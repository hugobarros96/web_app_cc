"""Restricted-exec sandbox for the `python_analytics` tool.

The agent will write small pandas/matplotlib snippets to answer analytical
questions ("how many smokers?", "compare ALT across readmitted vs not", "plot
medication count distribution by sex"). This module runs those snippets safely.

Safety boundary (POC-grade):
- AST whitelist rejects: `import`, `__dunder__` attribute access, the forbidden
  builtins (`open`, `eval`, `exec`, `compile`, `__import__`, `input`, `exit`).
- Globals contain only a curated allowlist (df, pd, np, plt, sns, safe builtins).
- 30-second wall-clock timeout via signal.SIGALRM.
- Captures stdout, the value of the last expression, and any matplotlib figure
  produced via `plt.show()` (or simply leaving a figure on `plt.gcf()`).

Production sandbox: e2b / Modal / gVisor. Documented in ARCHITECTURE.md.
"""
from __future__ import annotations

import ast
import base64
import io
import signal
import threading
import traceback
from contextlib import redirect_stdout
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

FORBIDDEN_NAMES = {"open", "eval", "exec", "compile", "__import__", "input", "exit", "quit"}


class _AstGuard(ast.NodeVisitor):
    def __init__(self):
        self.errors: list[str] = []

    def visit_Import(self, node: ast.Import) -> None:
        self.errors.append("import not allowed")

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        self.errors.append("import not allowed")

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if isinstance(node.attr, str) and node.attr.startswith("__"):
            self.errors.append(f"dunder attribute access not allowed: {node.attr}")
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id in FORBIDDEN_NAMES:
            self.errors.append(f"name not allowed: {node.id}")


class _Timeout(Exception):
    pass


def _alarm_handler(signum, frame):  # noqa: ARG001
    raise _Timeout("execution exceeded timeout")


_SAFE_BUILTINS = {
    "abs": abs, "min": min, "max": max, "sum": sum, "len": len,
    "range": range, "round": round, "int": int, "float": float,
    "str": str, "bool": bool, "list": list, "dict": dict, "tuple": tuple,
    "set": set, "sorted": sorted, "enumerate": enumerate, "zip": zip,
    "map": map, "filter": filter, "any": any, "all": all,
    "print": print, "True": True, "False": False, "None": None,
    "isinstance": isinstance, "type": type, "repr": repr,
}


def run_sandboxed(code: str, df: pd.DataFrame, timeout_seconds: int = 30) -> dict[str, Any]:
    """Execute `code` against a preloaded dataframe `df`. Return dict with
    {stdout, value, figure_png, error}. `figure_png` is base64-encoded if a
    matplotlib figure was produced, else None. `value` is the value of the
    last expression if the code ends in one, else None."""
    out: dict[str, Any] = {"stdout": "", "value": None, "figure_png": None, "error": None}

    # Parse
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as e:
        out["error"] = f"SyntaxError: {e}"
        return out

    # AST guard
    guard = _AstGuard()
    guard.visit(tree)
    if guard.errors:
        out["error"] = "; ".join(guard.errors)
        return out

    # If the last statement is an expression, capture its value
    last = tree.body[-1] if tree.body else None
    has_value = isinstance(last, ast.Expr)
    if has_value:
        tree.body[-1] = ast.Assign(
            targets=[ast.Name(id="__result__", ctx=ast.Store())],
            value=last.value,
        )
        ast.fix_missing_locations(tree)

    globals_dict = {
        "__builtins__": _SAFE_BUILTINS,
        "df": df,
        "pd": pd,
        "np": np,
        "plt": plt,
        "sns": sns,
    }
    locals_dict: dict[str, Any] = {}

    plt.close("all")
    buf = io.StringIO()

    # signal.alarm only works on the main thread. Agent frameworks run tools
    # in async/thread contexts where setting SIGALRM raises ValueError.
    # Guard with thread + platform check; if signals aren't usable, run
    # without the hard timeout (POC-acceptable; documented).
    use_signal_timeout = (
        hasattr(signal, "SIGALRM")
        and threading.current_thread() is threading.main_thread()
    )
    prev_handler = None
    if use_signal_timeout:
        try:
            prev_handler = signal.signal(signal.SIGALRM, _alarm_handler)
            signal.alarm(timeout_seconds)
        except (ValueError, OSError):
            prev_handler = None
            use_signal_timeout = False

    try:
        with redirect_stdout(buf):
            exec(compile(tree, "<sandbox>", "exec"), globals_dict, locals_dict)  # noqa: S102
        if has_value:
            out["value"] = locals_dict.get("__result__")
        if plt.get_fignums():
            fig = plt.gcf()
            img = io.BytesIO()
            fig.savefig(img, format="png", dpi=120, bbox_inches="tight")
            out["figure_png"] = base64.b64encode(img.getvalue()).decode("ascii")
            plt.close("all")
    except _Timeout as e:
        out["error"] = f"TimeoutError: {e}"
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=2)}"
    finally:
        if use_signal_timeout and prev_handler is not None:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, prev_handler)

    out["stdout"] = buf.getvalue()
    return out
