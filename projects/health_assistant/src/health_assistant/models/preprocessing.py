"""ColumnTransformer pipeline used by both predictive models.

Defaults to all 15 features. Pass an explicit `feature_subset` to build a
preprocessor over a subset of columns.
"""
from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

from .feature_schema import FEATURE_NAMES, FEATURE_SPEC


def _split_features(feature_subset: list[str] | None = None):
    selected = set(feature_subset) if feature_subset is not None else set(FEATURE_NAMES)
    numeric: list[str] = []
    binary: list[str] = []
    ord_cats: list[tuple[str, list]] = []
    unord_cats: list[str] = []
    for name, spec in FEATURE_SPEC.items():
        if name not in selected:
            continue
        kind = spec["kind"]
        if kind == "numeric":
            numeric.append(name)
        elif kind == "binary":
            binary.append(name)
        elif kind == "categorical" and spec.get("ordered"):
            ord_cats.append((name, spec["choices"]))
        elif kind == "categorical":
            unord_cats.append(name)
    return numeric, binary, ord_cats, unord_cats


def build_preprocessor(feature_subset: list[str] | None = None) -> ColumnTransformer:
    numeric, binary, ord_cats, unord_cats = _split_features(feature_subset)
    ord_names = [n for n, _ in ord_cats]
    ord_choices = [c for _, c in ord_cats]
    transformers = [
        ("num", "passthrough", numeric),
        ("bin", "passthrough", binary),
        (
            "ord",
            OrdinalEncoder(
                categories=ord_choices,
                handle_unknown="use_encoded_value",
                unknown_value=-1,
            ),
            ord_names,
        ),
        ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False), unord_cats),
    ]
    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )
