"""web_search - SerpAPI Google search, filtered to a medical-domain allowlist.

Uses SerpApi (serpapi.com) via langchain's SerpAPIWrapper. Note: this is a
different vendor from Serper.dev despite the similar name. Env var
`SERPA_API_KEY` (legacy spelling kept for compatibility) holds the SerpApi
account key.

Gated by SessionContext.web_search_enabled (sidebar toggle). When disabled,
the tool returns {"status": "disabled", "hits": []} so the LLM sees an
explicit no-op rather than thinking the tool failed.

Allowlist is intentional: medical web content is uneven, and a clinician POC
should never ground medical advice on a random blog.
"""
from __future__ import annotations

import os
from urllib.parse import urlparse

from health_assistant.agent.session_state import get_session_ctx

ALLOWED_DOMAINS = {
    # Government / public-health (English-speaking)
    "cdc.gov", "nih.gov", "medlineplus.gov", "fda.gov", "who.int", "escardio.org",
    # Academic / index
    "pubmed.ncbi.nlm.nih.gov", "ncbi.nlm.nih.gov",
    # Reputable secondary sources
    "uptodate.com", "merckmanuals.com", "msdmanuals.com",
    # Professional bodies / top journals
    "ahajournals.org", "diabetesjournals.org", "thelancet.com",
    "nejm.org", "bmj.com", "jamanetwork.com",
    # European institutions
    "ema.europa.eu", "ecdc.europa.eu",
    # Portuguese health system + societies
    "dgs.pt", "sns.gov.pt", "sns24.gov.pt", "infarmed.pt", "spms.min-saude.pt",
    "ordemdosmedicos.pt", "spginecologia.pt", "saudereprodutiva.dgs.pt",

}


def _domain_of(url: str) -> str:
    """Extract the registered host, stripping a leading 'www.' if present."""
    try:
        host = urlparse(url).netloc.lower()
        return host[4:] if host.startswith("www.") else host
    except Exception:
        return ""


def _domain_in_allowlist(url: str) -> bool:
    host = _domain_of(url)
    if not host:
        return False
    return any(host == d or host.endswith("." + d) for d in ALLOWED_DOMAINS)


def _serper_results(query: str) -> dict:
    """Thin wrapper around SerpAPI. Kept as a module-level function with the
    legacy name `_serper_results` so existing tests can patch by this path.

    Returns a normalized dict with an `organic` list (we remap SerpAPI's
    `organic_results` to the same shape Serper.dev uses, so the rest of the
    pipeline doesn't care which vendor produced the response). Any exception
    (rate limit, timeout, JSON parse failure, etc.) is caught here and
    reported via a sentinel `_error` key so the caller can surface a clean
    status to the LLM instead of having Strands flag it as a tool error.
    """
    from langchain_community.utilities import SerpAPIWrapper

    # Legacy env-var name preserved for compatibility with earlier docs.
    api_key = os.environ.get("SERPA_API_KEY") or os.environ.get("SERPAPI_API_KEY")
    if not api_key:
        return {"organic": [], "_error": "no_api_key"}
    try:
        wrapper = SerpAPIWrapper(serpapi_api_key=api_key)
        raw = wrapper.results(query)
        # SerpAPI returns "organic_results"; Serper.dev returned "organic".
        # Normalize so downstream code works with either response shape.
        organic = raw.get("organic_results") or raw.get("organic") or []
        return {"organic": organic}
    except Exception as e:  # rate limit, timeout, bad JSON, ...
        msg = f"{type(e).__name__}: {e}"
        # Print to stdout so it lands in the Streamlit terminal for debugging.
        print(f"[web_search] SerpAPI call failed for query={query!r}: {msg}")
        return {"organic": [], "_error": msg}


# Google quietly degrades on long site:-OR chains, so cap how many sites we
# splice into one query. ~10 is the sweet spot in our testing.
_MAX_SITES_PER_QUERY = 10


def _build_site_filter(sites: list[str] | None) -> tuple[str, list[str], list[str]]:
    """Validate LLM-supplied sites against ALLOWED_DOMAINS, build the Google
    `site:` OR-filter snippet, and report what was kept vs dropped.

    Returns (snippet_or_empty, kept, dropped). The snippet is empty when no
    valid sites were supplied (caller falls back to an unfiltered query)."""
    if not sites:
        return "", [], []
    kept: list[str] = []
    dropped: list[str] = []
    for s in sites:
        if not isinstance(s, str):
            continue
        s = s.strip().lower()
        # Tolerate "https://www.dgs.pt/foo" style inputs
        if "//" in s:
            s = _domain_of(s)
        elif s.startswith("www."):
            s = s[4:]
        if s in ALLOWED_DOMAINS:
            if s not in kept:
                kept.append(s)
        else:
            dropped.append(s)
    kept = kept[:_MAX_SITES_PER_QUERY]
    if not kept:
        return "", [], dropped
    snippet = "(" + " OR ".join(f"site:{d}" for d in kept) + ")"
    return snippet, kept, dropped


def web_search(query: str, k: int = 5, sites: list[str] | None = None) -> dict:
    """Run a medical-allowlisted Google search.

    Args:
        query: the search query (free text).
        k: how many top hits to return after the allowlist post-filter.
        sites: OPTIONAL list of domains to bias the search toward. Pass 5-10
            domains chosen from ALLOWED_DOMAINS that are most relevant to the
            question (e.g. Portuguese gynecology question -> ["dgs.pt",
            "sns.gov.pt", "spginecologia.pt", "saudereprodutiva.dgs.pt",
            "pubmed.ncbi.nlm.nih.gov"]). When supplied, the tool appends a
            Google `site:X OR site:Y ...` clause so Serper returns results
            from those domains directly, instead of returning the top-N
            globally and discarding most via post-filter. The allowlist
            post-filter still runs as a safety net.

            Unknown or off-allowlist sites are silently dropped (returned in
            sites_dropped). If all your sites get dropped or you pass None,
            the search is unbiased and the post-filter handles correctness.

    Returns:
        {
          "status": "disabled" | "ok" | "no_allowed_results" | "error",
          "hits": [{"title", "link", "snippet", "domain"}, ...],
          "sites_used":            list of allowlisted sites we biased to,
          "sites_dropped":         sites you passed that weren't allowlisted,
          "filtered_out_domains":  raw Serper hits that the post-filter cut,
          "error":                 (only on status=error) the underlying
                                    Serper error string. Do NOT retry blindly
                                    when you see this - tell the user there
                                    was a search-provider problem.
        }
    """
    if not get_session_ctx().web_search_enabled:
        return {
            "status": "disabled",
            "hits": [],
            "sites_used": [],
            "sites_dropped": [],
            "filtered_out_domains": [],
        }

    snippet, sites_used, sites_dropped = _build_site_filter(sites)
    serper_query = f"{query} {snippet}".strip() if snippet else query

    raw = _serper_results(serper_query)
    if isinstance(raw, dict) and raw.get("_error"):
        return {
            "status": "error",
            "hits": [],
            "sites_used": sites_used,
            "sites_dropped": sites_dropped,
            "filtered_out_domains": [],
            "error": raw["_error"],
        }
    organic = raw.get("organic", []) if isinstance(raw, dict) else []
    filtered = [
        {
            "title": h.get("title", ""),
            "link": h.get("link", ""),
            "snippet": h.get("snippet", ""),
            "domain": _domain_of(h.get("link", "")),
        }
        for h in organic
        if _domain_in_allowlist(h.get("link", ""))
    ][:k]
    filtered_out_domains = sorted({
        _domain_of(h.get("link", ""))
        for h in organic
        if h.get("link") and not _domain_in_allowlist(h.get("link", ""))
    })

    return {
        "status": "ok" if filtered else "no_allowed_results",
        "hits": filtered,
        "sites_used": sites_used,
        "sites_dropped": sites_dropped,
        "filtered_out_domains": filtered_out_domains,
    }
