# llm/synonyms.py
# Helpers to manage category synonyms for the interpreter.
# Keep tiny and dependency‑free.

from typing import Dict, Iterable, List


def map_synonym_hits(text: str, synonyms: Dict[str, Dict[str, List[str]]]) -> Dict[str, List[str]]:
    """Return merged hits for type_any/outcome_any given a text query.

    Args:
        text: User question (case‑insensitive).
        synonyms: Phrase→mapping dictionary like:
            {"sales-related": {"type_any": ["Billing/Sales"], "outcome_any": ["Sale-close"]}}

    Returns:
        Dict with keys "type_any" and "outcome_any". Missing keys default to [].
    """
    if not isinstance(text, str) or not text:
        return {"type_any": [], "outcome_any": []}
    t = text.lower()
    out = {"type_any": [], "outcome_any": []}
    for key, mapping in (synonyms or {}).items():
        if key.lower() in t:
            for target, vals in (mapping or {}).items():
                if target in out:
                    out[target].extend(vals or [])
    # de‑dup
    out["type_any"] = _dedup_list(out["type_any"])
    out["outcome_any"] = _dedup_list(out["outcome_any"])
    return out


def _dedup_list(xs: Iterable[str]) -> List[str]:
    seen, out = set(), []
    for x in xs:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out