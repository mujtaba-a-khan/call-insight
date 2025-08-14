# llm/interpreter.py
# Deterministic NL → filter spec (no external calls).
# Focus: precise filters that mirror dataset columns exactly.

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union
import re


# ------------------------- Public API ---------------------------------------


@dataclass
class Catalog:
    """Known entities from the dataset for grounding matches.

    Attributes:
        agents: Agent IDs or names (e.g., "A001", "Sara Ahmed").
        campaigns: Campaign names (e.g., "Black Friday").
        products: Product names (e.g., "Premium Plan").
    """
    agents: List[str] = field(default_factory=list)
    campaigns: List[str] = field(default_factory=list)
    products: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: Dict[str, Iterable[str]] | None) -> "Catalog":
        """Create a Catalog from a plain dictionary.

        Args:
            d: Optional mapping with keys "agents", "campaigns", "products".

        Returns:
            Catalog: Instance with empty lists for missing keys.
        """
        d = d or {}
        return cls(
            agents=[str(x) for x in (d.get("agents") or [])],
            campaigns=[str(x) for x in (d.get("campaigns") or [])],
            products=[str(x) for x in (d.get("products") or [])],
        )


def build_filter_spec(
    question: str,
    cfg: Dict,
    catalog: Catalog | Dict | None = None,
) -> Tuple[Dict, str]:
    """Interpret a natural-language question and emit a filter spec.

    Args:
        question: User text.
        cfg: Subset of configs/llm.yml. Recognized keys:
            - timezone: IANA timezone for relative dates.
            - synonyms: Optional phrase→targets mapping (type_any / outcome_any).
            - filter_schema.allowed_keys: Whitelist of keys to emit.
        catalog: Catalog or plain dict with known entities.

    Returns:
        Tuple (spec, notes):
            spec: Keys among {"date_from","date_to","type_any","outcome_any",
                              "agents","campaigns","products","min_amount","max_amount"}.
            notes: Short human-readable parse trace.
    """
    q = (question or "").strip()
    tz = (cfg or {}).get("timezone") or "Europe/Berlin"
    allowed = set(_allowed_keys(cfg))
    synonyms = (cfg or {}).get("synonyms") or {}
    cat = catalog if isinstance(catalog, Catalog) else Catalog.from_dict(catalog)

    spec: Dict[str, Union[str, float, List[str]]] = {}
    notes: List[str] = []

    # 1) Dates — explicit ranges → single dates → relative → week/month → month-only.
    dfrom, dto, date_note = _parse_dates(q, tz)
    if dfrom:
        spec["date_from"] = dfrom
    if dto:
        spec["date_to"] = dto
    if date_note:
        notes.append(date_note)

    # 2) Strong, exclusive TYPE intent.
    types = _detect_types_strict(q, synonyms)
    if types and "type_any" in allowed:
        spec["type_any"] = types

    # 3) Outcome directives.
    outs = _detect_outcomes(q, synonyms)
    if outs and "outcome_any" in allowed:
        spec["outcome_any"] = outs

    # 4) Campaigns / Products / Agents — strict substring grounding only.
    camps = _ground_substring(q, cat.campaigns)
    if camps and "campaigns" in allowed:
        spec["campaigns"] = camps

    prods = _ground_substring(q, cat.products)
    if prods and "products" in allowed:
        spec["products"] = prods

    ags = _ground_substring(q, cat.agents)
    ags += _after_words(q, ("by", "agent", "rep"), max_tokens=3)
    ags = sorted({a for a in ags if a})
    if ags and "agents" in allowed:
        resolved = _resolve_against_catalog(ags, cat.agents)
        spec["agents"] = resolved or ags

    # 5) Amounts.
    lo, hi = _parse_amounts(q)
    if lo is not None and "min_amount" in allowed:
        spec["min_amount"] = float(lo)
    if hi is not None and "max_amount" in allowed:
        spec["max_amount"] = float(hi)

    # Enforce allowed keys (dates always allowed).
    if allowed:
        keep = set(allowed) | {"date_from", "date_to"}
        spec = {k: v for k, v in spec.items() if k in keep}

    # Notes
    if spec.get("date_from") and spec.get("date_from") == spec.get("date_to"):
        notes.append(f"Date resolved to {spec['date_from']} ({tz}).")
    elif spec.get("date_from") and spec.get("date_to"):
        notes.append(f"Date range: {spec['date_from']} → {spec['date_to']} ({tz}).")

    return spec, " ".join(notes).strip()


# ------------------------- Allowed keys -------------------------------------

DEFAULT_ALLOWED_KEYS: Tuple[str, ...] = (
    "date_from",
    "date_to",
    "type_any",
    "outcome_any",
    "agents",
    "campaigns",
    "products",
    "min_amount",
    "max_amount",
)


def _allowed_keys(cfg: Dict | None) -> Sequence[str]:
    schema = (cfg or {}).get("filter_schema") or {}
    keys = schema.get("allowed_keys")
    if not keys:
        return DEFAULT_ALLOWED_KEYS
    return [k for k in DEFAULT_ALLOWED_KEYS if k in set(keys)]


# ------------------------- Dates --------------------------------------------

MONTHS = {
    "january": 1, "jan": 1,
    "february": 2, "feb": 2,
    "march": 3, "mar": 3,
    "april": 4, "apr": 4,
    "may": 5,
    "june": 6, "jun": 6,
    "july": 7, "jul": 7,
    "august": 8, "aug": 8,
    "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10,
    "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}


def _today_local(tz: str | None) -> date:
    """Return today's date in the configured timezone (best-effort)."""
    try:
        from zoneinfo import ZoneInfo
        return datetime.now(tz=ZoneInfo(tz)).date() if tz else datetime.now().date()
    except Exception:
        return datetime.now().date()


def _parse_dates(q: str, tz: str | None) -> Tuple[Optional[str], Optional[str], str]:
    """Parse date range or single date into ISO strings.

    Priority:
      1) Explicit ranges: "between X and Y", "from X to Y", "X - Y".
      2) Single explicit date: ISO / D.M.Y / "13 Aug 2025" / "Aug 13 2025".
      3) Relative: today, yesterday, last N days.
      4) Week/month windows: this week/last week, this month/last month.
      5) Month-year fallback "August 2025" → whole month;
         also **month-only** fallback "August" → whole month in the **current year**.
    """
    s = q.lower()
    today = _today_local(tz)

    # 1) Ranges
    m = re.search(r"between\s+(.*?)\s+and\s+(.*)", s)
    if m:
        d1 = _first_date(m.group(1))
        d2 = _first_date(m.group(2))
        if d1 and d2:
            a, b = sorted([d1, d2])
            return a.isoformat(), b.isoformat(), f"Date: {a.isoformat()}..{b.isoformat()}."
    m = re.search(r"from\s+(.*?)\s+to\s+(.*)", s)
    if m:
        d1 = _first_date(m.group(1))
        d2 = _first_date(m.group(2))
        if d1 and d2:
            a, b = sorted([d1, d2])
            return a.isoformat(), b.isoformat(), f"Date: {a.isoformat()}..{b.isoformat()}."
    m = re.search(r"(.*?)\s*-\s*(.*)", s)
    if m and any(tok in s for tok in ("jan","feb","mar","apr","may","jun","jul","aug","sep","sept","oct","nov","dec","202")):
        d1 = _first_date(m.group(1))
        d2 = _first_date(m.group(2))
        if d1 and d2:
            a, b = sorted([d1, d2])
            return a.isoformat(), b.isoformat(), f"Date: {a.isoformat()}..{b.isoformat()}."

    # 2) Single explicit
    d = _first_date(s)
    if d:
        return d.isoformat(), d.isoformat(), f"Date: {d.isoformat()}."

    # 3) Relative
    if "today" in s:
        return today.isoformat(), today.isoformat(), "Date: today."
    if "yesterday" in s:
        y = today - timedelta(days=1)
        return y.isoformat(), y.isoformat(), "Date: yesterday."
    m = re.search(r"(last|past)\s+(\d{1,3})\s+days?", s)
    if m:
        n = int(m.group(2))
        start = today - timedelta(days=n)
        return start.isoformat(), today.isoformat(), f"Date: last {n} days."

    # 4) Week/month windows
    if "last week" in s:
        wd = today.weekday()
        end = today - timedelta(days=wd + 1)
        start = end - timedelta(days=6)
        return start.isoformat(), end.isoformat(), "Date: last week."
    if "this week" in s:
        wd = today.weekday()
        start = today - timedelta(days=wd)
        end = start + timedelta(days=6)
        return start.isoformat(), min(end, today).isoformat(), "Date: this week."
    if "last month" in s:
        first_this = today.replace(day=1)
        last_prev = first_this - timedelta(days=1)
        start = last_prev.replace(day=1)
        end = last_prev
        return start.isoformat(), end.isoformat(), "Date: last month."
    if "this month" in s:
        start = today.replace(day=1)
        return start.isoformat(), today.isoformat(), "Date: this month."

    # 5a) Month-year fallback (whole month)
    my = re.search(r"\b([A-Za-z]{3,9})\s+(20\d{2})\b", s)
    if my and not _contains_any_explicit_date_tokens(s):
        mon, year = my.groups()
        mo = MONTHS.get(mon.lower())
        if mo:
            start = date(int(year), mo, 1)
            end = (date(int(year), mo + 1, 1) - timedelta(days=1)) if mo < 12 else date(int(year), 12, 31)
            return start.isoformat(), end.isoformat(), f"Date: {start.isoformat()}..{end.isoformat()}."

    # 5b) Month-only fallback (e.g., "calls of august")
    monly = re.search(r"\b(?:of|in)?\s*([A-Za-z]{3,9})\b", s)
    # Require the token to actually be a month, and avoid matching generic words.
    if monly:
        mon = monly.group(1).lower()
        if mon in MONTHS and not _contains_any_explicit_date_tokens(s):
            mo = MONTHS[mon]
            year = today.year
            start = date(year, mo, 1)
            end = (date(year, mo + 1, 1) - timedelta(days=1)) if mo < 12 else date(year, 12, 31)
            return start.isoformat(), end.isoformat(), f"Date: {start.isoformat()}..{end.isoformat()}."

    return None, None, ""


def _contains_any_explicit_date_tokens(s: str) -> bool:
    """Return True if the string already contains explicit date tokens."""
    return bool(
        re.search(r"\b(20\d{2})-(\d{1,2})-(\d{1,2})\b", s)
        or re.search(r"\b(\d{1,2})[./](\d{1,2})[./](20\d{2})\b", s)
        or re.search(r"\b(\d{1,2})\s+[A-Za-z]{3,9}\s+(20\d{2})\b", s)
        or re.search(r"\b[A-Za-z]{3,9}\s+(\d{1,2})\s+(20\d{2})\b", s)
    )


def _first_date(fragment: str) -> Optional[date]:
    """Extract the first date found in a fragment."""
    s = fragment.strip().replace(",", " ")
    # ISO yyyy-mm-dd
    m = re.search(r"\b(20\d{2})-(\d{1,2})-(\d{1,2})\b", s)
    if m:
        y, mo, d = map(int, m.groups())
        return _safe_date(y, mo, d)
    # D/M/Y or M/D/Y (prefer D/M when both ≤ 12)
    m = re.search(r"\b(\d{1,2})[./](\d{1,2})[./](20\d{2})\b", s)
    if m:
        a, b, y = map(int, m.groups())
        if a <= 12 and b <= 12:
            d, mo = a, b
        else:
            mo, d = (a, b) if a <= 12 else (b, a)
        return _safe_date(y, mo, d)
    # "13 Aug 2025"
    m = re.search(r"\b(\d{1,2})\s+([A-Za-z]{3,9})\s+(20\d{2})\b", s)
    if m:
        d, mon, y = m.groups()
        mo = MONTHS.get(mon.lower())
        if mo:
            return _safe_date(int(y), mo, int(d))
    # "Aug 13 2025"
    m = re.search(r"\b([A-Za-z]{3,9})\s+(\d{1,2})\s+(20\d{2})\b", s)
    if m:
        mon, d, y = m.groups()
        mo = MONTHS.get(mon.lower())
        if mo:
            return _safe_date(int(y), mo, int(d))
    return None


def _safe_date(y: int, m: int, d: int) -> Optional[date]:
    try:
        return date(y, m, d)
    except Exception:
        return None


# ------------------------- Types & Outcomes ---------------------------------

TYPE_CANONICAL = ("Billing/Sales", "Support", "Inquiry", "Complaint")
OUTCOME_CANONICAL = ("Unknown", "Resolved", "Callback", "Refund", "Sale-close", "Disconnected")

TYPE_PATTERNS = [
    (r"\b(inquiry|queries|general inquiry)\b", "Inquiry"),
    (r"\b(sales|billing|sales[- ]related)\b", "Billing/Sales"),
    (r"\b(support|technical support|tech support)\b", "Support"),
    (r"\b(complaint|escalation|issue)\b", "Complaint"),
    (r"\b(inquiry|sales|billing|support|complaint)[- ]related\b", None),
    (r"related to (inquiry|sales|billing|support|complaint)\b", None),
    (r"\btype\s+(inquiry|sales|billing|support|complaint)\b", None),
]

OUTCOME_PATTERNS = [
    (r"\bunknown\b", "Unknown"),
    (r"\bresolved|fixed|solved\b", "Resolved"),
    (r"\bcallback|call back\b", "Callback"),
    (r"\brefund|refunded\b", "Refund"),
    (r"\bsale[- ]close|purchase completed\b", "Sale-close"),
    (r"\bdisconnected|dropped|hang ?up|hung up\b", "Disconnected"),
    (r"outcome\s+is\s+([A-Za-z\- ]+)", None),
]


def _detect_types_strict(q: str, synonyms: Dict) -> List[str]:
    """Return an exclusive list of types inferred from the query."""
    s = q.lower()
    hits: List[Tuple[int, str]] = []

    for pat, label in TYPE_PATTERNS:
        m = re.search(pat, s)
        if not m:
            continue
        idx = m.start()
        if label is not None:
            hits.append((idx, label))
        else:
            word = (m.group(1) or "").lower()
            mapping = {
                "inquiry": "Inquiry",
                "queries": "Inquiry",
                "sales": "Billing/Sales",
                "billing": "Billing/Sales",
                "support": "Support",
                "complaint": "Complaint",
            }
            lab = mapping.get(word)
            if lab:
                hits.append((idx, lab))

    if hits:
        hits.sort(key=lambda x: x[0])
        return [hits[0][1]]

    if isinstance(synonyms, dict):
        for phrase, mapping in (synonyms or {}).items():
            if phrase and isinstance(mapping, dict) and phrase.lower() in s:
                vals = mapping.get("type_any") or []
                for v in vals:
                    vv = str(v).strip()
                    if vv in TYPE_CANONICAL:
                        return [vv]
    return []


def _detect_outcomes(q: str, synonyms: Dict) -> List[str]:
    """Return outcome labels inferred from the query (can be multiple)."""
    s = q.lower()
    outs: List[str] = []

    for pat, label in OUTCOME_PATTERNS:
        m = re.search(pat, s)
        if not m:
            continue
        if label is not None:
            outs.append(label)
        else:
            explicit = m.group(1).strip().title()
            outs.append(explicit)

    if isinstance(synonyms, dict):
        for phrase, mapping in (synonyms or {}).items():
            if phrase and isinstance(mapping, dict) and phrase.lower() in s:
                vals = mapping.get("outcome_any") or []
                outs.extend(str(v).strip() for v in vals if str(v).strip())

    # de-dup preserving order
    seen, out = set(), []
    for x in outs:
        if x not in seen:
            seen.add(x)
            out.append(x)
    normalized = [x for x in out if x in OUTCOME_CANONICAL] or out
    return normalized


# ------------------------- Entities & Amounts --------------------------------

def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def _ground_substring(q: str, candidates: Sequence[str]) -> List[str]:
    """Return catalog items that appear as substrings in the question."""
    if not candidates:
        return []
    qs = _normalize(q)
    hits = []
    for c in candidates:
        cs = _normalize(str(c))
        if cs and cs in qs:
            hits.append(str(c))
    # de-dup preserving order
    seen, out = set(), []
    for x in hits:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _resolve_against_catalog(tokens: Sequence[str], candidates: Sequence[str]) -> List[str]:
    """Resolve free tokens to catalog values via equality/substring (case-insensitive)."""
    if not tokens or not candidates:
        return []
    outs: List[str] = []
    for t in tokens:
        tn = _normalize(t)
        exact = next((c for c in candidates if _normalize(c) == tn), None)
        if exact:
            outs.append(str(exact))
            continue
        sub = next((c for c in candidates if tn in _normalize(c)), None)
        if sub:
            outs.append(str(sub))
    seen, out = set(), []
    for x in outs:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _after_words(q: str, words: Tuple[str, ...], max_tokens: int) -> List[str]:
    """Capture short phrases after words like 'by', 'agent', 'rep'."""
    s = " " + q.strip() + " "
    found: List[str] = []
    for w in words:
        for m in re.finditer(rf"\b{re.escape(w)}\s+([A-Za-z0-9][A-Za-z0-9_\- ]{{0,64}})", s, flags=re.IGNORECASE):
            frag = re.split(r"[;,.]|\n", m.group(1))[0]
            tokens = frag.split()
            if tokens:
                found.append(" ".join(tokens[:max_tokens]))
    return found


def _parse_amounts(q: str) -> Tuple[Optional[float], Optional[float]]:
    """Extract min/max amount constraints from common patterns."""
    s = q.lower().replace("–", "-").replace("—", "-")
    m = re.search(r"\$?\s*(\d+(?:[\.,]\d+)?)\s*-\s*\$?\s*(\d+(?:[\.,]\d+)?)", s)
    if m:
        a, b = (float(m.group(1).replace(",", ".")), float(m.group(2).replace(",", ".")))
        return (min(a, b), max(a, b))
    m = re.search(r"between\s+(\d+(?:[\.,]\d+)?)\s+and\s+(\d+(?:[\.,]\d+)?)", s)
    if m:
        a, b = (float(m.group(1).replace(",", ".")), float(m.group(2).replace(",", ".")))
        return (min(a, b), max(a, b))
    lo = hi = None
    m = re.search(r">=\s*(\d+(?:[\.,]\d+)?)", s)
    if m:
        lo = float(m.group(1).replace(",", "."))
    m = re.search(r"(?:<=|≤)\s*(\d+(?:[\.,]\d+)?)", s)
    if m:
        hi = float(m.group(1).replace(",", "."))
    m = re.search(r"\bmin\s*(\d+(?:[\.,]\d+)?)", s)
    if m:
        lo = float(m.group(1).replace(",", "."))
    m = re.search(r"\bmax\s*(\d+(?:[\.,]\d+)?)", s)
    if m:
        hi = float(m.group(1).replace(",", "."))
    return lo, hi
