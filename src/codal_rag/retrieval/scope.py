"""Conservative document routing from explicit metadata; never route by similarity."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import re

from codal_rag.normalization.persian import NormalizerConfig, normalize_plain_text


def canonical(value) -> str:
    return normalize_plain_text(str(value or ""), NormalizerConfig()).replace("\u200c", " ").strip().lower()


@dataclass(frozen=True)
class RetrievalScope:
    document_ids: tuple[str, ...] = ()
    companies: tuple[str, ...] = ()
    years: tuple[str, ...] = ()
    symbols: tuple[str, ...] = ()
    periods: tuple[str, ...] = ()

    def __post_init__(self):
        for name, values in asdict(self).items():
            if not isinstance(values, (tuple, list)) or any(not isinstance(v, str) or not v.strip() for v in values):
                raise ValueError(f"Scope {name} must be a list of nonempty strings")
            object.__setattr__(self, name, tuple(values))

    def matches(self, chunk: dict) -> bool:
        if self.document_ids and chunk.get('document_id') not in self.document_ids:
            return False
        fields = (("company", self.companies),
                  ("year", self.years), ("symbol", self.symbols), ("financial_period", self.periods))
        return all(not values or canonical(chunk.get(key)) in {canonical(v) for v in values}
                   for key, values in fields)

    def to_dict(self) -> dict:
        return {k: list(v) for k, v in asdict(self).items()}


def resolve_scope(question: str, catalog: list[dict], supplied: RetrievalScope | None = None):
    """Return (scope, reason). None requires explicit document/company/period input.

    Multiple values are only allowed when explicitly supplied or named in the
    question. One-document corpora are a controlled fallback, not top-hit routing.
    """
    q = canonical(question)
    named = {}
    for attr, key in (("companies", "company"), ("symbols", "symbol")):
        values = sorted({str(c[key]) for c in catalog if c.get(key)})
        named[attr] = tuple(v for v in values if re.search(r"(?<!\w)" + re.escape(canonical(v)) + r"(?!\w)", q))
    named["years"] = tuple(sorted(set(re.findall(r"\b(?:13|14)\d{2}\b", q))))
    if supplied is not None:
        for attr in ("companies", "symbols", "years"):
            values = getattr(supplied, attr)
            if values and named[attr] and not set(map(canonical, named[attr])) <= set(map(canonical, values)):
                return None, f"Question conflicts with supplied {attr}"
        scope, reason = supplied, "explicit_scope"
    else:
        scope, reason = RetrievalScope(**named), "literal_metadata_match"
    matches = [c for c in catalog if scope.matches(c)]
    if not matches:
        return None, "No document matches the requested company/period/document"
    for attr, key in (('companies', 'company'), ('symbols', 'symbol')):
        if named[attr] and any(canonical(c.get(key)) not in set(map(canonical, named[attr])) for c in matches):
            return None, f'Question identity conflicts with selected document {key}'
    if scope.document_ids:
        if set(scope.document_ids) - {c.get("document_id") for c in matches}:
            return None, "One or more requested documents are unavailable"
        return scope, reason
    docs = {c["document_id"] for c in matches}
    if len(docs) == 1 and len({c["document_id"] for c in catalog}) == 1:
        return RetrievalScope(document_ids=tuple(docs), **{k: v for k, v in scope.to_dict().items()
                             if k != "document_ids"}), "single_document_corpus"
    if not (scope.companies or scope.symbols):
        return None, "Company is ambiguous; provide explicit scope"
    if not (scope.years or scope.periods):
        periods = {(canonical(c.get("year")), canonical(c.get("financial_period"))) for c in matches}
        if len(periods) > 1 or any(not y and not p for y, p in periods):
            return None, "Reporting period is ambiguous; provide explicit scope"
    # Year alone cannot silently combine annual and interim reports.
    by_year = {}
    for c in matches:
        if c.get('financial_period'):
            by_year.setdefault(canonical(c.get('year')), set()).add(canonical(c['financial_period']))
    if not scope.periods and any(len(periods) > 1 for periods in by_year.values()):
        return None, "Multiple reporting periods match; specify periods or document IDs"
    return scope, reason
