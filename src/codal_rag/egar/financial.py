"""Small, explicit financial arithmetic checks; no execution of generated code."""
from __future__ import annotations

from decimal import Decimal, InvalidOperation
import re

from codal_rag.retrieval.scope import canonical


UNIT_SCALE = {
    "ریال": Decimal(1), "هزار ریال": Decimal(1000), "میلیون ریال": Decimal(10**6),
    "میلیارد ریال": Decimal(10**9), "rial": Decimal(1), "million rial": Decimal(10**6),
    "billion rial": Decimal(10**9), "%": Decimal(1), "درصد": Decimal(1),
}


def number(value: str) -> Decimal:
    text = canonical(value).replace(",", "").replace("٪", "%")
    if text.endswith("%"):
        text = text[:-1].strip()
    if text.startswith("(") and text.endswith(")"):
        text = "-" + text[1:-1]
    if not re.fullmatch(r"[+-]?\d+(?:\.\d+)?", text):
        raise ValueError("Unverifiable financial number")
    try:
        result = Decimal(text)
    except InvalidOperation as exc:
        raise ValueError("Invalid financial number") from exc
    if not result.is_finite():
        raise ValueError("Non-finite financial number")
    return result


def numeric_tokens(text: str):
    return [number(m) for m in re.findall(r"\(?[+-]?\d[\d,]*(?:\.\d+)?\)?", canonical(text))]


def quantities(text: str):
    """Explicit adjacent number+unit pairs; table-inherited units use calculations."""
    units = '|'.join(re.escape(u) for u in sorted(UNIT_SCALE, key=len, reverse=True))
    pattern = r'(\(?[+-]?\d[\d,]*(?:\.\d+)?\)?)\s*(' + units + r')(?!\w)'
    return {(number(value)*UNIT_SCALE[unit], 'percent' if unit in {'%', 'درصد'} else 'rial')
            for value, unit in re.findall(pattern, canonical(text).replace('٪', '%'))}


def unit_names(text: str):
    pattern = '|'.join(re.escape(u) for u in sorted(UNIT_SCALE, key=len, reverse=True))
    return set(re.findall(r'(?<!\w)(?:' + pattern + r')(?!\w)', canonical(text).replace('٪', '%')))


def validate_calculation(check: dict, evidence_by_id: dict) -> dict:
    """Anchored row/cell quotes plus explicit periods and units are mandatory.

    Supported operations: identity (one table reading), ratio, percentage,
    percent_change, difference, sum.
    Values are computed with Decimal; a semantic judge must still verify that
    the chosen rows/columns/concepts are appropriate for the question.
    """
    if not isinstance(check, dict) or check.get("operation") not in {
            "identity", "ratio", "percentage", "percent_change", "difference", "sum"}:
        raise ValueError("Unsupported calculation operation")
    operands = check.get("operands")
    if not isinstance(operands, list) or len(operands) != (1 if check['operation'] == 'identity' else 2):
        raise ValueError("A reading requires one anchored operand; arithmetic requires two")
    amounts, periods, families = [], [], []
    for operand in operands:
        if not isinstance(operand, dict):
            raise ValueError("Malformed operand")
        ev = evidence_by_id.get(operand.get("evidence_id"))
        quote, period, unit = (operand.get(k) for k in ("quote", "period", "unit"))
        if not ev or any(not isinstance(v, str) or not v.strip() for v in (quote, period, unit)):
            raise ValueError("Calculation lacks evidence, period or unit")
        if canonical(quote) not in canonical(ev.text):
            raise ValueError("Numerical quote is not in cited evidence")
        value = number(str(operand.get("value", "")))
        if value not in numeric_tokens(quote):
            raise ValueError("Numerical value is not in quoted row/cell")
        # Unit and period must occur in the evidence (including table headers),
        # not just in the generated arithmetic instruction.
        source = canonical(ev.text + " " + " ".join(ev.heading_path))
        unit = canonical(unit).replace("٪", "%")
        if unit not in UNIT_SCALE or unit not in source.replace("٪", "%"):
            raise ValueError("Unverifiable numerical unit")
        explicit_amounts = quantities(quote)
        family = 'percent' if unit in {'%', 'درصد'} else 'rial'
        if explicit_amounts and (value*UNIT_SCALE[unit], family) not in explicit_amounts:
            raise ValueError('Numerical unit conflicts with the quoted amount')
        if not explicit_amounts:
            available_units = unit_names(source)
            unit_quote = operand.get('unit_quote', '')
            if unit_quote and (canonical(unit_quote) not in source or unit not in unit_names(unit_quote)):
                raise ValueError('Unit header quote is not anchored')
            if unit not in available_units or (len(available_units) > 1 and not unit_quote):
                raise ValueError('Ambiguous table units require an exact unit_quote header')
        if canonical(period) not in source and canonical(period) != canonical(ev.year):
            raise ValueError("Unverifiable financial period")
        amounts.append(value * UNIT_SCALE[unit])
        families.append("percent" if unit in {"%", "درصد"} else "rial")
        periods.append(canonical(period))
    if len(set(families)) != 1:
        raise ValueError("Numerical unit mismatch")
    op = check["operation"]
    if op != "percent_change" and len(set(periods)) != 1:
        raise ValueError("Calculation periods must align")
    if op == "percent_change" and len(set(periods)) != 2:
        raise ValueError("Percent change requires distinct explicitly cited periods")
    a, b = amounts[0], amounts[1] if len(amounts) == 2 else Decimal(1)
    if op in {"ratio", "percentage", "percent_change"} and b == 0:
        raise ValueError("Division by zero")
    actual = {"identity": lambda: a, "sum": lambda: a+b, "difference": lambda: a-b,
              "ratio": lambda: a/b, "percentage": lambda: 100*a/b,
              "percent_change": lambda: 100*(a-b)/abs(b)}[op]()
    result_unit = canonical(check.get("result_unit", ""))
    if op in {"sum", "difference", "identity"}:
        if result_unit not in UNIT_SCALE or result_unit in {"%", "درصد"}:
            raise ValueError("A monetary result requires a known unit")
        actual /= UNIT_SCALE[result_unit]
    elif op in {"percentage", "percent_change"} and result_unit not in {"%", "درصد"}:
        raise ValueError("Percentage result unit must be percent")
    elif op == "ratio" and result_unit != "ratio":
        raise ValueError("Ratio result must be dimensionless")
    expected = number(str(check.get("result", "")))
    if abs(actual - expected) > Decimal("0.01"):
        raise ValueError("Incorrect numerical reasoning (absolute rounding tolerance 0.01)")
    return {"valid": True, "computed_result": str(actual), "result_unit": result_unit,
            "operation": op, "validation": "deterministic_arithmetic_with_anchored_operands"}


def unsupported_causality(claim: str) -> bool:
    text = canonical(claim)
    return any(term in text for term in ("caused", "causes", "because of", "باعث", "موجب", "به علت"))


def unqualified_distress(claim: str) -> bool:
    text = canonical(claim)
    distress = any(w in text for w in ("ورشکست", "درماندگی مالی", "financial distress", "bankrupt"))
    qualified = any(w in text for w in ("ممکن", "احتمال", "may", "could", "risk", "ریسک"))
    return distress and not qualified
