"""Structural citation validation only; this cannot establish factual entailment."""


def validate_citation_view(citation_view: dict) -> dict:
    errors, warnings = [], []
    if not isinstance(citation_view, dict) or not isinstance(citation_view.get('items'), list):
        return {"valid": False, "errors": ["citation_view.items must be a list"], "warnings": []}
    for item in citation_view['items']:
        if not isinstance(item, dict):
            errors.append("Citation item must be an object")
            continue
        label = item.get('finding_id', 'UNKNOWN')
        if not isinstance(item.get('statement'), str) or not item['statement'].strip():
            errors.append(f"{label}: missing statement")
        evidence = item.get('evidence', [])
        if not isinstance(evidence, list) or not evidence:
            errors.append(f"{label}: no evidence attached")
            continue
        ids = []
        for ev in evidence:
            if not isinstance(ev, dict) or not ev.get('resolved'):
                errors.append(f"{label}: evidence could not be resolved")
                continue
            if not ev.get('document_id') or not ev.get('chunk_id') or not ev.get('evidence_id'):
                errors.append(f"{label}: missing evidence, document, or chunk identifier")
            ids.append(ev.get('evidence_id'))
            if citation_view.get('method') == 'proposed' and not ev.get('heading_path'):
                warnings.append(f"{label}: heading_path is empty")
        if ids != item.get('evidence_ids'):
            errors.append(f"{label}: citation IDs and resolved evidence differ")
    return {"valid": not errors, "errors": errors, "warnings": warnings,
            "error_count": len(errors), "warning_count": len(warnings)}
