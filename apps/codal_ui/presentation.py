"""Pure formatters for answer, evidence and export. All source text is untrusted."""
from __future__ import annotations

from dataclasses import fields
from html import escape
import re

from codal_rag.extraction.citation_validation import validate_citation_view
from codal_rag.extraction.citations import build_citation_view
from codal_rag.extraction.validation import assert_valid_extraction
from codal_rag.models import EvidenceItem
from codal_rag.parsing.markdown import is_table_separator, parse_heading
from .safety import UIError, public
from .state import LEVELS, METHODS, STATUS


def passage_html(text):
    """Render only escaped text, Markdown headings and tables; never load source-supplied resources."""
    lines = str(text).splitlines()
    blocks, para = [], []

    def flush():
        if para:
            blocks.append('<div class="ci-prose" dir="auto">' + escape("\n".join(para)) + "</div>")
            para.clear()

    def cells(line, tag):
        parts = re.split(r"(?<!\\)\|", line.strip().strip("|"))
        return "<tr>" + "".join("<" + tag + '><bdi>' + escape(p.strip().replace(r"\|", "|")) +
                                "</bdi></" + tag + ">" for p in parts) + "</tr>"

    i = 0
    while i < len(lines):
        line = lines[i]
        if i + 1 < len(lines) and "|" in line and is_table_separator(lines[i + 1]):
            flush()
            rows = [cells(line, "th")]
            i += 2
            while i < len(lines) and "|" in lines[i] and lines[i].strip():
                rows.append(cells(lines[i], "td"))
                i += 1
            blocks.append('<div class="ci-table" tabindex="0" role="region" aria-label="جدول گزارش">'
                          "<table><thead>" + rows[0] + "</thead><tbody>" + "".join(rows[1:]) +
                          "</tbody></table></div>")
            continue
        heading = parse_heading(line)
        if heading:
            flush()
            blocks.append('<h4 dir="auto">' + escape(heading[1]) + "</h4>")
        elif not line.strip():
            flush()
        else:
            para.append(line)
        i += 1
    flush()
    return "\n".join(blocks)


def evidence_map(record):
    output = record.get("output", {})
    traces = output.get("source_trace", {})
    hits = record.get("initial", []) + output.get("final_retrieval", [])
    chunks = {(h["chunk"]["document_id"], h["chunk"]["chunk_id"]): h["chunk"] for h in hits}
    return {e["evidence_id"]: {**chunks.get((e["document_id"], e["chunk_id"]), {}),
                               **traces.get(e["evidence_id"], {}), **{k: v for k, v in e.items() if v is not None}}
            for e in output.get("evidence", [])}


def saved_extraction(obj):
    envelope = {}
    if "output" in obj:
        envelope, obj = obj, obj["output"]
        if not isinstance(obj, dict):
            raise UIError("خروجی استخراج خواندنی نیست.")
    if obj.get("method") not in ("baseline", "proposed") or not isinstance(obj.get("evidence"), list):
        raise UIError("فایل باید خروجی دستور استخراج باشد.")
    if obj.get("schema_version", 1) not in (1, 2):
        raise UIError("نسخهٔ خروجی استخراج پشتیبانی نمی‌شود.")
    allowed = {f.name for f in fields(EvidenceItem)}
    evidence = [EvidenceItem(**{k: v for k, v in row.items() if k in allowed}) for row in obj["evidence"]]
    if any(not e.document_id or not e.chunk_id or not isinstance(e.text, str) or not isinstance(e.evidence_id, str)
           for e in evidence) or len({e.evidence_id for e in evidence}) != len(evidence):
        raise UIError("شناسه یا متن منبع در فایل معتبر نیست.")
    ex = obj["extraction"]
    if not obj.get("query") or obj["query"] != ex.get("query"):
        raise UIError("پرسش فایل با پاسخ تطبیق ندارد.")
    assert_valid_extraction(ex, evidence, allow_single_chunk_latent=obj.get("schema_version") == 2)
    if obj.get("schema_version") == 2:
        for level in LEVELS:
            for claim in ex[level]:
                labels = claim.get("conceptual_labels")
                if not isinstance(labels, list) or any(not isinstance(v, str) or not v.strip() for v in labels):
                    raise UIError("برچسب‌های مفهومی خروجی نسخهٔ ۲ معتبر نیستند.")
    citations = build_citation_view(ex, evidence, query=obj["query"], method=obj["method"])
    check = validate_citation_view(citations)
    supplied = obj.get("citation_view")
    def identities(view):
        return [(item.get("level"), item["statement"], [(e["evidence_id"], e["document_id"], e["chunk_id"])
                 for e in item["evidence"]]) for item in view["items"]]

    if not check["valid"] or not validate_citation_view(supplied)["valid"] or identities(supplied) != identities(citations):
        raise UIError("ارجاع‌های ذخیره‌شده با شواهد و یافته‌های همین پاسخ تطبیق ندارند.")
    # Ignore extra user-supplied keys, including unrelated objects or private prompts.
    keys = ("schema_version", "method", "query", "llm", "extraction", "evidence", "source_trace",
            "final_retrieval", "egar", "operational", "context_budget")
    output = {k: obj[k] for k in keys if k in obj}
    output.update(citation_view=citations, citation_validation=check)
    method = "egar" if output.get("egar", {}).get("enabled") else obj["method"]
    if envelope and envelope.get("method") != method:
        raise UIError("روش ثبت‌شده با خروجی استخراج مطابقت ندارد.")
    record = {k: envelope[k] for k in ("configuration", "scope", "initial", "index_provenance",
                                        "initial_sha256", "normalized_query", "seconds") if k in envelope}
    record.update(method=method, status=envelope.get("status", "complete"), execution="saved", output=output,
                  usage=output.get("operational", envelope.get("usage", {})), stages=[])
    return public(record)


def answer_text(record):
    parts = [METHODS.get(record["method"], record["method"])]
    if record.get("error"):
        parts.append(record["error"]["message"])
    if record.get("answer"):
        parts += [record["answer"]["answer"], *record["answer"]["limitations"]]
    output = record.get("output", {})
    ex = output.get("extraction", {})
    for level, label in LEVELS.items():
        if ex.get(level):
            parts.append(label)
            for c in ex[level]:
                parts.append(c["claim"] + " [" + ", ".join(c["evidence_ids"]) + "]")
                parts.extend(str(c[k]) for k in ("basis", "disclaimer") if c.get(k))
                if c.get("conceptual_labels"):
                    parts.append("برچسب‌ها · " + "، ".join(c["conceptual_labels"]))
    if ex.get("insufficient_evidence"):
        parts.append("شواهد برای پاسخ کامل کافی نیست.")
    if ex.get("notes"):
        parts.append(ex["notes"])
    decision = output.get("egar", {}).get("final_sufficiency_decision", {}).get("sufficiency_status")
    if decision:
        parts.append("کفایت شواهد · " + STATUS.get(decision, decision))
    for eid, ev in evidence_map(record).items():
        parts.append(f"[{eid}] {ev['document_id']} / {ev['chunk_id']}\n{ev['text']}")
    return "\n\n".join(parts)


def comparison_text(run):
    blocks = [run["question"]]
    for r in run["methods"].values():
        blocks.append(answer_text(r))
        blocks.append(f"وضعیت: {r['status']} | زمان: {r['seconds']:.2f} s")
        if r.get("initial") is not None:
            blocks.append(f"قطعه‌های اولیه: {len(r['initial'])}")
    return "\n\n".join(blocks)
