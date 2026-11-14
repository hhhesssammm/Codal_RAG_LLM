"""Versioned CodalBench planning/validation. No fabricated gold data."""
from __future__ import annotations

from collections import Counter, defaultdict
import re

from codal_rag.retrieval.scope import RetrievalScope, canonical
from codal_rag.storage.json_io import load_json, save_json


PROTOCOL = {"reports": 60, "companies": 20, "industries": 5, "years": ["1401", "1402", "1403"],
            "questions": 260, "answerable": 240, "controls": 20,
            "development_questions": 60, "development_companies": 5,
            "test_questions": 200, "test_companies": 15, "test_answerable": 180,
            "cluster_bootstrap_resamples": 10000}


def create_benchmark_template(path):
    questions = [{"id": f"Q{i+1:03}", "question": "", "type": "", "company_id": "",
                  "split": "development" if i < 60 else "test", "answerable": i < 240,
                  "scope": {"document_ids": [], "companies": [], "years": []}}
                 for i in range(260)]
    return save_json({"schema_version": 2, "protocol": PROTOCOL, "synthetic": False,
                      "documents": [], "questions": questions, "annotations": {},
                      "note": "Unfilled planning template, not an evaluation dataset. Annotate from real reports."}, path)


def public_question(q):
    # Do not give generation answerability, type, answers or gold evidence.
    return {k: q[k] for k in ("id", "question", "scope")}


def validate_benchmark(data, *, require_full=False):
    if not isinstance(data, dict) or data.get("schema_version") != 2:
        raise ValueError("Expected CodalBench schema_version 2")
    questions, documents = data.get("questions"), data.get("documents")
    if not isinstance(questions, list) or not questions or not isinstance(documents, list):
        raise ValueError("Benchmark requires questions and a document manifest")
    ids, company_splits = set(), defaultdict(set)
    doc_ids = set()
    for doc in documents:
        if not isinstance(doc, dict) or any(not isinstance(doc.get(k), str) or not doc[k].strip() for k in
                ("document_id", "company_id", "company", "year", "industry", "split")):
            raise ValueError("Document manifest requires identity, company, year, industry and split")
        if doc['document_id'] in doc_ids or doc['split'] not in {'development', 'test'}:
            raise ValueError("Duplicate document or invalid document split")
        doc_ids.add(doc['document_id'])
        company_splits[doc['company_id']].add(doc['split'])
    doc_by_id = {d['document_id']: d for d in documents}
    for q in questions:
        if not isinstance(q, dict) or not isinstance(q.get("id"), str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,99}", q["id"]):
            raise ValueError("Invalid benchmark question ID")
        if q['id'] in ids:
            raise ValueError("Duplicate question ID")
        ids.add(q['id'])
        if q.get('type') not in {'explicit', 'implicit', 'latent'} or q.get('split') not in {'development', 'test'}:
            raise ValueError("Fill valid inference type and split before running")
        if type(q.get('answerable')) is not bool or any(not isinstance(q.get(k), str) or not q[k].strip()
                                                       for k in ('question', 'company_id')):
            raise ValueError("Fill question, company_id and answerability")
        scope = RetrievalScope(**q.get('scope', {}))
        if not any(scope.to_dict().values()):
            raise ValueError("Benchmark questions require explicit retrieval scope")
        company_splits[q['company_id']].add(q['split'])
        selected_docs = [d for d in documents if scope.matches(d)]
        if not selected_docs or set(scope.document_ids) - set(doc_by_id):
            raise ValueError("Question scope does not resolve to manifest documents")
        if any(d['company_id'] != q['company_id'] or d['split'] != q['split'] for d in selected_docs):
            raise ValueError("Clustered benchmark question must stay within its company and split")
    if any(len(parts) != 1 for parts in company_splits.values()):
        raise ValueError("Company-level development/test leakage")
    annotations = data.get('annotations', {})
    if not isinstance(annotations, dict) or set(annotations) - ids:
        raise ValueError("Annotations must reference existing question IDs")
    for q in questions:
        if q['id'] not in annotations:
            continue
        ann = annotations[q['id']]
        if not isinstance(ann, dict) or not isinstance(ann.get('claims'), list) or not isinstance(ann.get('evidence_sets'), list):
            raise ValueError("Annotation requires claims and alternative complete evidence_sets")
        if q['answerable'] and (not ann['claims'] or not ann['evidence_sets']):
            raise ValueError("Answerable annotation requires gold claims and evidence")
        if not q['answerable'] and (ann['claims'] or ann['evidence_sets']):
            raise ValueError("Insufficient-evidence controls cannot contain a complete gold answer")
        if ann.get('validation_status') not in {'human_validated', 'draft', 'synthetic'}:
            raise ValueError("Record the gold annotation validation status")
        claim_ids = set()
        for claim in ann['claims']:
            if not isinstance(claim, dict) or not claim.get('id') or claim['id'] in claim_ids or not claim.get('claim'):
                raise ValueError("Gold claims require unique IDs and text")
            claim_ids.add(claim['id'])
            if claim.get('inference_level') not in {'explicit', 'implicit', 'latent'} or not isinstance(claim.get('conceptual_labels'), list):
                raise ValueError("Gold claim requires inference level and independent conceptual labels")
        for evidence_set in ann['evidence_sets']:
            if not isinstance(evidence_set, list) or not evidence_set:
                raise ValueError("Gold evidence sets must be nonempty")
            for evidence in evidence_set:
                if not isinstance(evidence, dict) or evidence.get('document_id') not in doc_by_id or not isinstance(evidence.get('text'), str) or not evidence['text'].strip():
                    raise ValueError("Gold evidence requires real document identity and exact passage")
                if not RetrievalScope(**q['scope']).matches(doc_by_id[evidence['document_id']]):
                    raise ValueError("Gold evidence lies outside question scope")
    if require_full:
        validate_full_protocol(data, company_splits)
    return data


def validate_full_protocol(data, company_splits=None):
    docs, qs = data['documents'], data['questions']
    companies = {d['company_id'] for d in docs}
    industries = {d['industry'] for d in docs}
    expected = (len(docs), len(companies), len(industries), len(qs), sum(q['answerable'] for q in qs))
    if expected != (60, 20, 5, 260, 240):
        raise ValueError("Corpus does not meet the approved 60/20/5/260/240 plan")
    by_company = defaultdict(list)
    for d in docs:
        by_company[d['company_id']].append(d)
    for rows in by_company.values():
        if len(rows) != 3 or {canonical(d['year']) for d in rows} != set(PROTOCOL['years']):
            raise ValueError("Each company requires one report for each year 1401-1403")
        if len({d['industry'] for d in rows}) != 1:
            raise ValueError("Company industry must be stable")
    for industry in industries:
        members = {d['company_id'] for d in docs if d['industry'] == industry}
        dev = {d['company_id'] for d in docs if d['industry'] == industry and d['split'] == 'development'}
        if len(members) != 4 or len(dev) != 1:
            raise ValueError("Each industry needs four companies, exactly one in development")
    for split, count, company_count, answerable in (('development', 60, 5, 60), ('test', 200, 15, 180)):
        selected = [q for q in qs if q['split'] == split]
        if (len(selected), len({q['company_id'] for q in selected}), sum(q['answerable'] for q in selected)) != (count, company_count, answerable):
            raise ValueError("Question/answerability/company split differs from approved plan")
    if Counter(q['id'] in data.get('annotations', {}) for q in qs)[True] != 260:
        raise ValueError("All 260 questions require validated annotations for a full evaluation")
    if any(a['validation_status'] != 'human_validated' for a in data['annotations'].values()):
        raise ValueError("Full research evaluation requires human-validated gold annotations")


def load_benchmark(path, *, require_full=False):
    return validate_benchmark(load_json(path), require_full=require_full)
