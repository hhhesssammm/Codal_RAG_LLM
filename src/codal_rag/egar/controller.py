"""Bounded, single-controller EGAR. No source, embedding or index mutation."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict

from codal_rag.egar.analyzer import Assessment, EvidenceSufficiencyAnalyzer
from codal_rag.egar.evidence import (content, merge_hits, pack_hits, token_upper_bound,
                                     trace_sources)
from codal_rag.egar.runtime import BudgetedLLM, BudgetExceeded, QuestionBudget
from codal_rag.extraction.three_level import ThreeLevelExtractor
from codal_rag.retrieval.scope import canonical, resolve_scope
from codal_rag.storage.legacy import select_retrieval_results


def abstention(query, reason):
    return {"query": query, "explicit": [], "implicit": [], "latent": [],
            "insufficient_evidence": True, "notes": reason}


def run_adaptive(query, payload, method, config, *, engine, client, scope=None, judge_client=None):
    if not config.egar.enabled:
        raise ValueError("Adaptive controller requires egar.enabled=true")
    budget = QuestionBudget(config.egar)
    generator = BudgetedLLM(client, budget)
    judge = BudgetedLLM(judge_client or client, budget) if config.egar.judge == "llm" else None
    analyzer = EvidenceSufficiencyAnalyzer(judge)
    initial = deepcopy(select_retrieval_results(payload, method))
    trace = {"enabled": True, "activated": False, "activation_reason": None,
             "judge": {"mode": config.egar.judge, "model": getattr(judge_client or client, 'model', config.llm.model),
                       "separately_injected": judge_client is not None},
             "configuration": asdict(config.egar), "scope": None, "routing_reason": None,
             "initial_retrieval": initial, "iterations": [], "assessments": [],
             "missing_evidence_requirements": [], "supplementary_queries": [],
             "newly_retrieved_evidence": [], "duplicate_evidence": [],
             "iteration_count": 0, "stop_reason": None}
    selected, evidence, sources = [], [], {}
    extraction = abstention(query, "Evidence has not been assessed")
    assessment = Assessment(query, "unverifiable")
    total_candidates = len(initial)
    initial_tokens = sum(token_upper_bound(content(h)) for h in initial)
    additional_tokens = 0
    final_tokens = 0
    try:
        if len(initial) > config.egar.max_candidates:
            raise BudgetExceeded('Initial retrieval exceeds the candidate budget')
        if payload.get("index_config") is not None and payload["index_config"] != engine.index_config:
            raise ValueError("Initial and supplementary index provenance differ")
        if payload.get('retrieval_configuration') is not None and payload['retrieval_configuration'] != asdict(config.retrieval):
            raise ValueError('Initial and supplementary retrieval settings differ')
        if payload.get('index_config') is not None and 'retrieval_configuration' not in payload and config.retrieval.mode != 'dense':
            raise ValueError('Legacy initial retrieval is dense; regenerate it for hybrid EGAR')
        catalog = engine.catalog(method)
        resolved, reason = resolve_scope(query, catalog, scope)
        trace["routing_reason"] = reason
        if resolved is None:
            trace["stop_reason"] = "ambiguous_scope"
            assessment = Assessment(query, "unverifiable", justification=reason, validation_status="scope_unresolved")
        else:
            trace["scope"] = resolved.to_dict()
            pool, dropped = merge_hits([], [h for h in initial[:config.egar.max_candidates]
                                            if resolved.matches(h["chunk"])])
            trace["duplicate_evidence"].extend(dropped)
            headings = sorted({h for c in catalog if resolved.matches(c) for h in c.get("heading_path", [])})
            seen_queries = {canonical(query)}
            for iteration in range(config.egar.max_iterations + 1):
                if budget.remaining() <= 0:
                    raise BudgetExceeded("Question time budget exhausted")
                selected, evidence, excluded, final_tokens = pack_hits(pool, config.extraction.max_evidence_tokens)
                sources = trace_sources(evidence, selected)
                extraction = ThreeLevelExtractor(generator, schema_version=2).extract(query, evidence)
                assessment = analyzer.analyze(query, extraction, evidence, scope=resolved, sources=sources, headings=headings)
                trace["assessments"].append({"iteration": iteration, "evidence": [e.to_dict() for e in evidence],
                    "source_trace": sources, "excluded_by_context_budget": excluded,
                    "candidate_extraction": deepcopy(extraction), **assessment.to_dict()})
                trace["missing_evidence_requirements"].extend(g for g in assessment.missing_evidence_requirements
                    if g not in trace["missing_evidence_requirements"])
                if assessment.sufficiency_status == "sufficient":
                    trace["stop_reason"] = "sufficient"
                    break
                if assessment.sufficiency_status == "contradictory":
                    trace["stop_reason"] = "contradictory_evidence"
                    break
                if iteration == config.egar.max_iterations:
                    trace["stop_reason"] = "iteration_budget"
                    break
                if total_candidates >= config.egar.max_candidates:
                    trace["stop_reason"] = "candidate_budget"
                    break
                queries = []
                for gap in assessment.missing_evidence_requirements:
                    q = canonical(gap["query"])
                    if not q or q in seen_queries:
                        continue
                    # Generated hints can only select existing report headings.
                    hints = [h for h in gap.get("section_hints", []) if h in headings]
                    queries.append({**gap, "section_hints": hints,
                                    "section_routing": "matched_headings" if hints else "in_scope_fallback"})
                    seen_queries.add(q)
                    if len(queries) == config.egar.max_queries_per_iteration:
                        break
                if not queries:
                    trace["stop_reason"] = "no_new_targeted_queries"
                    break
                if budget.llm_calls + 2 > config.egar.max_llm_calls:
                    trace["stop_reason"] = "llm_budget"
                    break
                trace.update(activated=True, activation_reason=trace["activation_reason"] or assessment.sufficiency_status)
                step = {"iteration": iteration + 1, "queries": [], "new_chunk_ids": []}
                trace["iterations"].append(step)
                trace["iteration_count"] = iteration + 1
                additions = []
                for gap in queries:
                    limit = min(config.egar.supplementary_top_k, config.egar.max_candidates - total_candidates)
                    if limit <= 0:
                        break
                    step["queries"].append(gap)
                    trace["supplementary_queries"].append(gap["query"])
                    hits = budget.run(lambda g=gap, k=limit: engine.search_method(g["query"], method, k,
                        scope=resolved, section_hints=g["section_hints"]))
                    if not isinstance(hits, list) or len(hits) > limit:
                        raise ValueError("Retriever exceeded candidate budget")
                    total_candidates += len(hits)
                    additional_tokens += sum(token_upper_bound(content(h)) for h in hits)
                    if any(not resolved.matches(h["chunk"]) for h in hits):
                        raise ValueError("Supplementary retriever violated document/company/period scope")
                    additions.extend(hits)
                merged, dropped = merge_hits(pool, additions)
                trace["duplicate_evidence"].extend(dropped)
                old_ids = {(h["chunk"]["document_id"], h["chunk"]["chunk_id"]) for h in pool}
                new = [h for h in merged if (h["chunk"]["document_id"], h["chunk"]["chunk_id"]) not in old_ids]
                trace["newly_retrieved_evidence"].extend(deepcopy(new))
                step["new_chunk_ids"] = [h["chunk"]["chunk_id"] for h in new]
                if not new:
                    trace["stop_reason"] = "no_new_evidence"
                    break
                # Keep used initial evidence, then new complementary candidates,
                # then other initial hits. Reassessment sees the actual packed context.
                used = {eid for a in assessment.claims if a.sufficiency_status not in {'unverifiable', 'contradictory'}
                        for eid in a.evidence_ids}
                required = {(e.document_id, e.chunk_id) for e in evidence if e.evidence_id in used}
                pool = [h for h in pool if (h["chunk"]["document_id"], h["chunk"]["chunk_id"]) in required] + new + [
                    h for h in pool if (h["chunk"]["document_id"], h["chunk"]["chunk_id"]) not in required]
    except Exception as exc:
        # Provider failures may contain sensitive request details; log only class.
        reason = str(exc) if isinstance(exc, BudgetExceeded) else "Evidence assessment/retrieval failed validation"
        trace["stop_reason"] = "budget_or_timeout" if isinstance(exc, BudgetExceeded) else "service_or_validation_failure"
        trace["error_type"] = type(exc).__name__
        assessment = Assessment(query, "unverifiable", justification=reason, validation_status="failed_closed")
    if assessment.sufficiency_status != "sufficient":
        extraction = abstention(query, assessment.justification or "Insufficient verified complementary evidence")
    else:
        for analysis in assessment.claims:
            claim = extraction[analysis.inference_level][analysis.claim_index]
            claim.update(inference_level=analysis.inference_level,
                evidence_sufficiency_status=analysis.sufficiency_status,
                validation_status=analysis.validation_status,
                source_trace={eid: sources[eid] for eid in analysis.evidence_ids},
                joint_support=analysis.joint_support)
    trace["final_sufficiency_decision"] = assessment.to_dict()
    metrics = {**budget.metrics(), "retrieved_candidate_count": total_candidates,
               "initial_retrieval_token_upper_bound": initial_tokens,
               "additional_retrieval_token_upper_bound": additional_tokens,
               "retrieved_token_count": initial_tokens + additional_tokens,
               "final_evidence_tokens": final_tokens,
               "retrieved_token_count_basis": "utf8_byte_upper_bound_not_provider_tokens"}
    return {"extraction": extraction, "evidence": evidence, "source_trace": sources,
            "final_retrieval": selected, "egar": trace, "operational": metrics}
