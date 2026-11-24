from collections import Counter

import pandas as pd
import streamlit as st

from codal_rag.pipelines.research import ARMS
from codal_rag.storage.json_io import digest
from ..components import intro, metadata, prose, result, source
from ..evaluations import aggregate, details, discover, execution_kind, match_run, report_rows
from ..presentation import saved_extraction
from ..safety import UIError, json_bytes, parse_object, public, read_object
from ..state import LEVELS

METRICS = {"mean_answer_correctness": "میانگین شباهت واژگانی پاسخ", "answer_accuracy": "نرخ عبور از آستانهٔ واژگانی",
           "mean_semantic_correctness": "میانگین شباهت برداری پاسخ", "semantic_accuracy": "نرخ عبور از آستانهٔ شباهت",
           "citation_faithfulness": "اعتبار ساختاری ارجاع", "semantic_faithfulness": "تقریب پشتیبانی با شباهت برداری",
           "estimated_cost": "هزینهٔ برآوردی با تعرفهٔ پیکربندی", "latency_seconds": "زمان اجرا (ثانیه)",
           "claim_f1": "F1 ادعاهای داوری‌شده", "total_inference_tokens": "توکن واقعی مدل"}


def render(cfg, research):
    intro("ارزیابی پژوهشی", "اجرای ذخیره‌شده را انتخاب کنید تا معیارها، پاسخ مرجع و شواهد هر پرسش را بررسی کنید.")
    root = cfg.path().resolve()
    reports, runs = discover(root)
    manifests = {}
    invalid = 0
    for path in runs:
        try:
            manifest = read_object(path)
            if isinstance(manifest.get("complete"), bool) and isinstance(manifest.get("question_ids"), list):
                manifests[path] = manifest
            else:
                invalid += 1
        except (OSError, ValueError, TypeError):
            invalid += 1
    if manifests:
        with st.expander("فهرست اجراها"):
            st.dataframe([{"اجرا": str(p.parent.relative_to(root)), "پرسش": len(m["question_ids"]),
                           "منشأ": execution_kind(m)[0], "تکمیل": m["complete"]}
                          for p, m in manifests.items()], hide_index=True, use_container_width=True)
    if invalid:
        st.warning("شناسنامهٔ یک یا چند اجرا خواندنی نیست.")
    selected = st.selectbox("گزارش ذخیره‌شده", [None, *reports],
                            index=1 if reports else 0, format_func=lambda p: str(p.relative_to(root)) if p else "انتخاب گزارش")
    uploaded = st.file_uploader("یا گزارش ارزیابی را باز کنید", type="json", key="eval_upload")
    if not uploaded and not selected:
        st.info("گزارش ارزیابی موجود نیست. خروجی evaluate یا research-evaluate را انتخاب کنید.")
        return
    report = parse_object(uploaded.getvalue()) if uploaded else read_object(selected)
    rows = report_rows(report)
    if not rows:
        st.info("گزارش، پرسش امتیازگذاری‌شده‌ای ندارد.")
        return
    matched = None
    manifest = report.get("run")
    if manifests:
        path = st.selectbox("اتصال به فایل‌های همان اجرا برای بازرسی شواهد", [None, *manifests],
                            format_func=lambda p: str(p.parent.relative_to(root)) if p else "فقط گزارش")
        if path:
            matched_manifest = match_run(report, path)
            if matched_manifest is None:
                st.warning("فایل‌های این اجرا با نسخهٔ ثبت‌شده در گزارش مطابقت ندارند.")
            else:
                matched, manifest = path, matched_manifest
    kind, research_run = execution_kind(manifest, report.get("run_configuration") or report.get("configuration"))
    (st.info if research_run else st.warning)(kind)
    counts = {r["id"]: r["type"] for r in rows}
    metrics = {"questions": len(counts), "methods": len(report["overall"]),
               "complete": sum(m["complete"] for m in manifests.values()) if manifests else None}
    if manifest and manifest.get("index_config"):
        documents = {doc for index in manifest["index_config"].values() for method in index.values()
                     for doc in method.get("documents", [])}
        metrics["documents"] = len(documents)
    metadata(metrics, {"questions": "پرسش ارزیابی‌شده", "methods": "روش‌ها", "complete": "اجراهای کامل موجود",
                       "documents": "اسناد مجموعهٔ ارزیابی"})
    metadata(Counter(counts.values()), LEVELS)
    methods = st.multiselect("روش‌های نمایش", list(report["overall"]), default=list(report["overall"]))
    category = st.selectbox("ردهٔ پرسش", [None, *dict.fromkeys(counts.values())],
                            format_func=lambda x: LEVELS.get(x, "همه"))
    visible = [r for r in rows if r["method"] in methods and (not category or r["type"] == category)]
    summaries = aggregate(report, methods, category)
    keys = sorted({k for m in summaries.values() for k in m})
    if keys:
        metric = st.selectbox("معیار", keys, index=next((i for i, k in enumerate(keys) if k.startswith("recall@")), 0),
                              format_func=lambda k: METRICS.get(k, k))
        values = {m: data.get(metric) for m, data in summaries.items()}
        st.dataframe([{"روش": m, METRICS.get(metric, metric): v, "وضعیت": "امتیازدهی نشده" if v is None else "ثبت شده"}
                      for m, v in values.items()], hide_index=True, use_container_width=True)
        scored = {m: v for m, v in values.items() if type(v) in (int, float)}
        if research_run and scored:
            st.bar_chart(pd.DataFrame({"method": list(scored), "value": list(scored.values())}).set_index("method"),
                         color="#308873")
    if not visible:
        st.info("برای فیلترهای انتخاب‌شده نتیجه‌ای موجود نیست.")
        return
    st.subheader("بررسی پرسش‌ها")
    st.dataframe([{"شناسه": r["id"], "روش": r["method"], "رده": LEVELS[r["type"]],
                   **{k: v for k, v in r["metrics"].items() if v is None or type(v) in (float, int, bool)}}
                  for r in visible], hide_index=True, use_container_width=True)
    qid = st.selectbox("پرسش", list(dict.fromkeys(r["id"] for r in visible)))
    available = {r["method"]: r for r in visible if r["id"] == qid}
    m = st.selectbox("روش برای بررسی", list(available))
    row = available[m]
    if row.get("question"):
        prose(row["question"])
    if "answer" in row:
        st.caption("پاسخ مرجع")
        prose(row["answer"].get("expected_answer", ""))
        st.caption("پاسخ‌های استخراج‌شده برای ردهٔ پرسش")
        for candidate in row["answer"].get("candidate_answers", []):
            prose(candidate)
        if row["retrieval"].get("retrieval_details"):
            with st.expander("رتبه و شناسهٔ قطعه‌های بازیابی‌شده"):
                st.dataframe(row["retrieval"]["retrieval_details"], hide_index=True)
    if row["metrics"].get("errors"):
        st.dataframe(row["metrics"]["errors"], hide_index=True)
    if report["schema_version"] == 2:
        benchmark = st.file_uploader("مجموعهٔ ارزیابی همان اجرا برای پاسخ و شواهد مرجع (اختیاری)",
                                     type="json", key="eval_benchmark")
        if benchmark:
            dataset = parse_object(benchmark.getvalue())
            if digest(dataset) != report["run"]["dataset_sha256"]:
                raise UIError("اثر انگشت مجموعهٔ مرجع با اجرای گزارش یکسان نیست.")
            q = next(q for q in dataset["questions"] if q["id"] == qid)
            prose(q["question"])
            annotation = dataset.get("annotations", {}).get(qid, {})
            for claim in annotation.get("claims", []):
                prose(claim["claim"])
            for i, group in enumerate(annotation.get("evidence_sets", []), 1):
                with st.expander(f"مجموعهٔ شواهد مرجع {i}"):
                    for ev in group:
                        source(ev)
    if matched:
        assets = details(report, matched, qid, m)
        for ev in assets.get("gold", {}).get("gold_evidence", []):
            with st.expander("شاهد مرجع"):
                source(ev)
        retrieval = assets["retrieval"]
        base = ARMS[m][0] if m in ARMS else m
        from codal_rag.storage.legacy import select_retrieval_results
        hits = select_retrieval_results(retrieval.get("initial", retrieval), base)
        with st.expander("متن قطعه‌های بازیابی‌شده"):
            for hit in hits:
                source(hit["chunk"])
        record = saved_extraction(assets[m])
        prose(assets[m]["query"])
        result(record, "eval_" + qid + m, research)
    with st.expander("پیکربندی، تعریف معیارها و منشأ گزارش"):
        st.json(public({k: report[k] for k in ("schema_version", "configuration", "run_configuration", "run",
                          "semantic_evaluator", "metric_notes", "adjudications_sha256", "paired_comparisons") if k in report}))
    st.download_button("دریافت گزارش ارزیابی", json_bytes(report), "evaluation.json", "application/json")
