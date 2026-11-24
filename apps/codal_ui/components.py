from __future__ import annotations

from html import escape
import streamlit as st

from .analysis import STAGES
from .catalog import catalog_for, scope_for
from .presentation import answer_text, evidence_map, passage_html
from .safety import codal_url, json_bytes
from .state import LEVELS, STATUS

META = {"company": "شرکت", "year": "سال مالی", "financial_period": "دورهٔ گزارش", "document_type": "نوع گزارش",
        "document_title": "عنوان سند", "document_id": "شناسهٔ سند", "source_file": "فایل منبع",
        "section": "بخش", "subsection": "زیربخش", "chunk_id": "شناسهٔ قطعه", "page": "صفحه", "row": "ردیف",
        "cell": "سلول", "symbol": "نماد", "publication_date": "تاریخ انتشار", "report_type": "نوع گزارش ثبت‌شده"}
STOP = {"sufficient": "شواهد کافی", "ambiguous_scope": "شرکت یا دورهٔ گزارش مبهم است",
        "contradictory_evidence": "تعارض شواهد", "iteration_budget": "پایان بودجهٔ دورها",
        "llm_budget": "پایان بودجهٔ فراخوانی مدل", "candidate_budget": "پایان بودجهٔ قطعه‌ها",
        "no_new_evidence": "شاهد تازه‌ای بازیابی نشد", "no_new_targeted_queries": "پرسش تکمیلی تازه‌ای ثبت نشد",
        "budget_or_timeout": "پایان مهلت یا بودجه", "service_or_validation_failure": "خطای سرویس یا اعتبارسنجی"}
RESULT_STATUS = {"complete": "پایان اجرا", "failed": "خطا", "unavailable": "در دسترس نیست", "incomplete": "ناقص"}


def prose(text):
    st.markdown('<div class="ci-prose" dir="auto">' + escape(str(text)) + "</div>", unsafe_allow_html=True)


def passage(text):
    st.markdown(passage_html(text), unsafe_allow_html=True)


def intro(title, text):
    st.title(title)
    st.markdown('<div class="ci-intro">' + escape(text) + "</div>", unsafe_allow_html=True)


def metadata(values, labels=META):
    rows = [f'<span>{escape(label)} · <bdi>{escape(str(values[k]))}</bdi></span>'
            for k, label in labels.items() if values.get(k) is not None and values[k] != ""]
    if rows:
        st.markdown('<div class="ci-source">' + "　/　".join(rows) + '</div>', unsafe_allow_html=True)


def error_box(error, research):
    st.error(error["message"])
    if research:
        st.caption(f"{error['stage']} · {error['type']}")


def choose_scope(cfg, key):
    if not st.checkbox("محدود کردن جست‌وجو به اسناد", key=key + "_use_scope"):
        return None
    cat = catalog_for(cfg, "proposed")
    if not cat or not cat.docs:
        st.info("فرادادهٔ نمایه‌شده برای انتخاب دامنه در دسترس نیست.")
        return None
    chosen = {}
    for col, field in zip(st.columns(4), ("company", "year", "financial_period", "document_type")):
        options = sorted({m[field] for m in cat.docs.values() if m.get(field)})
        if options:
            chosen[field] = col.selectbox(META[field], [None, *options], format_func=lambda x: x or "همه",
                                          key=key + field)
    if not any(chosen.values()):
        selected = st.multiselect("سند مشخص", sorted(cat.docs), key=key + "_docs")
        if selected:
            from codal_rag.retrieval.scope import RetrievalScope
            return RetrievalScope(document_ids=tuple(selected))
    return scope_for(cat.docs, chosen)


def source(ev):
    metadata(ev)
    if ev.get("heading_path"):
        prose(" ‹ ".join(str(h) for h in ev["heading_path"]))
    passage(ev.get("text") or ev.get("content") or "")
    for loc in ev.get("source_locations", []):
        if isinstance(loc, dict):
            metadata(loc, {k: META.get(k, k) for k in loc})
    url = codal_url(ev.get("source_url"))
    if url:
        st.link_button("گزارش اصلی در کدال", url)


def trace_details(record):
    st.caption("مراحل ثبت‌شده در همین اجرا")
    if record.get("stages"):
        prose(" ← ".join(STAGES[s] for s in record["stages"]))
    if record.get("initial_sha256"):
        st.caption("اثر انگشت بازیابی اولیهٔ مشترک")
        st.code(record["initial_sha256"], language=None)
    if record.get("scope"):
        st.caption("دامنهٔ اعمال‌شده")
        st.json(record["scope"])
    trace = record.get("output", {}).get("egar", {})
    if trace.get("enabled"):
        st.subheader("مسیر بازیابی تطبیقی")
        counts = {"initial": len(trace.get("initial_retrieval", [])), "rounds": trace.get("iteration_count"),
                  "new": len(trace.get("newly_retrieved_evidence", []))}
        metadata(counts, {"initial": "قطعهٔ اولیه", "rounds": "دور تکمیلی", "new": "قطعهٔ تازه"})
        for a in trace.get("assessments", []):
            title = "ارزیابی اولیه" if a["iteration"] == 0 else f"ارزیابی پس از دور {a['iteration']}"
            with st.expander(title + " · " + STATUS.get(a["sufficiency_status"], "نامشخص")):
                for g in a.get("missing_evidence_requirements", []):
                    prose(g.get("description", ""))
                metadata({"sources": ", ".join(e["evidence_id"] for e in a.get("evidence", []))}, {"sources": "شواهد این دور"})
        for step in trace.get("iterations", []):
            with st.expander(f"جست‌وجوهای تکمیلی دور {step['iteration']}"):
                for gap in step["queries"]:
                    prose(gap["query"])
                for cid in step.get("new_chunk_ids", []):
                    prose(cid)
        reason = trace.get("stop_reason")
        prose("علت توقف · " + STOP.get(reason, str(reason)))
    if record.get("initial"):
        with st.expander("قطعه‌های بازیابی اولیه"):
            for hit in record["initial"]:
                st.caption(f"رتبه {hit['rank']} · {hit['score']:.4g}")
                source(hit["chunk"])
    if record.get("configuration"):
        with st.expander("پیکربندی مؤثر و منشأ ایندکس"):
            st.json({"configuration": record["configuration"], "index": record.get("index_provenance")})
    usage = record.get("usage", {})
    if usage:
        metadata(usage, {"llm_calls": "فراخوانی مدل", "prompt_tokens": "توکن ورودی", "completion_tokens": "توکن خروجی",
                         "retrieved_candidate_count": "کل نامزدها", "final_evidence_tokens": "کران بایتی متن نهایی"})


def answer(record, key):
    if "answer" in record:
        prose(record["answer"]["answer"])
        st.caption("این پاسخ بدون دریافت گزارش مالی تولید شده است.")
        for line in record["answer"]["limitations"]:
            prose(line)
        return
    output = record.get("output", {})
    ex = output.get("extraction")
    if ex is None:
        return
    refs = evidence_map(record)
    # No invented summary; the extractor's statements are the narrative.
    labels = []
    for level, heading in LEVELS.items():
        st.subheader(heading)
        if not ex[level]:
            st.caption("یافته‌ای در این سطح گزارش نشده است.")
        for i, claim in enumerate(ex[level], 1):
            prose(claim["claim"])
            if claim.get("basis"):
                prose(claim["basis"])
            if claim.get("disclaimer"):
                prose(claim["disclaimer"])
            labels.extend(claim.get("conceptual_labels", []))
            for eid in claim["evidence_ids"]:
                with st.expander(f"مشاهدهٔ شاهد {eid} · یافتهٔ {i}"):
                    source(refs[eid])
            if st.session_state.mode == "پژوهشگر" and claim.get("confidence") is not None:
                st.caption(f"اطمینان اعلام‌شدهٔ مدل، کالیبره‌نشده · {claim['confidence']}")
    st.subheader("برچسب‌های مفهومی")
    if labels:
        prose("، ".join(dict.fromkeys(labels)))
    else:
        st.caption("برچسب مستقلی در خروجی ثبت نشده است.")
    st.subheader("محدودیت‌ها و کفایت شواهد")
    if ex["insufficient_evidence"]:
        st.warning("شواهد برای پاسخ کامل به این پرسش کافی نیست.")
    trace = output.get("egar", {})
    decision = trace.get("final_sufficiency_decision", {}).get("sufficiency_status")
    if decision:
        st.caption("تصمیم ارزیاب شواهد · " + STATUS.get(decision, decision))
        prose(STOP.get(trace.get("stop_reason"), trace.get("stop_reason", "")))
    else:
        st.caption("ارزیابی مستقل کفایت شواهد در این اجرا ثبت نشده است.")
    if ex.get("notes"):
        prose(ex["notes"])
    check = output.get("citation_validation", {})
    if check.get("valid"):
        st.caption("ارجاع‌ها از نظر ساختاری معتبرند. صحت مالی و پشتیبانی معنایی نیاز به بررسی جداگانه دارند.")
        if check.get("warning_count"):
            st.caption(f"تعداد هشدارهای ساختاری · {check['warning_count']}")


def result(record, key, research):
    if record.get("error"):
        error_box(record["error"], research)
    if record.get("execution") == "saved":
        st.caption("بازخوانی خروجی ذخیره‌شده")
    if record.get("output", {}).get("llm", {}).get("mocked"):
        st.warning("این خروجی با مدل جایگزین آزمون تولید شده است.")
    if record.get("execution") == "smoke_retrieval":
        st.warning("بازیابی آزمایشی با بردارهای هش؛ امتیازها کاربرد پژوهشی ندارند.")
    tabs = st.tabs(["پاسخ", "شواهد و منابع"] + (["فرایند بازیابی"] if research else []))
    with tabs[0]:
        answer(record, key)
        if record.get("answer") or record.get("output"):
            with st.expander("کپی متن پاسخ"):
                st.code(answer_text(record), language=None)
    with tabs[1]:
        sources = evidence_map(record)
        if sources:
            for eid, ev in sources.items():
                with st.expander(f"شاهد {eid}"):
                    source(ev)
        elif record.get("initial"):
            st.caption("قطعه‌های بازیابی‌شده پیش از تکمیل استخراج")
            for hit in record["initial"]:
                with st.expander(f"قطعهٔ رتبهٔ {hit['rank']}"):
                    source(hit["chunk"])
        else:
            st.caption("برای این اجرا شاهدی ثبت نشده است.")
    if research:
        with tabs[2]:
            trace_details(record)
    st.download_button("دریافت نتیجهٔ JSON", json_bytes(record), key=key + "_download",
                       file_name=record["method"] + ".json", mime="application/json")


