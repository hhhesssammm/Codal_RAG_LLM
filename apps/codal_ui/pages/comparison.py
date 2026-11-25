import streamlit as st

from ..analysis import current_engine
from ..jobs import submitted_run
from ..components import RESULT_STATUS, choose_scope, data_table, intro, metadata, prose, result, source
from ..presentation import comparison_text
from ..safety import UIError, json_bytes, public
from ..state import METHODS, STATUS


def retrieval_only(cfg):
    with st.form("retrieval_only"):
        question = st.text_input("پرسش بازیابی")
        submit = st.form_submit_button("جست‌وجوی شواهد", type="primary")
    scope = choose_scope(cfg, "lookup")
    if submit:
        if not question.strip():
            raise UIError("متن پرسش را وارد کنید.")
        with st.spinner("بازیابی شواهد"):
            engine = current_engine(cfg)
            payload = engine.result_payload(engine.search(question.strip(), cfg.top_k, scope=scope))
            st.session_state.retrieved = public(payload)
    payload = st.session_state.get("retrieved")
    if payload:
        prose(payload["original_query"])
        if payload.get("embedding_model") == "smoke-hash-v1":
            st.warning("این بازیابی از بردارهای آزمایشی هش استفاده کرده است.")
        for col, method in zip(st.columns(2), ("baseline", "proposed")):
            with col:
                st.subheader(METHODS[method])
                if not payload[method]:
                    st.info("قطعه‌ای در دامنهٔ انتخابی پیدا نشد.")
                for hit in payload[method]:
                    with st.expander(f"رتبهٔ {hit['rank']} · {hit['score']:.4g}"):
                        source(hit["chunk"])
        st.download_button("دریافت خروجی بازیابی", json_bytes(payload), "retrieval.json", "application/json")


def render(cfg, research):
    state = st.session_state
    intro("مقایسه روش‌ها", "یک پرسش را با روش‌های منتخب اجرا کنید و پاسخ‌ها، منابع و هزینهٔ پردازش را کنار هم ببینید.")
    if research and st.radio("نوع مقایسه", ["پاسخ و شواهد", "فقط بازیابی"], horizontal=True) == "فقط بازیابی":
        retrieval_only(cfg)
        return
    with st.form("comparison_form_" + (state.comparison or {}).get("id", "first")):
        question = st.text_area("پرسش مشترک", max_chars=8000, height=100, key="compare_question")
        methods = st.multiselect("روش‌های آزمایش", list(METHODS), default=list(METHODS), format_func=METHODS.get)
        submit = st.form_submit_button("اجرای مقایسه", type="primary")
    scope = choose_scope(cfg, "compare") if any(m != "llm_only" for m in methods) else None
    st.caption("مدل و تنظیمات تولید مشترک‌اند. روش‌های RAG با همان رتبه‌بندی اولیه و قرارداد استخراج نسخهٔ ۲ اجرا می‌شوند. "
               "فقط EGAR بودجهٔ تکمیلی دارد. روش بدون RAG فقط متن پرسش را دریافت می‌کند.")
    submitted_run(state, "comparison", submit, question, methods, cfg, scope)
    run = state.comparison
    if not run:
        return
    st.divider()
    prose(run["question"])
    results = list(run["methods"].values())
    table = []
    for r in results:
        out, usage = r.get("output", {}), r.get("usage", {})
        trace = out.get("egar", {})
        decision = trace.get("final_sufficiency_decision", {}).get("sufficiency_status")
        valid = out.get("citation_validation", {}).get("valid")
        citations = ("معتبر" if out.get("citation_view", {}).get("items") else "بدون ارجاع") if valid else None
        sufficiency = STATUS.get(decision)
        if not sufficiency and out.get("extraction", {}).get("insufficient_evidence"):
            sufficiency = "ناکافی طبق استخراج"
        table.append({"روش": METHODS[r["method"]], "وضعیت": RESULT_STATUS[r["status"]],
                      "زمان (ثانیه)": round(r["seconds"], 3), "قطعهٔ اولیه": len(r["initial"]) if "initial" in r else None,
                      "شواهد نهایی": len(out["evidence"]) if "evidence" in out else None,
                      "توکن ورودی": usage.get("prompt_tokens"), "توکن خروجی": usage.get("completion_tokens"),
                      "اعتبار ساختاری": citations, "کفایت": sufficiency,
                      "دور تکمیلی": trace.get("iteration_count") if trace.get("enabled") else None})
    data_table(table, "comparison_table_page")
    st.caption("زمان هر روش RAG شامل سهم کامل بازیابی اولیهٔ مشترک است. خط تیره یعنی اندازه‌گیری نشده یا نامرتبط. "
               "تعرفهٔ تأییدشدهٔ ارائه‌دهنده در سامانه ثبت نشده است.")
    layout = st.radio("چیدمان پاسخ‌ها", ["کنار هم", "زبانه‌ای"], horizontal=True, key="compare_layout")
    if layout == "کنار هم":
        for start in range(0, len(results), 2):
            pair = results[start:start + 2]
            for column, r in zip(st.columns(len(pair)), pair):
                with column:
                    st.subheader(METHODS[r["method"]])
                    result(r, run["id"] + r["method"], research)
    else:
        for tab, r in zip(st.tabs([METHODS[r["method"]] for r in results]), results):
            with tab:
                result(r, run["id"] + r["method"], research)
    if run.get("scope"):
        metadata({"scope": "، ".join(run["scope"]["document_ids"])}, {"scope": "اسناد انتخابی"})
    left, right = st.columns(2)
    left.download_button("دریافت مقایسهٔ JSON", json_bytes(run), "comparison.json", "application/json")
    right.download_button("دریافت گزارش متنی", comparison_text(run), "comparison.txt", "text/plain")
