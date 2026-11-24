import streamlit as st

from ..components import error_box, intro, metadata
from ..configuration import config_for, profile_choices
from ..safety import problem, public


def appearance():
    state = st.session_state
    st.subheader("نمایش")
    for col, (label, key, choices) in zip(st.columns(2), [
            ("حالت نمایش", "mode", ["تحلیلگر", "پژوهشگر"]), ("پوسته", "theme", ["روشن", "تیره"])]):
        widget = "appearance_" + key
        col.radio(label, choices, index=choices.index(state[key]), horizontal=True, key=widget,
                  on_change=lambda k=key, w=widget: state.update({k: state[w]}))
    st.caption("تغییر نما بلافاصله اعمال می‌شود. پیکربندی آزمایش و پاسخ‌های ذخیره‌شده به اجرای خود وابسته‌اند.")


def render(cfg, research):
    state = st.session_state
    intro("تنظیمات", "نمای برنامه و پیکربندی اجرای بعدی را مدیریت کنید.")
    appearance()
    with st.expander("پیکربندی پیشرفته و ترتیب YAML", expanded=cfg is None):
        options = profile_choices()
        names = st.multiselect("پروفایل‌ها به ترتیب اعمال", list(options), default=state.profiles or
                               (["پیکربندی محیط اجرا"] if "پیکربندی محیط اجرا" in options else []))
        st.caption("ابتدا پیش‌فرض‌ها، سپس YAML، متغیرهای محیطی و در پایان تنظیمات همین نشست اعمال می‌شوند.")
        if st.button("اعمال پروفایل‌ها و بازنشانی تنظیمات نشست"):
            try:
                config_for(names, {})
                state.profiles, state.overrides = names, {}
                state["settings_revision"] = state.get("settings_revision", 0) + 1
                st.rerun()
            except (ValueError, OSError) as exc:
                error_box(problem(exc), research)
    if cfg is None:
        return
    st.subheader("مدل و ارائه‌دهنده")
    metadata({"provider": cfg.llm.provider, "model": cfg.llm.model or "تنظیم نشده",
              "embedding": cfg.embedding.model, "key": "تنظیم شده" if cfg.llm.api_key else "تنظیم نشده"},
             {"model": "مدل زبانی", "provider": "ارائه‌دهنده", "key": "کلید محیط اجرا", "embedding": "مدل بردارسازی"})
    st.caption("اتصال فقط هنگام درخواست واقعی آزموده می‌شود. تغییر مدل بردارسازی در YAML نیازمند بازسازی هر دو ایندکس است.")
    with st.form("settings_" + str(state.get("settings_revision", 0))):
        model = st.text_input("مدل زبانی برای اجرای بعدی", value=cfg.llm.model)
        left, right = st.columns(2)
        temp = left.number_input("دمای تولید", 0.0, 2.0, float(cfg.llm.temperature), step=0.1)
        tokens = right.number_input("سقف توکن خروجی", min_value=1, value=cfg.llm.max_output_tokens)
        st.subheader("بازیابی و بودجه")
        left, right = st.columns(2)
        mode = left.selectbox("روش بازیابی", ["dense", "bm25", "hybrid"],
                              index=["dense", "bm25", "hybrid"].index(cfg.retrieval.mode),
                              format_func=lambda x: {"dense": "معنایی", "bm25": "واژگانی (BM25)", "hybrid": "ترکیبی (RRF)"}[x])
        k = right.number_input("تعداد قطعه‌های اولیه (Top-K)", min_value=1, value=cfg.top_k)
        candidates = st.number_input("عمق نامزدهای بازیابی", min_value=1, value=cfg.retrieval.candidate_k)
        adaptive = st.checkbox("استفاده از EGAR برای دستیار", value=cfg.egar.enabled)
        a, b, c = st.columns(3)
        rounds = a.number_input("سقف دور تکمیلی", 0, 10, cfg.egar.max_iterations)
        queries = b.number_input("پرسش در هر دور", min_value=1, value=cfg.egar.max_queries_per_iteration)
        extra = c.number_input("قطعه در جست‌وجوی تکمیلی", min_value=1, value=cfg.egar.supplementary_top_k)
        a, b, c = st.columns(3)
        budget = a.number_input("سقف نامزدهای EGAR", min_value=1, value=cfg.egar.max_candidates)
        calls = b.number_input("سقف فراخوانی مدل", min_value=1, value=cfg.egar.max_llm_calls)
        timeout = c.number_input("مهلت EGAR (ثانیه)", min_value=0.1, value=float(cfg.egar.timeout_seconds))
        submit = st.form_submit_button("ذخیره برای اجرای بعدی", type="primary")
    if submit:
        changes = {"llm": {"model": model.strip(), "temperature": temp, "max_output_tokens": tokens},
                   "top_k": k, "retrieval": {"mode": mode, "candidate_k": candidates},
                   "egar": {"enabled": adaptive, "max_iterations": rounds, "max_queries_per_iteration": queries,
                            "supplementary_top_k": extra, "max_candidates": budget, "max_llm_calls": calls,
                            "timeout_seconds": timeout}}
        try:
            cfg = config_for(state.profiles, changes)
            state.overrides = changes
            st.success("تنظیمات این نشست ذخیره شد.")
        except (ValueError, OSError) as exc:
            error_box(problem(exc), research)
    st.caption("تغییر مدل، بودجه یا روش بر مقایسهٔ اجراهای جداگانه اثر دارد. بازیاب هنگام استفاده با پیکربندی و نسخهٔ ایندکس تطبیق داده می‌شود.")
    if research:
        with st.expander("مشاهدهٔ پیکربندی مؤثر"):
            st.json(public(cfg.public_dict()))
