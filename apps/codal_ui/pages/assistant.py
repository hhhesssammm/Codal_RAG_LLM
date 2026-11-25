from hashlib import sha256

import streamlit as st

from ..analysis import index_version
from ..jobs import submitted_run
from ..components import choose_scope, error_box, intro, prose, result
from ..presentation import saved_extraction
from ..safety import MAX_UPLOAD, parse_object, problem
from ..state import METHODS, clear_conversation, new_conversation

EXAMPLES = ["علت تغییرات فروش شرکت چیست؟", "چه نشانه‌هایی از ریسک نقدینگی وجود دارد؟",
            "عملکرد مالی شرکت نسبت به سال گذشته چگونه تغییر کرده است؟"]


def render(cfg, research):
    state = st.session_state
    cid = state.current_conversation
    history = state.conversations[cid]
    intro("تحلیل هوشمند", "از گزارش‌های مالی کدال بپرسید و هر یافته را تا متن و جدول منبع دنبال کنید.")
    key = "assistant_" + cid
    query_key = key + "_question"
    with st.form(key + "_form_" + str(len(history))):
        question = st.text_area("پرسش شما", height=116, key=query_key, max_chars=8000,
                                placeholder="دربارهٔ کدام تغییر، ریسک یا شاخص مالی می‌خواهید بدانید؟")
        submit = st.form_submit_button("تحلیل پرسش", type="primary")
    if not history:
        st.caption("پرسش‌های پیشنهادی")
        for col, example in zip(st.columns(len(EXAMPLES)), EXAMPLES):
            col.button(example, key=key + example, on_click=lambda text=example: state.update({query_key: text}),
                       use_container_width=True)
    scope = choose_scope(cfg, key)
    if not index_version(cfg):
        st.info("برای تحلیل گزارش‌ها ابتدا ایندکس‌های ساده و ساختارآگاه را آماده کنید. وضعیت آن‌ها در مدیریت اسناد دیده می‌شود.")
    method = "egar" if cfg.egar.enabled else "proposed"
    submitted_run(state, cid, submit, question, [method], cfg, scope)
    if history:
        st.divider()
        by_id = {r["id"]: r for r in reversed(history)}
        labels = {r["id"]: f"{i} · {r['question']}" for i, r in enumerate(history, 1)}
        selected = st.selectbox("تاریخچهٔ پرسش‌ها", list(by_id), format_func=labels.__getitem__,
                                key=key + "_selected")
        run = by_id[selected]
        method, record = next(iter(run["methods"].items()))
        prose(run["question"])
        st.caption(METHODS[method] + " · هر پرسش مستقل پردازش می‌شود.")
        result(record, run["id"], research)
    with st.expander("گفت‌وگوها و خروجی ذخیره‌شده"):
        left, right = st.columns(2)
        left.button("گفت‌وگوی جدید", key=key + "_new", on_click=new_conversation, args=(state,))
        right.button("پاک کردن این گفت‌وگو", key=key + "_clear", on_click=clear_conversation, args=(state,))
        previous = [k for k, values in state.conversations.items() if k != cid and values]
        if previous:
            titles = {k: f"{i} · {state.conversations[k][0]['question']}" for i, k in enumerate(previous, 1)}
            old = st.selectbox("گفت‌وگوهای پیشین", previous, format_func=titles.__getitem__)
            st.button("باز کردن گفت‌وگو", on_click=lambda: state.update(current_conversation=old))
        uploaded = st.file_uploader("بازخوانی خروجی استخراج (JSON)", type="json", key=key + "_upload")
        if uploaded:
            try:
                if uploaded.size > MAX_UPLOAD:
                    from ..safety import UIError
                    raise UIError("حجم فایل از سقف مجاز بیشتر است.")
                data = uploaded.getvalue()
                artifact = parse_object(data)
                record = saved_extraction(artifact)
                result(record, key + "_saved_" + sha256(data).hexdigest()[:12], research)
            except (ValueError, TypeError, KeyError, AttributeError, RecursionError) as exc:
                error_box(problem(exc, "artifact"), research)
