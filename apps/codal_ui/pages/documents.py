import streamlit as st

from ..analysis import INDEX_FILES
from ..catalog import catalog_for, original
from ..components import META, intro, metadata, passage, source
from ..safety import inside, public, read_object

PAGE = 15


def render(cfg, research):
    intro("مدیریت اسناد", "گزارش، دورهٔ مالی و مرز قطعه‌هایی را که در بازیابی استفاده می‌شوند بررسی کنید.")
    cats, manifests = {}, {}
    for method in ("baseline", "proposed"):
        cat = catalog_for(cfg, method)
        if cat:
            cats[method] = cat
        directory = inside(cfg.path(), "vector_db", method)
        ready = all(inside(cfg.path(), "vector_db", method, n).is_file() for n in INDEX_FILES)
        if ready:
            manifests[method] = read_object(inside(directory, "config.json"))
        st.caption(f"{method.title()} · " + ("فایل‌های ایندکس موجودند" if ready else "ایندکس کامل نیست"))
    if not cats:
        st.info("هنوز فهرست سندی موجود نیست. پس از اجرای دستور chunk یا build، گزارش‌ها در این صفحه دیده می‌شوند.")
        return
    docs = {doc: meta for cat in cats.values() for doc, meta in cat.docs.items()}
    counts = [("سند", len(docs)), ("شرکت ثبت‌شده", len({v["company"] for v in docs.values() if v.get("company")}))]
    for m, cat in cats.items():
        counts.append(("قطعهٔ " + m.title(), sum(map(len, cat.positions.values()))))
    for col, (label, val) in zip(st.columns(len(counts)), counts):
        col.metric(label, val)
    if manifests:
        with st.expander("مدل و منشأ ایندکس"):
            for method, manifest in manifests.items():
                st.caption(method.title())
                metadata(manifest, {"model_name": "مدل", "embedding_model": "مدل بردارسازی",
                         "embedding_dimension": "بُعد", "source_corpus_sha256": "اثر انگشت مجموعه"})
                if research:
                    st.json(public({k: v for k, v in manifest.items() if k != "documents"}))
    query = st.text_input("جست‌وجو در نام، شناسه یا نماد سند").strip().casefold()
    filters = {}
    for col, field in zip(st.columns(4), ("company", "year", "financial_period", "document_type")):
        options = sorted({v[field] for v in docs.values() if v.get(field)})
        if options:
            filters[field] = col.selectbox(META[field], [None, *options], key="doc_" + field,
                                           format_func=lambda x: x or "همه")
    matching = sorted(doc for doc, meta in docs.items() if
                      all(not v or meta.get(k) == v for k, v in filters.items())
                      and query in " ".join([doc, *map(str, meta.values())]).casefold())
    if not matching:
        st.info("سندی با این مشخصات پیدا نشد.")
        return
    pages = (len(matching) + PAGE - 1) // PAGE
    page = st.number_input("صفحهٔ اسناد", 1, pages, 1) if pages > 1 else 1
    doc = st.selectbox("انتخاب سند", matching[(page-1)*PAGE:page*PAGE])
    metadata({"document_id": doc, **docs[doc]})
    methods = [m for m in cats if doc in cats[m].docs]
    labels = {"baseline": "قطعه‌های ساده", "proposed": "قطعه‌های ساختارآگاه"}
    view = st.radio("نمای سند", ["مقایسهٔ قطعه‌ها", "متن اصلی"], horizontal=True)
    if view == "متن اصلی":
        text = original(cfg, doc, docs[doc], methods[-1])
        if text is None:
            st.info("فایل Markdown اصلی در پوشهٔ اسناد موجود نیست. متن قطعه‌های ثبت‌شده در نمای مقایسه در دسترس است.")
        else:
            passage(text)
        return
    for col, m in zip(st.columns(len(methods)), methods):
        with col:
            st.subheader(labels[m])
            cat = cats[m]
            n = len(cat.positions[doc])
            item = st.number_input(f"شمارهٔ قطعه از {n}", 1, n, 1, key=f"chunk_{doc}_{m}")
            row = cat.chunks(doc, item-1, 1)[0]
            metadata(row, {"token_count": "تعداد توکن", "chunk_type": "نوع قطعه", "source_block_ids": "بلوک‌های منبع",
                           "table_part": "بخش جدول", "table_parts": "تعداد بخش‌ها"})
            source(row)
