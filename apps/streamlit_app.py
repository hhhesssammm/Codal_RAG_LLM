"""Run with: python -m streamlit run apps/streamlit_app.py"""
from __future__ import annotations

import json
import os
from dataclasses import replace

import streamlit as st

from codal_rag.config import load_config
from codal_rag.retrieval.engine import DualRetrievalEngine
from codal_rag.extraction.citation_validation import validate_citation_view


@st.cache_resource(show_spinner='در حال بارگذاری مدل و ایندکس‌ها...')
def load_engine(config):
    return DualRetrievalEngine(config=config)


def render_result_card(result: dict) -> None:
    chunk = result['chunk']
    st.markdown(f"**رتبه {result['rank']}** · {result['score']:.4f}")
    st.caption(' > '.join(chunk.get('heading_path') or []))
    st.write(chunk.get('text') or chunk.get('content') or '')
    with st.expander('مشخصات منبع'):
        st.json({key: chunk.get(key) for key in ('document_id', 'chunk_id', 'source_file',
            'company', 'year', 'document_type', 'section', 'subsection', 'chunk_type', 'source_block_ids')})
    st.divider()


def main() -> None:
    st.set_page_config(page_title='Codal RAG', page_icon='📊', layout='wide')
    st.markdown('<style>.stMain {direction:rtl;text-align:right}</style>', unsafe_allow_html=True)
    st.title('مقایسه بازیابی گزارش‌های کدال')
    st.write('Baseline و Proposed با مدل، بردار پرس‌وجو و Top-K یکسان مقایسه می‌شوند.')
    config_text = st.sidebar.text_input('فایل‌های YAML (با کاما جدا کنید)', value=os.getenv('CODAL_CONFIG', ''))
    try:
        cfg = load_config([p.strip() for p in config_text.split(',') if p.strip()])
    except (ValueError, OSError) as exc:
        st.error(str(exc))
        return
    if cfg.embedding.backend == 'hash':
        st.warning('حالت smoke: امتیازها برای آزمون اتصال مراحل‌اند و معیار کیفیت معنایی نیستند.')
    adaptive = st.sidebar.checkbox('فعال‌سازی بازیابی هدایت‌شده با کمبود شواهد (EGAR)', value=cfg.egar.enabled)
    retrieval_mode = st.sidebar.selectbox('روش بازیابی', ['dense', 'hybrid', 'bm25'],
                                         index=['dense', 'hybrid', 'bm25'].index(cfg.retrieval.mode),
                                         format_func=lambda value: {'dense': 'معنایی', 'hybrid': 'ترکیبی', 'bm25': 'واژگانی (BM25)'}[value])
    cfg = replace(cfg, egar=replace(cfg.egar, enabled=adaptive),
                  retrieval=replace(cfg.retrieval, mode=retrieval_mode))
    inspect = st.sidebar.checkbox('نمای پژوهشی', value=False)
    if st.sidebar.button('بارگذاری دوباره ایندکس‌ها'):
        load_engine.clear()
    with st.form('retrieval_form'):
        query = st.text_input('سؤال کاربر', value='')
        top_k = st.number_input('Top-K', min_value=1, max_value=max(100, cfg.top_k), value=cfg.top_k, step=1)
        submitted = st.form_submit_button('جست‌وجو')
    missing = any(not cfg.path('vector_db', m, 'config.json').exists() for m in ('baseline', 'proposed'))
    if missing:
        st.info('ابتدا هر دو ایندکس را با دستور codal-rag build یا index بسازید.')
    elif submitted:
        try:
            engine = load_engine(cfg)
            result = engine.search(query, int(top_k))
            st.caption('پرس‌وجوی نرمال‌شده: ' + result.normalized_query)
            for column, method in zip(st.columns(2), ('baseline', 'proposed')):
                with column:
                    st.header(method.title())
                    hits = getattr(result, method)
                    if not hits:
                        st.info('نتیجه‌ای یافت نشد.')
                    for hit in hits:
                        render_result_card(hit)
            st.download_button('دریافت نتایج JSON',
                json.dumps(engine.result_payload(result), ensure_ascii=False, indent=2),
                file_name='retrieval.json', mime='application/json')
            if adaptive:
                from codal_rag.pipelines.extract import extract_retrieval
                with st.spinner('بررسی کفایت شواهد...'):
                    answer = extract_retrieval(engine.result_payload(result), 'proposed', cfg, engine=engine)
                final = answer['egar']['final_sufficiency_decision']
                st.subheader('نتیجهٔ بررسی شواهد')
                status_label = {'sufficient': 'کافی', 'partially_sufficient': 'نسبتاً کافی',
                                'insufficient': 'ناکافی', 'contradictory': 'متناقض',
                                'unverifiable': 'قابل تأیید نیست'}[final['sufficiency_status']]
                st.write(f"کفایت شواهد: {status_label} · دور تکمیلی: {answer['egar']['iteration_count']}")
                if answer['extraction']['insufficient_evidence']:
                    reason = answer['egar']['stop_reason']
                    message = {'ambiguous_scope': 'شرکت و دورهٔ گزارش روشن نیست؛ آن‌ها را در سؤال مشخص کنید.',
                               'service_or_validation_failure': 'بررسی شواهد به‌دلیل خطای سرویس یا خروجی نامعتبر کامل نشد.',
                               'budget_or_timeout': 'مهلت یا بودجهٔ بررسی تمام شد؛ پاسخی تأیید نشد.',
                               'contradictory_evidence': 'شواهد متناقض است؛ نتیجه‌ای تأیید نشد.'}
                    st.info(message.get(reason, 'شواهد کافی برای پاسخ مستند فراهم نشد.'))
                for level in ('explicit', 'implicit', 'latent'):
                    for claim in answer['extraction'][level]:
                        st.write(claim['claim'])
                        st.caption(' · '.join(claim.get('conceptual_labels', [])) + ' | ' + ', '.join(claim['evidence_ids']))
                with st.expander('شواهد و ارجاع‌ها'):
                    st.json(answer['citation_view'])
                if inspect:
                    with st.expander('کمبود شواهد و جست‌وجوهای تکمیلی'):
                        st.json(answer['egar'])
                st.download_button('دریافت پاسخ و شواهد JSON', json.dumps(answer, ensure_ascii=False, indent=2),
                                   file_name='extraction.json', mime='application/json')
        except (ValueError, OSError, RuntimeError) as exc:
            st.error(str(exc))
    with st.expander('نمایش ارجاع‌های خروجی استخراج'):
        uploaded = st.file_uploader('فایل JSON استخراج', type=['json'])
        if uploaded:
            try:
                payload = json.loads(uploaded.getvalue())
                citations = payload['citation_view']
                validation = validate_citation_view(citations)
                if not validation['valid']:
                    st.error('ارجاع نامعتبر: ' + '; '.join(validation['errors']))
                else:
                    st.json(citations)
                    st.caption('اعتبار ساختاری ارجاع به معنی تأیید صحت معنایی ادعا نیست.')
            except (ValueError, KeyError, TypeError):
                st.error('فایل باید خروجی JSON معتبر دستور extract باشد.')


if __name__ == '__main__':
    main()
