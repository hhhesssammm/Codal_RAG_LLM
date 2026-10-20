from __future__ import annotations

import streamlit as st

from config import (
    TOP_K,
)

from src.retrieval_engine import (
    DualRetrievalEngine,
    DualRetrievalResult,
)


# =========================================================
# Page configuration
# =========================================================

st.set_page_config(

    page_title=(
        "Codal RAG Retrieval Demo"
    ),

    page_icon="📊",

    layout="wide",
)


# =========================================================
# RTL / Persian styling
# =========================================================

st.markdown(
    """
    <style>

    html, body,
    [class*="css"] {
        direction: rtl;
        text-align: right;
    }

    .block-container {
        max-width: 1500px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .result-title {
        font-size: 1.05rem;
        font-weight: 700;
        margin-bottom: 0.35rem;
    }

    .path-box {
        direction: rtl;
        text-align: right;
        background-color: rgba(128,128,128,0.08);
        padding: 0.65rem;
        border-radius: 0.5rem;
        margin-top: 0.4rem;
        margin-bottom: 0.6rem;
    }

    .chunk-text {
        direction: rtl;
        text-align: right;
        line-height: 2;
        white-space: pre-wrap;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# Load expensive resources once
# =========================================================

@st.cache_resource(
    show_spinner=(
        "در حال بارگذاری مدل Embedding "
        "و پایگاه‌های برداری..."
    )
)
def load_engine():

    return DualRetrievalEngine()


# =========================================================
# Helpers
# =========================================================

def get_chunk_text(
    chunk: dict
) -> str:

    return (
        chunk.get("text")
        or chunk.get("content")
        or ""
    )


def get_chunk_type(
    chunk: dict
) -> str:

    return (
        chunk.get("chunk_type")
        or chunk.get("type")
        or "unknown"
    )


def get_heading_path(
    chunk: dict
) -> list[str]:

    return (
        chunk.get("heading_path")
        or []
    )


# =========================================================
# Result card
# =========================================================

def render_result_card(
    result: dict,
    *,
    show_structure: bool,
):

    chunk = result[
        "chunk"
    ]

    rank = result[
        "rank"
    ]

    score = result[
        "score"
    ]

    chunk_id = chunk.get(
        "chunk_id",
        "-"
    )

    chunk_type = (
        get_chunk_type(
            chunk
        )
    )

    text = get_chunk_text(
        chunk
    )

    heading_path = (
        get_heading_path(
            chunk
        )
    )

    # -----------------------------------------------------
    # Card
    # -----------------------------------------------------

    with st.container(
        border=True
    ):

        st.markdown(
            f"### رتبه {rank}"
        )

        metric_col1, metric_col2 = (
            st.columns(2)
        )

        with metric_col1:

            st.metric(
                "Similarity",
                f"{score:.4f}",
            )

        with metric_col2:

            st.metric(
                "نوع Chunk",
                chunk_type,
            )

        st.caption(
            f"Chunk ID: {chunk_id}"
        )

        # -------------------------------------------------
        # Structural information
        # -------------------------------------------------

        if (
            show_structure
            and heading_path
        ):

            path = " > ".join(
                heading_path
            )

            st.markdown(
                f"""
                <div class="path-box">
                    <strong>مسیر ساختاری:</strong><br>
                    {path}
                </div>
                """,
                unsafe_allow_html=True,
            )

        company = (
            chunk.get(
                "company"
            )
        )

        year = (
            chunk.get(
                "year"
            )
        )

        document_type = (
            chunk.get(
                "document_type"
            )
        )

        if (
            show_structure
            and (
                company
                or year
                or document_type
            )
        ):

            metadata_parts = []

            if company:

                metadata_parts.append(
                    f"شرکت: {company}"
                )

            if year:

                metadata_parts.append(
                    f"سال: {year}"
                )

            if document_type:

                metadata_parts.append(
                    f"نوع گزارش: {document_type}"
                )

            st.caption(
                " | ".join(
                    metadata_parts
                )
            )

        # -------------------------------------------------
        # Chunk text
        # -------------------------------------------------

        st.markdown(
            "**متن بازیابی‌شده:**"
        )

        st.markdown(
            f"""
            <div class="chunk-text">
            {text}
            </div>
            """,
            unsafe_allow_html=True,
        )

        # -------------------------------------------------
        # Debug information
        # -------------------------------------------------

        with st.expander(
            "جزئیات فنی Chunk"
        ):

            st.write(
                "Vector ID:",
                result.get(
                    "vector_id"
                ),
            )

            st.write(
                "Document ID:",
                chunk.get(
                    "document_id"
                ),
            )

            st.write(
                "Token Count:",
                chunk.get(
                    "token_count"
                ),
            )

            st.write(
                "Section:",
                chunk.get(
                    "section"
                ),
            )

            st.write(
                "Subsection:",
                chunk.get(
                    "subsection"
                ),
            )

            if (
                chunk.get(
                    "embedding_text"
                )
            ):

                st.markdown(
                    "**متنی که واقعاً Embed شده:**"
                )

                st.code(
                    chunk[
                        "embedding_text"
                    ]
                )


# =========================================================
# Result columns
# =========================================================

def render_comparison(
    result: DualRetrievalResult,
):

    # -----------------------------------------------------
    # Query information
    # -----------------------------------------------------

    st.markdown("---")

    st.subheader(
        "اطلاعات Query"
    )

    query_col1, query_col2 = (
        st.columns(2)
    )

    with query_col1:

        st.markdown(
            "**سؤال اصلی**"
        )

        st.write(
            result.original_query
        )

    with query_col2:

        st.markdown(
            "**سؤال پس از نرمال‌سازی**"
        )

        st.write(
            result.normalized_query
        )

    st.markdown("---")

    # -----------------------------------------------------
    # Two-system comparison
    # -----------------------------------------------------

    baseline_column, proposed_column = (
        st.columns(
            [1, 1],
            gap="large",
        )
    )

    # =====================================================
    # BASELINE
    # =====================================================

    with baseline_column:

        st.header(
            "Baseline Retrieval"
        )

        st.caption(
            "Fixed-size Chunking"
        )

        st.metric(
            "تعداد نتایج",
            len(
                result.baseline
            ),
        )

        if not result.baseline:

            st.warning(
                "نتیجه‌ای بازیابی نشد."
            )

        for item in result.baseline:

            render_result_card(
                item,
                show_structure=False,
            )

    # =====================================================
    # PROPOSED
    # =====================================================

    with proposed_column:

        st.header(
            "Markdown-Aware Retrieval"
        )

        st.caption(
            "Structure-Aware Chunking"
        )

        st.metric(
            "تعداد نتایج",
            len(
                result.proposed
            ),
        )

        if not result.proposed:

            st.warning(
                "نتیجه‌ای بازیابی نشد."
            )

        for item in result.proposed:

            render_result_card(
                item,
                show_structure=True,
            )


# =========================================================
# Main UI
# =========================================================

st.title(
    "مقایسه Retrieval در RAG گزارش‌های کدال"
)

st.write(
    """
    این دمو نتایج بازیابی دو Pipeline را برای یک Query یکسان
    مقایسه می‌کند. هر دو سیستم از یک مدل Embedding، یک Query
    embedding و مقدار Top-K یکسان استفاده می‌کنند.
    """
)


# =========================================================
# Engine
# =========================================================

try:

    engine = load_engine()

except Exception as error:

    st.error(
        "خطا در بارگذاری Retrieval Engine"
    )

    st.exception(
        error
    )

    st.stop()


# =========================================================
# Search form
# =========================================================

with st.form(
    "retrieval_form"
):

    query = st.text_input(

        "سؤال کاربر",

        value=(
            "علت کاهش فروش شرکت "
            "در سال ۱۴۰۳ چیست؟"
        ),
    )

    top_k = st.slider(

        "Top-K",

        min_value=1,

        max_value=10,

        value=TOP_K,

        step=1,
    )

    submitted = (
        st.form_submit_button(
            "اجرای Retrieval",
            type="primary",
        )
    )


# =========================================================
# Run retrieval
# =========================================================

if submitted:

    if not query.strip():

        st.warning(
            "لطفاً سؤال را وارد کنید."
        )

        st.stop()

    with st.spinner(
        "در حال بازیابی از هر دو سیستم..."
    ):

        try:

            result = engine.search(

                query=query,

                top_k=top_k,
            )

        except Exception as error:

            st.error(
                "خطا هنگام Retrieval"
            )

            st.exception(
                error
            )

            st.stop()

    # -----------------------------------------------------
    # Save experiment automatically
    # -----------------------------------------------------

    output_path = (
        engine.save_result(
            result
        )
    )

    # -----------------------------------------------------
    # Display
    # -----------------------------------------------------

    render_comparison(
        result
    )

    st.success(
        "نتیجه Retrieval ذخیره شد."
    )

    st.caption(
        f"Experiment file: {output_path}"
    )