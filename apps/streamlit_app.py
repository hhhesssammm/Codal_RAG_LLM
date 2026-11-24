"""Launch from the checkout with python -m streamlit run apps/streamlit_app.py."""
import streamlit as st

from codal_ui import theme
from codal_ui.components import error_box
from codal_ui.configuration import config_for
from codal_ui.pages import assistant, comparison, documents, evaluation, settings
from codal_ui.safety import problem
from codal_ui.state import PAGES, initialize

RENDER = {"assistant": assistant.render, "comparison": comparison.render, "documents": documents.render,
          "evaluation": evaluation.render, "settings": settings.render}


def main():
    st.set_page_config(page_title="Codal Intelligence", page_icon="📑", layout="wide")
    state = st.session_state
    initialize(state)
    theme.apply(state.theme)
    with st.sidebar:
        theme.brand()
        page = st.radio("فضای کار", list(PAGES), index=list(PAGES).index(state.selected_page),
                        format_func=PAGES.get, key="ci_navigation")
        state.selected_page = page
        st.divider()
        st.caption("نمای فعال · " + state.mode)
    research = state.mode == "پژوهشگر"
    try:
        cfg = config_for(state.profiles, state.overrides)
    except (ValueError, OSError) as exc:
        error_box(problem(exc), research)
        settings.render(None, research)
        return
    try:
        RENDER[page](cfg, research)
    except (ValueError, OSError, TypeError, KeyError, AttributeError, IndexError, RecursionError, StopIteration) as exc:
        error_box(problem(exc, "page"), research)


if __name__ == "__main__":
    main()
