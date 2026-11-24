from base64 import b64encode
from functools import lru_cache
from pathlib import Path

import streamlit as st

ASSETS = Path(__file__).with_name("assets")
COLORS = {
    "روشن": ("#f6f8f8", "#ffffff", "#102f3b", "#48616b", "#cedbd9", "#09685e", "#e1f0e9", "#f0f4f2"),
    "تیره": ("#101b23", "#172832", "#eef5f3", "#b2c7c7", "#38515a", "#88dbc1", "#24443f", "#1e313a"),
}


@lru_cache(maxsize=2)
def stylesheet(theme):
    keys = ("bg", "surface", "ink", "muted", "line", "accent", "tint", "soft")
    variables = ";".join(f"--ci-{k}:{v}" for k, v in zip(keys, COLORS[theme]))
    font = b64encode((ASSETS / "Vazirmatn.woff2").read_bytes()).decode("ascii")
    css = (ASSETS / "workspace.css").read_text(encoding="utf-8")
    return css.replace("FONT_DATA", font).replace("PALETTE", variables).replace(
        "COLOR_SCHEME", "dark" if theme == "تیره" else "light")


def apply(theme):
    st.markdown("<style>" + stylesheet(theme) + "</style>", unsafe_allow_html=True)


def brand():
    # All markup in this module is static.
    st.markdown('<div class="ci-brand" dir="ltr"><span class="ci-mark" aria-hidden="true">'
                '<i></i><i></i><i></i></span><strong>Codal Intelligence</strong></div>'
                '<div class="ci-subtitle">سامانه هوشمند تحلیل گزارش‌های مالی</div>',
                unsafe_allow_html=True)
