"""Keep durable session data independent of Streamlit's page-local widget cleanup."""
from uuid import uuid4

PAGES = {
    "assistant": "تحلیل هوشمند", "comparison": "مقایسه روش‌ها", "evaluation": "ارزیابی پژوهشی",
    "documents": "مدیریت اسناد", "settings": "تنظیمات",
}
METHODS = {
    "llm_only": "مدل زبانی بدون RAG", "baseline": "RAG ساده",
    "proposed": "RAG ساختارآگاه", "egar": "RAG ساختارآگاه + EGAR",
}
LEVELS = {"explicit": "اطلاعات صریح", "implicit": "اطلاعات ضمنی", "latent": "اطلاعات نهفته"}
STATUS = {"sufficient": "کافی", "partially_sufficient": "تا حدی کافی", "contradictory": "متناقض",
          "unverifiable": "تأییدناپذیر", "insufficient": "ناکافی", "unassessed": "ارزیابی نشده"}


def new_id():
    return uuid4().hex


def initialize(state):
    if "conversations" not in state:
        state.conversations = {}
        new_conversation(state)
    for key, value in dict(mode="تحلیلگر", theme="روشن", profiles=None, overrides={},
                           comparison=None, selected_page="assistant").items():
        state.setdefault(key, value)


def new_conversation(state):
    key = new_id()
    state.conversations[key] = []
    state.current_conversation = key


def clear_conversation(state):
    # Fresh widget keys also clear the composer and the selected sources.
    del state.conversations[state.current_conversation]
    new_conversation(state)
