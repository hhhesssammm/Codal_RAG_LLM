"""Only server-installed profiles and the operator's environment can select file roots."""
from __future__ import annotations

import os
from pathlib import Path

from codal_rag.config import load_config
from .safety import ROOT, UIError, inside


def profile_choices():
    directory = ROOT / "configs"
    choices = {p.name: inside(directory, p.name) for p in sorted(directory.glob("*.yaml"))
               if p.resolve().is_relative_to(directory.resolve())}
    if os.environ.get("CODAL_CONFIG"):
        choices["پیکربندی محیط اجرا"] = Path(os.environ["CODAL_CONFIG"])
    return choices


def config_for(profiles, overrides):
    options = profile_choices()
    if any(name not in options for name in profiles or []):
        raise UIError("پروفایل انتخاب‌شده در دسترس نیست.")
    return load_config([options[name] for name in profiles or []], overrides=overrides)
