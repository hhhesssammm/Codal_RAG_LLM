"""Boundary for UI-selected files, diagnostics and downloadable JSON."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path, PureWindowsPath
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[2]
MAX_UPLOAD = 24 * 1024 * 1024


class UIError(ValueError):
    """A fixed Persian message safe to show to an analyst."""


def inside(root: Path, *parts: str) -> Path:
    root = root.resolve()
    path = root.joinpath(*parts).resolve()
    if not path.is_relative_to(root):
        raise UIError("این فایل خارج از پوشهٔ مجاز است.")
    return path


def parse_object(data: bytes) -> dict:
    if len(data) > MAX_UPLOAD:
        raise UIError("حجم فایل از سقف ۲۴ مگابایت بیشتر است.")

    def invalid(_):
        raise ValueError("non-finite number")

    obj = json.loads(data.decode("utf-8-sig"), parse_constant=invalid)
    if not isinstance(obj, dict):
        raise UIError("ساختار فایل باید یک شیء JSON باشد.")
    return obj


def read_object(path: Path) -> dict:
    with path.open("rb") as source:
        return parse_object(source.read(MAX_UPLOAD + 1))


def codal_url(value):
    if not isinstance(value, str) or any(c.isspace() or ord(c) < 32 for c in value) or "\\" in value:
        return None
    try:
        parsed = urlsplit(value)
        host = (parsed.hostname or "").lower()
        if (parsed.scheme == "https" and (host == "codal.ir" or host.endswith(".codal.ir"))
                and not parsed.username and not parsed.password and parsed.port in (None, 443)):
            return value
    except ValueError:
        pass
    return None


def public(value):
    """Retain research settings, while removing private host and credential fields."""
    if isinstance(value, dict):
        return {k: public(v) for k, v in value.items()
                if k.lower() not in {"project_root", "data_dir", "base_url", "authorization", "password",
                                     "access_token", "refresh_token", "api_token", "headers", "token", "secret",
                                     "chain_of_thought", "reasoning_content", "private_reasoning"}
                and not re.search(r"api.?key|private.?key|password|secret_key", k, re.I)}
    if isinstance(value, (list, tuple)):
        return [public(v) for v in value]
    if isinstance(value, str):
        for key in ("CODAL_LLM_API_KEY", "OPENAI_API_KEY", "CODAL_LLM_BASE_URL", "OPENAI_BASE_URL"):
            secret = os.getenv(key)
            if secret and len(secret) >= 4:
                value = value.replace(secret, "••••")
        if "\n" not in value and (Path(value).is_absolute() or PureWindowsPath(value).is_absolute()):
            return PureWindowsPath(value).name
    return value


def json_bytes(obj) -> bytes:
    return json.dumps(public(obj), ensure_ascii=False, allow_nan=False, indent=2).encode("utf-8")


def problem(exc: Exception, stage: str = "") -> dict:
    # Provider errors, invalid YAML and OS exceptions may contain paths, keys or full request bodies.
    from codal_rag.config import ConfigurationError
    from codal_rag.llm.client import LLMError

    if isinstance(exc, UIError):
        message = str(exc)
    elif isinstance(exc, ConfigurationError):
        message = "پیکربندی معتبر نیست. پرونده‌ها، ترتیب اعمال و مقادیر تنظیمات را بررسی کنید."
    elif isinstance(exc, (ModuleNotFoundError, ImportError)):
        message = "بستهٔ لازم برای این عملیات در محیط پایتون موجود نیست."
    elif isinstance(exc, LLMError):
        detail = str(exc)
        if "API_KEY" in detail or "LLM_MODEL" in detail:
            message = "برای استخراج، نام مدل و کلید سرویس را در محیط اجرای برنامه تنظیم کنید."
        elif "token limit" in detail or "no choices" in detail or "declined" in detail:
            message = "مدل پاسخ قابل استفاده برنگرداند. سقف خروجی و وضعیت سرویس را بررسی کنید."
        else:
            message = "درخواست مدل کامل نشد. دسترسی شبکه، سهمیه و تنظیمات ارائه‌دهنده را بررسی کنید."
    elif "index" in str(exc).lower() and stage == "retrieve":
        message = "ایندکس‌ها در دسترس نیستند یا با پیکربندی فعلی هم‌خوانی ندارند."
    elif stage == "retrieve":
        message = "بازیابی انجام نشد. دسترسی به وزن‌های مدل و سازگاری هر دو ایندکس را بررسی کنید."
    elif stage == "extract":
        message = "استخراج کامل نشد. قالب پاسخ مدل، شناسه‌های شواهد یا اتصال سرویس را بررسی کنید."
    else:
        message = "خواندن داده امکان‌پذیر نبود. قالب، اندازه و کامل بودن فایل را بررسی کنید."
    return {"message": message, "type": type(exc).__name__, "stage": stage}
