"""Read-only catalog. Scan a JSON container once; retain only document metadata and byte ranges."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from codal_rag.retrieval.scope import RetrievalScope
from codal_rag.storage.json_io import digest
from .analysis import file_stamp
from .safety import UIError, inside, read_object

FIELDS = ("company", "year", "financial_period", "document_type", "symbol", "source_file", "source_url",
          "document_title", "publication_date")
CHUNK = 64 * 1024


def records(path):
    """Stream array elements or vector-ID/object members, including compact UTF-8 JSON."""
    decoder = json.JSONDecoder()
    with path.open(encoding="utf-8", newline="") as src:
        buffer, offset, end = "", 0, False

        def fill():
            nonlocal buffer, end
            block = src.read(CHUNK)
            buffer += block
            end = not block

        def advance(n):
            nonlocal offset, buffer
            offset += len(buffer[:n].encode("utf-8"))
            buffer = buffer[n:]

        def peek():
            while True:
                advance(len(buffer) - len(buffer.lstrip()))
                if buffer or end:
                    return buffer[:1]
                fill()

        def token(c):
            if peek() != c:
                raise UIError("قالب فهرست قطعه‌ها معتبر نیست.")
            advance(1)

        def item():
            while True:
                try:
                    value, n = decoder.raw_decode(buffer)
                    start = offset
                    advance(n)
                    return value, start, offset - start
                except json.JSONDecodeError:
                    if end or len(buffer) > 16 * 1024 * 1024:
                        raise UIError("یک رکورد در فهرست قطعه‌ها نامعتبر یا بیش از حد بزرگ است.") from None
                    fill()

        opening = peek()
        if opening not in ("[", "{"):
            raise UIError("فهرست قطعه‌ها معتبر نیست.")
        token(opening)
        closing = "]" if opening == "[" else "}"
        first = True
        while peek() != closing:
            if not first:
                token(",")
            first = False
            if opening == "{":
                peek()
                key, _, _ = item()
                if not isinstance(key, str):
                    raise UIError("شناسهٔ رکورد معتبر نیست.")
                token(":")
            peek()
            row, offset_start, length = item()
            if not isinstance(row, dict) or not isinstance(row.get("document_id"), str):
                raise UIError("شناسهٔ سند در فهرست قطعه‌ها موجود نیست.")
            yield row, offset_start, length
        token(closing)
        if peek():
            raise UIError("دادهٔ اضافی پس از فهرست قطعه‌ها دیده شد.")


@dataclass
class Catalog:
    path: Path
    version: tuple
    docs: dict = field(default_factory=dict)
    positions: dict = field(default_factory=dict)

    def chunks(self, document, start, count):
        with self.path.open("rb") as src:
            if file_stamp(self.path) != self.version:
                raise UIError("نسخهٔ قطعه‌ها تغییر کرد. صفحه را دوباره باز کنید.")
            rows = []
            for offset, size in self.positions.get(document, [])[start:start + count]:
                src.seek(offset)
                rows.append(json.loads(src.read(size)))
            return rows


@lru_cache(maxsize=6)
def read_catalog(path, version):
    catalog = Catalog(path, version)
    for row, offset, length in records(path):
        doc = row["document_id"]
        catalog.docs.setdefault(doc, {k: row[k] for k in FIELDS if row.get(k)})
        catalog.positions.setdefault(doc, []).append((offset, length))
    if file_stamp(path) != version:
        raise UIError("نسخهٔ فهرست حین خواندن تغییر کرد.")
    return catalog


def catalog_for(cfg, method):
    for parts in (("vector_db", method, "metadata.json"), ("chunks", method, "corpus.json")):
        path = inside(cfg.path(), *parts)
        if path.is_file():
            return read_catalog(path, file_stamp(path))
    return None


def scope_for(docs, selections):
    selected = {k: v for k, v in selections.items() if v}
    if not selected:
        return None
    matches = [doc for doc, meta in docs.items() if all(meta.get(k) == v for k, v in selected.items())]
    if not matches:
        raise UIError("هیچ سندی با این ترکیب فیلترها مطابقت ندارد.")
    # A report type has no native scope field. Resolve it to exact indexed document IDs.
    return RetrievalScope(document_ids=tuple(sorted(matches)),
                          companies=(selected["company"],) if "company" in selected else (),
                          years=(selected["year"],) if "year" in selected else (),
                          periods=(selected["financial_period"],) if "financial_period" in selected else ())


def original(cfg, doc, metadata, method):
    root = Path(os.getenv("CODAL_UI_DOCUMENT_ROOT") or cfg.path("documents"))
    source = metadata.get("source_file")
    if not source:
        return None
    path = inside(root, source)
    if path.suffix.lower() != ".md" or not path.is_file():
        return None
    if path.stat().st_size > 4 * 1024 * 1024:
        raise UIError("سند برای پیش‌نمایش کامل بزرگ است. از نمای قطعه‌ها استفاده کنید.")
    text = path.read_text(encoding="utf-8")
    manifest = inside(cfg.path(), "chunks", method, "manifest.json")
    if not manifest.is_file():
        raise UIError("اثر انگشت متن اصلی برای تطبیق با قطعه‌ها در دسترس نیست.")
    source_manifest = read_object(manifest)
    index = inside(cfg.path(), "vector_db", method, "config.json")
    if index.is_file() and read_object(index).get("source_corpus_sha256") != source_manifest.get("source_sha256"):
        raise UIError("نسخهٔ ایندکس و مجموعهٔ اسناد اصلی متفاوت است.")
    if source_manifest.get("documents", {}).get(doc) != digest(text):
        raise UIError("متن اصلی با نسخهٔ ثبت‌شده هنگام قطعه‌بندی مطابقت ندارد.")
    return text
