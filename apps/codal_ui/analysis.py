"""UI adapters. No data or configuration from one question is used to generate another."""
from __future__ import annotations

from dataclasses import replace
from functools import lru_cache
from threading import Lock
from time import monotonic

from codal_rag.config import AppConfig, validate_config
from codal_rag.egar.runtime import BudgetedLLM, QuestionBudget
from codal_rag.llm.client import LLMClient
from codal_rag.pipelines import extract as pipeline
from codal_rag.retrieval.engine import DualRetrievalEngine
from codal_rag.storage.json_io import digest
from .safety import UIError, inside, problem, public
from .state import METHODS, new_id

INDEX_FILES = ("config.json", "db_info.json", "metadata.json", "index.faiss")
STAGES = {"question": "پردازش سؤال", "retrieve": "بازیابی شواهد", "assess": "بررسی کفایت شواهد",
          "supplement": "بازیابی تکمیلی", "extract": "استخراج اطلاعات", "citations": "اعتبارسنجی ساختاری ارجاعات",
          "generate": "تولید پاسخ بدون سند"}
NO_CONTEXT_PROMPT = """به فارسی به سؤال مالی پاسخ بده. هیچ سند، ابزار یا دادهٔ شرکتی در اختیار تو نیست.
اطلاعات عمومی را می‌توانی توضیح دهی. اگر پاسخ دربارهٔ گزارش مشخص به داده نیاز دارد، محدودیت را بنویس.
واقعیت، استنباط و تفسیر احتمالی را از هم متمایز کن. عدد یا ارجاع نساز.
یک شیء JSON با فیلد answer (متن پاسخ) و limitations (آرایه‌ای از متن) برگردان."""


def file_stamp(path):
    s = path.stat()
    return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)


def index_version(cfg):
    root = cfg.path()
    try:
        return tuple(file_stamp(inside(root, "vector_db", m, name))
                     for m in ("baseline", "proposed") for name in INDEX_FILES)
    except FileNotFoundError:
        return ()


_load_lock = Lock()


@lru_cache(maxsize=4)
def _engine(cfg, version):
    engine = DualRetrievalEngine(config=cfg)
    if index_version(cfg) != version:
        raise UIError("ایندکس حین بارگذاری تغییر کرد. جست‌وجو را دوباره اجرا کنید.")
    return engine


def current_engine(cfg):
    version = index_version(cfg)
    if not version:
        raise UIError("ایندکس‌های ساده و ساختارآگاه کامل نیستند. ابتدا دستور ساخت ایندکس را برای مجموعهٔ گزارش‌ها اجرا کنید.")
    # A shared retriever must never retain the operator's LLM key or session-specific generation settings.
    resource_cfg = AppConfig(project_root=cfg.project_root, data_dir=cfg.data_dir, embedding=cfg.embedding,
                             retrieval=cfg.retrieval, top_k=cfg.top_k, random_seed=cfg.random_seed)
    with _load_lock:
        return _engine(resource_cfg, version)


def llm_only(question, cfg, client_factory=None):
    budget = QuestionBudget(cfg.egar)
    llm = replace(cfg.llm, max_retries=0, timeout=min(cfg.llm.timeout, cfg.egar.call_timeout_seconds))
    client = (client_factory or LLMClient)(llm)
    result = BudgetedLLM(client, budget).generate_json(NO_CONTEXT_PROMPT, question)
    if not isinstance(result.get("answer"), str) or not result["answer"].strip():
        raise ValueError("Missing answer")
    limits = result.get("limitations", [])
    if not isinstance(limits, list) or any(not isinstance(v, str) for v in limits):
        raise ValueError("Invalid limitations")
    return {"answer": result["answer"], "limitations": limits}, budget.metrics()


def effective(cfg, method, comparison):
    version = 2 if comparison or method == "egar" else cfg.extraction.schema_version
    return validate_config(replace(cfg, egar=replace(cfg.egar, enabled=method == "egar"),
                                   extraction=replace(cfg.extraction, schema_version=version)))


def execute(question, methods, cfg, *, scope=None, comparison=False, notify=None, engine_factory=None,
            client_factory=None):
    question = question.strip()
    if not question or len(question) > 8000:
        raise UIError("یک پرسش بین ۱ تا ۸۰۰۰ نویسه بنویسید.")
    if not methods or len(set(methods)) != len(methods) or any(m not in METHODS for m in methods):
        raise UIError("حداقل یک روش معتبر برای اجرا انتخاب کنید.")
    run = {"id": new_id(), "question": question, "methods": {}, "scope": scope.to_dict() if scope else None}
    shared = shared_error = engine = None
    retrieval_seconds = 0.0
    for method in methods:
        started = monotonic()
        record = {"method": method, "status": "failed", "stages": [], "usage": {},
                  "scope": run["scope"] if method != "llm_only" else None}

        def stage(name):
            record["stages"].append(name)
            if notify:
                notify(method, name)

        try:
            stage("question")
            config = effective(cfg, method, comparison)
            record["configuration"] = public(config.public_dict())
            runtime_llm = record["configuration"]["llm"]
            if comparison or config.extraction.schema_version == 2:
                runtime_llm.update(timeout=min(config.llm.timeout, config.egar.call_timeout_seconds),
                                   max_retries=0, budgeted_retries=config.egar.max_retries)
            record["execution"] = "live_provider_request" if method == "llm_only" else (
                "smoke_retrieval" if cfg.embedding.backend == "hash" else "configured_pipeline")
            if method == "llm_only":
                record["configuration"] = {"llm": runtime_llm,
                    "max_llm_calls": config.egar.max_llm_calls, "question_timeout": config.egar.timeout_seconds,
                    "per_call_timeout": min(config.llm.timeout, config.egar.call_timeout_seconds),
                    "retry_budget": config.egar.max_retries, "documents_supplied": 0}
                stage("generate")
                record["answer"], record["usage"] = llm_only(question, config, client_factory)
            else:
                if shared is None and shared_error is None:
                    stage("retrieve")
                    search_start = monotonic()
                    try:
                        engine = (engine_factory or current_engine)(cfg)
                        shared = engine.result_payload(engine.search(question, cfg.top_k, scope=scope))
                    except Exception as exc:
                        shared_error = problem(exc, "retrieve")
                    retrieval_seconds = monotonic() - search_start
                if shared_error:
                    record.update(error=shared_error, status="unavailable")
                    continue
                base = "proposed" if method == "egar" else method
                record.update(initial=shared[base], initial_sha256=digest(shared),
                              normalized_query=shared.get("normalized_query"), initial_seconds=retrieval_seconds,
                              index_provenance=shared.get("index_config", {}).get(base, {}))
                started = monotonic()
                result = pipeline.extract_retrieval(shared, base, config, scope=scope, engine=engine,
                    on_stage=stage, on_usage=record["usage"].update)
                record["output"] = result
                if "operational" in result:
                    record["usage"] = result["operational"]
                if result.get("egar", {}).get("stop_reason") in {"service_or_validation_failure", "budget_or_timeout"}:
                    record["status"] = "incomplete"
                    continue
            record["status"] = "complete"
        except Exception as exc:
            record["error"] = problem(exc, "extract")
            if isinstance(exc, (ImportError, UIError)) or "API_KEY" in str(exc) or "LLM_MODEL" in str(exc):
                record["status"] = "unavailable"
        finally:
            record["seconds"] = monotonic() - started + record.get("initial_seconds", 0.0)
            run["methods"][method] = record
    return public(run)
