"""Validated runtime configuration. Importing this module performs no I/O."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Mapping
import math
import os

import yaml


class ConfigurationError(ValueError):
    pass


@dataclass(frozen=True)
class ChunkingConfig:
    max_tokens: int = 500
    baseline_overlap: int = 100
    min_merge_tokens: int = 40
    include_path: bool = True


@dataclass(frozen=True)
class EmbeddingConfig:
    backend: str = "sentence_transformers"
    model: str = "intfloat/multilingual-e5-base"
    revision: str | None = None
    batch_size: int = 16
    normalize: bool = True
    similarity: str = "cosine"
    query_prefix: str = "query: "
    passage_prefix: str = "passage: "
    hash_dimension: int = 128
    tokenizer: str = "transformers"
    local_files_only: bool = False


@dataclass(frozen=True)
class LLMConfig:
    provider: str = "openai_compatible"
    model: str = ""
    base_url: str = field(default="https://api.openai.com/v1", repr=False)
    api_key: str = field(default="", repr=False)
    temperature: float = 0.0
    max_output_tokens: int = 2000
    timeout: float = 60.0
    max_retries: int = 2


@dataclass(frozen=True)
class EvaluationConfig:
    evidence_threshold: float = 0.70
    answer_threshold: float = 0.60
    semantic_threshold: float = 0.70
    faithfulness_threshold: float = 0.50
    semantic_enabled: bool = False
    semantic_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


@dataclass(frozen=True)
class RetrievalConfig:
    mode: str = "dense"
    candidate_k: int = 50
    rrf_k: int = 60
    bm25_k1: float = 1.5
    bm25_b: float = 0.75


@dataclass(frozen=True)
class ExtractionConfig:
    schema_version: int = 1
    # v1 remains unbounded for historical compatibility. v2 always uses this cap.
    max_evidence_tokens: int = 4096


@dataclass(frozen=True)
class EGARConfig:
    enabled: bool = False
    judge: str = "llm"
    max_iterations: int = 2
    max_queries_per_iteration: int = 2
    supplementary_top_k: int = 5
    max_candidates: int = 40
    max_llm_calls: int = 6
    timeout_seconds: float = 120.0
    call_timeout_seconds: float = 30.0
    max_retries: int = 0
    minimum_sufficiency: str = "sufficient"
    input_cost_per_million: float = 0.0
    output_cost_per_million: float = 0.0


@dataclass(frozen=True)
class AppConfig:
    project_root: Path = field(default_factory=lambda: Path.cwd().resolve())
    data_dir: str = "data"
    mode: str = "both"
    top_k: int = 5
    random_seed: int = 42
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    extraction: ExtractionConfig = field(default_factory=ExtractionConfig)
    egar: EGARConfig = field(default_factory=EGARConfig)

    def path(self, *parts: str) -> Path:
        return self.project_root / self.data_dir / Path(*parts)

    def public_dict(self) -> dict:
        result = asdict(self)
        result["project_root"] = str(self.project_root)
        result["llm"].pop("api_key")
        # URLs may contain credentials/query tokens. They are never persisted in reports.
        result["llm"].pop("base_url")
        return result


SECTIONS = {"chunking": ChunkingConfig, "embedding": EmbeddingConfig,
            "llm": LLMConfig, "evaluation": EvaluationConfig,
            "retrieval": RetrievalConfig, "extraction": ExtractionConfig, "egar": EGARConfig}


def _merge(base: dict, update: dict, prefix: str = "") -> None:
    for key, value in update.items():
        name = f"{prefix}{key}"
        if key not in base:
            raise ConfigurationError(f"Unknown configuration field: {name}")
        if isinstance(base[key], dict):
            if not isinstance(value, dict):
                raise ConfigurationError(f"{name} must be a mapping")
            _merge(base[key], value, name + ".")
        else:
            base[key] = value


def validate_config(cfg: AppConfig) -> AppConfig:
    defaults = AppConfig()
    for section in ("", *SECTIONS):
        value = getattr(cfg, section) if section else cfg
        default = getattr(defaults, section) if section else defaults
        for f in fields(value):
            actual, expected = getattr(value, f.name), getattr(default, f.name)
            if f.name in SECTIONS or f.name == "project_root":
                continue
            name = f"{section}.{f.name}".strip(".")
            if expected is None:
                valid = actual is None or isinstance(actual, str)
            elif isinstance(expected, float):
                valid = type(actual) in (int, float) and math.isfinite(actual)
            else:
                valid = type(actual) is type(expected)
            if not valid:
                raise ConfigurationError(f"Invalid type or non-finite value for {name}")
    c, e, llm = cfg.chunking, cfg.embedding, cfg.llm
    for name, value in {"top_k": cfg.top_k, "chunking.max_tokens": c.max_tokens,
                        "embedding.batch_size": e.batch_size, "embedding.hash_dimension": e.hash_dimension,
                        "llm.timeout": llm.timeout, "llm.max_output_tokens": llm.max_output_tokens}.items():
        if value <= 0:
            raise ConfigurationError(f"{name} must be positive")
    if not 0 <= c.baseline_overlap < c.max_tokens:
        raise ConfigurationError("chunking.baseline_overlap must be >= 0 and < max_tokens")
    if c.min_merge_tokens < 0 or llm.max_retries < 0 or cfg.random_seed < 0:
        raise ConfigurationError("Merge size, retries, and seed must be nonnegative")
    if cfg.mode not in {"both", "baseline", "proposed"}:
        raise ConfigurationError("mode must be both, baseline, or proposed")
    if e.backend not in {"sentence_transformers", "hash"}:
        raise ConfigurationError("embedding.backend must be sentence_transformers or hash")
    if e.tokenizer not in {"transformers", "whitespace"}:
        raise ConfigurationError("embedding.tokenizer must be transformers or whitespace")
    if (e.backend == "hash") != (e.tokenizer == "whitespace"):
        raise ConfigurationError("The smoke hash backend must use the whitespace tokenizer")
    if e.backend == "hash" and e.model != "smoke-hash-v1":
        raise ConfigurationError("Hash embeddings must be identified as smoke-hash-v1")
    if e.backend == "hash" and (e.query_prefix or e.passage_prefix):
        raise ConfigurationError("Smoke hash embeddings require empty query/passage prefixes")
    if not e.model.strip() or e.similarity not in {"cosine", "inner_product"}:
        raise ConfigurationError("Specify an embedding model and cosine or inner_product similarity")
    if e.similarity == "cosine" and not e.normalize:
        raise ConfigurationError("Cosine similarity requires normalized embeddings")
    if llm.provider != "openai_compatible":
        raise ConfigurationError("Supported LLM provider: openai_compatible (configurable base_url)")
    if not llm.base_url.startswith(("https://", "http://")) or not 0 <= llm.temperature <= 2:
        raise ConfigurationError("Invalid LLM base_url or temperature")
    for name in ("evidence_threshold", "answer_threshold", "semantic_threshold", "faithfulness_threshold"):
        if not 0 < getattr(cfg.evaluation, name) <= 1:
            raise ConfigurationError(f"evaluation.{name} must be in (0, 1]")
    if not cfg.data_dir.strip():
        raise ConfigurationError("data_dir cannot be empty")
    r, a = cfg.retrieval, cfg.egar
    if r.mode not in {"dense", "bm25", "hybrid"}:
        raise ConfigurationError("retrieval.mode must be dense, bm25, or hybrid")
    if r.candidate_k < cfg.top_k or r.rrf_k <= 0 or r.bm25_k1 <= 0 or not 0 <= r.bm25_b <= 1:
        raise ConfigurationError("Invalid retrieval parameters (candidate_k must be >= top_k)")
    if cfg.extraction.schema_version not in {1, 2} or cfg.extraction.max_evidence_tokens <= 0:
        raise ConfigurationError("Use extraction schema_version 1 or 2 and a positive context budget")
    if a.judge not in {"llm", "deterministic"} or a.minimum_sufficiency != "sufficient":
        raise ConfigurationError("Use llm/deterministic judge; only sufficient evidence permits final claims")
    if not 0 <= a.max_iterations <= 10 or not 0 <= a.max_retries <= 2:
        raise ConfigurationError("EGAR iterations must be 0..10 and retries 0..2")
    if any(v <= 0 for v in (a.max_queries_per_iteration, a.supplementary_top_k,
                            a.max_candidates, a.max_llm_calls, a.timeout_seconds,
                            a.call_timeout_seconds)):
        raise ConfigurationError("EGAR budgets must be positive")
    if a.input_cost_per_million < 0 or a.output_cost_per_million < 0:
        raise ConfigurationError("Token prices must be nonnegative")
    if a.enabled and cfg.top_k > a.max_candidates:
        raise ConfigurationError('Initial top_k cannot exceed the EGAR candidate budget')
    return cfg


def load_config(paths: list[str | Path] | None = None, *,
                environ: Mapping[str, str] | None = None,
                overrides: dict | None = None) -> AppConfig:
    """Defaults < explicitly listed YAMLs < environment < CLI/programmatic overrides.

    YAML project_root is relative to that YAML's directory; other paths are relative
    to the resulting project_root. No config.py, .env, or implicit local file reads.
    """
    env = os.environ if environ is None else environ
    values = asdict(AppConfig())
    selected = paths or ([env["CODAL_CONFIG"]] if env.get("CODAL_CONFIG") else [])
    for item in selected:
        path = Path(item).expanduser().resolve()
        try:
            update = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError) as exc:
            raise ConfigurationError(f"Cannot read YAML configuration: {path}") from exc
        if not isinstance(update, dict):
            raise ConfigurationError(f"Configuration must be a mapping: {path}")
        if isinstance(update.get("llm"), dict) and "api_key" in update["llm"]:
            raise ConfigurationError("Use CODAL_LLM_API_KEY or OPENAI_API_KEY, not a YAML credential")
        if "project_root" in update:
            if not isinstance(update["project_root"], str):
                raise ConfigurationError("project_root must be a path string")
            update["project_root"] = (path.parent / Path(update["project_root"]).expanduser()).resolve()
        _merge(values, update)
    # Every scalar config field can be overridden by CODAL_<SECTION>_<FIELD>.
    for section, section_values in list(values.items()):
        entries = section_values.items() if isinstance(section_values, dict) else [(section, section_values)]
        for name, default in entries:
            key = f"CODAL_{section}_{name}" if isinstance(section_values, dict) else f"CODAL_{name}"
            key = key.upper()
            if key not in env:
                continue
            raw = env[key]
            try:
                if isinstance(default, bool):
                    if raw.lower() not in {"true", "false", "1", "0"}:
                        raise ValueError()
                    parsed = raw.lower() in {"true", "1"}
                elif type(default) is int:
                    parsed = int(raw)
                elif type(default) is float:
                    parsed = float(raw)
                else:
                    parsed = raw
            except ValueError as exc:
                raise ConfigurationError(f"Invalid value for {key}") from exc
            if isinstance(section_values, dict):
                section_values[name] = parsed
            else:
                values[section] = parsed
    if "CODAL_LLM_API_KEY" not in env:
        values["llm"]["api_key"] = env.get("OPENAI_API_KEY", "")
    if "CODAL_LLM_BASE_URL" not in env and env.get("OPENAI_BASE_URL"):
        values["llm"]["base_url"] = env["OPENAI_BASE_URL"]
    if overrides:
        _merge(values, overrides)
    values["project_root"] = Path(values["project_root"]).expanduser().resolve()
    for section, cls in SECTIONS.items():
        values[section] = cls(**values[section])
    return validate_config(AppConfig(**values))
