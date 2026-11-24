"""Read the two CLI report contracts without rerunning evaluation or scoring missing values."""
from __future__ import annotations

from functools import lru_cache
import os
from pathlib import Path
import re

from codal_rag.pipelines.research import ARMS
from codal_rag.storage.json_io import digest
from .analysis import file_stamp
from .safety import UIError, inside, read_object
from .state import LEVELS

ID = re.compile(r"[A-Za-z0-9_-]{1,100}\Z")
SKIP = {"chunks", "embeddings", "vector_db", "documents", "retrieval_results"}
METHOD_NAMES = {"baseline", "proposed", *ARMS}


def discover(root):
    reports, runs = [], []
    if not root.is_dir():
        return reports, runs
    for directory, folders, filenames in os.walk(root):
        p = Path(directory)
        folders[:] = [f for f in folders if f not in SKIP and not f.startswith(".") and not (p/f).is_symlink()]
        if "run.json" in filenames:
            runs.append(inside(root, str(p.relative_to(root)), "run.json"))
            folders.clear()  # Never traverse hundreds of question output directories.
        for name in filenames:
            if name.endswith(".json") and ("report" in name.casefold() or p.name in {"evaluation_results", "research_results"}):
                if name != "run.json":
                    reports.append(inside(root, str(p.relative_to(root)), name))
    return sorted(reports), sorted(runs)


def report_rows(report):
    version = report.get("schema_version")
    if version not in (1, 2) or not isinstance(report.get("overall"), dict):
        raise UIError("این فایل گزارش evaluate یا research-evaluate نیست.")
    methods = list(report["overall"])
    if not methods or set(methods) - METHOD_NAMES:
        raise UIError("روش‌های گزارش شناخته نشدند.")
    rows = []
    if version == 2:
        if set(report["per_arm"]) != set(methods) or not isinstance(report["run"]["complete"], bool):
            raise UIError("قرارداد روش‌ها یا منشأ گزارش ناقص است.")
        for m, questions in report["per_arm"].items():
            for q in questions:
                if not isinstance(q["metrics"], dict):
                    raise UIError("ساختار معیارهای پرسش معتبر نیست.")
                rows.append({"id": q["id"], "type": q["type"], "method": m, "metrics": q["metrics"]})
    else:
        for q in report["per_question"]:
            for m in methods:
                data = q[m]
                measures = {k: v for k, v in data["retrieval"].items() if k != "retrieval_details"}
                measures.update({k: data["answer"].get(k) for k in ("answer_correctness", "semantic_correctness")})
                measures.update({k: v["score"] for k, v in data["faithfulness"].items() if isinstance(v, dict)})
                rows.append({"id": q["id"], "question": q["question"], "type": q["type"], "method": m,
                             "metrics": measures, "answer": data["answer"], "retrieval": data["retrieval"]})
    seen = set()
    for r in rows:
        pair = (r["id"], r["method"])
        if not ID.fullmatch(r["id"]) or r["type"] not in LEVELS or pair in seen:
            raise UIError("شناسه یا ردهٔ پرسش‌ها در گزارش معتبر نیست.")
        seen.add(pair)
    return rows


def aggregate(report, methods, category):
    output = {}
    for m in methods:
        if category and report["schema_version"] == 1:
            src = report.get("by_type", {}).get(category, {}).get(m, {})
        elif category:
            src = report.get("by_level", {}).get(m, {}).get(category, {})
        else:
            src = report["overall"].get(m, {})
        if report["schema_version"] == 2:
            output[m] = {k: v["mean"] for k, v in src.items() if isinstance(v, dict) and "mean" in v}
        else:
            output[m] = {k: v for section in ("retrieval", "answer", "faithfulness") for k, v in src.get(section, {}).items()}
    return output


def execution_kind(manifest, cfg=None):
    manifest = manifest or {}
    configs = list((manifest.get("configurations") or {}).values()) or [manifest.get("configuration") or cfg or {}]
    flags = []
    if manifest.get("complete") is False:
        flags.append("اجرای ناقص")
    if manifest.get("synthetic"):
        flags.append("دادهٔ مصنوعی")
    if manifest.get("mocked_llm") or manifest.get("separately_injected_judge"):
        flags.append("مدل یا داور جایگزین آزمون")
    if any(c.get("embedding", {}).get("backend") == "hash" for c in configs):
        flags.append("بازیابی آزمایشی (smoke)")
    if flags:
        return " · ".join(flags), False
    if manifest.get("complete") and manifest.get("mocked_llm") is False:
        return "اجرای کامل با مدل پیکربندی‌شده، طبق شناسنامه", True
    return "منشأ اجرای مدل در این گزارش قابل تأیید نیست", False


@lru_cache(maxsize=256)
def artifact_digest(path, version):
    return digest(read_object(path))


def match_run(report, path):
    """Only attach a local run to a report after verifying the recorded identity."""
    manifest = read_object(path)
    if report["schema_version"] == 2:
        return manifest if manifest == report["run"] else None
    if not manifest.get("complete") or manifest.get("configuration") != report.get("run_configuration"):
        return None
    hashes = report.get("artifact_sha256", {})
    if not hashes or set(manifest.get("question_ids", [])) != set(hashes):
        return None
    for qid, artifact_hashes in hashes.items():
        if not ID.fullmatch(qid) or set(artifact_hashes) != {"gold", "retrieval", "baseline", "proposed"}:
            return None
        for name, expected in artifact_hashes.items():
            asset = inside(path.parent, qid, name + ".json")
            if not asset.is_file() or artifact_digest(asset, file_stamp(asset)) != expected:
                return None
    return manifest


def details(report, run_path, qid, method):
    if not ID.fullmatch(qid) or method not in METHOD_NAMES:
        raise UIError("شناسهٔ جزئیات معتبر نیست.")
    hashes = report.get("artifact_sha256", {})
    if report["schema_version"] == 1:
        expected = hashes[qid]
        files = {k: (k, expected[k]) for k in ("gold", "retrieval", method)}
    else:
        expected = hashes[f"{qid}/{method}"]
        files = {method: (method, expected["output"]), "retrieval": (ARMS[method][1] + "_retrieval", expected["retrieval"])}
    values = {}
    for label, (name, fingerprint) in files.items():
        data = read_object(inside(run_path.parent, qid, name + ".json"))
        if digest(data) != fingerprint:
            raise UIError("اثر انگشت فایل پرسش با گزارش ارزیابی تطبیق ندارد.")
        values[label] = data
    return values
