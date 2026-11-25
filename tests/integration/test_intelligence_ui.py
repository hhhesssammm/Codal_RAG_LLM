"""Exercise the Streamlit entry point and its adapters with isolated, offline artifacts."""
from pathlib import Path
import json

import pytest

from codal_rag.config import load_config
from codal_rag.pipelines.batch import run_batch
from codal_rag.pipelines.chunk import chunk_documents
from codal_rag.pipelines.evaluate import evaluate_runs
from codal_rag.pipelines.index import index_corpus
from codal_rag.storage.json_io import load_json, save_json
from tests.fakes import FakeLLM

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[2] / "apps" / "streamlit_app.py"


@pytest.fixture
def indexed(smoke_config, corpus, monkeypatch):
    chunk_documents(corpus, smoke_config, metadata_by_id={"alpha/report": {"company": "الف", "year": "۱۴۰۳",
                    "document_type": "گزارش مدیریت"}, "beta/report": {"company": "ب", "year": "۱۴۰۲"}})
    index_corpus(smoke_config)
    p = save_json(smoke_config.public_dict(), smoke_config.project_root/"test-config.json")
    monkeypatch.setenv("CODAL_CONFIG", str(p))
    return smoke_config


def go(app, page):
    return app.radio(key="ci_navigation").set_value(page).run()


def click(app, label):
    return next(b for b in app.button if b.label == label).click().run()


def test_session_navigation_examples_scoped_evidence_and_exports(indexed, monkeypatch):
    from codal_rag.pipelines import extract
    fake = FakeLLM()
    monkeypatch.setattr(extract, "LLMClient", lambda config: fake)
    app = AppTest.from_file(str(APP)).run()
    assert app.title[0].value == "تحلیل هوشمند"
    example = "علت تغییرات فروش شرکت چیست؟"
    click(app, example)
    assert app.text_area[0].value == example
    click(app, "تحلیل پرسش")
    assert not app.error and not app.exception
    key = app.session_state["current_conversation"]
    item = app.session_state["conversations"][key][0]["methods"]["proposed"]
    assert item["output"]["citation_validation"]["valid"]
    assert any("مشاهدهٔ شاهد E1" in e.label for e in app.expander)
    assert len(app.get("download_button")) == 1
    assert len(fake.calls) == 1
    for page in ("comparison", "evaluation", "documents", "settings"):
        go(app, page)
        assert not app.exception and not app.error
    app.radio(key="appearance_mode").set_value("پژوهشگر").run()
    app.radio(key="appearance_theme").set_value("تیره").run()
    go(app, "assistant").run()
    assert app.tabs[-1].label == "فرایند بازیابی"
    assert app.session_state["conversations"][key][0]["methods"]["proposed"] == item
    assert len(fake.calls) == 1
    click(app, "گفت‌وگوی جدید")
    assert not app.session_state["conversations"][app.session_state["current_conversation"]]
    click(app, "باز کردن گفت‌وگو")
    assert app.session_state["current_conversation"] == key
    click(app, "پاک کردن این گفت‌وگو")
    assert key not in app.session_state["conversations"]
    assert len(fake.calls) == 1


def test_comparison_one_ranking_no_context_to_control_partial_failure(indexed, monkeypatch):
    from codal_ui import analysis
    from codal_rag.egar.demo import SyntheticClient
    from codal_rag.retrieval.engine import DualRetrievalEngine
    engine = DualRetrievalEngine(config=indexed)
    calls = []
    search = engine.search
    def counted(*args, **kwargs):
        calls.append(args)
        return search(*args, **kwargs)
    monkeypatch.setattr(engine, "search", counted)
    prompts = []
    class Plain:
        last_usage = {"prompt_tokens": 17, "completion_tokens": 4}
        def generate_json(self, system_prompt, user_prompt):
            prompts.append((system_prompt, user_prompt))
            return {"answer": "برای پاسخ اختصاصی به داده نیاز است.", "limitations": []}
    monkeypatch.setattr(analysis, "LLMClient", lambda cfg: Plain())
    monkeypatch.setattr(analysis.pipeline, "LLMClient", lambda cfg: SyntheticClient())
    original = analysis.pipeline.extract_retrieval
    def fail_one(payload, method, cfg, **kwargs):
        if method == "baseline":
            raise ValueError("sensitive provider detail")
        return original(payload, method, cfg, **kwargs)
    monkeypatch.setattr(analysis.pipeline, "extract_retrieval", fail_one)
    q = "وضعیت فروش چگونه است؟"
    run = analysis.execute(q, list(analysis.METHODS), indexed, comparison=True, engine_factory=lambda cfg: engine)
    r = run["methods"]
    assert len(calls) == 1 and [u for s, u in prompts] == [q]
    assert "report" not in prompts[0][0] and "E1" not in prompts[0][0]
    assert r["llm_only"]["usage"]["total_inference_tokens"] == 21
    assert r["baseline"]["status"] == "failed" and r["proposed"]["status"] == "complete"
    assert "sensitive provider detail" not in json.dumps(run)
    assert r["baseline"]["initial_sha256"] == r["proposed"]["initial_sha256"] == r["egar"]["initial_sha256"]
    assert r["egar"]["configuration"]["egar"]["enabled"] and not r["proposed"]["configuration"]["egar"]["enabled"]
    assert r["egar"]["configuration"]["extraction"] == r["baseline"]["configuration"]["extraction"]
    generation = r["llm_only"]["configuration"]["llm"]
    assert generation == r["proposed"]["configuration"]["llm"]
    assert generation["max_retries"] == 0 and generation["timeout"] == indexed.egar.call_timeout_seconds
    assert r["egar"]["output"]["egar"]["stop_reason"] == "ambiguous_scope"
    assert "supplement" not in r["egar"]["stages"]


def test_missing_indexes_do_not_disable_plain_control_or_generate_fake_result(smoke_config):
    from codal_ui.analysis import execute
    class Plain:
        def generate_json(self, **_):
            return {"answer": "پاسخ عمومی"}
    run = execute("پرسش", ["baseline", "llm_only"], smoke_config, comparison=True, client_factory=lambda c: Plain())
    assert run["methods"]["baseline"]["status"] == "unavailable"
    assert run["methods"]["llm_only"]["status"] == "complete"
    assert run["methods"]["llm_only"]["usage"]["prompt_tokens"] is None
    assert "output" not in run["methods"]["baseline"]


def test_invalid_config_settings_recovery(tmp_path, monkeypatch):
    secret = "hidden-value-1234"
    path = tmp_path/"bad.yaml"
    path.write_text("llm:\n  api_key: " + secret)
    monkeypatch.setenv("CODAL_CONFIG", str(path))
    app = AppTest.from_file(str(APP)).run()
    assert app.error and not app.exception
    assert secret not in app.error[0].value and str(tmp_path) not in app.error[0].value
    app.multiselect[0].set_value(["base.yaml"])
    click(app, "اعمال پروفایل‌ها و بازنشانی تنظیمات نشست")
    assert not app.error and app.title[0].value == "تحلیل هوشمند"


def test_catalog_scope_unicode_offsets_and_verified_original(indexed, corpus, monkeypatch, tmp_path):
    from codal_ui.catalog import catalog_for, original, records, scope_for
    from codal_ui.safety import UIError
    cat = catalog_for(indexed, "proposed")
    doc = "alpha/report"
    chunks = cat.chunks(doc, 0, 2)
    assert all(c["document_id"] == doc for c in chunks)
    scope = scope_for(cat.docs, {"document_type": "گزارش مدیریت", "year": "۱۴۰۳"})
    assert scope.document_ids == (doc,) and scope.matches(chunks[0])
    assert not scope.matches(cat.chunks("beta/report", 0, 1)[0])
    with pytest.raises(UIError):
        scope_for(cat.docs, {"company": "شرکت دیگر"})
    monkeypatch.setenv("CODAL_UI_DOCUMENT_ROOT", str(corpus))
    assert "جدول" in original(indexed, doc, cat.docs[doc], "proposed")
    (corpus/"alpha"/"report.md").write_text("تغییر")
    with pytest.raises(UIError):
        original(indexed, doc, cat.docs[doc], "proposed")
    with pytest.raises(UIError):
        original(indexed, doc, {"source_file": "../bad.md"}, "proposed")
    compact = tmp_path/"many.json"
    big = {"document_id": "الف", "text": "الف"*40000}
    compact.write_text(json.dumps([big, {"document_id": "ب"}], ensure_ascii=False))
    for row, start, n in records(compact):
        assert json.loads(compact.read_bytes()[start:start+n]) == row
    # A rebuilt index changes the engine stamp. Historical snapshots remain intact.
    from codal_ui.analysis import current_engine
    engine = current_engine(indexed)
    index_corpus(indexed)
    assert current_engine(indexed) is not engine


def test_saved_citations_untrusted_rendering_and_downloads(indexed, monkeypatch):
    from codal_ui.analysis import execute
    from codal_ui.presentation import answer_text, evidence_map, passage_html, saved_extraction
    from codal_ui.safety import UIError, codal_url, json_bytes, parse_object
    from codal_rag.pipelines import extract
    monkeypatch.setattr(extract, "LLMClient", lambda c: FakeLLM())
    record = execute("سؤال", ["proposed"], indexed)["methods"]["proposed"]
    saved = saved_extraction(record["output"])
    reopened = saved_extraction(record)
    assert reopened["configuration"]["embedding"]["backend"] == "hash"
    assert reopened["initial_sha256"] == record["initial_sha256"]
    assert evidence_map(reopened)["E1"]["chunk_id"] == evidence_map(record)["E1"]["chunk_id"]
    ref = evidence_map(saved)["E1"]
    assert ref["document_id"] in answer_text(record) and "[E1]" in answer_text(record)
    ev = record["output"]["citation_view"]["items"][0]["evidence"][0]
    ev["document_id"] = "different"
    with pytest.raises(UIError):
        saved_extraction(record["output"])
    html = passage_html("# فروش FY1403\n\n| سال | نرخ |\n| --- | --- |\n| ۱۴۰۳ | -3.2% |\n<script>bad()</script>")
    assert '<bdi>-3.2%</bdi>' in html and "<table>" in html and "<script>" not in html
    assert 'dir="auto"' in html and "&lt;script&gt;" in html
    assert codal_url("https://www.codal.ir/Reports/1") and not codal_url("https://codal.ir.evil.invalid/")
    monkeypatch.setenv("OPENAI_API_KEY", "my-private-key")
    blob = json_bytes({"llm": {"api_key": "unsafe", "base_url": "private"}, "detail": "my-private-key", "n": None})
    assert b"unsafe" not in blob and b"my-private-key" not in blob and parse_object(blob)["n"] is None
    with pytest.raises(ValueError):
        parse_object(b'{"bad":NaN}')


def test_saved_evaluation_views_provenance_integrity_and_nullable_metrics(indexed, tmp_path):
    from codal_ui.evaluations import aggregate, details, discover, execution_kind, match_run, report_rows
    from codal_ui.safety import UIError
    q = save_json([{"id": "Q1", "question": "فروش؟", "type": "explicit",
                    "expected_answer": "کاهش", "gold_evidence": []}], tmp_path/"questions.json")
    runs = run_batch(q, indexed, llm_client=FakeLLM())
    report = evaluate_runs(runs, indexed)
    reports, manifests = discover(indexed.path())
    assert len(reports) == len(manifests) == 1
    assert len(report_rows(report)) == 2
    metrics = aggregate(report, ["baseline"], "explicit")
    assert metrics["baseline"]["recall@3"] is None
    matched = match_run(report, manifests[0])
    assert matched and not execution_kind(matched)[1]
    assert details(report, manifests[0], "Q1", "proposed")["gold"]["expected_answer"] == "کاهش"
    app = go(AppTest.from_file(str(APP)).run(), "evaluation")
    assert not app.exception and not app.error
    assert any("smoke" in w.value for w in app.warning)
    assert any(a.label == "دریافت گزارش ارزیابی" for a in app.get("download_button"))
    bad_file = runs/"Q1"/"proposed.json"
    save_json({"query": "wrong"}, bad_file)
    assert match_run(report, manifests[0]) is None
    with pytest.raises(UIError):
        details(report, manifests[0], "Q1", "proposed")
    with pytest.raises(UIError):
        report_rows({"overall": {}})


def test_research_evaluation_contract_and_mocked_provenance(tmp_path, smoke_config):
    from tests.integration.test_egar_pipeline import prepare
    from codal_rag.pipelines.research import run_experiments, evaluate_experiments
    from codal_rag.egar.demo import SyntheticClient
    from codal_ui.evaluations import aggregate, details, execution_kind, match_run, report_rows
    cfg, dataset = prepare(tmp_path, smoke_config)
    path = run_experiments(dataset, cfg, arms=["D", "D+EGAR"], llm_client=SyntheticClient())
    report = evaluate_experiments(path, dataset, resamples=10)
    assert {r["method"] for r in report_rows(report)} == {"D", "D+EGAR"}
    assert aggregate(report, ["D"], "latent")["D"]["claim_f1"] is None
    m = match_run(report, path/"run.json")
    assert m == load_json(path/"run.json") and not execution_kind(m)[1]
    assert details(report, path/"run.json", "Q1", "D+EGAR")["D+EGAR"]["egar"]["iteration_count"] == 1


def test_comparison_ui_preserves_partial_results_and_downloads(indexed, monkeypatch):
    from codal_ui import analysis
    from codal_rag.egar.demo import SyntheticClient
    monkeypatch.setattr(analysis.pipeline, "LLMClient", lambda cfg: SyntheticClient())
    app = go(AppTest.from_file(str(APP)).run(), "comparison")
    app.text_area[0].set_value("فروش چگونه است؟")
    click(app, "اجرای مقایسه")
    before = app.session_state["comparison"]
    assert before["methods"]["llm_only"]["status"] == "unavailable"
    assert before["methods"]["proposed"]["status"] == "complete"
    next(r for r in app.radio if r.label == "چیدمان پاسخ‌ها").set_value("زبانه‌ای").run()
    assert app.session_state["comparison"] == before and not app.exception
    names = [b.label for b in app.get("download_button")]
    assert {"دریافت گزارش متنی", "دریافت مقایسهٔ JSON"} <= set(names)


def test_ui_profile_precedence_and_unchanged_cli_defaults(monkeypatch):
    from codal_ui.configuration import config_for
    monkeypatch.setenv("CODAL_TOP_K", "4")
    assert config_for(["base.yaml"], {}).top_k == 4
    assert config_for(["base.yaml"], {"top_k": 8}).top_k == 8
    cfg = load_config(environ={})
    assert not cfg.egar.enabled and cfg.extraction.schema_version == 1


def test_rerun_reattaches_to_inflight_submission(monkeypatch, smoke_config):
    from threading import Event
    from types import SimpleNamespace

    from codal_ui import jobs
    state = SimpleNamespace(jobs={}, comparison=None)
    release = Event()
    calls = []

    def slow_run(question, methods, cfg, *, notify, **kwargs):
        calls.append(question)
        notify("proposed", "retrieve")
        assert release.wait(5)
        return {"id": "one-submission", "question": question}

    class InterruptedView:
        def info(self, label):
            raise InterruptedError("Streamlit reran while the analysis was running")

        def empty(self):
            pass

    monkeypatch.setattr(jobs, "execute", slow_run)
    monkeypatch.setattr(jobs.st, "empty", InterruptedView)
    try:
        with pytest.raises(InterruptedError):
            jobs.submitted_run(state, "comparison", True, "فروش؟", ["proposed"], smoke_config, None)
        assert not state.jobs["comparison"][0].done()
        monkeypatch.setattr(jobs.st, "empty", lambda: (release.set() or InterruptedView()))
        jobs.submitted_run(state, "comparison", True, "فروش؟", ["proposed"], smoke_config, None)
        assert calls == ["فروش؟"] and state.comparison["id"] == "one-submission" and not state.jobs
    finally:
        release.set()
