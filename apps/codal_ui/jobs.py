"""An explicit submission owns one worker. Streamlit reruns can reattach to that worker."""
from concurrent.futures import Future
from queue import Empty, Queue
from threading import Thread

import streamlit as st

from .analysis import STAGES, execute
from .state import METHODS


def collect(state):
    """Commit completed snapshots on the UI thread, even after navigating to a different page."""
    for key, (future, _) in list(state.jobs.items()):
        if not future.done():
            continue
        if state.jobs.pop(key, None) is None:
            continue
        run = future.result()
        if key == "comparison":
            state.comparison = run
        elif key in state.conversations:
            state.conversations[key].append(run)
            state["assistant_" + key + "_selected"] = run["id"]


def submitted_run(state, key, submitted, question, methods, cfg, scope):
    if submitted and key not in state.jobs:
        # Validate before launching so input errors belong to the submitting page.
        from .safety import UIError
        if not question.strip() or len(question.strip()) > 8000 or not methods:
            raise UIError("یک پرسش و حداقل یک روش معتبر وارد کنید.")
        future, stages = Future(), Queue()
        state.jobs[key] = (future, stages)

        def work():
            try:
                future.set_result(execute(question, methods, cfg, scope=scope, comparison=key == "comparison",
                                          notify=lambda method, stage: stages.put((method, stage))))
            except Exception as exc:
                future.set_exception(exc)
        Thread(target=work, daemon=True, name="codal-analysis").start()
    if key not in state.jobs:
        return
    future, stages = state.jobs[key]
    status = st.empty()
    try:
        while not future.done():
            try:
                method, stage = stages.get(timeout=0.12)
            except Empty:
                continue
            status.info(METHODS[method] + " · " + STAGES[stage])
        collect(state)
    finally:
        status.empty()
