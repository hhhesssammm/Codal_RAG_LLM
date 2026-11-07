from codal_rag.retrieval.engine import DualRetrievalEngine


def retrieve(query, config, *, top_k=None, output=None, engine=None):
    engine = engine or DualRetrievalEngine(config=config)
    result = engine.search(query, top_k)
    if output is not None:
        engine.save_result(result, output)
    return engine.result_payload(result)
