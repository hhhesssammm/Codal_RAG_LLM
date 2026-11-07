from codal_rag.indexing.faiss_store import FaissVectorStore
from codal_rag.pipelines.embed import embed_corpus, load_embeddings, methods


def index_corpus(config, *, embedding_model=None, from_embeddings: bool = False) -> dict:
    artifacts = ({m: load_embeddings(config, m) for m in methods(config)} if from_embeddings
                 else embed_corpus(config, embedding_model=embedding_model))
    if len(artifacts) == 2 and any(artifacts['baseline']['manifest'][key] != artifacts['proposed']['manifest'][key] for key in ('documents', 'source_corpus_sha256')):
        raise ValueError("Baseline and Proposed must index the same document corpus")
    counts = {}
    for method, artifact in artifacts.items():
        store = FaissVectorStore(artifact['manifest']['embedding_dimension'])
        store.add(artifact['vectors'], artifact['records'])
        store.save(config.path("vector_db", method), artifact['manifest'])
        counts[method] = store.size
    return counts
