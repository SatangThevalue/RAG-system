import os
import time, json
from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.document_loaders.markdown import MarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from mlflow_utils import start_run, log_params, log_metrics, log_artifact_text, end_run

DOC_DIR = Path("/app/data/docs")
EMB_MODEL = os.getenv("HF_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
COLLECTION = os.getenv("CHROMA_COLLECTION", "rag_docs")
CHROMA_HOST = os.getenv("CHROMA_HOST", "chroma")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))

def load_docs():
    docs = []
    for path in sorted(DOC_DIR.rglob("*")):
        if path.suffix.lower() in [".pdf"]:
            docs += PyPDFLoader(str(path)).load()
        elif path.suffix.lower() in [".md", ".markdown"]:
            docs += MarkdownLoader(str(path)).load()
        elif path.suffix.lower() in [".txt"]:
            docs += TextLoader(str(path), encoding="utf-8").load()
    return docs

def main():
    run = start_run(run_name="ingest")
    t0 = time.time()

    print("Loading docs...")
    raw_docs = load_docs()
    n_docs = len(raw_docs)
    print(f"Loaded {n_docs} docs")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(raw_docs)
    n_chunks = len(chunks)

    # log params/metrics
    log_params({
        "emb_model": EMB_MODEL,
        "collection": COLLECTION,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "n_docs": n_docs,
    })
    log_metrics({"n_chunks": n_chunks})

    # manifest artifact
    manifest = {
        "docs": [str(d.metadata.get("source", d.metadata.get("file_path", "unknown"))) for d in raw_docs],
        "n_docs": n_docs,
        "n_chunks": n_chunks,
    }
    log_artifact_text("ingest_manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))

    embeddings = HuggingFaceEmbeddings(model_name=EMB_MODEL)

    print("Writing to Chroma (HTTP client)...")
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION,
        client_settings={
            "chroma_server_host": CHROMA_HOST,
            "chroma_server_http_port": CHROMA_PORT,
        },
    )

    dur = time.time() - t0
    log_metrics({"ingest_seconds": dur})
    end_run()
    print(f"Done. took {dur:.2f}s")

if __name__ == "__main__":
    main()