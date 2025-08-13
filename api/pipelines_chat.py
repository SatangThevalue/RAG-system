import os
import time, json
from typing import List, Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from mlflow_utils import start_run, log_params, log_metrics, log_artifact_text, end_run

CHROMA_HOST = os.getenv("CHROMA_HOST", "chroma")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))
COLLECTION = os.getenv("CHROMA_COLLECTION", "rag_docs")
EMB_MODEL = os.getenv("HF_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
GEN_MODEL = os.getenv("HF_GENERATOR_MODEL", "google/flan-t5-base")
TOP_K = int(os.getenv("TOP_K", "4"))

embeddings = HuggingFaceEmbeddings(model_name=EMB_MODEL)
vectordb = Chroma(
    collection_name=COLLECTION,
    embedding_function=embeddings,
    client_settings={
        "chroma_server_host": CHROMA_HOST,
        "chroma_server_http_port": CHROMA_PORT,
    },
)

_tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL)
_model = AutoModelForSeq2SeqLM.from_pretrained(GEN_MODEL)
gen = pipeline("text2text-generation", model=_model, tokenizer=_tokenizer)

retriever = vectordb.as_retriever(
    search_kwargs={"k": TOP_K, "search_type": "mmr", "fetch_k": 20}
)

CONDENSE_PROMPT = ChatPromptTemplate.from_template(
    """Rewrite the user's latest question into a standalone Thai query using the chat history if needed.
Chat history:
{history}

Latest user question: {question}

Standalone question (Thai):"""
)

ANSWER_PROMPT = ChatPromptTemplate.from_template(
    """คุณคือผู้ช่วยภาษาไทยที่อธิบายกระชับ ถูกต้อง และอ้างอิงแหล่งที่มาด้วย
- ใช้เฉพาะข้อมูลในบริบทด้านล่างเท่านั้น
- หากข้อมูลไม่พอ ให้ตอบว่า "ไม่ทราบจากข้อมูลที่มี"
- สรุปเป็นภาษาไทย และแนบ [source: filename] ตอนท้ายอย่างย่อ

คำถาม: {question}
ประวัติย่อ: {history}

บริบทอ้างอิง:
{context}

คำตอบ:"""
)


def _format_docs(docs) -> str:
    blocks = []
    for d in docs:
        src = d.metadata.get("source", "unknown")
        snippet = d.page_content.strip().replace("\n", " ")
        if len(snippet) > 350:
            snippet = snippet[:350] + "..."
        blocks.append(f"[{src}]\n{snippet}")
    return "\n\n".join(blocks)


def _collect_sources(docs) -> List[Dict[str, str]]:
    seen = set()
    items = []
    for d in docs:
        src = d.metadata.get("source", "unknown")
        if src in seen:
            continue
        seen.add(src)
        preview = d.page_content.strip().split("\n", 1)[0]
        items.append(
            {
                "source": src,
                "preview": (preview[:140] + "...") if len(preview) > 140 else preview,
            }
        )
    return items


def _generate(text: str, max_new_tokens: int = 384) -> str:
    return gen(text, max_new_tokens=max_new_tokens)[0]["generated_text"]


def conversational_rag_with_citations(
    question: str, history_text: str = ""
) -> Dict[str, Any]:
    run = start_run(run_name="chat", nested=False)
    t0 = time.time()

    # log params of this request
    log_params(
        {
            "emb_model": EMB_MODEL,
            "gen_model": GEN_MODEL,
            "top_k": TOP_K,
            "history_len_chars": len(history_text or ""),
        }
    )

    # 1) condense
    t1 = time.time()
    standalone = _generate(
        CONDENSE_PROMPT.format(history=history_text, question=question), 128
    )
    log_metrics({"t_condense_ms": (time.time() - t1) * 1000})

    # 2) retrieve
    t2 = time.time()
    docs = retriever.invoke(standalone)
    log_metrics({"t_retrieve_ms": (time.time() - t2) * 1000, "k_returned": len(docs)})

    # เก็บ sources เป็น artifact
    source_rows = []
    for d in docs:
        source_rows.append(
            {
                "source": d.metadata.get("source", "unknown"),
                "len": len(d.page_content or ""),
            }
        )
    log_artifact_text(
        "retrieved_sources.json", json.dumps(source_rows, ensure_ascii=False, indent=2)
    )

    # 3) answer
    context = _format_docs(docs)
    t3 = time.time()
    answer = _generate(
        ANSWER_PROMPT.format(question=question, history=history_text, context=context),
        384,
    )
    log_metrics({"t_generate_ms": (time.time() - t3) * 1000})

    # สรุป latency รวม
    log_metrics({"t_total_ms": (time.time() - t0) * 1000})

    end_run()
    sources = _collect_sources(docs)
    return {"answer": answer, "sources": sources}
