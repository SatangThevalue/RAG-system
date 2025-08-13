import os, time
from typing import List, Dict
from fastapi import FastAPI
from pydantic import BaseModel
from memory import get_store, history_as_text
from pipelines import chain
from pipelines_chat import conversational_rag_with_citations
from db import init_db, save_message

app = FastAPI(title="RAG Chatbot + Redis + LangFlow + Chroma", version="1.3.0")
store = get_store()
init_db()

class IngestOut(BaseModel):
    returncode: int
    stdout: List[str]
    stderr: List[str]

class QueryIn(BaseModel):
    question: str

class ChatIn(BaseModel):
    session_id: str
    message: str

class ChatOut(BaseModel):
    session_id: str
    answer: str
    sources: List[Dict[str, str]]

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/ingest", response_model=IngestOut)
def ingest():
    import subprocess
    cp = subprocess.run(["python", "ingest.py"], capture_output=True, text=True)
    return {"returncode": cp.returncode, "stdout": cp.stdout.splitlines()[-50:], "stderr": cp.stderr.splitlines()[-50:]}

@app.post("/query")
def query(inp: QueryIn):
    ans = chain.invoke(inp.question)
    return {"answer": ans}

@app.post("/chat", response_model=ChatOut)
def chat(inp: ChatIn):
    sid = inp.session_id
    ts = int(time.time())
    store.append(sid, "user", inp.message)
    save_message(sid, "user", inp.message, ts)
    hist_text = history_as_text(store.history(sid, limit=12))
    result = conversational_rag_with_citations(question=inp.message, history_text=hist_text)
    store.append(sid, "assistant", result["answer"])
    save_message(sid, "assistant", result["answer"], int(time.time()))
    return {"session_id": sid, **result}

@app.post("/chat/clear")
def clear(session_id: str):
    store.clear(session_id)
    return {"ok": True, "session_id": session_id}