import os
from typing import List, Any
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from chromadb import HttpClient
from chromadb.config import Settings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

CHROMA_HOST = os.getenv("CHROMA_HOST", "chroma")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))
COLLECTION = os.getenv("CHROMA_COLLECTION", "rag_docs")
EMB_MODEL = os.getenv("HF_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
GEN_MODEL = os.getenv("HF_GENERATOR_MODEL", "google/flan-t5-base")
TOP_K = int(os.getenv("TOP_K", "4"))

embeddings = HuggingFaceEmbeddings(model_name=EMB_MODEL)

chroma_client = HttpClient(
    host=CHROMA_HOST,
    port=CHROMA_PORT,
    settings=Settings(allow_reset=True)  # ใส่ได้/ไม่ใส่ก็ได้
)

vectordb = Chroma(
    client=chroma_client,
    collection_name=COLLECTION,
    embedding_function=embeddings,
)

_tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL)
_model = AutoModelForSeq2SeqLM.from_pretrained(GEN_MODEL)
gen = pipeline("text2text-generation", model=_model, tokenizer=_tokenizer)

retriever = vectordb.as_retriever(search_kwargs={"k": TOP_K})

prompt = ChatPromptTemplate.from_template(
    """You are a helpful assistant that answers in Thai.
Use the retrieved context to answer the question. If unsure, say you don't know.
Cite sources as [source: filename or metadata].

Question: {question}

Context:
{context}

Answer in Thai:"""
)

def format_docs(docs: List[Any]) -> str:
    out = []
    for d in docs:
        src = d.metadata.get("source", "unknown")
        out.append(f"[{src}]\n{d.page_content}")
    return "\n\n".join(out)

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | (lambda x: gen(x, max_new_tokens=384)[0]["generated_text"])
    | StrOutputParser()
)