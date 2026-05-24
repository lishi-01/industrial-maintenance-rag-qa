from pathlib import Path
import sys
from typing import List, Optional

sys.path.append(str(Path(__file__).resolve().parents[1]))

from fastapi import FastAPI
from pydantic import BaseModel

from src.rag_chain import IndustrialRAGChain


app = FastAPI(
    title="Industrial Maintenance RAG QA",
    description="Industrial equipment maintenance RAG question-answering system",
    version="0.1.0"
)

rag_chain = None


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5


class SourceItem(BaseModel):
    chunk_id: str
    source_file: str
    page_start: int
    page_end: int
    rrf_score: Optional[float] = None
    rerank_score: Optional[float] = None
    text: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: List[SourceItem]


def get_rag_chain():
    global rag_chain

    if rag_chain is None:
        rag_chain = IndustrialRAGChain(
            llm_model_name="Qwen/Qwen2.5-1.5B-Instruct",
            device="cpu"
        )

    return rag_chain


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "industrial-maintenance-rag-qa"
    }


@app.post("/query", response_model=QueryResponse)
def query_rag(request: QueryRequest):
    rag = get_rag_chain()

    result = rag.query(
        question=request.question,
        final_top_k=request.top_k
    )

    return {
        "question": result["question"],
        "answer": result["answer"],
        "sources": result["sources"]
    }
