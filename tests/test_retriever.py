import json
from pathlib import Path

from src.bm25_retriever import tokenize, BM25Retriever


def test_tokenize_english_and_number():
    text = "Bearing outer race fault BPFO 123.5 Hz"
    tokens = tokenize(text)

    assert "bearing" in tokens
    assert "outer" in tokens
    assert "race" in tokens
    assert "bpfo" in tokens
    assert "123.5" in tokens


def test_tokenize_chinese():
    text = "轴承外圈故障"
    tokens = tokenize(text)

    assert "轴" in tokens
    assert "承" in tokens
    assert "外" in tokens


def test_bm25_retriever_search(tmp_path):
    chunks_path = tmp_path / "chunks.jsonl"

    records = [
        {
            "chunk_id": "c1",
            "source_file": "bearing.pdf",
            "page_start": 1,
            "text": "Bearing outer race fault is related to BPFO vibration spectrum."
        },
        {
            "chunk_id": "c2",
            "source_file": "lubrication.pdf",
            "page_start": 2,
            "text": "Lubrication contamination may cause bearing damage."
        },
        {
            "chunk_id": "c3",
            "source_file": "gear.pdf",
            "page_start": 3,
            "text": "Gear tooth wear can cause vibration."
        }
    ]

    with chunks_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    retriever = BM25Retriever(str(chunks_path))
    results = retriever.search("outer race BPFO", top_k=2)

    assert len(results) == 2
    assert results[0]["chunk_id"] == "c1"
    assert results[0]["score"] > 0
