import re
from typing import List, Dict


def clean_page_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n+", "\n", text)
    text = re.sub(r"\s+\n", "\n", text)
    return text.strip()


def split_text_sliding_window(
    text: str,
    chunk_size: int = 600,
    chunk_overlap: int = 100
) -> List[str]:
    text = clean_page_text(text)

    if not text:
        return []

    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = end - chunk_overlap

    return chunks


def build_chunks_from_page(
    page_record: Dict,
    chunk_size: int = 600,
    chunk_overlap: int = 100
) -> List[Dict]:
    text = page_record.get("text", "")
    pieces = split_text_sliding_window(text, chunk_size, chunk_overlap)

    chunks = []

    for idx, piece in enumerate(pieces):
        chunk_id = f"{page_record['doc_id']}_p{page_record['page']:04d}_c{idx + 1:03d}"

        chunk = {
            "chunk_id": chunk_id,
            "doc_id": page_record["doc_id"],
            "source_file": page_record["source_file"],
            "page_start": page_record["page"],
            "page_end": page_record["page"],
            "section": page_record.get("section", ""),
            "text": piece,
            "char_len": len(piece),
            "language": page_record.get("language", "unknown"),
            "keywords": [],
            "metadata": {
                "domain": page_record.get("metadata", {}).get("domain", "bearing_fault_diagnosis"),
                "equipment": page_record.get("metadata", {}).get("equipment", "bearing"),
                "fault_type": ""
            }
        }

        chunks.append(chunk)

    return chunks
