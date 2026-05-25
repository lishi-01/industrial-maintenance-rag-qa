import json
from pathlib import Path
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import load_rag_config, get_config_value
from src.chunker import build_chunks_from_page


def main():
    config = load_rag_config()

    in_path = Path(get_config_value(config, "document.parsed_path", "data/parsed/documents.jsonl"))
    out_path = Path(get_config_value(config, "document.chunks_path", "data/chunks/chunks.jsonl"))
    chunk_size = int(get_config_value(config, "document.chunk_size", 600))
    chunk_overlap = int(get_config_value(config, "document.chunk_overlap", 100))

    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    total_pages = 0
    total_chunks = 0
    skipped_pages = 0

    with in_path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for line in tqdm(fin, desc="Building chunks"):
            total_pages += 1
            record = json.loads(line)

            if not record.get("text", "").strip():
                skipped_pages += 1
                continue

            chunks = build_chunks_from_page(
                record,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )

            for chunk in chunks:
                total_chunks += 1
                fout.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print(f"Input pages: {total_pages}")
    print(f"Skipped empty pages: {skipped_pages}")
    print(f"Output chunks: {total_chunks}")
    print(f"Chunk size: {chunk_size}")
    print(f"Chunk overlap: {chunk_overlap}")
    print(f"Output saved to: {out_path}")


if __name__ == "__main__":
    main()
