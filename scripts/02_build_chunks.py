import json
from pathlib import Path
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.chunker import build_chunks_from_page


IN_PATH = Path("data/parsed/documents.jsonl")
OUT_PATH = Path("data/chunks/chunks.jsonl")

CHUNK_SIZE = 600
CHUNK_OVERLAP = 100


def main():
    if not IN_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {IN_PATH}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    total_pages = 0
    total_chunks = 0
    skipped_pages = 0

    with IN_PATH.open("r", encoding="utf-8") as fin, OUT_PATH.open("w", encoding="utf-8") as fout:
        for line in tqdm(fin, desc="Building chunks"):
            total_pages += 1
            record = json.loads(line)

            if not record.get("text", "").strip():
                skipped_pages += 1
                continue

            chunks = build_chunks_from_page(
                record,
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP
            )

            for chunk in chunks:
                total_chunks += 1
                fout.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print(f"Input pages: {total_pages}")
    print(f"Skipped empty pages: {skipped_pages}")
    print(f"Output chunks: {total_chunks}")
    print(f"Output saved to: {OUT_PATH}")


if __name__ == "__main__":
    main()
