import json
from pathlib import Path
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.pdf_parser import parse_pdf


RAW_DIR = Path("data/raw_docs")
OUT_PATH = Path("data/parsed/documents.jsonl")


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(RAW_DIR.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {RAW_DIR}")

    total_pages = 0
    scanned_pages = 0

    with OUT_PATH.open("w", encoding="utf-8") as f:
        for pdf_path in tqdm(pdf_files, desc="Parsing PDFs"):
            records = parse_pdf(pdf_path)

            for record in records:
                total_pages += 1
                if record["possible_scanned_page"]:
                    scanned_pages += 1

                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Parsed PDF files: {len(pdf_files)}")
    print(f"Total pages: {total_pages}")
    print(f"Possible scanned/empty pages: {scanned_pages}")
    print(f"Output saved to: {OUT_PATH}")


if __name__ == "__main__":
    main()
