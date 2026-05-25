import json
from pathlib import Path
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import load_rag_config, get_config_value
from src.pdf_parser import parse_pdf


def main():
    config = load_rag_config()

    raw_dir = Path(get_config_value(config, "document.raw_dir", "data/raw_docs"))
    out_path = Path(get_config_value(config, "document.parsed_path", "data/parsed/documents.jsonl"))

    out_path.parent.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(raw_dir.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {raw_dir}")

    total_pages = 0
    scanned_pages = 0

    with out_path.open("w", encoding="utf-8") as f:
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
    print(f"Output saved to: {out_path}")


if __name__ == "__main__":
    main()
