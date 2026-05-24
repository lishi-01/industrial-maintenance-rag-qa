import re
import fitz
from pathlib import Path
from datetime import datetime


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def detect_language(text: str) -> str:
    zh_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
    en_chars = len(re.findall(r"[A-Za-z]", text))

    if zh_chars > 0 and en_chars > 0:
        return "mixed"
    if zh_chars > 0:
        return "zh"
    return "en"


def parse_pdf(pdf_path: Path, domain: str = "bearing_fault_diagnosis"):
    doc = fitz.open(pdf_path)
    doc_id = pdf_path.stem

    records = []

    for page_idx in range(len(doc)):
        page = doc[page_idx]
        raw_text = page.get_text("text")
        text = clean_text(raw_text)

        record = {
            "doc_id": doc_id,
            "source_file": pdf_path.name,
            "file_type": "pdf",
            "page": page_idx + 1,
            "section": "",
            "language": detect_language(text),
            "text": text,
            "char_len": len(text),
            "possible_scanned_page": len(text) < 50,
            "metadata": {
                "domain": domain,
                "equipment": "bearing",
                "created_at": datetime.now().strftime("%Y-%m-%d")
            }
        }

        records.append(record)

    doc.close()
    return records
