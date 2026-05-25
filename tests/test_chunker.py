from src.chunker import split_text_sliding_window, build_chunks_from_page


def test_split_text_sliding_window_short_text():
    text = "bearing fault diagnosis"
    chunks = split_text_sliding_window(text, chunk_size=100, chunk_overlap=20)

    assert len(chunks) == 1
    assert chunks[0] == text


def test_split_text_sliding_window_long_text():
    text = "a" * 1000
    chunks = split_text_sliding_window(text, chunk_size=300, chunk_overlap=50)

    assert len(chunks) > 1
    assert all(len(c) <= 300 for c in chunks)


def test_build_chunks_from_page():
    page_record = {
        "doc_id": "test_doc",
        "source_file": "test.pdf",
        "page": 1,
        "section": "bearing fault",
        "language": "en",
        "text": "bearing outer race fault BPFO vibration spectrum",
        "metadata": {
            "domain": "bearing_fault_diagnosis",
            "equipment": "bearing"
        }
    }

    chunks = build_chunks_from_page(page_record, chunk_size=100, chunk_overlap=20)

    assert len(chunks) == 1
    assert chunks[0]["chunk_id"] == "test_doc_p0001_c001"
    assert chunks[0]["source_file"] == "test.pdf"
    assert chunks[0]["page_start"] == 1
    assert "BPFO" in chunks[0]["text"]
