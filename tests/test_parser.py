from src.pdf_parser import clean_text, detect_language


def test_clean_text_removes_extra_spaces():
    text = "bearing   fault\t diagnosis\n\n\nouter race"
    cleaned = clean_text(text)

    assert "   " not in cleaned
    assert "\t" not in cleaned
    assert "bearing fault diagnosis" in cleaned


def test_detect_language_en():
    text = "Bearing outer race fault diagnosis using vibration spectrum."
    assert detect_language(text) == "en"


def test_detect_language_zh():
    text = "轴承外圈故障通常表现为周期性冲击。"
    assert detect_language(text) == "zh"


def test_detect_language_mixed():
    text = "轴承 outer race fault 与 BPFO 有关。"
    assert detect_language(text) == "mixed"
