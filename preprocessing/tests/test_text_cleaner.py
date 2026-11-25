import json
from pathlib import Path

import pytest

from text_cleaner import clean_document, clean_page_record, clean_text


def test_clean_text_preserves_line_and_paragraph_boundaries():
    source = (
        "  Question\t No. 481:  Generic medicines\r\n"
        "  Follow-up question.\r\n\r\n\r\n"
        "Minister:  The answer is here.  "
    )

    assert clean_text(source) == (
        "Question No. 481: Generic medicines\n"
        "Follow-up question.\n\n"
        "Minister: The answer is here."
    )


def test_clean_text_preserves_hindi_unicode_and_punctuation():
    source = "\u0928\u092e\u0938\u094d\u0924\u0947, \u092a\u094d\u0930\u0936\u094d\u0928 481!"

    assert clean_text(source) == source


def test_suspicious_hindi_markers_are_flagged_but_not_repaired():
    source = "\u0928\u092e\u0938\u094d\u0924\u0947 (cid:197) \ufffd à¤"
    page = {
        "pdf_page": 13,
        "text": source,
        "is_empty": False,
        "extraction_status": "ok",
    }

    cleaned_page = clean_page_record(page)

    assert cleaned_page["original_text"] == source
    assert "(cid:197)" in cleaned_page["cleaned_text"]
    assert "\ufffd" in cleaned_page["cleaned_text"]
    assert "à¤" in cleaned_page["cleaned_text"]
    assert cleaned_page["quality_warnings"] == [
        "cid_glyph_placeholder",
        "unicode_replacement_character",
        "possible_hindi_mojibake",
    ]


def test_clean_page_record_keeps_extraction_error_and_null_text():
    page = {
        "pdf_page": 8,
        "text": None,
        "is_empty": None,
        "extraction_status": "error",
        "extraction_error": {"type": "RuntimeError", "message": "failed"},
    }

    cleaned_page = clean_page_record(page)

    assert cleaned_page["original_text"] is None
    assert cleaned_page["cleaned_text"] is None
    assert cleaned_page["extraction_error"] == page["extraction_error"]
    assert cleaned_page["quality_warnings"] == []


def test_clean_document_sample_keeps_raw_text_and_source_report(tmp_path: Path):
    source_path = tmp_path / "source.json"
    output_path = tmp_path / "processed" / "sample.json"
    source_pages = [
        {
            "pdf_page": 1,
            "text": " Question\t 481\r\n\r\nAnswer ",
            "is_empty": False,
            "extraction_status": "ok",
        },
        {
            "pdf_page": 2,
            "text": "\u0928(cid:197)",
            "is_empty": False,
            "extraction_status": "ok",
        },
    ]
    source_document = {
        "dataset_name": "Test debate",
        "source_file": "debate.pdf",
        "date": "2026-03-27",
        "total_pdf_pages": 2,
        "extracted_page_count": 2,
        "selected_pdf_pages": [1, 2],
        "quality_report": {"pages_processed": 2, "failed_pages": []},
        "pages": source_pages,
    }
    source_path.write_text(
        json.dumps(source_document, ensure_ascii=False),
        encoding="utf-8",
    )

    result = clean_document(source_path, output_path, sample_pages=[2, 1])
    saved_document = json.loads(output_path.read_text(encoding="utf-8"))

    assert saved_document == result
    assert [page["pdf_page"] for page in result["pages"]] == [1, 2]
    assert result["pages"][0]["original_text"] == source_pages[0]["text"]
    assert result["pages"][0]["cleaned_text"] == "Question 481\n\nAnswer"
    assert result["pages"][1]["quality_warnings"] == ["cid_glyph_placeholder"]
    assert result["source_extraction_quality_report"] == source_document["quality_report"]
    assert source_document["pages"] == source_pages
    assert result["cleaning_report"]["pages_processed"] == 2
    assert result["cleaning_report"]["pages_with_warnings"] == 1


def test_clean_document_rejects_missing_sample_page(tmp_path: Path):
    source_path = tmp_path / "source.json"
    source_path.write_text(
        json.dumps({"pages": [{"pdf_page": 1, "text": "text"}]}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Requested sample pages are missing"):
        clean_document(source_path, tmp_path / "out.json", sample_pages=[2])
