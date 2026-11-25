import json
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from preprocess import preprocess_pdf


def test_preprocess_pdf_writes_only_selected_pages_and_flags_empty_text(
    tmp_path: Path,
):
    source_pdf = tmp_path / "debate.pdf"
    source_pdf.touch()
    output_json = tmp_path / "intermediate" / "sample.json"

    source_pages = [MagicMock() for _ in range(4)]
    source_pages[0].extract_text.return_value = " First page\n text "
    source_pages[2].extract_text.return_value = None

    pdf_context = MagicMock()
    pdf_context.__enter__.return_value = SimpleNamespace(pages=source_pages)

    with patch("preprocess.pdfplumber.open", return_value=pdf_context):
        result = preprocess_pdf(
            source_pdf,
            output_json,
            date(2026, 3, 27),
            expected_total_pages=4,
            sample_pages=[3, 1],
        )

    saved_document = json.loads(output_json.read_text(encoding="utf-8"))
    assert result["total_pdf_pages"] == 4
    assert result["extracted_page_count"] == 2
    assert result["selected_pdf_pages"] == [1, 3]
    assert [page["pdf_page"] for page in saved_document["pages"]] == [1, 3]
    assert saved_document["pages"][0]["text"] == " First page\n text "
    assert saved_document["pages"][0]["extraction_status"] == "ok"
    assert saved_document["pages"][1]["text"] == ""
    assert saved_document["pages"][1]["extraction_status"] == "empty"
    assert result["quality_report"]["empty_text_pages"] == [3]
    assert source_pages[1].extract_text.call_count == 0
    assert source_pages[3].extract_text.call_count == 0
    assert source_pages[0].close.call_count == 1
    assert source_pages[1].close.call_count == 0
    assert source_pages[2].close.call_count == 1
    assert source_pages[3].close.call_count == 0


def test_preprocess_pdf_all_pages_records_errors_and_continues(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    source_pdf = tmp_path / "debate.pdf"
    source_pdf.touch()
    output_json = tmp_path / "intermediate" / "full.json"

    source_pages = [MagicMock() for _ in range(3)]
    source_pages[0].extract_text.return_value = "First page"
    source_pages[1].extract_text.side_effect = RuntimeError("simulated failure")
    source_pages[2].extract_text.return_value = "   "

    pdf_context = MagicMock()
    pdf_context.__enter__.return_value = SimpleNamespace(pages=source_pages)

    with patch("preprocess.pdfplumber.open", return_value=pdf_context):
        result = preprocess_pdf(
            source_pdf,
            output_json,
            date(2026, 3, 27),
            expected_total_pages=3,
        )

    saved_document = json.loads(output_json.read_text(encoding="utf-8"))
    assert result["selected_pdf_pages"] == [1, 2, 3]
    assert [page["pdf_page"] for page in saved_document["pages"]] == [1, 2, 3]
    assert saved_document["pages"][1]["text"] is None
    assert saved_document["pages"][1]["extraction_status"] == "error"
    assert saved_document["pages"][2]["text"] == "   "
    assert saved_document["quality_report"]["pages_processed"] == 3
    assert saved_document["quality_report"]["pages_with_text"] == 1
    assert saved_document["quality_report"]["empty_text_pages"] == [3]
    assert saved_document["quality_report"]["extraction_errors"] == [
        {
            "pdf_page": 2,
            "error_type": "RuntimeError",
            "message": "simulated failure",
        }
    ]
    assert saved_document["quality_report"]["warnings"] == [
        {
            "pdf_page": 2,
            "category": "extraction_error",
            "message": "simulated failure",
        },
        {
            "pdf_page": 3,
            "category": "empty_text",
            "message": "No non-whitespace text was extracted.",
        },
    ]
    assert "PDF page 2" in capsys.readouterr().err
    assert all(page.close.call_count == 1 for page in source_pages)


def test_preprocess_pdf_rejects_page_count_mismatch(tmp_path: Path):
    source_pdf = tmp_path / "debate.pdf"
    source_pdf.touch()

    pdf_context = MagicMock()
    pdf_context.__enter__.return_value = SimpleNamespace(pages=[MagicMock()])

    with patch("preprocess.pdfplumber.open", return_value=pdf_context):
        with pytest.raises(ValueError, match="Expected 2 PDF pages but found 1"):
            preprocess_pdf(
                source_pdf,
                tmp_path / "sample.json",
                date(2026, 3, 27),
                expected_total_pages=2,
                sample_pages=[1],
            )


def test_preprocess_pdf_rejects_sample_page_outside_pdf(tmp_path: Path):
    source_pdf = tmp_path / "debate.pdf"
    source_pdf.touch()

    pdf_context = MagicMock()
    pdf_context.__enter__.return_value = SimpleNamespace(pages=[MagicMock()])

    with patch("preprocess.pdfplumber.open", return_value=pdf_context):
        with pytest.raises(ValueError, match="outside the PDF page range"):
            preprocess_pdf(
                source_pdf,
                tmp_path / "sample.json",
                date(2026, 3, 27),
                expected_total_pages=1,
                sample_pages=[2],
            )


def test_preprocess_pdf_rejects_same_input_and_output_path(tmp_path: Path):
    source_pdf = tmp_path / "debate.pdf"
    source_pdf.touch()

    with pytest.raises(ValueError, match="different from the input PDF"):
        preprocess_pdf(
            source_pdf,
            source_pdf,
            date(2026, 3, 27),
            expected_total_pages=1,
            sample_pages=[1],
        )
