import json
from pathlib import Path

import pytest

from combine_dataset import DocumentInput, combine_dataset


def _write_source(
    path: Path,
    source_file: str,
    debate_date: str,
    pages: list,
) -> dict:
    document = {
        "dataset_name": f"Lok Sabha Debate - {debate_date}",
        "source_file": source_file,
        "date": debate_date,
        "document_type": "Lok Sabha Debate",
        "total_pdf_pages": len(pages),
        "source_extracted_page_count": len(pages),
        "extracted_page_count": len(pages),
        "source_extraction_quality_report": {"failed_pages": []},
        "cleaning_report": {"pages_with_warnings": 1},
        "pages": pages,
    }
    path.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
    return document


def _page(pdf_page: int, original_text: str, cleaned_text: str) -> dict:
    return {
        "pdf_page": pdf_page,
        "original_text": original_text,
        "cleaned_text": cleaned_text,
        "is_empty": False,
        "extraction_status": "ok",
        "quality_warnings": ["cid_glyph_placeholder"] if "(cid:" in original_text else [],
        "custom_page_metadata": {"retained": True},
    }


def test_combine_dataset_preserves_pages_and_local_page_numbers(tmp_path: Path):
    march_path = tmp_path / "march.json"
    july_path = tmp_path / "july.json"
    output_path = tmp_path / "processed" / "combined.json"
    march_document = _write_source(
        march_path,
        "march.pdf",
        "2026-03-27",
        [_page(1, "नमस्ते (cid:164)", "नमस्ते (cid:164)")],
    )
    july_document = _write_source(
        july_path,
        "july.pdf",
        "2026-07-20",
        [
            _page(1, "first", "first"),
            _page(2, "second", "second"),
        ],
    )

    summary = combine_dataset(
        [
            DocumentInput(march_path, "LS_2026_03_27", 1),
            DocumentInput(july_path, "LS_2026_07_20", 2),
        ],
        output_path,
    )
    combined = json.loads(output_path.read_text(encoding="utf-8"))

    assert combined["dataset_name"] == "Lok Sabha Debate Dataset"
    assert combined["document_count"] == 2
    assert combined["total_pdf_pages"] == 3
    assert [document["document_id"] for document in combined["documents"]] == [
        "LS_2026_03_27",
        "LS_2026_07_20",
    ]
    assert [
        (page["document_id"], page["pdf_page"])
        for page in combined["pages"]
    ] == [
        ("LS_2026_03_27", 1),
        ("LS_2026_07_20", 1),
        ("LS_2026_07_20", 2),
    ]
    assert combined["pages"][0]["quality_warnings"] == ["cid_glyph_placeholder"]
    assert combined["pages"][0]["custom_page_metadata"] == {"retained": True}
    assert combined["pages"][0]["source_file"] == "march.pdf"
    assert combined["pages"][0]["date"] == "2026-03-27"
    assert combined["pages"][0]["original_text"] == march_document["pages"][0]["original_text"]
    assert combined["pages"][0]["cleaned_text"] == march_document["pages"][0]["cleaned_text"]
    assert combined["pages"][-1]["source_file"] == "july.pdf"
    assert combined["pages"][-1]["date"] == "2026-07-20"
    serialized_output = output_path.read_text(encoding="utf-8")
    assert "नमस्ते" in serialized_output
    assert "\\u0928" not in serialized_output
    assert json.loads(march_path.read_text(encoding="utf-8")) == march_document
    assert json.loads(july_path.read_text(encoding="utf-8")) == july_document
    assert summary["total_pdf_pages"] == 3
    assert summary["errors"] == []
    assert summary["input_files_unchanged"] is True


def test_combine_dataset_rejects_duplicate_page_numbers(tmp_path: Path):
    march_path = tmp_path / "march.json"
    july_path = tmp_path / "july.json"
    output_path = tmp_path / "combined.json"
    _write_source(
        march_path,
        "march.pdf",
        "2026-03-27",
        [_page(1, "first", "first"), _page(1, "duplicate", "duplicate")],
    )
    _write_source(
        july_path,
        "july.pdf",
        "2026-07-20",
        [_page(1, "july", "july")],
    )

    with pytest.raises(ValueError, match="unique and sequential"):
        combine_dataset(
            [
                DocumentInput(march_path, "LS_2026_03_27", 2),
                DocumentInput(july_path, "LS_2026_07_20", 1),
            ],
            output_path,
        )

    assert not output_path.exists()


def test_combine_dataset_rejects_output_that_overwrites_an_input(tmp_path: Path):
    march_path = tmp_path / "march.json"
    july_path = tmp_path / "july.json"
    _write_source(march_path, "march.pdf", "2026-03-27", [_page(1, "a", "a")])
    _write_source(july_path, "july.pdf", "2026-07-20", [_page(1, "b", "b")])

    with pytest.raises(ValueError, match="must not overwrite an input"):
        combine_dataset(
            [
                DocumentInput(march_path, "LS_2026_03_27", 1),
                DocumentInput(july_path, "LS_2026_07_20", 1),
            ],
            march_path,
        )
