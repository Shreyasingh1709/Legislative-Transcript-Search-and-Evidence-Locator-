import json
from pathlib import Path

import pytest

from english_only_dataset import (
    build_english_dataset,
    generate_english_dataset,
)


def _page(
    pdf_page: int,
    cleaned_text: str,
    original_text: str = None,
    quality_warnings=None,
) -> dict:
    return {
        "document_id": "LS_2026_03_27",
        "source_file": "march.pdf",
        "date": "2026-03-27",
        "pdf_page": pdf_page,
        "original_text": cleaned_text if original_text is None else original_text,
        "cleaned_text": cleaned_text,
        "is_empty": not cleaned_text,
        "extraction_status": "ok",
        "quality_warnings": [] if quality_warnings is None else quality_warnings,
        "other_page_metadata": {"keep": True},
    }


def _dataset(pages: list) -> dict:
    return {
        "dataset_name": "Lok Sabha Debate Dataset",
        "document_count": 1,
        "total_pdf_pages": len(pages),
        "documents": [
            {
                "document_id": "LS_2026_03_27",
                "source_file": "march.pdf",
                "date": "2026-03-27",
                "total_pdf_pages": len(pages),
            }
        ],
        "pages": pages,
    }


def test_build_english_dataset_partitions_pages_conservatively():
    english_page = _page(
        1,
        "The minister answered the question and provided information to the members.",
    )
    hindi_page = _page(2, "\u092f\u0939 \u0939\u093f\u0902\u0926\u0940 \u092a\u093e\u0920 \u0939\u0948\u0964")
    mixed_page = _page(
        3,
        "The minister answered the question.\n\u092f\u0939 \u0939\u093f\u0902\u0926\u0940 \u092a\u093e\u0920 \u0939\u0948\u0964",
    )
    cid_page = _page(
        4,
        "The minister answered the question (cid:164).",
        quality_warnings=["cid_glyph_placeholder"],
    )
    uncertain_latin_page = _page(5, "Question No. 481")
    repeated_word_page = _page(6, "the the the the the the the the the the")

    output, summary = build_english_dataset(
        _dataset(
            [
                english_page,
                hindi_page,
                mixed_page,
                cid_page,
                uncertain_latin_page,
                repeated_word_page,
            ]
        )
    )

    assert [page["pdf_page"] for page in output["pages"]] == [1]
    assert output["pages"][0]["retrieval_text"] == english_page["cleaned_text"]
    assert output["pages"][0]["other_page_metadata"] == {"keep": True}
    assert [page["pdf_page"] for page in output["uncertain_records"]] == [3, 4, 5, 6]
    assert output["uncertain_records"][0]["cleaned_text"] == mixed_page["cleaned_text"]
    assert output["uncertain_records"][1]["quality_warnings"] == [
        "cid_glyph_placeholder"
    ]
    assert output["uncertain_records"][1]["cleaned_text"] == cid_page["cleaned_text"]
    assert output["excluded_record_count"] == 1
    assert output["uncertain_record_count"] == 4
    assert summary["retained"] + summary["excluded"] + summary["uncertain"] == 6


def test_generate_english_dataset_preserves_input_and_writes_utf8(tmp_path: Path):
    input_path = tmp_path / "combined.json"
    output_path = tmp_path / "processed" / "english.json"
    mixed_text = (
        "The minister answered the question.\n"
        "\u092f\u0939 \u0939\u093f\u0902\u0926\u0940 \u092a\u093e\u0920 \u0939\u0948\u0964"
    )
    source = _dataset(
        [
            _page(
                1,
                "The minister answered the question and provided information to the members.",
            ),
            _page(2, mixed_text),
        ]
    )
    input_path.write_text(json.dumps(source, ensure_ascii=False), encoding="utf-8")

    summary = generate_english_dataset(input_path, output_path)
    output_text = output_path.read_text(encoding="utf-8")
    output = json.loads(output_text)

    assert len(output["pages"]) == 1
    assert len(output["uncertain_records"]) == 1
    assert output["pages"][0]["document_id"] == "LS_2026_03_27"
    assert output["pages"][0]["source_file"] == "march.pdf"
    assert output["pages"][0]["date"] == "2026-03-27"
    assert output["pages"][0]["quality_warnings"] == []
    assert output["uncertain_records"][0]["cleaned_text"] == mixed_text
    assert "\u0939" in output_text
    assert "\\u0939" not in output_text
    assert summary["input_unchanged"] is True
    assert json.loads(input_path.read_text(encoding="utf-8")) == source


def test_generate_english_dataset_rejects_output_equal_to_input(tmp_path: Path):
    input_path = tmp_path / "combined.json"
    input_path.write_text(json.dumps(_dataset([_page(1, "The minister answered the question and provided information.")])) , encoding="utf-8")

    with pytest.raises(ValueError, match="must not overwrite any source JSON"):
        generate_english_dataset(input_path, input_path)


def test_generate_english_dataset_protects_cleaned_source_files(tmp_path: Path):
    input_path = tmp_path / "combined.json"
    input_path.write_text(
        json.dumps(
            _dataset(
                [
                    _page(
                        1,
                        "The minister answered the question and provided information to members.",
                    )
                ]
            )
        ),
        encoding="utf-8",
    )
    protected_cleaned_input = Path(r"data\processed\march_27_cleaned.json")

    with pytest.raises(ValueError, match="must not overwrite any source JSON"):
        generate_english_dataset(input_path, protected_cleaned_input)
