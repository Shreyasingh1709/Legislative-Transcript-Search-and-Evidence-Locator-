"""Clean extracted page text while preserving raw text and paragraph breaks."""

import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional


_HORIZONTAL_WHITESPACE = re.compile(r"[^\S\n]+")
_EXCESS_BLANK_LINES = re.compile(r"\n{3,}")
_CID_PLACEHOLDER = re.compile(r"\(cid:\d+\)", re.IGNORECASE)
_MOJIBAKE_MARKER = re.compile(r"(?:à¤|à¥)")


def clean_text(text: str) -> str:
    """Normalize whitespace without joining lines or changing non-whitespace."""
    normalized_line_endings = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [
        _HORIZONTAL_WHITESPACE.sub(" ", line).strip()
        for line in normalized_line_endings.split("\n")
    ]
    cleaned = "\n".join(lines)
    return _EXCESS_BLANK_LINES.sub("\n\n", cleaned).strip("\n")


def find_suspicious_sequences(text: str) -> List[str]:
    """Return warning codes for suspicious text without modifying the input."""
    warnings: List[str] = []
    if _CID_PLACEHOLDER.search(text):
        warnings.append("cid_glyph_placeholder")
    if "\ufffd" in text:
        warnings.append("unicode_replacement_character")
    if any(unicodedata.category(character) == "Co" for character in text):
        warnings.append("private_use_character")
    if _MOJIBAKE_MARKER.search(text):
        warnings.append("possible_hindi_mojibake")
    if any(
        unicodedata.category(character) == "Cc"
        and character not in "\n\r\t"
        for character in text
    ):
        warnings.append("unexpected_control_character")
    return warnings


def validate_sample_pages(sample_pages: List[int]) -> List[int]:
    """Return unique, positive page numbers in ascending order."""
    if not sample_pages:
        raise ValueError("At least one sample page must be selected.")
    if any(
        not isinstance(page, int) or isinstance(page, bool) or page < 1
        for page in sample_pages
    ):
        raise ValueError("Sample page numbers must be positive integers.")
    if len(set(sample_pages)) != len(sample_pages):
        raise ValueError("Sample page numbers must not contain duplicates.")
    return sorted(sample_pages)


def parse_sample_pages(value: str) -> List[int]:
    """Parse comma-separated, one-based PDF page numbers for argparse."""
    try:
        pages = [int(item.strip()) for item in value.split(",")]
        return validate_sample_pages(pages)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def clean_page_record(page: Dict[str, Any]) -> Dict[str, Any]:
    """Create a cleaned page record while retaining extraction fields and raw text."""
    if "pdf_page" not in page or "text" not in page:
        raise ValueError("Each page record must contain pdf_page and text.")

    original_text = page["text"]
    if original_text is not None and not isinstance(original_text, str):
        raise TypeError(f"Page {page['pdf_page']} text must be a string or null.")

    cleaned_page = {key: value for key, value in page.items() if key != "text"}
    cleaned_page["original_text"] = original_text
    cleaned_page["cleaned_text"] = (
        clean_text(original_text) if original_text is not None else None
    )
    cleaned_page["quality_warnings"] = (
        find_suspicious_sequences(original_text) if original_text is not None else []
    )
    return cleaned_page


def clean_document(
    input_json: Path,
    output_json: Path,
    sample_pages: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Clean one extracted JSON document, optionally limiting output to sample pages."""
    input_json = input_json.resolve()
    output_json = output_json.resolve()
    if input_json == output_json:
        raise ValueError("The output JSON path must be different from the input JSON path.")

    with input_json.open("r", encoding="utf-8") as input_file:
        source_document = json.load(input_file)
    if not isinstance(source_document, dict) or not isinstance(
        source_document.get("pages"), list
    ):
        raise ValueError("Input JSON must contain a document object with a pages array.")

    source_pages = source_document["pages"]
    page_numbers = [page.get("pdf_page") for page in source_pages]
    if any(not isinstance(number, int) or number < 1 for number in page_numbers):
        raise ValueError("Every source page must have a positive integer pdf_page.")
    if len(set(page_numbers)) != len(page_numbers):
        raise ValueError("Source JSON contains duplicate PDF page numbers.")

    if sample_pages is None:
        selected_pages = source_pages
        selected_page_numbers = page_numbers
    else:
        selected_page_numbers = validate_sample_pages(sample_pages)
        pages_by_number = {page["pdf_page"]: page for page in source_pages}
        missing_pages = [
            number for number in selected_page_numbers if number not in pages_by_number
        ]
        if missing_pages:
            raise ValueError(f"Requested sample pages are missing: {missing_pages}")
        selected_pages = [pages_by_number[number] for number in selected_page_numbers]

    cleaned_pages = [clean_page_record(page) for page in selected_pages]
    warning_pages = [
        {
            "pdf_page": page["pdf_page"],
            "warnings": page["quality_warnings"],
        }
        for page in cleaned_pages
        if page["quality_warnings"]
    ]
    source_quality_report = source_document.get("quality_report", {})
    cleaned_document = {
        key: value
        for key, value in source_document.items()
        if key
        not in {
            "pages",
            "quality_report",
            "extracted_page_count",
            "selected_pdf_pages",
        }
    }
    cleaned_document.update(
        {
            "source_extracted_page_count": source_document.get(
                "extracted_page_count", len(source_pages)
            ),
            "extracted_page_count": len(cleaned_pages),
            "selected_pdf_pages": selected_page_numbers,
            "source_extraction_quality_report": source_quality_report,
            "cleaning_report": {
                "pages_processed": len(cleaned_pages),
                "pages_with_text": sum(
                    bool(page["cleaned_text"] and page["cleaned_text"].strip())
                    for page in cleaned_pages
                ),
                "pages_with_warnings": len(warning_pages),
                "warning_pages": warning_pages,
            },
            "pages": cleaned_pages,
        }
    )

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8", newline="\n") as output_file:
        json.dump(cleaned_document, output_file, ensure_ascii=False, indent=2)
        output_file.write("\n")
    return cleaned_document


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clean extracted page text while preserving original text."
    )
    parser.add_argument("--input", type=Path, required=True, help="Extracted JSON file")
    parser.add_argument("--output", type=Path, required=True, help="Cleaned JSON file")
    page_mode = parser.add_mutually_exclusive_group(required=True)
    page_mode.add_argument(
        "--sample-pages",
        type=parse_sample_pages,
        help="Comma-separated, one-based PDF page numbers to clean",
    )
    page_mode.add_argument(
        "--all-pages",
        action="store_true",
        help="Clean every page in this one input JSON document",
    )
    args = parser.parse_args()

    cleaned_document = clean_document(
        args.input,
        args.output,
        args.sample_pages,
    )
    report = cleaned_document["cleaning_report"]
    print(f"Pages processed: {report['pages_processed']}")
    print(f"Pages with text: {report['pages_with_text']}")
    print(f"Pages with suspicious-text warnings: {report['pages_with_warnings']}")
    if report["warning_pages"]:
        print(f"Warning page details: {report['warning_pages']}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
