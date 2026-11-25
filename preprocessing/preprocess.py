"""Extract raw page text from Lok Sabha debate PDFs into intermediate JSON."""

import argparse
import json
import sys
import tempfile
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, TextIO

import pdfplumber


def validate_sample_pages(sample_pages: List[int]) -> List[int]:
    """Return unique, positive sample page numbers in ascending order."""
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


def _write_indented_json(output_file: TextIO, value: Any, indentation: int) -> None:
    """Write one JSON value with indentation relative to its parent object."""
    encoded_value = json.dumps(value, ensure_ascii=False, indent=2)
    prefix = " " * indentation
    output_file.write("\n".join(prefix + line for line in encoded_value.splitlines()))


def preprocess_pdf(
    input_pdf: Path,
    output_json: Path,
    debate_date: date,
    expected_total_pages: int,
    sample_pages: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Write selected pages or the complete PDF to JSON one page at a time.

    Set ``sample_pages`` to a list for sample mode. Leave it as ``None`` to
    extract every page. Raw text is streamed to disk without cleaning.
    """
    input_pdf = input_pdf.resolve()
    output_json = output_json.resolve()
    if input_pdf == output_json:
        raise ValueError("The output JSON path must be different from the input PDF path.")
    if expected_total_pages < 1:
        raise ValueError("Expected total pages must be a positive integer.")

    if sample_pages is not None:
        sample_pages = validate_sample_pages(sample_pages)

    empty_pages: List[int] = []
    extraction_errors: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []
    problem_pages: List[Dict[str, Any]] = []

    with pdfplumber.open(str(input_pdf)) as pdf:
        total_pdf_pages = len(pdf.pages)
        if total_pdf_pages != expected_total_pages:
            raise ValueError(
                f"Expected {expected_total_pages} PDF pages but found {total_pdf_pages}."
            )

        page_numbers = (
            list(range(1, total_pdf_pages + 1))
            if sample_pages is None
            else sample_pages
        )
        if page_numbers[-1] > total_pdf_pages:
            raise ValueError(
                f"Sample page {page_numbers[-1]} is outside the PDF page range "
                f"1-{total_pdf_pages}."
            )

        output_json.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Optional[Path] = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="\n",
                dir=output_json.parent,
                prefix=f".{output_json.name}.",
                suffix=".tmp",
                delete=False,
            ) as output_file:
                temporary_path = Path(output_file.name)
                metadata = {
                    "dataset_name": (
                        f"Lok Sabha Debate - {debate_date.strftime('%d %B %Y')}"
                    ),
                    "source_file": input_pdf.name,
                    "date": debate_date.isoformat(),
                    "document_type": "Lok Sabha Debate",
                    "extraction_method": "pdfplumber page.extract_text()",
                    "ocr_performed": False,
                    "total_pdf_pages": total_pdf_pages,
                    "extracted_page_count": len(page_numbers),
                    "selected_pdf_pages": page_numbers,
                }

                output_file.write("{\n")
                for key, value in metadata.items():
                    output_file.write(
                        f"  {json.dumps(key)}: "
                        f"{json.dumps(value, ensure_ascii=False)},\n"
                    )
                output_file.write('  "pages": [')

                for index, page_number in enumerate(page_numbers):
                    page = pdf.pages[page_number - 1]
                    try:
                        text = page.extract_text()
                    except Exception as exc:
                        error_record = {
                            "pdf_page": page_number,
                            "error_type": type(exc).__name__,
                            "message": str(exc),
                        }
                        extraction_errors.append(error_record)
                        warning_record = {
                            "pdf_page": page_number,
                            "category": "extraction_error",
                            "message": str(exc),
                        }
                        warnings.append(warning_record)
                        problem_pages.append(
                            {
                                "pdf_page": page_number,
                                "issue": "extraction_error",
                                "error_type": type(exc).__name__,
                                "message": str(exc),
                            }
                        )
                        page_record = {
                            "pdf_page": page_number,
                            "text": None,
                            "is_empty": None,
                            "extraction_status": "error",
                            "extraction_error": {
                                "type": type(exc).__name__,
                                "message": str(exc),
                            },
                        }
                        print(
                            f"ERROR: text extraction failed on PDF page "
                            f"{page_number}: {type(exc).__name__}: {exc}",
                            file=sys.stderr,
                        )
                    else:
                        text = text or ""
                        is_empty = not text.strip()
                        page_record = {
                            "pdf_page": page_number,
                            "text": text,
                            "is_empty": is_empty,
                            "extraction_status": "empty" if is_empty else "ok",
                        }
                        if is_empty:
                            empty_pages.append(page_number)
                            warning_record = {
                                "pdf_page": page_number,
                                "category": "empty_text",
                                "message": "No non-whitespace text was extracted.",
                            }
                            warnings.append(warning_record)
                            problem_pages.append(
                                {"pdf_page": page_number, "issue": "empty_text"}
                            )
                    finally:
                        page.close()

                    output_file.write("\n")
                    _write_indented_json(output_file, page_record, indentation=4)
                    output_file.write("\n")
                    if index < len(page_numbers) - 1:
                        output_file.write("    ,")
                    else:
                        output_file.write("  ")

                quality_report = {
                    "pages_processed": len(page_numbers),
                    "pages_with_text": (
                        len(page_numbers) - len(empty_pages) - len(extraction_errors)
                    ),
                    "empty_text_pages": empty_pages,
                    "failed_pages": [
                        issue["pdf_page"] for issue in extraction_errors
                    ],
                    "extraction_errors": extraction_errors,
                    "warnings": warnings,
                    "problem_pages": problem_pages,
                }
                output_file.write('\n  ],\n  "quality_report": ')
                _write_indented_json(output_file, quality_report, indentation=2)
                output_file.write("\n}\n")

            temporary_path.replace(output_json)
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()

    return {**metadata, "quality_report": quality_report}


def parse_sample_pages(value: str) -> List[int]:
    """Parse comma-separated, one-based PDF page numbers for argparse."""
    try:
        pages = [int(item.strip()) for item in value.split(",")]
        return validate_sample_pages(pages)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract selected pages or the complete PDF into JSON."
    )
    parser.add_argument("--input", type=Path, required=True, help="Source PDF file")
    parser.add_argument("--output", type=Path, required=True, help="Destination JSON file")
    parser.add_argument(
        "--date",
        type=date.fromisoformat,
        required=True,
        help="Debate date in YYYY-MM-DD format",
    )
    page_mode = parser.add_mutually_exclusive_group(required=True)
    page_mode.add_argument(
        "--sample-pages",
        type=parse_sample_pages,
        help="Comma-separated, one-based PDF page numbers, e.g. 1,650,1300",
    )
    page_mode.add_argument(
        "--all-pages",
        action="store_true",
        help="Extract every page in the PDF",
    )
    parser.add_argument(
        "--expected-total-pages",
        type=int,
        required=True,
        help="Expected page count of the complete source PDF",
    )
    args = parser.parse_args()

    document = preprocess_pdf(
        args.input,
        args.output,
        args.date,
        args.expected_total_pages,
        args.sample_pages,
    )
    quality_report = document["quality_report"]
    error_count = len(quality_report["extraction_errors"])
    print(f"Total pages: {document['total_pdf_pages']}")
    print(f"Pages processed: {quality_report['pages_processed']}")
    print(f"Pages with text: {quality_report['pages_with_text']}")
    print(f"Empty pages: {len(quality_report['empty_text_pages'])}")
    print(f"Failed pages: {error_count}")
    if quality_report["failed_pages"]:
        print(f"Failed PDF page numbers: {quality_report['failed_pages']}")
    print(f"Output: {args.output}")
    if error_count:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
