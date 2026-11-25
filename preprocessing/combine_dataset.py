"""Combine the two cleaned Lok Sabha debate JSON documents."""

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple


DATASET_NAME = "Lok Sabha Debate Dataset"
REQUIRED_PAGE_FIELDS = {
    "pdf_page",
    "original_text",
    "cleaned_text",
    "is_empty",
    "quality_warnings",
}


@dataclass(frozen=True)
class DocumentInput:
    path: Path
    fallback_document_id: str
    expected_page_count: int


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_and_validate_source(
    source: DocumentInput,
) -> Tuple[Dict[str, Any], str]:
    if source.expected_page_count < 1:
        raise ValueError(
            f"{source.path}: expected page count must be a positive integer."
        )
    if not source.fallback_document_id.strip():
        raise ValueError(f"{source.path}: document ID must not be empty.")

    with source.path.open("r", encoding="utf-8") as input_file:
        document = json.load(input_file)
    if not isinstance(document, dict):
        raise ValueError(f"{source.path}: JSON root must be an object.")

    source_file = document.get("source_file")
    debate_date = document.get("date")
    pages = document.get("pages")
    if not isinstance(source_file, str) or not source_file:
        raise ValueError(f"{source.path}: missing valid source_file metadata.")
    if not isinstance(debate_date, str) or not debate_date:
        raise ValueError(f"{source.path}: missing valid date metadata.")
    if not isinstance(pages, list):
        raise ValueError(f"{source.path}: pages must be a JSON array.")

    for count_field in (
        "total_pdf_pages",
        "source_extracted_page_count",
        "extracted_page_count",
    ):
        if count_field in document and document[count_field] != source.expected_page_count:
            raise ValueError(
                f"{source.path}: {count_field} is {document[count_field]}, "
                f"expected {source.expected_page_count}."
            )

    if len(pages) != source.expected_page_count:
        raise ValueError(
            f"{source.path}: contains {len(pages)} page records, "
            f"expected {source.expected_page_count}."
        )

    page_numbers: List[int] = []
    for index, page in enumerate(pages, start=1):
        if not isinstance(page, dict):
            raise ValueError(
                f"{source.path}: page record {index} must be a JSON object."
            )
        missing_fields = REQUIRED_PAGE_FIELDS.difference(page)
        if missing_fields:
            raise ValueError(
                f"{source.path}: page record {index} is missing required fields "
                f"{sorted(missing_fields)}."
            )
        page_number = page["pdf_page"]
        if (
            not isinstance(page_number, int)
            or isinstance(page_number, bool)
            or page_number < 1
        ):
            raise ValueError(
                f"{source.path}: page record {index} has an invalid pdf_page."
            )
        page_numbers.append(page_number)
        if not isinstance(page["quality_warnings"], list):
            raise ValueError(
                f"{source.path}: page {page_number} quality_warnings must be an array."
            )
        if "source_file" in page and page["source_file"] != source_file:
            raise ValueError(
                f"{source.path}: page {page_number} has conflicting source_file metadata."
            )
        if "date" in page and page["date"] != debate_date:
            raise ValueError(
                f"{source.path}: page {page_number} has conflicting date metadata."
            )

    duplicates = sorted(
        number
        for number, count in Counter(page_numbers).items()
        if count > 1
    )
    expected_numbers = list(range(1, source.expected_page_count + 1))
    if duplicates or page_numbers != expected_numbers:
        missing = sorted(set(expected_numbers).difference(page_numbers))
        unexpected = sorted(set(page_numbers).difference(expected_numbers))
        raise ValueError(
            f"{source.path}: PDF page numbers must be unique and sequential "
            f"from 1 to {source.expected_page_count}; duplicates={duplicates[:10]}, "
            f"missing={missing[:10]}, unexpected={unexpected[:10]}."
        )

    existing_document_id = document.get("document_id")
    if existing_document_id is not None and (
        not isinstance(existing_document_id, str) or not existing_document_id.strip()
    ):
        raise ValueError(f"{source.path}: document_id must be a non-empty string.")
    document_id = existing_document_id or source.fallback_document_id

    for page in pages:
        page_document_id = page.get("document_id")
        if page_document_id is not None and page_document_id != document_id:
            raise ValueError(
                f"{source.path}: page {page['pdf_page']} has conflicting document_id."
            )

    return document, document_id


def _build_combined_document(
    sources: Sequence[DocumentInput],
    loaded_sources: Sequence[Tuple[Dict[str, Any], str]],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if len(sources) != 2 or len(loaded_sources) != 2:
        raise ValueError("The combined PS 160 dataset requires exactly two documents.")

    document_ids = [document_id for _, document_id in loaded_sources]
    if len(set(document_ids)) != len(document_ids):
        raise ValueError("Each source document must have a unique document_id.")

    documents: List[Dict[str, Any]] = []
    pages: List[Dict[str, Any]] = []
    document_summaries: List[Dict[str, Any]] = []
    source_page_fields = set()

    for source, (source_document, document_id) in zip(sources, loaded_sources):
        source_pages = source_document["pages"]
        source_file = source_document["source_file"]
        debate_date = source_document["date"]

        document_metadata = {
            key: value for key, value in source_document.items() if key != "pages"
        }
        document_metadata["document_id"] = document_id
        documents.append(document_metadata)

        for source_page in source_pages:
            source_page_fields.update(source_page)
            combined_page = dict(source_page)
            combined_page.setdefault("document_id", document_id)
            combined_page.setdefault("source_file", source_file)
            combined_page.setdefault("date", debate_date)
            pages.append(combined_page)

        document_summaries.append(
            {
                "document_id": document_id,
                "source_file": source_file,
                "date": debate_date,
                "expected_pages": source.expected_page_count,
                "actual_pages": len(source_pages),
                "first_pdf_page": source_pages[0]["pdf_page"],
                "last_pdf_page": source_pages[-1]["pdf_page"],
                "duplicate_page_numbers": 0,
                "page_numbers_sequential": True,
                "page_source_metadata_complete": all(
                    page["document_id"] == document_id
                    and page["source_file"] == source_file
                    and page["date"] == debate_date
                    for page in pages[-len(source_pages) :]
                ),
                "original_page_fields_preserved": all(
                    all(combined_page.get(key) == value for key, value in source_page.items())
                    for source_page, combined_page in zip(
                        source_pages, pages[-len(source_pages) :]
                    )
                ),
                "pages_with_quality_warnings": sum(
                    bool(page["quality_warnings"]) for page in source_pages
                ),
            }
        )

    expected_total_pages = sum(source.expected_page_count for source in sources)
    combined_document = {
        "dataset_name": DATASET_NAME,
        "document_count": len(documents),
        "total_pdf_pages": len(pages),
        "documents": documents,
        "pages": pages,
    }

    character_count_fields = sorted(
        field
        for field in source_page_fields
        if "char" in field.lower() and "count" in field.lower()
    )
    summary = {
        "document_count": len(documents),
        "total_pdf_pages": len(pages),
        "expected_total_pdf_pages": expected_total_pages,
        "documents": document_summaries,
        "preserved_page_fields": sorted(source_page_fields),
        "character_count_fields": character_count_fields,
        "errors": [],
    }

    if len(pages) != expected_total_pages:
        raise ValueError(
            f"Combined page count is {len(pages)}, expected {expected_total_pages}."
        )
    if not all(
        item["page_source_metadata_complete"]
        and item["original_page_fields_preserved"]
        and item["page_numbers_sequential"]
        for item in document_summaries
    ):
        raise ValueError("Combined page validation failed.")

    return combined_document, summary


def combine_dataset(
    sources: Sequence[DocumentInput],
    output_path: Path,
) -> Dict[str, Any]:
    """Combine two cleaned documents and validate the persisted UTF-8 JSON."""
    output_path = output_path.resolve()
    resolved_inputs = [source.path.resolve() for source in sources]
    if output_path in resolved_inputs:
        raise ValueError("The combined output path must not overwrite an input file.")
    if len(set(resolved_inputs)) != len(resolved_inputs):
        raise ValueError("The two input paths must refer to different files.")

    input_hashes = {
        source.path.resolve(): _sha256_file(source.path) for source in sources
    }
    loaded_sources = [
        _load_and_validate_source(source) for source in sources
    ]
    combined_document, summary = _build_combined_document(sources, loaded_sources)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as output_file:
            temporary_path = Path(output_file.name)
            json.dump(combined_document, output_file, ensure_ascii=False, indent=2)
            output_file.write("\n")
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()

    with output_path.open("r", encoding="utf-8") as output_file:
        saved_document = json.load(output_file)
    if saved_document != combined_document:
        raise ValueError("The saved combined JSON does not match the validated data.")

    for source, (source_document, document_id) in zip(sources, loaded_sources):
        if _sha256_file(source.path) != input_hashes[source.path.resolve()]:
            raise ValueError(f"Input file changed during combination: {source.path}")
        saved_pages = [
            page
            for page in saved_document["pages"]
            if page["document_id"] == document_id
        ]
        if len(saved_pages) != len(source_document["pages"]):
            raise ValueError(
                f"Page records were not preserved for input file: {source.path}"
            )
        for source_page, saved_page in zip(source_document["pages"], saved_pages):
            if any(
                key not in saved_page or saved_page[key] != value
                for key, value in source_page.items()
            ):
                raise ValueError(
                    f"Page fields were not preserved for PDF page "
                    f"{source_page['pdf_page']} from {source.path}."
                )
            if (
                saved_page["source_file"] != source_document["source_file"]
                or saved_page["date"] != source_document["date"]
            ):
                raise ValueError(
                    f"Page provenance validation failed for PDF page "
                    f"{source_page['pdf_page']} from {source.path}."
                )

    summary["input_files_unchanged"] = True
    summary["output_path"] = str(output_path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Combine the two cleaned PS 160 Lok Sabha debate JSON files."
    )
    parser.add_argument(
        "--march-input",
        type=Path,
        default=Path(r"data\processed\march_27_cleaned.json"),
        help="Cleaned March debate JSON input",
    )
    parser.add_argument(
        "--july-input",
        type=Path,
        default=Path(r"data\processed\july_20_cleaned.json"),
        help="Cleaned July debate JSON input",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(r"data\processed\combined_processed.json"),
        help="Combined JSON output",
    )
    args = parser.parse_args()

    sources = [
        DocumentInput(args.march_input, "LS_2026_03_27", 1300),
        DocumentInput(args.july_input, "LS_2026_07_20", 1315),
    ]

    try:
        summary = combine_dataset(sources, args.output)
    except (OSError, ValueError) as exc:
        print("Validation: FAILED")
        print(f"Errors: {exc}")
        raise SystemExit(2) from exc

    print("Validation: PASSED")
    for document in summary["documents"]:
        print(
            f"{document['document_id']}: expected {document['expected_pages']}, "
            f"actual {document['actual_pages']}; PDF pages "
            f"{document['first_pdf_page']}-{document['last_pdf_page']}; "
            f"duplicates {document['duplicate_page_numbers']}; "
            f"warning pages {document['pages_with_quality_warnings']}."
        )
    print(f"Documents: {summary['document_count']}")
    print(
        f"Combined pages: {summary['total_pdf_pages']} "
        f"(expected {summary['expected_total_pdf_pages']})"
    )
    print("Input files unchanged: yes")
    if summary["character_count_fields"]:
        print(
            "Preserved character-count fields: "
            f"{summary['character_count_fields']}"
        )
    else:
        print("Character-count fields: none present in the input page records")
    print(f"Preserved page fields: {summary['preserved_page_fields']}")
    print("Errors: none")
    print(f"Output: {summary['output_path']}")


if __name__ == "__main__":
    main()
