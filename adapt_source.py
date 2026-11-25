"""Adapt page-level English retrieval records for the transcript chunker."""

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple


BASE_DIR = Path(__file__).resolve().parent
REQUIRED_FIELDS = {
    "document_id",
    "source_file",
    "date",
    "pdf_page",
    "retrieval_text",
    "quality_warnings",
    "language_status",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def adapt_dataset(dataset: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Adapt only retained English pages without fabricating attribution."""
    if not isinstance(dataset, dict):
        raise ValueError("English-only input JSON root must be an object.")
    pages = dataset.get("pages")
    documents = dataset.get("documents")
    if not isinstance(pages, list) or not isinstance(documents, list):
        raise ValueError("Input must contain documents and pages arrays.")
    if dataset.get("retained_record_count") != len(pages):
        raise ValueError(
            "retained_record_count does not match the pages array length."
        )

    documents_by_id = {}
    for document in documents:
        if not isinstance(document, dict):
            raise ValueError("Each document metadata record must be an object.")
        document_id = document.get("document_id")
        if not isinstance(document_id, str) or not document_id:
            raise ValueError("Each source document must have a document_id.")
        if document_id in documents_by_id:
            raise ValueError(f"Duplicate document_id: {document_id}")
        documents_by_id[document_id] = document

    records: List[Dict[str, Any]] = []
    seen_page_ids = set()
    seen_record_ids = set()
    record_counts = Counter()
    for index, page in enumerate(pages, start=1):
        if not isinstance(page, dict):
            raise ValueError(f"Page record {index} must be a JSON object.")
        missing_fields = REQUIRED_FIELDS.difference(page)
        if missing_fields:
            raise ValueError(
                f"Page record {index} is missing fields {sorted(missing_fields)}."
            )
        document_id = page["document_id"]
        if document_id not in documents_by_id:
            raise ValueError(
                f"Page record {index} references unknown document_id {document_id!r}."
            )
        document = documents_by_id[document_id]
        if (
            page["source_file"] != document.get("source_file")
            or page["date"] != document.get("date")
        ):
            raise ValueError(
                f"Page record {index} has inconsistent document provenance."
            )
        page_number = page["pdf_page"]
        if (
            not isinstance(page_number, int)
            or isinstance(page_number, bool)
            or page_number < 1
        ):
            raise ValueError(f"Page record {index} has an invalid pdf_page.")
        if page["language_status"] != "english_only":
            raise ValueError(
                f"Page record {index} is not marked english_only; "
                "uncertain records must not enter retrieval."
            )
        retrieval_text = page["retrieval_text"]
        if not isinstance(retrieval_text, str) or not retrieval_text.strip():
            raise ValueError(
                f"Page record {index} has missing or empty retrieval_text."
            )
        if not isinstance(page["quality_warnings"], list):
            raise ValueError(f"Page record {index} quality_warnings must be an array.")

        page_key = (document_id, page_number)
        if page_key in seen_page_ids:
            raise ValueError(
                f"Duplicate source page: {document_id}, PDF page {page_number}."
            )
        seen_page_ids.add(page_key)

        existing_record_id = page.get("record_id")
        record_id = existing_record_id or f"{document_id}_page_{page_number:04d}"
        if not isinstance(record_id, str) or not record_id.strip():
            raise ValueError(f"Page record {index} has an invalid record_id.")
        if record_id in seen_record_ids:
            raise ValueError(f"Duplicate record_id: {record_id}")
        seen_record_ids.add(record_id)

        record = dict(page)
        record["record_id"] = record_id
        record["text"] = retrieval_text
        record["speaker"] = "Unknown"
        record["attribution_status"] = "unverified"
        records.append(record)
        record_counts[document_id] += 1

    report = {
        "source_page_count": len(pages),
        "adapted_record_count": len(records),
        "record_ids_unique": len(seen_record_ids) == len(records),
        "source_pages_unique": len(seen_page_ids) == len(pages),
        "speaker_attribution": "Unknown / unverified; source records are page-level",
        "records_by_document": dict(sorted(record_counts.items())),
        "source_page_fields_preserved": sorted(
            {key for page in pages for key in page}
        ),
        "errors": [],
    }
    if len(records) != len(pages):
        raise ValueError("Not every retained English page was adapted.")
    return records, report


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as output_file:
            temporary_path = Path(output_file.name)
            json.dump(payload, output_file, ensure_ascii=False, indent=2)
            output_file.write("\n")
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def adapt_file(input_path: Path, output_path: Path) -> Dict[str, Any]:
    input_path = input_path.resolve()
    output_path = output_path.resolve()
    if input_path == output_path:
        raise ValueError("Adapter output must not overwrite its input.")

    original_project = input_path.parents[2]
    protected_paths = {
        input_path,
        original_project / "data" / "processed" / "combined_processed.json",
        original_project / "data" / "processed" / "march_27_cleaned.json",
        original_project / "data" / "processed" / "july_20_cleaned.json",
    }
    if output_path in protected_paths:
        raise ValueError("Adapter output must not overwrite a source dataset.")

    input_hash = _sha256_file(input_path)
    with input_path.open("r", encoding="utf-8") as input_file:
        dataset = json.load(input_file)
    records, report = adapt_dataset(dataset)
    _write_json_atomic(output_path, records)

    with output_path.open("r", encoding="utf-8") as output_file:
        saved_records = json.load(output_file)
    if saved_records != records:
        raise ValueError("Saved adapter records differ from validated records.")
    if _sha256_file(input_path) != input_hash:
        raise ValueError("Source English-only dataset changed during adaptation.")

    report["input_unchanged"] = True
    report["output_path"] = str(output_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Adapt English-only page records without adding attribution."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=BASE_DIR / "transcript_records.json",
    )
    args = parser.parse_args()

    try:
        report = adapt_file(args.input, args.output)
    except (OSError, ValueError) as exc:
        print("Adapter validation: FAILED")
        print(f"Errors: {exc}")
        raise SystemExit(2) from exc

    print("Adapter validation: PASSED")
    print(f"Source pages considered: {report['source_page_count']}")
    print(f"Transcript records created: {report['adapted_record_count']}")
    print(f"Unique record IDs: {report['record_ids_unique']}")
    print(f"Page counts by document: {report['records_by_document']}")
    print(f"Speaker attribution: {report['speaker_attribution']}")
    print("English-only input unchanged: yes")
    print("Errors: none")
    print(f"Output: {report['output_path']}")


if __name__ == "__main__":
    main()
