"""Build a conservative English retrieval set from the combined dataset."""

import argparse
import hashlib
import json
import os
import re
import tempfile
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple


_CID_PLACEHOLDER = re.compile(r"\(cid:\d+\)", re.IGNORECASE)
_WORD = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)*", re.UNICODE)
_ENGLISH_FUNCTION_WORDS = {
    "a",
    "about",
    "after",
    "all",
    "also",
    "an",
    "and",
    "any",
    "are",
    "as",
    "at",
    "be",
    "been",
    "before",
    "between",
    "by",
    "can",
    "could",
    "did",
    "do",
    "for",
    "from",
    "had",
    "has",
    "have",
    "he",
    "her",
    "him",
    "his",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "may",
    "might",
    "more",
    "most",
    "must",
    "no",
    "not",
    "of",
    "on",
    "or",
    "our",
    "she",
    "should",
    "such",
    "than",
    "that",
    "the",
    "their",
    "them",
    "there",
    "these",
    "they",
    "this",
    "those",
    "to",
    "under",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "will",
    "with",
    "would",
    "you",
    "your",
}
_MIN_ENGLISH_FUNCTION_WORDS = 3
_MIN_ENGLISH_WORDS = 10
_PROJECT_ROOT = Path(__file__).resolve().parent
_PROTECTED_SOURCE_PATHS = (
    _PROJECT_ROOT / "data" / "processed" / "march_27_cleaned.json",
    _PROJECT_ROOT / "data" / "processed" / "july_20_cleaned.json",
)
_REQUIRED_PAGE_FIELDS = {
    "document_id",
    "source_file",
    "date",
    "pdf_page",
    "original_text",
    "cleaned_text",
    "quality_warnings",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _script_flags(text: str) -> Tuple[bool, bool, bool]:
    has_latin = False
    has_devanagari = False
    has_other_non_latin = False
    for character in text:
        if not character.isalpha():
            continue
        character_name = unicodedata.name(character, "")
        if character_name.startswith("LATIN "):
            has_latin = True
        elif character_name.startswith("DEVANAGARI "):
            has_devanagari = True
        else:
            has_other_non_latin = True
    return has_latin, has_devanagari, has_other_non_latin


def classify_page(page: Dict[str, Any]) -> Tuple[str, str]:
    """Classify a page conservatively without changing its text."""
    text = page.get("cleaned_text")
    if not isinstance(text, str) or not text.strip():
        return "excluded", "empty_text"

    warnings = page.get("quality_warnings")
    if not isinstance(warnings, list):
        raise ValueError(
            f"Page {page.get('pdf_page')} quality_warnings must be an array."
        )
    if (
        warnings
        or _CID_PLACEHOLDER.search(text)
        or "\ufffd" in text
        or any(unicodedata.category(character) == "Co" for character in text)
    ):
        return "uncertain", "unresolved_quality_warning_or_glyph"

    has_latin, has_devanagari, has_other_non_latin = _script_flags(text)
    if has_latin and (has_devanagari or has_other_non_latin):
        return "uncertain", "mixed_scripts"
    if has_devanagari:
        return "excluded", "hindi_only"
    if has_other_non_latin:
        return "excluded", "non_latin_text_without_english"
    if not has_latin:
        return "excluded", "no_english_letters"

    words = [word.casefold() for word in _WORD.findall(text)]
    english_evidence = {
        word for word in words if word in _ENGLISH_FUNCTION_WORDS
    }
    if (
        len(words) < _MIN_ENGLISH_WORDS
        or len(english_evidence) < _MIN_ENGLISH_FUNCTION_WORDS
    ):
        return "uncertain", "insufficient_english_language_evidence"
    return "retained", "latin_script_with_english_function_words"


def _validate_combined_input(dataset: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not isinstance(dataset, dict):
        raise ValueError("Combined input JSON root must be an object.")
    documents = dataset.get("documents")
    pages = dataset.get("pages")
    if not isinstance(documents, list) or not isinstance(pages, list):
        raise ValueError("Combined input must contain documents and pages arrays.")
    if dataset.get("document_count") != len(documents):
        raise ValueError("Combined input document_count does not match documents.")
    if dataset.get("total_pdf_pages") != len(pages):
        raise ValueError("Combined input total_pdf_pages does not match pages.")

    documents_by_id = {}
    for document in documents:
        if not isinstance(document, dict):
            raise ValueError("Each document metadata record must be an object.")
        document_id = document.get("document_id")
        if not isinstance(document_id, str) or not document_id:
            raise ValueError("Each source document must have a document_id.")
        if document_id in documents_by_id:
            raise ValueError(f"Duplicate document_id in input: {document_id}")
        documents_by_id[document_id] = document

    pages_by_document = defaultdict(list)
    seen_page_keys = set()
    for index, page in enumerate(pages, start=1):
        if not isinstance(page, dict):
            raise ValueError(f"Page record {index} must be an object.")
        missing_fields = _REQUIRED_PAGE_FIELDS.difference(page)
        if missing_fields:
            raise ValueError(
                f"Page record {index} is missing required fields "
                f"{sorted(missing_fields)}."
            )
        document_id = page["document_id"]
        if document_id not in documents_by_id:
            raise ValueError(
                f"Page record {index} references unknown document_id {document_id!r}."
            )
        if (
            page["source_file"] != documents_by_id[document_id].get("source_file")
            or page["date"] != documents_by_id[document_id].get("date")
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
        page_key = (document_id, page_number)
        if page_key in seen_page_keys:
            raise ValueError(
                f"Duplicate page record for document {document_id}, page {page_number}."
            )
        seen_page_keys.add(page_key)
        if not isinstance(page["quality_warnings"], list):
            raise ValueError(
                f"Page {page_number} quality_warnings must be an array."
            )
        pages_by_document[document_id].append(page)

    for document_id, document in documents_by_id.items():
        document_pages = pages_by_document[document_id]
        expected_page_count = document.get("total_pdf_pages")
        if not isinstance(expected_page_count, int) or isinstance(
            expected_page_count, bool
        ):
            raise ValueError(
                f"Document {document_id} has an invalid total_pdf_pages value."
            )
        actual_numbers = [page["pdf_page"] for page in document_pages]
        expected_numbers = list(range(1, expected_page_count + 1))
        if len(document_pages) != expected_page_count or actual_numbers != expected_numbers:
            raise ValueError(
                f"Document {document_id} must have sequential pages "
                f"1-{expected_page_count}."
            )
    return pages


def build_english_dataset(dataset: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Partition combined page records into retained, excluded, and uncertain."""
    source_pages = _validate_combined_input(dataset)
    retained_pages: List[Dict[str, Any]] = []
    uncertain_pages: List[Dict[str, Any]] = []
    excluded_counts = Counter()
    uncertain_counts = Counter()
    counts_by_document = defaultdict(Counter)

    for source_page in source_pages:
        status, reason = classify_page(source_page)
        document_id = source_page["document_id"]
        counts_by_document[document_id][status] += 1
        if status == "retained":
            record = dict(source_page)
            record["retrieval_text"] = source_page["cleaned_text"]
            record["language_status"] = "english_only"
            retained_pages.append(record)
        elif status == "uncertain":
            record = dict(source_page)
            record["language_status"] = "uncertain"
            record["language_reason"] = reason
            uncertain_pages.append(record)
            uncertain_counts[reason] += 1
        else:
            excluded_counts[reason] += 1

    retained_count = len(retained_pages)
    excluded_count = sum(excluded_counts.values())
    uncertain_count = len(uncertain_pages)
    if retained_count + excluded_count + uncertain_count != len(source_pages):
        raise ValueError("Language classification counts do not cover every source page.")

    output_dataset = {
        "dataset_name": "PS 160 English-only Retrieval Dataset",
        "source_dataset": dataset.get("dataset_name"),
        "document_count": dataset["document_count"],
        "documents": dataset["documents"],
        "source_record_count": len(source_pages),
        "retained_record_count": retained_count,
        "excluded_record_count": excluded_count,
        "uncertain_record_count": uncertain_count,
        "language_policy": {
            "retained": (
                "Latin-script text with at least ten words, at least three "
                "distinct common English function words, and no quality "
                "warnings or unresolved glyphs."
            ),
            "excluded": (
                "Hindi-only text, other non-Latin-only text, empty text, or text "
                "without English letters."
            ),
            "uncertain": (
                "Mixed scripts, unresolved quality warnings/glyphs, or Latin "
                "text without sufficient English-language evidence. These records "
                "are preserved separately and are not retrieval pages."
            ),
        },
        "classification_summary": {
            "retained": retained_count,
            "excluded": excluded_count,
            "uncertain": uncertain_count,
            "excluded_by_reason": dict(sorted(excluded_counts.items())),
            "uncertain_by_reason": dict(sorted(uncertain_counts.items())),
            "by_document": {
                document_id: {
                    "retained": counts["retained"],
                    "excluded": counts["excluded"],
                    "uncertain": counts["uncertain"],
                }
                for document_id, counts in sorted(counts_by_document.items())
            },
        },
        "pages": retained_pages,
        "uncertain_records": uncertain_pages,
    }

    source_pages_by_key = {
        (page["document_id"], page["pdf_page"]): page for page in source_pages
    }
    partitioned_pages = retained_pages + uncertain_pages
    partitioned_keys = [
        (page["document_id"], page["pdf_page"]) for page in partitioned_pages
    ]
    if len(set(partitioned_keys)) != len(partitioned_keys):
        raise ValueError("A source page appears more than once in the output.")
    for output_page in partitioned_pages:
        source_page = source_pages_by_key[
            (output_page["document_id"], output_page["pdf_page"])
        ]
        if any(
            key not in output_page or output_page[key] != value
            for key, value in source_page.items()
        ):
            raise ValueError(
                f"Source page fields changed for {output_page['document_id']}:"
                f"{output_page['pdf_page']}."
            )

    for page in retained_pages:
        if (
            page["retrieval_text"] != page["cleaned_text"]
            or page["quality_warnings"]
            or _script_flags(page["retrieval_text"])[1:]
                != (False, False)
        ):
            raise ValueError(
                f"Retained page {page['document_id']}:{page['pdf_page']} "
                "failed English-only validation."
            )
    return output_dataset, output_dataset["classification_summary"]


def generate_english_dataset(input_path: Path, output_path: Path) -> Dict[str, Any]:
    """Create and validate the English-only retrieval dataset without altering input."""
    input_path = input_path.resolve()
    output_path = output_path.resolve()
    protected_paths = {input_path}
    protected_paths.update(path.resolve() for path in _PROTECTED_SOURCE_PATHS)
    if output_path in protected_paths:
        raise ValueError("The output path must not overwrite any source JSON file.")

    input_hash = _sha256_file(input_path)
    with input_path.open("r", encoding="utf-8") as input_file:
        source_dataset = json.load(input_file)
    output_dataset, summary = build_english_dataset(source_dataset)

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
            json.dump(output_dataset, output_file, ensure_ascii=False, indent=2)
            output_file.write("\n")
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()

    with output_path.open("r", encoding="utf-8") as output_file:
        saved_dataset = json.load(output_file)
    if saved_dataset != output_dataset:
        raise ValueError("Saved English-only JSON differs from validated output.")
    if _sha256_file(input_path) != input_hash:
        raise ValueError("Combined input changed during English dataset generation.")
    if (
        saved_dataset["retained_record_count"]
        + saved_dataset["excluded_record_count"]
        + saved_dataset["uncertain_record_count"]
        != saved_dataset["source_record_count"]
    ):
        raise ValueError("Persisted English-only classification counts do not balance.")

    summary = dict(summary)
    summary["source_record_count"] = output_dataset["source_record_count"]
    summary["input_unchanged"] = True
    summary["output_path"] = str(output_path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build an English-only retrieval dataset from combined pages."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(r"data\processed\combined_processed.json"),
        help="Combined processed JSON input",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(r"data\processed\english_only_dataset.json"),
        help="English-only JSON output",
    )
    args = parser.parse_args()

    try:
        summary = generate_english_dataset(args.input, args.output)
    except (OSError, ValueError) as exc:
        print("Validation: FAILED")
        print(f"Errors: {exc}")
        raise SystemExit(2) from exc

    print("Validation: PASSED")
    print(f"Source records: {summary['source_record_count']}")
    print(f"Retained records: {summary['retained']}")
    print(f"Excluded records: {summary['excluded']}")
    print(f"Uncertain records: {summary['uncertain']}")
    print(
        "Classification totals balance: "
        f"{summary['retained'] + summary['excluded'] + summary['uncertain']} "
        f"of {summary['source_record_count']}"
    )
    print(f"Excluded by reason: {summary['excluded_by_reason']}")
    print(f"Uncertain by reason: {summary['uncertain_by_reason']}")
    print(f"Counts by document: {summary['by_document']}")
    print("Combined input unchanged: yes")
    print("Errors: none")
    print(f"Output: {summary['output_path']}")


if __name__ == "__main__":
    main()
