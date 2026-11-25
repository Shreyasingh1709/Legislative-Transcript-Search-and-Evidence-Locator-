"""Create provenance-preserving transcript chunks using BGE-M3 token counts."""

import argparse
import json
import os
import re
import tempfile
from bisect import bisect_right
from pathlib import Path
from typing import Any

import pandas as pd
from transformers import AutoTokenizer


MODEL_NAME = "BAAI/bge-m3"
TARGET_TOKENS = 400
MAX_TOKENS = 500
OVERLAP_TOKENS = 50
MIN_TOKENS = 80
BASE_DIR = Path(__file__).resolve().parent


def load_records(input_path: Path) -> list[dict[str, Any]]:
    """Load adapted records from a JSON array or an object containing records."""
    with input_path.open("r", encoding="utf-8") as input_file:
        payload = json.load(input_file)

    if isinstance(payload, dict):
        payload = payload.get("records")
    if not isinstance(payload, list):
        raise ValueError("Input must contain a JSON array of transcript records.")
    if not payload:
        raise ValueError("Input contains no transcript records.")
    if not all(isinstance(record, dict) for record in payload):
        raise ValueError("Every transcript record must be a JSON object.")

    return [dict(record) for record in payload]


def _token_windows(
    text: str,
    tokenizer: Any,
    target_tokens: int = TARGET_TOKENS,
    max_tokens: int = MAX_TOKENS,
    overlap_tokens: int = OVERLAP_TOKENS,
    min_tokens: int = MIN_TOKENS,
) -> list[tuple[int, int, list[tuple[int, int]]]]:
    """Return token index windows and offsets, preferring sentence boundaries."""
    if not 0 <= overlap_tokens < target_tokens <= max_tokens:
        raise ValueError(
            "Chunk settings must satisfy 0 <= overlap < target <= maximum."
        )
    if min_tokens <= 0 or min_tokens > max_tokens:
        raise ValueError("Minimum chunk size must be between 1 and maximum.")

    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    offsets = [tuple(offset) for offset in encoded["offset_mapping"]]
    token_count = len(offsets)
    if token_count == 0:
        return []
    if token_count <= max_tokens:
        return [(0, token_count, offsets)]

    token_ends = [end for _, end in offsets]
    sentence_ends = {
        bisect_right(token_ends, match.end())
        for match in re.finditer(
            r"[.!?\u0964]+(?:[\"'\u2019\u201d)]*)?(?=\s|$)", text
        )
    }
    paragraph_ends = {
        bisect_right(token_ends, match.end())
        for match in re.finditer(r"\n[ \t]*\n(?:[ \t]*\n)*", text)
    }
    sentence_ends.update(paragraph_ends)

    windows: list[tuple[int, int, list[tuple[int, int]]]] = []
    start = 0
    while start < token_count:
        remaining = token_count - start
        if remaining <= max_tokens:
            if (
                remaining < min_tokens
                and windows
                and token_count - windows[-1][0] <= max_tokens
            ):
                previous_start = windows[-1][0]
                windows[-1] = (previous_start, token_count, offsets)
            else:
                windows.append((start, token_count, offsets))
            break

        target_end = min(start + target_tokens, token_count)
        minimum_end = min(start + min_tokens, token_count)
        maximum_end = min(start + max_tokens, token_count)
        boundaries = [
            boundary
            for boundary in sentence_ends
            if minimum_end <= boundary <= maximum_end
        ]
        before_target = [boundary for boundary in boundaries if boundary <= target_end]
        if before_target:
            end = max(before_target)
        elif boundaries:
            end = min(boundaries)
        else:
            end = target_end

        if end <= start:
            raise RuntimeError("Tokenizer produced an invalid chunk boundary.")
        windows.append((start, end, offsets))
        start = end - overlap_tokens

    return windows


def chunk_text(
    text: str,
    tokenizer: Any,
    target_tokens: int = TARGET_TOKENS,
    max_tokens: int = MAX_TOKENS,
    overlap_tokens: int = OVERLAP_TOKENS,
    min_tokens: int = MIN_TOKENS,
) -> list[tuple[str, int]]:
    """Split one source record into token-limited text chunks and token counts."""
    if not isinstance(text, str):
        raise TypeError("Transcript text must be a string.")
    if not text.strip():
        return []

    windows = _token_windows(
        text,
        tokenizer,
        target_tokens=target_tokens,
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
        min_tokens=min_tokens,
    )
    chunks = []
    for start, end, offsets in windows:
        start_char = offsets[start][0]
        end_char = offsets[end - 1][1]
        chunk = text[start_char:end_char].strip()
        token_count = len(
            tokenizer.encode(chunk, add_special_tokens=False)
        )
        if chunk:
            chunks.append((chunk, token_count))
    return chunks


def chunk_records(
    records: list[dict[str, Any]],
    tokenizer: Any,
) -> list[dict[str, Any]]:
    """Chunk each source record independently and retain its metadata."""
    chunks: list[dict[str, Any]] = []
    seen_record_ids: set[str] = set()
    text_fields = {"text", "retrieval_text", "original_text", "cleaned_text"}

    for row_number, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            raise ValueError(f"Record {row_number} must be a JSON object.")
        text = record.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(
                f"Record {row_number} has missing or empty transcript text."
            )

        speaker = record.get("speaker")
        if not isinstance(speaker, str) or not speaker.strip():
            raise ValueError(
                f"Record {row_number} has transcript text but no speaker field."
            )

        record_id = str(record.get("record_id") or f"record_{row_number:06d}")
        if record_id in seen_record_ids:
            raise ValueError(f"Duplicate record_id: {record_id}")
        seen_record_ids.add(record_id)

        metadata = {
            key: value
            for key, value in record.items()
            if key not in text_fields
            and not (isinstance(value, float) and pd.isna(value))
        }
        metadata["record_id"] = record_id
        metadata["speaker"] = speaker.strip()

        for chunk_index, (chunk_text_value, token_count) in enumerate(
            chunk_text(text, tokenizer), start=0
        ):
            chunks.append(
                {
                    **metadata,
                    "chunk_id": f"{record_id}::chunk::{chunk_index:04d}",
                    "chunk_index": chunk_index,
                    "token_count": token_count,
                    "text": chunk_text_value,
                }
            )

    if not chunks:
        raise ValueError("No non-empty transcript statements were found.")
    return chunks


def validate_chunk_output(
    records: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
) -> dict[str, Any]:
    """Validate chunk limits, page provenance, IDs, and parent traceability."""
    errors = []
    records_by_id = {}
    for record in records:
        record_id = record.get("record_id")
        if not isinstance(record_id, str) or not record_id:
            errors.append("Source record has a missing or invalid record_id.")
        elif record_id in records_by_id:
            errors.append(f"Duplicate source record_id: {record_id}")
        else:
            records_by_id[record_id] = record
        for field in ("document_id", "source_file", "date", "speaker"):
            value = record.get(field)
            if not isinstance(value, str) or not value.strip():
                errors.append(f"Source record {record_id} is missing {field}.")
        page_number = record.get("pdf_page")
        if (
            not isinstance(page_number, int)
            or isinstance(page_number, bool)
            or page_number < 1
        ):
            errors.append(f"Source record {record_id} has invalid pdf_page.")
        if record.get("attribution_status") != "unverified":
            errors.append(
                f"Source record {record_id} must have unverified attribution."
            )

    chunk_ids = [chunk.get("chunk_id") for chunk in chunks]
    if any(not isinstance(chunk_id, str) or not chunk_id for chunk_id in chunk_ids):
        errors.append("One or more chunks have a missing or invalid chunk_id.")
    duplicate_chunk_ids = sorted(
        chunk_id
        for chunk_id in set(chunk_ids)
        if chunk_ids.count(chunk_id) > 1
    )
    chunks_by_record: dict[str, list[dict[str, Any]]] = {}
    for chunk in chunks:
        record_id = chunk.get("record_id")
        if not isinstance(record_id, str) or not record_id:
            errors.append(f"Chunk {chunk.get('chunk_id')} has an invalid record_id.")
            continue
        chunks_by_record.setdefault(record_id, []).append(chunk)
        source_record = records_by_id.get(record_id)
        if source_record is None:
            errors.append(f"Chunk {chunk.get('chunk_id')} has no source record.")
            continue
        if not isinstance(chunk.get("text"), str) or not chunk["text"].strip():
            errors.append(f"Chunk {chunk.get('chunk_id')} has empty text.")
        else:
            source_text = source_record.get("text")
            if not isinstance(source_text, str) or chunk["text"] not in source_text:
                errors.append(
                    f"Chunk {chunk.get('chunk_id')} text is not traceable to its source."
                )
        token_count = chunk.get("token_count")
        if not isinstance(token_count, int) or isinstance(token_count, bool) or token_count <= 0:
            errors.append(f"Chunk {chunk.get('chunk_id')} has invalid token_count.")
        elif token_count > MAX_TOKENS:
            errors.append(
                f"Chunk {chunk.get('chunk_id')} has {token_count} tokens "
                f"(maximum {MAX_TOKENS})."
            )
        for key, value in source_record.items():
            if key in {"text", "retrieval_text", "original_text", "cleaned_text"}:
                continue
            if chunk.get(key) != value:
                errors.append(
                    f"Chunk {chunk.get('chunk_id')} lost or changed metadata field {key}."
                )

    records_without_chunks = sorted(
        record_id for record_id in records_by_id if record_id not in chunks_by_record
    )
    if records_without_chunks:
        errors.append(
            f"{len(records_without_chunks)} source records have no chunks."
        )

    counts_by_document: dict[str, dict[str, int]] = {}
    for record in records:
        document_id = record.get("document_id", "unknown")
        counts_by_document.setdefault(document_id, {"records": 0, "chunks": 0})
        counts_by_document[document_id]["records"] += 1
    for chunk in chunks:
        document_id = chunk.get("document_id", "unknown")
        counts_by_document.setdefault(document_id, {"records": 0, "chunks": 0})
        counts_by_document[document_id]["chunks"] += 1

    token_counts = [
        chunk["token_count"]
        for chunk in chunks
        if isinstance(chunk.get("token_count"), int)
        and not isinstance(chunk.get("token_count"), bool)
    ]
    return {
        "input_record_count": len(records),
        "records_with_chunks": len(records_by_id) - len(records_without_chunks),
        "records_without_chunks": records_without_chunks,
        "chunk_count": len(chunks),
        "unique_record_ids": len(records_by_id) == len(records),
        "unique_chunk_ids": len(chunk_ids) == len(set(chunk_ids)),
        "duplicate_chunk_ids": duplicate_chunk_ids,
        "empty_chunk_count": sum(
            not isinstance(chunk.get("text"), str) or not chunk["text"].strip()
            for chunk in chunks
        ),
        "max_tokens_per_chunk": max(token_counts, default=0),
        "chunks_over_max_tokens": sum(
            isinstance(chunk.get("token_count"), int)
            and not isinstance(chunk.get("token_count"), bool)
            and chunk["token_count"] > MAX_TOKENS
            for chunk in chunks
        ),
        "counts_by_document": counts_by_document,
        "validation_errors": errors,
    }


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Chunk adapted transcript records.")
    parser.add_argument(
        "--input",
        type=Path,
        default=BASE_DIR / "transcript_records.json",
        help="Adapted transcript records JSON",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=BASE_DIR / "chunks.json",
        help="Chunk output JSON",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=BASE_DIR / "chunking_report.json",
        help="Chunk validation report JSON",
    )
    args = parser.parse_args()
    input_path = args.input.resolve()
    output_path = args.output.resolve()
    report_path = args.report.resolve()
    if input_path in {output_path, report_path} or output_path == report_path:
        parser.error("Input, chunk output, and report paths must be different.")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    records = load_records(input_path)
    chunks = chunk_records(records, tokenizer)
    report = validate_chunk_output(records, chunks)
    report["model_tokenizer"] = MODEL_NAME
    report["target_tokens"] = TARGET_TOKENS
    report["maximum_tokens"] = MAX_TOKENS
    report["overlap_tokens"] = OVERLAP_TOKENS
    report["minimum_tokens"] = MIN_TOKENS
    report["source_path"] = str(input_path)
    report["chunk_output_path"] = str(output_path)
    report["validation_status"] = (
        "passed" if not report["validation_errors"] else "failed"
    )
    _write_json_atomic(report_path, report)
    if report["validation_errors"]:
        print("Chunk validation: FAILED")
        for error in report["validation_errors"]:
            print(f"- {error}")
        raise SystemExit(2)

    _write_json_atomic(output_path, chunks)
    print("Chunk validation: PASSED")
    print(f"Source records considered: {report['input_record_count']}")
    print(f"Records with chunks: {report['records_with_chunks']}")
    print(f"Chunks created: {report['chunk_count']}")
    print(f"Maximum chunk tokens: {report['max_tokens_per_chunk']}")
    print(f"Chunks over {MAX_TOKENS} tokens: {report['chunks_over_max_tokens']}")
    print(f"Unique chunk IDs: {report['unique_chunk_ids']}")
    print(f"Empty chunks: {report['empty_chunk_count']}")
    print(f"Counts by document: {report['counts_by_document']}")
    print("Errors: none")
    print(f"Chunks: {output_path}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
