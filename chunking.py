"""Create speaker-attributed transcript chunks using BGE-M3 token counts."""

import json
import re
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
    """Load cleaned records from a JSON array or an object containing records."""
    with input_path.open("r", encoding="utf-8") as input_file:
        payload = json.load(input_file)

    if isinstance(payload, dict):
        payload = payload.get("records")
    if not isinstance(payload, list):
        raise ValueError("cleaned_data.json must contain a JSON array of records.")
    if not payload:
        raise ValueError("cleaned_data.json contains no transcript records.")
    if not all(isinstance(record, dict) for record in payload):
        raise ValueError("Every transcript record must be a JSON object.")

    # DataFrame normalization makes missing values consistent before validation.
    return pd.DataFrame.from_records(payload).to_dict(orient="records")


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
    """Split one speaker turn into token-limited text chunks and token counts."""
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
    """Chunk each speaker turn independently and retain its scalar metadata."""
    chunks: list[dict[str, Any]] = []
    seen_record_ids: set[str] = set()

    for row_number, record in enumerate(records, start=1):
        text = record.get("text")
        if not isinstance(text, str) or not text.strip():
            continue

        speaker = record.get("speaker")
        if not isinstance(speaker, str) or not speaker.strip():
            raise ValueError(
                f"Record {row_number} has transcript text but no speaker; "
                "refusing to create unattributed evidence."
            )

        record_id = str(record.get("record_id") or f"record_{row_number:06d}")
        if record_id in seen_record_ids:
            raise ValueError(f"Duplicate record_id: {record_id}")
        seen_record_ids.add(record_id)

        metadata = {
            key: value
            for key, value in record.items()
            if key != "text"
            and isinstance(value, (str, int, float, bool))
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


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    records = load_records(BASE_DIR / "cleaned_data.json")
    chunks = chunk_records(records, tokenizer)
    output_path = BASE_DIR / "chunks.json"
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(chunks, output_file, ensure_ascii=False, indent=2)
    print(f"Created {len(chunks)} attributed chunks at {output_path}")


if __name__ == "__main__":
    main()
