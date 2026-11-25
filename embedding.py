"""Generate dense BGE-M3 vectors for chunks.json."""

import json
from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer


MODEL_NAME = "BAAI/bge-m3"
BASE_DIR = Path(__file__).resolve().parent
BATCH_SIZE = 8


def load_chunks(chunks_path: Path) -> list[dict[str, Any]]:
    with chunks_path.open("r", encoding="utf-8") as input_file:
        chunks = json.load(input_file)
    if not isinstance(chunks, list) or not chunks:
        raise ValueError("chunks.json must contain a non-empty JSON array.")
    for index, chunk in enumerate(chunks):
        if not isinstance(chunk, dict) or not chunk.get("chunk_id"):
            raise ValueError(f"Chunk at index {index} is missing chunk_id.")
        if not isinstance(chunk.get("text"), str) or not chunk["text"].strip():
            raise ValueError(f"Chunk at index {index} is missing text.")
        if not chunk.get("speaker"):
            raise ValueError(f"Chunk {chunk['chunk_id']} is missing speaker.")
    return chunks


def embed_chunks(
    chunks: list[dict[str, Any]],
    model: Any | None = None,
    batch_size: int = BATCH_SIZE,
) -> np.ndarray:
    """Encode chunk text into normalized BGE-M3 dense vectors."""
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero.")
    if not chunks:
        raise ValueError("No chunks supplied for embedding.")

    if model is None:
        model = SentenceTransformer(MODEL_NAME, trust_remote_code=True)

    vectors = model.encode(
        [chunk["text"] for chunk in chunks],
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[0] != len(chunks):
        raise RuntimeError("Embedding model returned an unexpected vector shape.")
    if not np.isfinite(vectors).all():
        raise RuntimeError("Embedding model returned non-finite vector values.")
    return vectors


def main() -> None:
    chunks_path = BASE_DIR / "chunks.json"
    chunks = load_chunks(chunks_path)
    vectors = embed_chunks(chunks)

    embeddings_path = BASE_DIR / "embeddings.npy"
    np.save(embeddings_path, vectors)
    manifest_path = BASE_DIR / "embeddings_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as output_file:
        json.dump(
            {
                "model": MODEL_NAME,
                "chunk_ids": [chunk["chunk_id"] for chunk in chunks],
                "dimensions": int(vectors.shape[1]),
            },
            output_file,
            ensure_ascii=False,
            indent=2,
        )

    print(
        f"Saved {vectors.shape[0]} BGE-M3 embeddings "
        f"({vectors.shape[1]} dimensions) to {embeddings_path}"
    )


if __name__ == "__main__":
    main()
