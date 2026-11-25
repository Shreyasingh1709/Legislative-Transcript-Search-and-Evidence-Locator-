"""Generate TF-IDF vectors or optional BGE-M3 vectors for chunks.json."""

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


MODEL_NAME = "BAAI/bge-m3"
TFIDF_MODEL_NAME = "TF-IDF (scikit-learn)"
BASE_DIR = Path(__file__).resolve().parent
BATCH_SIZE = 8
TFIDF_MAX_FEATURES = 4096


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
        from sentence_transformers import SentenceTransformer

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


def create_embeddings(
    chunks: list[dict[str, Any]],
    method: str = "tfidf",
) -> tuple[np.ndarray, str, Path | None]:
    """Create BGE-M3 or explicitly selected TF-IDF vectors."""
    if method not in {"bge-m3", "tfidf"}:
        raise ValueError(f"Unsupported embedding method: {method}")
    if method == "bge-m3":
        return embed_chunks(chunks), MODEL_NAME, None

    vectorizer = TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        ngram_range=(1, 2),
        sublinear_tf=True,
        stop_words="english",
        dtype=np.float32,
    )
    vectors = vectorizer.fit_transform(
        [chunk["text"] for chunk in chunks]
    ).toarray()
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[0] != len(chunks):
        raise RuntimeError("TF-IDF vectorizer returned an unexpected vector shape.")
    if not np.isfinite(vectors).all():
        raise RuntimeError("TF-IDF vectorizer returned non-finite values.")
    vectorizer_path = BASE_DIR / "tfidf_vectorizer.joblib"
    joblib.dump(vectorizer, vectorizer_path)
    return vectors, TFIDF_MODEL_NAME, vectorizer_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed validated transcript chunks.")
    parser.add_argument(
        "--model",
        choices=("bge-m3", "tfidf"),
        default="tfidf",
        help="TF-IDF lexical retrieval is the default; BGE-M3 is optional",
    )
    args = parser.parse_args()

    chunks_path = BASE_DIR / "chunks.json"
    chunks = load_chunks(chunks_path)
    vectors, model_name, vectorizer_path = create_embeddings(chunks, args.model)

    embeddings_path = BASE_DIR / "embeddings.npy"
    np.save(embeddings_path, vectors)
    manifest_path = BASE_DIR / "embeddings_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as output_file:
        json.dump(
            {
                "model": model_name,
                "method": "tfidf" if vectorizer_path else "dense",
                "chunk_ids": [chunk["chunk_id"] for chunk in chunks],
                "dimensions": int(vectors.shape[1]),
                "vectorizer_file": (
                    vectorizer_path.name if vectorizer_path else None
                ),
            },
            output_file,
            ensure_ascii=False,
            indent=2,
        )

    print(
        f"Saved {vectors.shape[0]} {model_name} vectors "
        f"({vectors.shape[1]} dimensions) to {embeddings_path}"
    )


if __name__ == "__main__":
    main()
