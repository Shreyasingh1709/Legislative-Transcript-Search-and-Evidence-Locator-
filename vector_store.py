"""Persist transcript chunks and vectors in ChromaDB."""

import json
from pathlib import Path
from typing import Any

import numpy as np

from embedding import MODEL_NAME, TFIDF_MODEL_NAME


BASE_DIR = Path(__file__).resolve().parent
DATABASE_DIR = BASE_DIR / "lok_sabha_vector_db"
COLLECTION_NAME = "lok_sabha_debates"
MODEL_NAME = "BAAI/bge-m3"
UPSERT_BATCH_SIZE = 128


def load_index_inputs(
    chunks_path: Path,
    vectors_path: Path,
    manifest_path: Path,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    with chunks_path.open("r", encoding="utf-8") as input_file:
        chunks = json.load(input_file)
    with manifest_path.open("r", encoding="utf-8") as input_file:
        manifest = json.load(input_file)
    vectors = np.load(vectors_path, mmap_mode="r")

    chunk_ids = [chunk.get("chunk_id") for chunk in chunks]
    if not chunks or any(not chunk_id for chunk_id in chunk_ids):
        raise ValueError("chunks.json must contain chunks with chunk_id values.")
    if chunk_ids != manifest.get("chunk_ids"):
        raise ValueError(
            "Embedding manifest does not match chunks.json. "
            "Run chunking.py and embedding.py again."
        )
    if manifest.get("model") not in {MODEL_NAME, TFIDF_MODEL_NAME}:
        raise ValueError("Unsupported embedding model in manifest.")
    if manifest.get("model") == TFIDF_MODEL_NAME:
        vectorizer_file = manifest.get("vectorizer_file")
        if not isinstance(vectorizer_file, str) or not vectorizer_file:
            raise ValueError("TF-IDF manifest is missing its vectorizer file.")
    if vectors.ndim != 2 or vectors.shape[0] != len(chunks):
        raise ValueError("Embedding matrix rows do not match the chunk count.")
    if vectors.shape[1] != manifest.get("dimensions"):
        raise ValueError("Embedding dimensions do not match the manifest.")
    if len(set(chunk_ids)) != len(chunk_ids):
        raise ValueError("chunks.json contains duplicate chunk_id values.")
    for chunk in chunks:
        required_fields = (
            "text",
            "speaker",
            "attribution_status",
            "document_id",
            "source_file",
            "date",
            "pdf_page",
        )
        if any(chunk.get(field) in (None, "") for field in required_fields):
            raise ValueError(f"Chunk {chunk.get('chunk_id')} is missing metadata.")
    return chunks, vectors


def _chromadb_metadata(chunk: dict[str, Any]) -> dict[str, str | int | float | bool]:
    metadata = {}
    for key, value in chunk.items():
        if key == "text" or value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            metadata[key] = value
        elif isinstance(value, (dict, list)):
            metadata[key] = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return metadata


def store_chunks(
    chunks: list[dict[str, Any]],
    vectors: np.ndarray,
    database_dir: Path = DATABASE_DIR,
    collection_name: str = COLLECTION_NAME,
    batch_size: int = UPSERT_BATCH_SIZE,
) -> int:
    """Upsert chunks and explicit embeddings into the persistent collection."""
    try:
        import chromadb
    except ImportError as error:
        raise RuntimeError(
            "ChromaDB is required to build the persistent vector store. "
            "Install project requirements first."
        ) from error

    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero.")
    if len(chunks) != vectors.shape[0]:
        raise ValueError("Number of chunks and embedding vectors must match.")

    client = chromadb.PersistentClient(path=str(database_dir))
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )
    incoming_ids = {chunk["chunk_id"] for chunk in chunks}
    existing_ids = set(collection.get(include=[])["ids"])
    stale_ids = existing_ids.difference(incoming_ids)
    if stale_ids:
        raise ValueError(
            f"ChromaDB collection contains {len(stale_ids)} IDs absent from "
            "the validated chunks. Refusing to mix stale or sample records."
        )

    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        batch_vectors = np.asarray(vectors[start : start + batch_size]).tolist()
        collection.upsert(
            ids=[chunk["chunk_id"] for chunk in batch],
            documents=[chunk["text"] for chunk in batch],
            metadatas=[_chromadb_metadata(chunk) for chunk in batch],
            embeddings=batch_vectors,
        )
    stored_count = collection.count()
    if stored_count != len(chunks):
        raise RuntimeError(
            f"ChromaDB contains {stored_count} records after upsert; "
            f"expected {len(chunks)}."
        )
    return stored_count


def main() -> None:
    chunks, vectors = load_index_inputs(
        BASE_DIR / "chunks.json",
        BASE_DIR / "embeddings.npy",
        BASE_DIR / "embeddings_manifest.json",
    )
    count = store_chunks(chunks, vectors)
    print(
        f"Stored {len(chunks)} chunks in {DATABASE_DIR} "
        f"(collection now contains {count})."
    )


if __name__ == "__main__":
    main()
