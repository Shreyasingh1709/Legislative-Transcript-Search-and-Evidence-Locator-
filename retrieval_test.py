"""Search legislative transcript chunks using BGE-M3 and ChromaDB."""

from pathlib import Path
from typing import Any

import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer

from embedding import MODEL_NAME
from vector_store import COLLECTION_NAME, DATABASE_DIR


def search_transcripts(
    query: str,
    top_k: int = 5,
    model: Any | None = None,
    database_dir: Path = DATABASE_DIR,
    collection_name: str = COLLECTION_NAME,
) -> list[dict[str, Any]]:
    """Return semantically relevant chunks together with their attribution."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Search query must not be empty.")
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero.")

    client = chromadb.PersistentClient(path=str(database_dir))
    try:
        collection = client.get_collection(name=collection_name)
    except chromadb.errors.NotFoundError as error:
        raise RuntimeError(
            f"ChromaDB collection '{collection_name}' was not found. "
            "Run chunking.py, embedding.py, and vector_store.py first."
        ) from error

    result_count = collection.count()
    if result_count == 0:
        return []

    if model is None:
        model = SentenceTransformer(MODEL_NAME, trust_remote_code=True)
    query_vector = model.encode(
        [query],
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    query_vector = np.asarray(query_vector, dtype=np.float32)
    if query_vector.ndim != 2 or query_vector.shape[0] != 1:
        raise RuntimeError("Embedding model returned an invalid query vector.")

    result = collection.query(
        query_embeddings=query_vector.tolist(),
        n_results=min(top_k, result_count),
        include=["documents", "metadatas", "distances"],
    )
    hits = []
    for document, metadata, distance in zip(
        result["documents"][0],
        result["metadatas"][0],
        result["distances"][0],
        strict=True,
    ):
        hits.append(
            {
                "text": document,
                "speaker": metadata.get("speaker", "Unknown"),
                "date": metadata.get("document_date", metadata.get("date", "")),
                "document_id": metadata.get("document_id", ""),
                "pdf_page": metadata.get("pdf_page"),
                "printed_page": metadata.get("printed_page"),
                "record_id": metadata.get("record_id", ""),
                "distance": distance,
            }
        )
    return hits


def main() -> None:
    query = "Who discussed the difficulties faced by farmers?"
    results = search_transcripts(query, top_k=3)
    print(f"Query: {query}\n")
    if not results:
        print("No matching transcript statements found.")
        return

    for rank, result in enumerate(results, start=1):
        page = result["printed_page"] or result["pdf_page"] or "not recorded"
        print(
            f"{rank}. {result['speaker']} | {result['date']} | "
            f"{result['document_id']} | page {page} | "
            f"cosine distance {result['distance']:.4f}"
        )
        print(result["text"])
        print()


if __name__ == "__main__":
    main()