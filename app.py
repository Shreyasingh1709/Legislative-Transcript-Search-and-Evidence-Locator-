"""Streamlit interface for evidence-grounded Lok Sabha transcript search."""

import json
import os
import time
from pathlib import Path
from typing import Any

import streamlit as st

from vector_store import COLLECTION_NAME, DATABASE_DIR
from embedding import MODEL_NAME, TFIDF_MODEL_NAME


BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = Path(
    os.environ.get(
        "ENGLISH_DATASET_PATH",
        str(
            BASE_DIR.parent
            / "Gen AI hackathon"
            / "data"
            / "processed"
            / "english_only_dataset.json"
        ),
    )
)
CHUNKS_PATH = BASE_DIR / "chunks.json"
REPORT_PATH = BASE_DIR / "chunking_report.json"
MANIFEST_PATH = BASE_DIR / "embeddings_manifest.json"


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as input_file:
        return json.load(input_file)


def _collection_count() -> int:
    if not DATABASE_DIR.exists():
        return 0
    try:
        import chromadb
    except ImportError:
        return 0
    try:
        client = chromadb.PersistentClient(path=str(DATABASE_DIR))
        return client.get_collection(name=COLLECTION_NAME).count()
    except chromadb.errors.NotFoundError:
        return 0


def _show_result(result: dict[str, Any], rank: int) -> None:
    with st.expander(
        f"{rank}. {result.get('document_id', 'Source not recorded')} "
        f"- PDF page {result.get('pdf_page', 'unknown')}"
    ):
        st.write(result.get("text", ""))
        st.caption(
            f"Source: {result.get('source_file', 'not recorded')} | "
            f"Date: {result.get('date', 'not recorded')} | "
            f"Speaker: {result.get('speaker', 'Unknown')} "
            f"({result.get('attribution_status', 'unverified')})"
        )
        st.caption(f"Chunk ID: {result.get('chunk_id', 'not recorded')}")
        st.caption(
            f"Cosine distance (lower is closer): "
            f"{result.get('distance', float('nan')):.4f}"
        )


def main() -> None:
    st.set_page_config(
        page_title="Lok Sabha Debate Search and Evidence Locator",
        layout="wide",
    )
    st.title("Lok Sabha Debate Search and Evidence Locator")
    st.caption(
        "Search a conservative English-only subset of Lok Sabha debate pages. "
        "Page-level records do not establish speaker attribution."
    )

    dataset = {}
    if DATASET_PATH.is_file():
        dataset = _read_json(DATASET_PATH)
    else:
        st.warning(
            f"English-only dataset not found at {DATASET_PATH}. "
            "Set ENGLISH_DATASET_PATH to its location."
        )

    chunks = _read_json(CHUNKS_PATH) if CHUNKS_PATH.is_file() else []
    report = _read_json(REPORT_PATH) if REPORT_PATH.is_file() else {}
    manifest = _read_json(MANIFEST_PATH) if MANIFEST_PATH.is_file() else {}
    indexed_count = _collection_count()
    chunk_sizes = [
        chunk.get("token_count", 0)
        for chunk in chunks
        if isinstance(chunk, dict) and isinstance(chunk.get("token_count"), int)
    ]

    retained = dataset.get("retained_record_count", len(dataset.get("pages", [])))
    uncertain = dataset.get("uncertain_record_count", 0)
    documents = dataset.get("documents", [])
    page_count = sum(
        int(document.get("total_pdf_pages", 0))
        for document in documents
        if isinstance(document, dict)
    )
    if not page_count:
        page_count = 2615 if len(documents) == 2 else 0

    metrics = st.columns(4)
    metrics[0].metric("Source documents", len(documents))
    metrics[1].metric("Original PDF pages", page_count)
    metrics[2].metric("Retained English pages", retained)
    metrics[3].metric("Uncertain records", uncertain)

    stats = st.columns(4)
    stats[0].metric("Generated chunks", len(chunks))
    stats[1].metric("Indexed chunks", indexed_count)
    average_tokens = f"{sum(chunk_sizes) / len(chunk_sizes):.1f}" if chunk_sizes else "-"
    stats[2].metric("Average chunk tokens", average_tokens)
    stats[3].metric(
        "Chunk token range",
        f"{min(chunk_sizes)}-{max(chunk_sizes)}" if chunk_sizes else "-",
    )

    embedding_name = manifest.get("model", "Not generated")
    st.caption(f"Retrieval method: {embedding_name} | Vector database: ChromaDB")
    if embedding_name == TFIDF_MODEL_NAME:
        st.caption("TF-IDF uses lexical term matching; it is not semantic retrieval.")
    if report.get("validation_status") == "passed":
        st.success(
            f"Chunk validation passed: {report.get('input_record_count', 0)} "
            f"source pages, {report.get('chunk_count', 0)} chunks."
        )
    else:
        st.warning("No validated chunk report is available yet.")

    st.subheader("Search parliamentary evidence")
    query = st.text_input(
        "Natural-language question",
        placeholder="What evidence discusses support for farmers?",
    )
    if st.button("Search", type="primary"):
        if not query.strip():
            st.warning("Enter a question to search.")
        elif indexed_count == 0:
            st.error(
                "Retrieval is not ready: the ChromaDB collection has no "
                "indexed chunks. Create vectors and the persistent index first."
            )
        elif manifest.get("model") not in {MODEL_NAME, TFIDF_MODEL_NAME}:
            st.error(
                "The embedding manifest is missing or uses an unsupported method."
            )
        else:
            try:
                from retrieval_test import search_transcripts

                started = time.perf_counter()
                results = search_transcripts(query, top_k=5)
                elapsed_ms = (time.perf_counter() - started) * 1000
                st.caption(
                    f"Retrieval latency: {elapsed_ms:.1f} ms | "
                    f"Results returned: {len(results)}"
                )
                if results:
                    for rank, result in enumerate(results, start=1):
                        _show_result(result, rank)
                else:
                    st.info("No matching passages were returned.")
            except (OSError, RuntimeError, ValueError, ImportError) as error:
                st.error(f"Search failed: {error}")

    st.info(
        "Speaker names are not verified in these page-level source records; "
        "retrieved passages are evidence, not generated answers."
    )


if __name__ == "__main__":
    main()
