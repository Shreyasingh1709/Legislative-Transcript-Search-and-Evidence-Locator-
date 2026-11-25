import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ============================================================
# PS 160 - Create Embeddings and FAISS Vector Database
# ============================================================

INPUT_FILE = "chunks.json"
INDEX_FILE = "transcript.index"
METADATA_FILE = "metadata.json"

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


def create_vector_database():

    # Load prepared transcript chunks
    with open(INPUT_FILE, "r", encoding="utf-8") as file:
        chunks = json.load(file)

    print("Total chunks:", len(chunks))

    # Extract statements
    statements = [
        chunk["statement"]
        for chunk in chunks
    ]

    # Create embeddings
    print("Creating embeddings...")

    embeddings = model.encode(
        statements,
        show_progress_bar=True
    )

    # Convert to NumPy format
    embeddings = np.array(
        embeddings,
        dtype="float32"
    )

    # Create FAISS index
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    # Add embeddings to FAISS
    index.add(embeddings)

    # Save FAISS index
    faiss.write_index(
        index,
        INDEX_FILE
    )

    # Save metadata separately
    with open(
        METADATA_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            chunks,
            file,
            ensure_ascii=False,
            indent=2
        )

    print()
    print("========================================")
    print("Vector database created successfully")
    print("========================================")
    print("Embeddings:", len(embeddings))
    print("Vector dimension:", dimension)
    print("FAISS index:", INDEX_FILE)
    print("Metadata:", METADATA_FILE)


if __name__ == "__main__":
    create_vector_database()