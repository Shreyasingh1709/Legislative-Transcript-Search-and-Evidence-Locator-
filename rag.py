import json
import faiss
from sentence_transformers import SentenceTransformer

# ============================================================
# PS 160 - RAG Retrieval
# ============================================================

INDEX_FILE = "transcript.index"
METADATA_FILE = "metadata.json"

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


def load_database():

    # Load FAISS index
    index = faiss.read_index(
        INDEX_FILE
    )

    # Load transcript metadata
    with open(
        METADATA_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        metadata = json.load(file)

    return index, metadata


def search_transcript(query, top_k=5):

    index, metadata = load_database()

    # Convert user query into an embedding
    query_embedding = model.encode(
        [query]
    )

    # Search FAISS
    distances, indices = index.search(
        query_embedding,
        top_k
    )

    results = []

    for distance, index_number in zip(
        distances[0],
        indices[0]
    ):

        if index_number == -1:
            continue

        record = metadata[index_number]

        result = {
            "statement": record.get(
                "statement",
                ""
            ),
            "speaker": record.get(
                "speaker",
                "Not explicitly identified"
            ),
            "date": record.get(
                "date",
                ""
            ),
            "pdf_page": record.get(
                "pdf_page",
                ""
            ),
            "printed_page": record.get(
                "printed_page",
                ""
            ),
            "section": record.get(
                "section",
                ""
            ),
            "source": record.get(
                "source",
                ""
            ),
            "similarity_distance": float(distance)
        }

        results.append(result)

    return results


# ============================================================
# Simple testing
# ============================================================

if __name__ == "__main__":

    question = input(
        "Enter your question: "
    )

    results = search_transcript(
        question,
        top_k=5
    )

    print()
    print("========================================")
    print("Retrieved Evidence")
    print("========================================")

    for number, result in enumerate(
        results,
        start=1
    ):

        print()
        print(f"Result {number}")
        print("----------------------------------------")
        print("Statement:")
        print(result["statement"])

        print("Speaker:",
              result["speaker"])

        print("Date:",
              result["date"])

        print("PDF Page:",
              result["pdf_page"])

        print("Printed Page:",
              result["printed_page"])

        print("Section:",
              result["section"])

        print("Source:",
              result["source"])

        print("Distance:",
              result["similarity_distance"])