import json
import os

# ============================================================
# PS 160 - Legislative Transcript Search and Evidence Locator
# Document Ingestion and RAG Data Preparation
# ============================================================

# Input transcript files
INPUT_FILES = [
    "data/lok_sabha_20-07-2026_member1_standardized.json",
    "data/lok_sabha_27-03-2026_member1_standardized.json"
]

# Output file
OUTPUT_FILE = "chunks.json"


def load_json(filename):
    """Load a JSON file."""

    with open(filename, "r", encoding="utf-8") as file:
        return json.load(file)


def get_pages(data):
    """
    Get page records from the standardized JSON file.

    The two datasets use different top-level keys:
    - 20 July 2026: records
    - 27 March 2026: pages
    """

    if "records" in data:
        return data["records"]

    if "pages" in data:
        return data["pages"]

    return []


def prepare_chunks():

    chunks = []
    chunk_number = 1

    for filename in INPUT_FILES:

        if not os.path.exists(filename):
            print("File not found:", filename)
            continue

        print("Reading:", filename)

        data = load_json(filename)
        pages = get_pages(data)

        print("Pages found:", len(pages))

        for page in pages:

            date = page.get("date")
            pdf_page = page.get("pdf_page")
            printed_page = page.get("printed_page")
            section = page.get("section")
            source = page.get("source", filename)

            speaker_turns = page.get("speaker_turns", [])

            # ------------------------------------------------
            # If speaker information is explicitly available
            # ------------------------------------------------

            if speaker_turns:

                for turn in speaker_turns:

                    speaker = turn.get(
                        "speaker",
                        "Not explicitly identified"
                    )

                    statement = turn.get(
                        "statement",
                        ""
                    ).strip()

                    if not statement:
                        continue

                    chunk = {
                        "chunk_id": f"chunk_{chunk_number}",
                        "statement": statement,
                        "speaker": speaker,
                        "date": date,
                        "pdf_page": pdf_page,
                        "printed_page": printed_page,
                        "section": section,
                        "source": source
                    }

                    chunks.append(chunk)
                    chunk_number += 1

            # ------------------------------------------------
            # If speaker information is not explicitly available
            # ------------------------------------------------

            else:

                text = page.get(
                    "clean_text",
                    page.get("text", "")
                ).strip()

                if not text:
                    continue

                chunk = {
                    "chunk_id": f"chunk_{chunk_number}",
                    "statement": text,
                    "speaker": "Not explicitly identified",
                    "date": date,
                    "pdf_page": pdf_page,
                    "printed_page": printed_page,
                    "section": section,
                    "source": source
                }

                chunks.append(chunk)
                chunk_number += 1

    # Save the RAG-ready data
    with open(
        OUTPUT_FILE,
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
    print("Data preparation completed")
    print("========================================")
    print("Total RAG-ready records:", len(chunks))
    print("Output file:", OUTPUT_FILE)


if __name__ == "__main__":
    prepare_chunks()