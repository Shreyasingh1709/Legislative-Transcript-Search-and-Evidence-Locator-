"""Focused tests for adapting and chunking page-level English records."""

import re
import unittest

from adapt_source import adapt_dataset
from chunking import chunk_records, validate_chunk_output


class FakeTokenizer:
    def __call__(self, text, add_special_tokens, return_offsets_mapping):
        return {
            "offset_mapping": [
                (match.start(), match.end()) for match in re.finditer(r"\S+", text)
            ]
        }

    def encode(self, text, add_special_tokens):
        return re.findall(r"\S+", text)


class AdapterChunkingTests(unittest.TestCase):
    def make_dataset(self, retrieval_text):
        page = {
            "document_id": "LS_2026_03_27",
            "source_file": "march_source.pdf",
            "date": "2026-03-27",
            "pdf_page": 2,
            "retrieval_text": retrieval_text,
            "original_text": "Original page text.",
            "cleaned_text": "Cleaned page text.",
            "quality_warnings": ["review"],
            "language_status": "english_only",
            "nested_metadata": {"origin": ["pdf", "page"]},
        }
        return {
            "retained_record_count": 1,
            "documents": [
                {
                    "document_id": "LS_2026_03_27",
                    "source_file": "march_source.pdf",
                    "date": "2026-03-27",
                }
            ],
            "pages": [page],
        }, page

    def test_adapter_preserves_page_metadata_without_attribution(self):
        dataset, source_page = self.make_dataset("A short English passage.")

        records, report = adapt_dataset(dataset)

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["text"], source_page["retrieval_text"])
        self.assertEqual(records[0]["speaker"], "Unknown")
        self.assertEqual(records[0]["attribution_status"], "unverified")
        for key, value in source_page.items():
            self.assertEqual(records[0][key], value)
        self.assertTrue(report["record_ids_unique"])
        self.assertTrue(report["source_pages_unique"])

    def test_chunking_respects_token_limit_and_source_traceability(self):
        passage = " ".join(f"statement{i}." for i in range(650))
        dataset, _ = self.make_dataset(passage)
        records, _ = adapt_dataset(dataset)

        chunks = chunk_records(records, FakeTokenizer())
        report = validate_chunk_output(records, chunks)

        self.assertEqual(report["validation_errors"], [])
        self.assertEqual(report["input_record_count"], 1)
        self.assertTrue(report["unique_chunk_ids"])
        self.assertTrue(all(chunk["pdf_page"] == 2 for chunk in chunks))
        self.assertTrue(
            all(chunk["attribution_status"] == "unverified" for chunk in chunks)
        )
        self.assertTrue(all(chunk["token_count"] <= 500 for chunk in chunks))
        self.assertTrue(
            all(chunk["nested_metadata"] == {"origin": ["pdf", "page"]} for chunk in chunks)
        )

    def test_adapter_rejects_empty_page_text(self):
        dataset, _ = self.make_dataset(" ")

        with self.assertRaisesRegex(ValueError, "missing or empty retrieval_text"):
            adapt_dataset(dataset)


if __name__ == "__main__":
    unittest.main()
