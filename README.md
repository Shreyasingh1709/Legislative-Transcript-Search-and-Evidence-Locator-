# Lok Sabha Debate Preprocessing

This project prepares page-level text datasets from two Lok Sabha debate PDFs:
27 March 2026 and 20 July 2026. It extracts selectable PDF text, cleans
whitespace without discarding the original text, combines the cleaned
documents, and creates a conservative English-only dataset for downstream
retrieval.

The pipeline does not perform OCR, translate text, infer speakers or speaker
turns, or create embeddings and a vector store.

## Pipeline

1. **Extract** selected pages or all pages from each PDF with `pdfplumber`.
   Extraction output includes page numbers, source metadata, and a quality
   report. Empty pages and extraction errors are recorded rather than silently
   dropped.
2. **Clean** extracted text in a separate step. Original text is retained;
   whitespace is normalized and suspicious sequences are reported as warnings,
   not repaired.
3. **Combine** the two complete cleaned documents after checking their page
   counts, page-number sequences, and provenance.
4. **Filter** the combined dataset into retained, excluded, and uncertain
   records. The English-only filter is heuristic; it does not translate or
   guarantee language identification.

## Project files and data

```text
.
├── dataset/                 # Source debate PDFs
├── data/
│   ├── intermediate/        # Raw extraction JSON
│   └── processed/           # Cleaned and derived JSON datasets
├── preprocess.py            # PDF text extraction
├── text_cleaner.py          # Whitespace normalization and warnings
├── combine_dataset.py       # Validated two-document merge
├── english_only_dataset.py  # Conservative English retrieval filter
├── requirements.txt
└── tests/
```

The input PDFs are:

- `dataset/lsd_18_VII_27-03-2026_original_corrected.pdf` (1,300 pages)
- `dataset/lsd_18_VIII_20-07-2026_original_corrected.pdf` (1,315 pages)

The scripts create their output directories when needed. Sample runs write to
`data/intermediate/`; cleaned, combined, and English-only datasets belong in
`data/processed/`.

## Requirements and setup

Python 3.10 is the development version used to verify this project. Install
dependencies from the project directory:

```powershell
py -3.10 -m pip install -r requirements.txt
```

The shared `requirements.txt` includes dependencies for PDF preprocessing,
the related transcript search application, and the test suite. `pdfplumber`
handles text extraction; no external PDF utility or OCR executable is
required.

## Run the pipeline

Run commands from the project directory. The following commands extract a few
representative pages from each PDF for review:

```powershell
py -3.10 preprocess.py --input "dataset\lsd_18_VII_27-03-2026_original_corrected.pdf" --output "data\intermediate\lok_sabha_27-03-2026_sample.json" --date 2026-03-27 --sample-pages "1,650,1300" --expected-total-pages 1300
py -3.10 preprocess.py --input "dataset\lsd_18_VIII_20-07-2026_original_corrected.pdf" --output "data\intermediate\lok_sabha_20-07-2026_sample.json" --date 2026-07-20 --sample-pages "1,658,1315" --expected-total-pages 1315
```

Review the sample output and any quality warnings before processing all pages.
For complete extraction, use `--all-pages` instead of `--sample-pages`:

```powershell
py -3.10 preprocess.py --input "dataset\lsd_18_VII_27-03-2026_original_corrected.pdf" --output "data\intermediate\march_27_extracted.json" --date 2026-03-27 --all-pages --expected-total-pages 1300
py -3.10 preprocess.py --input "dataset\lsd_18_VIII_20-07-2026_original_corrected.pdf" --output "data\intermediate\july_20_extracted.json" --date 2026-07-20 --all-pages --expected-total-pages 1315
```

Clean the complete extraction files:

```powershell
py -3.10 text_cleaner.py --input "data\intermediate\march_27_extracted.json" --output "data\processed\march_27_cleaned.json" --all-pages
py -3.10 text_cleaner.py --input "data\intermediate\july_20_extracted.json" --output "data\processed\july_20_cleaned.json" --all-pages
```

Then combine the cleaned files and build the English-only retrieval dataset:

```powershell
py -3.10 combine_dataset.py
py -3.10 english_only_dataset.py --input "data\processed\combined_processed.json" --output "data\processed\english_only_dataset.json"
```

`combine_dataset.py` expects 1,300 March pages and 1,315 July pages, in
sequential one-based PDF page order. It writes 2,615 combined page records.
`english_only_dataset.py` retains pages with at least ten Latin-script words,
at least three distinct common English function words, and no quality warnings
or unresolved glyphs. Pages it cannot confidently include are preserved in
`uncertain_records`; excluded records are summarized by reason.

Use each script's `--help` option to see its command-line arguments. Extraction
and cleaning can also be run on selected pages with `--sample-pages`.

## Output and quality notes

- Extraction JSON preserves raw PDF text and reports empty pages and extraction
  failures. If extraction errors occur, the output and report are still
  written, and the script exits with status 2.
- Cleaned page records include `original_text`, `cleaned_text`, and
  `quality_warnings`. Cleaning normalizes line endings and horizontal
  whitespace, and limits runs of blank lines; it preserves remaining line
  breaks, punctuation, and Unicode text.
- The cleaner flags `(cid:...)` placeholders, replacement and private-use
  characters, likely mojibake, and unexpected control characters without
  changing them.
- `pdfplumber` extracts embedded/selectable text only. Image-only pages,
  Hindi shaping, columns, tables, reading order, and other source-PDF issues
  may require manual review. OCR is not implemented.
- Page numbers refer to positions in each PDF, not printed page numbers.
  Speaker annotations in any external source dataset are not generated or
  included by this pipeline.

## Tests

Run the test suite from the project directory:

```powershell
py -3.10 -m pytest
```

The tests cover extraction behavior and error reporting, text cleaning and
warnings, combined-dataset validation and provenance, and English-only page
classification. PDF extraction tests use mocked pages.
