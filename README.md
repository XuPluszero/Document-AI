Document AI
 (No Raw Data are showned due to confidential reason)
====

Analyze insurance policy documents with LLMs and extract structured information. 

## Overview
This repository provides an end-to-end pipeline to:
- chunk long policy documents,
- extract policy-level metadata,
- retrieve relevant sections per extraction target, and
- perform structured information extraction, followed by evaluation.

## Prerequisites
1. Git LFS (large JSON files are stored with LFS)
   - Install and initialize:
     - Ubuntu/Debian: `sudo apt-get install git-lfs && git lfs install`
     - macOS (Homebrew): `brew install git-lfs && git lfs install`
     - Otherwise: see `https://git-lfs.com` and then run `git lfs install`
   - After cloning, make sure LFS content is pulled: `git lfs pull`
2. Python environment
   - Python 3.10+ recommended
   - Install dependencies: `pip install -r requirements.txt`
3. OpenAI API key
   - Export: `export OPENAI_API_KEY="YOUR_API_KEY"`

## Workflow
0. Convert the PDF policy to Markdown (e.g., via Google Docs or your preferred tool).
1. Because policies are often long (100+ pages), use an LLM to chunk the document into sections.
2. Extract useful policy metadata for downstream steps.
3. For each extraction target, retrieve relevant document sections.
4. Using the retrieved sections and metadata, run structured information extraction.
5. Evaluate extraction quality.

## Data Overview
- `raw_data/ground_truths/`: Each JSON file corresponds to one policy. The raw text is already segmented into document sections under `chunker_result/document_sections`. Concatenating those sections reconstructs the original document.
- `raw_data/pdf_sample/`: Example PDF file. Note: all essential content already exists within the JSON document sections.
- `processed_data/`: Intermediate and final outputs from the pipeline (e.g., retrieval and extraction logs/results).

## How to Run

### 1) Retrieval
```bash
python code/step_3_retrieval.py --test-run true --model-name gpt-4.1
```
- **test-run**: `true` runs a single extraction target for a quick check. Set to `false` for the full run (longer and higher cost).
- **model-name**: Defaults to `gpt-4.1`; you may use other OpenAI models available to your account.

### 2) Extraction
```bash
python code/step_4_extraction.py --test-run true --model-name gpt-4.1 --reasoning-model NO
```
- **test-run**: Same behavior as above.
- **model-name**: Defaults to `gpt-4.1`.
- **reasoning-model**: Set to `YES` for reasoning models; set to `NO` for non-reasoning models (e.g., `gpt-4.1`).

### 3) Evaluation
```bash
python code/step_5_evaluation.py --model-generation-path processed_data/step_4_extraction_log_gpt-4.1.json
```
Evaluates the correctness of the extracted outputs.

## Notes
- Running full retrieval and extraction will invoke an external API and incur latency and cost.
- Model availability and names may vary by account/region; adjust `--model-name` accordingly.
