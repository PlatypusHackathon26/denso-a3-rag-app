# Project review before/after refactor

## Overall assessment

The original repository is already a **real RAG engine**, not a toy prototype. The strongest parts are the ingestion lifecycle, source tracking, privacy guard and automated tests. The main weakness is that the project was optimized around the RAG core rather than the hackathon deliverable: a visible end-to-end product that demonstrates diverse document understanding and measurable quality.

For DENSO Factory Hacks 2026, the public A3 challenge explicitly asks for multi-format understanding (PDFs, images, tables and bilingual documents), a RAG chatbot demo, and an accuracy/latency report. The original code directly addressed only part of that scope.

## Strengths in the original code

| Strength | Why it matters |
|---|---|
| Incremental ingest using SHA-256 | avoids needless re-embedding and handles file updates/deletions |
| Source/page/heading metadata | makes answers inspectable instead of opaque |
| Threshold gate | prevents obviously irrelevant questions from being sent to the LLM |
| Local-only Ollama guard | gives the privacy story a concrete technical control |
| Optional local reranker | leaves room for better retrieval without a cloud API |
| Evaluation module | shows the team is thinking about measurement, not only demo polish |
| Offline unit tests | catches pipeline regressions without requiring an LLM during CI |

## Main weaknesses to address

### 1. Vector-only retrieval was too weak for exact industrial identifiers

Factory knowledge contains many exact values: part numbers, quantities, dates, phone numbers, tolerance values and codes. Pure semantic retrieval can miss or mis-rank exact matches.

**Changed:** add a lightweight lexical score and hybrid ranking. Numeric strings are normalized before scoring.

### 2. The project had no judge-friendly application UI

A CLI is excellent for engineering, but it does not demonstrate the whole product quickly during a live hackathon presentation.

**Changed:** add a Streamlit local demo with chat, ingest, index statistics, retrieved evidence and latency.

### 3. Images were not a supported input

The original supported PDF/Markdown/TXT only.

**Changed:** add optional image ingestion through a local Ollama vision model for PNG/JPG/JPEG/WEBP. The extracted content is indexed like normal text.

### 4. The prompt structure was harder to reason about under prompt injection

The original context used nested XML-like tags. Even with closing-tag neutralization, the model still received document text that looked structurally similar to control markup.

**Changed:** serialize retrieved evidence as JSON and clearly distinguish data from instructions in the system prompt.

### 5. Evaluation did not measure latency

A3 explicitly calls for an accuracy/latency report.

**Changed:** record retrieval, generation and end-to-end latency, plus mean and p95 in the evaluation report.

### 6. Chat had no memory

The original `chat` command treated every question independently.

**Changed:** retain the last four turns for contextual follow-up questions.

### 7. The benchmark was too small and weakly labeled

Only a small fraction of the sample questions had expected source labels. That makes retrieval metrics incomplete.

**Not hidden:** the sample benchmark is still synthetic. For the real hackathon result, build a DENSO-specific evaluation set with source labels, hard negatives, bilingual variants and unanswerable questions.

## What the refactored project can demonstrate now

```text
1. Put documents into data/raw/
2. localrag ingest
3. Open the web app
4. Ask a normal question
5. Show exact evidence + page + score
6. Ask an exact-number question
7. Show hybrid retrieval
8. Ask an unsupported question
9. Show that generation is skipped
10. Run evaluation and show accuracy + latency
```

## What should be built next on the real DENSO data

The next work should prioritize **evidence quality**, not adding more framework names.

1. Preserve page + region provenance for tables and images.
2. Add OCR/VLM fallback for scanned PDFs.
3. Benchmark Japanese/Vietnamese/English queries if the real corpus needs them.
4. Add multi-hop retrieval for questions spanning several manuals.
5. Add a human-reviewed golden set and compare retrieval against answer accuracy.
6. Package the exact demo machine environment so a judge can reproduce it.

The important presentation principle is: **show the evidence and the measurement next to the answer**. That makes the RAG behavior easy for a technical judge to verify.
