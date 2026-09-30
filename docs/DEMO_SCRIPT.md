# 5-minute demo script

## 0:00 — Problem

“Factory knowledge lives in PDFs, manuals, tables, images and bilingual documents. Searching manually costs time, and an LLM without evidence can hallucinate. Our assistant indexes the knowledge locally, retrieves the relevant evidence, then answers with citations.”

## 0:30 — Show the architecture

Open the README architecture diagram and point to:

- multi-format ingestion
- hybrid retrieval
- grounded answer generation
- evidence + metrics

## 1:00 — Ingest

```bash
localrag doctor
localrag ingest
localrag stats
```

Then open the Streamlit UI:

```bash
streamlit run app.py
```

## 2:00 — Normal question

Ask a question whose answer is clearly present in the sample documents.

Show:

- concise answer
- citations `[1]`
- source filename + page
- retrieved chunk text
- latency

## 3:00 — Exact numeric lookup

Ask for a phone number / ABN / date.

Explain that lexical scoring complements embeddings for exact identifiers.

## 3:30 — Unsupported question

Ask something unrelated to the corpus.

Show that the retriever returns no qualifying evidence and the LLM is not called.

## 4:00 — Image input (A3 demo path)

Set:

```env
VISION_ENABLED=true
VISION_MODEL=<your local vision model>
```

Place an image containing a table or bilingual label in `data/raw/`, ingest it, and ask for one exact field.

## 4:30 — Evaluation

```bash
localrag eval
```

Show:

- answer accuracy
- Recall@K / MRR for labeled questions
- average and p95 latency
- citation rate

Close with: “The next production step is to connect the same pipeline to DENSO’s real factory knowledge and preserve richer layout/region provenance.”
