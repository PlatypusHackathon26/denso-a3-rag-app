# Architecture

## Goal

This repository is a local-first enterprise knowledge assistant designed around a simple demo loop:

**ingest → retrieve evidence → answer with citations → measure quality/latency**.

For DENSO Factory Hacks 2026, the design is especially relevant to challenge **A3 — AI Agent Understanding Diverse Document Formats**, whose public description calls for a multi-format ingestion pipeline, a RAG chatbot demo, and accuracy/latency evaluation.

## Components

```text
                    ┌─────────────────────────────┐
                    │        Streamlit UI         │
                    │ chat / ingest / evidence   │
                    └──────────────┬──────────────┘
                                   │
                            ┌──────▼──────┐
                            │ RAGPipeline │
                            └──────┬──────┘
                                   │
           ┌───────────────────────┼────────────────────────┐
           │                       │                        │
      INGEST PATH             QUERY PATH               EVALUATION
           │                       │                        │
  PDF/MD/TXT/Image        query embedding           eval/questions.json
           │                       │                        │
     parser / vision        LanceDB cosine               LLM judge
           │                       │                        │
        chunking          lexical exact-match              │
           │                       │                        │
        Ollama embed      hybrid scoring → rerank          │
           │                       │                        │
        LanceDB ◄──────── evidence chunks ────────────────┘
                                   │
                              grounded LLM
                                   │
                             answer + [1][2]
```

## Retrieval strategy

The project intentionally uses **hybrid retrieval** rather than vector similarity alone:

1. Vector similarity handles semantic matches and bilingual phrasing.
2. A lightweight lexical scorer boosts exact words, phrases and normalized digit strings. This is useful for part numbers, ABNs, phone numbers, addresses and dates.
3. Optional cross-encoder reranking can refine the top candidate set.
4. A source repetition cap prevents one document from consuming the whole context.
5. A vector threshold blocks obviously off-topic queries from reaching the LLM.

This is deliberately dependency-light. A full BM25 implementation can be added later when corpus size requires it.

## Grounding / safety

Document text is passed as **data**, not instructions. The system prompt explicitly says to ignore instructions inside retrieved documents. Context is serialized as JSON rather than nested XML tags, which avoids structural tag injection.

The model is instructed to cite factual claims as `[1]`, `[2]`, etc. The application validates that cited numbers exist in the retrieved set and exposes the underlying evidence to the user.

## Multimodal path

Text/PDF files are parsed directly. Image files (`.png`, `.jpg`, `.jpeg`, `.webp`) can be enabled with a local Ollama vision model. The vision model extracts visible text, numbers, labels and tables into a retrieval-friendly text representation; that text is then embedded with the same pipeline.

This keeps the architecture simple while still giving the hackathon demo a concrete story for heterogeneous documents. For production use, the next step is to preserve page/region bounding boxes and use a layout-aware multimodal index.

## Reproducibility

`data/index/index_meta.json` stores the embedding model, parser and chunking signature. Any incompatible change triggers a rebuild instead of silently mixing vectors from different configurations.

`eval/reports/latest.json` stores answer accuracy, retrieval metrics, citation rate and latency statistics so the demo can show measurable system behavior.
