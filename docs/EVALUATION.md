# Evaluation plan

The sample benchmark is intentionally small and synthetic. It is useful for regression testing, not for claiming production accuracy.

## Retrieval metrics

Questions may contain a `sources` array. For those questions the evaluator computes:

- **Recall@K**: fraction of expected source files present in the retrieved list.
- **MRR**: reciprocal rank of the first expected source.

The current sample file contains only a small subset of source labels. For a serious benchmark, label the correct source(s) for every question and add hard negatives.

## Answer accuracy

A local LLM judge compares the generated answer with the reference answer. Because an LLM judge is not a gold standard, review a sample manually and keep the judge model separate from the generation model when possible.

## Latency

Each answer records:

- retrieval latency
- generation latency
- end-to-end latency

The report exposes mean and p95 latency. Run at least 20–50 representative questions on the target demo machine before presenting a final number.

## Recommended DENSO benchmark expansion

For the actual factory dataset, build a test set covering:

1. direct fact lookup (part number, limit, phone, date)
2. table lookup
3. multi-hop questions across two documents
4. bilingual questions
5. image / scanned-document questions
6. unanswerable questions
7. prompt-injection strings embedded in documents
8. long-context questions

Report retrieval and answer metrics separately so a strong generation model cannot hide weak retrieval.
