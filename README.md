# cite-this-paper

`cite-this-paper` is a local, source-grounded claim-verification tool for
academic PDFs. A corpus is an independent SQLite-backed database: add PDFs,
rebuild its retrieval index when a batch is complete, then verify claims
against its papers.

## Quick start

```bash
python -m pip install -e .

cite-this-paper init-db data/corpora/water
cite-this-paper add-directory --database data/corpora/water papers --defer-rebuild
cite-this-paper rebuild-index --database data/corpora/water
cite-this-paper verify-claim --database data/corpora/water "Your scientific claim"
```

The standard verification workflow always performs dense and lexical
retrieval, hybrid rank fusion, Qwen passage reranking, and local Qwen evidence
verification. The default model stages expect CUDA; pass `--device cpu` to
`verify-claim` when the selected models support CPU execution.

Each corpus contains `corpus.sqlite`, managed PDF copies, and one current
`vectors/embeddings.npy` matrix. If a PDF has identical SHA-256 content to one
already stored, the CLI asks whether to discard it or replace its managed copy.
After documents are added, the CLI asks whether to rebuild indexes. Deferring a
rebuild keeps the old complete index active and causes verification to warn
that pending documents were not searched.

The original scripts remain available as reference and are reused for the
established extraction, sentence reconstruction, passage construction, and
end-matter classification heuristics during this transition.
