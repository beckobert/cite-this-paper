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

The verifier distinguishes `NOT_MENTIONED` from `RELATED_ONLY`: the former
means the passage is unrelated to the claim, while the latter means it is
topically related or merely references relevant work without supporting or
contradicting the claim. A lack of supporting information is never treated as
`CONTRADICTS`; contradiction requires explicitly incompatible passage evidence.

`verify-claim` prints each verdict with its source metadata, verifier reason,
and the evidence sentences selected by the verifier. Every result includes one
copy-pasteable `show-sentences` command for all displayed evidence; it renders
one combined highlighted image for each affected PDF page. If the verifier
selects no individual sentence, the command clearly labels and displays every
sentence in the passage as passage-wide evidence.
Pass `--verbose` to include dense, lexical, fusion, and reranker diagnostics.

Each corpus contains `corpus.sqlite`, managed PDF copies, and one current
`vectors/embeddings.npy` matrix. If a PDF has identical SHA-256 content to one
already stored, the CLI asks whether to discard it or replace its managed copy.
After documents are added, the CLI asks whether to rebuild indexes. When a
corpus still has a pending rebuild, an interactive `verify-claim` run asks
whether to rebuild, continue with the previous index, or quit. In
non-interactive use it stops unless `--allow-stale-index` explicitly permits
using the previous index; pending documents are then not searched.

Verification output is advisory only. It is not guaranteed correct, and the
user is solely responsible for verifying and deciding how to use the result.

The package includes its own PDF extraction, sentence reconstruction, passage
construction, and end-matter classification modules. The top-level `scripts/`
directory remains only as a legacy reference and is not required at runtime or
included in the package distribution.
