# cite-this-paper

`cite-this-paper` is a local package for checking scientific claims against a
curated collection of academic PDFs. It extracts and indexes the source
documents, retrieves relevant passages, reranks them, and asks a local verifier
model to classify their relationship to a claim.

The package is designed to keep results traceable to their original PDFs. Its
output is advisory: model verdicts can be wrong, incomplete, or misleading.
You are solely responsible for checking the evidence and deciding how to use
the result.

## How the package is built

The unit of work is a **corpus**: an independent database for one research
purpose. Every command selects a corpus explicitly with `--database`, so PDFs,
indexes, verification history, and source-review images are never shared
implicitly between projects.

A corpus directory contains:

- `corpus.sqlite` — documents, extracted pages, sentences, passages, search
  data, and verification audit records;
- `pdfs/` — managed copies of the ingested source PDFs;
- `vectors/embeddings.npy` — the current dense retrieval index;
- `corpus-config.json` — model names and passage-building settings;
- `review/` — generated source-page images when evidence is rendered.

Internally, the package has five stages:

1. PDF ingestion extracts page text and word geometry, then reconstructs
   sentences and retrieval passages.
2. Passage classification excludes terminal material such as references from
   retrieval.
3. Index rebuilding creates dense BGE-M3 embeddings and a SQLite FTS lexical
   index.
4. Claim verification combines dense and lexical retrieval, Qwen reranking,
   and local Qwen passage verification.
5. Source review renders the selected evidence sentences directly on their PDF
   pages.

## Standard workflow

Install the package, create a corpus, add PDFs, rebuild its index, and verify a
claim:

```bash
python -m pip install .

cite-this-paper init-db data/corpora/corpus-001
cite-this-paper add-directory --database data/corpora/corpus-001 /path/to/papers --defer-rebuild
cite-this-paper rebuild-index --database data/corpora/corpus-001
cite-this-paper verify-claim --database data/corpora/corpus-001 "Your scientific claim"
```

Use `add-pdf` instead of `add-directory` when adding a single file. New PDFs
are stored immediately, but do not become searchable until `rebuild-index`
finishes. The ingestion report states whether rebuilding was completed or
deferred.

Verification always follows the same sequence:

1. Create an embedding for the claim and retrieve dense and lexical candidates.
2. Fuse both result lists into a shared candidate ranking.
3. Rerank the best candidates with the Qwen reranker.
4. Verify the highest-ranked passages with the Qwen verifier.

The verifier can return `DIRECT_SUPPORT`, `PARTIAL_SUPPORT`, `CONTRADICTS`,
`RELATED_ONLY`, or `NOT_MENTIONED`. A passage that omits information is not a
contradiction; `CONTRADICTS` requires explicitly incompatible evidence.

Each verification result includes source metadata, the verifier’s reason, and
one copy-pasteable command for displaying all selected evidence:

```bash
cite-this-paper show-sentences --database data/corpora/water <sentence-id> [<sentence-id> ...]
```

This creates one highlighted image per affected PDF page. If the verifier does
not select individual sentences, the result treats the entire passage as the
evidence and provides its sentence IDs instead.

## Additional commands and options

### Ingestion

`add-pdf` and `add-directory` report the PDF currently being processed, any
duplicate decision, and a final summary. Exact-content duplicates are detected
by SHA-256 hash.

- `--on-duplicate discard` keeps the stored copy; `replace` updates it. Without
  either option, an interactive session asks which action to take.
- `--rebuild` rebuilds immediately; `--defer-rebuild` leaves it for later.
- `--title`, `--author`, `--year`, `--journal`, `--doi`, and `--citation-key`
  override document metadata.
- `--debug` shows low-level extraction diagnostics, including merged physical
  PDF blocks.
- `--quiet` hides interim processing messages while keeping the final report.

Currently, only the metadata attached to the PDF or supplied by the user is
considered. In the future, the package will try to extract the metadata directly
from the PDF.

### Indexing and verification

`rebuild-index` and `verify-claim` show model and processing progress by
default. Pass `--quiet` to hide this progress. `verify-claim --verbose` adds
dense, lexical, fusion, and reranker scores to each result.

If documents have been added since the last index rebuild, interactive
verification asks whether to rebuild, continue with the old index, or quit. In
non-interactive use, pass `--allow-stale-index` to search the previous index;
pending PDFs will not be searched.

The default reranker and verifier use CUDA. Use `--device cpu` when the chosen
models support CPU execution. Candidate counts can be tuned with
`--candidate-k`, `--rerank-k`, and `--verify-k`; the standard retrieval,
reranking, and verification stages remain mandatory.

### Corpus cleanup

`cleanup-databases` permanently removes whole corpus directories, including
their PDFs, database, vectors, and review images. It previews first; deletion
requires `--apply`:

```bash
cite-this-paper cleanup-databases data/corpora/old-project
cite-this-paper cleanup-databases data/corpora/old-project --apply
```

To find inactive corpora, scan `data/corpora` by default, or provide another
parent with `--root`:

```bash
cite-this-paper cleanup-databases --unused-for 90
cite-this-paper cleanup-databases --unused-for 90 --root path/to/corpora --apply
```

Normal corpus commands update the last-access timestamp used by age-based
cleanup. The schema is intentionally development-oriented and has no migration
path: recreate corpora after incompatible schema changes. Explicit cleanup can
still remove an older corpus by path.
