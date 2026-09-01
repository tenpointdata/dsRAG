# Tests and test corpora

```bash
python -m unittest discover tests/unit          # no API keys needed for some; most call a model
python -m unittest tests.unit.test_fetch_corpora
```

## `tests/data/` — the committed fixtures

Eleven small files, all committed as plain blobs, all referenced by tests.
Two of them are text corpora:

| File | Size | Read by |
|---|---|---|
| `nike_2023_annual_report.txt` | 376 KB | `test_auto_context.py`, `semantic_sectioning_demo.py` |
| `les_miserables.txt` | 34 KB | `test_auto_context.py` (the non-English path) |

**`les_miserables.txt` is an excerpt, not the novel.** It is the first ~40 pages
of the Bibliothèque électronique du Québec edition and it stops mid-sentence.
Replacing it with the full Project Gutenberg text would substitute a 3.3 MB
document for a 34 KB one and change the cost and behaviour of every model call
in `test_auto_context.py`. Nothing should overwrite these two files —
`fetch_corpora.py` does not.

## `tests/fetch_corpora.py` — larger corpora, fetched on demand

```bash
python tests/fetch_corpora.py                     # what is available, and why each one
python tests/fetch_corpora.py rfcs
python tests/fetch_corpora.py --all
python tests/fetch_corpora.py squad --limit 25
python tests/fetch_corpora.py tatqa --file ./tatqa_dataset_train.json
```

Downloads land in `tests/data/corpora/<id>/`, which is gitignored. Nothing is
vendored: a committed copy is tens of megabytes of somebody else's text, and a
stale copy is worse than none — it retrieves plausibly while no longer being
what the publisher serves.

Standard library only, so it runs before dsRAG's dependencies are installed.

### What is available, and why each one

Each corpus is here because it is the only one that puts some part of the
pipeline under real load. A fixture is written by the person testing the code,
which is exactly why it never surprises them.

| Corpus | What it loads | Size |
|---|---|---|
| `rfcs` | **Semantic sectioning** with real numbered sections, and the identifiers hybrid retrieval exists for — `RFC 9110`, `Retry-After`, `HTTP/1.1`. 9110/9111/9112 restate much of the 2616 they obsolete, so the same question has several near-identical right answers. | 1.7 MB |
| `gutenberg` | **Sectioning with nothing to lean on** — chapter headings and no other structure, in documents up to 3.3 MB. `les_miserables.txt` is a 34 KB excerpt of one of these; this is the whole novel. | 9 MB |
| `pgessays` | The same shape at 235 KB, so a full ingest-and-query cycle finishes while you watch it. Right while changing chunk geometry; wrong for measuring. | 0.2 MB |
| `multihoprag` | **Relevant segment extraction under real distractors.** Its published queries need two to four of the 609 articles, and four outlets supply half the corpus while covering the same stories. | 12 MB |
| `squad` | **Whether a pipeline declines.** 442 articles behind 43,498 questions the corpus deliberately does not answer, written by people looking at the paragraph — plausible in exactly the way a guess is. | 42 MB |
| `tatqa` | **AutoContext on tables.** 2,201 real annual-report tables with the prose that explains them. A table row is the least self-describing text in any corpus, which is where a contextual chunk header earns the most. | 9 MB |

### Using one from a test

```python
from tests.fetch_corpora import corpus_dir

path = os.path.join(corpus_dir("rfcs"), "rfc9110 HTTP Semantics.txt")
```

`corpus_dir` rather than a hand-built path: `tests/`, `tests/unit/` and
`dsrag/dsparse/tests/unit/` each reach `tests/data` through a different number
of `../`, and every one of those is written out by hand today.

A test that needs a fetched corpus must **skip** when it is absent rather than
fail — CI has no reason to download 42 MB of Wikipedia.

```python
if not os.path.isdir(corpus_dir("rfcs")):
    self.skipTest("run tests/fetch_corpora.py rfcs")
```

### Licences

Recorded on every entry and printed before every fetch. Nothing is
redistributed by this repository; the script retrieves from the publisher.

| Corpus | Licence |
|---|---|
| `rfcs` | IETF Trust — verbatim reproduction and redistribution permitted |
| `gutenberg` | Public domain (US); files keep the Project Gutenberg boilerplate |
| `pgessays` | Harness MIT; **the essays are © Paul Graham with no redistribution grant** — a local fixture, not something to republish |
| `multihoprag` | ODC-BY 1.0 |
| `squad` | CC-BY-SA 4.0 |
| `tatqa` | CC-BY 4.0 |

### Adding one

Add an entry to `CORPORA` in `fetch_corpora.py`. It declares `files` (one GET
per document) **or** `bundle` + `split` (one GET whose contents become many
documents) — never both and never neither; `test_fetch_corpora.py` asserts it,
because declaring both would leave which one is fetched to reading order.

Only add a corpus if it loads something none of the others already does. If it
does not, it is a duplicate however different its subject matter.
