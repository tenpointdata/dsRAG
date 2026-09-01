#!/usr/bin/env python3
"""Fetch public plain-text corpora to test dsRAG against.

    python tests/fetch_corpora.py                     # what is available
    python tests/fetch_corpora.py rfcs
    python tests/fetch_corpora.py --all
    python tests/fetch_corpora.py squad --limit 25
    python tests/fetch_corpora.py tatqa --file ./tatqa_dataset_train.json

`tests/data/` already holds two .txt corpora, and both are committed because
tests assert on their contents: `nike_2023_annual_report.txt` (a whole 10-K) and
`les_miserables.txt` (a 34 KB French EXCERPT, not the novel -- swapping in the
full Gutenberg text would change every assertion in test_auto_context.py). This
script never touches either. It writes into `tests/data/corpora/`, which is
gitignored, so a fetched corpus can be as large as it needs to be.

Each corpus is here because it is the only one that puts some part of the
pipeline under real load -- see WHY on each entry. Nothing is vendored: a
committed copy is tens of megabytes of somebody else's text, and a stale copy
retrieves plausibly while no longer being what the publisher serves.

Standard library only, so this runs before dsRAG's dependencies are installed.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import unicodedata
import urllib.error
import urllib.request

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "data"))
CORPORA_DIR = os.path.join(DATA_DIR, "corpora")

# Below this a 200 is an error page, not a document. Writing that body as
# rfc9110.txt puts a file in the corpus that parses to nothing and is
# indistinguishable afterwards from a document that genuinely had little text.
MIN_PLAUSIBLE_BYTES = 512

# Filesystems cap a name in BYTES, not characters: ext4 stops at 255, and 120
# CJK characters is 360 bytes. The budget below 255 leaves room for ".txt" and
# for the ".partial" suffix every write goes through.
MAX_NAME_BYTES = 200

TIMEOUT_SECONDS = 180


def corpus_dir(corpus_id: str) -> str:
    """Where a fetched corpus lives. Use this from a test rather than a path."""
    return os.path.join(CORPORA_DIR, corpus_id)


# --------------------------------------------------------------------------
# Names
#
# A document name becomes a file name, and dsRAG derives a document's title
# from the file it was read from -- so this is not cosmetic. Titles inside a
# fetched corpus carry slashes, quotation marks and the occasional control
# character, and a name of "../../etc/thing" writes outside the destination.
# --------------------------------------------------------------------------

_UNSAFE = re.compile(r"[^\w .-]+", re.UNICODE)


def _byte_length(name: str) -> int:
    return len(name.encode("utf-8"))


def is_safe_name(name: str) -> bool:
    if not name or name != name.strip() or name.startswith("."):
        return False
    if ".." in name or "/" in name or "\\" in name:
        return False
    if any(unicodedata.category(character) == "Cc" for character in name):
        return False
    return _byte_length(name) <= MAX_NAME_BYTES


def as_document_name(raw: str, fallback: str) -> str:
    """A title from inside a corpus, turned into something writable."""
    cleaned = _UNSAFE.sub(" ", raw)
    cleaned = re.sub(r"\.{2,}", ".", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    # Truncate by bytes rather than characters, or a non-Latin title passes the
    # length check and fails the write.
    while _byte_length(cleaned) > MAX_NAME_BYTES:
        cleaned = cleaned[:-1]
    cleaned = cleaned.strip(". ")

    return cleaned if is_safe_name(cleaned) else fallback


# --------------------------------------------------------------------------
# Splits -- the only per-corpus code here
# --------------------------------------------------------------------------


def _records(contents: str, corpus_id: str) -> list:
    parsed = json.loads(contents)
    if not isinstance(parsed, list):
        raise ValueError(f"{corpus_id}: expected a JSON array at the top level")
    return parsed


def split_multihop_rag(contents: str) -> list:
    """One .txt per news article.

    The outlet and the date are carried INTO the text because a .txt has
    nowhere else to put them, and MultiHop-RAG's queries condition on both --
    "which outlet reported..." is unanswerable from the body alone.

    Named "{ordinal} {headline}": two outlets covering one story file
    near-identical headlines, which is exactly why this corpus is here, and two
    documents that collide on name are one document on disk.
    """
    documents = []
    for index, entry in enumerate(_records(contents, "multihoprag"), start=1):
        if not isinstance(entry, dict) or not isinstance(entry.get("body"), str):
            continue
        ordinal = f"{index:04d}"
        heading = [
            str(entry.get(key))
            for key in ("title", "source", "published_at")
            if isinstance(entry.get(key), str) and entry[key]
        ]
        documents.append(
            (
                as_document_name(f"{ordinal} {entry.get('title', '')}", ordinal),
                "\n".join(heading) + "\n\n" + entry["body"] + "\n",
            )
        )
    return documents


def split_squad(contents: str) -> list:
    """One .txt per Wikipedia ARTICLE, not per paragraph.

    SQuAD's unanswerable questions were written by people looking at the
    paragraph they sit beside, so a corpus of loose paragraphs hands the
    retriever the answer's neighbourhood for free. Rejoined into articles,
    finding the right passage is work again -- and only then does declining to
    answer measure anything.
    """
    parsed = json.loads(contents)
    if not isinstance(parsed, dict):
        raise ValueError("squad: expected a JSON object at the top level")

    documents = []
    for index, article in enumerate(parsed.get("data", []), start=1):
        if not isinstance(article, dict):
            continue
        ordinal = f"{index:04d}"
        title = str(article.get("title", ordinal)).replace("_", " ")
        paragraphs = [
            paragraph["context"]
            for paragraph in article.get("paragraphs", [])
            if isinstance(paragraph, dict) and isinstance(paragraph.get("context"), str)
        ]
        if not paragraphs:
            continue
        documents.append(
            (as_document_name(title, ordinal), title + "\n\n" + "\n\n".join(paragraphs) + "\n")
        )
    return documents


def split_tatqa(contents: str) -> list:
    """One .txt per table, with the prose that explains it.

    Rows are pipe-delimited rather than aligned: alignment is padding, padding
    is tokens, and a chunker that split on width would cut a row in half. The
    first paragraph is TAT-QA's caption -- it names the figures, so it leads the
    document and becomes the title.
    """
    documents = []
    for index, entry in enumerate(_records(contents, "tatqa"), start=1):
        if not isinstance(entry, dict):
            continue
        ordinal = f"{index:04d}"
        table = entry.get("table")
        rows = table.get("table", []) if isinstance(table, dict) else []
        prose = [
            paragraph["text"]
            for paragraph in entry.get("paragraphs", [])
            if isinstance(paragraph, dict) and isinstance(paragraph.get("text"), str)
        ]
        if not rows and not prose:
            continue

        caption = prose[0] if prose else f"Financial table {ordinal}"
        rendered = "\n".join(
            " | ".join(str(cell).strip() for cell in row) for row in rows if isinstance(row, list)
        )
        documents.append(
            (
                as_document_name(f"{ordinal} {caption}", ordinal),
                caption + "\n\n" + rendered + "\n\n" + "\n\n".join(prose[1:]) + "\n",
            )
        )
    return documents


# --------------------------------------------------------------------------
# The catalogue
# --------------------------------------------------------------------------


def _rfc(number: int, name: str) -> tuple:
    return (f"rfc{number} {name}", f"https://www.rfc-editor.org/rfc/rfc{number}.txt")


def _book(repository: str, book_id: int, name: str) -> tuple:
    return (
        name,
        f"https://raw.githubusercontent.com/GITenberg/{repository}_{book_id}"
        f"/master/{book_id}.txt",
    )


_ESSAYS = (
    "https://raw.githubusercontent.com/gkamradt/LLMTest_NeedleInAHaystack"
    "/main/needlehaystack/PaulGrahamEssays"
)


def _essay(name: str) -> tuple:
    return (name, f"{_ESSAYS}/{name}.txt")


CORPORA = {
    "rfcs": {
        "title": "IETF Request for Comments",
        "why": (
            "Numbered sections that become citation locators without a model inventing one, "
            "and identifiers no embedding has an opinion about -- RFC 9110, Retry-After, "
            "HTTP/1.1. 9110/9111/9112 restate much of the 2616 they obsolete, so the same "
            "question has several near-identical right answers in the corpus."
        ),
        "licence": "IETF Trust -- verbatim reproduction and redistribution permitted",
        "home": "https://www.rfc-editor.org/retrieve/",
        "approx_bytes": 1_700_000,
        "files": [
            _rfc(9110, "HTTP Semantics"),
            _rfc(9111, "HTTP Caching"),
            _rfc(9112, "HTTP1.1"),
            _rfc(2616, "HTTP1.1 obsoleted"),
            _rfc(2119, "Key words for requirement levels"),
            _rfc(8259, "The JSON data interchange format"),
            _rfc(9000, "QUIC"),
            _rfc(9562, "Universally unique identifiers"),
        ],
    },
    "gutenberg": {
        "title": "Project Gutenberg novels",
        "why": (
            "Semantic sectioning with nothing to lean on: chapter headings and no other "
            "structure, in single documents of up to 3.3 MB. les_miserables.txt in "
            "tests/data is a 34 KB excerpt of one of these -- this is the whole novel."
        ),
        "licence": "Public domain (US); files keep the Project Gutenberg boilerplate",
        "home": "https://www.gutenberg.org/policy/permission.html",
        "approx_bytes": 9_000_000,
        "files": [
            _book("Moby-Dick--Or-The-Whale", 2701, "Moby Dick"),
            _book("Pride-and-Prejudice", 1342, "Pride and Prejudice"),
            _book("Les-Mis-rables", 135, "Les Miserables"),
            _book("War-and-Peace", 2600, "War and Peace"),
            _book("Adventures-of-Huckleberry-Finn", 76, "Huckleberry Finn"),
        ],
    },
    "pgessays": {
        "title": "Paul Graham essays",
        "why": (
            "The same unstructured shape at 235 KB, so a full ingest-and-query cycle "
            "finishes while you watch it. The right corpus while changing chunk geometry; "
            "the wrong one for measuring anything."
        ),
        "licence": (
            "Harness MIT; the essays are (c) Paul Graham with no redistribution grant -- "
            "fine as a local fixture, not to republish"
        ),
        "home": "https://github.com/gkamradt/LLMTest_NeedleInAHaystack",
        "approx_bytes": 235_000,
        "files": [
            _essay("worked"),
            _essay("popular"),
            _essay("gap"),
            _essay("gh"),
            _essay("philosophy"),
            _essay("startuplessons"),
        ],
    },
    "multihoprag": {
        "title": "MultiHop-RAG news corpus",
        "why": (
            "609 articles whose published queries need evidence from two to four of them. "
            "Four outlets supply half the corpus and cover the same stories, so the "
            "distractors are real rather than sampled -- which is what relevant segment "
            "extraction has to survive."
        ),
        "licence": "ODC-BY 1.0",
        "home": "https://github.com/yixuantt/MultiHop-RAG",
        "approx_bytes": 12_000_000,
        # Pinned to the commit before dataset/ was removed from the default
        # branch. A main URL 404s today, and a commit is immutable.
        "bundle": (
            "https://media.githubusercontent.com/media/yixuantt/MultiHop-RAG"
            "/3dd4d4e79fc9843008b8f832da99086a82f1a805/dataset/corpus.json"
        ),
        "split": split_multihop_rag,
    },
    "squad": {
        "title": "SQuAD 2.0 Wikipedia articles",
        "why": (
            "442 articles behind 43,498 questions the corpus deliberately does not answer, "
            "written by people looking at the paragraph -- so they are plausible in exactly "
            "the way a guess is. The corpus for testing whether a pipeline declines."
        ),
        "licence": "CC-BY-SA 4.0",
        "home": "https://rajpurkar.github.io/SQuAD-explorer/",
        "approx_bytes": 42_000_000,
        "bundle": (
            "https://raw.githubusercontent.com/rajpurkar/SQuAD-explorer"
            "/master/dataset/train-v2.0.json"
        ),
        "split": split_squad,
    },
    "tatqa": {
        "title": "TAT-QA financial tables",
        "why": (
            "2,201 real annual-report tables with the prose that explains them. A table row "
            "is the least self-describing text in any corpus, which is what makes this the "
            "sharpest test of AutoContext's contextual chunk headers."
        ),
        "licence": "CC-BY 4.0",
        "home": "https://nextplusplus.github.io/TAT-QA/",
        "approx_bytes": 9_000_000,
        "bundle": (
            "https://raw.githubusercontent.com/NExTplusplus/TAT-QA"
            "/master/dataset_raw/tatqa_dataset_train.json"
        ),
        "split": split_tatqa,
    },
}


# --------------------------------------------------------------------------
# Fetching
# --------------------------------------------------------------------------

_NEVER_REACHED_HOST = re.compile(
    r"tunnel|CONNECT|proxy|certificate|Name or service not known|Temporary failure",
    re.IGNORECASE,
)


def explain_failure(message: str) -> str:
    """A blocked CONNECT and a moved file both look like a failed GET.

    They want opposite responses -- one is a network to fix, the other an
    address to correct -- so saying which stops an hour going into the wrong one.
    """
    if _NEVER_REACHED_HOST.search(message):
        return (
            f"the request never reached the host -- {message}. This is your network or "
            "proxy, not the corpus. Download the file and pass --file."
        )
    return message


def _write_atomically(path: str, payload: bytes) -> None:
    """A half-written file that the resume check then SKIPS is permanent."""
    partial = path + ".partial"
    with open(partial, "wb") as handle:
        handle.write(payload)
    os.replace(partial, path)


def _existing_size(path: str) -> int:
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


def _get(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=TIMEOUT_SECONDS) as response:
        return response.read()


def download_file(url: str, path: str) -> str:
    """'written', 'skipped', or a reason. Resumable by skipping what is there."""
    if _existing_size(path) >= MIN_PLAUSIBLE_BYTES:
        return "skipped"
    try:
        payload = _get(url)
    except urllib.error.HTTPError as err:
        return f"HTTP {err.code}"
    except Exception as err:  # noqa: BLE001 - the reason is the whole output
        return explain_failure(str(err))

    if len(payload) < MIN_PLAUSIBLE_BYTES:
        return f"{len(payload)} bytes -- an error page, not a document"
    _write_atomically(path, payload)
    return "written"


def write_documents(documents, destination: str) -> dict:
    """Write split output.

    Names derived from fetched text are a BOUNDARY, not internal code: the name
    came out of a file on the internet. A collision is not a duplicate download
    -- it is the second document overwriting the first, and a corpus quietly one
    document short.
    """
    outcome = {"written": 0, "skipped": 0, "failed": []}
    seen = set()

    for name, text in documents:
        if not is_safe_name(name):
            outcome["failed"].append((name, "a name that cannot be written here"))
            continue
        if name in seen:
            outcome["failed"].append((name, "two documents derived the same name"))
            continue
        seen.add(name)

        path = os.path.join(destination, name + ".txt")
        if _existing_size(path) >= MIN_PLAUSIBLE_BYTES:
            outcome["skipped"] += 1
            continue
        _write_atomically(path, text.encode("utf-8"))
        outcome["written"] += 1
    return outcome


def fetch(corpus_id: str, limit: int, local_file: str = None) -> dict:
    corpus = CORPORA[corpus_id]
    destination = corpus_dir(corpus_id)
    os.makedirs(destination, exist_ok=True)

    print(f"\n> {corpus_id} -- {corpus['title']}")
    print(f"  licence: {corpus['licence']}")

    if "bundle" in corpus:
        # --file is not a convenience. A corporate proxy, an air-gapped machine
        # and a rate-limited host all end with the operator already holding the
        # file, and a fetcher that can only fetch is useless in all three.
        if local_file:
            with open(local_file, encoding="utf-8") as handle:
                contents = handle.read()
        else:
            print(f"  one file -> {destination}, split into documents")
            try:
                contents = _get(corpus["bundle"]).decode("utf-8")
            except Exception as err:  # noqa: BLE001
                return {"written": 0, "skipped": 0, "failed": [(corpus_id, explain_failure(str(err)))]}

        try:
            documents = corpus["split"](contents)
        except (ValueError, json.JSONDecodeError) as err:
            # A moved schema must not read as an empty corpus: an empty corpus
            # retrieves nothing and reports success.
            return {
                "written": 0,
                "skipped": 0,
                "failed": [(corpus_id, f"could not read the published file -- {err}")],
            }

        print(f"  {len(documents)} documents found, writing {min(limit, len(documents))}")
        return write_documents(documents[:limit], destination)

    outcome = {"written": 0, "skipped": 0, "failed": []}
    files = corpus["files"][:limit]
    print(f"  {len(files)} documents -> {destination}")
    for name, url in files:
        result = download_file(url, os.path.join(destination, name + ".txt"))
        if result in ("written", "skipped"):
            outcome[result] += 1
        else:
            outcome["failed"].append((name, result))
        print(f"    {name}: {result}")
    return outcome


def _describe() -> None:
    print("\nAvailable corpora:\n")
    for corpus_id, corpus in CORPORA.items():
        shape = (
            "documents inside one file"
            if "bundle" in corpus
            else f"{len(corpus['files'])} documents"
        )
        print(f"  {corpus_id}")
        print(f"      {corpus['title']} -- {shape}, ~{corpus['approx_bytes'] / 1e6:.1f} MB")
        print(f"      {corpus['why']}")
        print(f"      licence: {corpus['licence']}")
        print("")
    print(f"  Fetched corpora land in {CORPORA_DIR}\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", nargs="?", help="which corpus to fetch")
    parser.add_argument("--all", action="store_true", help="fetch every corpus")
    parser.add_argument("--limit", type=int, default=sys.maxsize, help="at most N documents")
    parser.add_argument("--file", help="read the bundle from disk instead of the network")
    arguments = parser.parse_args()

    if not arguments.corpus and not arguments.all:
        _describe()
        return 0

    wanted = list(CORPORA) if arguments.all else [arguments.corpus]
    unknown = [corpus_id for corpus_id in wanted if corpus_id not in CORPORA]
    if unknown:
        print(f"Unknown corpus {unknown[0]!r}. Available: {', '.join(CORPORA)}", file=sys.stderr)
        return 1

    failures = 0
    for corpus_id in wanted:
        outcome = fetch(corpus_id, arguments.limit, arguments.file)
        print(f"  wrote {outcome['written']}, already had {outcome['skipped']}")
        for name, reason in outcome["failed"][:20]:
            print(f"  FAILED {name}: {reason}")
        failures += len(outcome["failed"])

    # A partial corpus measures the gap rather than the system.
    print("")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
