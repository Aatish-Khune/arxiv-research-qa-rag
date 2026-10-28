"""Fetch recent cs.LG paper metadata from the arXiv API and save a clean corpus.

Only descriptive metadata (title, abstract, authors, dates, categories) is fetched, which the
arXiv API Terms of Use place under CC0. No PDFs or full text.
"""

from __future__ import annotations

import time
import xml.etree.ElementTree as ET

import pandas as pd
import requests

from paperqa.config import (
    ARTIFACTS_DIR,
    ARXIV_API,
    ARXIV_CATEGORY,
    DATA_DIR,
    N_PAPERS,
    PAGE_SIZE,
    REQUEST_DELAY_S,
)
from paperqa.text import clean_text

NS = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
USER_AGENT = "arxiv-research-qa-rag/0.1 (student portfolio project; polite rate-limited client)"


def parse_feed(xml_text: str) -> list[dict]:
    """Parse one arXiv Atom response into a list of paper dicts."""
    root = ET.fromstring(xml_text)
    papers = []
    for entry in root.findall("a:entry", NS):
        raw_id = entry.findtext("a:id", default="", namespaces=NS)
        arxiv_id = raw_id.rsplit("/abs/", 1)[-1]
        base_id = arxiv_id.rsplit("v", 1)[0] if "v" in arxiv_id else arxiv_id
        primary = entry.find("arxiv:primary_category", NS)
        papers.append(
            {
                "arxiv_id": base_id,
                "title": clean_text(entry.findtext("a:title", default="", namespaces=NS)),
                "abstract": clean_text(entry.findtext("a:summary", default="", namespaces=NS)),
                "authors": ", ".join(
                    a.findtext("a:name", default="", namespaces=NS)
                    for a in entry.findall("a:author", NS)
                ),
                "published": entry.findtext("a:published", default="", namespaces=NS)[:10],
                "primary_category": primary.get("term") if primary is not None else "",
                "categories": " ".join(
                    c.get("term", "") for c in entry.findall("a:category", NS)
                ),
            }
        )
    return papers


def fetch_page(start: int, session: requests.Session, retries: int = 4) -> str:
    """Fetch one page of results, retrying politely on errors or empty pages."""
    params = {
        "search_query": f"cat:{ARXIV_CATEGORY}",
        "sortBy": "submittedDate",
        "sortOrder": "descending",
        "start": start,
        "max_results": PAGE_SIZE,
    }
    for attempt in range(retries):
        resp = session.get(ARXIV_API, params=params, timeout=120)
        if resp.ok and "<entry>" in resp.text:
            return resp.text
        time.sleep(REQUEST_DELAY_S * (attempt + 2))  # back off; the API sometimes returns empty
    raise RuntimeError(f"arXiv API returned no entries for start={start} after {retries} tries")


def fetch_corpus(n_papers: int = N_PAPERS) -> pd.DataFrame:
    """Download `n_papers` most recent cs.LG papers (raw pages cached in data/)."""
    raw_dir = DATA_DIR / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    papers: list[dict] = []
    for start in range(0, n_papers, PAGE_SIZE):
        cache = raw_dir / f"page_{start:05d}.xml"
        if cache.exists():
            xml_text = cache.read_text(encoding="utf-8")
        else:
            xml_text = fetch_page(start, session)
            cache.write_text(xml_text, encoding="utf-8")
            time.sleep(REQUEST_DELAY_S)
        page = parse_feed(xml_text)
        papers += page
        print(f"start={start:>5}: {len(page)} papers")
    df = pd.DataFrame(papers).drop_duplicates("arxiv_id").reset_index(drop=True)
    df = df[(df["abstract"].str.len() > 0) & (df["title"].str.len() > 0)]
    return df.reset_index(drop=True)


def main() -> None:
    df = fetch_corpus()
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(ARTIFACTS_DIR / "papers.parquet", index=False)
    print(
        f"Saved {len(df):,} papers ({df['published'].min()} to {df['published'].max()}) "
        f"to {ARTIFACTS_DIR / 'papers.parquet'}"
    )


if __name__ == "__main__":
    main()
