"""
scripts/fetch_regulations.py

One-time fetcher for the source regulation texts used by the ingestion
pipeline. GDPR, CCPA and HIPAA text changes on the order of years, not
days, so this is meant to be run manually/occasionally rather than on
a schedule - re-run it if a regulation is amended and you want to
refresh data/regulations/.

Sources (official / canonical):
  GDPR  -> gdpr-info.eu (article-by-article mirror of Regulation 2016/679;
           EUR-Lex itself is JS-rendered and blocks simple scraping)
  CCPA  -> leginfo.legislature.ca.gov (California Civil Code Title 1.81.5)
  HIPAA -> ecfr.gov versioner API (45 CFR Part 164, official XML)

Usage:
  python scripts/fetch_regulations.py
  python scripts/fetch_regulations.py --only gdpr,hipaa
"""

import argparse
import re
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import requests
from bs4 import BeautifulSoup

OUT_DIR = Path(__file__).parent.parent / "data" / "regulations"
HEADERS = {"User-Agent": "Mozilla/5.0 (graphrag-privacy research script)"}
TIMEOUT = 20


def fetch_gdpr(max_articles: int = 99) -> str:
    """Fetch all GDPR articles from gdpr-info.eu into one text blob."""
    blocks = []
    for n in range(1, max_articles + 1):
        url = f"https://gdpr-info.eu/art-{n}-gdpr/"
        resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        if resp.status_code != 200:
            print(f"  ! skipping Article {n}: HTTP {resp.status_code}")
            continue

        soup = BeautifulSoup(resp.text, "html.parser")
        title_el = soup.select_one("h1.entry-title")
        body_el = soup.select_one("div.entry-content")
        if not title_el or not body_el:
            print(f"  ! skipping Article {n}: missing title/body markup")
            continue

        title = title_el.select_one("span.dsgvo-title")
        title_text = title.get_text(strip=True) if title else ""

        # Join with spaces, not newlines: get_text(separator="\n") inserts
        # a newline between *every* text-node boundary, including around
        # inline <a> cross-reference links, splitting "Article 6(1)" into
        # "Article 6\n(1)" - which then looks like a real article header to
        # the downstream regex. Space-joining keeps the whole body newline-free.
        body_text = body_el.get_text(separator=" ", strip=True)

        blocks.append(f"Article {n}\n{title_text}\n\n{body_text}\n")
        print(f"  Article {n}: {title_text[:60]}")
        time.sleep(0.2)  # be polite to the mirror

    return "\n\n".join(blocks)


CCPA_URL = (
    "https://leginfo.legislature.ca.gov/faces/codes_displayText.xhtml"
    "?division=3.&part=4.&lawCode=CIV&title=1.81.5"
)
CCPA_SECTION_RE = re.compile(r"^1798\.1\d{2}(\.\d+)?\.?$")


def fetch_ccpa() -> str:
    """Fetch CCPA (Civil Code Title 1.81.5) sections 1798.100-1798.199.100."""
    resp = requests.get(CCPA_URL, headers=HEADERS, timeout=TIMEOUT)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    blocks = []
    headers = soup.select("h6")
    for i, h6 in enumerate(headers):
        link = h6.select_one("a")
        if not link:
            continue
        section_num = link.get_text(strip=True).rstrip(".")
        if not CCPA_SECTION_RE.match(section_num + "."):
            continue

        # Title is the next inline <p> sibling right after the <h6>'s parent tag
        parent = h6.parent
        title_el = parent.select_one("p")
        title_text = title_el.get_text(strip=True) if title_el else ""

        # Body = all text in the parent container after the title paragraph
        parent_text = parent.get_text(separator="\n", strip=True)
        body_text = parent_text
        if title_text and parent_text.startswith(title_text):
            body_text = parent_text[len(title_text):].strip()

        blocks.append(f"Section {section_num} {title_text}\n\n{body_text}\n")
        print(f"  Section {section_num}: {title_text[:60]}")

    return "\n\n".join(blocks)


def _get_latest_ecfr_date(title_number: int = 45) -> str:
    resp = requests.get(
        "https://www.ecfr.gov/api/versioner/v1/titles.json",
        headers=HEADERS, timeout=TIMEOUT,
    )
    resp.raise_for_status()
    for t in resp.json().get("titles", []):
        if t.get("number") == title_number:
            return t["up_to_date_as_of"]
    raise RuntimeError(f"Title {title_number} not found in eCFR titles list")


def fetch_hipaa(part: int = 164) -> str:
    """Fetch 45 CFR Part 164 (Security and Privacy) from the eCFR API."""
    as_of = _get_latest_ecfr_date()
    print(f"  Using eCFR snapshot date: {as_of}")

    url = f"https://www.ecfr.gov/api/versioner/v1/full/{as_of}/title-45.xml"
    resp = requests.get(url, params={"part": part}, headers=HEADERS, timeout=TIMEOUT)
    resp.raise_for_status()

    root = ET.fromstring(resp.content)
    blocks = []
    for section in root.iter("DIV8"):
        if section.get("TYPE") != "SECTION":
            continue
        head = section.find("HEAD")
        head_text = (head.text or "").strip() if head is not None else ""
        if not head_text:
            continue

        body_parts = []
        for el in section.iter():
            if el is section or el.tag == "HEAD":
                continue
            if el.text and el.text.strip():
                body_parts.append(el.text.strip())
            if el.tail and el.tail.strip():
                body_parts.append(el.tail.strip())
        body_text = "\n".join(body_parts)

        blocks.append(f"{head_text}\n\n{body_text}\n")
        print(f"  {head_text[:70]}")

    return "\n\n".join(blocks)


FETCHERS = {
    "gdpr": fetch_gdpr,
    "ccpa": fetch_ccpa,
    "hipaa": fetch_hipaa,
}


def main():
    parser = argparse.ArgumentParser(description="Fetch privacy regulation source texts")
    parser.add_argument(
        "--only", type=str, default=None,
        help="Comma-separated subset to fetch, e.g. gdpr,hipaa",
    )
    args = parser.parse_args()

    targets = list(FETCHERS.keys())
    if args.only:
        targets = [t.strip().lower() for t in args.only.split(",")]
        unknown = set(targets) - set(FETCHERS.keys())
        if unknown:
            print(f"Unknown regulation(s): {unknown}. Choices: {list(FETCHERS.keys())}")
            sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for name in targets:
        print(f"\n{'='*60}\nFetching {name.upper()}\n{'='*60}")
        try:
            text = FETCHERS[name]()
        except Exception as e:
            print(f"FAILED to fetch {name}: {e}")
            continue

        if not text.strip():
            print(f"FAILED: no content extracted for {name}, not writing file")
            continue

        out_path = OUT_DIR / f"{name}.txt"
        out_path.write_text(text, encoding="utf-8")
        print(f"\n-> wrote {out_path} ({len(text):,} chars)")


if __name__ == "__main__":
    main()
