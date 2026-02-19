"""Fetch crypto news headlines from RSS feeds (no API key required)."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET

import requests


FEEDS = [
    # Google News — Bitcoin
    "https://news.google.com/rss/search?q=Bitcoin+crypto&hl=en-US&gl=US&ceid=US:en",
]

# Strip HTML tags from descriptions
_TAG_RE = re.compile(r"<[^>]+>")


def fetch_headlines(max_items: int = 10) -> list[dict]:
    headlines: list[dict] = []
    for url in FEEDS:
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            root = ET.fromstring(resp.content)
            for item in root.iter("item"):
                title = item.findtext("title", "")
                pub_date = item.findtext("pubDate", "")
                desc = _TAG_RE.sub("", item.findtext("description", ""))
                headlines.append({
                    "title": title,
                    "description": desc[:200],
                    "published": pub_date,
                })
                if len(headlines) >= max_items:
                    break
        except Exception:
            continue
    return headlines[:max_items]


if __name__ == "__main__":
    for h in fetch_headlines(5):
        print(f"  [{h['published'][:16]}] {h['title']}")
