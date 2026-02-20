"""Fetch crypto news headlines from RSS feeds (no API key required)."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from urllib.parse import quote

import requests

# Map trading symbols to search terms
SYMBOL_KEYWORDS: dict[str, str] = {
    "BTC/USDT": "Bitcoin",
    "ETH/USDT": "Ethereum",
    "SOL/USDT": "Solana",
    "DOGE/USDT": "Dogecoin",
    "XRP/USDT": "XRP Ripple",
}

_TAG_RE = re.compile(r"<[^>]+>")


def _build_feed_url(keyword: str) -> str:
    q = quote(f"{keyword} crypto")
    return f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"


def fetch_headlines(max_items: int = 10, symbol: str = "BTC/USDT") -> list[dict]:
    keyword = SYMBOL_KEYWORDS.get(symbol, symbol.split("/")[0])
    url = _build_feed_url(keyword)

    headlines: list[dict] = []
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
        pass
    return headlines[:max_items]


if __name__ == "__main__":
    import sys
    symbol = sys.argv[1] if len(sys.argv) > 1 else "BTC/USDT"
    print(f"Headlines for {symbol}:")
    for h in fetch_headlines(5, symbol=symbol):
        print(f"  [{h['published'][:16]}] {h['title']}")
