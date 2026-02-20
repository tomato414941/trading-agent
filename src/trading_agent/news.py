"""Fetch crypto news headlines from multiple RSS feeds (no API key required)."""

from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
from urllib.parse import quote

import requests

log = logging.getLogger(__name__)

# Map trading symbols to search terms / tags
SYMBOL_KEYWORDS: dict[str, str] = {
    "BTC/USDT": "Bitcoin",
    "ETH/USDT": "Ethereum",
    "SOL/USDT": "Solana",
    "DOGE/USDT": "Dogecoin",
    "XRP/USDT": "XRP Ripple",
}

SYMBOL_CT_TAGS: dict[str, str] = {
    "BTC/USDT": "bitcoin",
    "ETH/USDT": "ethereum",
    "SOL/USDT": "solana",
    "DOGE/USDT": "dogecoin",
    "XRP/USDT": "xrp",
}

_TAG_RE = re.compile(r"<[^>]+>")
_TIMEOUT = 10


def _parse_rss(content: bytes, max_items: int) -> list[dict]:
    """Parse RSS XML and return headline dicts."""
    items: list[dict] = []
    try:
        root = ET.fromstring(content)
        for item in root.iter("item"):
            title = item.findtext("title", "").strip()
            if not title:
                continue
            pub_date = item.findtext("pubDate", "")
            desc = _TAG_RE.sub("", item.findtext("description", ""))
            items.append({
                "title": title,
                "description": desc[:200],
                "published": pub_date,
            })
            if len(items) >= max_items:
                break
    except ET.ParseError:
        pass
    return items


def _fetch_feed(url: str, max_items: int) -> list[dict]:
    """Fetch a single RSS feed and return parsed headlines."""
    try:
        resp = requests.get(url, timeout=_TIMEOUT, headers={
            "User-Agent": "TradingAgent/1.0",
        })
        resp.raise_for_status()
        return _parse_rss(resp.content, max_items)
    except Exception as e:
        log.debug("Feed fetch failed (%s): %s", url[:60], e)
        return []


def _google_news_url(keyword: str) -> str:
    q = quote(f"{keyword} crypto")
    return f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"


def _cointelegraph_url(symbol: str) -> str:
    tag = SYMBOL_CT_TAGS.get(symbol, symbol.split("/")[0].lower())
    return f"https://cointelegraph.com/rss/tag/{tag}"


COINDESK_MARKETS = "https://www.coindesk.com/arc/outboundfeeds/rss/?outputType=xml&category=markets"
DECRYPT_FEED = "https://decrypt.co/feed"


def fetch_headlines(max_items: int = 10, symbol: str = "BTC/USDT") -> list[dict]:
    """Fetch headlines from multiple sources, deduplicate, return up to max_items."""
    keyword = SYMBOL_KEYWORDS.get(symbol, symbol.split("/")[0])

    feeds = [
        _google_news_url(keyword),
        _cointelegraph_url(symbol),
        COINDESK_MARKETS,
        DECRYPT_FEED,
    ]

    # Collect from all sources (5 per feed to stay balanced)
    per_feed = max(3, max_items // len(feeds) + 1)
    all_headlines: list[dict] = []
    seen_titles: set[str] = set()

    for url in feeds:
        for h in _fetch_feed(url, per_feed):
            # Deduplicate by normalized title prefix
            key = h["title"][:50].lower().strip()
            if key not in seen_titles:
                seen_titles.add(key)
                all_headlines.append(h)

    # Filter: for non-BTC symbols, prefer headlines mentioning the keyword
    if symbol != "BTC/USDT":
        kw_lower = keyword.lower()
        relevant = [h for h in all_headlines if kw_lower in h["title"].lower()]
        # Fill with general crypto headlines if not enough
        if len(relevant) < max_items:
            for h in all_headlines:
                if h not in relevant:
                    relevant.append(h)
                    if len(relevant) >= max_items:
                        break
        all_headlines = relevant

    return all_headlines[:max_items]


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.DEBUG)
    symbol = sys.argv[1] if len(sys.argv) > 1 else "BTC/USDT"
    headlines = fetch_headlines(10, symbol=symbol)
    print(f"Headlines for {symbol} ({len(headlines)} items):")
    for h in headlines:
        print(f"  [{h['published'][:16]}] {h['title'][:80]}")
