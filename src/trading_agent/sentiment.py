"""LLM-based sentiment analysis using Claude API."""

from __future__ import annotations

import json
import logging
import os

from anthropic import Anthropic

from trading_agent.news import fetch_headlines

log = logging.getLogger(__name__)

SENTIMENT_PROMPT = """\
You are a crypto market sentiment analyst.
Given the following news headlines about Bitcoin/crypto, rate the overall market sentiment.

Headlines:
{headlines}

Respond with ONLY a JSON object (no markdown, no explanation):
{{"score": <float from -1.0 to 1.0>, "summary": "<one sentence>"}}

Where:
- score -1.0 = extremely bearish
- score  0.0 = neutral
- score +1.0 = extremely bullish
"""


def _get_client() -> Anthropic:
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        secrets = os.path.expanduser("~/.secrets/anthropic")
        if os.path.exists(secrets):
            with open(secrets) as f:
                for line in f:
                    if "ANTHROPIC_API_KEY" in line:
                        # handle: export ANTHROPIC_API_KEY=...
                        val = line.strip().split("=", 1)[-1].strip().strip("'\"")
                        key = val
                        break
    return Anthropic(api_key=key)


def analyze_sentiment(max_headlines: int = 8) -> dict:
    """Fetch news and return sentiment score + summary.

    Returns:
        {"score": float, "summary": str, "headline_count": int}
        score is clamped to [-1.0, 1.0]
        Returns score=0.0 on any failure (safe fallback).
    """
    headlines = fetch_headlines(max_headlines)
    if not headlines:
        log.warning("No headlines fetched, returning neutral sentiment")
        return {"score": 0.0, "summary": "No news available", "headline_count": 0}

    headline_text = "\n".join(
        f"- {h['title']}" for h in headlines
    )

    try:
        client = _get_client()
        resp = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=150,
            messages=[{
                "role": "user",
                "content": SENTIMENT_PROMPT.format(headlines=headline_text),
            }],
        )
        raw = resp.content[0].text.strip()
    except Exception as e:
        log.warning("LLM API call failed: %s", e)
        return {"score": 0.0, "summary": f"API error: {e}", "headline_count": len(headlines)}

    try:
        data = json.loads(raw)
        score = max(-1.0, min(1.0, float(data["score"])))
        summary = data.get("summary", "")
    except (json.JSONDecodeError, KeyError, ValueError):
        log.warning("Failed to parse LLM response: %s", raw)
        return {"score": 0.0, "summary": raw[:100], "headline_count": len(headlines)}

    return {"score": score, "summary": summary, "headline_count": len(headlines)}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = analyze_sentiment()
    print(f"Score:     {result['score']:+.2f}")
    print(f"Summary:   {result['summary']}")
    print(f"Headlines: {result['headline_count']}")
