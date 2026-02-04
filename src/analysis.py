"""End-of-conversation analysis: summary and sentiment."""

import json
import logging
from typing import Dict, List

import google.generativeai as genai

from src import config

logger = logging.getLogger(__name__)


def analyze_conversation(history: List[Dict[str, str]]) -> Dict:
    """Generate summary and sentiment analysis for a conversation.

    Args:
        history: List of {"role": "user"|"assistant", "text": "..."}.

    Returns:
        {
            "summary": "...",
            "sentiment": {
                "overall": "positive|negative|neutral|mixed",
                "score": 0.0-1.0,
                "details": "..."
            },
            "turn_count": int,
            "languages_used": [...]
        }
    """
    if not history:
        return {
            "summary": "No conversation took place.",
            "sentiment": {"overall": "neutral", "score": 0.5, "details": "Empty conversation."},
            "turn_count": 0,
            "languages_used": [],
        }

    genai.configure(api_key=config.gemini_api_key())
    model = genai.GenerativeModel(config.get("llm.model", "gemini-2.5-flash"))

    # Format the conversation transcript
    transcript_lines = []
    for turn in history:
        role_label = "User" if turn["role"] == "user" else "Assistant"
        transcript_lines.append(f"{role_label}: {turn['text']}")
    transcript = "\n".join(transcript_lines)

    prompt = f"""Analyze the following conversation transcript and return a JSON object with:
1. "summary": A concise 2-3 sentence summary of what was discussed.
2. "sentiment": An object with:
   - "overall": one of "positive", "negative", "neutral", "mixed"
   - "score": a float from 0.0 (very negative) to 1.0 (very positive)
   - "details": a brief explanation of the sentiment

Return ONLY valid JSON, no markdown formatting.

TRANSCRIPT:
{transcript}"""

    try:
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.3,
                max_output_tokens=512,
            ),
        )

        raw = response.text.strip()
        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
        if raw.endswith("```"):
            raw = raw[:-3]
        raw = raw.strip()
        if raw.startswith("json"):
            raw = raw[4:].strip()

        result = json.loads(raw)
        result["turn_count"] = len(history)
        logger.info("Analysis complete: %s", result.get("sentiment", {}).get("overall"))
        return result

    except (json.JSONDecodeError, Exception) as e:
        logger.error("Analysis failed: %s", e)
        return {
            "summary": "Analysis could not be completed.",
            "sentiment": {"overall": "unknown", "score": 0.5, "details": str(e)},
            "turn_count": len(history),
            "languages_used": [],
        }
