"""Text-to-Speech engine using Edge-TTS (Microsoft Neural Voices)."""

import io
import logging

import edge_tts

from src import config

logger = logging.getLogger(__name__)


def load():
    """No-op — Edge-TTS has no local models to preload."""
    voice = config.get("tts.voice", "fr-FR-VivienneMultilingualNeural")
    logger.info("TTS ready (edge-tts, voice=%s). No models to preload.", voice)


async def synthesize(text: str, detected_language: str = "fr") -> bytes:
    """Synthesize text to MP3 audio bytes using Edge-TTS.

    Args:
        text: The text to speak.
        detected_language: The detected language code (unused — voice is fixed in config).

    Returns:
        MP3-format audio bytes.
    """
    voice = config.get("tts.voice", "fr-FR-VivienneMultilingualNeural")
    rate = config.get("tts.rate", "+0%")
    volume = config.get("tts.volume", "+0%")
    pitch = config.get("tts.pitch", "+0Hz")

    logger.info("TTS: voice=%s, rate=%s, volume=%s", voice, rate, volume)

    communicate = edge_tts.Communicate(text, voice, rate=rate, volume=volume, pitch=pitch)
    buf = io.BytesIO()

    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            buf.write(chunk["data"])

    mp3_bytes = buf.getvalue()

    if not mp3_bytes:
        logger.warning("TTS produced no audio for text: %s", text[:80])
        return b""

    return mp3_bytes
