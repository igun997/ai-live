"""Text-to-Speech engine using Gemini TTS (gemini-2.5-flash-preview-tts)."""

import io
import logging
import wave

from google import genai
from google.genai import types

from src import config

logger = logging.getLogger(__name__)

_client: genai.Client | None = None


def _get_client() -> genai.Client:
    """Lazy-init the Gemini genai client."""
    global _client
    if _client is None:
        _client = genai.Client(api_key=config.gemini_api_key())
    return _client


def _pcm_to_wav(pcm_data: bytes, sample_rate: int = 24000,
                num_channels: int = 1, sample_width: int = 2) -> bytes:
    """Wrap raw PCM bytes in a WAV header."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(num_channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_data)
    return buf.getvalue()


def load():
    """Initialise the Gemini TTS client and log readiness."""
    _get_client()
    model = config.get("tts.model", "gemini-2.5-flash-preview-tts")
    voice = config.get("tts.voice_name", "Kore")
    logger.info("TTS ready (Gemini %s, voice=%s).", model, voice)


async def synthesize(text: str, detected_language: str = "fr") -> bytes:
    """Synthesize text to WAV audio bytes using Gemini TTS.

    Args:
        text: The text to speak.
        detected_language: The detected language code (unused — voice is fixed in config).

    Returns:
        WAV-format audio bytes (PCM 24 kHz mono 16-bit).
    """
    client = _get_client()
    model = config.get("tts.model", "gemini-2.5-flash-preview-tts")
    voice_name = config.get("tts.voice_name", "Kore")
    sample_rate = config.get("tts.sample_rate", 24000)

    logger.info("TTS: model=%s, voice=%s, lang=%s", model, voice_name, detected_language)

    response = await client.aio.models.generate_content(
        model=model,
        contents=f"Say: {text}",
        config=types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=voice_name,
                    ),
                ),
            ),
        ),
    )

    if (not response.candidates
            or not response.candidates[0].content
            or not response.candidates[0].content.parts):
        logger.warning("TTS produced no audio for text: %s", text[:80])
        return b""

    part = response.candidates[0].content.parts[0]
    pcm_data = part.inline_data.data
    if not pcm_data:
        logger.warning("TTS returned empty audio data for text: %s", text[:80])
        return b""

    wav_bytes = _pcm_to_wav(pcm_data, sample_rate=sample_rate)
    return wav_bytes
