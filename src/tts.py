"""Text-to-Speech engine using Kokoro TTS."""

import io
import logging
from typing import Dict

import numpy as np
import soundfile as sf

from src import config

logger = logging.getLogger(__name__)

# Cache pipelines per language code
_pipelines: Dict[str, object] = {}


def _get_pipeline(lang_code: str):
    """Get or create a Kokoro KPipeline for the given language code."""
    if lang_code not in _pipelines:
        from kokoro import KPipeline

        logger.info("Initializing Kokoro TTS pipeline for lang_code='%s'", lang_code)
        _pipelines[lang_code] = KPipeline(lang_code=lang_code)
        logger.info("Kokoro TTS pipeline ready for lang_code='%s'", lang_code)
    return _pipelines[lang_code]


def load():
    """Pre-load TTS pipelines for all configured languages at startup."""
    supported = config.get("languages.supported", {})
    for lang_key, lang_cfg in supported.items():
        lang_code = lang_cfg.get("tts_lang_code", "a")
        _get_pipeline(lang_code)


def _resolve_voice(detected_language: str) -> tuple[str, str]:
    """Resolve the TTS voice and lang_code from the detected language.

    Returns:
        (voice_name, lang_code)
    """
    supported = config.get("languages.supported", {})

    # Map common Whisper language codes to our config keys
    lang_map = {
        "en": "en",
        "fr": "fr",
        "id": "en",  # Indonesian -> respond in English
    }

    lang_key = lang_map.get(detected_language, config.get("languages.default", "en"))
    lang_cfg = supported.get(lang_key, supported.get("en", {}))

    voice = lang_cfg.get("tts_voice", "af_heart")
    lang_code = lang_cfg.get("tts_lang_code", "a")

    return voice, lang_code


def synthesize(text: str, detected_language: str = "en") -> bytes:
    """Synthesize text to WAV audio bytes.

    Args:
        text: The text to speak.
        detected_language: The detected language code from STT (e.g. 'en', 'fr').

    Returns:
        WAV-format audio bytes at 24kHz.
    """
    voice, lang_code = _resolve_voice(detected_language)
    pipeline = _get_pipeline(lang_code)
    speed = config.get("tts.speed", 1.0)
    sample_rate = config.get("tts.sample_rate", 24000)

    logger.info("TTS: voice=%s, lang_code=%s, speed=%s", voice, lang_code, speed)

    # Kokoro KPipeline returns a generator of (graphemes, phonemes, audio) tuples
    audio_segments = []
    for _gs, _ps, audio in pipeline(text, voice=voice, speed=speed):
        if audio is not None:
            if not isinstance(audio, np.ndarray):
                audio = np.array(audio, dtype=np.float32)
            audio_segments.append(audio)

    if not audio_segments:
        logger.warning("TTS produced no audio for text: %s", text[:80])
        # Return silence (0.5s)
        silence = np.zeros(int(sample_rate * 0.5), dtype=np.float32)
        audio_segments = [silence]

    full_audio = np.concatenate(audio_segments)

    # Convert to WAV bytes
    buf = io.BytesIO()
    sf.write(buf, full_audio, sample_rate, format="WAV", subtype="PCM_16")
    buf.seek(0)
    return buf.read()
