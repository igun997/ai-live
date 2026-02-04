"""Speech-to-Text engine using faster-whisper."""

import io
import logging
from typing import Tuple

import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel

from src import config

logger = logging.getLogger(__name__)

_model: WhisperModel | None = None


def _get_model() -> WhisperModel:
    global _model
    if _model is None:
        model_size = config.get("stt.model_size", "small")
        device = config.get("stt.device", "cuda")
        compute_type = config.get("stt.compute_type", "float16")
        logger.info("Loading Whisper model: %s on %s (%s)", model_size, device, compute_type)
        _model = WhisperModel(model_size, device=device, compute_type=compute_type)
        logger.info("Whisper model loaded.")
    return _model


def load():
    """Pre-load the Whisper model at startup."""
    _get_model()


def transcribe(audio_bytes: bytes) -> Tuple[str, str]:
    """Transcribe audio bytes to text.

    Args:
        audio_bytes: WAV-format audio data.

    Returns:
        (transcribed_text, detected_language_code)
        Language codes: 'en', 'fr', 'id', etc.
    """
    model = _get_model()

    buf = io.BytesIO(audio_bytes)
    audio_data, sample_rate = sf.read(buf, dtype="float32")

    # Convert stereo to mono if needed
    if audio_data.ndim > 1:
        audio_data = audio_data.mean(axis=1)

    # Resample to 16kHz if needed (Whisper expects 16kHz)
    if sample_rate != 16000:
        from scipy import signal
        num_samples = int(len(audio_data) * 16000 / sample_rate)
        audio_data = signal.resample(audio_data, num_samples)

    # First attempt with VAD filter (lower threshold for browser audio)
    segments, info = model.transcribe(
        audio_data,
        beam_size=5,
        vad_filter=True,
        vad_parameters=dict(
            threshold=0.3,
            min_silence_duration_ms=600,
            min_speech_duration_ms=100,
            speech_pad_ms=400,
        ),
    )

    text_parts = []
    for segment in segments:
        text_parts.append(segment.text.strip())

    full_text = " ".join(text_parts).strip()

    # If VAD filtered everything out, retry without VAD
    if not full_text:
        logger.info("VAD filtered all audio, retrying without VAD filter")
        segments, info = model.transcribe(audio_data, beam_size=5, vad_filter=False)
        text_parts = []
        for segment in segments:
            text_parts.append(segment.text.strip())
        full_text = " ".join(text_parts).strip()

    detected_lang = info.language or "en"

    logger.info("STT: [%s] %s", detected_lang, full_text)
    return full_text, detected_lang
