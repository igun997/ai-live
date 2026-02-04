"""Agent orchestrator — wires STT, LLM, and TTS into a pipeline."""

import logging
from dataclasses import dataclass
from typing import Optional

from src import stt, tts, llm, analysis
from src.session import Session

logger = logging.getLogger(__name__)


@dataclass
class AgentResponse:
    user_text: str
    detected_language: str
    response_text: str
    audio_bytes: bytes


def load_models():
    """Pre-load all models at startup."""
    logger.info("Loading all models...")
    stt.load()
    tts.load()
    llm.load()
    logger.info("All models loaded.")


async def process_audio(audio_bytes: bytes, session: Session) -> Optional[AgentResponse]:
    """Run the full voice agent pipeline.

    1. Transcribe audio (STT)
    2. Generate response (LLM)
    3. Synthesize speech (TTS)

    Args:
        audio_bytes: WAV-format audio from the user's microphone.
        session: The current conversation session.

    Returns:
        AgentResponse with all results, or None if transcription was empty.
    """
    # Step 1: Speech-to-Text
    user_text, detected_lang = stt.transcribe(audio_bytes)

    if not user_text.strip():
        logger.info("Empty transcription, skipping.")
        return None

    # Record user turn
    session.add_turn("user", user_text, detected_lang)

    # Step 2: LLM response
    response_text = llm.generate_response(
        user_text=user_text,
        conversation_history=session.get_history()[:-1],  # exclude current turn
        detected_language=detected_lang,
    )

    # Always French
    tts_lang = "fr"

    # Record assistant turn
    session.add_turn("assistant", response_text, tts_lang)

    # Step 3: Text-to-Speech
    audio = await tts.synthesize(response_text, tts_lang)

    return AgentResponse(
        user_text=user_text,
        detected_language=detected_lang,
        response_text=response_text,
        audio_bytes=audio,
    )


def end_session(session: Session) -> dict:
    """End a session and produce summary + sentiment analysis.

    Returns:
        Analysis dict with summary, sentiment, turn_count.
    """
    session.end()
    history = session.get_history()
    result = analysis.analyze_conversation(history)
    session.summary = result.get("summary")
    session.sentiment = result.get("sentiment")
    return result
