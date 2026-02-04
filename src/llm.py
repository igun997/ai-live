"""LLM client using Google Gemini 2.5 Flash."""

import logging
from typing import List, Dict

import google.generativeai as genai

from src import config

logger = logging.getLogger(__name__)

_model = None


def _get_model():
    global _model
    if _model is None:
        genai.configure(api_key=config.gemini_api_key())
        model_name = config.get("llm.model", "gemini-2.5-flash")
        logger.info("Initializing Gemini model: %s", model_name)
        _model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=config.build_system_prompt(),
        )
        logger.info("Gemini model ready.")
    return _model


def load():
    """Pre-initialize the Gemini model."""
    _get_model()


def generate_response(
    user_text: str,
    conversation_history: List[Dict[str, str]],
    detected_language: str = "en",
) -> str:
    """Generate a response from the LLM.

    Args:
        user_text: The user's transcribed speech.
        conversation_history: List of {"role": "user"|"assistant", "text": "..."}.
        detected_language: Detected language code from STT.

    Returns:
        The model's response text.
    """
    model = _get_model()
    temperature = config.get("llm.temperature", 0.7)
    max_tokens = config.get("llm.max_output_tokens", 512)

    # Build Gemini conversation history
    gemini_history = []
    for turn in conversation_history:
        role = "user" if turn["role"] == "user" else "model"
        gemini_history.append({"role": role, "parts": [turn["text"]]})

    chat = model.start_chat(history=gemini_history)

    # Add language hint to guide response language
    lang_hints = {
        "fr": "(The user is speaking French. Respond in French.)\n",
        "id": "(The user is speaking Indonesian. Respond in English.)\n",
    }
    prefix = lang_hints.get(detected_language, "")

    response = chat.send_message(
        f"{prefix}{user_text}",
        generation_config=genai.types.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        ),
    )

    reply = response.text.strip()
    logger.info("LLM response [%s]: %s", detected_language, reply[:120])
    return reply
