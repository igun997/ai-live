import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

load_dotenv()

_BASE_DIR = Path(__file__).resolve().parent.parent
_CONFIG_PATH = _BASE_DIR / "config" / "agent.yaml"

_config: dict | None = None


def _load() -> dict:
    global _config
    if _config is None:
        with open(_CONFIG_PATH) as f:
            _config = yaml.safe_load(f)
    return _config


def get(key: str, default: Any = None) -> Any:
    """Get a dot-separated config key, e.g. 'agent.name'."""
    data = _load()
    for part in key.split("."):
        if isinstance(data, dict):
            data = data.get(part)
        else:
            return default
        if data is None:
            return default
    return data


def build_system_prompt() -> str:
    """Build the full system prompt from the agent config template."""
    cfg = _load()
    agent = cfg["agent"]
    tpl = agent["system_prompt"]

    traits_str = ", ".join(agent["personality"]["traits"])
    allowed = ", ".join(agent["boundaries"]["allowed_topics"])
    forbidden = ", ".join(agent["boundaries"]["forbidden_topics"])
    knowledge = agent["knowledge"]["base"]

    return tpl.format(
        name=agent["name"],
        description=agent["personality"]["description"],
        traits=traits_str,
        tone=agent["personality"]["tone"],
        knowledge=knowledge,
        allowed_topics=allowed,
        forbidden_topics=forbidden,
        redirect_message=agent["boundaries"]["redirect_message"],
        max_words=agent["boundaries"]["max_response_words"],
    )


def gemini_api_key() -> str:
    key = os.environ.get("GEMINI_API_KEY", "")
    if not key:
        raise RuntimeError(
            "GEMINI_API_KEY not set. Copy .env.example to .env and add your key."
        )
    return key
