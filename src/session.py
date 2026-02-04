"""Conversation session manager."""

import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Turn:
    role: str  # "user" or "assistant"
    text: str
    language: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class Session:
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    turns: List[Turn] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    ended: bool = False
    summary: Optional[str] = None
    sentiment: Optional[dict] = None

    def add_turn(self, role: str, text: str, language: str):
        self.turns.append(Turn(role=role, text=text, language=language))

    def get_history(self) -> List[Dict[str, str]]:
        """Return history in the format expected by the LLM."""
        return [{"role": t.role, "text": t.text} for t in self.turns]

    def last_activity(self) -> float:
        if self.turns:
            return self.turns[-1].timestamp
        return self.created_at

    def end(self):
        self.ended = True


# In-memory session store
_sessions: Dict[str, Session] = {}


def create_session() -> Session:
    session = Session()
    _sessions[session.session_id] = session
    return session


def get_session(session_id: str) -> Optional[Session]:
    return _sessions.get(session_id)


def remove_session(session_id: str):
    _sessions.pop(session_id, None)
