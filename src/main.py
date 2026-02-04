"""FastAPI application — serves the web UI and WebSocket audio endpoint."""

import io
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydub import AudioSegment

from src import agent
from src.session import Session, create_session, get_session

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup, cleanup on shutdown."""
    logger.info("Starting up — loading models...")
    agent.load_models()
    logger.info("Models loaded. Server ready.")
    yield
    logger.info("Shutting down.")


app = FastAPI(title="Live Voice AI Agent", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
async def index():
    index_file = STATIC_DIR / "index.html"
    return HTMLResponse(content=index_file.read_text())


def _webm_to_wav(webm_bytes: bytes) -> bytes:
    """Convert WebM/Opus audio from the browser to WAV for Whisper."""
    audio = AudioSegment.from_file(io.BytesIO(webm_bytes), format="webm")
    audio = audio.set_channels(1).set_frame_rate(16000).set_sample_width(2)
    buf = io.BytesIO()
    audio.export(buf, format="wav")
    buf.seek(0)
    return buf.read()


@app.websocket("/ws/audio")
async def websocket_audio(ws: WebSocket):
    await ws.accept()
    session: Session = create_session()
    logger.info("WebSocket connected — session %s", session.session_id)

    await ws.send_json({
        "type": "session_start",
        "session_id": session.session_id,
    })

    try:
        while True:
            message = await ws.receive()

            # Handle text messages (JSON commands)
            if "text" in message:
                data = json.loads(message["text"])
                msg_type = data.get("type")

                if msg_type == "end_session":
                    logger.info("End session requested — %s", session.session_id)
                    result = agent.end_session(session)
                    await ws.send_json({"type": "summary", **result})
                    break

                if msg_type == "ping":
                    await ws.send_json({"type": "pong"})
                    continue

            # Handle binary messages (audio data)
            if "bytes" in message:
                audio_webm = message["bytes"]
                if not audio_webm:
                    continue

                try:
                    # Convert WebM from browser to WAV
                    audio_wav = _webm_to_wav(audio_webm)
                except Exception as e:
                    logger.error("Audio conversion failed: %s", e)
                    await ws.send_json({
                        "type": "error",
                        "message": "Audio format conversion failed.",
                    })
                    continue

                # Run agent pipeline
                try:
                    result = agent.process_audio(audio_wav, session)
                except Exception as e:
                    logger.error("Agent pipeline error: %s", e)
                    await ws.send_json({
                        "type": "error",
                        "message": "Processing error. Please try again.",
                    })
                    continue

                if result is None:
                    await ws.send_json({
                        "type": "transcription",
                        "text": "",
                        "language": "",
                    })
                    continue

                # Send transcription
                await ws.send_json({
                    "type": "transcription",
                    "text": result.user_text,
                    "language": result.detected_language,
                })

                # Send response text
                await ws.send_json({
                    "type": "response",
                    "text": result.response_text,
                })

                # Send audio response as binary
                await ws.send_bytes(result.audio_bytes)

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected — session %s", session.session_id)
    except Exception as e:
        logger.error("WebSocket error: %s", e)
    finally:
        if not session.ended:
            session.end()
