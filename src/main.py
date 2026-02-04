"""FastAPI application — serves the web UI and WebSocket audio endpoint."""

import io
import json
import logging
import subprocess
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

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
    """Convert WebM/Opus audio from the browser to WAV for Whisper.

    Uses ffmpeg with temp files to avoid pipe/seek issues with WebM containers.
    """
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp_in:
        tmp_in.write(webm_bytes)
        tmp_in_path = tmp_in.name

    tmp_out_path = tmp_in_path.replace(".webm", ".wav")

    try:
        result = subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", tmp_in_path,
                "-ac", "1",
                "-ar", "16000",
                "-sample_fmt", "s16",
                "-f", "wav",
                tmp_out_path,
            ],
            capture_output=True,
            timeout=15,
        )

        if result.returncode != 0:
            stderr = result.stderr.decode(errors="replace")
            raise RuntimeError(f"ffmpeg failed (code {result.returncode}): {stderr[-300:]}")

        return Path(tmp_out_path).read_bytes()

    finally:
        Path(tmp_in_path).unlink(missing_ok=True)
        Path(tmp_out_path).unlink(missing_ok=True)


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
                if not audio_webm or len(audio_webm) < 1000:
                    # Skip empty or too-small chunks (corrupt/incomplete WebM)
                    logger.debug("Skipping tiny audio chunk (%d bytes)", len(audio_webm) if audio_webm else 0)
                    await ws.send_json({"type": "transcription", "text": "", "language": ""})
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
