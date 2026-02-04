(() => {
  "use strict";

  // DOM elements
  const micBtn      = document.getElementById("mic-btn");
  const endBtn      = document.getElementById("end-btn");
  const chatArea    = document.getElementById("chat-area");
  const statusDot   = document.getElementById("status-dot");
  const statusText  = document.getElementById("status-text");
  const langBadge   = document.getElementById("lang-badge");
  const processing  = document.getElementById("processing");
  const procText    = document.getElementById("processing-text");
  const summaryPanel    = document.getElementById("summary-panel");
  const summaryText     = document.getElementById("summary-text");
  const sentimentBadge  = document.getElementById("sentiment-badge");
  const sentimentDetail = document.getElementById("sentiment-details");
  const turnCount       = document.getElementById("turn-count");

  // State
  let ws = null;
  let mediaRecorder = null;
  let audioChunks = [];
  let isRecording = false;
  let audioCtx = null;
  let sessionActive = false;

  // ── WebSocket ──────────────────────────────────────────────

  function connect() {
    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    ws = new WebSocket(`${proto}//${location.host}/ws/audio`);

    ws.binaryType = "arraybuffer";

    ws.onopen = () => {
      setStatus("connected", "Connected");
      sessionActive = true;
    };

    ws.onmessage = (evt) => {
      if (typeof evt.data === "string") {
        handleJsonMessage(JSON.parse(evt.data));
      } else {
        // Binary = audio response
        playAudio(evt.data);
      }
    };

    ws.onclose = () => {
      setStatus("disconnected", "Disconnected");
      sessionActive = false;
    };

    ws.onerror = () => {
      setStatus("disconnected", "Connection error");
    };
  }

  function handleJsonMessage(msg) {
    switch (msg.type) {
      case "session_start":
        addMessage("agent", "Session started. Hold the mic button and speak.");
        break;

      case "transcription":
        if (msg.text) {
          addMessage("user", msg.text, msg.language);
          langBadge.textContent = (msg.language || "--").toUpperCase();
        }
        setStatus("connected", "Connected");
        break;

      case "response":
        addMessage("agent", msg.text);
        showProcessing("Speaking...");
        break;

      case "summary":
        hideProcessing();
        showSummary(msg);
        break;

      case "error":
        hideProcessing();
        addMessage("agent", `Error: ${msg.message}`);
        setStatus("connected", "Connected");
        break;

      case "pong":
        break;
    }
  }

  // ── Audio Capture ──────────────────────────────────────────

  async function startRecording() {
    if (isRecording) return;

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          sampleRate: 16000,
          echoCancellation: true,
          noiseSuppression: true,
        }
      });

      audioChunks = [];
      mediaRecorder = new MediaRecorder(stream, {
        mimeType: getSupportedMimeType(),
      });

      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) audioChunks.push(e.data);
      };

      mediaRecorder.onstop = () => {
        // Stop all tracks
        stream.getTracks().forEach(t => t.stop());

        if (audioChunks.length === 0) return;

        const blob = new Blob(audioChunks, { type: mediaRecorder.mimeType });
        sendAudio(blob);
      };

      mediaRecorder.start();
      isRecording = true;
      micBtn.classList.add("recording");

    } catch (err) {
      console.error("Mic access denied:", err);
      addMessage("agent", "Microphone access denied. Please allow microphone permissions.");
    }
  }

  function stopRecording() {
    if (!isRecording || !mediaRecorder) return;
    isRecording = false;
    micBtn.classList.remove("recording");
    mediaRecorder.stop();
    showProcessing("Transcribing...");
    setStatus("processing", "Processing");
  }

  function getSupportedMimeType() {
    const types = [
      "audio/webm;codecs=opus",
      "audio/webm",
      "audio/ogg;codecs=opus",
      "audio/mp4",
    ];
    for (const t of types) {
      if (MediaRecorder.isTypeSupported(t)) return t;
    }
    return "audio/webm";
  }

  function sendAudio(blob) {
    if (!ws || ws.readyState !== WebSocket.OPEN) {
      addMessage("agent", "Not connected. Please refresh the page.");
      hideProcessing();
      return;
    }

    blob.arrayBuffer().then(buf => {
      ws.send(buf);
      showProcessing("Thinking...");
    });
  }

  // ── Audio Playback ─────────────────────────────────────────

  function playAudio(arrayBuffer) {
    if (!audioCtx) {
      audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    }

    audioCtx.decodeAudioData(arrayBuffer.slice(0), (buffer) => {
      const source = audioCtx.createBufferSource();
      source.buffer = buffer;
      source.connect(audioCtx.destination);
      source.onended = () => hideProcessing();
      source.start(0);
    }, (err) => {
      console.error("Audio decode error:", err);
      hideProcessing();
    });
  }

  // ── UI Helpers ─────────────────────────────────────────────

  function addMessage(role, text, lang) {
    const div = document.createElement("div");
    div.className = `message ${role}`;

    const meta = document.createElement("div");
    meta.className = "meta";
    meta.textContent = role === "user" ? "You" : "Agent";
    if (lang) {
      const badge = document.createElement("span");
      badge.className = "lang-badge";
      badge.textContent = lang.toUpperCase();
      meta.appendChild(badge);
    }

    const body = document.createElement("div");
    body.textContent = text;

    div.appendChild(meta);
    div.appendChild(body);
    chatArea.appendChild(div);
    chatArea.scrollTop = chatArea.scrollHeight;
  }

  function setStatus(state, text) {
    statusDot.className = `status-dot ${state}`;
    statusText.textContent = text;
  }

  function showProcessing(text) {
    processing.classList.add("visible");
    procText.textContent = text || "Processing...";
  }

  function hideProcessing() {
    processing.classList.remove("visible");
  }

  function showSummary(data) {
    summaryPanel.classList.add("visible");
    summaryText.textContent = data.summary || "No summary available.";

    const sentiment = data.sentiment || {};
    sentimentBadge.textContent = sentiment.overall || "unknown";
    sentimentBadge.className = `sentiment-badge ${sentiment.overall || "neutral"}`;
    sentimentDetail.textContent = sentiment.details || "";
    turnCount.textContent = `${data.turn_count || 0} turns`;

    // Hide controls
    document.getElementById("controls").style.display = "none";
  }

  // ── Event Listeners ────────────────────────────────────────

  // Press-and-hold to record
  micBtn.addEventListener("pointerdown", (e) => {
    e.preventDefault();
    if (!sessionActive) return;
    startRecording();
  });

  micBtn.addEventListener("pointerup", (e) => {
    e.preventDefault();
    stopRecording();
  });

  micBtn.addEventListener("pointerleave", (e) => {
    if (isRecording) stopRecording();
  });

  // Prevent context menu on long press (mobile)
  micBtn.addEventListener("contextmenu", (e) => e.preventDefault());

  // End session
  endBtn.addEventListener("click", () => {
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    showProcessing("Generating summary...");
    ws.send(JSON.stringify({ type: "end_session" }));
  });

  // ── Init ───────────────────────────────────────────────────
  connect();
})();
