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
  let isPlaying = false;
  let recordStartTime = 0;
  let audioCtx = null;
  let sessionActive = false;

  const MIN_RECORD_MS = 600; // minimum recording duration

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
        } else {
          // Empty transcription — nothing heard, reset UI
          hideProcessing();
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
    if (isRecording || isPlaying) return;

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
      recordStartTime = Date.now();
      micBtn.classList.add("recording");

    } catch (err) {
      console.error("Mic access denied:", err);
      addMessage("agent", "Microphone access denied. Please allow microphone permissions.");
    }
  }

  function stopRecording() {
    if (!isRecording || !mediaRecorder) return;

    const elapsed = Date.now() - recordStartTime;
    isRecording = false;
    micBtn.classList.remove("recording");

    if (elapsed < MIN_RECORD_MS) {
      // Too short — discard and stop tracks
      mediaRecorder.stream.getTracks().forEach(t => t.stop());
      mediaRecorder = null;
      audioChunks = [];
      return;
    }

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

  function ensureAudioCtx() {
    if (!audioCtx) {
      audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    }
    // Resume if suspended (browsers require user gesture)
    if (audioCtx.state === "suspended") {
      audioCtx.resume().catch(() => {});
    }
    return audioCtx;
  }

  function playAudio(arrayBuffer) {
    const ctx = ensureAudioCtx();

    isPlaying = true;

    ctx.decodeAudioData(arrayBuffer.slice(0))
      .then((buffer) => {
        const source = ctx.createBufferSource();
        source.buffer = buffer;
        source.connect(ctx.destination);
        source.onended = () => { isPlaying = false; hideProcessing(); };
        source.start(0);
      })
      .catch((err) => {
        console.error("Audio decode error:", err);
        playAudioFallback(arrayBuffer);
      });
  }

  function playAudioFallback(arrayBuffer) {
    try {
      const blob = new Blob([arrayBuffer], { type: "audio/mpeg" });
      const url = URL.createObjectURL(blob);
      const audio = new Audio(url);
      audio.onended = () => {
        URL.revokeObjectURL(url);
        isPlaying = false;
        hideProcessing();
      };
      audio.onerror = () => {
        URL.revokeObjectURL(url);
        isPlaying = false;
        hideProcessing();
      };
      audio.play().catch(() => { isPlaying = false; hideProcessing(); });
    } catch (e) {
      console.error("Fallback audio playback failed:", e);
      hideProcessing();
    }
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

  // Press-and-hold to record — touch events for mobile, pointer for desktop
  function onMicDown(e) {
    e.preventDefault();
    e.stopPropagation();
    // Unlock AudioContext on user gesture so playback works later
    ensureAudioCtx();
    if (!sessionActive) return;
    startRecording();
  }

  function onMicUp(e) {
    e.preventDefault();
    e.stopPropagation();
    if (isRecording) stopRecording();
  }

  // Touch events (mobile-first)
  micBtn.addEventListener("touchstart", onMicDown, { passive: false });
  micBtn.addEventListener("touchend", onMicUp, { passive: false });
  micBtn.addEventListener("touchcancel", onMicUp, { passive: false });

  // Mouse events (desktop fallback)
  micBtn.addEventListener("mousedown", onMicDown);
  micBtn.addEventListener("mouseup", onMicUp);
  micBtn.addEventListener("mouseleave", (e) => { if (isRecording) stopRecording(); });

  // Prevent context menu and text selection on long press (mobile)
  micBtn.addEventListener("contextmenu", (e) => e.preventDefault());
  micBtn.addEventListener("selectstart", (e) => e.preventDefault());

  // End session
  endBtn.addEventListener("click", () => {
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    showProcessing("Generating summary...");
    ws.send(JSON.stringify({ type: "end_session" }));
  });

  // ── Init ───────────────────────────────────────────────────
  connect();
})();
