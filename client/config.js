window.AUDIO8_CONFIG = {
  publicVllmPort: "18190",
  // Optional second vLLM instance. When set, the page shows an instance picker:
  //   * plain-HTTP page  -> ws://<host>:<publicVllmPort2>/v1/realtime (direct)
  //   * HTTPS page       -> wss://<host>:<page port>/v1/realtime?instance=2
  //                        (routed by the TLS proxy, docker/web_tls_proxy.py)
  publicVllmPort2: "18193",
  // Contextual biasing: default hotword list (one term per line) and boost.
  // The page keeps these in localStorage; each session.update carries them.
  hotwords: "",
  hotwordBoost: "1.5",
  modelName: "audio8-asr-infinite",
  zhEnModelName: "audio8-asr-infinite",
  baseModelName: "audio8-asr-infinite",
  language: "zh",
  delayProfile: "480ms",
  // Master VAD switch: "0" = no VAD at all -- Silero never loads (no onnxruntime-web,
  // no onnx model, no per-32ms inference) and the semantic VAD panel/parsing stay off.
  // Set to "1" to bring the VAD features back.
  vadEnabled: "0",
  // Semantic VAD indicator defaults to c0 at 1.0s (written as "horizonSeconds:class").
  vadIndicator: "1.0:0",
  // Semantic VAD panel: hidden by default (rendering/parsing code is kept),
  // enable it at deploy time with AUDIO8_SHOW_VAD_PANEL=1.
  showVadPanel: "0",
  // Probability grid (one bar per horizon x class): hidden by default, timeline only;
  // enable it at deploy time with AUDIO8_SHOW_VAD_GRID=1.
  showVadGrid: "0",
  // Per-class live values (only meaningful when the grid is shown): hidden by default,
  // enable at deploy time with AUDIO8_SHOW_VAD_CLASS_VALUES=1.
  showVadClassValues: "0",
  // Silero comparison VAD: threshold uses the official 0.5 default.
  sileroVadThreshold: "0.5",
  sileroVadModelUrl: "./vendor/silero_vad.onnx",
  ortScriptUrl: "./vendor/ort.min.js"
};
