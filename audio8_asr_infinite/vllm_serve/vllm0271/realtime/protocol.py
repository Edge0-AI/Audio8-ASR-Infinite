# SPDX-License-Identifier: Apache-2.0
"""Wire protocol models for the Audio8 ASR Infinite realtime endpoint overlay.

Re-exports the native realtime protocol classes and extends ``SessionUpdate``
with the session fields the Audio8 ASR Infinite adapter consumes (language, delay profile,
token-duration knobs).  The native 0.27.1 ``SessionUpdate`` only carries
``type``/``model``; pydantic subclassing keeps the rest of the protocol
byte-identical.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from vllm.entrypoints.speech_to_text.realtime.protocol import (
    ErrorEvent,
    InputAudioBufferAppend,
    InputAudioBufferCommit,
    SessionCreated,
    TranscriptionDelta,
    TranscriptionDone,
    SessionUpdate as _NativeSessionUpdate,
)

__all__ = [
    "ErrorEvent",
    "InputAudioBufferAppend",
    "InputAudioBufferCommit",
    "MetricsDelta",
    "SemanticVADDelta",
    "SessionCreated",
    "SessionUpdate",
    "TranscriptionDelta",
    "TranscriptionDone",
]


class SessionUpdate(_NativeSessionUpdate):
    """Native SessionUpdate + Audio8 ASR Infinite session knobs (all optional)."""

    language: str | None = None
    delay_profile: str | None = None
    target_delay_ms: int | None = None
    num_delay_tokens: int | None = None
    token_duration_ms: int | None = None
    # Contextual biasing for the decoder: terms the transcription should
    # prefer, and how strongly to bias towards them (see connection.py).
    hotwords: list[str] | None = None
    hotword_boost: float | None = None


class MetricsDelta(BaseModel):
    """Per-step serving metrics for the web client's GPU-memory/RTF timelines.

    ``rtf`` is this step's model-side time (forward through logits) divided by
    the audio duration the step consumed, so < 1 means the model runs faster
    than real time; ``rtf_ema`` is an exponential moving average of it.  The
    GPU memory numbers are whole-device usage as reported by the driver, which
    is what ``nvidia-smi`` shows for the serving GPU.

    Like ``SemanticVADDelta`` this is plain pydantic because the connection
    only needs ``model_dump_json``.  It is published for every checkpoint,
    whether or not the served model carries semantic VAD heads.
    """

    type: Literal["metrics.delta"] = "metrics.delta"
    step_index: int = 0
    step_ms: float = 0.0
    audio_ms: float = 0.0
    rtf: float = 0.0
    rtf_ema: float = 0.0
    gpu_memory_used_bytes: int = 0
    gpu_memory_total_bytes: int = 0
    gpu_memory_used_ratio: float = 0.0


class SemanticVADDelta(BaseModel):
    """Incremental semantic VAD head prediction (one event per generated step).

    ``probabilities[h][c]`` is the probability that horizon
    ``horizons_seconds[h]`` contains exactly ``c`` future semantic units; class
    0 is end-of-turn.  ``selected_eot_probability`` is that class-0 probability
    at ``selected_eot_horizon_seconds``, i.e. the value the web client
    thresholds on.

    Convention: class 0 means *no speech within that horizon* (end-of-turn), so
    a **speech** probability — the direction Silero VAD reports — is
    ``1 - P(class 0)``.  The payload keeps the model's own orientation (higher
    ``selected_eot_probability`` = more likely silent); consumers that display a
    speech probability, including this repository's web client, invert it so
    that 1 means speaking and 0 means not speaking for both VADs.

    Plain pydantic is used rather than a vLLM protocol base class because the
    connection only needs ``model_dump_json``; that keeps this event
    independent of vLLM's internal protocol module layout.
    """

    type: Literal["semantic_vad.delta"] = "semantic_vad.delta"
    step_index: int = 0
    horizons_seconds: list[float] = Field(default_factory=list)
    predicted_classes: list[int] = Field(default_factory=list)
    probabilities: list[list[float]] = Field(default_factory=list)
    eot_probabilities: list[float] = Field(default_factory=list)
    selected_eot_horizon_seconds: float = 2.0
    selected_eot_probability: float = 0.0
