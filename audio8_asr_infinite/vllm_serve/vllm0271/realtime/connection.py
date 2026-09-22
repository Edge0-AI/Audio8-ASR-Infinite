# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import json
import os
from collections.abc import AsyncGenerator
from http import HTTPStatus
from uuid import uuid4

from typing import Any, Sequence

import numpy as np
import pybase64 as base64
from fastapi import WebSocket
from starlette.websockets import WebSocketDisconnect

from vllm import envs
from vllm.entrypoints.openai.engine.protocol import ErrorResponse, UsageInfo
from vllm.entrypoints.serve.utils.api_utils import sanitize_message
from vllm.exceptions import VLLMValidationError
from vllm.logger import init_logger

from .protocol import (
    ErrorEvent,
    InputAudioBufferAppend,
    InputAudioBufferCommit,
    MetricsDelta,
    SemanticVADDelta,
    SessionCreated,
    SessionUpdate,
    TranscriptionDelta,
    TranscriptionDone,
)
from audio8_asr_infinite.vllm_serve.vllm0271 import audio8_vad_channel
from audio8_asr_infinite.vllm_serve.vllm0271.realtime.serving import Audio8OpenAIServingRealtime

logger = init_logger(__name__)


_AUDIO8_HOTWORD_TOKENIZER: Any = None


def _audio8_hotword_tokenizer(serving: Any) -> Any:
    """Best-effort tokenizer for hotword biasing (cached; None if unreachable)."""
    global _AUDIO8_HOTWORD_TOKENIZER
    if _AUDIO8_HOTWORD_TOKENIZER is not None:
        return _AUDIO8_HOTWORD_TOKENIZER

    try:
        from vllm.tokenizers import cached_tokenizer_from_config
    except Exception:  # pragma: no cover - vLLM always ships it
        cached_tokenizer_from_config = None

    candidates = []
    getter = getattr(getattr(serving, "model_cls", None), "get_tokenizer", None)
    if callable(getter):
        candidates.append(getter)
    model_config = getattr(serving, "model_config", None)
    if cached_tokenizer_from_config is not None and model_config is not None:
        candidates.append(lambda: cached_tokenizer_from_config(model_config))

    for candidate in candidates:
        try:
            tokenizer = candidate()
        except Exception:
            continue
        if tokenizer is not None:
            _AUDIO8_HOTWORD_TOKENIZER = tokenizer
            return tokenizer
    logger.warning(
        "Hotwords were requested but no tokenizer is reachable from the serving "
        "object; the session continues without contextual biasing."
    )
    return None


def _audio8_hotword_logit_bias(
    serving: Any,
    hotwords: Sequence[str],
    boost: float,
) -> dict[int, float]:
    """Turn hotwords into a vLLM ``logit_bias`` map.

    vLLM 0.27.1 exposes no per-request logits-processor hook on
    ``SamplingParams`` (only the engine-wide pattern), so contextual biasing is
    expressed as token logit bias: the first token of each hotword gets the full
    boost and the remaining tokens half of it.  The model still chooses the word
    itself, it just stops losing it to a homophone; raise the boost to make the
    bias stronger, or use 0 to disable it.
    """
    if not hotwords or boost <= 0:
        return {}

    tokenizer = _audio8_hotword_tokenizer(serving)
    if tokenizer is None:
        return {}

    bias: dict[int, float] = {}
    for word in hotwords:
        try:
            encoded = tokenizer.encode(word, add_special_tokens=False)
        except TypeError:
            try:
                encoded = tokenizer.encode(word)
            except Exception:
                continue
        except Exception:
            continue
        try:
            token_ids = [int(token) for token in encoded]
        except Exception:
            continue
        if not token_ids:
            continue
        bias[token_ids[0]] = max(bias.get(token_ids[0], 0.0), float(boost))
        for token in token_ids[1:]:
            bias[token] = max(bias.get(token, 0.0), float(boost) * 0.5)
    return bias


def _audio8_dump_audio_dir() -> str:
    """Optional audio capture for reproducing streaming degradations offline.

    Set ``AUDIO8_REALTIME_DUMP_AUDIO`` to a directory to write each session's
    incoming audio as raw 16 kHz mono s16le PCM (``*.pcm``).  Off by default:
    this stores user audio, so only enable it while debugging.
    """
    return (os.environ.get("AUDIO8_REALTIME_DUMP_AUDIO", "") or "").strip()


def _audio8_dump_max_seconds() -> float:
    try:
        return float(os.environ.get("AUDIO8_REALTIME_DUMP_MAX_SECONDS", "300"))
    except ValueError:
        return 300.0


def _audio8_audio_log_every() -> int:
    """每 N 个上行音频块打一条 RMS/峰值日志；0=关闭（默认）。

    与 AUDIO8_REALTIME_DELTA_LOG_EVERY 配对使用：文本 delta 变空时，需要知道那一刻
    上行音频到底是「静音」还是「还有语音」。静音则模型不出文本是正确行为；有语音而
    模型哑掉才是真故障。
    """
    try:
        return max(0, int(os.environ.get("AUDIO8_REALTIME_AUDIO_LOG_EVERY", "0")))
    except ValueError:
        return 0


def _audio8_delta_log_every() -> int:
    """每 N 个生成步打一条 delta/累计文本长度日志；0=关闭（默认）。

    24 小时长跑排查「前端不再显示新转写」时必须先分清是哪一侧停了：如果服务端这里
    的 delta_len 一直 >0、text_len 一直增长，那文本是发出去的，问题在渲染/链路；如果
    delta_len 恒为 0（模型只出 pad）或日志本身不再出现（引擎不再产出），那问题在推理。
    """
    try:
        return max(0, int(os.environ.get("AUDIO8_REALTIME_DELTA_LOG_EVERY", "0")))
    except ValueError:
        return 0


class RealtimeConnection:
    """Manages WebSocket lifecycle and state for realtime transcription.

    This class handles:
    - WebSocket connection lifecycle (accept, receive, send, close)
    - Event routing (session.update, append, commit)
    - Audio buffering via asyncio.Queue
    - Generation task management
    - Error handling and cleanup
    """

    def __init__(self, websocket: WebSocket, serving: Audio8OpenAIServingRealtime):
        self.websocket = websocket
        self.connection_id = f"ws-{uuid4()}"
        self.serving = serving
        self.audio_queue: asyncio.Queue[np.ndarray | None] = asyncio.Queue()
        self._audio_log_every = _audio8_audio_log_every()
        self._audio_chunk_count = 0
        self._audio_dump_dir = _audio8_dump_audio_dir()
        self._audio_dump_max_samples = int(
            _audio8_dump_max_seconds() * 16000
        )
        self._audio_dump_handle = None
        self._audio_dump_samples = 0
        self._audio_dump_path = ""
        self._empty_delta_logged = 0
        self._hotwords: list[str] = []
        self._hotword_boost = 1.5
        self.generation_task: asyncio.Task | None = None
        self._request_id: str | None = None
        self._cleanup_started = False

        self._is_connected = False
        self._is_model_validated = False
        # Session settings are mutable until a generation starts.  A copy is
        # passed to the model buffer for each generation so a later
        # session.update cannot change an already-running rolling context.
        self._session_language: str | None = None
        self._realtime_options: dict[str, object] = {}

        # Semantic VAD events: `_semantic_vad_counter` is how much of the side
        # channel this connection has consumed, `_semantic_vad_index` is the
        # client-visible step counter carried on every event.
        self._semantic_vad_counter = 0
        self._semantic_vad_index = 0
        self._semantic_vad_salt: int | None = None
        # Serving metrics (RTF / GPU memory) share the same side channel and
        # drain, so they are reported per generated step with their own index.
        self._metrics_index = 0

        self._max_audio_filesize_mb = envs.VLLM_MAX_AUDIO_CLIP_FILESIZE_MB

    async def handle_connection(self):
        """Main connection loop."""
        await self.websocket.accept()
        logger.debug("WebSocket connection accepted: %s", self.connection_id)
        self._is_connected = True

        # Send session created event
        await self.send(SessionCreated())

        try:
            while True:
                message = await self.websocket.receive_text()
                try:
                    event = json.loads(message)
                    await self.handle_event(event)
                except json.JSONDecodeError:
                    await self.send_error("Invalid JSON", "invalid_json")
                except Exception as e:
                    logger.exception("Error handling event: %s", e)
                    await self.send_error(sanitize_message(str(e)), "processing_error")
        except WebSocketDisconnect:
            logger.debug("WebSocket disconnected: %s", self.connection_id)
            self._is_connected = False
        except Exception as e:
            logger.exception("Unexpected error in connection: %s", e)
        finally:
            await self.cleanup()

    def _check_model(self, model: str | None) -> None | ErrorResponse:
        if self.serving._is_model_supported(model):
            return None

        return self.serving.create_error_response(
            message=f"The model `{model}` does not exist.",
            err_type="NotFoundError",
            status_code=HTTPStatus.NOT_FOUND,
            param="model",
        )

    async def handle_event(self, event: dict):
        """Route events to handlers.

        Supported event types:
        - session.update: Configure model
        - input_audio_buffer.append: Add audio chunk to queue
        - input_audio_buffer.commit: Start transcription generation
        """
        event_type = event.get("type")
        if event_type == "session.update":
            update_event = SessionUpdate(**event)
            logger.debug("Session updated: %s", update_event)
            model = update_event.model
            if model is None and not self._is_model_validated:
                await self.send_error("Missing required field: model", "invalid_event")
                return
            if model is not None:
                err = self._check_model(model)
                if err is not None:
                    await self.send_error(err.error.message, "model_not_found")
                    return

            if update_event.language is not None:
                self._session_language = str(update_event.language)
            if update_event.hotwords is not None:
                cleaned: list[str] = []
                for word in update_event.hotwords:
                    text = str(word).strip()
                    if text and text not in cleaned:
                        cleaned.append(text[:64])
                    if len(cleaned) >= 200:
                        break
                self._hotwords = cleaned
            if update_event.hotword_boost is not None:
                try:
                    self._hotword_boost = max(
                        0.0, min(float(update_event.hotword_boost), 10.0)
                    )
                except (TypeError, ValueError):
                    pass
            option_values = {
                "delay_profile": update_event.delay_profile,
                "target_delay_ms": update_event.target_delay_ms,
                "num_delay_tokens": update_event.num_delay_tokens,
            }
            # session.update is intentionally partial: a language-only update
            # keeps the delay settings selected earlier in this session.
            self._realtime_options.update(
                {key: value for key, value in option_values.items() if value is not None}
            )
            logger.info(
                "Realtime session %s language=%s delay_options=%s hotwords=%s "
                "hotword_boost=%s",
                self.connection_id,
                self._session_language,
                self._realtime_options,
                len(self._hotwords),
                self._hotword_boost,
            )
            self._is_model_validated = True
        elif event_type == "input_audio_buffer.append":
            append_event = InputAudioBufferAppend(**event)
            try:
                audio_bytes = base64.b64decode(append_event.audio)
                # Convert PCM16 bytes to float32 numpy array
                audio_array = (
                    np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
                    / 32768.0
                )

                if len(audio_array) / 1024**2 > self._max_audio_filesize_mb:
                    raise VLLMValidationError(
                        "Maximum file size exceeded",
                        parameter="audio_filesize_mb",
                        value=len(audio_array) / 1024**2,
                    )
                if len(audio_array) == 0:
                    raise VLLMValidationError("Can't process empty audio.")

                # Put audio chunk in queue
                self.audio_queue.put_nowait(audio_array)
                self._audio8_dump_chunk(audio_array)
                if self._audio_log_every:
                    self._audio_chunk_count += 1
                    if self._audio_chunk_count % self._audio_log_every == 0:
                        chunk_rms = (
                            float(np.sqrt(np.mean(np.square(audio_array))))
                            if audio_array.size
                            else 0.0
                        )
                        chunk_peak = (
                            float(np.max(np.abs(audio_array)))
                            if audio_array.size
                            else 0.0
                        )
                        logger.info(
                            "AUDIO8_REALTIME_AUDIO chunks=%s samples=%s rms=%.4f "
                            "peak=%.4f",
                            self._audio_chunk_count,
                            int(audio_array.size),
                            chunk_rms,
                            chunk_peak,
                        )

            except Exception as e:
                logger.error("Failed to decode audio: %s", e)
                await self.send_error("Invalid audio data", "invalid_audio")

        elif event_type == "input_audio_buffer.commit":
            if not self._is_model_validated:
                err_msg = (
                    "Model not validated. Make sure to validate the"
                    " model by sending a session.update event."
                )
                await self.send_error(
                    err_msg,
                    "model_not_validated",
                )
                return

            commit_event = InputAudioBufferCommit(**event)
            # final signals that the audio is finished
            if commit_event.final:
                self.audio_queue.put_nowait(None)
            else:
                await self.start_generation()
        else:
            await self.send_error(f"Unknown event type: {event_type}", "unknown_event")

    async def audio_stream_generator(self) -> AsyncGenerator[np.ndarray, None]:
        """Generator that yields audio chunks from the queue."""
        while True:
            audio_chunk = await self.audio_queue.get()
            if audio_chunk is None:  # Sentinel value to stop
                break
            yield audio_chunk

    async def start_generation(self):
        """Start the transcription generation task."""
        if self.generation_task is not None and not self.generation_task.done():
            logger.warning("Generation already in progress, ignoring commit")
            return

        # Create audio stream generator
        audio_stream = self.audio_stream_generator()
        input_stream = asyncio.Queue[list[int]]()
        # Prefix caching is request-scoped in vLLM, while the Audio8 ASR Infinite HF decoder
        # cache lives on the long-lived model instance.  A fresh salt per
        # generation prevents a reconnect or a later utterance from reusing
        # an older session's prefix blocks before the model sees position 0.
        cache_salt = f"{self.connection_id}-{uuid4()}"
        # The model adapter derives the same digest when it publishes semantic
        # VAD predictions, so this is the handle for reading them back.
        self._semantic_vad_salt = audio8_vad_channel.session_salt(cache_salt)
        self._semantic_vad_counter = 0
        self._semantic_vad_index = 0
        self._metrics_index = 0

        # Transform to StreamingInput generator
        # Snapshot session configuration at generation start.  This keeps the
        # language token and delay profile fixed for all rolling windows in
        # this generation while still allowing a later utterance to update it.
        session_options = dict(self._realtime_options)
        session_language = self._session_language
        if session_language is not None:
            # Keep the value in the options mapping as well as the explicit
            # keyword so adapters that only implement the older options hook
            # still receive the selected language.
            session_options.setdefault("language", session_language)
        streaming_input_gen = self.serving.transcribe_realtime(
            audio_stream,
            input_stream,
            realtime_options=session_options,
            language=session_language,
            cache_salt=cache_salt,
        )

        # Start generation task
        self.generation_task = asyncio.create_task(
            self._run_generation(streaming_input_gen, input_stream)
        )

    async def _run_generation(
        self,
        streaming_input_gen: AsyncGenerator,
        input_stream: asyncio.Queue[list[int]],
    ):
        """Run the generation and stream results back to the client.

        This method:
        1. Creates sampling parameters from session config
        2. Passes the streaming input generator to engine.generate()
        3. Streams transcription.delta events as text is generated
        4. Sends final transcription.done event with usage stats
        5. Feeds generated token IDs back to input_stream for next iteration
        6. Cleans up the audio queue
        """
        request_id = f"rt-{self.connection_id}-{uuid4()}"
        self._request_id = request_id
        full_text = ""

        prompt_token_ids_len: int = 0
        completion_tokens_len: int = 0
        delta_log_every = _audio8_delta_log_every()
        delta_steps = 0

        try:
            # Create sampling params
            from vllm.sampling_params import RequestOutputKind, SamplingParams

            # Contextual biasing: express the session's hotwords as token-level
            # logit bias (vLLM 0.27.1 has no per-request logits processor).
            hotword_bias = _audio8_hotword_logit_bias(
                self.serving,
                self._hotwords,
                self._hotword_boost,
            )
            if self._hotwords:
                logger.info(
                    "Realtime session %s hotwords=%s boost=%s biased_token_ids=%s",
                    self.connection_id,
                    len(self._hotwords),
                    self._hotword_boost,
                    len(hotword_bias),
                )
            sampling_params = SamplingParams.from_optional(
                temperature=0.0,
                max_tokens=self.serving.model_cls.realtime_max_tokens,
                output_kind=RequestOutputKind.DELTA,
                skip_clone=True,
                logit_bias=hotword_bias if hotword_bias else None,
            )

            # Pass the streaming input generator to the engine
            # The engine will consume audio chunks as they arrive and
            # stream back transcription results incrementally
            result_gen = self.serving.engine_client.generate(
                prompt=streaming_input_gen,
                sampling_params=sampling_params,
                request_id=request_id,
            )

            # Stream results back to client as they're generated
            async for output in result_gen:
                if output.outputs and len(output.outputs) > 0:
                    if not prompt_token_ids_len and output.prompt_token_ids:
                        prompt_token_ids_len = len(output.prompt_token_ids)

                    delta = output.outputs[0].text
                    full_text += delta
                    if delta_log_every:
                        delta_steps += 1
                        if delta_steps % delta_log_every == 0:
                            logger.info(
                                "AUDIO8_REALTIME_DELTA steps=%s delta_len=%s "
                                "text_len=%s tail=%r",
                                delta_steps,
                                len(delta or ""),
                                len(full_text),
                                full_text[-48:],
                            )

                    if not delta:
                        # A step whose sampled token decodes to nothing (special /
                        # placeholder / pad) is the onset of the "transcript stops
                        # updating" degradation: the token still enters the rolling
                        # context, so log its id to identify it after the fact.
                        self._empty_delta_logged += 1
                        if self._empty_delta_logged <= 10 or self._empty_delta_logged % 500 == 0:
                            logger.info(
                                "AUDIO8_REALTIME_EMPTY_DELTA count=%s ids=%s step=%s "
                                "text_len=%s",
                                self._empty_delta_logged,
                                list(output.outputs[0].token_ids),
                                delta_steps,
                                len(full_text),
                            )

                    # append output to input
                    input_stream.put_nowait(list(output.outputs[0].token_ids))
                    # Semantic VAD for this step is sent before the text it
                    # belongs to, so the client's VAD panel and transcript stay
                    # on the same step.
                    await self._send_semantic_vad()
                    await self.send(TranscriptionDelta(delta=delta))

                    completion_tokens_len += len(output.outputs[0].token_ids)

                if not self._is_connected:
                    # finish because websocket connection was killed
                    await self._abort_request(request_id)
                    break

            usage = UsageInfo(
                prompt_tokens=prompt_token_ids_len,
                completion_tokens=completion_tokens_len,
                total_tokens=prompt_token_ids_len + completion_tokens_len,
            )

            # Send final completion event
            await self.send(TranscriptionDone(text=full_text, usage=usage))

            # Clear queue for next utterance
            while not self.audio_queue.empty():
                self.audio_queue.get_nowait()

        except asyncio.CancelledError:
            await self._abort_request(request_id)
            raise
        except Exception as e:
            logger.exception("Error in generation: %s", e)
            if self._is_connected:
                await self.send_error(sanitize_message(str(e)), "processing_error")
        finally:
            self._audio8_close_dump()
            if self._request_id == request_id:
                self._request_id = None

    def _audio8_dump_chunk(self, audio_array) -> None:
        """Append one incoming chunk to the session's diagnostic PCM file."""
        if not self._audio_dump_dir:
            return
        remaining = self._audio_dump_max_samples - self._audio_dump_samples
        if remaining <= 0:
            return
        try:
            if self._audio_dump_handle is None:
                os.makedirs(self._audio_dump_dir, exist_ok=True)
                name = f"{self.connection_id}-16k-mono-s16le.pcm"
                self._audio_dump_path = os.path.join(self._audio_dump_dir, name)
                self._audio_dump_handle = open(self._audio_dump_path, "wb")
                logger.info(
                    "AUDIO8_REALTIME_DUMP_AUDIO start path=%s max_seconds=%.0f",
                    self._audio_dump_path,
                    self._audio_dump_max_samples / 16000,
                )
            pcm = (np.clip(np.asarray(audio_array), -1.0, 1.0) * 32767.0).astype(
                "<i2"
            ).tobytes()
            keep = min(remaining, len(pcm) // 2 * 2)
            self._audio_dump_handle.write(pcm[:keep])
            self._audio_dump_handle.flush()
            self._audio_dump_samples += keep // 2
        except Exception:
            logger.exception("AUDIO8_REALTIME_DUMP_AUDIO failed; disabling capture")
            self._audio_dump_dir = ""

    def _audio8_close_dump(self) -> None:
        if self._audio_dump_handle is None:
            return
        try:
            self._audio_dump_handle.close()
        except Exception:
            pass
        logger.info(
            "AUDIO8_REALTIME_DUMP_AUDIO done path=%s seconds=%.1f",
            self._audio_dump_path,
            self._audio_dump_samples / 16000,
        )
        self._audio_dump_handle = None

    async def _abort_request(self, request_id: str) -> None:
        """Abort the engine request so its paged KV blocks are released."""
        logger.info("Aborting realtime request on session cleanup: %s", request_id)
        try:
            await self.serving.engine_client.abort(request_id)
        except Exception:
            logger.exception("Failed to abort realtime request: %s", request_id)

    async def _send_semantic_vad(self) -> None:
        """Forward any side-channel predictions the model produced this step.

        The adapter scores the VAD heads on the hidden states it feeds the
        language head and publishes them on ``audio8_vad_channel`` keyed by the
        session salt; nothing VAD-related arrives here for a transcription-only
        checkpoint, so that part is a no-op unless the served model has VAD
        heads.  Per-step serving metrics (RTF / GPU memory) travel the same way
        and are forwarded as ``metrics.delta`` for every checkpoint.
        """

        salt = self._semantic_vad_salt
        if salt is None:
            return
        counter, payloads = audio8_vad_channel.drain(salt, self._semantic_vad_counter)
        if not payloads:
            return
        self._semantic_vad_counter = counter
        for payload in payloads:
            if str(payload.get("type") or "semantic_vad") == "metrics":
                await self.send(
                    MetricsDelta(
                        step_index=self._metrics_index,
                        step_ms=float(payload.get("step_ms") or 0.0),
                        audio_ms=float(payload.get("audio_ms") or 0.0),
                        rtf=float(payload.get("rtf") or 0.0),
                        rtf_ema=float(payload.get("rtf_ema") or 0.0),
                        gpu_memory_used_bytes=int(
                            payload.get("gpu_memory_used_bytes") or 0
                        ),
                        gpu_memory_total_bytes=int(
                            payload.get("gpu_memory_total_bytes") or 0
                        ),
                        gpu_memory_used_ratio=float(
                            payload.get("gpu_memory_used_ratio") or 0.0
                        ),
                    )
                )
                self._metrics_index += 1
                continue
            probabilities = payload["probabilities"]
            await self.send(
                SemanticVADDelta(
                    step_index=self._semantic_vad_index,
                    horizons_seconds=payload["horizons_seconds"],
                    predicted_classes=payload["predicted_classes"],
                    probabilities=probabilities,
                    eot_probabilities=[
                        float(horizon_probabilities[0])
                        for horizon_probabilities in probabilities
                    ],
                    selected_eot_horizon_seconds=payload[
                        "selected_eot_horizon_seconds"
                    ],
                    selected_eot_probability=payload["selected_eot_probability"],
                )
            )
            self._semantic_vad_index += 1

    async def send(
        self,
        event: (
            SessionCreated
            | SemanticVADDelta
            | MetricsDelta
            | TranscriptionDelta
            | TranscriptionDone
        ),
    ):
        """Send event to client."""
        data = event.model_dump_json()
        await self.websocket.send_text(data)

    async def send_error(self, message: str, code: str | None = None):
        """Send error event to client."""
        error_event = ErrorEvent(error=message, code=code)
        await self.websocket.send_text(error_event.model_dump_json())

    async def cleanup(self):
        """Cleanup resources."""
        if self._cleanup_started:
            return
        self._cleanup_started = True
        self._is_connected = False

        # Signal audio stream to stop
        self.audio_queue.put_nowait(None)

        # Cancel generation task if running
        task = self.generation_task
        if task and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

        # The task can be cancelled before it reaches the async generator. Keep
        # a direct abort here as the final request-scoped KV cleanup boundary.
        if self._request_id is not None:
            request_id = self._request_id
            await self._abort_request(request_id)
            self._request_id = None

        logger.debug("Connection cleanup complete: %s", self.connection_id)
