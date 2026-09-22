"""Shared simulated-streaming audio helpers for Audio8 ASR Infinite.

These helpers implement the definition-aligned simulated-streaming audio
frontend: real-time look-back / look-ahead windowing, a streaming audio
KV-cache, and per-80ms-clock text token emission support.  They were extracted
from the original Audio8 ASR Infinite simulated-streaming inference module so
that both the torch decoding entrypoint in this package and external
evaluation harnesses can share the exact same audio semantics.

Design constraints (kept from the original module):
- Only depends on `math`, `numpy`, `torch`.
- The model, feature extractor, tokenizer and duck-typed `audio_config`
  (must expose `.raw_audio_samples_per_token`,
  `.streaming_n_left_pad_tokens`, `.sampling_rate`) are provided by the
  caller; no framework-specific dependencies here.
"""

from __future__ import annotations

import math
from contextlib import nullcontext
from typing import Any, Callable, ContextManager, Iterator, Sequence

import numpy as np
import torch
import torch.nn.functional as F


def _resolve_model_frame_len(
    model: Any,
    frame_len: int | None = None,
) -> int:
    if frame_len is not None:
        return int(frame_len)
    supported_frame_lens = getattr(
        model.config,
        "supported_frame_lens",
        None,
    )
    if supported_frame_lens is None:
        return int(model.config.downsample_factor)
    return int(supported_frame_lens[0])


def ms_to_samples(milliseconds: float, *, sampling_rate: int) -> int:
    samples = sampling_rate * float(milliseconds) / 1000.0
    if not samples.is_integer():
        raise ValueError(
            "Streaming window duration must map to integer samples: "
            f"milliseconds={milliseconds} sampling_rate={sampling_rate}"
        )
    return int(samples)


def cache_seq_length(cache: Any | None) -> int | None:
    if cache is None or not hasattr(cache, "get_seq_length"):
        return None
    return int(cache.get_seq_length())


def iter_realtime_audio_windows(
    waveform: np.ndarray,
    *,
    audio_config: Any,
    num_delay_tokens: int,
    right_pad_text_tokens: int,
    streaming_look_ahead_ms: float,
    streaming_look_back_ms: float,
    initial_prefill_tokens: int | None = None,
) -> tuple[Iterator[dict[str, Any]], dict[str, int | float]]:
    samples_per_token = int(audio_config.raw_audio_samples_per_token)
    left_pad_tokens = int(audio_config.streaming_n_left_pad_tokens)
    if initial_prefill_tokens is None:
        initial_prefill_tokens = left_pad_tokens
    initial_prefill_tokens = int(initial_prefill_tokens)
    if initial_prefill_tokens < left_pad_tokens:
        raise ValueError(
            "initial_prefill_tokens cannot be shorter than the left audio pad: "
            f"prefill={initial_prefill_tokens} left_pad={left_pad_tokens}"
        )
    right_pad_tokens = int(num_delay_tokens) + 1 + int(right_pad_text_tokens)
    look_ahead_samples = ms_to_samples(streaming_look_ahead_ms, sampling_rate=audio_config.sampling_rate)
    look_back_samples = ms_to_samples(streaming_look_back_ms, sampling_rate=audio_config.sampling_rate)

    left_pad = np.zeros(left_pad_tokens * samples_per_token, dtype=np.float32)
    right_pad = np.zeros(right_pad_tokens * samples_per_token, dtype=np.float32)
    stream = np.concatenate([left_pad, waveform.astype(np.float32, copy=False), right_pad])

    timeline = {
        "samples_per_token": samples_per_token,
        "left_pad_tokens": left_pad_tokens,
        "initial_prefill_tokens": initial_prefill_tokens,
        "real_audio_samples": int(waveform.shape[0]),
        "num_delay_tokens": int(num_delay_tokens),
        "right_pad_text_tokens": int(right_pad_text_tokens),
        "right_pad_tokens": right_pad_tokens,
        "look_ahead_samples": look_ahead_samples,
        "look_back_samples": look_back_samples,
        "stream_total_samples": int(stream.shape[0]),
    }

    def generate() -> Iterator[dict[str, Any]]:
        start = 0
        end = initial_prefill_tokens * samples_per_token
        if end <= start:
            end = samples_per_token
        frame_index = 0
        while True:
            frame_start = max(start - look_back_samples, 0)
            frame_end = end + look_ahead_samples
            if frame_end > stream.shape[0]:
                return
            token_count = (end - start) / samples_per_token
            if not token_count.is_integer():
                raise RuntimeError(
                    "Streaming token count must be integral: "
                    f"start={start} end={end} samples_per_token={samples_per_token}"
                )
            yield {
                "frame_index": frame_index,
                "frame_start_sample": int(frame_start),
                "frame_end_sample": int(frame_end),
                "clock_start_sample": int(start),
                "clock_end_sample": int(end),
                "num_tokens": int(token_count),
                "audio": stream[frame_start:frame_end],
            }
            frame_index += 1
            start = end
            end += samples_per_token

    return generate(), timeline


def select_current_20ms_frames(
    conv_embeds: torch.Tensor,
    *,
    expected_20ms_frames: int,
) -> tuple[torch.Tensor, dict[str, int]]:
    actual_20ms_frames = int(conv_embeds.shape[1])
    if actual_20ms_frames < expected_20ms_frames:
        raise ValueError(
            "Streaming audio window produced fewer 20ms tower frames than required: "
            f"actual={actual_20ms_frames} expected={expected_20ms_frames}"
        )
    dropped_left_20ms_frames = actual_20ms_frames - expected_20ms_frames
    selected = conv_embeds[:, dropped_left_20ms_frames:, :]
    return selected, {
        "conv_20ms_frames_before_select": actual_20ms_frames,
        "selected_20ms_frames": int(selected.shape[1]),
        "dropped_left_20ms_frames": dropped_left_20ms_frames,
    }


def _group_streaming_audio_hidden_states(
    *,
    model: Any,
    audio_hidden_states: torch.Tensor,
    num_80ms_tokens: int,
    frame_len: int,
) -> torch.Tensor:
    """Group a homogeneous streaming chunk without the general row scatter.

    Streaming batches are partitioned by one frame length, and the audio tower
    output is already exactly ``num_80ms_tokens * frame_len`` frames.  The
    general ``group_audio_hidden_states`` implementation resolves per-row
    frame lengths and scatters through a zero tensor, which is unnecessary for
    this hot path.  Keep its fallback for non-Realtime adapters.
    """
    config = getattr(model, "config", None)
    audio_config = getattr(config, "audio_config", None)
    max_frame_len = getattr(config, "max_frame_len", None)
    hidden_size = getattr(audio_config, "hidden_size", None)
    group_fn = getattr(model, "group_audio_hidden_states", None)
    if (
        max_frame_len is None
        or hidden_size is None
        or int(frame_len) > int(max_frame_len)
    ):
        if callable(group_fn):
            return group_fn(audio_hidden_states, frame_len=int(frame_len))
        return audio_hidden_states.reshape(
            audio_hidden_states.shape[0],
            int(num_80ms_tokens),
            audio_hidden_states.shape[-1] * int(frame_len),
        )

    batch_size = int(audio_hidden_states.shape[0])
    grouped = audio_hidden_states.reshape(
        batch_size,
        int(num_80ms_tokens),
        int(frame_len),
        int(hidden_size),
    )
    if int(frame_len) < int(max_frame_len):
        grouped = F.pad(
            grouped,
            (0, 0, 0, int(max_frame_len) - int(frame_len)),
        )
    return grouped.reshape(
        batch_size,
        int(num_80ms_tokens),
        int(max_frame_len) * int(hidden_size),
    )


def _extract_streaming_features_batch(
    *,
    feature_extractor: Any,
    audio_windows: Sequence[np.ndarray] | torch.Tensor,
    sampling_rate: int,
    device: torch.device,
    center: bool = True,
) -> torch.Tensor:
    """Extract the realtime mel batch without a CPU round-trip.

    ``VoxtralRealtimeFeatureExtractor.__call__`` computes its STFT with torch
    but deliberately moves the result back to CPU. Streaming windows are already
    uniform and live on the target GPU, so reproducing that small fixed formula
    here avoids a host copy and the repeated Python feature-extractor wrapper.
    Backends without the Voxtral attributes use the canonical extractor path.
    """
    required = ("n_fft", "hop_length", "mel_filters")
    if not all(hasattr(feature_extractor, name) for name in required):
        feature_batch = feature_extractor(
            audio_windows,
            sampling_rate=sampling_rate,
            padding="longest",
            return_tensors="pt",
            center=center,
        )
        return feature_batch["input_features"].to(
            device=device,
            dtype=torch.float32,
        )

    if isinstance(audio_windows, torch.Tensor):
        waveform = audio_windows
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        waveform = waveform.to(device=device, dtype=torch.float32)
    else:
        if not audio_windows:
            return torch.empty(
                (0, int(getattr(feature_extractor, "feature_size", 128)), 0),
                device=device,
                dtype=torch.float32,
            )
        lengths = [int(np.asarray(window).shape[0]) for window in audio_windows]
        max_length = max(lengths)
        waveform = torch.zeros(
            (len(audio_windows), max_length),
            device=device,
            dtype=torch.float32,
        )
        for row, window in enumerate(audio_windows):
            values = torch.as_tensor(
                np.asarray(window),
                device=device,
                dtype=torch.float32,
            ).reshape(-1)
            waveform[row, : values.numel()] = values

    n_fft = int(feature_extractor.n_fft)
    hop_length = int(feature_extractor.hop_length)
    # Match VoxtralRealtimeFeatureExtractor exactly. Although the processor
    # exposes ``win_length``, its reference implementation constructs an
    # ``n_fft``-wide Hann window and leaves torch.stft's win_length unset.
    window_key = (
        device.type,
        device.index,
        n_fft,
        waveform.dtype,
    )
    window_cache = getattr(feature_extractor, "_audio8_streaming_window_cache", None)
    if window_cache is None:
        window_cache = {}
        setattr(feature_extractor, "_audio8_streaming_window_cache", window_cache)
    stft_window = window_cache.get(window_key)
    if stft_window is None:
        stft_window = torch.hann_window(
            n_fft,
            device=device,
            dtype=waveform.dtype,
        )
        window_cache[window_key] = stft_window

    mel_cache = getattr(feature_extractor, "_audio8_streaming_mel_cache", None)
    if mel_cache is None:
        mel_cache = {}
        setattr(feature_extractor, "_audio8_streaming_mel_cache", mel_cache)
    # Cache the transposed matrix actually consumed by the matmul.  The
    # realtime loop extracts one short window per clock tick, so repeatedly
    # materializing even a view and its scalar floor shows up in kernel launch
    # overhead on otherwise small feature batches.
    mel_key = (device.type, device.index, waveform.dtype)
    mel_filters = mel_cache.get(mel_key)
    if mel_filters is None:
        mel_filters = torch.as_tensor(
            feature_extractor.mel_filters,
            device=device,
            dtype=torch.float32,
        ).transpose(0, 1).contiguous()
        mel_cache[mel_key] = mel_filters

    stft = torch.stft(
        waveform,
        n_fft,
        hop_length,
        window=stft_window,
        return_complex=True,
        center=center,
    )
    magnitudes = stft[..., :-1].abs().square()
    mel_spec = mel_filters @ magnitudes
    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    # Processors that predate the Audio8 ASR Infinite config use the reference dynamic floor
    # when this field is absent; the production Audio8 ASR Infinite extractor carries 1.5
    # explicitly, so this fallback does not alter the optimized path there.
    global_log_mel_max = getattr(feature_extractor, "global_log_mel_max", None)
    if global_log_mel_max is not None:
        # No autograd is needed during streaming decode; clamping in place
        # avoids a scalar GPU allocation and an extra pointwise output tensor
        # per tick.
        log_spec.clamp_min_(float(global_log_mel_max) - 8.0)
    else:
        # Match the reference extractor's dynamic path, which uses one scalar
        # maximum over the complete input batch.
        log_spec_max = log_spec.amax()
        log_spec = torch.maximum(log_spec, log_spec_max - 8.0)
    return (log_spec + 4.0) / 4.0


def streaming_audio_embeds_by_definition(
    *,
    model: Any,
    feature_extractor: Any,
    audio_window: np.ndarray,
    audio_past_key_values: Any | None,
    audio_config: Any,
    num_80ms_tokens: int,
    frame_len: int | None = None,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, Any, dict[str, int | None]]:
    downsample_factor = _resolve_model_frame_len(model, frame_len)
    expected_20ms_frames = int(num_80ms_tokens) * downsample_factor

    input_features = _extract_streaming_features_batch(
        feature_extractor=feature_extractor,
        audio_windows=[audio_window],
        sampling_rate=audio_config.sampling_rate,
        device=device,
    ).to(dtype=dtype)

    conv_embeds = model.audio_tower.embedder(input_features)
    conv_embeds, frame_diag = select_current_20ms_frames(
        conv_embeds,
        expected_20ms_frames=expected_20ms_frames,
    )

    cache_before = cache_seq_length(audio_past_key_values)
    audio_outputs = model.audio_tower(
        inputs_embeds=conv_embeds,
        past_key_values=audio_past_key_values,
        use_cache=True,
        return_dict=True,
    )
    audio_hidden_states = audio_outputs.last_hidden_state
    if int(audio_hidden_states.shape[1]) != expected_20ms_frames:
        raise ValueError(
            "Streaming audio tower returned unexpected 20ms hidden length: "
            f"actual={audio_hidden_states.shape[1]} expected={expected_20ms_frames}"
        )

    audio_hidden_states = _group_streaming_audio_hidden_states(
        model=model,
        audio_hidden_states=audio_hidden_states,
        num_80ms_tokens=int(num_80ms_tokens),
        frame_len=downsample_factor,
    )
    projector = model.multi_modal_projector
    projector_input_size = getattr(
        getattr(projector, "linear_1", None),
        "in_features",
        None,
    )
    if (
        projector_input_size is not None
        and int(projector_input_size) > int(audio_hidden_states.shape[-1])
    ):
        audio_hidden_states = torch.nn.functional.pad(
            audio_hidden_states,
            (0, int(projector_input_size) - int(audio_hidden_states.shape[-1])),
        )
    audio_embeds = model.multi_modal_projector(audio_hidden_states).to(device=device, dtype=dtype)
    if int(audio_embeds.shape[1]) != int(num_80ms_tokens):
        raise ValueError(
            "Projected audio embeddings must align to the 80ms text clock: "
            f"audio_embeds={tuple(audio_embeds.shape)} num_80ms_tokens={num_80ms_tokens}"
        )

    diag: dict[str, int | None] = {
        "mel_10ms_frames": int(input_features.shape[-1]),
        "expected_20ms_frames": expected_20ms_frames,
        "projected_80ms_tokens": int(audio_embeds.shape[1]),
        "audio_cache_20ms_before": cache_before,
        "audio_cache_20ms_after": cache_seq_length(audio_outputs.past_key_values),
        **frame_diag,
    }
    return audio_embeds, audio_outputs.past_key_values, diag


def decode_visible_text(
    *,
    tokenizer: Any,
    generated_token_ids: list[int],
    special_ids: dict[str, int],
) -> tuple[str, list[int], dict[str, int]]:
    visible_token_ids: list[int] = []
    skipped_counts = {
        "streaming_pad": 0,
        "streaming_word": 0,
        "bos": 0,
        "eos": 0,
        "pad": 0,
    }
    for token_id in generated_token_ids:
        if token_id == special_ids["streaming_pad_token_id"]:
            skipped_counts["streaming_pad"] += 1
            continue
        if token_id == special_ids["streaming_word_token_id"]:
            skipped_counts["streaming_word"] += 1
            continue
        if token_id == special_ids["bos_token_id"]:
            skipped_counts["bos"] += 1
            continue
        if token_id == special_ids["pad_token_id"]:
            skipped_counts["pad"] += 1
            continue
        if token_id == special_ids["eos_token_id"]:
            skipped_counts["eos"] += 1
            break
        visible_token_ids.append(int(token_id))

    return (
        tokenizer.decode(
            visible_token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ),
        visible_token_ids,
        skipped_counts,
    )


def _precompute_streaming_conv_embeds(
    *,
    model: Any,
    feature_extractor: Any,
    streams: torch.Tensor,
    windows: Sequence[dict[str, Any]],
    frame_len: int,
    sampling_rate: int,
    dtype: torch.dtype,
    device: torch.device,
    window_batch_size: int = 64,
) -> list[torch.Tensor | None] | None:
    """Batch the independent frontend work while preserving stream order.

    Every streaming window still enters the audio tower separately, carrying
    the same incremental KV cache as the definition path.  Mel extraction and
    the convolutional embedder do not depend on that cache, so they can be
    evaluated in batches. Windows are grouped by exact sample length; this is
    important because centered STFT padding at a window boundary is part of
    the streaming definition.
    """
    if not all(
        hasattr(feature_extractor, name)
        for name in ("n_fft", "hop_length", "mel_filters")
    ):
        return None
    if not windows:
        return []

    batch_size = int(streams.shape[0])
    grouped_indices: dict[tuple[int, int], list[int]] = {}
    for index, window in enumerate(windows):
        frame_start = int(window["frame_start_sample"])
        frame_end = int(window["frame_end_sample"])
        num_tokens = int(window["num_tokens"])
        grouped_indices.setdefault(
            (frame_end - frame_start, num_tokens),
            [],
        ).append(index)

    precomputed: list[torch.Tensor | None] = [None] * len(windows)
    for (_window_samples, num_tokens), indices in grouped_indices.items():
        for chunk_start in range(0, len(indices), int(window_batch_size)):
            chunk_indices = indices[
                chunk_start : chunk_start + int(window_batch_size)
            ]
            audio_chunks = torch.cat(
                [
                    streams[
                        :,
                        int(windows[index]["frame_start_sample"]) : int(
                            windows[index]["frame_end_sample"]
                        ),
                    ]
                    for index in chunk_indices
                ],
                dim=0,
            )
            input_features = _extract_streaming_features_batch(
                feature_extractor=feature_extractor,
                audio_windows=audio_chunks,
                sampling_rate=sampling_rate,
                device=device,
            ).to(dtype=dtype)
            conv_embeds = model.audio_tower.embedder(input_features)
            expected_20ms_frames = int(num_tokens) * int(frame_len)
            conv_embeds, _ = select_current_20ms_frames(
                conv_embeds,
                expected_20ms_frames=expected_20ms_frames,
            )
            conv_embeds = conv_embeds.reshape(
                len(chunk_indices),
                batch_size,
                conv_embeds.shape[1],
                conv_embeds.shape[2],
            )
            for local_index, window_index in enumerate(chunk_indices):
                precomputed[window_index] = conv_embeds[local_index]

    # Keep one slot per input window, including a possible ``None`` fallback
    # slot.  The realtime loop uses this index to pair frontend output with
    # its streaming clock; filtering empty slots would silently misalign all
    # following windows.
    return precomputed


@torch.inference_mode()


def precompute_simulated_streaming_audio_embeds(
    *,
    model: Any,
    feature_extractor: Any,
    waveforms: Sequence[np.ndarray],
    audio_config: Any,
    frame_len: int,
    num_delay_tokens: int,
    right_pad_text_tokens: int,
    dtype: torch.dtype,
    device: torch.device,
    sampling_rate: int,
    forward_context_factory: Callable[[], ContextManager[Any]] = nullcontext,
) -> torch.Tensor | None:
    """Encode one frame-length clock once for all delay configurations.

    The audio tower is causal. Delay changes only how the resulting causal
    frame sequence is partitioned into the first language-model prefill, not
    the chronological audio states. Frontend windows still use the exact
    simulated-streaming definition, so centered STFT and overlap trimming are
    unchanged.
    """
    if not waveforms:
        return None
    required = ("n_fft", "hop_length", "mel_filters")
    if not all(hasattr(feature_extractor, name) for name in required):
        return None
    audio_tower = getattr(model, "audio_tower", None)
    if not callable(getattr(audio_tower, "embedder", None)):
        return None

    import math as _math

    samples_per_token = int(audio_config.raw_audio_samples_per_token)
    left_pad_tokens = int(audio_config.streaming_n_left_pad_tokens)
    real_audio_tokens = [
        max(1, _math.ceil(int(waveform.shape[0]) / samples_per_token))
        for waveform in waveforms
    ]
    max_audio_tokens = max(real_audio_tokens)
    dummy = np.zeros(
        max_audio_tokens * samples_per_token,
        dtype=np.float32,
    )
    windows_iter, timeline = iter_realtime_audio_windows(
        dummy,
        audio_config=audio_config,
        num_delay_tokens=int(num_delay_tokens),
        right_pad_text_tokens=int(right_pad_text_tokens),
        streaming_look_ahead_ms=2.5,
        streaming_look_back_ms=52.5,
        initial_prefill_tokens=(left_pad_tokens + int(num_delay_tokens) + 1),
    )
    windows = list(windows_iter)
    if not windows:
        return None
    stream_total_samples = int(timeline["stream_total_samples"])
    streams_cpu = np.zeros(
        (len(waveforms), stream_total_samples),
        dtype=np.float32,
    )
    offset = left_pad_tokens * samples_per_token
    for row, waveform in enumerate(waveforms):
        values = np.asarray(waveform, dtype=np.float32)
        streams_cpu[row, offset : offset + int(values.shape[0])] = values
    streams = torch.as_tensor(
        streams_cpu,
        device=device,
        dtype=torch.float32,
    )
    del streams_cpu, dummy

    # Compute the exact selected frontend frames per clock, then concatenate
    # the causal sequence. This removes repeated audio-tower forwards across
    # delay configurations without changing the streaming frontend.
    with forward_context_factory():
        conv_windows = _precompute_streaming_conv_embeds(
            model=model,
            feature_extractor=feature_extractor,
            streams=streams,
            windows=windows,
            frame_len=int(frame_len),
            sampling_rate=int(sampling_rate),
            dtype=dtype,
            device=device,
        )
    del streams
    if conv_windows is None or any(chunk is None for chunk in conv_windows):
        return None
    conv_sequence = torch.cat(
        [chunk for chunk in conv_windows if chunk is not None],
        dim=1,
    )
    del conv_windows
    expected_tokens = sum(int(window["num_tokens"]) for window in windows)
    expected_frames = expected_tokens * int(frame_len)
    if int(conv_sequence.shape[1]) != expected_frames:
        return None

    position_ids = torch.arange(
        expected_frames,
        device=device,
        dtype=torch.long,
    ).unsqueeze(0)
    with forward_context_factory():
        audio_outputs = audio_tower(
            inputs_embeds=conv_sequence,
            position_ids=position_ids,
            use_cache=False,
            return_dict=True,
        )
    del conv_sequence, position_ids
    audio_hidden_states = audio_outputs.last_hidden_state
    grouped_hidden_states = _group_streaming_audio_hidden_states(
        model=model,
        audio_hidden_states=audio_hidden_states,
        num_80ms_tokens=expected_tokens,
        frame_len=int(frame_len),
    )
    del audio_hidden_states, audio_outputs
    return model.multi_modal_projector(grouped_hidden_states).to(
        device=device,
        dtype=dtype,
    )


def _streaming_audio_embeds_batch(
    *,
    model: Any,
    feature_extractor: Any,
    audio_windows: Sequence[np.ndarray] | torch.Tensor | None,
    audio_past_key_values: Any | None,
    sampling_rate: int,
    num_80ms_tokens: int,
    frame_len: int | None = None,
    precomputed_conv_embeds: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, Any]:
    """Batched version of streaming_audio_embeds_by_definition. All windows in the
    batch have identical length (uniform window schedule), so no padding occurs and
    select_current_20ms_frames slices the same number of frames for every sample."""
    downsample_factor = _resolve_model_frame_len(model, frame_len)
    expected_20ms_frames = int(num_80ms_tokens) * downsample_factor
    if precomputed_conv_embeds is None:
        input_features = _extract_streaming_features_batch(
            feature_extractor=feature_extractor,
            audio_windows=audio_windows,
            sampling_rate=sampling_rate,
            device=device,
        ).to(dtype=dtype)
        conv_embeds = model.audio_tower.embedder(input_features)
        conv_embeds, _ = select_current_20ms_frames(
            conv_embeds,
            expected_20ms_frames=expected_20ms_frames,
        )
    else:
        # The precompute path has already applied the exact per-window frame
        # selection. Keep this branch free of feature-extractor and embedder
        # calls; the audio tower below remains incremental and canonical.
        conv_embeds = precomputed_conv_embeds.to(
            device=device,
            dtype=dtype,
        )
    if position_ids is None:
        audio_outputs = model.audio_tower(
            inputs_embeds=conv_embeds,
            past_key_values=audio_past_key_values,
            use_cache=True,
            return_dict=True,
        )
    else:
        audio_outputs = model.audio_tower(
            inputs_embeds=conv_embeds,
            position_ids=position_ids,
            past_key_values=audio_past_key_values,
            use_cache=True,
            return_dict=True,
        )
    audio_hidden_states = audio_outputs.last_hidden_state
    if int(audio_hidden_states.shape[1]) != expected_20ms_frames:
        raise ValueError(
            "Streaming audio tower returned unexpected 20ms hidden length: "
            f"actual={audio_hidden_states.shape[1]} expected={expected_20ms_frames}"
        )
    audio_hidden_states = _group_streaming_audio_hidden_states(
        model=model,
        audio_hidden_states=audio_hidden_states,
        num_80ms_tokens=int(num_80ms_tokens),
        frame_len=downsample_factor,
    )
    audio_embeds = model.multi_modal_projector(audio_hidden_states).to(device=device, dtype=dtype)
    return audio_embeds, audio_outputs.past_key_values


def _encode_audio_causal(
    *, model, feature_extractor, streams, sampling_rate, dtype, device, audio_chunk_tokens, downsample_factor
):
    """Encode the full (already padded) audio to per-80ms-token embeds, causally.

    The audio tower is fully causal (causal conv + causal sliding-window attention), so a
    single forward over the whole audio yields exactly the streaming per-frame embeds. mel
    is extracted ONCE here (vs once-per-window in the legacy windowed decode).

    audio_chunk_tokens>0 splits the encode into sequential chunks carrying the encoder KV
    cache + conv padding cache, bounding memory/compute for long audio; 0 = single pass.
    """
    feature_batch = feature_extractor(
        [streams[i] for i in range(streams.shape[0])],
        sampling_rate=sampling_rate,
        padding="longest",
        return_tensors="pt",
        center=True,
    )
    feats = feature_batch["input_features"].to(device=device, dtype=dtype)
    if audio_chunk_tokens and audio_chunk_tokens > 0:
        # mel frames per 80ms token = downsample_factor * 2 (20ms tower frames) ... derive
        # from feats length vs expected tokens instead of hardcoding: chunk along mel time.
        total_mel = int(feats.shape[-1])
        # tokens ~ total_mel / mel_per_token; compute mel_per_token from a 1-token probe is
        # overkill — chunk by a safe mel width and rely on causal conv padding cache.
        embeds_chunks = []
        encoder_pkv = None
        # mel width per chunk: audio_chunk_tokens tokens worth of mel.
        # mel_per_token estimated as total_mel / ceil(total tokens). We instead split feats
        # into fixed mel windows; downsample alignment is preserved because each chunk's mel
        # length is a multiple of mel_per_token (enforced by caller via padding).
        mel_per_token = max(1, total_mel // max(1, (total_mel // (downsample_factor * 2))))
        step = audio_chunk_tokens * downsample_factor * 2
        for start in range(0, total_mel, step):
            chunk = feats[:, :, start : start + step]
            out = model.audio_tower(input_features=chunk, past_key_values=encoder_pkv, use_cache=True, return_dict=True)
            encoder_pkv = getattr(out, "past_key_values", None)
            hs = out.last_hidden_state
            hs = hs.reshape(hs.shape[0], -1, int(model.config.audio_config.hidden_size) * downsample_factor)
            embeds_chunks.append(model.multi_modal_projector(hs).to(device=device, dtype=dtype))
        return torch.cat(embeds_chunks, dim=1)
    with torch.inference_mode():
        return model.get_audio_features(input_features=feats).to(device=device, dtype=dtype)
