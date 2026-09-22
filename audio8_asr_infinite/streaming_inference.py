"""Simulated-streaming inference for Audio8 ASR Infinite (Voxtral-style delay).

This module implements the definition-aligned simulated-streaming decoding used
for real inference: audio is consumed in real-time look-back / look-ahead
windows with a streaming audio KV cache, and one text token is emitted per 80ms
clock step.  Text decoding goes through
``model.forward_language_model_with_delay(...)`` so the checkpoint's
``num_delay_tokens -> t_cond -> per-layer adaptive scaling`` path is applied at
inference time.  The shared audio-windowing helpers live in
``audio8_asr_infinite.simulated_streaming_audio``.
"""

from __future__ import annotations

import os
from contextlib import nullcontext
from typing import Any, Callable, ContextManager, Sequence

import numpy as np
import torch

from audio8_asr_infinite.simulated_streaming_audio import (
    _encode_audio_causal,
    _resolve_model_frame_len,
    _precompute_streaming_conv_embeds,
    _streaming_audio_embeds_batch,
    decode_visible_text,
    iter_realtime_audio_windows,
    precompute_simulated_streaming_audio_embeds,
    streaming_audio_embeds_by_definition,
)


def _initial_prompt_token_ids(
    *,
    bos_token_id: int,
    language_token_id: int,
    streaming_pad_token_id: int,
    left_pad_tokens: int,
    num_delay_tokens: int,
) -> list[int]:
    if left_pad_tokens < 2:
        raise ValueError(
            "left_pad_tokens must be at least 2 when the "
            "language token follows BOS."
        )
    if num_delay_tokens < 0:
        raise ValueError("num_delay_tokens must be non-negative.")
    # The prompt is left-padded by the delay so that the first greedy logit
    # lands exactly on the first emission step of the 80ms clock.
    prefill_tokens = left_pad_tokens + num_delay_tokens + 1
    return [
        int(bos_token_id),
        int(language_token_id),
    ] + [
        int(streaming_pad_token_id)
    ] * (prefill_tokens - 2)


def _cache_supports_batch_select(cache: Any | None) -> bool:
    if cache is None:
        return True
    layers = getattr(cache, "layers", None)
    if layers is None:
        return callable(getattr(cache, "batch_select_indices", None))
    # Dynamic cache layers expose ``batch_select_indices``.  Static cache
    # layers intentionally do not: slicing their backing tensors would defeat
    # the fixed-address contract used by compile/cudagraph paths.  Keep the
    # full batch for those caches instead.
    return all(
        callable(getattr(layer, "batch_select_indices", None))
        for layer in layers
    )


def _batch_select_cache(
    cache: Any | None,
    indices: torch.Tensor,
) -> Any | None:
    if cache is None:
        return None
    layers = getattr(cache, "layers", None)
    if layers is None:
        cache.batch_select_indices(indices)
        return cache
    # Do this layer-by-layer so mixed cache implementations are handled
    # without assuming that every layer has the container method.
    for layer in layers:
        select = getattr(layer, "batch_select_indices", None)
        if callable(select):
            select(indices)
            continue
        raise TypeError(
            "Cache layer does not support batch compaction: "
            f"{type(layer).__name__}"
        )
    return cache


def _resolve_batch_delay_tokens(
    num_delay_tokens: int | Sequence[int] | torch.Tensor,
    batch_size: int,
) -> tuple[list[int], int]:
    if isinstance(num_delay_tokens, int):
        delay_values = [int(num_delay_tokens)] * batch_size
    elif isinstance(num_delay_tokens, torch.Tensor):
        if int(num_delay_tokens.numel()) != batch_size:
            raise ValueError(
                "num_delay_tokens tensor must contain one value per waveform: "
                f"got {int(num_delay_tokens.numel())} for batch_size "
                f"{batch_size}."
            )
        delay_values = [
            int(value)
            for value in num_delay_tokens.detach().cpu().tolist()
        ]
    else:
        raw_values = list(num_delay_tokens)
        if len(raw_values) != batch_size:
            raise ValueError(
                "num_delay_tokens sequence must contain one value per "
                f"waveform: got {len(raw_values)} for batch_size {batch_size}."
            )
        delay_values = [int(value) for value in raw_values]
    return delay_values, max(delay_values)


def simulated_streaming_greedy_decode(
    *,
    model: Any,
    tokenizer: Any,
    feature_extractor: Any,
    waveform: np.ndarray,
    language_token_id: int,
    special_ids: dict[str, int],
    audio_config: Any,
    num_delay_tokens: int,
    frame_len: int | None = None,
    right_pad_text_tokens: int,
    dtype: torch.dtype,
    device: torch.device,
    streaming_look_ahead_ms: float = 2.5,
    streaming_look_back_ms: float = 52.5,
    min_new_tokens: int = 0,
    max_new_tokens: int | None = None,
) -> dict[str, Any]:
    min_new_tokens = int(min_new_tokens)
    max_new_tokens = (
        int(max_new_tokens)
        if max_new_tokens is not None
        else None
    )
    if min_new_tokens < 0:
        raise ValueError("min_new_tokens must be non-negative.")
    if max_new_tokens is not None and max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive when set.")
    if (
        max_new_tokens is not None
        and min_new_tokens > max_new_tokens
    ):
        raise ValueError(
            "min_new_tokens cannot exceed max_new_tokens."
        )
    samples_per_token = int(audio_config.raw_audio_samples_per_token)
    real_audio_tokens = max(
        1,
        (int(waveform.shape[0]) + samples_per_token - 1)
        // samples_per_token,
    )
    effective_right_pad_text_tokens = int(right_pad_text_tokens)
    required_new_tokens = max(
        min_new_tokens,
        max_new_tokens or 0,
    )
    if required_new_tokens:
        effective_right_pad_text_tokens = max(
            effective_right_pad_text_tokens,
            required_new_tokens
            - real_audio_tokens
            - int(num_delay_tokens),
        )
    token_queue = _initial_prompt_token_ids(
        bos_token_id=special_ids["bos_token_id"],
        language_token_id=language_token_id,
        streaming_pad_token_id=(
            special_ids["streaming_pad_token_id"]
        ),
        left_pad_tokens=int(audio_config.streaming_n_left_pad_tokens),
        num_delay_tokens=int(num_delay_tokens),
    )
    windows, _ = iter_realtime_audio_windows(
        waveform,
        audio_config=audio_config,
        num_delay_tokens=num_delay_tokens,
        right_pad_text_tokens=effective_right_pad_text_tokens,
        streaming_look_ahead_ms=streaming_look_ahead_ms,
        streaming_look_back_ms=streaming_look_back_ms,
        initial_prefill_tokens=len(token_queue),
    )
    language_past_key_values = None
    audio_past_key_values = None
    seen_text_tokens = 0
    generated_token_ids: list[int] = []

    with torch.inference_mode():
        t_cond = model.build_t_cond(
            num_delay_tokens,
            batch_size=1,
            device=device,
            dtype=dtype,
            frame_len=frame_len,
        )
        for window in windows:
            num_80ms_tokens = int(window["num_tokens"])
            prompt_token_ids: list[int] = []
            for _ in range(num_80ms_tokens):
                if token_queue:
                    prompt_token_ids.append(int(token_queue.pop(0)))
                else:
                    prompt_token_ids.append(special_ids["streaming_pad_token_id"])

            audio_embeds, audio_past_key_values, audio_diag = streaming_audio_embeds_by_definition(
                model=model,
                feature_extractor=feature_extractor,
                audio_window=window["audio"],
                audio_past_key_values=audio_past_key_values,
                audio_config=audio_config,
                num_80ms_tokens=num_80ms_tokens,
                frame_len=frame_len,
                dtype=dtype,
                device=device,
            )

            input_ids = torch.tensor([prompt_token_ids], device=device, dtype=torch.long)
            inputs_embeds = model.build_text_inputs_embeds(input_ids=input_ids, audio_embeds=audio_embeds).to(dtype=dtype)
            seen_text_tokens += num_80ms_tokens
            outputs = model.forward_language_model_with_delay(
                inputs_embeds=inputs_embeds,
                attention_mask=torch.ones((1, seen_text_tokens), device=device, dtype=torch.long),
                past_key_values=language_past_key_values,
                use_cache=True,
                logits_to_keep=1,
                num_delay_tokens=num_delay_tokens,
                frame_len=frame_len,
                t_cond=t_cond,
            )
            language_past_key_values = outputs.past_key_values
            next_token_logits = outputs.logits[0, -1, :]
            # EOS is always suppressed: the streaming audio clock bounds
            # generation, so decoding never relies on emitting EOS to stop.
            next_token_logits = next_token_logits.clone()
            next_token_logits[
                int(special_ids["eos_token_id"])
            ] = float("-inf")
            next_token_id = int(next_token_logits.argmax(dim=-1).item())
            generated_token_ids.append(next_token_id)
            token_queue.append(next_token_id)
            generated_count = len(generated_token_ids)
            if (
                max_new_tokens is not None
                and generated_count >= max_new_tokens
            ):
                break

    final_text, visible_token_ids, skipped_counts = decode_visible_text(
        tokenizer=tokenizer,
        generated_token_ids=generated_token_ids,
        special_ids=special_ids,
    )
    return {
        "final_text": final_text,
        "generated_token_ids": generated_token_ids,
    }


def simulated_streaming_greedy_decode_batch(
    *,
    model: Any,
    tokenizer: Any,
    feature_extractor: Any,
    waveforms: list[np.ndarray],
    language_token_ids: Sequence[int],
    special_ids: dict[str, int],
    audio_config: Any,
    num_delay_tokens: int | Sequence[int] | torch.Tensor,
    frame_len: int | None = None,
    right_pad_text_tokens: int,
    dtype: torch.dtype,
    device: torch.device,
    streaming_look_ahead_ms: float = 2.5,
    streaming_look_back_ms: float = 52.5,
    min_new_tokens: int = 0,
    max_new_tokens: int | None = None,
    precomputed_audio_embeds: torch.Tensor | None = None,
    synchronize_across_ranks: bool = False,
    forward_context_factory: Callable[[], ContextManager[Any]] = nullcontext,
    inference_context_factory: Callable[[], ContextManager[Any]] = (
        torch.inference_mode
    ),
) -> list[dict[str, Any]]:
    batch_size = len(waveforms)
    if batch_size == 0:
        # Every rank must enter the same initial schedule collective.  A
        # configuration can be absent on one rank after stratified batching;
        # returning before this reduction would leave the other ranks stuck.
        if synchronize_across_ranks:
            if not torch.distributed.is_initialized():
                raise RuntimeError(
                    "synchronize_across_ranks requires an initialized "
                    "distributed process group."
                )
            empty_max_audio_tokens = torch.zeros(
                (),
                dtype=torch.long,
                device=device,
            )
            torch.distributed.all_reduce(
                empty_max_audio_tokens,
                op=torch.distributed.ReduceOp.MAX,
            )
        return []
    frame_len = _resolve_model_frame_len(model, frame_len)
    if len(language_token_ids) != batch_size:
        raise ValueError(
            "language_token_ids must match waveforms: "
            f"{len(language_token_ids)} != {batch_size}."
        )
    min_new_tokens = int(min_new_tokens)
    max_new_tokens = (
        int(max_new_tokens)
        if max_new_tokens is not None
        else None
    )
    if min_new_tokens < 0:
        raise ValueError("min_new_tokens must be non-negative.")
    if max_new_tokens is not None and max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive when set.")
    if (
        max_new_tokens is not None
        and min_new_tokens > max_new_tokens
    ):
        raise ValueError(
            "min_new_tokens cannot exceed max_new_tokens."
        )
    delay_values, max_delay_tokens = _resolve_batch_delay_tokens(
        num_delay_tokens,
        batch_size,
    )
    heterogeneous_delays = len(set(delay_values)) > 1
    samples_per_token = int(audio_config.raw_audio_samples_per_token)
    left_pad_tokens = int(audio_config.streaming_n_left_pad_tokens)
    sampling_rate = int(audio_config.sampling_rate)
    # Delay now varies per row. The window *schedule* is built from the largest
    # delay so one common clock covers every row.
    # Heterogeneous rows are LEFT-padded so every row shares the same
    # max-delay timeline length: a shorter-delay row's real audio is shifted
    # right by ``(max_delay - its_delay)`` tokens (batch-consistent leading
    # silence), which exactly cancels its smaller right pad.  This makes the
    # whole batch uniform in prefill and total length, so the decode no longer
    # relies on FlashAttention varlen padding-mask compaction and works with a
    # plain SDPA/eager attention backend.
    prefill_lens = [
        left_pad_tokens + max_delay_tokens + 1
        for _ in delay_values
    ]
    max_prefill_tokens = max(prefill_lens)

    import math as _math

    real_audio_tokens = [max(1, _math.ceil(int(w.shape[0]) / samples_per_token)) for w in waveforms]
    total_counts = [
        left_pad_tokens
        + max_delay_tokens
        + int(real_audio_tokens[row_index])
        + 1
        + int(right_pad_text_tokens)
        for row_index in range(batch_size)
    ]
    max_audio_tokens = max(real_audio_tokens)
    if synchronize_across_ranks:
        if not torch.distributed.is_initialized():
            raise RuntimeError(
                "synchronize_across_ranks requires an initialized "
                "distributed process group."
            )
        global_max_audio_tokens = torch.tensor(
            max_audio_tokens,
            dtype=torch.long,
            device=device,
        )
        torch.distributed.all_reduce(
            global_max_audio_tokens,
            op=torch.distributed.ReduceOp.MAX,
        )
        max_audio_tokens = int(global_max_audio_tokens.item())

    # Test/eval generation limits count one greedy output per simulated-streaming
    # window. Extend only the right-side silence so short utterances can still
    # reach the requested minimum without changing their real audio.
    effective_right_pad_text_tokens_list = [
        int(right_pad_text_tokens)
        for _ in range(batch_size)
    ]
    required_new_tokens = max(
        min_new_tokens,
        max_new_tokens or 0,
    )
    if required_new_tokens:
        if heterogeneous_delays:
            effective_right_pad_text_tokens_list = [
                max(
                    int(right_pad_text_tokens),
                    int(required_new_tokens)
                    - int(real_audio_tokens[row_index])
                    - int(delay_values[row_index]),
                )
                for row_index in range(batch_size)
            ]
        else:
            homogeneous_right_pad = max(
                int(right_pad_text_tokens),
                int(required_new_tokens)
                - max_audio_tokens
                - int(delay_values[0]),
            )
            effective_right_pad_text_tokens_list = [
                homogeneous_right_pad
                for _ in range(batch_size)
            ]
    effective_right_pad_text_tokens = max(
        effective_right_pad_text_tokens_list
    )

    dummy = np.zeros(max_audio_tokens * samples_per_token, dtype=np.float32)
    windows_iter, timeline = iter_realtime_audio_windows(
        dummy,
        audio_config=audio_config,
        num_delay_tokens=max_delay_tokens,
        right_pad_text_tokens=effective_right_pad_text_tokens,
        streaming_look_ahead_ms=streaming_look_ahead_ms,
        streaming_look_back_ms=streaming_look_back_ms,
        initial_prefill_tokens=max_prefill_tokens,
    )
    windows = list(windows_iter)
    stream_total_samples = int(timeline["stream_total_samples"])

    streams = None
    if precomputed_audio_embeds is None:
        streams_cpu = np.zeros(
            (batch_size, stream_total_samples),
            dtype=np.float32,
        )
        for i, waveform in enumerate(waveforms):
            n = int(waveform.shape[0])
            # Left-pad each row's real audio so all rows share the max-delay
            # timeline length (batch-consistent leading silence).
            offset = (
                left_pad_tokens + (max_delay_tokens - int(delay_values[i]))
            ) * samples_per_token
            streams_cpu[i, offset : offset + n] = waveform.astype(
                np.float32,
                copy=False,
            )
        # Keep the padded source streams resident for this configuration. The
        # previous path copied every window through the host feature extractor.
        streams = torch.as_tensor(
            streams_cpu,
            device=device,
            dtype=torch.float32,
        )
        del streams_cpu

    pad_id = int(special_ids["streaming_pad_token_id"])
    seen_text_tokens = 0

    # Dynamic caches are required here because completed rows can be removed
    # from a streaming batch. StaticCache keeps fixed backing addresses and
    # cannot compact its batch dimension; retaining it across windows also
    # raises the per-window memory high-water mark.
    language_past_key_values = None
    audio_past_key_values = None
    # Only the first prefill needs a prompt matrix. After that every window
    # consumes the previous greedy token, so retain one next-token vector
    # instead of a dense [batch, timeline] schedule.
    initial_prompt = torch.full(
        (batch_size, int(windows[0]["num_tokens"])),
        pad_id,
        device=device,
        dtype=torch.long,
    )
    initial_prompt[:, 0] = int(special_ids["bos_token_id"])
    initial_prompt[:, 1] = torch.as_tensor(
        [int(token_id) for token_id in language_token_ids],
        device=device,
        dtype=torch.long,
    )
    next_input_ids = torch.full(
        (batch_size,),
        pad_id,
        device=device,
        dtype=torch.long,
    )
    generated_tokens = torch.full(
        (batch_size, len(windows)),
        pad_id,
        device=device,
        dtype=torch.long,
    )
    generated_valid = torch.zeros(
        (batch_size, len(windows)),
        device=device,
        dtype=torch.bool,
    )
    generated_counts = torch.zeros(
        batch_size,
        device=device,
        dtype=torch.long,
    )
    total_counts_tensor = torch.as_tensor(
        total_counts,
        device=device,
        dtype=torch.long,
    )
    finished = torch.zeros(
        batch_size,
        device=device,
        dtype=torch.bool,
    )
    # Heterogeneous-delay batches pad every row's prefill to the common
    # schedule width and mask the padded slots. FA2/FA3 consume the 2D mask
    # as a varlen padding mask: masked slots are dropped from the unpadded
    # sequence, so each row's real tokens see exactly the canonical
    # per-delay KV and RoPE positions.
    full_attention_mask = None
    current_prefill_lens = None
    if heterogeneous_delays:
        full_attention_mask = torch.ones(
            (batch_size, len(windows)),
            device=device,
            dtype=torch.bool,
        )
        prefill_cols = torch.arange(
            max_prefill_tokens,
            device=device,
            dtype=torch.long,
        ).unsqueeze(0)
        padded_prefill = prefill_cols >= torch.as_tensor(
            prefill_lens,
            device=device,
            dtype=torch.long,
        ).unsqueeze(1)
        full_attention_mask[:, :max_prefill_tokens] = ~padded_prefill
        current_prefill_lens = torch.as_tensor(
            prefill_lens,
            device=device,
            dtype=torch.long,
        )
    active_indices = torch.arange(
        batch_size,
        device=device,
        dtype=torch.long,
    )
    # The cache grows by the homogeneous window width on every tick. Reuse
    # these positions instead of launching a fresh arange in the Transformers
    # decoder for each short forward.
    cache_positions: list[torch.Tensor] = []
    audio_position_ids: list[torch.Tensor] = []
    cache_position_start = 0
    audio_position_start = 0
    for window in windows:
        window_width = int(window["num_tokens"])
        cache_positions.append(
            torch.arange(
                cache_position_start,
                cache_position_start + window_width,
                device=device,
                dtype=torch.long,
            )
        )
        cache_position_start += window_width
        audio_width = window_width * int(frame_len)
        audio_position_ids.append(
            torch.arange(
                audio_position_start,
                audio_position_start + audio_width,
                device=device,
                dtype=torch.long,
            ).unsqueeze(0)
        )
        audio_position_start += audio_width
    # In synchronized multi-rank decoding the cache shape must stay identical on
    # every rank.  We compact the *union* of locally active rows after an
    # all-reduce, so a row is kept whenever any rank still needs it.  This
    # preserves the distributed collective schedule while avoiding forwards for
    # rows that have finished everywhere.
    compact_finished_rows = True
    cache_compaction_checked = False
    cache_supports_batch_select: bool | None = None
    precomputed_conv_embeds = None
    t_cond = None
    with inference_context_factory():
        # FSDP2 keeps module parameters as DTensors outside its forward
        # context.  `build_t_cond` uses the frame-length embedding directly,
        # so materialize that parameter for the short construction step too.
        with forward_context_factory():
            t_cond = model.build_t_cond(
                (
                    torch.as_tensor(
                        delay_values,
                        device=device,
                        dtype=dtype,
                    )
                    if heterogeneous_delays
                    else delay_values[0]
                ),
                batch_size=batch_size,
                device=device,
                dtype=dtype,
                frame_len=frame_len,
            )
        if precomputed_audio_embeds is None:
            # Frontend work is independent across windows. Batch it once while
            # retaining one audio-tower call per clock so its KV-cache
            # semantics stay identical to the canonical simulated stream.
            with forward_context_factory():
                precomputed_conv_embeds = _precompute_streaming_conv_embeds(
                    model=model,
                    feature_extractor=feature_extractor,
                    streams=streams,
                    windows=windows,
                    frame_len=frame_len,
                    sampling_rate=sampling_rate,
                    dtype=dtype,
                    device=device,
                )
        for window_index, window in enumerate(windows):
            num_80ms_tokens = int(window["num_tokens"])
            frame_start = int(window["frame_start_sample"])
            frame_end = int(window["frame_end_sample"])
            window_start = seen_text_tokens
            window_end = window_start + num_80ms_tokens
            seen_text_tokens += num_80ms_tokens
            if heterogeneous_delays:
                # Window 0 covers [0, prefill_i), window k covers the token
                # at canonical position prefill_i + k - 1.
                current_targets = (
                    current_prefill_lens + window_index - 1
                )
            else:
                target_index = seen_text_tokens - 1

            if active_indices.numel():
                if active_indices.numel() == batch_size:
                    current_indices = active_indices
                    current_t_cond = t_cond
                    input_ids = (
                        initial_prompt
                        if window_index == 0
                        else next_input_ids.unsqueeze(1)
                    )
                    current_counts = generated_counts
                    current_finished = finished
                    current_total_counts = total_counts_tensor
                else:
                    current_indices = active_indices
                    current_t_cond = t_cond.index_select(0, current_indices)
                    if window_index == 0:
                        input_ids = initial_prompt.index_select(
                            0,
                            current_indices,
                        )
                    else:
                        input_ids = next_input_ids.index_select(
                            0,
                            current_indices,
                        ).unsqueeze(1)
                    current_counts = generated_counts.index_select(
                        0,
                        current_indices,
                    )
                    current_finished = finished.index_select(
                        0,
                        current_indices,
                    )
                    current_total_counts = total_counts_tensor.index_select(
                        0,
                        current_indices,
                    )

                window_conv_embeds = None
                if precomputed_audio_embeds is not None:
                    if current_indices.numel() == batch_size:
                        audio_embeds = precomputed_audio_embeds[
                            :, window_start:window_end, :
                        ]
                    else:
                        audio_embeds = precomputed_audio_embeds.index_select(
                            0,
                            current_indices,
                        )[:, window_start:window_end, :]
                    audio_past_key_values = None
                elif precomputed_conv_embeds is not None:
                    window_conv_embeds = precomputed_conv_embeds[window_index]
                    if current_indices.numel() != batch_size:
                        if window_conv_embeds is not None:
                            window_conv_embeds = window_conv_embeds.index_select(
                                0,
                                current_indices,
                            )
                if precomputed_audio_embeds is None and window_conv_embeds is None:
                    if current_indices.numel() == batch_size:
                        # STFT accepts this strided view; avoid copying the
                        # full batch to a new contiguous buffer on every tick.
                        audio_windows = streams[:, frame_start:frame_end]
                    else:
                        audio_windows = streams.index_select(
                            0,
                            current_indices,
                        )[:, frame_start:frame_end]
                elif precomputed_audio_embeds is None:
                    # Frontend output is already resident on the target
                    # device; avoid slicing/copying the source waveform.
                    audio_windows = None
                if precomputed_audio_embeds is None:
                    with forward_context_factory():
                        audio_embeds, audio_past_key_values = (
                            _streaming_audio_embeds_batch(
                                model=model,
                                feature_extractor=feature_extractor,
                                audio_windows=audio_windows,
                                audio_past_key_values=audio_past_key_values,
                                sampling_rate=sampling_rate,
                                num_80ms_tokens=num_80ms_tokens,
                                frame_len=frame_len,
                                precomputed_conv_embeds=window_conv_embeds,
                                position_ids=audio_position_ids[window_index],
                                dtype=dtype,
                                device=device,
                            )
                        )
                if precomputed_conv_embeds is not None:
                    precomputed_conv_embeds[window_index] = None
                window_attention_mask = None
                window_position_ids = None
                if heterogeneous_delays:
                    window_attention_mask = full_attention_mask[
                        :, : max_prefill_tokens + window_index
                    ]
                    if window_index > 0:
                        # The cache slot is shared at max_prefill + k, but
                        # each row's RoPE position must stay canonical.
                        window_position_ids = (
                            current_prefill_lens + window_index - 1
                        ).unsqueeze(1)
                with forward_context_factory():
                    inputs_embeds = model.build_text_inputs_embeds(
                        input_ids=input_ids,
                        audio_embeds=audio_embeds,
                    ).to(dtype=dtype)
                    outputs = model.forward_language_model_with_delay(
                        inputs_embeds=inputs_embeds,
                        attention_mask=window_attention_mask,
                        past_key_values=language_past_key_values,
                        use_cache=True,
                        logits_to_keep=(
                            0
                            if (
                                heterogeneous_delays
                                and window_index == 0
                            )
                            else 1
                        ),
                        cache_position=cache_positions[window_index],
                        position_ids=window_position_ids,
                        num_delay_tokens=max_delay_tokens,
                        frame_len=frame_len,
                        t_cond=current_t_cond,
                    )
                language_past_key_values = outputs.past_key_values
                if cache_supports_batch_select is None:
                    cache_supports_batch_select = (
                        _cache_supports_batch_select(
                            language_past_key_values
                        )
                        and _cache_supports_batch_select(
                            audio_past_key_values
                        )
                )
                if heterogeneous_delays and window_index == 0:
                    # Each row's first greedy logit comes from its own last
                    # real prefill position, not the padded schedule tail.
                    full_prefill_logits = outputs.logits
                    row_indices = torch.arange(
                        full_prefill_logits.shape[0],
                        device=device,
                        dtype=torch.long,
                    )
                    next_token_logits = full_prefill_logits[
                        row_indices,
                        (current_prefill_lens - 1).clamp_min(0),
                        :,
                    ]
                    del full_prefill_logits
                    # Release the [rows, max_prefill, vocab] logits before the
                    # remaining per-window bookkeeping and next forward.
                    outputs = None
                else:
                    next_token_logits = outputs.logits[:, -1, :]
                # Decide which rows actually emit a token. Finished rows only
                # carry padding forward.
                if max_new_tokens is None:
                    below_maximum = torch.ones_like(
                        current_counts,
                        dtype=torch.bool,
                    )
                else:
                    below_maximum = current_counts < int(max_new_tokens)
                if heterogeneous_delays:
                    target_within_counts = (
                        current_targets < current_total_counts
                    )
                else:
                    target_within_counts = (
                        target_index < current_total_counts
                    )
                active = (
                    ~current_finished
                    & below_maximum
                    & (
                        target_within_counts
                        | (current_counts < int(min_new_tokens))
                    )
                )
                # EOS is always suppressed: the streaming audio clock bounds
                # generation, so decoding never relies on emitting EOS to stop.
                next_token_logits = next_token_logits.clone()
                next_token_logits[
                    :, int(special_ids["eos_token_id"])
                ] = float("-inf")
                next_token_ids = next_token_logits.argmax(dim=-1)
                generated_tokens[current_indices, window_index] = (
                    next_token_ids.masked_fill(~active, pad_id)
                )
                generated_valid[current_indices, window_index] = active
                updated_counts = current_counts + active.to(
                    dtype=current_counts.dtype
                )
                generated_counts[current_indices] = updated_counts
                if max_new_tokens is None:
                    max_now = torch.zeros_like(active)
                else:
                    max_now = active & (
                        updated_counts >= int(max_new_tokens)
                    )
                finished_now = current_finished | max_now
                finished[current_indices] = finished_now
                next_input_ids.index_copy_(
                    0,
                    current_indices,
                    next_token_ids.masked_fill(~active, pad_id),
                )
                if heterogeneous_delays:
                    next_active = (
                        ~finished_now
                        & (
                            (current_targets + 1 < current_total_counts)
                            | (updated_counts < int(min_new_tokens))
                        )
                    )
                else:
                    next_active = (
                        ~finished_now
                        & (
                            (target_index + 1 < current_total_counts)
                            | (updated_counts < int(min_new_tokens))
                        )
                    )
            else:
                next_active = torch.zeros(
                    0,
                    device=device,
                    dtype=torch.bool,
                )

            if synchronize_across_ranks:
                if compact_finished_rows and not cache_compaction_checked:
                    # DynamicCache supports batch selection; StaticCache does
                    # not. Agree on the capability before the first row-mask
                    # collective so all ranks take the same branch.
                    local_support = int(cache_supports_batch_select)
                    support_tensor = torch.tensor(
                        local_support,
                        dtype=torch.int32,
                        device=device,
                    )
                    torch.distributed.all_reduce(
                        support_tensor,
                        op=torch.distributed.ReduceOp.MIN,
                    )
                    compact_finished_rows = bool(support_tensor.item())
                    cache_compaction_checked = True
                if compact_finished_rows:
                    # A local row can finish earlier than its peer on another
                    # rank. Keep the global OR so cache dimensions and FSDP
                    # forward participation remain rank-identical.
                    global_next_active = next_active.to(dtype=torch.int32)
                    torch.distributed.all_reduce(
                        global_next_active,
                        op=torch.distributed.ReduceOp.MAX,
                    )
                    next_active = global_next_active.to(dtype=torch.bool)
                else:
                    # Without batch-select support, retain the fixed schedule
                    # through the last window. The loop itself is identical on
                    # every rank because max_audio_tokens was synchronized.
                    any_active_next = True

            can_compact = (
                compact_finished_rows
                and bool(cache_supports_batch_select)
                and active_indices.numel() > 0
            )
            if can_compact:
                keep_positions = torch.nonzero(
                    next_active,
                    as_tuple=False,
                ).flatten()
                if keep_positions.numel() < active_indices.numel():
                    _batch_select_cache(
                        language_past_key_values,
                        keep_positions,
                    )
                    _batch_select_cache(
                        audio_past_key_values,
                        keep_positions,
                    )
                    active_indices = active_indices.index_select(
                        0,
                        keep_positions,
                    )
                    if heterogeneous_delays:
                        full_attention_mask = (
                            full_attention_mask.index_select(
                                0,
                                keep_positions,
                            )
                        )
                        current_prefill_lens = (
                            current_prefill_lens.index_select(
                                0,
                                keep_positions,
                            )
                        )
                # `active_indices` has a host-known shape, so this avoids a
                # GPU-to-CPU synchronization on every streaming clock tick.
                any_active_next = active_indices.numel() > 0
            elif not synchronize_across_ranks:
                any_active_next = bool(next_active.any().item())
            if not any_active_next:
                break

    # Drop the cache and resident source tensor before materializing the small
    # CPU result objects. This keeps the decode allocations from overlapping the
    # next call's forwards.
    outputs = None
    audio_embeds = None
    inputs_embeds = None
    input_ids = None
    audio_windows = None
    current_t_cond = None
    next_token_logits = None
    next_token_ids = None
    current_indices = None
    current_counts = None
    current_finished = None
    current_total_counts = None
    updated_counts = None
    active = None
    next_active = None
    del language_past_key_values, audio_past_key_values, streams
    precomputed_audio_embeds = None
    del precomputed_conv_embeds
    del initial_prompt, next_input_ids
    del generated_counts, total_counts_tensor, finished
    del active_indices, t_cond, cache_positions, audio_position_ids
    del full_attention_mask, current_prefill_lens
    generated_tokens_cpu = generated_tokens.detach().cpu().tolist()
    generated_valid_cpu = generated_valid.detach().cpu().tolist()
    results: list[dict[str, Any]] = []
    for i in range(batch_size):
        valid_steps = [
            step
            for step, is_valid in enumerate(generated_valid_cpu[i])
            if is_valid
        ]
        generated = [
            int(generated_tokens_cpu[i][step])
            for step in valid_steps
        ]
        final_text, visible_token_ids, skipped_counts = decode_visible_text(
            tokenizer=tokenizer,
            generated_token_ids=generated,
            special_ids=special_ids,
        )
        results.append(
            {
                "final_text": final_text,
                "generated_token_ids": generated,
            }
        )
    return results


def transformers_streaming_greedy_decode_batch(
    *,
    model: Any,
    tokenizer: Any,
    feature_extractor: Any,
    waveforms: list[np.ndarray],
    language_token_ids: Sequence[int],
    special_ids: dict[str, int],
    audio_config: Any,
    num_delay_tokens: int,
    frame_len: int | None = None,
    right_pad_text_tokens: int,
    dtype: torch.dtype,
    device: torch.device,
    max_new_tokens: int | None = None,
    forward_context_factory: Callable[[], ContextManager[Any]] = nullcontext,
) -> list[dict[str, Any]]:
    """Decode through the native Transformers streaming-generation interface."""

    batch_size = len(waveforms)
    if batch_size == 0:
        return []
    frame_len = _resolve_model_frame_len(model, frame_len)
    samples_per_token = int(audio_config.raw_audio_samples_per_token)
    sampling_rate = int(audio_config.sampling_rate)
    left_pad_tokens = int(audio_config.streaming_n_left_pad_tokens)
    right_pad_tokens = (
        int(num_delay_tokens) + 1 + int(right_pad_text_tokens)
    )
    real_audio_tokens = [
        max(
            1,
            (
                int(waveform.shape[0])
                + samples_per_token
                - 1
            )
            // samples_per_token,
        )
        for waveform in waveforms
    ]
    maximum_real_audio_tokens = max(real_audio_tokens)
    total_audio_tokens = (
        left_pad_tokens
        + maximum_real_audio_tokens
        + right_pad_tokens
    )
    stream_samples = total_audio_tokens * samples_per_token
    streams = np.zeros(
        (batch_size, stream_samples),
        dtype=np.float32,
    )
    waveform_offset = left_pad_tokens * samples_per_token
    for index, waveform in enumerate(waveforms):
        waveform_samples = int(waveform.shape[0])
        streams[
            index,
            waveform_offset : waveform_offset + waveform_samples,
        ] = waveform.astype(np.float32, copy=False)

    prompt_rows = [
        _initial_prompt_token_ids(
            bos_token_id=special_ids["bos_token_id"],
            language_token_id=language_token_ids[index],
            streaming_pad_token_id=(
                special_ids["streaming_pad_token_id"]
            ),
            left_pad_tokens=left_pad_tokens,
            num_delay_tokens=int(num_delay_tokens),
        )
        for index in range(batch_size)
    ]
    input_ids = torch.tensor(
        prompt_rows,
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.ones_like(input_ids)
    prompt_token_count = int(input_ids.shape[1])

    # Match VoxtralRealtimeProcessor: one centered prefill followed by
    # non-centered 80 ms chunks whose convolution overlap lives in padding_cache.
    mel_frames_per_token = int(model.config.audio_length_per_tok)
    hop_length = int(feature_extractor.hop_length)
    window_length = int(feature_extractor.win_length)
    first_chunk_stream_mel_frames = (
        prompt_token_count * mel_frames_per_token
    )
    first_chunk_stream_samples = (
        (first_chunk_stream_mel_frames - 1) * hop_length
        + window_length // 2
    )
    first_features = feature_extractor(
        [stream[:first_chunk_stream_samples] for stream in streams],
        sampling_rate=sampling_rate,
        padding="longest",
        return_tensors="pt",
        center=True,
    )["input_features"].to(device=device, dtype=dtype)
    feature_chunks = [first_features]

    mel_frame_index = first_chunk_stream_mel_frames
    samples_per_chunk = (
        mel_frames_per_token * hop_length + window_length
    )
    chunk_start = (
        mel_frame_index * hop_length - window_length // 2
    )
    while (
        chunk_start + samples_per_chunk < stream_samples
        and (
            max_new_tokens is None
            or len(feature_chunks) < int(max_new_tokens)
        )
    ):
        chunk_end = chunk_start + samples_per_chunk
        chunk_features = feature_extractor(
            [stream[chunk_start:chunk_end] for stream in streams],
            sampling_rate=sampling_rate,
            padding="longest",
            return_tensors="pt",
            center=False,
        )["input_features"].to(device=device, dtype=dtype)
        feature_chunks.append(chunk_features)
        mel_frame_index += mel_frames_per_token
        chunk_start = (
            mel_frame_index * hop_length - window_length // 2
        )

    def feature_generator():
        yield from feature_chunks

    # Audio8 ASR Infinite's GenerationMixin hooks consume one feature chunk per greedy token
    # and stop generation when the stream is exhausted.
    generate_kwargs: dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "input_features": feature_generator(),
        "num_delay_tokens": int(num_delay_tokens),
        "frame_len": int(frame_len),
        "do_sample": False,
        "use_cache": True,
        "max_new_tokens": len(feature_chunks) + 1,
        "eos_token_id": int(special_ids["eos_token_id"]),
        "pad_token_id": int(special_ids["pad_token_id"]),
    }
    with torch.inference_mode(), forward_context_factory():
        output_ids = model.generate(**generate_kwargs)

    generated_rows = (
        output_ids[:, prompt_token_count:].detach().cpu().tolist()
    )
    first_chunk_mel_frames = (
        (prompt_token_count - left_pad_tokens)
        * mel_frames_per_token
    )
    first_chunk_samples = (
        (first_chunk_mel_frames - 1) * hop_length
        + window_length // 2
    )
    results: list[dict[str, Any]] = []
    for index, generated_row in enumerate(generated_rows):
        generated_token_ids = [
            int(token_id) for token_id in generated_row
        ]
        final_text, visible_token_ids, skipped_counts = (
            decode_visible_text(
                tokenizer=tokenizer,
                generated_token_ids=generated_token_ids,
                special_ids=special_ids,
            )
        )
        eos_offset = next(
            (
                offset
                for offset, token_id in enumerate(
                    generated_token_ids
                )
                if token_id == special_ids["eos_token_id"]
            ),
            None,
        )
        generated_target_indices = [
            prompt_token_count - 1 + offset
            for offset in range(len(generated_token_ids))
        ]
        results.append(
            {
                "final_text": final_text,
                "generated_token_ids": generated_token_ids,
                "generated_target_indices": generated_target_indices,
                "generated_topk_ids": [],
                "visible_token_ids": visible_token_ids,
                "special_token_ids": special_ids,
                "skipped_special_token_counts": skipped_counts,
                "eos_target_index": (
                    None
                    if eos_offset is None
                    else prompt_token_count - 1 + eos_offset
                ),
                "timeline": {
                    "streaming_interface": (
                        "transformers_input_features_generator"
                    ),
                    "real_audio_tokens": real_audio_tokens[index],
                    "left_pad_tokens": left_pad_tokens,
                    "right_pad_tokens": right_pad_tokens,
                    "prompt_token_count": prompt_token_count,
                    "first_chunk_samples": first_chunk_samples,
                    "first_chunk_audio_ms": (
                        1000.0
                        * first_chunk_samples
                        / sampling_rate
                    ),
                    "first_chunk_mel_frames": (
                        first_chunk_mel_frames
                    ),
                    "first_chunk_stream_samples": (
                        first_chunk_stream_samples
                    ),
                    "first_chunk_stream_mel_frames": (
                        first_chunk_stream_mel_frames
                    ),
                    "feature_chunk_count": len(feature_chunks),
                },
                "frames": [],
            }
        )
    return results


def causal_streaming_greedy_decode_batch(
    *,
    model,
    tokenizer,
    feature_extractor,
    waveforms,
    language_token_ids,
    special_ids,
    audio_config,
    num_delay_tokens,
    right_pad_text_tokens,
    dtype,
    device,
    audio_chunk_tokens: int = 0,
):
    import math as _math

    batch_size = len(waveforms)
    if batch_size == 0:
        return []
    if len(language_token_ids) != batch_size:
        raise ValueError(
            "language_token_ids must match waveforms: "
            f"{len(language_token_ids)} != {batch_size}."
        )
    spt = int(audio_config.raw_audio_samples_per_token)
    left_pad = int(audio_config.streaming_n_left_pad_tokens)
    sampling_rate = int(audio_config.sampling_rate)
    right_pad_tokens = int(num_delay_tokens) + 1 + int(right_pad_text_tokens)
    frame_len = _resolve_model_frame_len(model)
    pad_id = int(special_ids["streaming_pad_token_id"])

    real_tokens = [max(1, _math.ceil(int(w.shape[0]) / spt)) for w in waveforms]
    total_counts = [left_pad + r + right_pad_tokens for r in real_tokens]
    max_audio_tokens = max(real_tokens)
    stream_len = (left_pad + max_audio_tokens + right_pad_tokens) * spt
    streams = np.zeros((batch_size, stream_len), dtype=np.float32)
    offset = left_pad * spt
    for i, w in enumerate(waveforms):
        streams[i, offset : offset + int(w.shape[0])] = w.astype(np.float32, copy=False)

    ae = _encode_audio_causal(
        model=model,
        feature_extractor=feature_extractor,
        streams=streams,
        sampling_rate=sampling_rate,
        dtype=dtype,
        device=device,
        audio_chunk_tokens=audio_chunk_tokens,
        downsample_factor=frame_len,
    )
    max_total = max(total_counts)
    if int(ae.shape[1]) < max_total:
        raise ValueError(f"audio embeds {tuple(ae.shape)} shorter than timeline {max_total}")

    generated = [[] for _ in range(batch_size)]

    with torch.inference_mode():
        prompt = [
            _initial_prompt_token_ids(
                bos_token_id=special_ids["bos_token_id"],
                language_token_id=language_token_ids[index],
                streaming_pad_token_id=pad_id,
                left_pad_tokens=left_pad,
                num_delay_tokens=int(num_delay_tokens),
            )
            for index in range(batch_size)
        ]
        prefill_tokens = len(prompt[0])
        ids = torch.tensor(prompt, device=device, dtype=torch.long)
        ie = model.build_text_inputs_embeds(
            input_ids=ids,
            audio_embeds=ae[:, :prefill_tokens, :],
        ).to(dtype)
        out = model.forward_language_model_with_delay(
            inputs_embeds=ie,
            attention_mask=torch.ones(
                (batch_size, prefill_tokens),
                device=device,
                dtype=torch.long,
            ),
            use_cache=True,
            num_delay_tokens=num_delay_tokens,
        )
        pkv = out.past_key_values
        last = out.logits[:, -1, :].argmax(dim=-1)
        ti = prefill_tokens - 1
        for i in range(batch_size):
            if ti < total_counts[i]:
                token_id = int(last[i].item())
                generated[i].append(token_id)
        for token_index in range(prefill_tokens, max_total):
            step = model.build_text_inputs_embeds(
                input_ids=last.unsqueeze(1),
                audio_embeds=ae[:, token_index : token_index + 1, :],
            ).to(dtype)
            out = model.forward_language_model_with_delay(
                inputs_embeds=step,
                attention_mask=torch.ones((batch_size, token_index + 1), device=device, dtype=torch.long),
                past_key_values=pkv,
                use_cache=True,
                num_delay_tokens=num_delay_tokens,
            )
            pkv = out.past_key_values
            last = out.logits[:, -1, :].argmax(dim=-1)
            any_active = False
            for i in range(batch_size):
                if token_index < total_counts[i]:
                    tk = int(last[i].item())
                    generated[i].append(tk)
                if token_index + 1 < total_counts[i]:
                    any_active = True
            if not any_active:
                break

    results = []
    for i in range(batch_size):
        text, vis, skipped = decode_visible_text(
            tokenizer=tokenizer,
            generated_token_ids=generated[i],
            special_ids=special_ids,
        )
        results.append(
            {
                "final_text": text,
                "generated_token_ids": generated[i],
                "visible_token_ids": vis,
                "special_token_ids": special_ids,
                "skipped_special_token_counts": skipped,
                "eos_target_index": None,
            }
        )
    return results


__all__ = [
    "causal_streaming_greedy_decode_batch",
    "simulated_streaming_greedy_decode",
    "simulated_streaming_greedy_decode_batch",
    "transformers_streaming_greedy_decode_batch",
]
