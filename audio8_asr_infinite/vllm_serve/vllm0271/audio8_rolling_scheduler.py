# SPDX-License-Identifier: Apache-2.0
"""Scheduler subclass adding a rolling context window for Audio8 ASR Infinite realtime.

This module is loaded at engine startup via ``--scheduler-cls`` (vLLM's
official scheduler-class plugin point, resolved by
``SchedulerConfig.get_scheduler_cls`` / ``resolve_obj_by_qualname``); the
installed vLLM wheel stays pristine.

The only override is ``_update_request_as_session``: after the native
streaming-update bookkeeping, the session's prompt ledger is trimmed back
under ``AUDIO8_REALTIME_ROLLING_CONTEXT_TOKENS`` (default 375 = 30s @ 80ms
token clock).  RoPE attention scores depend only on relative position
offsets, so re-basing the window by a constant shift is numerically exact
for the model adapter's self-managed KV (HF decoder + eager audio tower);
the paged KV of the trimmed range is simply freed.
"""

from __future__ import annotations

import os
from dataclasses import replace
from typing import Any

from vllm.logger import init_logger
from vllm.v1.engine import EngineCoreEventType
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import Request, RequestStatus, StreamingUpdate

logger = init_logger(__name__)


def _audio8_realtime_env_int(name: str, default: int = 0) -> int:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    try:
        return int(raw_value)
    except ValueError:
        return default


def _audio8_realtime_rolling_context_tokens() -> int:
    # LAW.md fixed semantics: default 375 tokens = 30s @ 80ms/token; an
    # explicit 0 disables the rolling window.
    return max(0, _audio8_realtime_env_int("AUDIO8_REALTIME_ROLLING_CONTEXT_TOKENS", 375))


def _audio8_realtime_rolling_trim_interval_tokens() -> int:
    # Throttle so trims fire in batches (default 3000ms -> 38 tokens) instead
    # of on every update: between trims the ledger grows but never beyond
    # the window bound.
    milliseconds = _audio8_realtime_env_int("AUDIO8_REALTIME_ROLLING_TRIM_INTERVAL_MS", 3000)
    if milliseconds <= 0:
        return 1
    # Audio8 ASR Infinite realtime advances one language/audio token every 80ms.
    return max(1, (milliseconds + 79) // 80)


def _audio8_realtime_rolling_stable_prefix_tokens() -> int:
    # Session header (BOS + language token + left pad) must survive rolling
    # trims: the model conditions its language/task on this prefix. 16 =
    # streaming_n_left_pad_tokens (BOS + language + 14 pads).
    return max(0, _audio8_realtime_env_int("AUDIO8_REALTIME_ROLLING_STABLE_PREFIX_TOKENS", 16))


def _audio8_realtime_trim_mm_features_for_window(
    mm_features: list[Any],
    trim_count: int,
    trim_start: int = 0,
) -> int:
    if trim_count <= 0 or not mm_features:
        return 0

    lo = trim_start
    hi = trim_start + trim_count
    kept: list[Any] = []
    dropped = 0
    for mm_feature in mm_features:
        mm_position = getattr(mm_feature, "mm_position", None)
        offset = getattr(mm_position, "offset", None)
        length = getattr(mm_position, "length", None)
        if offset is None or length is None:
            kept.append(mm_feature)
            continue
        offset = int(offset)
        length = int(length)
        if offset + length <= lo or (lo <= offset < hi):
            # Fully inside the trimmed middle range (or before it when no
            # stable prefix is kept): drop.
            dropped += 1
            continue
        if offset < hi < offset + length:
            # Straddles the trim boundary: drop rather than shift a partial
            # feature; the model re-schedules only fully aligned features.
            dropped += 1
            continue
        if offset >= hi:
            mm_feature.mm_position = replace(mm_position, offset=offset - trim_count)
        kept.append(mm_feature)

    mm_features[:] = kept
    return dropped


def _audio8_realtime_update_log_every() -> int:
    """Diagnostic cadence for AUDIO8_REALTIME_SESSION_UPDATE (0 = off).

    Each streaming window appends the connection's fed tokens plus whatever the
    engine kept from the previous window; if those two overlap, the rolling
    prompt accumulates duplicated transcript tokens, which is what makes the
    model repeat itself and finally stop.  This line makes the overlap visible:
    ``fed_tokens`` is the update's own prompt tokens, ``kept_output_tokens`` the
    previous window's sampled tokens that were folded back into the prompt.
    """
    return _audio8_realtime_env_int("AUDIO8_REALTIME_UPDATE_LOG_EVERY", 0)


def _audio8_realtime_align_trim_to_mm_features(
    mm_features: list[Any],
    trim_start: int,
    trim_count: int,
) -> tuple[int, int]:
    """Snap a rolling trim to whole audio features.

    One mm feature covers ``mm_position.length`` placeholder tokens starting at
    ``mm_position.offset``.  Cutting the ledger inside that span drops the
    feature while keeping part of its placeholder run, so those tokens reach the
    decoder without rows and every later feature is shifted relative to its
    tokens: the model then transcribes misaligned audio (it first loops on
    plausible text, and once the argmax collapses onto a non-text token the
    transcript stops for good).  The cut is therefore moved to the nearest
    feature boundaries, so a trim always removes whole features and their whole
    placeholder runs.  Returns the adjusted ``(trim_start, trim_count)``.
    """
    if trim_count <= 0 or not mm_features:
        return trim_start, trim_count

    starts: list[int] = []
    for mm_feature in mm_features:
        mm_position = getattr(mm_feature, "mm_position", None)
        offset = getattr(mm_position, "offset", None)
        if offset is None:
            continue
        starts.append(int(offset))
    if not starts:
        return trim_start, trim_count

    starts.sort()
    requested_end = trim_start + trim_count
    aligned_start = next((s for s in starts if s >= trim_start), trim_start)
    aligned_end = next((s for s in starts if s >= requested_end), requested_end)
    if aligned_end <= aligned_start:
        # Nothing feature-aligned to drop here; keep the arithmetic trim rather
        # than trimming zero tokens.
        return trim_start, trim_count
    return aligned_start, aligned_end - aligned_start


def _audio8_realtime_compute_session_trim(
    session: Request,
    window_tokens: int,
    interval_tokens: int,
) -> int:
    """Decide how many prompt-ledger tokens a streaming session should trim.

    The ledger is bounded by ``window_tokens`` AT ALL TIMES: the trim removes
    tokens as soon as the ledger would exceed the window, cutting back to
    ``window_tokens - interval_tokens`` so the next ``interval_tokens`` steps
    fit under the bound before the next trim fires.  Positions beyond the
    window are not merely a KV concern: each language row's audio sub-token
    positions are row_position * 4, and the audio tower supports exactly
    ``max_position_embeddings`` (1500 = 375 rows * 4): rows past position 375
    feed the audio RoPE out-of-range positions and corrupt the per-row audio
    features.  Bounding the ledger keeps every row in-distribution.

    Pure check without mutation so callers can release request-scoped
    resources (paged blocks, encoder-cache references) while the pre-trim
    multimodal feature indices are still valid.
    """
    if window_tokens <= 0:
        return 0

    assert session.prompt_token_ids is not None
    overflow_tokens = max(0, len(session.prompt_token_ids) - window_tokens)
    if overflow_tokens <= 0:
        return 0

    # Trim as soon as the ledger exceeds the window, cutting back far enough
    # (overflow + interval - 1) that the next trim fires a full interval later
    # while the ledger never exceeds the window in between.
    trim_count = overflow_tokens
    if interval_tokens > 1:
        trim_count += interval_tokens - 1
    trim_count = min(trim_count, max(0, session.num_computed_tokens))
    # Keep at least the stable prefix + 1 token.
    trim_count = min(trim_count, len(session.prompt_token_ids))
    return trim_count


def _audio8_realtime_apply_session_trim(
    session: Request,
    trim_count: int,
    trim_start: int = 0,
) -> int:
    """Apply a rolling trim to the streaming session's token ledger.

    The trim removes ``trim_count`` tokens starting at ``trim_start``, which is
    at or after the stable prefix, so the session header (BOS + language token +
    left pad) survives every rolling window.  The removed range is contiguous,
    so all mm feature offsets beyond it shift by ``-trim_count``.
    """
    if trim_count <= 0:
        return 0

    if trim_start > 0:
        del session.prompt_token_ids[trim_start : trim_start + trim_count]
        del session._all_token_ids[trim_start : trim_start + trim_count]
    else:
        del session.prompt_token_ids[:trim_count]
        del session._all_token_ids[:trim_count]
    session.num_computed_tokens = max(0, session.num_computed_tokens - trim_count)
    if session.num_computed_tokens > session.num_tokens:
        session.num_computed_tokens = session.num_tokens
    dropped_mm_features = _audio8_realtime_trim_mm_features_for_window(
        session.mm_features,
        trim_count,
        trim_start=trim_start,
    )
    session.block_hashes.clear()
    session.update_block_hashes()
    session._audio8_realtime_last_rolling_trim_token = len(session.prompt_token_ids)
    return dropped_mm_features


class Audio8RollingScheduler(Scheduler):
    """Native Scheduler + rolling-context trim for Audio8 ASR Infinite realtime sessions."""

    def _update_request_as_session(
        self, session: Request, update: StreamingUpdate
    ) -> None:
        """
        Updates the waiting session with the next streaming update.

        Discards the last sampled output token from the prior input chunk.
        """
        # Current streaming input behaviour: Keep only computed output tokens
        # (discard final sampled output token).
        num_computed_tokens = session.num_computed_tokens
        kept_output_tokens = session._all_token_ids[
            session.num_prompt_tokens:num_computed_tokens
        ]
        del session._all_token_ids[num_computed_tokens:]
        session._output_token_ids.clear()
        assert session.prompt_token_ids is not None
        # Extend prompt with kept output tokens.
        session.prompt_token_ids.extend(kept_output_tokens)

        if update.mm_features:
            base = session.num_tokens
            for mm_feature in update.mm_features:
                mm_feature.mm_position = replace(
                    mm_feature.mm_position,
                    offset=mm_feature.mm_position.offset + base,
                )
            session.mm_features.extend(update.mm_features)

        session._all_token_ids.extend(update.prompt_token_ids or ())
        session.prompt_token_ids.extend(update.prompt_token_ids or ())
        update_log_every = _audio8_realtime_update_log_every()
        if update_log_every:
            self._update_log_count = getattr(self, "_update_log_count", 0) + 1
            if self._update_log_count % update_log_every == 0:
                logger.info(
                    "AUDIO8_REALTIME_SESSION_UPDATE n=%s fed_tokens=%s "
                    "kept_output_tokens=%s mm_features=%s mm_feature_tokens=%s "
                    "prompt_tokens=%s",
                    self._update_log_count,
                    len(update.prompt_token_ids or ()),
                    len(kept_output_tokens),
                    len(session.mm_features),
                    sum(
                        int(getattr(getattr(f, "mm_position", None), "length", 0) or 0)
                        for f in session.mm_features
                    ),
                    len(session.prompt_token_ids),
                )
        # Update block hashes for the new tokens.
        session.update_block_hashes()

        # Audio8 ASR Infinite realtime rolling context: keep the streaming session's prompt
        # ledger bounded so a continuous session never reaches max_model_len.
        # Release request-scoped resources first, while the pre-trim
        # multimodal feature indices are still valid: the paged block table is
        # re-allocated on the next schedule for the re-based window, and the
        # encoder-cache references are dropped so no later access can index
        # mm_features with a stale pre-trim index.  The tokens that "remain
        # computed" are only bookkeeping for the Audio8 ASR Infinite HF decoder path: its KV
        # lives on the model adapter, which trims it in lockstep, so vLLM's
        # paged KV is never read for them.
        rolling_context_tokens = _audio8_realtime_rolling_context_tokens()
        rolling_trim_interval_tokens = _audio8_realtime_rolling_trim_interval_tokens()
        if rolling_context_tokens > 0:
            session.skip_reading_prefix_cache = True
        trim_count = _audio8_realtime_compute_session_trim(
            session,
            rolling_context_tokens,
            rolling_trim_interval_tokens,
        )
        if trim_count > 0:
            self.encoder_cache_manager.free(session)
            stable_prefix = min(
                _audio8_realtime_rolling_stable_prefix_tokens(),
                max(0, rolling_context_tokens - 1),
            )
            # Never cut inside an audio feature's placeholder span: a split
            # feature is dropped while its surviving placeholders lose their
            # rows and every later feature shifts, which is what turned long
            # sessions into repeated text and finally silence.
            trim_start, trim_count = _audio8_realtime_align_trim_to_mm_features(
                session.mm_features,
                stable_prefix,
                trim_count,
            )
            dropped_rolling_mm_features = _audio8_realtime_apply_session_trim(
                session,
                trim_count,
                trim_start=trim_start,
            )
            self.kv_cache_manager.free(session)
            logger.info(
                "AUDIO8_REALTIME_ROLLING_TRIM request_id=%s "
                "rolling_context_tokens=%s rolling_trim_interval_tokens=%s "
                "stable_prefix=%s trim_start=%s "
                "trimmed_rolling_tokens=%s dropped_rolling_mm_features=%s "
                "num_prompt_tokens=%s num_computed_tokens=%s",
                session.request_id,
                rolling_context_tokens,
                rolling_trim_interval_tokens,
                stable_prefix,
                trim_start,
                trim_count,
                dropped_rolling_mm_features,
                len(session.prompt_token_ids),
                session.num_computed_tokens,
            )

        session.num_prompt_tokens = len(session.prompt_token_ids)
        session.arrival_time = update.arrival_time
        session.sampling_params = update.sampling_params
        if session.status == RequestStatus.WAITING_FOR_STREAMING_REQ:
            self.num_waiting_for_streaming_input -= 1
        session.status = RequestStatus.WAITING

        if self.log_stats:
            session.record_event(EngineCoreEventType.QUEUED)
