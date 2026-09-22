# SPDX-License-Identifier: Apache-2.0
"""vLLM realtime adapter for Audio8 ASR Infinite.

The Audio8 ASR Infinite checkpoint is a Hugging Face style composition:
`audio_tower.*`, `multi_modal_projector.*`, `delay_embedding.*`, and
`language_model.*`.  This adapter keeps those prefixes and drives generation
one configurable streaming token at a time.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import time
from itertools import count
from collections.abc import AsyncGenerator, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MethodType
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from mistral_common.audio import mel_filter_bank
from transformers import Qwen2ForCausalLM as HFQwen2ForCausalLM
from transformers import Qwen3ForCausalLM as HFQwen3ForCausalLM
from transformers import WhisperConfig
from transformers.models.voxtral_realtime import modeling_voxtral_realtime as _voxtral_realtime_modeling
from transformers.feature_extraction_utils import BatchFeature
from transformers.activations import ACT2FN

from vllm.config import ModelConfig, SpeechToTextConfig, VllmConfig
from vllm.envs import VLLM_ENGINE_ITERATION_TIMEOUT_S
from vllm.forward_context import get_forward_context
from vllm.inputs import PromptType, TokensPrompt
from vllm.logger import init_logger
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import (
    MultiModalEmbeddings,
    SupportsMultiModal,
    SupportsMRoPE,
    SupportsPP,
    SupportsRealtime,
)
from vllm.model_executor.models.module_mapping import MultiModelKeys
from vllm.model_executor.models.qwen3 import Qwen3ForCausalLM as VllmQwen3ForCausalLM
from vllm.model_executor.models.qwen2 import Qwen2ForCausalLM as VllmQwen2ForCausalLM
from vllm.model_executor.models.utils import (
    WeightsMapper,
    _flatten_embeddings,
    maybe_prefix,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.cache import BaseMultiModalProcessorCache
from vllm.multimodal.inputs import MultiModalFieldConfig, MultiModalKwargsItems
from vllm.multimodal.parse import AudioProcessorItems, MultiModalDataItems, MultiModalDataParser
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseProcessingInfo,
    PromptUpdate,
)
from vllm.multimodal.processing.processor import (
    BaseMultiModalProcessor,
    MultiModalPromptUpdates,
    PlaceholderFeaturesInfo,
    PromptReplacement,
)
from vllm.sequence import IntermediateTensors
from vllm.tokenizers import cached_tokenizer_from_config
from vllm.model_executor.models.voxtral import VoxtralEncoderModel
from audio8_asr_infinite.modeling.modeling_audio8_asr_infinite import (
    Audio8ASRInfiniteForCausalLM,
    Audio8ASRInfiniteQwen2ForCausalLM,
    resolve_qwen_language_token_id,
)
from audio8_asr_infinite.vllm_serve.vllm0271 import audio8_vad_channel

logger = init_logger(__name__)

_STREAMING_PAD_TOKEN = "[STREAMING_PAD]"
_AUDIO8_REALTIME_DELAY_FRAME_MS = 80
# 每 N 步采一次显存（驱动查询本身很便宜，但显存变化很慢，没必要每步都采）。
_AUDIO8_METRICS_GPU_SAMPLE_STEPS = 8
# RTF 滑动平均系数：0.2 与前端读数平滑一致。
_AUDIO8_METRICS_RTF_EMA_ALPHA = 0.2
_AUDIO8_REALTIME_TOKEN_DURATION_OPTIONS_MS = (80,)
_AUDIO8_REALTIME_DELAY_PROFILE_TO_MS = {
    **{
        f"{milliseconds}ms": milliseconds
        for milliseconds in range(
            _AUDIO8_REALTIME_DELAY_FRAME_MS,
            1200 + _AUDIO8_REALTIME_DELAY_FRAME_MS,
            _AUDIO8_REALTIME_DELAY_FRAME_MS,
        )
    },
    # extra selectable delay targets beyond the 80ms sweep
    "1600ms": 1600,
    "2400ms": 2400,
}


class Audio8VoxtralRealtimeFeatureExtractor:
    model_input_names = ["input_features", "attention_mask"]

    def __init__(
        self,
        *,
        feature_size: int = 128,
        sampling_rate: int = 16000,
        n_fft: int = 400,
        hop_length: int = 160,
        win_length: int = 400,
        padding_value: float = 0.0,
        return_attention_mask: bool = True,
        global_log_mel_max: float | None = 1.5,
        conv_stride: int = 2,
        block_pool_size: int = 4,
        **kwargs: object,
    ) -> None:
        _ = kwargs
        self.feature_size = int(feature_size)
        self.sampling_rate = int(sampling_rate)
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.win_length = int(win_length)
        self.padding_value = float(padding_value)
        self.return_attention_mask = bool(return_attention_mask)
        self.global_log_mel_max = None if global_log_mel_max is None else float(global_log_mel_max)
        self.conv_stride = int(conv_stride)
        self.block_pool_size = int(block_pool_size)
        self.mel_filters = torch.tensor(
            mel_filter_bank(
                num_frequency_bins=1 + self.n_fft // 2,
                num_mel_bins=self.feature_size,
                min_frequency=0.0,
                max_frequency=float(self.sampling_rate // 2),
                sampling_rate=self.sampling_rate,
            ),
            dtype=torch.float32,
        )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        **kwargs: object,
    ) -> "Audio8VoxtralRealtimeFeatureExtractor":
        _ = kwargs
        config_path = Path(pretrained_model_name_or_path) / "preprocessor_config.json"
        if not config_path.is_file():
            raise FileNotFoundError(f"Missing preprocessor_config.json under {pretrained_model_name_or_path}.")
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        return cls(**payload)

    def __call__(
        self,
        raw_speech: object,
        *,
        sampling_rate: int | None = None,
        return_tensors: str | None = None,
        return_attention_mask: bool | None = None,
        padding: str | None = "longest",
        center: bool = True,
        **kwargs: object,
    ) -> BatchFeature:
        _ = kwargs
        if sampling_rate is not None and int(sampling_rate) != self.sampling_rate:
            raise ValueError(
                f"Audio8 ASR Infinite feature extractor expected {self.sampling_rate}Hz audio, got {sampling_rate}Hz."
            )
        audio_arrays = self._normalize_audio_batch(raw_speech)
        features = [self._compute_log_mel(audio, center=center).numpy() for audio in audio_arrays]
        lengths = [feature.shape[1] for feature in features]
        target_len = max(lengths) if padding in (None, "longest") else max(lengths)
        padded_features = np.full(
            (len(features), self.feature_size, target_len),
            self.padding_value,
            dtype=np.float32,
        )
        attention_mask = np.zeros((len(features), target_len), dtype=np.int64)
        for index, feature in enumerate(features):
            padded_features[index, :, : feature.shape[1]] = feature
            attention_mask[index, : feature.shape[1]] = 1

        output: dict[str, object] = {"input_features": padded_features}
        should_return_mask = self.return_attention_mask if return_attention_mask is None else bool(return_attention_mask)
        if should_return_mask:
            output["attention_mask"] = attention_mask
        if return_tensors == "pt":
            output = {
                key: torch.as_tensor(value)
                for key, value in output.items()
            }
        return BatchFeature(output)

    def get_num_audio_tokens(self, audio_len: int) -> int:
        mel_frame_count = max(0, int(audio_len) // self.hop_length)
        encoder_frame_count = mel_frame_count // self.conv_stride
        return encoder_frame_count // self.block_pool_size

    def _normalize_audio_batch(self, raw_speech: object) -> list[np.ndarray]:
        if isinstance(raw_speech, torch.Tensor):
            raw_speech = raw_speech.detach().cpu().numpy()
        if isinstance(raw_speech, np.ndarray):
            if raw_speech.ndim == 1:
                return [raw_speech.astype(np.float32, copy=False)]
            if raw_speech.ndim == 2:
                return [sample.astype(np.float32, copy=False) for sample in raw_speech]
        if isinstance(raw_speech, (list, tuple)):
            if not raw_speech:
                return []
            if all(isinstance(item, (float, int, np.floating, np.integer)) for item in raw_speech):
                return [np.asarray(raw_speech, dtype=np.float32)]
            normalized: list[np.ndarray] = []
            for item in raw_speech:
                if isinstance(item, torch.Tensor):
                    item = item.detach().cpu().numpy()
                normalized.append(np.asarray(item, dtype=np.float32).reshape(-1))
            return normalized
        raise TypeError(f"Unsupported Audio8 ASR Infinite realtime audio input type: {type(raw_speech)}")

    def _compute_log_mel(self, audio_array: np.ndarray, *, center: bool) -> torch.Tensor:
        audio = torch.as_tensor(audio_array, dtype=torch.float32).reshape(-1)
        window = torch.hann_window(self.win_length, device=audio.device)
        stft = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            center=center,
            return_complex=True,
        )
        magnitudes = stft[..., :-1].abs() ** 2
        mel_filters = self.mel_filters.to(device=magnitudes.device, dtype=magnitudes.dtype)
        mel_spec = mel_filters.T @ magnitudes
        log_spec = torch.clamp(mel_spec, min=1e-10).log10()
        if self.global_log_mel_max is None:
            log_spec_max = log_spec.max()
        else:
            log_spec_max = torch.tensor(
                self.global_log_mel_max,
                device=log_spec.device,
                dtype=log_spec.dtype,
            )
        log_spec = torch.maximum(log_spec, log_spec_max - 8.0)
        return (log_spec + 4.0) / 4.0


def _get_if_not_none(mapping: Mapping[str, Any], *keys: str) -> Any | None:
    for key in keys:
        value = mapping.get(key)
        if value is not None:
            return value
    return None


def _set_if_missing_or_none(payload: dict[str, Any], key: str, value: Any) -> None:
    if value is not None and payload.get(key) is None:
        payload[key] = value


def _normalize_audio8_realtime_delay_profile(profile: str | None) -> str | None:
    if profile is None:
        return None
    normalized = str(profile).strip().lower().replace("_", "-")
    if not normalized:
        return None
    aliases = {
        "default": "480ms",
        "balanced": "480ms",
        "normal": "480ms",
        "low": "240ms",
        "low-latency": "240ms",
        "fast": "240ms",
        "high": "960ms",
        "stable": "960ms",
        "max": "2400ms",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized.endswith("s") and not normalized.endswith("ms"):
        try:
            seconds = float(normalized[:-1])
        except ValueError:
            return normalized
        return f"{int(round(seconds * 1000))}ms"
    if normalized.isdigit():
        return f"{int(normalized)}ms"
    return normalized


def _resolve_audio8_realtime_token_duration_ms(
    token_duration_ms: int | float | str | None = None,
    *,
    default_token_duration_ms: int = _AUDIO8_REALTIME_DELAY_FRAME_MS,
) -> int:
    value = default_token_duration_ms if token_duration_ms is None else token_duration_ms
    try:
        duration_ms = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid token_duration_ms={token_duration_ms!r}.") from exc
    if duration_ms not in _AUDIO8_REALTIME_TOKEN_DURATION_OPTIONS_MS:
        allowed = ", ".join(f"{item}ms" for item in _AUDIO8_REALTIME_TOKEN_DURATION_OPTIONS_MS)
        raise ValueError(f"Audio8 ASR Infinite realtime token_duration_ms must be one of: {allowed}.")
    return duration_ms


def _per_item_int_values(
    value: object,
    item_count: int,
    *,
    default: int = 0,
) -> list[int]:
    """Read a batched mm field as one integer per item.

    ``MultiModalFieldConfig.batched`` fields arrive stacked (tensor or list);
    ``keep_on_cpu=True`` keeps them off-device.  Falls back to a single
    broadcast value for pre-salt/dummy inputs.
    """
    if value is None:
        return [default] * item_count
    if isinstance(value, torch.Tensor):
        flat = value.detach().reshape(-1)
        values = [int(item) for item in flat.tolist()]
    elif isinstance(value, np.ndarray):
        values = [int(item) for item in value.reshape(-1).tolist()]
    elif isinstance(value, (list, tuple)):
        values = [
            int(item.detach().reshape(-1)[0].item())
            if isinstance(item, torch.Tensor)
            else int(item)
            for item in value
        ]
    else:
        values = [int(value)]
    if len(values) == 1 and item_count > 1:
        return values * item_count
    if len(values) != item_count:
        raise ValueError(
            "Audio8 ASR Infinite realtime per-item mm field must contain one value per "
            f"audio item: got {len(values)} values for {item_count} items."
        )
    return values


def _first_scalar_value(value: object) -> object | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return None
        return value.reshape(-1)[0].item()
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return None
        return value.reshape(-1)[0].item()
    if isinstance(value, (list, tuple)):
        if not value:
            return None
        return _first_scalar_value(value[0])
    return value


def _audio8_realtime_token_group_size(
    token_duration_ms: object,
    *,
    default_token_duration_ms: int = _AUDIO8_REALTIME_DELAY_FRAME_MS,
) -> int:
    duration_ms = _resolve_audio8_realtime_token_duration_ms(
        _first_scalar_value(token_duration_ms),
        default_token_duration_ms=default_token_duration_ms,
    )
    if duration_ms % _AUDIO8_REALTIME_DELAY_FRAME_MS != 0:
        raise ValueError(
            "Audio8 ASR Infinite realtime token duration must be a multiple of the base 80ms frame: "
            f"token_duration_ms={duration_ms}."
        )
    group_size = duration_ms // _AUDIO8_REALTIME_DELAY_FRAME_MS
    if group_size <= 0:
        raise ValueError(f"Invalid Audio8 ASR Infinite realtime token group size: {group_size}.")
    return group_size


def _audio8_realtime_frame_len(
    token_duration_ms: object,
    *,
    config: Any,
    default_token_duration_ms: int = _AUDIO8_REALTIME_DELAY_FRAME_MS,
) -> int:
    """Resolve a streaming token duration to encoder 20 ms frame slots."""
    duration_ms = _resolve_audio8_realtime_token_duration_ms(
        _first_scalar_value(token_duration_ms),
        default_token_duration_ms=default_token_duration_ms,
    )
    frame_ms = int(getattr(config, "audio_tower_frame_ms", 20) or 20)
    if duration_ms % frame_ms != 0:
        raise ValueError(
            "Audio8 ASR Infinite realtime token duration must be divisible by the audio tower "
            f"frame duration: token_duration_ms={duration_ms} "
            f"audio_tower_frame_ms={frame_ms}."
        )
    frame_len = duration_ms // frame_ms
    supported = tuple(int(value) for value in getattr(config, "supported_frame_lens", ()))
    if supported and frame_len not in supported:
        raise ValueError(
            f"Audio8 ASR Infinite realtime frame_len={frame_len} is not supported; "
            f"supported={supported}."
        )
    return int(frame_len)


def _audio8_gpu_memory_bytes() -> tuple[int, int]:
    """Whole-device ``(used, total)`` GPU memory in bytes, ``(0, 0)`` if unknown.

    This is the driver-reported device usage (what ``nvidia-smi`` shows for the
    serving GPU), so it also covers memory held outside this process.  Measured
    in the EngineCore process, which is the one holding the CUDA context.
    """

    try:
        if not torch.cuda.is_available():
            return 0, 0
        free_bytes, total_bytes = torch.cuda.mem_get_info()
    except Exception:  # pragma: no cover - driver/context hiccups must not break serving
        return 0, 0
    total = int(total_bytes)
    return max(0, total - int(free_bytes)), total


def _resolve_audio8_realtime_delay_options(
    *,
    delay_profile: str | None = None,
    target_delay_ms: int | float | str | None = None,
    num_delay_tokens: int | str | None = None,
    default_num_delay_tokens: int = 6,
    token_duration_ms: int | float | str | None = None,
    default_token_duration_ms: int = _AUDIO8_REALTIME_DELAY_FRAME_MS,
) -> dict[str, int | str]:
    duration_ms = _resolve_audio8_realtime_token_duration_ms(
        token_duration_ms,
        default_token_duration_ms=default_token_duration_ms,
    )
    if num_delay_tokens is not None:
        tokens = int(num_delay_tokens)
        milliseconds = tokens * duration_ms
    elif target_delay_ms is not None:
        milliseconds = int(target_delay_ms)
        tokens = milliseconds // duration_ms
    else:
        profile = _normalize_audio8_realtime_delay_profile(delay_profile)
        if profile is None:
            milliseconds = int(default_num_delay_tokens) * int(default_token_duration_ms)
            tokens = milliseconds // duration_ms
        else:
            if profile not in _AUDIO8_REALTIME_DELAY_PROFILE_TO_MS:
                allowed = ", ".join(sorted(_AUDIO8_REALTIME_DELAY_PROFILE_TO_MS))
                raise ValueError(f"Unsupported Audio8 ASR Infinite realtime delay_profile={delay_profile!r}; allowed: {allowed}.")
            milliseconds = int(_AUDIO8_REALTIME_DELAY_PROFILE_TO_MS[profile])
            tokens = milliseconds // duration_ms
    if milliseconds % duration_ms != 0:
        raise ValueError(
            "Audio8 ASR Infinite realtime delay must be divisible by the streaming token duration: "
            f"target_delay_ms={milliseconds} token_duration_ms={duration_ms}."
        )
    profile = f"{milliseconds}ms"
    if profile not in _AUDIO8_REALTIME_DELAY_PROFILE_TO_MS:
        allowed = ", ".join(sorted(_AUDIO8_REALTIME_DELAY_PROFILE_TO_MS))
        raise ValueError(f"Unsupported Audio8 ASR Infinite realtime delay {milliseconds}ms; allowed: {allowed}.")
    return {
        "delay_profile": profile,
        "target_delay_ms": int(milliseconds),
        "num_delay_tokens": int(tokens),
        "token_duration_ms": int(duration_ms),
    }


def _load_preprocessor_audio_config(model_dir: str | Path | None) -> dict[str, Any]:
    if model_dir is None:
        return {}
    config_path = Path(model_dir) / "preprocessor_config.json"
    if not config_path.is_file():
        return {}
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object in {config_path}.")
    return payload


def _build_whisper_audio_config(
    config: Any,
    model_dir: str | Path | None = None,
) -> WhisperConfig:
    audio_config_dict = config.audio_config.to_dict()
    preprocessor_config = _load_preprocessor_audio_config(model_dir)

    _set_if_missing_or_none(audio_config_dict, "model_type", "whisper")
    _set_if_missing_or_none(audio_config_dict, "is_causal", True)
    _set_if_missing_or_none(
        audio_config_dict,
        "num_mel_bins",
        _get_if_not_none(preprocessor_config, "feature_size"),
    )
    _set_if_missing_or_none(
        audio_config_dict,
        "sampling_rate",
        _get_if_not_none(preprocessor_config, "sampling_rate"),
    )
    _set_if_missing_or_none(
        audio_config_dict,
        "window_size",
        _get_if_not_none(preprocessor_config, "n_fft", "win_length"),
    )
    _set_if_missing_or_none(
        audio_config_dict,
        "hop_length",
        _get_if_not_none(preprocessor_config, "hop_length"),
    )
    _set_if_missing_or_none(
        audio_config_dict,
        "global_log_mel_max",
        _get_if_not_none(preprocessor_config, "global_log_mel_max"),
    )

    audio_hidden_size = int(
        _get_if_not_none(audio_config_dict, "hidden_size", "d_model") or 1280
    )
    downsample_factor = int(
        getattr(config, "downsample_factor", None)
        or _get_if_not_none(audio_config_dict, "downsample_factor")
        or next(
            iter(getattr(config, "supported_frame_lens", ()) or ()),
            None,
        )
        or 4
    )
    encoder_attention_heads = int(
        _get_if_not_none(
            audio_config_dict,
            "encoder_attention_heads",
            "num_attention_heads",
        )
        or 32
    )
    _set_if_missing_or_none(audio_config_dict, "d_model", audio_hidden_size)
    _set_if_missing_or_none(
        audio_config_dict,
        "encoder_layers",
        _get_if_not_none(audio_config_dict, "num_hidden_layers") or 32,
    )
    _set_if_missing_or_none(
        audio_config_dict,
        "encoder_ffn_dim",
        _get_if_not_none(audio_config_dict, "intermediate_size") or 5120,
    )
    _set_if_missing_or_none(
        audio_config_dict,
        "encoder_attention_heads",
        encoder_attention_heads,
    )
    _set_if_missing_or_none(
        audio_config_dict,
        "encoder_head_dim",
        _get_if_not_none(audio_config_dict, "head_dim")
        or audio_hidden_size // encoder_attention_heads,
    )
    _set_if_missing_or_none(
        audio_config_dict,
        "max_source_positions",
        _get_if_not_none(audio_config_dict, "max_position_embeddings") or 1500,
    )
    _set_if_missing_or_none(audio_config_dict, "scale_embedding", False)
    _set_if_missing_or_none(audio_config_dict, "num_mel_bins", 128)
    _set_if_missing_or_none(audio_config_dict, "sampling_rate", 16000)
    _set_if_missing_or_none(audio_config_dict, "window_size", 400)
    _set_if_missing_or_none(audio_config_dict, "hop_length", 160)
    _set_if_missing_or_none(audio_config_dict, "downsample_factor", downsample_factor)
    _set_if_missing_or_none(audio_config_dict, "block_pool_size", downsample_factor)
    _set_if_missing_or_none(audio_config_dict, "pos_embed", "rope")
    if audio_config_dict.get("global_log_mel_max") is not None:
        audio_config_dict["global_log_mel_max"] = float(audio_config_dict["global_log_mel_max"])
    # vLLM's Voxtral realtime path expands every pooled token back to its
    # encoder frames before applying RoPE, so the RoPE table uses frame units.
    audio_config_dict["max_position_embeddings"] = (
        downsample_factor * int(audio_config_dict["max_source_positions"])
    )
    return WhisperConfig(**audio_config_dict)


def _load_raw_text_rope_theta(model_dir: str | Path | None) -> float | None:
    if model_dir is None:
        return None
    config_path = Path(model_dir) / "config.json"
    if not config_path.is_file():
        return None
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    text_config = payload.get("text_config")
    if not isinstance(text_config, Mapping):
        return None
    rope_parameters = text_config.get("rope_parameters")
    if not isinstance(rope_parameters, Mapping):
        return None
    rope_theta = rope_parameters.get("rope_theta")
    return None if rope_theta is None else float(rope_theta)


def _normalize_qwen_text_config_for_container_transformers(
    config: Any,
    model_dir: str | Path | None = None,
) -> Any:
    rope_parameters = getattr(config, "rope_parameters", None)
    raw_rope_theta = _load_raw_text_rope_theta(model_dir)
    if isinstance(rope_parameters, Mapping):
        rope_theta = raw_rope_theta if raw_rope_theta is not None else rope_parameters.get("rope_theta")
        if rope_theta is not None:
            config.rope_theta = float(rope_theta)
            rope_parameters = dict(rope_parameters)
            rope_parameters["rope_theta"] = float(rope_theta)
            config.rope_parameters = rope_parameters
        if getattr(config, "rope_scaling", None) is None:
            config.rope_scaling = dict(rope_parameters)
        elif isinstance(config.rope_scaling, Mapping) and rope_theta is not None:
            rope_scaling = dict(config.rope_scaling)
            rope_scaling["rope_theta"] = float(rope_theta)
            config.rope_scaling = rope_scaling
    return config


def _load_feature_extractor_from_model_config(model_config: ModelConfig) -> Any:
    _ = model_config.revision, model_config.trust_remote_code
    return Audio8VoxtralRealtimeFeatureExtractor.from_pretrained(model_config.model)


def _ms_to_samples(milliseconds: float, *, sampling_rate: int) -> int:
    samples = sampling_rate * milliseconds / 1000.0
    if not samples.is_integer():
        raise ValueError(
            "Audio8 ASR Infinite realtime window duration must map to integer samples: "
            f"milliseconds={milliseconds} sampling_rate={sampling_rate}"
        )
    return int(samples)


def _expand_positions(positions: torch.Tensor, scaling: int) -> torch.Tensor:
    base = positions * scaling
    offsets = torch.arange(scaling, device=positions.device, dtype=positions.dtype)
    return (base.unsqueeze(1) + offsets).reshape(-1)


def _as_1d_positions(positions: torch.Tensor) -> torch.Tensor:
    if positions.ndim > 1:
        return positions[-1]
    return positions


def _stable_session_salt(cache_salt: str | None) -> int:
    """Map a per-generation cache_salt string to a stable int64 digest.

    The salt rides every window's mm item so the model adapter can key its
    per-session KV/t_cond state; 0 keeps the historical single-session
    behavior when no salt is provided.  The digest itself lives in
    ``audio8_vad_channel`` so the realtime connection derives the same key when
    it reads semantic VAD predictions back.
    """
    return audio8_vad_channel.session_salt(cache_salt)


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError:
        logger.warning("Ignoring invalid integer env %s=%r.", name, value)
        return default


def _env_language_backend() -> str:
    value = os.environ.get("AUDIO8_REALTIME_LANGUAGE_BACKEND", "hf").strip().lower()
    if value in {"", "hf", "torch", "correctness"}:
        return "hf"
    if value in {"vllm", "native", "performance"}:
        return "vllm"
    logger.warning(
        "Ignoring invalid AUDIO8_REALTIME_LANGUAGE_BACKEND=%r; using HF correctness backend.",
        value,
    )
    return "hf"


def _env_rolling_context_tokens() -> int:
    # Audio8 ASR Infinite defaults to the 30 s rolling window (375
    # tokens at the 80 ms token clock).  Only an explicit
    # AUDIO8_REALTIME_ROLLING_CONTEXT_TOKENS=0 keeps the old non-rolling
    # behavior, which caps a continuous realtime session at max_model_len
    # (the prompt ledger grows by one token per 80 ms of audio).  The HF
    # decoder cache is trimmed here in lockstep with the scheduler-side
    # rolling trim, keeping RoPE-relative alignment inside the window.
    return max(0, _env_int("AUDIO8_REALTIME_ROLLING_CONTEXT_TOKENS", 375))


def _env_rolling_trim_interval_tokens() -> int:
    milliseconds = _env_int("AUDIO8_REALTIME_ROLLING_TRIM_INTERVAL_MS", 3000)
    if milliseconds <= 0:
        return 0
    # Audio8 ASR Infinite realtime advances one language/audio token every 80ms.
    return max(1, (milliseconds + 79) // 80)


def _cache_seq_length(cache: object | None) -> int:
    if cache is None:
        return 0
    get_seq_length = getattr(cache, "get_seq_length", None)
    if callable(get_seq_length):
        try:
            return int(get_seq_length())
        except Exception:
            logger.debug("Failed to read Audio8 ASR Infinite realtime cache length.", exc_info=True)
            return 0
    layers = getattr(cache, "layers", None)
    if layers:
        get_layer_seq_length = getattr(layers[0], "get_seq_length", None)
        if callable(get_layer_seq_length):
            return int(get_layer_seq_length())
    return 0


def _rotate_half_plain_rope(value: torch.Tensor) -> torch.Tensor:
    first_half = value[..., : value.shape[-1] // 2]
    second_half = value[..., value.shape[-1] // 2 :]
    return torch.cat((-second_half, first_half), dim=-1)


def _apply_plain_rope_position_delta(
    keys: torch.Tensor,
    *,
    inv_freq: torch.Tensor | None,
    delta: int,
) -> torch.Tensor:
    if inv_freq is None or delta == 0 or keys.numel() == 0:
        return keys
    freqs = inv_freq.to(device=keys.device, dtype=torch.float32) * float(delta)
    emb = torch.cat((freqs, freqs), dim=-1)
    view_shape = (1,) * (keys.ndim - 1) + (emb.shape[-1],)
    cos = emb.cos().view(view_shape).to(dtype=keys.dtype)
    sin = emb.sin().view(view_shape).to(dtype=keys.dtype)
    return (keys * cos) + (_rotate_half_plain_rope(keys) * sin)


def _plain_rope_inv_freq_from_module(
    rotary_emb: object | None,
    *,
    device: torch.device,
) -> torch.Tensor | None:
    inv_freq = getattr(rotary_emb, "inv_freq", None)
    if isinstance(inv_freq, torch.Tensor):
        return inv_freq
    base = getattr(rotary_emb, "base", None)
    rotary_dim = getattr(rotary_emb, "rotary_dim", None)
    if base is None or rotary_dim is None:
        return None
    rotary_dim = int(rotary_dim)
    if rotary_dim <= 0:
        return None
    return 1.0 / (
        float(base)
        ** (
            torch.arange(0, rotary_dim, 2, dtype=torch.float32, device=device)
            / float(rotary_dim)
        )
    )


def _env_rolling_stable_prefix_tokens() -> int:
    # Session header (BOS + language token + left pads) kept stable across
    # rolling trims.  The language token conditions which language the model
    # transcribes; trimming it away collapses post-window generation into
    # pads and mixed-language filler.  Default 16 = the initial realtime
    # prompt ([BOS, language] + left-pad-2 streaming pads).
    return max(0, _env_int("AUDIO8_REALTIME_ROLLING_STABLE_PREFIX_TOKENS", 16))


def _trim_hf_cache_to_window_with_stable_prefix(
    cache: object | None,
    *,
    stable_prefix: int,
    max_tokens: int,
    inv_freq: torch.Tensor | None,
) -> int:
    """Trim the HF decoder cache to ``max_tokens`` keeping a stable head.

    The kept layout is ``[stable_prefix head][last max_tokens - stable_prefix
    tail]``; the removed tokens sit strictly between them.  The tail entries
    get their plain-HF RoPE positions shifted by ``-trim_count`` so they match
    the scheduler's re-based token ledger exactly.
    """
    if cache is None or max_tokens <= 0:
        return 0
    trim_count = _cache_seq_length(cache) - max_tokens
    if trim_count <= 0:
        return 0
    stable_prefix = max(0, min(stable_prefix, max_tokens))

    layers = getattr(cache, "layers", None)
    if layers is None:
        return 0

    trimmed = 0
    for layer in layers:
        get_seq_length = getattr(layer, "get_seq_length", None)
        if not callable(get_seq_length):
            continue
        seq_length = int(get_seq_length())
        layer_trim = max(0, seq_length - max_tokens)
        if layer_trim <= 0:
            continue
        keys = getattr(layer, "keys", None)
        values = getattr(layer, "values", None)
        if not isinstance(keys, torch.Tensor) or not isinstance(values, torch.Tensor):
            continue
        head_keys = keys[:, :, :stable_prefix, :].contiguous()
        head_values = values[:, :, :stable_prefix, :].contiguous()
        tail_start = stable_prefix + layer_trim
        tail_keys = keys[:, :, tail_start:, :].contiguous()
        tail_values = values[:, :, tail_start:, :].contiguous()
        tail_keys = _apply_plain_rope_position_delta(
            tail_keys,
            inv_freq=inv_freq,
            delta=-layer_trim,
        ).contiguous()
        layer.keys = torch.cat([head_keys, tail_keys], dim=2).contiguous()
        layer.values = torch.cat([head_values, tail_values], dim=2).contiguous()
        trimmed = max(trimmed, layer_trim)
    return trimmed


def _trim_hf_cache_to_last_tokens(
    cache: object | None,
    *,
    max_tokens: int,
    inv_freq: torch.Tensor | None,
) -> int:
    if cache is None or max_tokens <= 0:
        return 0
    layers = getattr(cache, "layers", None)
    if layers is None:
        key_cache = getattr(cache, "key_cache", None)
        value_cache = getattr(cache, "value_cache", None)
        if key_cache is None or value_cache is None:
            return 0
        max_trimmed = 0
        for index, keys in enumerate(key_cache):
            if not isinstance(keys, torch.Tensor):
                continue
            trim_count = max(0, int(keys.shape[-2]) - max_tokens)
            if trim_count <= 0:
                continue
            key_cache[index] = _apply_plain_rope_position_delta(
                keys[..., trim_count:, :].contiguous(),
                inv_freq=inv_freq,
                delta=-trim_count,
            ).contiguous()
            values = value_cache[index]
            if isinstance(values, torch.Tensor):
                value_cache[index] = values[..., trim_count:, :].contiguous()
            max_trimmed = max(max_trimmed, trim_count)
        return max_trimmed

    max_trimmed = 0
    for layer in layers:
        get_seq_length = getattr(layer, "get_seq_length", None)
        if not callable(get_seq_length):
            continue
        seq_length = int(get_seq_length())
        trim_count = max(0, seq_length - max_tokens)
        if trim_count <= 0:
            continue
        keys = getattr(layer, "keys", None)
        values = getattr(layer, "values", None)
        if isinstance(keys, torch.Tensor):
            kept_keys = keys[..., trim_count:, :].contiguous()
            layer.keys = _apply_plain_rope_position_delta(
                kept_keys,
                inv_freq=inv_freq,
                delta=-trim_count,
            ).contiguous()
        if isinstance(values, torch.Tensor):
            layer.values = values[..., trim_count:, :].contiguous()
        if hasattr(layer, "cumulative_length"):
            target_length = seq_length - trim_count
            cumulative_length = getattr(layer, "cumulative_length")
            if isinstance(cumulative_length, torch.Tensor):
                cumulative_length.fill_(target_length)
            else:
                layer.cumulative_length = target_length
        max_trimmed = max(max_trimmed, trim_count)
    return max_trimmed


def _map_audio8_audio_tower_weight_name(name: str) -> str:
    if name.startswith("embedder."):
        name = f"whisper_encoder.{name.removeprefix('embedder.')}"
    elif name.startswith("layers."):
        name = f"whisper_encoder.{name}"
    elif name.startswith("norm."):
        name = f"whisper_encoder.layer_norm.{name.removeprefix('norm.')}"

    return (
        name.replace(".self_attn.o_proj.", ".self_attn.out_proj.")
        .replace(".mlp.gate_proj.", ".mlp.fc1.")
        .replace(".mlp.up_proj.", ".mlp.fc3.")
        .replace(".mlp.down_proj.", ".mlp.fc2.")
    )


def _iter_audio8_audio_tower_load_items(
    name: str,
    weight: torch.Tensor,
) -> Iterable[tuple[str, torch.Tensor]]:
    mapped_name = _map_audio8_audio_tower_weight_name(name)
    yield mapped_name, weight
    if name.endswith(".self_attn.k_proj.weight"):
        yield mapped_name.replace(".weight", ".bias"), torch.zeros(
            weight.size(0),
            dtype=weight.dtype,
            device=weight.device,
        )


def _get_audio_rope_theta(audio_config: Any) -> float:
    rope_parameters = getattr(audio_config, "rope_parameters", None)
    if isinstance(rope_parameters, Mapping):
        return float(rope_parameters.get("rope_theta", 1e6))
    return 1e6


def _get_language_rope_theta(text_config: Any) -> float:
    rope_parameters = getattr(text_config, "rope_parameters", None)
    if isinstance(rope_parameters, Mapping):
        return float(rope_parameters.get("rope_theta", 1e6))
    return 1e6


def _patch_audio_rope_to_hf_rotate_half_style(
    audio_tower: nn.Module,
    audio_config: Any,
) -> int:
    """Align vLLM audio RoPE layout with HF VoxtralRealtime rotate_half."""
    whisper_encoder = getattr(audio_tower, "whisper_encoder", None)
    layers = getattr(whisper_encoder, "layers", None)
    if layers is None:
        return 0

    max_position_embeddings = int(getattr(audio_config, "max_position_embeddings"))
    rope_parameters = {"rope_theta": _get_audio_rope_theta(audio_config)}
    patched = 0
    for layer in layers:
        self_attn = getattr(layer, "self_attn", None)
        if self_attn is None or not hasattr(self_attn, "rotary_emb"):
            continue
        self_attn.rotary_emb = get_rope(
            int(self_attn.head_dim),
            max_position=max_position_embeddings,
            is_neox_style=True,
            rope_parameters=rope_parameters,
        )
        patched += 1
    return patched


def _native_audio_rms_norm_forward(
    module: nn.Module,
    hidden_states: torch.Tensor,
    residual: torch.Tensor | None = None,
) -> torch.Tensor:
    if residual is not None:
        hidden_states = hidden_states + residual
    weight = getattr(module, "weight", None)
    if weight is None:
        return F.rms_norm(hidden_states, (hidden_states.shape[-1],), None, module.variance_epsilon)
    return F.rms_norm(
        hidden_states,
        (int(weight.numel()),),
        weight,
        float(getattr(module, "variance_epsilon", 1e-5)),
    )


def _patch_audio_rms_norm_to_torch(audio_tower: nn.Module) -> int:
    """Avoid the vLLM CUDA RMSNorm path for transposed Whisper conv output."""
    whisper_encoder = getattr(audio_tower, "whisper_encoder", None)
    if whisper_encoder is None:
        return 0
    modules: list[nn.Module] = []
    modules.extend(list(getattr(whisper_encoder, "layers", [])))
    final_norm = getattr(whisper_encoder, "layer_norm", None)
    if final_norm is not None:
        modules.append(final_norm)
    patched = 0
    for layer in modules:
        for name in ("self_attn_layer_norm", "final_layer_norm"):
            norm = getattr(layer, name, None)
            if norm is None or not hasattr(norm, "weight"):
                continue
            norm.forward = MethodType(_native_audio_rms_norm_forward, norm)
            patched += 1
        if layer is final_norm and hasattr(layer, "weight"):
            layer.forward = MethodType(_native_audio_rms_norm_forward, layer)
            patched += 1
    return patched


def _patch_language_rope_to_plain_hf_style(
    language_model: nn.Module,
    text_config: Any,
) -> int:
    """Use the plain Qwen3 RoPE path that HF torch applies to this checkpoint."""
    model = getattr(language_model, "model", language_model)
    layers = getattr(model, "layers", None)
    if layers is None:
        return 0

    max_position_embeddings = int(getattr(text_config, "max_position_embeddings"))
    rope_parameters = {
        "rope_type": "default",
        "rope_theta": _get_language_rope_theta(text_config),
    }
    patched = 0
    for layer in layers:
        self_attn = getattr(layer, "self_attn", None)
        if self_attn is None or not hasattr(self_attn, "rotary_emb"):
            continue
        self_attn.rotary_emb = get_rope(
            int(self_attn.head_dim),
            max_position=max_position_embeddings,
            is_neox_style=True,
            rope_parameters=rope_parameters,
        )
        patched += 1
    return patched


def _audio8_realtime_hf_eager_attention_forward(
    self_attn: nn.Module,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    """HF-style Qwen attention for the Audio8 ASR Infinite realtime single-request path."""
    positions = _as_1d_positions(positions).to(device=hidden_states.device)
    qkv, _ = self_attn.qkv_proj(hidden_states)
    q, k, v = qkv.split([self_attn.q_size, self_attn.kv_size, self_attn.kv_size], dim=-1)

    q_by_head = q.view(*q.shape[:-1], q.shape[-1] // self_attn.head_dim, self_attn.head_dim)
    q_by_head = self_attn.q_norm(q_by_head)
    q = q_by_head.view(q.shape)
    k_by_head = k.view(*k.shape[:-1], k.shape[-1] // self_attn.head_dim, self_attn.head_dim)
    k_by_head = self_attn.k_norm(k_by_head)
    k = k_by_head.view(k.shape)
    q, k = self_attn.rotary_emb(positions, q, k)

    query_length = int(q.shape[0])
    q_heads = q.view(query_length, self_attn.num_heads, self_attn.head_dim)
    k_heads = k.view(query_length, self_attn.num_kv_heads, self_attn.head_dim)
    v_heads = v.view(query_length, self_attn.num_kv_heads, self_attn.head_dim)
    q_heads = q_heads.transpose(0, 1).unsqueeze(0)
    k_heads = k_heads.transpose(0, 1).unsqueeze(0)
    v_heads = v_heads.transpose(0, 1).unsqueeze(0)

    cached_k = getattr(self_attn, "_audio8_realtime_k_cache", None)
    cached_v = getattr(self_attn, "_audio8_realtime_v_cache", None)
    cached_positions = getattr(self_attn, "_audio8_realtime_k_positions", None)
    rolling_context_tokens = int(
        getattr(self_attn, "_audio8_realtime_rolling_context_tokens", 0) or 0
    )
    rolling_trim_interval_tokens = int(
        getattr(self_attn, "_audio8_realtime_rolling_trim_interval_tokens", 0) or 0
    )
    reset_cache = (
        cached_k is None
        or cached_v is None
        or cached_positions is None
        or int(positions.reshape(-1)[0].item()) == 0
        or cached_k.device != k_heads.device
        or cached_k.dtype != k_heads.dtype
        or cached_v.device != v_heads.device
        or cached_v.dtype != v_heads.dtype
    )
    if reset_cache:
        k_cache = k_heads
        v_cache = v_heads
        key_positions = positions
    else:
        target_cache_tokens = int(positions.reshape(-1)[0].item())
        if rolling_context_tokens > 0 and cached_k.shape[2] > target_cache_tokens:
            trim_count = int(cached_k.shape[2] - target_cache_tokens)
            cached_k = cached_k[:, :, trim_count:, :].contiguous()
            cached_v = cached_v[:, :, trim_count:, :].contiguous()
            inv_freq = _plain_rope_inv_freq_from_module(
                getattr(self_attn, "rotary_emb", None),
                device=cached_k.device,
            )
            cached_k = _apply_plain_rope_position_delta(
                cached_k,
                inv_freq=inv_freq,
                delta=-trim_count,
            ).contiguous()
            cached_positions = (
                cached_positions[trim_count:] - trim_count
            ).contiguous()
        k_cache = torch.cat([cached_k, k_heads], dim=2)
        v_cache = torch.cat([cached_v, v_heads], dim=2)
        key_positions = torch.cat([cached_positions, positions], dim=0)

    k_for_attention = k_cache
    v_for_attention = v_cache
    if self_attn.num_heads != self_attn.num_kv_heads:
        if self_attn.num_heads % self_attn.num_kv_heads != 0:
            raise ValueError(
                "Audio8 ASR Infinite realtime Qwen attention requires num_heads to be divisible by "
                "num_kv_heads for grouped-query attention."
            )
        repeat_factor = self_attn.num_heads // self_attn.num_kv_heads
        k_for_attention = k_for_attention.repeat_interleave(repeat_factor, dim=1)
        v_for_attention = v_for_attention.repeat_interleave(repeat_factor, dim=1)

    causal_mask = key_positions.unsqueeze(0) <= positions.unsqueeze(1)
    attention_scores = torch.matmul(
        q_heads,
        k_for_attention.transpose(2, 3),
    ) * float(self_attn.scaling)
    attention_scores = attention_scores.masked_fill(
        ~causal_mask.view(1, 1, query_length, -1),
        torch.finfo(attention_scores.dtype).min,
    )
    attention_probs = torch.softmax(
        attention_scores,
        dim=-1,
        dtype=torch.float32,
    ).to(dtype=q_heads.dtype)
    attention_output = torch.matmul(attention_probs, v_for_attention)
    attention_output = (
        attention_output.squeeze(0)
        .transpose(0, 1)
        .contiguous()
        .view(query_length, self_attn.q_size)
    )
    # Same bound as the scheduler trim: never exceed the window (audio
    # sub-token positions are row_position * 4 and must stay under
    # max_source_positions); cut back to context - interval + 1.
    trim_target_tokens = rolling_context_tokens
    if rolling_trim_interval_tokens > 1:
        trim_target_tokens -= rolling_trim_interval_tokens - 1
    if rolling_context_tokens > 0 and k_cache.shape[2] > rolling_context_tokens:
        trim_count = int(k_cache.shape[2] - trim_target_tokens)
        k_cache = k_cache[:, :, trim_count:, :].contiguous()
        v_cache = v_cache[:, :, trim_count:, :].contiguous()
        inv_freq = _plain_rope_inv_freq_from_module(
            getattr(self_attn, "rotary_emb", None),
            device=k_cache.device,
        )
        k_cache = _apply_plain_rope_position_delta(
            k_cache,
            inv_freq=inv_freq,
            delta=-trim_count,
        ).contiguous()
        key_positions = (key_positions[trim_count:] - trim_count).contiguous()

    self_attn._audio8_realtime_k_cache = k_cache
    self_attn._audio8_realtime_v_cache = v_cache
    self_attn._audio8_realtime_k_positions = key_positions

    output, _ = self_attn.o_proj(attention_output)
    return output


def _audio8_realtime_tower_eager_attention_forward(
    self_attn: nn.Module,
    hidden_states: torch.Tensor,
    positions: torch.Tensor | None = None,
) -> torch.Tensor:
    """Eager sliding-window attention for the Audio8 ASR Infinite realtime audio tower.

    The tower's vLLM attention is DECODER-typed against the paged KV cache:
    its blocks hold the cross-row audio context, and the scheduler's rolling
    trim frees those blocks (they cannot be re-based in place), which corrupts
    every audio feature after the first trim.  RoPE attention scores depend
    only on relative position offsets, so re-basing the tower KV by a constant
    shift (RoPE delta) is exact: a rolling window is numerically identical to
    a fresh session over the same audio.
    """
    if positions is None:
        raise ValueError("Audio8 ASR Infinite realtime tower attention requires positions.")
    positions = _as_1d_positions(positions).to(device=hidden_states.device)
    qkv, _ = self_attn.qkv_proj(hidden_states)
    q, k, v = qkv.split([self_attn.q_size, self_attn.kv_size, self_attn.kv_size], dim=-1)
    q, k = self_attn.rotary_emb(positions, q, k)

    query_length = int(q.shape[0])
    q_heads = q.view(query_length, self_attn.num_heads, self_attn.head_dim)
    k_heads = k.view(query_length, self_attn.num_heads, self_attn.head_dim)
    v_heads = v.view(query_length, self_attn.num_heads, self_attn.head_dim)
    q_heads = q_heads.transpose(0, 1).unsqueeze(0)
    k_heads = k_heads.transpose(0, 1).unsqueeze(0)
    v_heads = v_heads.transpose(0, 1).unsqueeze(0)

    sliding_window = int(getattr(self_attn, "_audio8_realtime_tower_sliding_window", 0) or 0)
    cached_k = getattr(self_attn, "_audio8_realtime_tower_k_cache", None)
    cached_v = getattr(self_attn, "_audio8_realtime_tower_v_cache", None)
    cached_positions = getattr(self_attn, "_audio8_realtime_tower_key_positions", None)
    first_position = int(positions.reshape(-1)[0].item())
    reset_cache = (
        cached_k is None
        or cached_v is None
        or cached_positions is None
        or first_position == 0
        or cached_k.device != k_heads.device
        or cached_k.dtype != k_heads.dtype
    )
    if not reset_cache:
        # Re-base when the scheduler rolling trim moved the ledger window
        # backward: keys keep their content, positions shift by the gap.
        expected_start = int(cached_positions[-1].item()) + 1
        if first_position != expected_start:
            delta = first_position - expected_start
            if delta < 0 and int(cached_positions[-1].item()) + delta >= 0:
                keep_from = 0 if sliding_window <= 0 else max(0, first_position - sliding_window)
                keep_mask = (cached_positions + delta) >= keep_from
                cached_k = cached_k[:, :, keep_mask, :].contiguous()
                cached_v = cached_v[:, :, keep_mask, :].contiguous()
                cached_positions = (cached_positions[keep_mask] + delta).contiguous()
                inv_freq = _plain_rope_inv_freq_from_module(
                    getattr(self_attn, "rotary_emb", None),
                    device=cached_k.device,
                )
                cached_k = _apply_plain_rope_position_delta(
                    cached_k,
                    inv_freq=inv_freq,
                    delta=delta,
                ).contiguous()
            else:
                reset_cache = True

    if reset_cache:
        k_cache = k_heads
        v_cache = v_heads
        key_positions = positions
    else:
        k_cache = torch.cat([cached_k, k_heads], dim=2)
        v_cache = torch.cat([cached_v, v_heads], dim=2)
        key_positions = torch.cat([cached_positions, positions], dim=0)

    k_for_attention = k_cache
    v_for_attention = v_cache
    if self_attn.num_heads != self_attn.num_kv_heads:
        if self_attn.num_heads % self_attn.num_kv_heads != 0:
            raise ValueError(
                "Audio8 ASR Infinite realtime tower attention requires num_heads divisible by "
                "num_kv_heads."
            )
        repeat_factor = self_attn.num_heads // self_attn.num_kv_heads
        k_for_attention = k_for_attention.repeat_interleave(repeat_factor, dim=1)
        v_for_attention = v_for_attention.repeat_interleave(repeat_factor, dim=1)

    causal_mask = key_positions.unsqueeze(0) <= positions.unsqueeze(1)
    if sliding_window > 0:
        window_mask = key_positions.unsqueeze(0) > (
            positions.unsqueeze(1) - sliding_window
        )
        causal_mask = causal_mask & window_mask
    attention_scores = torch.matmul(
        q_heads,
        k_for_attention.transpose(2, 3),
    ) * float(self_attn.scaling)
    attention_scores = attention_scores.masked_fill(
        ~causal_mask.view(1, 1, query_length, -1),
        torch.finfo(attention_scores.dtype).min,
    )
    attention_probs = torch.softmax(
        attention_scores,
        dim=-1,
        dtype=torch.float32,
    ).to(dtype=q_heads.dtype)
    attention_output = torch.matmul(attention_probs, v_for_attention)
    attention_output = (
        attention_output.squeeze(0)
        .transpose(0, 1)
        .contiguous()
        .view(query_length, self_attn.q_size)
    )

    # Bound the tower cache to the sliding window (plus the live query):
    # older keys can never be attended again.
    if sliding_window > 0 and k_cache.shape[2] > sliding_window:
        trim_count = int(k_cache.shape[2] - sliding_window)
        k_cache = k_cache[:, :, trim_count:, :].contiguous()
        v_cache = v_cache[:, :, trim_count:, :].contiguous()
        key_positions = key_positions[trim_count:].contiguous()

    self_attn._audio8_realtime_tower_k_cache = k_cache
    self_attn._audio8_realtime_tower_v_cache = v_cache
    self_attn._audio8_realtime_tower_key_positions = key_positions

    output, _ = self_attn.out_proj(attention_output)
    return output


def _patch_audio_tower_attention_to_eager_realtime(
    audio_tower: nn.Module,
    sliding_window: int,
) -> int:
    """Bypass the vLLM paged attention for the realtime audio tower."""
    whisper_encoder = getattr(audio_tower, "whisper_encoder", None)
    layers = getattr(whisper_encoder, "layers", None)
    if layers is None:
        return 0

    patched = 0
    for layer in layers:
        self_attn = getattr(layer, "self_attn", None)
        if self_attn is None:
            continue
        required_attrs = (
            "qkv_proj",
            "out_proj",
            "rotary_emb",
            "num_heads",
            "num_kv_heads",
            "head_dim",
            "q_size",
            "kv_size",
            "scaling",
        )
        if not all(hasattr(self_attn, attr) for attr in required_attrs):
            continue
        self_attn.forward = MethodType(
            _audio8_realtime_tower_eager_attention_forward,
            self_attn,
        )
        self_attn._audio8_realtime_tower_eager = True
        self_attn._audio8_realtime_tower_sliding_window = int(sliding_window)
        patched += 1
    return patched


def _patch_language_attention_to_hf_eager_realtime(language_model: nn.Module) -> int:
    """Bypass vLLM paged attention for Audio8 ASR Infinite realtime and use HF eager attention."""
    model = getattr(language_model, "model", language_model)
    layers = getattr(model, "layers", None)
    if layers is None:
        return 0

    patched = 0
    for layer in layers:
        self_attn = getattr(layer, "self_attn", None)
        if self_attn is None:
            continue
        required_attrs = (
            "qkv_proj",
            "q_norm",
            "k_norm",
            "rotary_emb",
            "o_proj",
            "num_heads",
            "num_kv_heads",
            "head_dim",
            "q_size",
            "kv_size",
        )
        if not all(hasattr(self_attn, attr) for attr in required_attrs):
            continue
        self_attn.forward = MethodType(
            _audio8_realtime_hf_eager_attention_forward,
            self_attn,
        )
        self_attn._audio8_realtime_hf_eager_attention = True
        self_attn._audio8_realtime_rolling_context_tokens = _env_rolling_context_tokens()
        self_attn._audio8_realtime_rolling_trim_interval_tokens = (
            _env_rolling_trim_interval_tokens()
        )
        patched += 1
    return patched


def _audio8_realtime_hf_residual_decoder_layer_forward(
    layer: nn.Module,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    residual: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """HF-style Qwen decoder residual order for the Audio8 ASR Infinite realtime path."""
    if residual is not None:
        hidden_states = hidden_states + residual
    layer_residual = hidden_states
    hidden_states = layer.input_layernorm(hidden_states)
    hidden_states = layer.self_attn(
        positions=positions,
        hidden_states=hidden_states,
    )
    hidden_states = layer_residual + hidden_states

    layer_residual = hidden_states
    hidden_states = layer.post_attention_layernorm(hidden_states)
    hidden_states = layer.mlp(hidden_states)
    hidden_states = layer_residual + hidden_states
    return hidden_states, torch.zeros_like(hidden_states)


def _patch_language_layers_to_hf_residual_realtime(language_model: nn.Module) -> int:
    """Use HF's explicit residual order instead of vLLM residual-carry layers."""
    model = getattr(language_model, "model", language_model)
    layers = getattr(model, "layers", None)
    if layers is None:
        return 0

    patched = 0
    for layer in layers:
        required_attrs = (
            "input_layernorm",
            "self_attn",
            "post_attention_layernorm",
            "mlp",
        )
        if not all(hasattr(layer, attr) for attr in required_attrs):
            continue
        layer.forward = MethodType(
            _audio8_realtime_hf_residual_decoder_layer_forward,
            layer,
        )
        layer._audio8_realtime_hf_residual_layer = True
        patched += 1
    return patched


class Audio8AudioToQwenProjector(nn.Module):
    def __init__(self, config: Any) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(
            int(config.audio_config.hidden_size)
            * int(getattr(config, "downsample_factor", getattr(config, "max_frame_len", 4))),
            int(config.text_config.hidden_size),
            bias=False,
        )
        self.act = ACT2FN[str(getattr(config, "projector_hidden_act", "gelu"))]
        self.linear_2 = nn.Linear(
            int(config.text_config.hidden_size),
            int(config.text_config.hidden_size),
            bias=False,
        )

    def forward(self, audio_features: torch.Tensor) -> torch.Tensor:
        hidden_states = self.linear_1(audio_features)
        hidden_states = self.act(hidden_states)
        return self.linear_2(hidden_states)


@dataclass(frozen=True)
class Audio8RealtimeBufferConfig:
    sampling_rate: int
    samples_per_token: int
    left_pad_tokens: int
    right_pad_tokens: int
    num_delay_tokens: int
    token_duration_ms: int
    look_ahead_samples: int
    look_back_samples: int
    streaming_pad_token_id: int
    eos_token_id: int
    bos_token_id: int
    language_token_id: int
    # Stable per-generation identity (int64 digest of the connection's
    # cache_salt).  Carried on every window's mm item so the model can route
    # per-session KV/t_cond when several realtime sessions share one engine.
    session_salt: int = 0


@dataclass
class Audio8RealtimeSessionState:
    """Per-session rolling state owned by the model adapter.

    With ``--max-num-seqs > 1`` several realtime sessions share one engine and
    one model instance, so the two self-managed rolling caches (HF decoder KV
    and the audio tower's eager sliding-window KV) plus the active delay
    condition must be keyed by session instead of living on the instance.
    The key is the ``audio8_session_salt`` digest carried on every window's mm
    item; salt 0 keeps the historical single-session behavior.
    """

    salt: int
    # HF Qwen2 decoder DynamicCache (or equivalent) for this session.
    hf_past_key_values: Any = None
    # Audio tower eager cache per patched layer:
    # layer index -> (k_cache, v_cache, key_positions) or None.
    tower_caches: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None] = field(
        default_factory=dict
    )
    active_num_delay_tokens: int = 6
    active_frame_len: int | None = None
    # Ledger position just past this session's last scheduled row; used for
    # diagnostics and position-continuity checks across steps.
    next_position: int | None = None
    reset_count: int = 0
    last_used_tick: int = 0
    # Wall-clock stamp used to tell a finished generation's leftovers apart
    # from a live session (a live session is touched once per audio window).
    last_used_monotonic: float = 0.0


class Audio8RealtimeAudioTokenBuffer:
    def __init__(self, config: Audio8RealtimeBufferConfig) -> None:
        self.config = config
        self._audio_queue: asyncio.Queue[np.ndarray | None] = asyncio.Queue()
        self._token_queue: asyncio.Queue[int] = asyncio.Queue()
        self._leftover: np.ndarray | None = None
        self._saw_audio_end = False
        # Per-window nonce carried in mm_processor_kwargs.  vLLM hashes mm
        # items from the raw audio plus these kwargs, so without a nonce the
        # trailing silence windows of one session (identical zero frames)
        # would share an mm hash and hit the encoder cache, hiding their mm
        # items from the model adapter's per-session routing.
        self._window_nonce = 0

        for token_id in self._initial_prompt_token_ids():
            self._token_queue.put_nowait(token_id)

    def _initial_prompt_token_ids(self) -> list[int]:
        if self.config.left_pad_tokens < 2:
            return []
        return [self.config.bos_token_id, self.config.language_token_id] + [
            self.config.streaming_pad_token_id
        ] * (self.config.left_pad_tokens - 2)

    async def append_audio(self, audio_array: np.ndarray | None) -> None:
        if audio_array is None:
            await self._audio_queue.put(None)
            return
        audio = np.asarray(audio_array, dtype=np.float32).reshape(-1)
        if audio.size:
            await self._audio_queue.put(audio)

    async def append_tokens(self, tokens: Iterable[int]) -> None:
        for token_id in tokens:
            token_id = int(token_id)
            if token_id == self.config.eos_token_id:
                token_id = self.config.streaming_pad_token_id
            await self._token_queue.put(token_id)

    async def get_input_stream(self) -> AsyncGenerator[PromptType, None]:
        for frame_size, num_tokens in self._generate_frame_size_and_num_tokens():
            prompt_token_ids = [await self._read_token_id() for _ in range(num_tokens)]
            audio = await self._read_audio_frame(frame_size)
            if audio is None:
                return

            self._window_nonce += 1
            yield TokensPrompt(
                prompt_token_ids=prompt_token_ids,
                multi_modal_data={"audio": (audio, self.config.sampling_rate)},
                mm_processor_kwargs={
                    "num_delay_tokens": self.config.num_delay_tokens,
                    "token_duration_ms": self.config.token_duration_ms,
                    "audio8_session_salt": self.config.session_salt,
                    "audio8_window_nonce": self._window_nonce,
                },
            )

    async def _read_token_id(self) -> int:
        return int(await self._token_queue.get())

    def _generate_frame_size_and_num_tokens(self) -> Iterable[tuple[int, int]]:
        start = 0
        end = len(self._initial_prompt_token_ids()) * self.config.samples_per_token
        if end <= start:
            end = self.config.samples_per_token

        for _ in count():
            frame_start = max(start - self.config.look_back_samples, 0)
            frame_end = end + self.config.look_ahead_samples
            frame_size = frame_end - frame_start
            token_count = (end - start) / self.config.samples_per_token
            if not token_count.is_integer():
                raise RuntimeError(
                    "Audio8 ASR Infinite realtime token count must be integral: "
                    f"start={start} end={end} samples_per_token={self.config.samples_per_token}"
                )
            yield frame_size, int(token_count)
            start = end
            end += self.config.samples_per_token

    async def _read_audio_frame(self, frame_size: int) -> np.ndarray | None:
        audio_arrays: list[np.ndarray] = []
        if self._leftover is not None:
            audio_arrays.append(self._leftover)

        while sum(len(array) for array in audio_arrays) < frame_size and not self._saw_audio_end:
            chunk = await self._audio_queue.get()
            if chunk is None:
                self._saw_audio_end = True
                break

            audio_arrays.append(chunk)

        if not audio_arrays:
            return None

        audio_array = np.concatenate(audio_arrays)
        if audio_array.shape[0] < frame_size:
            return None

        frame = audio_array[:frame_size]
        stride = frame_size - self.config.look_ahead_samples - self.config.look_back_samples
        if stride <= 0:
            raise RuntimeError(f"Audio8 ASR Infinite realtime stride must be positive, got {stride}.")
        self._leftover = audio_array[stride:]
        return frame


class Audio8StreamingASRProcessingInfo(BaseProcessingInfo):
    def get_tokenizer(self) -> Any:
        return cached_tokenizer_from_config(self.ctx.model_config)

    def get_default_token_duration_ms(self) -> int:
        hf_config = self.ctx.model_config.hf_config
        configured_frame_ms = (
            getattr(hf_config, "streaming_frame_ms", None)
            or getattr(hf_config, "token_duration_ms", None)
            or _AUDIO8_REALTIME_DELAY_FRAME_MS
        )
        return int(configured_frame_ms)

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"audio": None}

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Mapping[str, int]:
        _ = seq_len, mm_counts
        return {"audio": self.ctx.model_config.max_model_len}

    def get_feature_extractor(self, **kwargs: object) -> Any:
        _ = kwargs
        return _load_feature_extractor_from_model_config(self.ctx.model_config)

    def get_data_parser(self) -> MultiModalDataParser:
        feature_extractor = self.get_feature_extractor()
        return MultiModalDataParser(
            target_sr=int(getattr(feature_extractor, "sampling_rate", 16000)),
            target_channels=1,
            expected_hidden_size=self._get_expected_hidden_size(),
        )


class Audio8StreamingASRDummyInputsBuilder(BaseDummyInputsBuilder[Audio8StreamingASRProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        _ = mm_counts
        return ""

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, object],
    ) -> Mapping[str, object]:
        _ = seq_len
        num_audios = mm_counts.get("audio", 0)
        feature_extractor = self.info.get_feature_extractor()
        duration_ms = _resolve_audio8_realtime_token_duration_ms(
            _first_scalar_value(mm_options.get("token_duration_ms")),
            default_token_duration_ms=self.info.get_default_token_duration_ms(),
        )
        duration_ms = max(
            duration_ms,
            self.info.get_default_token_duration_ms()
            * int(getattr(self.info.ctx.model_config.hf_config, "max_frame_len", 1) or 1),
        )
        target_audio_length = int(
            getattr(feature_extractor, "sampling_rate", 16000) * duration_ms / 1000.0
        )
        return {"audio": [np.zeros(target_audio_length, dtype=np.float32) for _ in range(num_audios)]}


class Audio8StreamingASRMultiModalProcessor(BaseMultiModalProcessor[Audio8StreamingASRProcessingInfo]):
    def __init__(
        self,
        info: Audio8StreamingASRProcessingInfo,
        dummy_inputs: BaseDummyInputsBuilder[Audio8StreamingASRProcessingInfo],
        *,
        cache: BaseMultiModalProcessorCache | None = None,
    ) -> None:
        super().__init__(info, dummy_inputs, cache=cache)

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        _ = hf_processor_mm_kwargs
        _ = hf_inputs
        return {
            "audio_arrays": MultiModalFieldConfig.batched("audio"),
            "num_delay_tokens": MultiModalFieldConfig.batched("audio", keep_on_cpu=True),
            "token_duration_ms": MultiModalFieldConfig.batched("audio", keep_on_cpu=True),
            "audio8_session_salt": MultiModalFieldConfig.batched("audio", keep_on_cpu=True),
            "audio8_window_nonce": MultiModalFieldConfig.batched("audio", keep_on_cpu=True),
        }

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        _ = prompt, tok_kwargs
        mm_data = dict(mm_data)
        audios = mm_data.pop("audios", mm_data.pop("audio", None))
        if audios is None:
            input_ids = self.info.get_tokenizer().encode(prompt)
            return BatchFeature({"input_ids": torch.tensor([input_ids], dtype=torch.long)})

        processor = self.info.get_feature_extractor()
        sampling_rate = int(mm_kwargs.get("sampling_rate", getattr(processor, "sampling_rate", 16000)))
        num_delay_tokens = int(mm_kwargs.get("num_delay_tokens", 6))
        session_salt = int(mm_kwargs.get("audio8_session_salt", 0) or 0)
        window_nonce = int(mm_kwargs.get("audio8_window_nonce", 0) or 0)
        token_duration_ms = _resolve_audio8_realtime_token_duration_ms(
            mm_kwargs.get("token_duration_ms"),
            default_token_duration_ms=self.info.get_default_token_duration_ms(),
        )
        input_ids = self.info.get_tokenizer().encode(prompt)
        features = processor(
            audios,
            sampling_rate=sampling_rate,
            return_tensors="pt",
            return_attention_mask=True,
            padding="longest",
            center=True,
        )
        return BatchFeature({
            "input_ids": torch.tensor([input_ids], dtype=torch.long),
            "audio_arrays": [torch.as_tensor(audio, dtype=torch.float32) for audio in audios],
            "num_delay_tokens": torch.full((len(audios),), num_delay_tokens, dtype=torch.long),
            "token_duration_ms": torch.full((len(audios),), token_duration_ms, dtype=torch.long),
            "audio8_session_salt": torch.full((len(audios),), session_salt, dtype=torch.long),
            "audio8_window_nonce": torch.full((len(audios),), window_nonce, dtype=torch.long),
            **features,
        })

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, Any],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        feature_extractor = self.info.get_feature_extractor(**hf_processor_mm_kwargs)
        out_mm_data = out_mm_kwargs.require_data()
        out_audio_items = out_mm_data.get("audio", [])

        def get_replacement(item_idx: int) -> list[int]:
            token_duration_value = hf_processor_mm_kwargs.get("token_duration_ms")
            if item_idx < len(out_audio_items):
                out_audio_data = out_audio_items[item_idx].get_data()
                audio_array = out_audio_data["audio_arrays"]
                token_duration_value = out_audio_data.get(
                    "token_duration_ms",
                    token_duration_value,
                )
                if isinstance(audio_array, (torch.Tensor, np.ndarray)):
                    audio_len = len(audio_array)
                else:
                    raise TypeError(
                        "Unexpected type for Audio8 ASR Infinite audio_arrays in out_mm_kwargs: "
                        f"{type(audio_array)}"
                    )
            else:
                audios = mm_items.get_items("audio", AudioProcessorItems)
                audio_len = audios.get_audio_length(item_idx)

            token_group_size = _audio8_realtime_token_group_size(
                token_duration_value,
                default_token_duration_ms=self.info.get_default_token_duration_ms(),
            )
            num_audio_tokens = max(
                1,
                feature_extractor.get_num_audio_tokens(audio_len)
                // token_group_size,
            )
            return [0] * num_audio_tokens

        return [
            PromptReplacement(
                modality="audio",
                target="",
                replacement=get_replacement,
            ),
        ]

    def _maybe_apply_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        prompt_ids: list[int],
        mm_kwargs: Mapping[str, Any],
        mm_prompt_updates: MultiModalPromptUpdates,
        is_update_applied: bool,
    ) -> tuple[list[int], Mapping[str, list[PlaceholderFeaturesInfo]]]:
        _ = mm_items, mm_prompt_updates, is_update_applied
        features_info = PlaceholderFeaturesInfo(
            modality="audio",
            item_idx=0,
            start_idx=0,
            tokens=[0] * len(prompt_ids),
            is_embed=None,
        )
        return prompt_ids, {"audio": [features_info]}


@MULTIMODAL_REGISTRY.register_processor(
    Audio8StreamingASRMultiModalProcessor,
    info=Audio8StreamingASRProcessingInfo,
    dummy_inputs=Audio8StreamingASRDummyInputsBuilder,
)
class Audio8StreamingASRRealtimeGeneration(
    nn.Module,
    SupportsMultiModal,
    SupportsMRoPE,
    SupportsPP,
    SupportsRealtime,
):
    supports_realtime = True
    requires_raw_input_tokens = True
    disable_full_model_cudagraph = True
    realtime_max_tokens = 1

    hf_to_vllm_mapper = WeightsMapper(orig_to_new_prefix={"language_model.": ""})

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        self.vllm_config = vllm_config
        self.config = vllm_config.model_config.hf_config
        self.quant_config = vllm_config.quant_config
        self.multimodal_config = vllm_config.model_config.multimodal_config
        self.downsample_factor = int(
            getattr(self.config, "downsample_factor", None)
            or next(
                iter(getattr(self.config, "supported_frame_lens", ()) or ()),
                None,
            )
            or 4
        )
        self.config.text_config = _normalize_qwen_text_config_for_container_transformers(
            self.config.text_config,
            vllm_config.model_config.model,
        )
        # SupportsRealtime passes post-conv audio blocks through the runner's
        # inputs_embeds buffer. Match VoxtralRealtimeGeneration here: that
        # buffer is one fixed audio block wide. Audio8 ASR Infinite's larger shared projector
        # width is handled only after the audio encoder.
        audio_block_projection_size = self.downsample_factor * int(
            getattr(
                self.config.audio_config,
                "hidden_size",
                getattr(self.config.audio_config, "d_model", 1280),
            )
        )
        self.config.projection_size = audio_block_projection_size
        self.config.text_config.projection_size = audio_block_projection_size
        self.whisper_audio_config = _build_whisper_audio_config(
            self.config,
            vllm_config.model_config.model,
        )

        with self._mark_tower_model(vllm_config, "audio"):
            self.audio_tower = VoxtralEncoderModel(
                vllm_config.with_hf_config(self.whisper_audio_config),
                prefix=maybe_prefix(prefix, "audio_tower"),
            )
        self._audio_rms_norm_patch_count = _patch_audio_rms_norm_to_torch(
            self.audio_tower
        )
        self._audio_tower_eager_patch_count = 0
        if self._env_tower_eager_attention():
            self._audio_tower_eager_patch_count = (
                _patch_audio_tower_attention_to_eager_realtime(
                    self.audio_tower,
                    int(getattr(self.whisper_audio_config, "sliding_window", 0) or 0),
                )
            )
            if self._audio_tower_eager_patch_count:
                logger.info(
                    "Audio8 ASR Infinite realtime audio tower uses eager rolling attention: "
                    "layers=%s sliding_window=%s",
                    self._audio_tower_eager_patch_count,
                    getattr(self.whisper_audio_config, "sliding_window", None),
                )
        self._audio_rope_patch_count = _patch_audio_rope_to_hf_rotate_half_style(
            self.audio_tower,
            self.whisper_audio_config,
        )
        # Ordered tower attention modules (index-stable across the model's
        # lifetime) used to swap the per-session eager KV in and out around
        # each session's tower forward.
        self._tower_self_attns: list[nn.Module] = [
            layer.self_attn
            for layer in (
                getattr(
                    getattr(self.audio_tower, "whisper_encoder", None),
                    "layers",
                    (),
                )
                or ()
            )
            if getattr(layer, "self_attn", None) is not None
        ]

        self.multi_modal_projector = Audio8AudioToQwenProjector(self.config)
        self.time_embedding = _voxtral_realtime_modeling.VoxtralRealtimeTimeEmbedding(
            int(self.config.text_config.hidden_size)
        )
        self.frame_len_embedding = (
            nn.Embedding(
                len(getattr(self.config, "supported_frame_lens", ())),
                int(self.config.text_config.hidden_size),
            )
            if bool(getattr(self.config, "use_frame_len_embedding", False))
            else None
        )
        # Audio8 ASR Infinite conditions the decoder with per-layer ada_rms_norm(t_cond).
        # It has no standalone delay-embedding tensor in the checkpoint.
        self.delay_embedding = None

        is_qwen2 = getattr(self.config.text_config, "model_type", None) == "qwen2"
        if is_qwen2:
            # V1 checkpoints contain adaptive RMS modules that are part of the
            # repository's HF Audio8 ASR Infinite decoder.  Keep that decoder intact while
            # vLLM supplies the realtime scheduler and WebSocket endpoint.
            self.language_model = Audio8ASRInfiniteQwen2ForCausalLM(
                self.config.text_config
            )
        else:
            self.language_model = Audio8ASRInfiniteForCausalLM(
                self.config.text_config
            )
        # Semantic VAD heads are built only when the checkpoint declares
        # horizons: their weight keys (`semantic_vad_heads.{i}.weight/bias`) sit
        # at the top level, so a transcription-only checkpoint is unaffected.
        # Each head reads the same hidden states the language head consumes and
        # the predictions are handed to the realtime endpoint through
        # `audio8_vad_channel` (see that module for why a side channel is used).
        self.semantic_vad_horizons_seconds: tuple[float, ...] = tuple(
            float(horizon)
            for horizon in (
                getattr(self.config, "semantic_vad_horizons_seconds", None) or ()
            )
        )
        self.semantic_vad_num_classes: int = int(
            getattr(self.config, "semantic_vad_num_classes", 0) or 0
        )
        if self.semantic_vad_horizons_seconds and self.semantic_vad_num_classes >= 2:
            reference_parameter = next(self.language_model.parameters())
            self.semantic_vad_heads: nn.ModuleList | None = nn.ModuleList(
                [
                    nn.Linear(
                        int(self.config.text_config.hidden_size),
                        self.semantic_vad_num_classes,
                        bias=True,
                        dtype=reference_parameter.dtype,
                    )
                    for _ in self.semantic_vad_horizons_seconds
                ]
            )
        else:
            self.semantic_vad_heads = None
        # Salts of the session groups seen by the latest forward, used to route
        # the predictions published out of `compute_logits`.
        self._vad_recent_salts: list[int] = []
        # 每步的服务指标（RTF / 显存）：`forward` 打时间戳、`compute_logits` 发布。
        self._metrics_step_started_at: float | None = None
        self._metrics_rtf_ema: float | None = None
        self._metrics_gpu_memory: tuple[int, int] | None = None
        self._metrics_step_counter = 0
        self._language_backend = "hf" if is_qwen2 else _env_language_backend()
        self._rolling_context_tokens = _env_rolling_context_tokens()
        self._rolling_trim_interval_tokens = _env_rolling_trim_interval_tokens()
        self._rolling_stable_prefix_tokens = min(
            _env_rolling_stable_prefix_tokens(),
            max(1, self._rolling_context_tokens - 1) if self._rolling_context_tokens > 0 else 0,
        )
        self._hf_language_model = None
        if self._language_backend == "hf":
            if is_qwen2:
                self._hf_language_model = self.language_model
            else:
                hf_language_model_cls = (
                    HFQwen2ForCausalLM
                    if getattr(self.config.text_config, "model_type", None) == "qwen2"
                    else HFQwen3ForCausalLM
                )
                self._hf_language_model = hf_language_model_cls(self.config.text_config)
            self._hf_language_model.eval()
            self._hf_language_model.to(dtype=torch.bfloat16)
        self._hf_language_past_key_values = None
        self._realtime_context_reset_count = 0
        self._active_frame_len = int(
            getattr(self.config, "supported_frame_lens", (self.downsample_factor,))[0]
        )
        self._active_num_delay_tokens = int(
            getattr(self.config, "default_num_delay_tokens", None) or 6
        )
        # Per-session rolling state (KV / delay condition / tower caches),
        # keyed by the audio8_session_salt digest.  Salt 0 is the historical
        # single-session slot so a no-salt deployment keeps its exact
        # previous behavior.
        self._session_states: dict[int, Audio8RealtimeSessionState] = {}
        self._session_state_cap = max(
            1, _env_int("AUDIO8_REALTIME_SESSION_STATE_CAP", 4)
        )
        # States of finished generations are dropped after this much idle time
        # (0 disables).  Without it the dict sits at the cap after a handful of
        # reconnects, so later routing fallbacks see "several live sessions"
        # although only one request can be in flight.
        self._session_state_idle_seconds = max(
            0, _env_int("AUDIO8_REALTIME_SESSION_STATE_IDLE_SECONDS", 120)
        )
        # Never LRU-evict a state touched this recently: dropping a generating
        # session's HF KV mid-stream would leave it attending to an empty cache
        # for the rest of the connection.
        self._session_state_protect_seconds = max(
            0, _env_int("AUDIO8_REALTIME_SESSION_STATE_PROTECT_SECONDS", 15)
        )
        self._session_state_tick = 0
        # (salt, num_delay_tokens, row_count) per mm item encoded in this
        # engine step, in encoder scheduling order.  Consumed (and cleared)
        # by forward() to partition the step's token rows per session.
        self._pending_mm_items: list[tuple[int, int, int]] = []
        # The HF Audio8 ASR Infinite decoder already owns the correct Qwen2 attention,
        # cache, RoPE, residual order, and adaptive RMS layer.  Do not replace
        # those methods with vLLM-internal tensor signatures.
        self._language_rope_patch_count = 0
        self._language_attention_patch_count = 0
        self._language_layer_patch_count = 0

        self.make_empty_intermediate_tensors = getattr(
            self.language_model,
            "make_empty_intermediate_tensors",
            lambda *args, **kwargs: None,
        )

    def get_mrope_input_positions(
        self,
        input_tokens: list[int],
        mm_features: list[object],
    ) -> tuple[torch.Tensor, int]:
        _ = mm_features
        positions = torch.arange(len(input_tokens), dtype=torch.long)
        return positions.unsqueeze(0).expand(3, -1).clone(), 0

    @classmethod
    async def buffer_realtime_audio(
        cls,
        audio_stream: AsyncGenerator[np.ndarray, None],
        input_stream: asyncio.Queue[list[int]],
        model_config: ModelConfig,
        realtime_options: Mapping[str, object] | None = None,
        language: str | None = None,
        cache_salt: str | None = None,
    ) -> AsyncGenerator[PromptType, None]:
        tokenizer = cached_tokenizer_from_config(model_config)
        hf_config = model_config.hf_config
        audio_config = getattr(hf_config, "audio_config", None)
        feature_extractor = _load_feature_extractor_from_model_config(model_config)
        realtime_options = dict(realtime_options or {})
        selected_language = language or realtime_options.pop("language", None) or "zh"
        language_token_id = resolve_qwen_language_token_id(
            tokenizer,
            str(selected_language),
        )
        configured_frame_ms = (
            getattr(hf_config, "streaming_frame_ms", None)
            or getattr(hf_config, "token_duration_ms", None)
            or _AUDIO8_REALTIME_DELAY_FRAME_MS
        )
        configured_delay_tokens = getattr(hf_config, "default_num_delay_tokens", None)
        if configured_delay_tokens is None:
            configured_delay_tokens = 6
        delay_options = _resolve_audio8_realtime_delay_options(
            delay_profile=realtime_options.get("delay_profile"),
            target_delay_ms=realtime_options.get("target_delay_ms"),
            num_delay_tokens=realtime_options.get("num_delay_tokens"),
            default_num_delay_tokens=int(configured_delay_tokens),
            token_duration_ms=realtime_options.get("token_duration_ms"),
            default_token_duration_ms=int(configured_frame_ms),
        )
        num_delay_tokens = int(delay_options["num_delay_tokens"])
        token_duration_ms = int(delay_options["token_duration_ms"])

        sampling_rate = int(getattr(feature_extractor, "sampling_rate", getattr(audio_config, "sampling_rate", 16000)))
        samples_per_token_float = sampling_rate * token_duration_ms / 1000.0
        if not samples_per_token_float.is_integer():
            raise ValueError(
                "sampling_rate * token_duration_ms / 1000 must be an integer: "
                f"sampling_rate={sampling_rate} token_duration_ms={token_duration_ms}"
            )
        samples_per_token = int(samples_per_token_float)
        look_ahead_samples = _ms_to_samples(
            float(getattr(hf_config, "streaming_look_ahead_ms", 2.5)),
            sampling_rate=sampling_rate,
        )
        look_back_samples = _ms_to_samples(
            float(getattr(hf_config, "streaming_look_back_ms", 52.5)),
            sampling_rate=sampling_rate,
        )
        right_pad_tokens = (
            num_delay_tokens + 1 + int(getattr(hf_config, "right_pad_text_tokens", 10))
        )
        default_token_duration_ms = int(configured_frame_ms)
        base_left_pad_tokens = int(getattr(hf_config, "streaming_n_left_pad_tokens", 32))
        left_pad_tokens = max(1, int(round(base_left_pad_tokens * default_token_duration_ms / token_duration_ms)))
        buffer_config = Audio8RealtimeBufferConfig(
            sampling_rate=sampling_rate,
            samples_per_token=samples_per_token,
            left_pad_tokens=left_pad_tokens,
            right_pad_tokens=right_pad_tokens,
            num_delay_tokens=num_delay_tokens,
            token_duration_ms=token_duration_ms,
            look_ahead_samples=look_ahead_samples,
            look_back_samples=look_back_samples,
            streaming_pad_token_id=int(tokenizer.convert_tokens_to_ids(_STREAMING_PAD_TOKEN)),
            eos_token_id=int(
                getattr(tokenizer, "eos_token_id", None)
                or tokenizer.convert_tokens_to_ids("<|im_end|>")
            ),
            bos_token_id=int(getattr(tokenizer, "bos_token_id", None) or tokenizer.convert_tokens_to_ids("<|im_start|>")),
            language_token_id=int(language_token_id),
            session_salt=_stable_session_salt(cache_salt),
        )
        buffer = Audio8RealtimeAudioTokenBuffer(buffer_config)

        async def feed_audio() -> None:
            left_pad = np.zeros(buffer_config.left_pad_tokens * samples_per_token, dtype=np.float32)
            right_pad = np.zeros(buffer_config.right_pad_tokens * samples_per_token, dtype=np.float32)
            yielded_first_chunk = False
            async for audio_chunk in audio_stream:
                if not yielded_first_chunk:
                    yielded_first_chunk = True
                    await buffer.append_audio(left_pad)
                await buffer.append_audio(audio_chunk)
            if not yielded_first_chunk:
                await buffer.append_audio(left_pad)
            await buffer.append_audio(right_pad)
            await buffer.append_audio(None)

        async def feed_tokens() -> None:
            while True:
                try:
                    generated_token_ids = await asyncio.wait_for(
                        input_stream.get(),
                        timeout=VLLM_ENGINE_ITERATION_TIMEOUT_S,
                    )
                except TimeoutError:
                    logger.exception(
                        "Audio8 ASR Infinite realtime timed out waiting for generated token "
                        "feedback after %ss. Increase "
                        "VLLM_ENGINE_ITERATION_TIMEOUT_S or reduce streaming "
                        "session backlog.",
                        VLLM_ENGINE_ITERATION_TIMEOUT_S,
                    )
                    raise
                await buffer.append_tokens(generated_token_ids[-1:])

        audio_task = asyncio.create_task(feed_audio())
        token_task = asyncio.create_task(feed_tokens())
        try:
            async for prompt in buffer.get_input_stream():
                yield prompt
        finally:
            audio_task.cancel()
            token_task.cancel()

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings | None:
        audio_arrays = kwargs.get("audio_arrays")
        if audio_arrays is None:
            logger.warning("Audio8 ASR Infinite realtime received no audio_arrays in embed_multimodal.")
            return []
        if isinstance(audio_arrays, torch.Tensor):
            audio_inputs = list(audio_arrays.unbind(0))
        elif isinstance(audio_arrays, list):
            audio_inputs = [torch.as_tensor(audio, dtype=torch.float32, device=self._audio_device()) for audio in audio_arrays]
        else:
            raise TypeError(f"audio_arrays must be a tensor or tensor list, got {type(audio_arrays)}")

        token_duration_value = _first_scalar_value(kwargs.get("token_duration_ms"))
        default_token_duration_ms = int(
            getattr(self.config, "streaming_frame_ms", None)
            or getattr(self.config, "token_duration_ms", None)
            or _AUDIO8_REALTIME_DELAY_FRAME_MS
        )
        frame_len = _audio8_realtime_frame_len(
            token_duration_value,
            config=self.config,
            default_token_duration_ms=default_token_duration_ms,
        )
        # The native realtime scheduler does not pass multimodal kwargs into
        # the language-model forward call.  Keep the active session condition
        # alongside the cached audio embedding; with a single session (or a
        # pre-salt deployment) the next forward belongs to this same window.
        self._active_frame_len = int(frame_len)
        active_delay = _first_scalar_value(kwargs.get("num_delay_tokens"))
        if active_delay is not None:
            self._active_num_delay_tokens = int(active_delay)

        mel_features = [
            self.audio_tower.compute_whisper_melspec(audio).to(self.audio_tower.dtype)
            for audio in audio_inputs
        ]
        seq_lens = [mel.shape[1] for mel in mel_features]
        audio_embeddings = self.audio_tower.whisper_encoder.forward_conv(mel_features)
        conv_stride = self.audio_tower.whisper_encoder.total_stride
        conv_actual_split_lengths = [
            math.ceil(seq_len / conv_stride)
            for seq_len in seq_lens
        ]
        conv_target_lengths = [
            seq_len // conv_stride
            for seq_len in seq_lens
        ]
        audio_embeddings_per_sample = audio_embeddings.split(
            conv_actual_split_lengths,
            dim=0,
        )
        # The causal conv emits one right-padding frame for some mel lengths.
        # Keep the same floor alignment as the reference streaming path before
        # removing the left overlap for the current token group.
        audio_embeddings_per_sample = [
            sample[:target_length]
            for sample, target_length in zip(
                audio_embeddings_per_sample,
                conv_target_lengths,
                strict=True,
            )
        ]
        hidden_size = int(
            getattr(
                self.config.audio_config,
                "hidden_size",
                getattr(self.config.audio_config, "d_model", 1280),
            )
        )
        grouped_embeddings: list[torch.Tensor] = []
        for sample in audio_embeddings_per_sample:
            frame_count = int(sample.shape[0])
            if frame_count < frame_len:
                # Complete a short current window before applying Voxtral's
                # fixed block grouping.
                sample = F.pad(sample, (0, 0, 0, frame_len - frame_count))
            else:
                remainder = frame_count % frame_len
                if remainder:
                    sample = sample[remainder:]
            grouped_embeddings.append(
                sample.reshape(-1, frame_len * hidden_size)
            )
        # Per-item session identity for forward()'s row partitioning.  The
        # pending list is consumed (cleared) by the paired forward call in the
        # same engine step; a bound guards against unpaired calls from
        # profiling/dummy runs.
        session_salts = _per_item_int_values(
            kwargs.get("audio8_session_salt"),
            len(audio_inputs),
        )
        delay_values = _per_item_int_values(
            kwargs.get("num_delay_tokens"),
            len(audio_inputs),
        )
        for item_index, grouped in enumerate(grouped_embeddings):
            self._pending_mm_items.append(
                (
                    int(session_salts[item_index]),
                    int(delay_values[item_index]),
                    int(grouped.shape[0]),
                )
            )
        if len(self._pending_mm_items) > 64:
            self._pending_mm_items.clear()
        return grouped_embeddings

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _ = is_multimodal
        if multimodal_embeddings is None or len(multimodal_embeddings) == 0:
            pool_size = int(self.downsample_factor)
            audio_hidden_size = int(getattr(self.config.audio_config, "hidden_size", getattr(self.config.audio_config, "d_model", 1280)))
            return torch.zeros(
                input_ids.shape[0],
                audio_hidden_size * pool_size,
                dtype=self.audio_tower.dtype,
                device=input_ids.device,
            )
        return _flatten_embeddings(multimodal_embeddings).to(
            device=input_ids.device,
            dtype=self.audio_tower.dtype,
        )

    def _project_audio_hidden_states(
        self,
        audio_hidden_states: torch.Tensor,
        text_embeds: torch.Tensor,
    ) -> torch.Tensor:
        if audio_hidden_states.shape[0] % self.downsample_factor != 0:
            raise ValueError(
                "Audio8 ASR Infinite realtime audio hidden length must be divisible by downsample_factor: "
                f"length={audio_hidden_states.shape[0]} downsample_factor={self.downsample_factor}"
            )
        audio_hidden_size = int(getattr(self.config.audio_config, "hidden_size", getattr(self.config.audio_config, "d_model", 1280)))
        audio_hidden_states = audio_hidden_states.reshape(
            audio_hidden_states.shape[0] // self.downsample_factor,
            self.downsample_factor,
            audio_hidden_size,
        )
        max_frame_len = int(
            getattr(self.config, "max_frame_len", self.downsample_factor)
        )
        if self.downsample_factor < max_frame_len:
            audio_hidden_states = F.pad(
                audio_hidden_states,
                (0, 0, 0, max_frame_len - self.downsample_factor),
            )
        audio_hidden_states = audio_hidden_states.reshape(
            -1,
            max_frame_len * audio_hidden_size,
        )
        audio_embeds = self.multi_modal_projector(audio_hidden_states).to(
            device=text_embeds.device,
            dtype=text_embeds.dtype,
        )
        if audio_embeds.shape != text_embeds.shape:
            raise ValueError(
                "Audio8 ASR Infinite realtime audio/text embedding shape mismatch: "
                f"audio={tuple(audio_embeds.shape)} text={tuple(text_embeds.shape)}"
            )
        return audio_embeds

    def _hf_text_device(self) -> torch.device:
        if self._hf_language_model is None:
            raise RuntimeError("Audio8 ASR Infinite HF language backend is not initialized.")
        return next(self._hf_language_model.parameters()).device

    def _hf_text_dtype(self) -> torch.dtype:
        if self._hf_language_model is None:
            raise RuntimeError("Audio8 ASR Infinite HF language backend is not initialized.")
        return next(self._hf_language_model.parameters()).dtype

    def _env_tower_eager_attention(self) -> bool:
        # Enabled by default: the tower's vLLM paged KV cannot survive the
        # scheduler rolling trim (freed blocks), while the eager path re-bases
        # exactly thanks to RoPE's relative-position property.
        value = os.environ.get("AUDIO8_REALTIME_TOWER_EAGER")
        if value is None:
            return True
        return value.strip().lower() not in ("0", "false", "no", "off")

    def _touch_session_state(
        self,
        salt: int,
        *,
        num_delay_tokens: int | None = None,
        frame_len: int | None = None,
    ) -> Audio8RealtimeSessionState:
        """Return (creating if needed) the rolling state for one session."""
        self._session_state_tick += 1
        state = self._session_states.get(salt)
        if state is None:
            state = Audio8RealtimeSessionState(
                salt=salt,
                active_num_delay_tokens=(
                    int(num_delay_tokens)
                    if num_delay_tokens is not None
                    else self._active_num_delay_tokens
                ),
                active_frame_len=(
                    int(frame_len) if frame_len is not None else self._active_frame_len
                ),
            )
            self._session_states[salt] = state
            if salt != 0 and 0 in self._session_states:
                # A salted (real) session exists, so this deployment passes
                # cache_salt; any salt-0 state can only be a startup
                # profiling artifact, not a legacy live session.
                del self._session_states[0]
            logger.info(
                "Audio8 ASR Infinite realtime session state created: salt=%s delay_tokens=%s",
                salt,
                state.active_num_delay_tokens,
            )
        state.last_used_tick = self._session_state_tick
        state.last_used_monotonic = time.monotonic()
        if num_delay_tokens is not None:
            state.active_num_delay_tokens = int(num_delay_tokens)
        if frame_len is not None:
            state.active_frame_len = int(frame_len)
        self._prune_session_states()
        return state

    def _prune_idle_session_states(self) -> None:
        """Drop session states whose generation has been gone for a while."""
        ttl = self._session_state_idle_seconds
        if ttl <= 0 or not self._session_states:
            return
        now = time.monotonic()
        ordered = sorted(
            self._session_states.items(),
            key=lambda item: item[1].last_used_monotonic,
        )
        # The most recently used state is the live session (a generating
        # session is touched once per window, a paused one was still touched
        # after every finished generation), so it is never dropped here.
        for salt, state in ordered[:-1]:
            stamp = state.last_used_monotonic
            if stamp and now - stamp > ttl:
                del self._session_states[salt]
                logger.info(
                    "Audio8 ASR Infinite realtime session state dropped after %.1fs "
                    "idle: salt=%s hf_tokens=%s",
                    now - stamp,
                    salt,
                    _cache_seq_length(state.hf_past_key_values),
                )

    def _prune_session_states(self) -> None:
        """Bound the per-session state dict (idle TTL + LRU) to cap residency."""
        self._prune_idle_session_states()
        if len(self._session_states) <= self._session_state_cap:
            return
        protect = self._session_state_protect_seconds
        now = time.monotonic()
        evictable = [
            (salt, state)
            for salt, state in self._session_states.items()
            if not (
                protect > 0
                and state.last_used_monotonic
                and now - state.last_used_monotonic <= protect
            )
        ]
        if len(self._session_states) - len(evictable) >= self._session_state_cap:
            # Every session currently generating is protected; run over the cap
            # instead of freeing a live session's KV.
            return
        ordered = sorted(evictable, key=lambda item: item[1].last_used_tick)
        evict_count = len(self._session_states) - self._session_state_cap
        for salt, state in ordered[:evict_count]:
            del self._session_states[salt]
            logger.info(
                "Audio8 ASR Infinite realtime session state evicted (LRU): salt=%s hf_tokens=%s",
                salt,
                _cache_seq_length(state.hf_past_key_values),
            )

    def _reset_session_state_kv(self, state: Audio8RealtimeSessionState) -> None:
        """Drop one session's decoder/tower KV (its generation restarted).

        Scoped replacement of the historical global reset: with several
        realtime sessions on one engine, session B starting at position zero
        must not clear session A's rolling caches.  The (never-installed)
        language-layer legacy attributes need no handling here; the tower
        attributes are swapped per session around each tower forward.
        """
        previous_tokens = _cache_seq_length(state.hf_past_key_values)
        state.hf_past_key_values = None
        state.tower_caches.clear()
        state.reset_count += 1
        self._realtime_context_reset_count += 1
        logger.info(
            "Audio8 ASR Infinite realtime context KV reset at session start: salt=%s "
            "reset_count=%s previous_hf_tokens=%s",
            state.salt,
            state.reset_count,
            previous_tokens,
        )

    def _load_tower_caches_for_state(self, state: Audio8RealtimeSessionState) -> None:
        """Install one session's tower eager caches onto the layer modules."""
        for index, self_attn in enumerate(self._tower_self_attns):
            cached = state.tower_caches.get(index)
            if cached is None:
                self_attn._audio8_realtime_tower_k_cache = None
                self_attn._audio8_realtime_tower_v_cache = None
                self_attn._audio8_realtime_tower_key_positions = None
            else:
                k_cache, v_cache, key_positions = cached
                self_attn._audio8_realtime_tower_k_cache = k_cache
                self_attn._audio8_realtime_tower_v_cache = v_cache
                self_attn._audio8_realtime_tower_key_positions = key_positions

    def _save_tower_caches_for_state(self, state: Audio8RealtimeSessionState) -> None:
        """Persist the layer-attribute tower caches back into one session."""
        for index, self_attn in enumerate(self._tower_self_attns):
            k_cache = getattr(self_attn, "_audio8_realtime_tower_k_cache", None)
            if k_cache is None:
                state.tower_caches.pop(index, None)
                continue
            state.tower_caches[index] = (
                k_cache,
                getattr(self_attn, "_audio8_realtime_tower_v_cache", None),
                getattr(self_attn, "_audio8_realtime_tower_key_positions", None),
            )

    def _reset_realtime_context_kv(self) -> None:
        """Drop all decoder KV owned by this long-lived model instance.

        vLLM's paged request KV is request-scoped, but the Audio8 ASR Infinite HF decoder keeps
        its cache on the model object so it can advance one realtime frame at a
        time.  A new request starts at position zero; clear both the HF cache
        and any eager-attention side caches before that frame is evaluated.
        """
        self._hf_language_past_key_values = None
        for state in self._session_states.values():
            state.hf_past_key_values = None
            state.tower_caches.clear()
        self._realtime_context_reset_count += 1

    def _embed_text_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        if self._language_backend == "vllm":
            return self.language_model.embed_input_ids(input_ids)
        return self._hf_language_model.get_input_embeddings()(input_ids.to(self._hf_text_device()))

    def _run_hf_language_model(
        self,
        *,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor,
        t_cond: torch.Tensor,
        state: Audio8RealtimeSessionState,
    ) -> torch.Tensor:
        text_device = self._hf_text_device()
        positions = _as_1d_positions(positions).to(device=text_device, dtype=torch.long)
        if int(positions.reshape(-1)[0].item()) == 0:
            self._reset_session_state_kv(state)
        elif self._rolling_context_tokens > 0:
            current_cache_length = _cache_seq_length(state.hf_past_key_values)
            target_cache_length = int(positions.reshape(-1)[0].item())
            if current_cache_length > target_cache_length:
                # Re-align the HF cache with the scheduler's rolling trim:
                # stable session head + re-based tail.
                _trim_hf_cache_to_window_with_stable_prefix(
                    state.hf_past_key_values,
                    stable_prefix=self._rolling_stable_prefix_tokens,
                    max_tokens=target_cache_length,
                    inv_freq=self._hf_language_model.model.rotary_emb.inv_freq,
                )
        past_length = _cache_seq_length(state.hf_past_key_values)
        attention_length = past_length + int(positions.numel())
        outputs = self._hf_language_model.model(
            inputs_embeds=inputs_embeds.to(device=text_device, dtype=self._hf_text_dtype()).unsqueeze(0),
            position_ids=positions.unsqueeze(0),
            past_key_values=state.hf_past_key_values,
            attention_mask=torch.ones(
                1,
                attention_length,
                device=text_device,
                dtype=torch.long,
            ),
            use_cache=True,
            output_hidden_states=False,
            t_cond=t_cond.to(device=text_device, dtype=self._hf_text_dtype()),
        )
        state.hf_past_key_values = outputs.past_key_values
        if self._rolling_context_tokens > 0:
            # Same bound as the scheduler trim: the cache must never exceed
            # the window because the next row's audio sub-token positions are
            # row_position * 4 and the audio tower supports at most
            # max_source_positions (375 rows * 4 = 1500).  Trim back to
            # context - interval + 1 so the next trim fires a full interval
            # later while staying under the bound.
            trim_target_tokens = self._rolling_context_tokens
            if self._rolling_trim_interval_tokens > 1:
                trim_target_tokens -= self._rolling_trim_interval_tokens - 1
            if _cache_seq_length(state.hf_past_key_values) > self._rolling_context_tokens:
                _trim_hf_cache_to_window_with_stable_prefix(
                    state.hf_past_key_values,
                    stable_prefix=self._rolling_stable_prefix_tokens,
                    max_tokens=trim_target_tokens,
                    inv_freq=self._hf_language_model.model.rotary_emb.inv_freq,
                )
        return outputs.last_hidden_state.squeeze(0)

    def _run_language_model(
        self,
        *,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor,
        t_cond: torch.Tensor,
        state: Audio8RealtimeSessionState,
        intermediate_tensors: IntermediateTensors | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if self._language_backend == "vllm":
            if state.salt != 0 and len(self._session_states) > 1:
                raise NotImplementedError(
                    "Multiple concurrent realtime sessions require the HF "
                    "language backend (the vLLM paged backend is not "
                    "session-scoped in this adapter)."
                )
            return self.language_model(
                input_ids,
                positions,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=inputs_embeds,
                t_cond=t_cond,
            )
        _ = intermediate_tensors
        return self._run_hf_language_model(
            input_ids=input_ids,
            positions=positions,
            inputs_embeds=inputs_embeds,
            t_cond=t_cond,
            state=state,
        )

    def _language_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        if self._language_backend == "vllm":
            return self.language_model.compute_logits(hidden_states)
        return self._hf_language_model.lm_head(
            hidden_states.to(
                device=self._hf_text_device(),
                dtype=self._hf_text_dtype(),
            )
        )

    def _build_t_cond_for_forward(
        self,
        num_delay_tokens: object,
        *,
        device: torch.device,
        dtype: torch.dtype,
        frame_len: object = None,
    ) -> torch.Tensor:
        """Build the Audio8 ASR Infinite adaptive-RMS condition for one realtime row."""
        if self._hf_language_model is None:
            return torch.zeros(
                1,
                1,
                int(self.config.text_config.hidden_size),
                device=device,
                dtype=dtype,
            )
        if num_delay_tokens is None:
            num_delay_tokens = self._active_num_delay_tokens
        if num_delay_tokens is None:
            num_delay_tokens = 6
        if isinstance(num_delay_tokens, torch.Tensor):
            num_delay_tokens = num_delay_tokens.reshape(-1)[0]
        elif isinstance(num_delay_tokens, (list, tuple)):
            num_delay_tokens = num_delay_tokens[0] if num_delay_tokens else 6
        if frame_len is None:
            frame_len = self._active_frame_len
        if isinstance(frame_len, torch.Tensor):
            frame_len = frame_len.reshape(-1)[0]
        elif isinstance(frame_len, (list, tuple)):
            frame_len = frame_len[0] if frame_len else None
        delay_value = torch.as_tensor(
            num_delay_tokens,
            device=device,
            dtype=dtype,
        ).reshape(1)
        delay_embedding = self.time_embedding(delay_value)
        if self.frame_len_embedding is not None:
            if frame_len is None:
                frame_len = self.downsample_factor
            frame_value = int(frame_len)
            supported = [int(value) for value in self.config.supported_frame_lens]
            if frame_value not in supported:
                frame_value = self.downsample_factor
            if frame_value not in supported:
                raise ValueError(
                    f"Unsupported Audio8 ASR Infinite realtime frame_len={frame_value}; "
                    f"supported={supported}"
                )
            frame_embedding = self.frame_len_embedding(
                torch.tensor(
                    [supported.index(frame_value)],
                    device=device,
                    dtype=torch.long,
                )
            ).to(dtype=dtype)
            delay_embedding = delay_embedding + frame_embedding
        return delay_embedding.unsqueeze(1)

    def _resolve_forward_delay_ids(
        self,
        num_delay_tokens: object,
        *,
        token_count: int,
        device: torch.device,
    ) -> torch.Tensor:
        if num_delay_tokens is None:
            num_delay_tokens = int(getattr(self.config, "default_num_delay_tokens", 6))
        if isinstance(num_delay_tokens, torch.Tensor):
            delay_ids = num_delay_tokens.to(device=device, dtype=torch.long).reshape(-1)
            if delay_ids.numel() == 1:
                delay_ids = delay_ids.expand(token_count)
            elif delay_ids.numel() != token_count:
                delay_ids = delay_ids[:1].expand(token_count)
        elif isinstance(num_delay_tokens, (list, tuple)):
            if not num_delay_tokens:
                value = int(getattr(self.config, "default_num_delay_tokens", 6))
            else:
                value = int(num_delay_tokens[0])
            delay_ids = torch.full((token_count,), value, dtype=torch.long, device=device)
        else:
            delay_ids = torch.full(
                (token_count,),
                int(num_delay_tokens),
                dtype=torch.long,
                device=device,
            )
        max_delay_tokens = int(getattr(self.config, "max_delay_tokens", 64))
        return delay_ids.clamp_(min=0, max=max_delay_tokens)

    def _forward_query_start_loc_candidates(
        self,
    ) -> list[tuple[torch.Tensor, int]]:
        """All (query_start_loc, num_reqs) candidates from the forward context.

        The context can carry several attention metadata objects (encoder
        sub-position units vs decoder token units).  Callers pick the one
        consistent with the scheduled token-row count; the first available
        metadata is not guaranteed to be in token units.
        """
        candidates: list[tuple[torch.Tensor, int]] = []
        try:
            from vllm.forward_context import (
                get_forward_context,
                is_forward_context_available,
            )

            if not is_forward_context_available():
                return candidates
            attn_metadata = get_forward_context().attn_metadata
            if isinstance(attn_metadata, dict):
                entries = list(attn_metadata.values())
            elif isinstance(attn_metadata, (list, tuple)):
                entries = [
                    item
                    for entry in attn_metadata
                    for item in (
                        entry.values() if isinstance(entry, dict) else (entry,)
                    )
                ]
            else:
                entries = [attn_metadata]
            for metadata in entries:
                query_start_loc = getattr(metadata, "query_start_loc", None)
                if query_start_loc is None or not torch.is_tensor(query_start_loc):
                    continue
                num_reqs = getattr(metadata, "num_reqs", None)
                if num_reqs is None:
                    num_reqs = int(query_start_loc.numel()) - 1
                candidates.append((query_start_loc, int(num_reqs)))
        except Exception:
            logger.debug(
                "Audio8 ASR Infinite realtime failed to read query_start_loc from the "
                "forward context.",
                exc_info=True,
            )
        return candidates

    def _default_single_session_state(
        self,
        *,
        num_row_groups: int,
    ) -> Audio8RealtimeSessionState:
        """Session state for a legacy (no pending mm item) forward.

        Real realtime windows always carry mm items, so this path only fires
        for pre-salt deployments (single salt-0 state) and for steps without
        audio identity (e.g. startup profiling).  With several states live we
        pick the most recently used one rather than guessing silently; a
        genuine multi-session step always routes via the mm items instead.
        """
        if len(self._session_states) == 1:
            return self._touch_session_state(next(iter(self._session_states)))
        if self._session_states:
            if num_row_groups > 1:
                raise RuntimeError(
                    "Audio8 ASR Infinite realtime forward received multiple token row "
                    "groups without mm item identity; cannot route "
                    "per-session KV safely."
                )
            salt = max(
                self._session_states,
                key=lambda key: self._session_states[key].last_used_tick,
            )
            logger.warning(
                "Audio8 ASR Infinite realtime forward without mm item identity while "
                "several session states are live; routing to the most "
                "recently used session (salt=%s).",
                salt,
            )
            return self._touch_session_state(salt)
        return self._touch_session_state(0)

    def _resolve_forward_session_groups(
        self,
        *,
        total_rows: int,
        positions: torch.Tensor,
    ) -> list[tuple[int, int, int, int]] | None:
        """Partition this step's token rows per realtime session.

        Returns a list of ``(salt, delay_tokens, row_start, row_end)`` tuples
        in row order, or ``None`` for the legacy single-session forward (no
        pending mm items).  Row groups come from the forward context's
        ``query_start_loc`` (per-request boundaries, input-batch order); the
        (salt, delay) identity comes from the mm items recorded by
        ``embed_multimodal`` in the same engine step, whose scheduling order
        matches the batch order because streaming requests are removed and
        re-added to the input batch on every window update.
        """
        items = self._pending_mm_items
        self._pending_mm_items = []
        if not items:
            return None
        if not any(int(salt) != 0 for salt, _, _ in items):
            # No session identity (startup profiling/dummy run, or a
            # pre-cache_salt deployment): keep the historical single-session
            # forward untouched.
            return None
        # Merge consecutive same-salt items (windows of one request scheduled
        # together) into one run.
        runs: list[tuple[int, int, int]] = []
        for salt, delay, rows in items:
            if runs and runs[-1][0] == salt:
                merged_salt, merged_delay, merged_rows = runs[-1]
                runs[-1] = (merged_salt, int(delay), merged_rows + rows)
            else:
                runs.append((int(salt), int(delay), int(rows)))
        total_run_rows = sum(rows for _, _, rows in runs)
        if total_run_rows > total_rows:
            # Profiling/dummy runs can call embed_multimodal without a
            # row-matched forward, and an aborted or preempted window can leave
            # items behind.  Degrade to the legacy single-session routing
            # instead of failing the live generation: with the supported
            # realtime configuration (one request per engine step) that routing
            # is exact, so a routing hiccup must not kill the connection.
            logger.warning(
                "Audio8 ASR Infinite realtime mm item rows (%s) exceed the scheduled "
                "token rows (%s); falling back to legacy single-session routing.",
                total_run_rows,
                total_rows,
            )
            return None

        spans: list[tuple[int, int]] | None = None
        for query_start_loc, num_reqs in self._forward_query_start_loc_candidates():
            boundaries = query_start_loc.detach().reshape(-1).tolist()
            if len(boundaries) < num_reqs + 1:
                continue
            candidate_spans = [
                (int(boundaries[index]), int(boundaries[index + 1]))
                for index in range(num_reqs)
                if int(boundaries[index + 1]) > int(boundaries[index])
            ]
            # Only accept a metadata whose spans are in the same units as the
            # scheduled token rows: encoder-side metadata counts sub-positions
            # (pool_size per token row) and must be skipped.
            if (
                candidate_spans
                and candidate_spans[0][0] == 0
                and candidate_spans[-1][1] == total_rows
            ):
                spans = candidate_spans
                break
        if spans is None:
            # Fallback (no token-unit metadata available): derive boundaries
            # from the mm runs themselves.
            offset = 0
            spans = []
            for _, _, rows in runs:
                spans.append((offset, offset + rows))
                offset += rows

        if len(spans) == 1:
            # One request in the batch: every item belongs to it; leftover
            # rows (beyond the mm items) are its text-prefix rows.
            if len(set(salt for salt, _, _ in runs)) != 1:
                logger.warning(
                    "Audio8 ASR Infinite realtime single request scheduled items from "
                    "multiple session salts %s; falling back to legacy "
                    "single-session routing.",
                    sorted({int(salt) for salt, _, _ in runs}),
                )
                return None
            salt, delay, _ = runs[0]
            start, end = spans[0]
            return [(salt, delay, start, end)]

        # Several requests in the batch: greedily fill each request's span
        # with whole mm runs in batch order; a request's leftover span rows
        # (mm rows never exceed its span) are its text-prefix rows and route
        # to the same session.
        groups: list[tuple[int, int, int, int]] = []
        cursor = 0
        matched_rows = 0
        for start, end in spans:
            span_rows = end - start
            first_run: tuple[int, int, int] | None = None
            span_matched_rows = 0
            while (
                cursor < len(runs)
                and span_matched_rows + runs[cursor][2] <= span_rows
            ):
                run = runs[cursor]
                if first_run is None:
                    first_run = run
                span_matched_rows += run[2]
                cursor += 1
            matched_rows += span_matched_rows
            if first_run is None:
                logger.warning(
                    "Audio8 ASR Infinite realtime request row group without any mm item "
                    "(span=(%s, %s)); falling back to legacy single-session routing.",
                    start,
                    end,
                )
                return None
            groups.append((first_run[0], first_run[1], start, end))
        if cursor < len(runs):
            logger.warning(
                "Audio8 ASR Infinite realtime %s mm runs left over after matching all "
                "request row groups (%s rows); falling back to legacy "
                "single-session routing.",
                len(runs) - cursor,
                total_run_rows - matched_rows,
            )
            return None
        for group_salt, _, start, end in groups:
            span_positions = positions.detach()[start:end]
            if span_positions.numel() > 1:
                deltas = span_positions[1:] - span_positions[:-1]
                if not bool((deltas == 1).all()):
                    # Not fatal: every row keeps its own absolute position, so
                    # the session still receives a valid window.  Warn rather
                    # than kill a live generation.
                    logger.warning(
                        "Audio8 ASR Infinite realtime session row positions are not "
                        "consecutive within one forward (salt=%s span=(%s, %s)).",
                        group_salt,
                        start,
                        end,
                    )
        return groups

    def _forward_single_session(
        self,
        *,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        raw_multimodal_embeds: torch.Tensor,
        state: Audio8RealtimeSessionState,
        num_delay_tokens: object,
        frame_len: object,
    ) -> torch.Tensor:
        """Run one session's rows through tower, projector and HF decoder.

        Both the single-session path and the multi-session fan-out use this
        helper; the session state supplies the rolling HF KV and the tower
        eager caches, so the rolling semantics verified in the single-session
        runbook apply per session unchanged.
        """
        pool_size = int(self.downsample_factor)
        inputs_embeds = raw_multimodal_embeds.reshape(
            raw_multimodal_embeds.shape[0] * pool_size,
            raw_multimodal_embeds.shape[1] // pool_size,
        ).clone()
        expanded_audio_positions = _expand_positions(positions, pool_size)
        self._load_tower_caches_for_state(state)
        try:
            audio_hidden_states = self.audio_tower.whisper_encoder(
                inputs_embeds,
                expanded_audio_positions,
            )
        finally:
            self._save_tower_caches_for_state(state)
        text_embeds = self._embed_text_input_ids(input_ids)
        audio_embeds = self._project_audio_hidden_states(audio_hidden_states, text_embeds)
        inputs_embeds = text_embeds + audio_embeds
        t_cond = self._build_t_cond_for_forward(
            num_delay_tokens,
            device=inputs_embeds.device,
            dtype=inputs_embeds.dtype,
            frame_len=(
                frame_len
                if frame_len is not None
                else (state.active_frame_len or self._active_frame_len)
            ),
        )
        state.active_frame_len = int(
            frame_len if frame_len is not None else self._active_frame_len
        )
        hidden_states = self._run_language_model(
            input_ids=input_ids,
            positions=positions,
            inputs_embeds=inputs_embeds,
            t_cond=t_cond,
            state=state,
        )
        if positions.numel():
            state.next_position = int(positions.detach().reshape(-1)[-1].item()) + 1
        return hidden_states

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        # 只在这里打时间戳：`compute_logits` 紧跟本步 forward，两者之间的墙钟时间就是
        # 这一步的模型侧耗时，用来算 RTF（= 本步耗时 / 本步音频时长）并发布到侧信道。
        self._metrics_step_started_at = time.monotonic()
        return self._forward_step(
            input_ids,
            positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

    def _forward_step(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        positions = _as_1d_positions(positions)
        total_rows = int(positions.numel())
        session_groups = self._resolve_forward_session_groups(
            total_rows=total_rows,
            positions=positions,
        )
        # Remember which sessions this step served so `compute_logits` (called
        # right after this forward) can route semantic VAD predictions back to
        # the matching realtime connection.
        self._vad_recent_salts = (
            [int(group[0]) for group in session_groups] if session_groups else []
        )
        if intermediate_tensors is not None:
            if session_groups and len(session_groups) > 1:
                raise NotImplementedError(
                    "Audio8 ASR Infinite realtime intermediate-tensors forward does not "
                    "support multiple concurrent sessions."
                )
            if session_groups:
                state = self._touch_session_state(
                    session_groups[0][0],
                    num_delay_tokens=session_groups[0][1],
                )
            else:
                state = self._default_single_session_state(num_row_groups=1)
                self._vad_recent_salts = [int(state.salt)]
            if input_ids is None:
                raise ValueError("Audio8 ASR Infinite realtime intermediate forward requires input_ids.")
            text_embeds = self._embed_text_input_ids(input_ids)
            t_cond = self._build_t_cond_for_forward(
                kwargs.get("num_delay_tokens"),
                device=text_embeds.device,
                dtype=text_embeds.dtype,
                frame_len=kwargs.get("frame_len", self._active_frame_len),
            )
            hidden_states = self._run_language_model(
                input_ids=input_ids,
                positions=positions,
                inputs_embeds=text_embeds,
                t_cond=t_cond,
                state=state,
                intermediate_tensors=intermediate_tensors,
            )
            return hidden_states
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("Audio8 ASR Infinite realtime forward requires input_ids or inputs_embeds.")
            if session_groups and len(session_groups) > 1:
                raise NotImplementedError(
                    "Audio8 ASR Infinite realtime text-only forward does not support "
                    "multiple concurrent sessions."
                )
            if session_groups:
                state = self._touch_session_state(
                    session_groups[0][0],
                    num_delay_tokens=session_groups[0][1],
                )
            else:
                state = self._default_single_session_state(num_row_groups=1)
                self._vad_recent_salts = [int(state.salt)]
            text_embeds = self._embed_text_input_ids(input_ids)
            audio_embeds = torch.zeros_like(text_embeds)
            inputs_embeds = text_embeds + audio_embeds
            t_cond = self._build_t_cond_for_forward(
                kwargs.get("num_delay_tokens"),
                device=inputs_embeds.device,
                dtype=inputs_embeds.dtype,
                frame_len=kwargs.get("frame_len", self._active_frame_len),
            )
            if input_ids is None:
                raise ValueError("Audio8 ASR Infinite realtime HF language forward requires input_ids.")
            hidden_states = self._run_language_model(
                input_ids=input_ids,
                positions=positions,
                inputs_embeds=inputs_embeds,
                t_cond=t_cond,
                state=state,
            )
            return hidden_states
        if input_ids is None:
            raise ValueError("Audio8 ASR Infinite realtime forward requires raw input_ids with inputs_embeds.")
        if session_groups and len(session_groups) > 1:
            # Multi-session fan-out: run each session's rows through its own
            # rolling state (HF KV + tower eager caches + t_cond), then
            # reassemble the hidden states in the original row order.
            hidden_parts: list[torch.Tensor] = []
            for salt, delay, start, end in session_groups:
                state = self._touch_session_state(salt, num_delay_tokens=delay)
                hidden_part = self._forward_single_session(
                    input_ids=input_ids[start:end],
                    positions=positions[start:end],
                    raw_multimodal_embeds=inputs_embeds[start:end],
                    state=state,
                    num_delay_tokens=delay,
                    frame_len=_first_scalar_value(kwargs.get("frame_len")),
                )
                hidden_parts.append(hidden_part)
            hidden_states = torch.cat(hidden_parts, dim=0)
            return hidden_states
        if session_groups:
            salt, delay, start, end = session_groups[0]
            state = self._touch_session_state(salt, num_delay_tokens=delay)
            forward_delay: object = delay
        else:
            state = self._default_single_session_state(num_row_groups=1)
            self._vad_recent_salts = [int(state.salt)]
            forward_delay = kwargs.get("num_delay_tokens")
        hidden_states = self._forward_single_session(
            input_ids=input_ids[start:end] if session_groups else input_ids,
            positions=positions[start:end] if session_groups else positions,
            raw_multimodal_embeds=(
                inputs_embeds[start:end] if session_groups else inputs_embeds
            ),
            state=state,
            num_delay_tokens=forward_delay,
            frame_len=_first_scalar_value(kwargs.get("frame_len")),
        )
        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        # 指标先发布：它要读本次 forward 记下的 salt 与起始时间戳，而
        # `_publish_semantic_vad` 会把 salt 列表消费掉。
        self._publish_metrics()
        self._publish_semantic_vad(hidden_states)
        logits = self._language_logits(hidden_states)
        return logits

    def _publish_metrics(self) -> None:
        """Publish this step's RTF and GPU memory on the side channel.

        ``rtf`` is the model-side time for one step (forward through logits, as
        measured by the wall clock between ``forward`` and ``compute_logits``)
        divided by the audio duration that step consumed: below 1 means the
        model keeps up with real time.  Unlike semantic VAD this is published
        for every checkpoint, so the client's GPU/RTF timelines also work for a
        transcription-only model.  The connection turns each payload into a
        ``metrics.delta`` event.
        """

        started = self._metrics_step_started_at
        if started is None:
            # `compute_logits` was already served for this step.
            return
        self._metrics_step_started_at = None
        salts = self._vad_recent_salts
        if not salts:
            return
        step_ms = max(0.0, (time.monotonic() - started) * 1000.0)
        frame_ms = int(getattr(self.config, "audio_tower_frame_ms", 20) or 20)
        frame_len = int(self._active_frame_len or 0) or int(self.downsample_factor)
        audio_ms = float(frame_len * frame_ms)
        rtf = (step_ms / audio_ms) if audio_ms > 0 else 0.0
        previous = self._metrics_rtf_ema
        self._metrics_rtf_ema = (
            rtf
            if previous is None
            else (1.0 - _AUDIO8_METRICS_RTF_EMA_ALPHA) * previous
            + _AUDIO8_METRICS_RTF_EMA_ALPHA * rtf
        )
        self._metrics_step_counter += 1
        if (
            self._metrics_gpu_memory is None
            or self._metrics_step_counter % _AUDIO8_METRICS_GPU_SAMPLE_STEPS == 0
        ):
            self._metrics_gpu_memory = _audio8_gpu_memory_bytes()
        used_bytes, total_bytes = self._metrics_gpu_memory
        payload = {
            "type": "metrics",
            "step_ms": step_ms,
            "audio_ms": audio_ms,
            "rtf": rtf,
            "rtf_ema": float(self._metrics_rtf_ema or 0.0),
            "gpu_memory_used_bytes": int(used_bytes),
            "gpu_memory_total_bytes": int(total_bytes),
            "gpu_memory_used_ratio": (float(used_bytes) / float(total_bytes)) if total_bytes else 0.0,
        }
        for salt in dict.fromkeys(int(value) for value in salts):
            audio8_vad_channel.publish(salt, payload)

    def _publish_semantic_vad(self, hidden_states: torch.Tensor) -> None:
        """Score the semantic VAD heads on this step's sampled hidden states.

        `hidden_states` are the states the language head consumes for sampling,
        i.e. exactly the positions the realtime client is about to receive text
        for.  Predictions are pushed to the side channel keyed by the session
        salt observed in the forward that produced them, and the connection
        turns them into `semantic_vad.delta` events.
        """

        heads = self.semantic_vad_heads
        if heads is None:
            return
        # Consume the salts recorded by the forward that produced these states,
        # so a second `compute_logits` call on the same step cannot publish the
        # same prediction twice.
        salts = self._vad_recent_salts
        self._vad_recent_salts = []
        if not salts:
            return
        with torch.inference_mode():
            stacked_logits = torch.stack(
                [head(hidden_states) for head in heads],
                dim=1,
            )
            probabilities = torch.softmax(stacked_logits.float(), dim=-1).cpu()
        num_rows = int(probabilities.shape[0])
        if len(salts) == num_rows:
            row_salts = salts
        elif len(salts) == 1:
            row_salts = [salts[0]] * num_rows
        else:
            # Ambiguous row-to-session mapping (more sessions than row groups);
            # skip rather than attribute a prediction to the wrong session.
            return
        horizons = [float(horizon) for horizon in self.semantic_vad_horizons_seconds]
        selected_horizon = float(
            os.environ.get("AUDIO8_VAD_EOT_HORIZON_SECONDS", "2.0")
        )
        try:
            selected_index = horizons.index(selected_horizon)
        except ValueError:
            selected_index = min(2, len(horizons) - 1)
            selected_horizon = horizons[selected_index]
        for row, salt in enumerate(row_salts):
            row_probabilities = probabilities[row]
            audio8_vad_channel.publish(
                int(salt),
                {
                    "type": "semantic_vad",
                    "horizons_seconds": horizons,
                    "probabilities": row_probabilities.tolist(),
                    "predicted_classes": [
                        int(value) for value in row_probabilities.argmax(dim=-1).tolist()
                    ],
                    "selected_eot_horizon_seconds": selected_horizon,
                    # 类别 0 是「该 horizon 内不讲话 / EOT」，所以这里是“不讲话”的概率；
                    # 前端展示的“在说话”概率 = 1 - 该值，与 Silero 同向（1=在说话）。
                    "selected_eot_probability": float(
                        row_probabilities[selected_index][0].item()
                    ),
                },
            )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loaded_weights: set[str] = set()
        local_params = {
            **{
                f"multi_modal_projector.{name}": parameter
                for name, parameter in self.multi_modal_projector.named_parameters()
            },
            **(
                {
                    f"frame_len_embedding.{name}": parameter
                    for name, parameter in self.frame_len_embedding.named_parameters()
                }
                if self.frame_len_embedding is not None
                else {}
            ),
            **(
                {
                    f"semantic_vad_heads.{name}": parameter
                    for name, parameter in self.semantic_vad_heads.named_parameters()
                }
                if self.semantic_vad_heads is not None
                else {}
            ),
        }
        language_weights: list[tuple[str, torch.Tensor]] = []

        for name, weight in weights:
            if name.startswith("language_model."):
                language_weights.append((name.removeprefix("language_model."), weight))
                continue
            if name.startswith("audio_tower."):
                audio_name = name.removeprefix("audio_tower.")
                for audio_weight in _iter_audio8_audio_tower_load_items(
                    audio_name,
                    weight,
                ):
                    loaded_name = self.audio_tower.load_weight(audio_weight)
                    loaded_weights.add(f"audio_tower.{loaded_name}")
                continue
            parameter = local_params.get(name)
            if parameter is not None:
                default_weight_loader(parameter, weight)
                loaded_weights.add(name)

        if self._hf_language_model is None:
            for name in self.language_model.load_weights(language_weights):
                loaded_weights.add(f"language_model.{name}")
        hf_language_state: dict[str, torch.Tensor] = {}
        if self._hf_language_model is not None:
            hf_language_state = {
                name: weight.detach().cpu()
                for name, weight in language_weights
                if not (self.config.text_config.tie_word_embeddings and name.startswith("lm_head."))
            }
            hf_load_result = self._hf_language_model.load_state_dict(
                hf_language_state,
                strict=False,
            )
            allowed_hf_missing_names = (
                {"lm_head.weight"} if self.config.text_config.tie_word_embeddings else set()
            )
            unexpected_hf_names = set(hf_load_result.unexpected_keys)
            missing_hf_names = set(hf_load_result.missing_keys) - allowed_hf_missing_names
            if unexpected_hf_names or missing_hf_names:
                raise RuntimeError(
                    "Audio8 ASR Infinite HF language decoder weight load mismatch: "
                    f"missing={sorted(missing_hf_names)} "
                    f"unexpected={sorted(unexpected_hf_names)}"
                )
            for name in hf_language_state:
                loaded_weights.add(f"language_model.{name}")
            for name in allowed_hf_missing_names:
                loaded_weights.add(f"language_model.{name}")
            self._hf_language_model.to(device=self._audio_device(), dtype=torch.bfloat16)
            self._hf_language_model.eval()
        return loaded_weights

    def get_mm_mapping(self) -> MultiModelKeys:
        return MultiModelKeys.from_string_field(
            language_model="language_model",
            tower_model=["audio_tower."],
        )

    @classmethod
    def get_speech_to_text_config(
        cls,
        model_config: ModelConfig,
        task_type: str,
    ) -> SpeechToTextConfig:
        _ = task_type
        feature_extractor = _load_feature_extractor_from_model_config(model_config)
        return SpeechToTextConfig(
            max_audio_clip_s=None,
            sample_rate=int(getattr(feature_extractor, "sampling_rate", 16000)),
            min_energy_split_window_size=None,
        )

    def _audio_device(self) -> torch.device:
        for parameter in self.audio_tower.parameters():
            return parameter.device
        return torch.device("cpu")
