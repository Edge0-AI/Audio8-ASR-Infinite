"""Configuration for Audio8 ASR Infinite."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from transformers import PretrainedConfig, Qwen2Config, Qwen3Config
from transformers.models.voxtral_realtime.configuration_voxtral_realtime import (
    VoxtralRealtimeEncoderConfig,
)


AUDIO8_ASR_INFINITE_WEIGHT_FORMAT_VERSION = 2

# Semantic VAD head contract: one classifier per future horizon, each predicting
# the number of semantic units (0..num_classes-1) that will appear within that
# horizon.  Class 0 is the end-of-turn class the realtime client thresholds on.
DEFAULT_SEMANTIC_VAD_NUM_CLASSES = 8
DEFAULT_SEMANTIC_VAD_HORIZONS_SECONDS: tuple[float, ...] = (0.5, 1.0, 2.0, 3.0)


class Audio8ASRInfiniteConfig(PretrainedConfig):
    """Configuration for the Audio8 ASR Infinite model."""

    model_type = "audio8_asr_infinite"
    sub_configs = {
        "audio_config": VoxtralRealtimeEncoderConfig,
        "text_config": Qwen3Config,
    }

    def __init__(
        self,
        audio_config: Mapping[str, Any] | VoxtralRealtimeEncoderConfig | None = None,
        text_config: Mapping[str, Any] | Qwen2Config | Qwen3Config | None = None,
        audio_length_per_tok: int = 8,
        default_num_delay_tokens: int | None = None,
        supported_frame_lens: Sequence[int] = (4, 6, 8),
        audio_tower_frame_ms: int = 20,
        use_frame_len_embedding: bool = False,
        projector_hidden_act: str = "gelu",
        semantic_vad_horizons_seconds: Sequence[float] | None = None,
        semantic_vad_num_classes: int = DEFAULT_SEMANTIC_VAD_NUM_CLASSES,
        weight_format_version: int = AUDIO8_ASR_INFINITE_WEIGHT_FORMAT_VERSION,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if audio_config is None:
            audio_config = VoxtralRealtimeEncoderConfig()
        if isinstance(audio_config, Mapping):
            audio_config = VoxtralRealtimeEncoderConfig(**dict(audio_config))
        if text_config is None:
            text_config = Qwen3Config()
        if isinstance(text_config, Mapping):
            text_config_payload = dict(text_config)
            text_model_type = text_config_payload.pop(
                "model_type",
                Qwen3Config.model_type,
            )
            text_config_class = {
                Qwen2Config.model_type: Qwen2Config,
                Qwen3Config.model_type: Qwen3Config,
            }.get(str(text_model_type))
            if text_config_class is None:
                raise ValueError(
                    "Audio8 ASR Infinite text_config must use Qwen2 or Qwen3, "
                    f"got model_type={text_model_type!r}."
                )
            text_config = text_config_class(**text_config_payload)

        self.audio_config = audio_config
        self.text_config = text_config
        self.tie_word_embeddings = bool(text_config.tie_word_embeddings)
        self.audio_length_per_tok = int(audio_length_per_tok)
        self.default_num_delay_tokens = (
            None
            if default_num_delay_tokens is None
            else int(default_num_delay_tokens)
        )
        self.supported_frame_lens = tuple(
            int(value) for value in supported_frame_lens
        )
        self.audio_tower_frame_ms = int(audio_tower_frame_ms)
        if self.audio_tower_frame_ms <= 0:
            raise ValueError("audio_tower_frame_ms must be positive.")
        if (
            not self.supported_frame_lens
            or any(value <= 0 for value in self.supported_frame_lens)
            or len(set(self.supported_frame_lens))
            != len(self.supported_frame_lens)
        ):
            raise ValueError(
                "supported_frame_lens must contain unique positive integers."
            )
        self.max_frame_len = max(self.supported_frame_lens)
        self.use_frame_len_embedding = bool(
            use_frame_len_embedding
            and len(self.supported_frame_lens) > 1
        )
        self.projector_hidden_act = str(projector_hidden_act)
        # `None` / empty means "no semantic VAD heads": plain transcription
        # checkpoints are unaffected and their weight keys are unchanged.
        self.semantic_vad_horizons_seconds = (
            None
            if semantic_vad_horizons_seconds is None
            else [float(horizon) for horizon in semantic_vad_horizons_seconds]
        )
        if self.semantic_vad_horizons_seconds is not None and (
            not self.semantic_vad_horizons_seconds
            or any(
                horizon <= 0.0 for horizon in self.semantic_vad_horizons_seconds
            )
        ):
            raise ValueError(
                "semantic_vad_horizons_seconds must be a non-empty sequence of "
                "positive numbers, or None for a transcription-only checkpoint."
            )
        self.semantic_vad_num_classes = int(semantic_vad_num_classes)
        if self.semantic_vad_num_classes < 2:
            raise ValueError("semantic_vad_num_classes must be at least 2.")
        self.weight_format_version = int(weight_format_version)
        if (
            self.weight_format_version
            != AUDIO8_ASR_INFINITE_WEIGHT_FORMAT_VERSION
        ):
            raise ValueError(
                "Unsupported Audio8 ASR Infinite weight format: "
                f"expected={AUDIO8_ASR_INFINITE_WEIGHT_FORMAT_VERSION} "
                f"got={self.weight_format_version}. Convert the checkpoint "
                "before loading it."
            )
        self.projection_size = (
            int(self.audio_config.hidden_size)
            * self.max_frame_len
        )
        self.text_config.projection_size = self.projection_size
        self.vocab_size = int(text_config.vocab_size)
        self.hidden_size = int(text_config.hidden_size)
        self.pad_token_id = text_config.pad_token_id
        self.bos_token_id = text_config.bos_token_id
        self.eos_token_id = text_config.eos_token_id

    @classmethod
    def from_dict(
        cls,
        config_dict: dict[str, Any],
        **kwargs: Any,
    ) -> Audio8ASRInfiniteConfig:
        if config_dict.get("model_type") != cls.model_type:
            raise ValueError(
                "Audio8 ASR Infinite only loads its own checkpoint format: "
                f"expected model_type={cls.model_type!r}, got "
                f"{config_dict.get('model_type')!r}. Convert the checkpoint "
                "before loading it."
            )
        if "weight_format_version" not in config_dict:
            raise ValueError(
                "Audio8 ASR Infinite checkpoint is missing "
                "`weight_format_version`. Convert the checkpoint to the "
                "current format before loading it."
            )
        return super().from_dict(config_dict, **kwargs)

    def to_dict(self) -> dict[str, Any]:
        output = super().to_dict()
        output["audio_config"] = self.audio_config.to_dict()
        output["text_config"] = self.text_config.to_dict()
        output["audio_length_per_tok"] = self.audio_length_per_tok
        output["default_num_delay_tokens"] = self.default_num_delay_tokens
        output["supported_frame_lens"] = list(
            self.supported_frame_lens
        )
        output["max_frame_len"] = self.max_frame_len
        output["audio_tower_frame_ms"] = self.audio_tower_frame_ms
        output["use_frame_len_embedding"] = (
            self.use_frame_len_embedding
        )
        output["projector_hidden_act"] = self.projector_hidden_act
        output["weight_format_version"] = self.weight_format_version
        output["projection_size"] = self.projection_size
        output["text_config"]["projection_size"] = self.projection_size
        output["model_type"] = self.model_type
        return output

    def resolve_frame_len(
        self,
        streaming_frame_ms: int,
    ) -> int:
        streaming_frame_ms = int(streaming_frame_ms)
        if streaming_frame_ms <= 0:
            raise ValueError("streaming_frame_ms must be positive.")
        if streaming_frame_ms % self.audio_tower_frame_ms != 0:
            raise ValueError(
                "streaming_frame_ms must be divisible by "
                f"audio_tower_frame_ms={self.audio_tower_frame_ms}, got "
                f"{streaming_frame_ms}."
            )
        frame_len = (
            streaming_frame_ms // self.audio_tower_frame_ms
        )
        if frame_len not in self.supported_frame_lens:
            raise ValueError(
                "streaming_frame_ms resolves to unsupported "
                f"frame_len={frame_len}; supported="
                f"{self.supported_frame_lens}."
            )
        return frame_len

    def resolve_num_delay_tokens(
        self,
        *,
        target_delay_ms: int,
        streaming_frame_ms: int,
    ) -> int:
        self.resolve_frame_len(streaming_frame_ms)
        target_delay_ms = int(target_delay_ms)
        if target_delay_ms <= 0:
            raise ValueError("target_delay_ms must be positive.")
        if target_delay_ms % int(streaming_frame_ms) != 0:
            raise ValueError(
                "target_delay_ms must be divisible by streaming_frame_ms, "
                f"got target_delay_ms={target_delay_ms} "
                f"streaming_frame_ms={streaming_frame_ms}."
            )
        return target_delay_ms // int(streaming_frame_ms)


__all__ = [
    "AUDIO8_ASR_INFINITE_WEIGHT_FORMAT_VERSION",
    "Audio8ASRInfiniteConfig",
]
