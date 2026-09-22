"""Audio8 ASR Infinite variant with Voxtral-style delay conditioning."""

from .modeling.configuration_audio8_asr_infinite import (
    AUDIO8_ASR_INFINITE_WEIGHT_FORMAT_VERSION,
    Audio8ASRInfiniteConfig,
)
from .modeling.modeling_audio8_asr_infinite import (
    Audio8ASRInfiniteForCausalLM,
    Audio8ASRInfiniteForConditionalGeneration,
    Audio8ASRInfiniteMaxFrameLenProjector,
    Audio8ASRInfiniteQwen2ForCausalLM,
    Audio8ASRInfiniteQwen2TextModel,
    Audio8ASRInfiniteTextModel,
    LANGUAGE_EN_TOKEN,
    LANGUAGE_ZH_TOKEN,
    STREAMING_PAD_TOKEN,
    STREAMING_WORD_TOKEN,
    Qwen2RealtimeV1DecoderLayer,
    ensure_voxtral_streaming_tokens,
    resolve_qwen_language_token_id,
    resolve_qwen_streaming_special_token_ids,
)

__all__ = [
    "AUDIO8_ASR_INFINITE_WEIGHT_FORMAT_VERSION",
    "Audio8ASRInfiniteConfig",
    "Audio8ASRInfiniteForCausalLM",
    "Audio8ASRInfiniteForConditionalGeneration",
    "Audio8ASRInfiniteMaxFrameLenProjector",
    "Audio8ASRInfiniteQwen2ForCausalLM",
    "Audio8ASRInfiniteQwen2TextModel",
    "Audio8ASRInfiniteTextModel",
    "LANGUAGE_EN_TOKEN",
    "LANGUAGE_ZH_TOKEN",
    "STREAMING_PAD_TOKEN",
    "STREAMING_WORD_TOKEN",
    "Qwen2RealtimeV1DecoderLayer",
    "ensure_voxtral_streaming_tokens",
    "resolve_qwen_language_token_id",
    "resolve_qwen_streaming_special_token_ids",
]
