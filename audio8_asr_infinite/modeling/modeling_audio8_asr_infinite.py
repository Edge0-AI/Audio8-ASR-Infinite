"""Audio8 ASR Infinite with Voxtral Realtime delay conditioning and a Qwen decoder.

This model intentionally lives next to, not on top of, the existing
legacy ``audio8_streaming_asr`` implementation.  The audio tower,
projector, tokenizer contract, and Qwen decoder/head are preserved.  The delay
path is changed from Audio8 ASR Infinite's input-level learned ``delay_embedding`` to the
Voxtral Realtime mechanism:

``num_delay_tokens -> sinusoidal time embedding -> per-layer adaptive MLP -> post-attention hidden scaling``.
"""

from __future__ import annotations

from types import GeneratorType
from typing import Any, Sequence

import torch
from torch import nn
from torch.nn import functional as F
from transformers import (
    AutoModel,
    PreTrainedModel,
    Qwen2Config,
    Qwen3Config,
)
from transformers.generation import GenerationMixin
from transformers.activations import ACT2FN
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2DecoderLayer,
    Qwen2ForCausalLM,
    Qwen2Model,
    Qwen2PreTrainedModel,
    Qwen2RMSNorm,
    Qwen2RotaryEmbedding,
)
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3DecoderLayer,
    Qwen3ForCausalLM,
    Qwen3Model,
    Qwen3PreTrainedModel,
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
)
from transformers.models.voxtral_realtime import modeling_voxtral_realtime as _voxtral_realtime_modeling

from .configuration_audio8_asr_infinite import (
    AUDIO8_ASR_INFINITE_WEIGHT_FORMAT_VERSION,
    DEFAULT_SEMANTIC_VAD_NUM_CLASSES,
    Audio8ASRInfiniteConfig,
)

VoxtralRealtimeTextAdaRmsNorm = _voxtral_realtime_modeling.VoxtralRealtimeTextAdaRmsNorm
VoxtralRealtimeTimeEmbedding = _voxtral_realtime_modeling.VoxtralRealtimeTimeEmbedding

# The model's Qwen tokenizer special-token contract: the five core special ids
# consumed by the simulated-streaming decoder, plus the streaming / language
# special-token strings.
STREAMING_PAD_TOKEN = "[STREAMING_PAD]"
STREAMING_WORD_TOKEN = "[STREAMING_WORD]"
LANGUAGE_ZH_TOKEN = "[LANGUAGE_ZH]"
LANGUAGE_EN_TOKEN = "[LANGUAGE_EN]"
QWEN_AUDIO_PAD_TOKEN = "<|audio_pad|>"
QWEN_ASR_TEXT_TOKEN = "<asr_text>"


def resolve_token_id(tokenizer: Any, *, attr_name: str | None, token: str) -> int:
    if attr_name:
        token_id = getattr(tokenizer, attr_name, None)
        if token_id is not None:
            return int(token_id)
    token_id = tokenizer.convert_tokens_to_ids(token)
    if token_id is not None and int(token_id) >= 0:
        return int(token_id)
    raise ValueError(f"Tokenizer cannot resolve required token id for {token!r}.")


def resolve_qwen_streaming_special_token_ids(tokenizer: Any) -> dict[str, int]:
    bos_token_id = resolve_token_id(tokenizer, attr_name="bos_token_id", token="<|im_start|>")
    eos_token_id = resolve_token_id(tokenizer, attr_name="eos_token_id", token="<|im_end|>")
    pad_token_id = resolve_token_id(tokenizer, attr_name="pad_token_id", token="<|endoftext|>")
    streaming_pad_token_id = resolve_token_id(tokenizer, attr_name=None, token=STREAMING_PAD_TOKEN)
    streaming_word_token_id = resolve_token_id(tokenizer, attr_name=None, token=STREAMING_WORD_TOKEN)
    return {
        "bos_token_id": bos_token_id,
        "eos_token_id": eos_token_id,
        "pad_token_id": pad_token_id,
        "streaming_pad_token_id": streaming_pad_token_id,
        "streaming_word_token_id": streaming_word_token_id,
    }


def ensure_voxtral_streaming_tokens(tokenizer: Any) -> int:
    added = tokenizer.add_special_tokens(
        {
            "additional_special_tokens": [
                token
                for token in (
                    STREAMING_PAD_TOKEN,
                    STREAMING_WORD_TOKEN,
                    LANGUAGE_ZH_TOKEN,
                    LANGUAGE_EN_TOKEN,
                )
                if tokenizer.convert_tokens_to_ids(token) is None
                or int(tokenizer.convert_tokens_to_ids(token)) < 0
            ]
        }
    )
    resolve_qwen_streaming_special_token_ids(tokenizer)
    return int(added)



def resolve_qwen_language_token_id(
    tokenizer: Any,
    language: str,
) -> int:
    """把规范语言字段映射到对应的 prompt token。"""

    token_by_language = {
        "zh": LANGUAGE_ZH_TOKEN,
        "en": LANGUAGE_EN_TOKEN,
    }
    normalized = str(language).strip().lower()
    token = token_by_language.get(normalized)
    if token is None:
        raise ValueError(
            "language must be exactly 'zh' or 'en', "
            f"got {language!r}."
        )
    return resolve_token_id(
        tokenizer,
        attr_name=None,
        token=token,
    )


class Audio8ASRInfiniteMaxFrameLenProjector(nn.Module):
    """Project audio groups padded to the configured maximum frame length."""

    def __init__(self, config: Audio8ASRInfiniteConfig) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(
            config.projection_size,
            config.text_config.hidden_size,
            bias=False,
        )
        self.act = ACT2FN[config.projector_hidden_act]
        self.linear_2 = nn.Linear(
            config.text_config.hidden_size,
            config.text_config.hidden_size,
            bias=False,
        )

    def forward(self, audio_features: torch.Tensor) -> torch.Tensor:
        hidden_states = self.linear_1(audio_features)
        hidden_states = self.act(hidden_states)
        return self.linear_2(hidden_states)


class Qwen3RealtimeV1DecoderLayer(Qwen3DecoderLayer):
    """Qwen3 decoder layer with Voxtral-style delay modulation before the MLP."""

    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.ada_rms_norm = VoxtralRealtimeTextAdaRmsNorm(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Any | None = None,
        use_cache: bool | None = False,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        t_cond: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        if t_cond is None:
            raise ValueError("Qwen3RealtimeV1DecoderLayer requires `t_cond`.")
        hidden_states = hidden_states * (1 + self.ada_rms_norm(t_cond).to(dtype=hidden_states.dtype))
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class Qwen2RealtimeV1DecoderLayer(Qwen2DecoderLayer):
    """Qwen2 decoder layer with Voxtral-style delay modulation before the MLP."""

    def __init__(self, config: Qwen2Config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.ada_rms_norm = VoxtralRealtimeTextAdaRmsNorm(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Any | None = None,
        use_cache: bool | None = False,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        t_cond: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        if t_cond is None:
            raise ValueError("Qwen2RealtimeV1DecoderLayer requires `t_cond`.")
        hidden_states = hidden_states * (
            1
            + self.ada_rms_norm(t_cond).to(
                dtype=hidden_states.dtype
            )
        )
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class Audio8ASRInfiniteQwen2TextModel(Qwen2Model):
    """Qwen2 text backbone built directly from Realtime V1 decoder layers."""

    _no_split_modules = ["Qwen2RealtimeV1DecoderLayer"]

    def __init__(self, config: Qwen2Config) -> None:
        Qwen2PreTrainedModel.__init__(self, config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            self.padding_idx,
        )
        self.layers = nn.ModuleList(
            [
                Qwen2RealtimeV1DecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = Qwen2RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.rotary_emb = Qwen2RotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.has_sliding_layers = (
            "sliding_attention" in self.config.layer_types
        )
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Any | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        t_cond: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> Any:
        if t_cond is None:
            raise ValueError(
                "Audio8ASRInfiniteQwen2TextModel requires `t_cond`."
            )
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            t_cond=t_cond,
            **kwargs,
        )


class Audio8ASRInfiniteQwen2ForCausalLM(Qwen2ForCausalLM):
    """Qwen2 causal LM built directly on the Realtime V1 text backbone."""

    _no_split_modules = ["Qwen2RealtimeV1DecoderLayer"]

    def __init__(self, config: Qwen2Config) -> None:
        Qwen2PreTrainedModel.__init__(self, config)
        self.model = Audio8ASRInfiniteQwen2TextModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
        )
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Any | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        t_cond: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> CausalLMOutputWithPast:
        if t_cond is None:
            raise ValueError(
                "Audio8ASRInfiniteQwen2ForCausalLM requires `t_cond`."
            )
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            logits_to_keep=logits_to_keep,
            t_cond=t_cond,
            **kwargs,
        )


class Audio8ASRInfiniteTextModel(Qwen3Model):
    """Qwen3 text backbone built directly from Realtime V1 decoder layers."""

    _no_split_modules = ["Qwen3RealtimeV1DecoderLayer"]

    def __init__(self, config: Qwen3Config) -> None:
        # Skip Qwen3Model.__init__: it would allocate vanilla decoder layers.
        Qwen3PreTrainedModel.__init__(self, config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            self.padding_idx,
        )
        self.layers = nn.ModuleList(
            [
                Qwen3RealtimeV1DecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = Qwen3RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.rotary_emb = Qwen3RotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.has_sliding_layers = (
            "sliding_attention" in self.config.layer_types
        )
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Any | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        t_cond: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> Any:
        if t_cond is None:
            raise ValueError(
                "Audio8ASRInfiniteTextModel requires `t_cond`."
            )
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            t_cond=t_cond,
            **kwargs,
        )


class Audio8ASRInfiniteForCausalLM(Qwen3ForCausalLM):
    """Qwen3 causal LM built directly on the Realtime V1 text backbone."""

    _no_split_modules = ["Qwen3RealtimeV1DecoderLayer"]

    def __init__(self, config: Qwen3Config) -> None:
        # Skip Qwen3ForCausalLM.__init__: the backbone must be native V1.
        Qwen3PreTrainedModel.__init__(self, config)
        self.model = Audio8ASRInfiniteTextModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
        )
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Any | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        t_cond: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> CausalLMOutputWithPast:
        if t_cond is None:
            raise ValueError(
                "Audio8ASRInfiniteForCausalLM requires `t_cond`."
            )
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            logits_to_keep=logits_to_keep,
            t_cond=t_cond,
            **kwargs,
        )


class Audio8ASRInfiniteForConditionalGeneration(PreTrainedModel, GenerationMixin):
    """Voxtral audio tower + Qwen decoder/head + Voxtral-style delay conditioning."""

    config_class = Audio8ASRInfiniteConfig
    base_model_prefix = "audio8_asr_infinite"
    _tied_weights_keys = {
        "language_model.lm_head.weight": (
            "language_model.model.embed_tokens.weight"
        ),
    }
    _no_split_modules = [
        "VoxtralRealtimeEncoderLayer",
        "Qwen2RealtimeV1DecoderLayer",
        "Qwen3RealtimeV1DecoderLayer",
    ]
    supports_gradient_checkpointing = True
    _supports_cache_class = True
    _supports_sdpa = True
    _supports_flash_attn = True
    _supports_flex_attn = True

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | None,
        *model_args: Any,
        **kwargs: Any,
    ) -> Any:
        if kwargs.get("ignore_mismatched_sizes", False):
            raise ValueError(
                "Audio8 ASR Infinite forbids `ignore_mismatched_sizes`; "
                "convert the checkpoint to the exact current weight format."
            )
        return_loading_info = bool(
            kwargs.pop("output_loading_info", False)
        )
        model, loading_info = super().from_pretrained(
            pretrained_model_name_or_path,
            *model_args,
            output_loading_info=True,
            **kwargs,
        )
        incompatible = {
            name: loading_info.get(name)
            for name in (
                "missing_keys",
                "unexpected_keys",
                "mismatched_keys",
                "error_msgs",
            )
            if loading_info.get(name)
        }
        if incompatible:
            raise RuntimeError(
                "Audio8 ASR Infinite checkpoint does not exactly match the "
                "current weight format. Convert it before loading. "
                f"incompatible={incompatible}"
            )
        loaded_time_embedding = model.time_embedding
        time_embedding = VoxtralRealtimeTimeEmbedding(
            model.config.text_config.hidden_size,
            theta=float(getattr(loaded_time_embedding, "theta", 10000.0)),
        )
        loaded_buffer = loaded_time_embedding.inv_freq
        if loaded_buffer.device.type != "meta":
            time_embedding.to(device=loaded_buffer.device)
        if not torch.isfinite(time_embedding.inv_freq).all():
            raise RuntimeError(
                "Voxtral time embedding initialization is non-finite."
            )
        model.time_embedding = time_embedding
        if return_loading_info:
            return model, loading_info
        return model

    def __init__(self, config: Audio8ASRInfiniteConfig) -> None:
        super().__init__(config)
        self.vocab_size = config.text_config.vocab_size
        self.audio_tower = AutoModel.from_config(config.audio_config)
        language_model_class = (
            Audio8ASRInfiniteQwen2ForCausalLM
            if config.text_config.model_type == Qwen2Config.model_type
            else Audio8ASRInfiniteForCausalLM
        )
        self.language_model = language_model_class(config.text_config)
        self.multi_modal_projector = Audio8ASRInfiniteMaxFrameLenProjector(
            config
        )
        self.time_embedding = VoxtralRealtimeTimeEmbedding(config.text_config.hidden_size)
        self.frame_len_embedding = (
            nn.Embedding(
                len(config.supported_frame_lens),
                config.text_config.hidden_size,
            )
            if config.use_frame_len_embedding
            else None
        )
        self.post_init()
        if self.frame_len_embedding is not None:
            nn.init.normal_(
                self.frame_len_embedding.weight,
                mean=0.0,
                std=float(config.text_config.initializer_range),
            )
        # Semantic VAD heads only exist when the checkpoint declares horizons:
        # a plain transcription checkpoint builds none, so its weight keys are
        # unchanged.  One classifier per horizon predicts how many semantic
        # units will appear within that horizon; class 0 is end-of-turn.
        self.semantic_vad_heads: nn.ModuleList | None = None
        self.semantic_vad_horizons_seconds: tuple[float, ...] = ()
        self.semantic_vad_num_classes: int = 0
        configured_horizons = tuple(
            float(horizon)
            for horizon in (
                getattr(config, "semantic_vad_horizons_seconds", None) or ()
            )
        )
        if configured_horizons:
            self.attach_semantic_vad_heads(
                horizons_seconds=configured_horizons,
                num_classes=int(
                    getattr(
                        config,
                        "semantic_vad_num_classes",
                        DEFAULT_SEMANTIC_VAD_NUM_CLASSES,
                    )
                ),
            )

    def attach_semantic_vad_heads(
        self,
        *,
        horizons_seconds: "Sequence[float]",
        num_classes: int = DEFAULT_SEMANTIC_VAD_NUM_CLASSES,
    ) -> nn.ModuleList:
        """Attach the semantic VAD heads: one "future semantic units" classifier
        per horizon.

        Each head reads the text backbone's final hidden state and emits that
        horizon's class logits, matching the ``[batch, horizon, token]`` shape of
        the training labels.  The horizons and the class count are written back
        into the config so a saved checkpoint rebuilds the same heads on load.
        """

        horizons = tuple(float(horizon) for horizon in horizons_seconds)
        if not horizons:
            raise ValueError("semantic VAD horizons must not be empty.")
        num_classes = int(num_classes)
        if num_classes < 2:
            raise ValueError("semantic_vad_num_classes must be at least 2.")
        hidden_size = int(self.config.text_config.hidden_size)
        reference = next(self.language_model.parameters())
        heads = nn.ModuleList(
            [
                nn.Linear(
                    hidden_size,
                    num_classes,
                    bias=True,
                    dtype=reference.dtype,
                )
                for _ in horizons
            ]
        )
        self.semantic_vad_heads = heads
        self.semantic_vad_horizons_seconds = horizons
        self.semantic_vad_num_classes = num_classes
        self.config.semantic_vad_horizons_seconds = list(horizons)
        self.config.semantic_vad_num_classes = num_classes
        return heads

    def _semantic_vad_hidden_norm(self) -> nn.Module:
        """Return the text backbone's final-hidden-state norm layer."""

        text_model = getattr(self.language_model, "model", None)
        norm = getattr(text_model, "norm", None)
        if norm is None:
            raise RuntimeError(
                "Audio8 ASR Infinite text backbone does not expose `norm`; "
                "semantic VAD heads cannot read the final hidden state."
            )
        return norm

    def get_input_embeddings(self) -> nn.Module:
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.language_model.set_input_embeddings(value)

    def get_output_embeddings(self) -> nn.Module:
        return self.language_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings: nn.Module) -> None:
        self.language_model.set_output_embeddings(new_embeddings)

    def set_decoder(self, decoder: nn.Module) -> None:
        self.language_model.set_decoder(decoder)

    def get_decoder(self) -> nn.Module:
        return self.language_model.get_decoder()

    def _prepare_model_inputs(
        self,
        inputs: torch.Tensor | None = None,
        bos_token_id: torch.Tensor | None = None,
        model_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, str | None, dict[str, Any]]:
        inputs, input_name, model_kwargs = super()._prepare_model_inputs(
            inputs,
            bos_token_id,
            model_kwargs,
        )
        input_features = model_kwargs.get("input_features")
        if isinstance(input_features, GeneratorType):
            input_features_generator = model_kwargs.pop("input_features")
            model_kwargs["input_features_generator"] = (
                input_features_generator
            )
            try:
                model_kwargs["input_features"] = next(
                    input_features_generator
                )
            except StopIteration:
                self._stream_exhausted = True
        return inputs, input_name, model_kwargs

    def _has_unfinished_sequences(
        self,
        this_peer_finished: bool,
        synced_gpus: bool,
        device: torch.device,
    ) -> bool:
        if getattr(self, "_stream_exhausted", False):
            self._stream_exhausted = False
            return False
        return super()._has_unfinished_sequences(
            this_peer_finished,
            synced_gpus,
            device,
        )

    def _update_model_kwargs_for_generation(
        self,
        outputs: Any,
        model_kwargs: dict[str, Any],
        is_encoder_decoder: bool = False,
        num_new_tokens: int = 1,
    ) -> dict[str, Any]:
        model_kwargs = super()._update_model_kwargs_for_generation(
            outputs,
            model_kwargs,
            is_encoder_decoder,
            num_new_tokens,
        )
        if hasattr(outputs, "encoder_past_key_values"):
            model_kwargs["encoder_past_key_values"] = (
                outputs.encoder_past_key_values
            )
        if hasattr(outputs, "padding_cache"):
            model_kwargs["padding_cache"] = outputs.padding_cache

        input_features_generator = model_kwargs.get(
            "input_features_generator"
        )
        if input_features_generator is not None:
            try:
                model_kwargs["input_features"] = next(
                    input_features_generator
                )
            except StopIteration:
                self._stream_exhausted = True
        return model_kwargs

    def _prepare_generation_config(
        self,
        generation_config: Any,
        **kwargs: Any,
    ) -> tuple[Any, dict[str, Any]]:
        generation_config, model_kwargs = (
            super()._prepare_generation_config(
                generation_config,
                **kwargs,
            )
        )
        if isinstance(
            model_kwargs.get("input_features"),
            GeneratorType,
        ):
            generation_config.max_new_tokens = None
            generation_config.max_length = int(1e9)
            generation_config._voxtral_set_max_length = True
        return generation_config, model_kwargs

    def _prepare_generated_length(
        self,
        generation_config: Any,
        has_default_max_length: bool,
        has_default_min_length: bool,
        model_input_name: str,
        input_ids_length: int,
        inputs_tensor: torch.Tensor,
    ) -> Any:
        if getattr(
            generation_config,
            "_voxtral_set_max_length",
            False,
        ):
            has_default_max_length = False
        return super()._prepare_generated_length(
            generation_config,
            has_default_max_length,
            has_default_min_length,
            model_input_name,
            input_ids_length,
            inputs_tensor,
        )

    def resolve_frame_lens(
        self,
        frame_len: int | torch.Tensor | None,
        *,
        batch_size: int,
        device: torch.device,
    ) -> torch.LongTensor:
        # frame_lens has one value per model row. Several rows may refer to the
        # same source audio while using different frame lengths (and delays).
        if frame_len is None:
            frame_len = self.config.supported_frame_lens[0]
        if torch.is_tensor(frame_len):
            frame_lens = frame_len.to(
                device=device,
                dtype=torch.long,
            ).view(-1)
            if frame_lens.numel() == 1:
                frame_lens = frame_lens.expand(batch_size)
            elif frame_lens.numel() != batch_size:
                raise ValueError(
                    "frame_len tensor must contain 1 or batch_size values, "
                    f"got {frame_lens.numel()}."
                )
        else:
            frame_lens = torch.full(
                (batch_size,),
                int(frame_len),
                device=device,
                dtype=torch.long,
            )
        supported = torch.tensor(
            self.config.supported_frame_lens,
            device=device,
            dtype=torch.long,
        )
        if not torch.isin(frame_lens, supported).all():
            raise ValueError(
                "frame_len values must be drawn from "
                f"{self.config.supported_frame_lens}."
            )
        return frame_lens

    def group_audio_hidden_states(
        self,
        audio_hidden_states: torch.Tensor,
        *,
        frame_len: int | torch.Tensor | None,
        target_token_count: int | None = None,
    ) -> torch.Tensor:
        frame_lens = self.resolve_frame_lens(
            frame_len,
            batch_size=audio_hidden_states.shape[0],
            device=audio_hidden_states.device,
        )
        max_frame_len = int(self.config.max_frame_len)
        hidden_size = int(self.config.audio_config.hidden_size)
        grouped_batches: list[tuple[torch.Tensor, torch.Tensor]] = []
        # Partition the expanded batch by frame length. A row is processed by
        # exactly one branch; supported frame lengths are not fused together.
        for row_frame_len in self.config.supported_frame_lens:
            row_indices = torch.nonzero(
                frame_lens == row_frame_len,
                as_tuple=False,
            ).flatten()
            if row_indices.numel() == 0:
                continue
            rows = audio_hidden_states.index_select(0, row_indices)
            # Complete the last temporal group before reshaping consecutive
            # audio-tower frames into one streaming-token group.
            temporal_padding = (-rows.shape[1]) % row_frame_len
            if temporal_padding:
                rows = F.pad(rows, (0, 0, 0, temporal_padding))
            grouped = rows.reshape(
                rows.shape[0],
                -1,
                row_frame_len,
                hidden_size,
            )
            # Every gear shares one projector. Pad the frame slots inside each
            # group so its flattened width is always max_frame_len * hidden_size.
            if row_frame_len < max_frame_len:
                grouped = F.pad(
                    grouped,
                    (0, 0, 0, max_frame_len - row_frame_len),
                )
            grouped_batches.append(
                (
                    row_indices,
                    grouped.reshape(
                        rows.shape[0],
                        -1,
                        max_frame_len * hidden_size,
                    ),
                )
            )

        # Text rows share one padded sequence length. Each gear therefore pads
        # or truncates its number of grouped audio tokens to that same length.
        max_token_count = (
            int(target_token_count)
            if target_token_count is not None
            else max(grouped.shape[1] for _, grouped in grouped_batches)
        )
        projector_inputs = audio_hidden_states.new_zeros(
            audio_hidden_states.shape[0],
            max_token_count,
            int(self.config.projection_size),
        )
        for row_indices, grouped in grouped_batches:
            if grouped.shape[1] < max_token_count:
                grouped = F.pad(
                    grouped,
                    (0, 0, 0, max_token_count - grouped.shape[1]),
                )
            else:
                grouped = grouped[:, :max_token_count]
            # Restore the original expanded-batch order after per-gear work.
            projector_inputs.index_copy_(0, row_indices, grouped)
        return projector_inputs

    def get_audio_tower_hidden_states(
        self,
        input_features: torch.FloatTensor | None = None,
        padding_cache: Any | None = None,
        encoder_inputs_embeds: torch.FloatTensor | None = None,
        past_key_values: Any | None = None,
        use_cache: bool | None = None,
        return_outputs: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        """Run the frozen-capable audio tower at its native frame clock."""

        if (input_features is None) == (encoder_inputs_embeds is None):
            raise ValueError("Specify exactly one of input_features or encoder_inputs_embeds.")

        audio_outputs = self.audio_tower(
            input_features=input_features,
            inputs_embeds=encoder_inputs_embeds,
            past_key_values=past_key_values,
            padding_cache=padding_cache,
            return_dict=True,
            use_cache=use_cache,
            use_padding_cache=use_cache,
            **kwargs,
        )
        if return_outputs:
            return audio_outputs.last_hidden_state, audio_outputs
        return audio_outputs.last_hidden_state

    def get_audio_projector_input_features(
        self,
        input_features: torch.FloatTensor | None = None,
        padding_cache: Any | None = None,
        encoder_inputs_embeds: torch.FloatTensor | None = None,
        past_key_values: Any | None = None,
        use_cache: bool | None = None,
        frame_len: int | torch.Tensor | None = None,
        target_token_count: int | None = None,
        return_outputs: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        """Run the audio tower and group states for the max-frame-len projector."""

        audio_hidden_states, audio_outputs = (
            self.get_audio_tower_hidden_states(
                input_features=input_features,
                encoder_inputs_embeds=encoder_inputs_embeds,
                past_key_values=past_key_values,
                padding_cache=padding_cache,
                use_cache=use_cache,
                return_outputs=True,
                **kwargs,
            )
        )
        projector_inputs = self.group_audio_hidden_states(
            audio_hidden_states,
            frame_len=frame_len,
            target_token_count=target_token_count,
        )
        if return_outputs:
            return projector_inputs, audio_outputs
        return projector_inputs

    def get_audio_features(
        self,
        input_features: torch.FloatTensor | None = None,
        padding_cache: Any | None = None,
        encoder_inputs_embeds: torch.FloatTensor | None = None,
        past_key_values: Any | None = None,
        use_cache: bool | None = None,
        frame_len: int | torch.Tensor | None = None,
        target_token_count: int | None = None,
        return_outputs: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        audio_hidden_states, audio_outputs = (
            self.get_audio_projector_input_features(
                input_features=input_features,
                encoder_inputs_embeds=encoder_inputs_embeds,
                past_key_values=past_key_values,
                padding_cache=padding_cache,
                use_cache=use_cache,
                frame_len=frame_len,
                target_token_count=target_token_count,
                return_outputs=True,
                **kwargs,
            )
        )
        audio_embeds = self.multi_modal_projector(
            audio_hidden_states
        )
        audio_outputs.pooler_output = audio_embeds
        if return_outputs:
            return audio_embeds, audio_outputs
        return audio_embeds

    def get_source_audio_embeds(
        self,
        source_input_features: torch.FloatTensor,
        audio_source_indices: torch.LongTensor,
        *,
        frame_len: int | torch.Tensor | None = None,
        target_token_count: int | None = None,
    ) -> torch.Tensor:
        """Compute the audio tower result once and fan it out to expanded rows.

        ``forward(source_input_features=...)`` already computes the audio tower
        once per call and then fans its states out to expanded rows.  This method
        exposes that operation explicitly so callers that need several
        language-model forwards can share one audio result.
        """
        audio_hidden_states = self.get_audio_tower_hidden_states(
            input_features=source_input_features,
            use_cache=False,
        )
        expanded_audio_hidden_states = audio_hidden_states.index_select(
            0,
            audio_source_indices.to(device=audio_hidden_states.device),
        )
        projector_inputs = self.group_audio_hidden_states(
            expanded_audio_hidden_states,
            frame_len=frame_len,
            target_token_count=target_token_count,
        )
        return self.multi_modal_projector(projector_inputs)

    def build_text_inputs_embeds(
        self,
        *,
        input_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        audio_embeds: torch.FloatTensor | None = None,
    ) -> torch.FloatTensor:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds.")
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)
        if audio_embeds is not None:
            audio_embeds = audio_embeds.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
            if audio_embeds.shape[:2] != inputs_embeds.shape[:2]:
                raise ValueError(
                    "Audio embedding shape must match token embedding shape before fusion: "
                    f"audio={tuple(audio_embeds.shape)} tokens={tuple(inputs_embeds.shape)}"
                )
            inputs_embeds = inputs_embeds + audio_embeds
        return inputs_embeds

    def build_t_cond(
        self,
        num_delay_tokens: int | torch.Tensor | None,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        frame_len: int | torch.Tensor | None = None,
    ) -> torch.Tensor:
        if num_delay_tokens is None:
            num_delay_tokens = self.config.default_num_delay_tokens
        if num_delay_tokens is None:
            raise ValueError("Audio8 ASR Infinite requires explicit `num_delay_tokens`.")
        if torch.is_tensor(num_delay_tokens):
            delay_values = num_delay_tokens.to(device=device, dtype=dtype).view(-1)
            if delay_values.numel() == 1:
                delay_values = delay_values.expand(batch_size)
            elif delay_values.numel() != batch_size:
                raise ValueError(
                    f"num_delay_tokens tensor must contain 1 or batch_size values, got {delay_values.numel()}."
                )
        else:
            delay_values = torch.full((batch_size,), float(num_delay_tokens), device=device, dtype=dtype)
        # The time embedding is sinusoidal.  Evaluating it row by row creates
        # three tiny GPU kernels (and Python iteration) for every LM window;
        # build the same [batch, hidden] tensor in one vectorized operation.
        inv_freq = self.time_embedding.inv_freq.to(
            device=device,
            dtype=dtype,
        )
        phase = delay_values.unsqueeze(-1) * inv_freq.unsqueeze(0)
        delay_embeddings = torch.cat(
            (phase.cos(), phase.sin()),
            dim=-1,
        )
        if self.frame_len_embedding is not None:
            frame_lens = self.resolve_frame_lens(
                frame_len,
                batch_size=batch_size,
                device=device,
            )
            frame_len_indices = torch.empty_like(frame_lens)
            for index, supported_frame_len in enumerate(
                self.config.supported_frame_lens
            ):
                frame_len_indices[
                    frame_lens == supported_frame_len
                ] = index
            delay_embeddings = delay_embeddings + self.frame_len_embedding(
                frame_len_indices
            ).to(dtype=delay_embeddings.dtype)
        return delay_embeddings.unsqueeze(1)

    def forward_language_model_with_delay(
        self,
        *,
        inputs_embeds: torch.FloatTensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Any | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        num_delay_tokens: int | torch.Tensor | None = None,
        frame_len: int | torch.Tensor | None = None,
        t_cond: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> CausalLMOutputWithPast:
        if t_cond is None:
            t_cond = self.build_t_cond(
                num_delay_tokens,
                batch_size=inputs_embeds.shape[0],
                device=inputs_embeds.device,
                dtype=inputs_embeds.dtype,
                frame_len=frame_len,
            )
        return self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            logits_to_keep=logits_to_keep,
            t_cond=t_cond,
            **kwargs,
        )

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        input_features: torch.FloatTensor | None = None,
        source_input_features: torch.FloatTensor | None = None,
        audio_source_indices: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Any | None = None,
        encoder_past_key_values: Any | None = None,
        padding_cache: Any | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        encoder_inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        num_delay_tokens: int | torch.Tensor | None = None,
        frame_len: int | torch.Tensor | None = None,
        **kwargs: Any,
    ) -> CausalLMOutputWithPast:
        audio_input_count = sum(
            value is not None
            for value in (
                input_features,
                source_input_features,
                encoder_inputs_embeds,
            )
        )
        if audio_input_count != 1:
            raise ValueError(
                "Specify exactly one of input_features, source_input_features, "
                "or encoder_inputs_embeds."
            )
        uses_source_rows = source_input_features is not None
        if not uses_source_rows:
            if audio_source_indices is not None:
                raise ValueError(
                    "audio_source_indices is only valid with source audio rows."
                )
            audio_features = input_features
        else:
            if audio_source_indices is None:
                raise ValueError(
                    "Source audio rows require audio_source_indices."
                )
            if not torch.is_tensor(audio_source_indices):
                raise TypeError("audio_source_indices must be a tensor.")
            if audio_source_indices.dtype != torch.long:
                raise TypeError("audio_source_indices must have dtype torch.long.")
            if audio_source_indices.ndim != 1:
                raise ValueError("audio_source_indices must be one-dimensional.")
            expanded_batch_size = (
                input_ids.shape[0]
                if input_ids is not None
                else inputs_embeds.shape[0]
                if inputs_embeds is not None
                else None
            )
            if expanded_batch_size is None:
                raise ValueError(
                    "source_input_features requires input_ids or inputs_embeds."
                )
            if audio_source_indices.numel() != expanded_batch_size:
                raise ValueError(
                    "audio_source_indices length must match the expanded text batch: "
                    f"indices={audio_source_indices.numel()} batch={expanded_batch_size}."
                )
            source_batch_size = source_input_features.shape[0]
            if source_batch_size <= 0:
                raise ValueError("Source audio rows must contain at least one row.")
            if (
                audio_source_indices.device.type == "cpu"
                and audio_source_indices.numel()
                and (
                    int(audio_source_indices.min().item()) < 0
                    or int(audio_source_indices.max().item()) >= source_batch_size
                )
            ):
                raise ValueError(
                    "audio_source_indices contains an out-of-range source row."
                )
            audio_features = source_input_features

        if source_input_features is not None:
            # The collator stores each source waveform once, then expands its
            # text targets across frame-length/delay configurations. Run the
            # audio tower once per source and fan its states out to those rows.
            audio_hidden_states, audio_outputs = (
                self.get_audio_tower_hidden_states(
                    input_features=source_input_features,
                    past_key_values=encoder_past_key_values,
                    padding_cache=padding_cache,
                    use_cache=use_cache,
                    return_outputs=True,
                )
            )
            expanded_audio_hidden_states = (
                audio_hidden_states.index_select(
                    0,
                    audio_source_indices.to(
                        device=audio_hidden_states.device
                    ),
                )
            )
            projector_inputs = self.group_audio_hidden_states(
                expanded_audio_hidden_states,
                frame_len=frame_len,
                target_token_count=(
                    input_ids.shape[1]
                    if input_ids is not None
                    else inputs_embeds.shape[1]
                ),
            )
            audio_embeds = self.multi_modal_projector(
                projector_inputs
            )
        else:
            audio_embeds, audio_outputs = self.get_audio_features(
                input_features=audio_features,
                encoder_inputs_embeds=encoder_inputs_embeds,
                past_key_values=encoder_past_key_values,
                padding_cache=padding_cache,
                use_cache=use_cache,
                frame_len=frame_len,
                target_token_count=(
                    input_ids.shape[1]
                    if input_ids is not None
                    else inputs_embeds.shape[1]
                ),
                return_outputs=True,
            )
        inputs_embeds = self.build_text_inputs_embeds(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            audio_embeds=audio_embeds,
        )
        # Semantic VAD heads read the text backbone norm layer's output (the
        # final hidden state).  A forward hook is used instead of
        # output_hidden_states: only the last layer is needed, so no other
        # activations are retained.
        captured_final_hidden_state: dict[str, torch.Tensor] = {}
        semantic_vad_hook: Any | None = None
        if self.semantic_vad_heads is not None:

            def _capture_final_hidden_state(
                _module: nn.Module,
                _hook_inputs: tuple[Any, ...],
                output: torch.Tensor,
            ) -> None:
                captured_final_hidden_state["final"] = output

            semantic_vad_hook = (
                self._semantic_vad_hidden_norm().register_forward_hook(
                    _capture_final_hidden_state
                )
            )
        try:
            outputs = self.forward_language_model_with_delay(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                logits_to_keep=logits_to_keep,
                num_delay_tokens=num_delay_tokens,
                frame_len=frame_len,
                **kwargs,
            )
        finally:
            if semantic_vad_hook is not None:
                semantic_vad_hook.remove()
        if self.semantic_vad_heads is not None:
            final_hidden_state = captured_final_hidden_state.get("final")
            if final_hidden_state is None:
                raise RuntimeError(
                    "Audio8 ASR Infinite did not expose the final hidden state "
                    "for the semantic VAD heads."
                )
            outputs["semantic_vad_logits"] = torch.stack(
                [head(final_hidden_state) for head in self.semantic_vad_heads],
                dim=1,
            )
        outputs.encoder_past_key_values = (
            getattr(audio_outputs, "past_key_values", None)
            if use_cache and audio_outputs is not None
            else None
        )
        outputs.padding_cache = (
            getattr(audio_outputs, "padding_cache", None)
            if use_cache and audio_outputs is not None
            else None
        )
        return outputs


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
    "QWEN_ASR_TEXT_TOKEN",
    "QWEN_AUDIO_PAD_TOKEN",
    "Qwen2RealtimeV1DecoderLayer",
    "STREAMING_PAD_TOKEN",
    "STREAMING_WORD_TOKEN",
    "ensure_voxtral_streaming_tokens",
    "resolve_qwen_language_token_id",
    "resolve_qwen_streaming_special_token_ids",
]
