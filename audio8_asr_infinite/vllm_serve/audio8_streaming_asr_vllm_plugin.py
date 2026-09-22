# SPDX-License-Identifier: Apache-2.0
"""vLLM general plugin for registering Audio8 ASR Infinite realtime."""

from __future__ import annotations


def register_audio8_streaming_asr_config() -> None:
    from transformers import AutoConfig

    # Reuse the V1 checkpoint's actual HF config.  The old shim silently
    # converted the V2 config into a different model format and also failed
    # on its intentional default_num_delay_tokens=null field.
    from audio8_asr_infinite.modeling.configuration_audio8_asr_infinite import (
        Audio8ASRInfiniteConfig,
    )

    try:
        AutoConfig.register("audio8_asr_infinite", Audio8ASRInfiniteConfig, exist_ok=True)
    except TypeError:
        AutoConfig.register("audio8_asr_infinite", Audio8ASRInfiniteConfig)


def register_audio8_streaming_asr_realtime() -> None:
    register_audio8_streaming_asr_config()

    from vllm.model_executor.models import ModelRegistry

    model_path = (
        "audio8_asr_infinite.vllm_serve.vllm0271."
        "audio8_streaming_asr_realtime:Audio8StreamingASRRealtimeGeneration"
    )
    # vLLM resolves the model class from the `architectures` field of the
    # checkpoint's config.json, so these strings must match the checkpoint.
    for architecture in (
        "Audio8ASRInfiniteForConditionalGeneration",
        "Audio8StreamingASRRealtimeGeneration",
    ):
        ModelRegistry.register_model(architecture, model_path)


def make_audio8_realtime_endpoint_plugin():
    from audio8_asr_infinite.vllm_serve.vllm0271.realtime.endpoint import make_audio8_realtime_endpoint_plugin as _make

    return _make()
