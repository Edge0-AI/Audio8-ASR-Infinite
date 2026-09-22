#!/usr/bin/env python3
"""Run real audio through the Audio8 ASR Infinite torch simulated-streaming decoder.

This is the canonical torch-side inference entrypoint: it loads a merged
``weight_format_version=2`` checkpoint through the HF-style modeling class and
decodes real waveform(s) with the definition-aligned simulated-streaming greedy
decoder (per-80ms text-token clock, streaming audio KV cache).

Example:
    python -m audio8_asr_infinite.examples.torch_streaming_decode \
        --checkpoint /path/to/merged-v2-checkpoint \
        --audio sample.wav --language zh --transcription-delay-ms 480
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from transformers import AutoFeatureExtractor, AutoTokenizer

from audio8_asr_infinite.modeling.modeling_audio8_asr_infinite import (
    Audio8ASRInfiniteForConditionalGeneration,
    resolve_qwen_language_token_id,
    resolve_qwen_streaming_special_token_ids,
)
from audio8_asr_infinite.streaming_inference import (
    simulated_streaming_greedy_decode_batch,
)


@dataclass(frozen=True)
class StreamingDecodeConfig:
    """Duck-typed audio config consumed by the simulated-streaming decoder."""

    transcription_delay_ms: int = 480
    token_period_ms: int = 80
    sampling_rate: int = 16000
    left_pad_tokens: int = 18
    right_pad_text_tokens: int = 10
    max_new_tokens: int = 512

    @property
    def num_delay_tokens(self) -> int:
        return self.transcription_delay_ms // self.token_period_ms

    @property
    def raw_audio_samples_per_token(self) -> int:
        return self.sampling_rate * self.token_period_ms // 1000

    @property
    def streaming_n_left_pad_tokens(self) -> int:
        return self.left_pad_tokens


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Audio8 ASR Infinite torch streaming decode")
    parser.add_argument("--checkpoint", required=True, help="Merged weight_format_version=2 checkpoint directory")
    parser.add_argument("--audio", required=True, help="Real audio file (wav/mp3/flac)")
    parser.add_argument("--language", default="zh", choices=["zh", "en"])
    parser.add_argument("--transcription-delay-ms", type=int, default=480)
    parser.add_argument("--left-pad-tokens", type=int, default=None, help="Override left prompt tokens (default: checkpoint value or 18)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--output", default=None, help="Optional JSON output path")
    return parser.parse_args()


def resolve_dtype(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def load_waveform(path: str, *, sampling_rate: int) -> np.ndarray:
    import librosa
    import soundfile as sf

    waveform, source_rate = sf.read(path, dtype="float32", always_2d=False)
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
    if int(source_rate) != sampling_rate:
        waveform = librosa.resample(waveform, orig_sr=int(source_rate), target_sr=sampling_rate)
    return np.clip(np.asarray(waveform, dtype=np.float32).reshape(-1), -1.0, 1.0)


def main() -> None:
    args = parse_args()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    device = torch.device(args.device)
    dtype = resolve_dtype(args.dtype)

    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint), trust_remote_code=True)
    feature_extractor = AutoFeatureExtractor.from_pretrained(str(checkpoint), trust_remote_code=True)
    model = Audio8ASRInfiniteForConditionalGeneration.from_pretrained(
        str(checkpoint),
        trust_remote_code=True,
        torch_dtype=dtype,
    )
    model.to(device=device)
    model.eval()
    model.config.use_cache = True
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()

    left_pad_tokens = args.left_pad_tokens
    if left_pad_tokens is None:
        left_pad_tokens = int(getattr(model.config, "streaming_n_left_pad_tokens", 0) or 18)
    config = StreamingDecodeConfig(
        transcription_delay_ms=args.transcription_delay_ms,
        left_pad_tokens=left_pad_tokens,
    )

    waveform = load_waveform(args.audio, sampling_rate=config.sampling_rate)
    special_ids = resolve_qwen_streaming_special_token_ids(tokenizer)
    language_token_id = resolve_qwen_language_token_id(tokenizer, args.language)

    results = simulated_streaming_greedy_decode_batch(
        model=model,
        tokenizer=tokenizer,
        feature_extractor=feature_extractor,
        waveforms=[waveform],
        language_token_ids=[language_token_id],
        special_ids=special_ids,
        audio_config=config,
        num_delay_tokens=[config.num_delay_tokens],
        right_pad_text_tokens=config.right_pad_text_tokens,
        dtype=dtype,
        device=next(model.parameters()).device,
        max_new_tokens=config.max_new_tokens,
    )
    result = results[0]
    summary = {
        "audio": str(Path(args.audio).resolve()),
        "language": args.language,
        "transcription_delay_ms": config.transcription_delay_ms,
        "left_pad_tokens": config.left_pad_tokens,
        "final_text": result.get("final_text", ""),
        "duration_seconds": round(waveform.size / config.sampling_rate, 3),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.output:
        output = Path(args.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
