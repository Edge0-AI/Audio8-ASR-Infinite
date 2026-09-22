#!/usr/bin/env python3
"""HuggingFace Spaces demo for Audio8 ASR Infinite.

Runs the canonical torch simulated-streaming decoder on real audio.  Two modes:

- 整段解码：对整段音频做一次完整的模拟流式解码（定义对齐，反映部署行为）。
- 增量解码：每 1 秒对"累积音频"重新跑一次真实解码，逐段返回更新的转写结果，
  用来直观展示流式转写的渐进输出（每一步都是真实模型前向，不是伪造的分段）。

Needs a GPU Space (24GB 级即可)。
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

import gradio as gr
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

MODEL_ID = os.environ.get("AUDIO8_MODEL_ID", "REPLACE_WITH_HF_MODEL_REPO_ID")
SAMPLING_RATE = 16000
TOKEN_PERIOD_MS = 80
STREAM_STEP_SECONDS = 1.0
DEFAULT_DELAY_MS = 480
MAX_NEW_TOKENS = 512
RIGHT_PAD_TEXT_TOKENS = 10


@dataclass(frozen=True)
class AudioConfig:
    """Duck-typed audio config consumed by the simulated-streaming decoder."""

    transcription_delay_ms: int
    sampling_rate: int = SAMPLING_RATE
    token_period_ms: int = TOKEN_PERIOD_MS
    left_pad_tokens: int = 18
    right_pad_text_tokens: int = RIGHT_PAD_TEXT_TOKENS

    @property
    def num_delay_tokens(self) -> int:
        return self.transcription_delay_ms // self.token_period_ms

    @property
    def raw_audio_samples_per_token(self) -> int:
        return self.sampling_rate * self.token_period_ms // 1000

    @property
    def streaming_n_left_pad_tokens(self) -> int:
        return self.left_pad_tokens


class Runtime:
    """Lazily loaded model / tokenizer / feature extractor."""

    def __init__(self) -> None:
        self.model = None
        self.tokenizer = None
        self.feature_extractor = None
        self.special_ids: dict[str, int] = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.bfloat16

    def load(self) -> None:
        if self.model is not None:
            return
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID, trust_remote_code=True
        )
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            MODEL_ID, trust_remote_code=True
        )
        self.model = Audio8ASRInfiniteForConditionalGeneration.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,
            torch_dtype=self.dtype,
        )
        self.model.to(device=self.device)
        self.model.eval()
        self.model.config.use_cache = True
        if hasattr(self.model, "gradient_checkpointing_disable"):
            self.model.gradient_checkpointing_disable()
        self.special_ids = resolve_qwen_streaming_special_token_ids(self.tokenizer)

    def left_pad_tokens(self) -> int:
        value = getattr(self.model.config, "streaming_n_left_pad_tokens", None)
        return int(value or 18)


RUNTIME = Runtime()


def load_waveform(path: str) -> np.ndarray:
    import librosa
    import soundfile as sf

    waveform, source_rate = sf.read(path, dtype="float32", always_2d=False)
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
    if int(source_rate) != SAMPLING_RATE:
        waveform = librosa.resample(
            waveform, orig_sr=int(source_rate), target_sr=SAMPLING_RATE
        )
    return np.clip(np.asarray(waveform, dtype=np.float32).reshape(-1), -1.0, 1.0)


def decode(
    waveform: np.ndarray,
    *,
    language: str,
    delay_ms: int,
    left_pad_tokens: int,
) -> tuple[dict, float]:
    """Run one real simulated-streaming decode and return (result, seconds)."""
    config = AudioConfig(
        transcription_delay_ms=int(delay_ms),
        left_pad_tokens=int(left_pad_tokens),
    )
    start = time.time()
    results = simulated_streaming_greedy_decode_batch(
        model=RUNTIME.model,
        tokenizer=RUNTIME.tokenizer,
        feature_extractor=RUNTIME.feature_extractor,
        waveforms=[waveform],
        language_token_ids=[
            resolve_qwen_language_token_id(RUNTIME.tokenizer, language)
        ],
        special_ids=RUNTIME.special_ids,
        audio_config=config,
        num_delay_tokens=[config.num_delay_tokens],
        right_pad_text_tokens=RIGHT_PAD_TEXT_TOKENS,
        dtype=RUNTIME.dtype,
        device=next(RUNTIME.model.parameters()).device,
        max_new_tokens=MAX_NEW_TOKENS,
    )
    return results[0], time.time() - start


def transcribe(
    audio_path: str | None,
    language: str,
    delay_ms: int,
    incremental: bool,
):
    if not audio_path:
        yield "请先录制或上传一段音频。", ""
        return

    yield "正在加载模型…", ""
    RUNTIME.load()
    left_pad_tokens = RUNTIME.left_pad_tokens()

    waveform = load_waveform(audio_path)
    if waveform.size == 0:
        yield "音频为空。", ""
        return
    duration = waveform.size / SAMPLING_RATE
    device_note = f"{RUNTIME.device.type} / {str(RUNTIME.dtype).replace('torch.', '')}"

    if not incremental:
        yield "解码中…", ""
        result, elapsed = decode(
            waveform,
            language=language,
            delay_ms=delay_ms,
            left_pad_tokens=left_pad_tokens,
        )
        status = (
            f"整段解码完成｜音频 {duration:.2f}s｜耗时 {elapsed:.2f}s｜"
            f"实时率 {elapsed / duration:.2f}×｜延迟 {delay_ms}ms｜{device_note}"
        )
        yield result.get("final_text", ""), status
        return

    step_samples = max(int(SAMPLING_RATE * STREAM_STEP_SECONDS), 1)
    boundaries = list(range(step_samples, waveform.size, step_samples)) + [
        waveform.size
    ]
    text = ""
    stream_start = time.time()
    for end in boundaries:
        decoded_seconds = end / SAMPLING_RATE
        result, elapsed = decode(
            waveform[:end],
            language=language,
            delay_ms=delay_ms,
            left_pad_tokens=left_pad_tokens,
        )
        text = result.get("final_text", "")
        wall = time.time() - stream_start
        status = (
            f"增量解码中｜已处理 {decoded_seconds:.1f}s / {duration:.1f}s｜"
            f"本步 {elapsed:.2f}s｜累计 {wall:.2f}s｜延迟 {delay_ms}ms｜{device_note}"
        )
        yield text, status
    status = (
        f"增量解码完成｜音频 {duration:.2f}s｜累计 {time.time() - stream_start:.2f}s｜"
        f"延迟 {delay_ms}ms｜{device_note}"
    )
    yield text, status


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="Audio8 ASR Infinite") as demo:
        gr.Markdown(
            """
            # Audio8 ASR Infinite 流式语音识别
            低延迟流式自动语音识别：80ms 音频时钟、可配置转写延迟（`target_delay_ms`）、
            中英双语。推理走定义对齐的**模拟流式解码**（逐窗音频 + 流式 KV cache），
            与真实部署行为一致。
            """
        )
        with gr.Row():
            with gr.Column():
                audio = gr.Audio(
                    sources=["microphone", "upload"],
                    type="filepath",
                    label="音频（麦克风录制或上传文件）",
                )
                language = gr.Radio(
                    choices=[("中文", "zh"), ("English", "en")],
                    value="zh",
                    label="语言",
                )
                delay = gr.Slider(
                    minimum=80,
                    maximum=2000,
                    step=80,
                    value=DEFAULT_DELAY_MS,
                    label="转写延迟 target_delay_ms（80ms 的整数倍）",
                )
                incremental = gr.Checkbox(
                    value=True,
                    label="增量解码（每 1 秒对累积音频重新真实解码，展示渐进输出）",
                )
                run = gr.Button("开始转写", variant="primary")
            with gr.Column():
                transcript = gr.Textbox(
                    label="转写结果",
                    lines=8,
                )
                status = gr.Textbox(label="运行状态", lines=2)
        run.click(
            transcribe,
            inputs=[audio, language, delay, incremental],
            outputs=[transcript, status],
        )
        gr.Markdown(
            """
            说明：增量模式每一步都会对"累积到当前的音频"重新跑一次完整解码，
            因此输出是真实的模型结果，但单步耗时随音频变长而增加。
            追求最低延迟请使用仓库里的 vLLM `/v1/realtime` 实时服务（docker compose
            一键部署，见 `docker/docker-compose.yml`）。
            """
        )
    return demo


if __name__ == "__main__":
    build_demo().launch()
