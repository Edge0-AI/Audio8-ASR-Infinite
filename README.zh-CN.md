<div align="center">

# Audio8 ASR Infinite

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Audio8--ASR--Infinite-yellow?style=for-the-badge)](https://huggingface.co/Edge0/Audio8-ASR-Infinite)
[![GitHub](https://img.shields.io/badge/GitHub-Audio8--ASR--Infinite-black?style=for-the-badge&logo=github)](https://github.com/Edge0-AI/Audio8-ASR-Infinite)
[![arXiv](https://img.shields.io/badge/arXiv-coming%20soon-b31b1b?style=for-the-badge&logo=arxiv&logoColor=white)](https://github.com/Edge0-AI/Audio8-ASR-Infinite)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue?style=for-the-badge)](LICENSE)

[English](README.md) | 中文

</div>

**Audio8 ASR Infinite** 是一个原生流式语音识别模型，以尽可能低的响应延迟为目标。支持
80/120/160 ms 三档音频时钟，转写延迟可在 240–560 ms 之间配置；配合改造后的 vLLM，
可 **7×24** 小时持续转写不限时长的音频，长时间运行也不会漂移。

## 亮点

- **响应快**：原生流式架构，每秒解码 12.5 次。
- **无限时长监听转录**：滚动 KV cache 让显存与延迟保持恒定，**7×24 运行**也不会劣化。
- **时钟可选**：每步输出一个 token，80 / 120 / 160 ms 分别对应每秒 12.5 / 8.3 / 6.25 次判断，兼顾感知粒度与算力开销。
- **延迟可调**：可按需设置用多少延迟换取准确率。
- **语义 VAD**：判断话是否说完靠语义而非声音，思考停顿、磕巴不会被误判成结束。
- **中英双语**

## 路线图

本次发布为**预览版**，交付转写能力。实时语义感知正在同一套帧网格和同一次声学前向上构建。

| 阶段 | 状态 | 范围 |
| --- | --- | --- |
| **预览版 · ASR 基座** | ✅ 已完成 | 中英流式转写：80/120/160 ms 时钟可选、`target_delay_ms` 可配、不限时长的滚动 KV window |
| **正式版** | 🏃 进行中 | 同一网格上的帧级语义感知，转写之外的能力 |

## 架构

音频侧沿用 Voxtral realtime，流式方案采用 DSM。

| 组件 | 初始权重 | 是否训练 |
| --- | --- | --- |
| Causal Audio Tower | Voxtral Realtime 4B | ✅ |
| Audio Projector | 随机初始化 | ✅ |
| Frame Length Embedding | 随机初始化 | ✅ |
| Decoder | Qwen2.5-3B-Instruct | ✅ |
| LM Head | Qwen2.5-3B-Instruct | ✅ |

## 延迟和帧长组合

下表中的帧长与延迟组合经过后训练，其他组合也能使用，但效果可能不是最优。

| 音频时钟 | `frame_len` | `streaming_n_left_pad_tokens` | 可选 `target_delay_ms` |
| --- | --- | --- | --- |
| 80 ms | 4 | 18 | 240 / 320 / 480 / 560 |
| 120 ms | 6 | 12 | 240 / 480 |
| 160 ms | 8 | 9 | 320 / 480 |

`target_delay_ms` 必须是所选时钟的整数倍，因此即使未列出，每个时钟下仍可使用更长的延迟。

## 评测

### 480 ms 延迟，80 ms 帧长

| 测试集 | 指标 | Audio8 ASR Infinite | Voxtral-Mini-4B-Realtime-2602 | nemotron-3.5-asr-streaming-0.6b |
| --- | --- | --- | --- | --- |
| aishell1/test | CER | **1.750** | 16.795 | 12.927@560ms |
| aishell4/test | CER | **2.893** | 16.456 | 14.677@560ms |
| librispeech test.clean | WER | 3.042 | **2.210** | 3.353@560ms |
| librispeech test.other | WER | 6.808 | **5.552** | 7.140@560ms |
| **平均** | | **3.623** | 10.253（2 库） | 9.524 |

## 使用 vLLM 做 7×24 推理

部署走 docker compose，同时提供 web demo：

```bash
cd docker
AUDIO8_MODEL_DIR=/path/to/checkpoint docker compose up -d
```

用自带的网页客户端验证：

```
http://localhost:8080/     # 明文 HTTP
https://localhost:8443/    # TLS 代理，接受自签证书即可
```

不开网页，也可以在终端直接连接：

```bash
python -m audio8_asr_infinite.examples.vllm_realtime_client \
    --ws-url ws://127.0.0.1:18191/v1/realtime \
    --audio sample.wav --language zh --target-delay-ms 480 --pace
```

## torch 推理（模拟流式解码）

```bash
python -m audio8_asr_infinite.examples.torch_streaming_decode \
    --checkpoint /path/to/checkpoint \
    --audio sample.wav --language zh --transcription-delay-ms 480
```
