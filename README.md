<div align="center">

# Audio8 ASR Infinite


[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Audio8--ASR--Infinite-yellow?style=for-the-badge)](https://huggingface.co/Edge0/Audio8-ASR-Infinite)
[![GitHub](https://img.shields.io/badge/GitHub-Audio8--ASR--Infinite-black?style=for-the-badge&logo=github)](https://github.com/Edge0-AI/Audio8-ASR-Infinite)
[![arXiv](https://img.shields.io/badge/arXiv-coming%20soon-b31b1b?style=for-the-badge&logo=arxiv&logoColor=white)](https://github.com/Edge0-AI/Audio8-ASR-Infinite)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue?style=for-the-badge)](LICENSE)

English | [中文](README.zh-CN.md)

</div>

**Audio8 ASR Infinite** is a native streaming speech recognition model built to be
as responsive as possible. It offers a selectable audio clock (80/120/160 ms) and
a transcription delay (240–560 ms). 
With our adapted vLLM build it transcribes unlimited-length audio **24/7** without drifting.

## Highlights

- **Super responsive** — the native streaming architecture decodes 12.5 times per second.
- **Unlimited-length transcription** — a rolling KV cache keeps memory and
  latency bounded, even in **24/7 operation**.
- **Selectable streaming clock** — one text token per clock step
  (12.5 / 8.3 / 6.25 decisions per second), balancing perception granularity and resource cost.
- **Configurable transcription delay** — set how much delay to trade for accuracy.
- **Semantic VAD** — distinguishes thinking pauses, stuttering and real end of turn, where traditional acoustic VAD fails.
- **Bilingual** — Chinese and English.

## Roadmap

This is the **preview release**: it delivers the transcription base. Realtime
semantic perception is being built on the same frame grid and the same acoustic
forward pass.


| Stage | Status | Scope |
| --- | --- | --- |
| **Preview — ASR base** | ✅ done | Streaming Chinese/English transcription: selectable 80/120/160 ms clock, configurable `target_delay_ms`, unlimited-length rolling KV window |
| **Formal release** | 🏃in progress | Frame-level semantic perception on the same grid, beyond transcription |

## Architecture

Inherits the Voxtral realtime audio architecture and DSM-style streaming.


| Component | Initial weights | Trained |
| --- | --- | --- |
| Causal Audio Tower | Voxtral Realtime 4B | ✅ |
| Audio Projector  | random initialisation | ✅ |
| Frame Length Embedding | random initialisation | ✅ |
| Decoder | Qwen2.5-3B-Instruct | ✅ |
| LM Head | Qwen2.5-3B-Instruct | ✅ |


## Optimized operation points

The following combinations of frame length and delay are post-trained. Other combinations can be used but performance may not be optimum.

| audio clock | `frame_len` | `streaming_n_left_pad_tokens` | selectable `target_delay_ms` |
| --- | --- | --- | --- |
| 80 ms | 4 | 18 | 240 / 320 / 480 / 560 |
| 120 ms | 6 | 12 | 240 / 480 |
| 160 ms | 8 | 9 | 320 / 480 |

`target_delay_ms` must be an integer multiple of the selected clock, so longer
delays stay available at every clock even when they are not listed above.



## Evaluation


### 480 ms Delay, 80ms frame length

| test set | metric | Audio8 ASR Infinite | Voxtral-Mini-4B-Realtime-2602 | nemotron-3.5-asr-streaming-0.6b |
| --- | --- | --- | --- | --- |
| aishell1/test | CER | **1.750** | 16.795 | 12.927@560ms |
| aishell4/test | CER | **2.893** | 16.456 | 14.677@560ms |
| librispeech test.clean | WER | 3.042 | **2.210** | 3.353@560ms |
| librispeech test.other | WER | 6.808 | **5.552** | 7.140@560ms |
| **average** | | **3.623** | 10.253 (2 sets) | 9.524 |


## 24/7 inference with vLLM

Docker compose is the canonical deployment path; it also serves the web demo:

```bash
cd docker
AUDIO8_MODEL_DIR=/path/to/checkpoint docker compose up -d
```

Verify with the web client shipped in the same stack:

```
http://localhost:8080/     # plain HTTP
https://localhost:8443/    # TLS proxy; accept the self-signed certificate
```

The same socket can be driven from a terminal:

```bash
python -m audio8_asr_infinite.examples.vllm_realtime_client \
    --ws-url ws://127.0.0.1:18191/v1/realtime \
    --audio sample.wav --language zh --target-delay-ms 480 --pace
```

## Torch inference (simulated streaming decode)

```bash
python -m audio8_asr_infinite.examples.torch_streaming_decode \
    --checkpoint /path/to/checkpoint \
    --audio sample.wav --language zh --transcription-delay-ms 480
```
