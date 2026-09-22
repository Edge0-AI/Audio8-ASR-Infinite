---
title: Audio8 ASR Infinite
emoji: 🎙️
colorFrom: indigo
colorTo: blue
sdk: gradio
app_file: app.py
pinned: false
license: apache-2.0
---

# Audio8 ASR Infinite — 流式语音识别 Demo

HuggingFace Space 版 demo，加载 Audio8 ASR Infinite 的 merged 权重，用仓库里的
**模拟流式解码**（`audio8_asr_infinite.streaming_inference`）对真实音频做推理。

## 硬件要求

需要 **GPU Space**（24GB 显存级，如 Nvidia L4 / A10G / L40S）。模型 bf16 权重约 8GB。

## 配置

- `AUDIO8_MODEL_ID`（Space 变量，可选）：模型仓库 id 或本地路径。缺省用 `app.py` 里的
  `MODEL_ID` 默认值（部署前需替换成实际的 HuggingFace 模型仓库 id）。

## 两种模式

- **整段解码**：对整段音频做一次完整模拟流式解码，输出终稿转写。
- **增量解码**（默认开）：每 1 秒对"累积到当前的音频"重新跑一次真实解码，逐段返回
  更新的转写结果。每一步都是真实模型前向（不是伪造分段），因此单步耗时随音频变长
  而增加；这只用于直观展示渐进输出。

追求最低延迟的真实流式服务请走 vLLM `/v1/realtime`：docker compose 一键部署，
见 `docker/docker-compose.yml`。

## 已验证环境

gradio `6.27.0` + transformers `5.13.0` + torch `2.9.1`（CPU 上真实音频端到端跑通：
整段解码与增量解码终稿一致）。gradio 5 / 6 均可用（demo 未使用版本特有参数）。

## 本地运行

```bash
pip install -r requirements.txt
AUDIO8_MODEL_ID=/path/to/merged-checkpoint python app.py
```
