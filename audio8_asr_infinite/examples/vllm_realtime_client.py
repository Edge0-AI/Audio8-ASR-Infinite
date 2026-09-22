#!/usr/bin/env python3
"""Run real audio through Audio8 ASR Infinite vLLM /v1/realtime WebSocket.

Requires a running Audio8 ASR Infinite vLLM service (see
``docker/docker-compose.yml`` for the canonical deployment).  The
session is configured through ``session.update`` with model + language +
target delay, audio is streamed as 16kHz PCM16 chunks, and the transcription
deltas / final text are collected from the realtime event stream.

Example:
    python -m audio8_asr_infinite.examples.vllm_realtime_client \
        --audio sample.wav --ws-url ws://127.0.0.1:18190/v1/realtime \
        --language zh --target-delay-ms 480 --pace
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

try:
    import websockets
except ImportError as exc:  # pragma: no cover
    raise SystemExit("websockets is required: pip install websockets") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Audio8 ASR Infinite vLLM realtime client")
    parser.add_argument("--audio", required=True, help="Real audio file (wav/mp3/flac)")
    parser.add_argument("--ws-url", default="ws://127.0.0.1:18190/v1/realtime")
    parser.add_argument("--model", default="audio8-asr-infinite")
    parser.add_argument("--language", default="zh", choices=["zh", "en"])
    parser.add_argument("--target-delay-ms", type=int, default=480)
    parser.add_argument("--chunk-ms", type=int, default=100)
    parser.add_argument("--pace", action="store_true", help="Send chunks in real time")
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument("--output", default="vllm_realtime_result.json")
    return parser.parse_args()


def load_pcm16(path: str) -> np.ndarray:
    import librosa
    import soundfile as sf

    waveform, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
    if int(sample_rate) != 16000:
        waveform = librosa.resample(waveform, orig_sr=int(sample_rate), target_sr=16000)
    waveform = np.clip(np.asarray(waveform, dtype=np.float32).reshape(-1), -1.0, 1.0)
    return (waveform * 32767.0).astype(np.int16)


async def run(args: argparse.Namespace) -> dict:
    pcm16 = load_pcm16(args.audio)
    chunk_samples = 16000 * args.chunk_ms // 1000

    events: list[dict] = []
    deltas: list[str] = []
    error_event = None
    received_done = False
    start = time.time()

    connect_kwargs = {"max_size": None, "ping_interval": None}
    if args.ws_url.startswith("wss://"):
        import ssl

        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        connect_kwargs["ssl"] = ssl_context
    async with websockets.connect(args.ws_url, **connect_kwargs) as ws:
        async def receive():
            nonlocal error_event, received_done
            while True:
                remaining = args.timeout_seconds - (time.time() - start)
                if remaining <= 0:
                    return
                msg = await asyncio.wait_for(ws.recv(), timeout=remaining)
                ev = json.loads(msg)
                ev["_received_at"] = time.time() - start
                events.append(ev)
                event_type = ev.get("type")
                if event_type == "transcription.delta":
                    deltas.append(ev.get("delta") or "")
                elif event_type == "transcription.done":
                    received_done = True
                    return
                elif event_type == "error":
                    error_event = ev
                    return

        receiver = asyncio.create_task(receive())

        # session.update with model + language + delay options.
        update = {
            "type": "session.update",
            "model": args.model,
            "language": args.language,
            "target_delay_ms": args.target_delay_ms,
        }
        await ws.send(json.dumps(update))
        await ws.send(json.dumps({"type": "input_audio_buffer.commit", "final": False}))

        send_start = time.time()
        for start_idx in range(0, len(pcm16), chunk_samples):
            chunk = pcm16[start_idx : start_idx + chunk_samples]
            await ws.send(
                json.dumps(
                    {
                        "type": "input_audio_buffer.append",
                        "audio": base64.b64encode(chunk.tobytes()).decode("ascii"),
                    }
                )
            )
            if args.pace:
                await asyncio.sleep(len(chunk) / 16000.0)
        await ws.send(json.dumps({"type": "input_audio_buffer.commit", "final": True}))
        send_end = time.time()
        await receiver

    done_text = ""
    for ev in reversed(events):
        if ev.get("type") == "transcription.done":
            done_text = ev.get("text") or ""
            break
    final_text = done_text or "".join(deltas)
    result = {
        "audio": str(Path(args.audio).resolve()),
        "ws_url": args.ws_url,
        "model": args.model,
        "language": args.language,
        "target_delay_ms": args.target_delay_ms,
        "chunk_ms": args.chunk_ms,
        "duration_seconds": round(len(pcm16) / 16000.0, 3),
        "send_duration_seconds": round(send_end - send_start, 3),
        "wall_seconds": round(time.time() - start, 3),
        "received_done": received_done,
        "error_event": error_event,
        "event_counts": dict(Counter(ev.get("type") for ev in events)),
        "delta_count": len(deltas),
        "delta_text": "".join(deltas),
        "done_text": done_text,
        "final_text": final_text,
        "events": events,
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "final_text": final_text,
                "received_done": received_done,
                "error_event": error_event,
                "event_counts": result["event_counts"],
                "delta_count": len(deltas),
                "result_path": str(output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return result


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
