# SPDX-License-Identifier: Apache-2.0
"""Cross-process side channel for semantic VAD predictions.

The model adapter scores the semantic VAD heads on the same hidden states it
feeds the language head, but a vLLM ``RequestOutput`` has no field for extra
per-step model outputs, and this stack deliberately does not patch vLLM.  With
``vllm serve`` in vLLM 0.27.1 the EngineCore always runs in its own process
(``EngineCoreClient.make_client`` refuses asyncio mode without multiprocessing),
so the handoff has to cross a process boundary: the model writes each
prediction to a small JSON file keyed by the session salt, and the realtime
connection reads it back whenever it forwards a generated token.

The directory defaults to ``/tmp/audio8_vad`` (override with
``AUDIO8_VAD_CHANNEL_DIR``).  It must be writable and shared by the EngineCore
process and the API server process — inside one container that is simply
``/tmp``; the repository mount is read-only, so nothing is written there.

Writes are atomic (temp file + ``os.replace``) and each file keeps only the
most recent predictions, so a reader that falls behind still receives the
pending predictions in order.  Only the model process writes and only the
connection reads, so they never have to coordinate through a cross-process lock.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Any


_LOCK = threading.Lock()
_MAX_PENDING_PER_SESSION = 16
_DEFAULT_DIRECTORY = "/tmp/audio8_vad"


def session_salt(cache_salt: str | None) -> int:
    """Digest a per-generation ``cache_salt`` into the int64 session key.

    This is the single definition of the digest.  The model adapter uses it to
    route per-session rolling KV/t_cond state and the realtime connection uses
    it to read this channel back, so both sides always agree; an empty salt maps
    to 0, the historical single-session slot.
    """

    if not cache_salt:
        return 0
    digest = hashlib.sha256(str(cache_salt).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=True)


def channel_dir() -> Path:
    """Directory shared by the model process and the realtime endpoint."""

    directory = Path(os.environ.get("AUDIO8_VAD_CHANNEL_DIR") or _DEFAULT_DIRECTORY)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _session_path(salt: int) -> Path:
    return channel_dir() / f"{int(salt)}.json"


def _read(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as handle:
            entry = json.load(handle)
    except (FileNotFoundError, ValueError, OSError):
        return None
    return entry if isinstance(entry, dict) else None


def publish(salt: int, payload: dict[str, Any]) -> None:
    """Append one semantic VAD prediction for ``salt``."""

    path = _session_path(salt)
    with _LOCK:
        entry = _read(path) or {"counter": 0, "items": []}
        entry["counter"] = int(entry.get("counter", 0)) + 1
        items = list(entry.get("items") or [])
        items.append([entry["counter"], payload])
        entry["items"] = items[-_MAX_PENDING_PER_SESSION:]
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(entry, handle)
        os.replace(temporary, path)


def drain(salt: int, after: int) -> tuple[int, list[dict[str, Any]]]:
    """Return ``(last_counter, payloads)`` published for ``salt`` after ``after``.

    ``after`` is the counter value the caller last consumed (0 initially).
    """

    entry = _read(_session_path(salt))
    if not entry:
        return after, []
    items = [
        item
        for item in (entry.get("items") or [])
        if isinstance(item, (list, tuple)) and len(item) == 2 and int(item[0]) > after
    ]
    if not items:
        return after, []
    return int(items[-1][0]), [item[1] for item in items]


def reset(salt: int) -> None:
    """Drop all state for ``salt`` (used when a session ends)."""

    try:
        _session_path(salt).unlink()
    except OSError:
        pass


__all__ = ["channel_dir", "drain", "publish", "reset", "session_salt"]
