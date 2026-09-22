# SPDX-License-Identifier: Apache-2.0
"""Realtime endpoint overlay for Audio8 ASR Infinite on pristine vLLM 0.27.1.

Loaded at server startup through the ``vllm.endpoint_plugins`` entry point
(``VLLM_PLUGINS=audio8_realtime_endpoint``): our WebSocket route shadows the
native ``/v1/realtime`` so language / delay-profile session options reach the
Audio8 ASR Infinite model adapter, while the installed vLLM wheel stays untouched.
"""

from audio8_asr_infinite.vllm_serve.vllm0271.realtime.protocol import (  # noqa: F401
    SessionUpdate as Audio8SessionUpdate,
)
