# SPDX-License-Identifier: Apache-2.0
"""Endpoint plugin that shadows /v1/realtime for the Audio8 ASR Infinite adapter.

Registered under the ``vllm.endpoint_plugins`` entry point group and enabled
via ``VLLM_PLUGINS=audio8_realtime_endpoint``.  Phase A attaches a WebSocket
route on ``/v1/realtime`` that uses the Audio8 ASR Infinite connection class (language /
delay-profile session options); because endpoint plugins attach after all
core routers, this route shadows the native one.  Phase B replaces
``state.openai_serving_realtime`` with the Audio8 ASR Infinite serving subclass so the
connection's ``transcribe_realtime`` call forwards the session options to the
model adapter's ``buffer_realtime_audio``.
"""

from __future__ import annotations

from argparse import Namespace
from typing import Any

from fastapi import APIRouter, FastAPI, WebSocket
from starlette.datastructures import State
from starlette.requests import Request
from vllm.engine.protocol import EngineClient
from vllm.logger import init_logger

from audio8_asr_infinite.vllm_serve.vllm0271.realtime.connection import RealtimeConnection
from audio8_asr_infinite.vllm_serve.vllm0271.realtime.serving import Audio8OpenAIServingRealtime

logger = init_logger(__name__)

_PLUGIN_NAME = "audio8_realtime_endpoint"


class Audio8RealtimeEndpointPlugin:
    name = _PLUGIN_NAME
    required_tasks = ("realtime",)

    def __init__(self) -> None:
        self.router = APIRouter()

        @self.router.websocket("/v1/realtime")
        async def realtime_endpoint(websocket: WebSocket) -> None:
            app = websocket.app
            serving = getattr(app.state, "openai_serving_realtime", None)
            if serving is None:
                await websocket.close(
                    code=1011, reason="realtime serving not initialized"
                )
                return
            connection = RealtimeConnection(websocket, serving)
            await connection.handle_connection()

    def attach_router(self, app: FastAPI) -> None:
        # Starlette matches the FIRST registered route for a path, so simply
        # appending a second /v1/realtime route would leave the native one
        # winning.  Remove the native realtime route(s), then register ours.
        native_paths = [
            route
            for route in list(app.routes)
            if getattr(route, "path", None) == "/v1/realtime"
        ]
        for route in native_paths:
            app.routes.remove(route)
        app.include_router(self.router)
        logger.info(
            "Audio8 ASR Infinite realtime endpoint plugin attached: replaced %s native "
            "/v1/realtime route(s)",
            len(native_paths),
        )

    async def init_state(
        self,
        engine_client: EngineClient | None,
        state: State,
        args: Namespace,
    ) -> None:
        native_serving = getattr(state, "openai_serving_realtime", None)
        if native_serving is None:
            logger.warning(
                "Audio8 ASR Infinite realtime endpoint plugin: no native realtime serving "
                "found on app state; skipping swap."
            )
            return
        if isinstance(native_serving, Audio8OpenAIServingRealtime):
            return
        # Swap the serving instance for the Audio8 ASR Infinite subclass.  The subclass only
        # overrides transcribe_realtime (session-option passthrough), so
        # rebind the native instance's already-initialized attributes instead
        # of reconstructing its engine/model plumbing.
        swapped = object.__new__(Audio8OpenAIServingRealtime)
        swapped.__dict__.update(native_serving.__dict__)
        state.openai_serving_realtime = swapped
        logger.info(
            "Audio8 ASR Infinite realtime serving installed (engine-backed): task=realtime"
        )


def make_audio8_realtime_endpoint_plugin(**kwargs: Any) -> Audio8RealtimeEndpointPlugin:
    return Audio8RealtimeEndpointPlugin()
