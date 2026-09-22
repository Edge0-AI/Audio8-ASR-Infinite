#!/usr/bin/env python3.12
"""Audio8 ASR Infinite web HTTPS 入口。

单端口（默认 8443）同时提供：
- 静态文件（/opt/audio8_client 复制来的 /srv/audio8_client，含改写 publicVllmPort 后的 config.js）
- /v1/realtime 的 WebSocket 反代（wss -> 后端 vLLM 的 ws，浏览器只需接受一次自签证书）

后端地址由 AUDIO8_BACKEND_WS 控制（默认 compose 内 asr 服务 18190）。
"""

from __future__ import annotations

import asyncio
import logging
import os
import ssl

from aiohttp import ClientSession, WSMsgType, web

ROOT = "/srv/audio8_client"
PUBLIC_PORT = os.environ.get("AUDIO8_PUBLIC_PORT", "8443")
BACKEND_WS = os.environ.get("AUDIO8_BACKEND_WS", "ws://audio8-asr-infinite:18190/v1/realtime")
# Optional second vLLM backend, selected with ``?instance=2`` on the same TLS
# port, so one HTTPS entry point can front several instances and the page can
# offer an instance picker.  Empty = no second instance.
BACKEND_WS_2 = os.environ.get("AUDIO8_BACKEND_WS_2", "").strip()
LISTEN_PORT = int(os.environ.get("AUDIO8_LISTEN_PORT", "8443"))

logger = logging.getLogger("audio8.web_tls_proxy")
# The proxy runs as a plain script (no vLLM logging config), where the root
# logger defaults to WARNING and would swallow the per-connection routing line
# ("realtime proxy <peer> -> <backend>").  Give our own logger a handler at INFO
# so the backend choice is visible in `docker logs web-https`.
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
logger.setLevel(logging.INFO)


async def realtime_proxy(request: web.Request) -> web.WebSocketResponse:
    client = web.WebSocketResponse()
    await client.prepare(request)
    session = request.app["client_session"]
    try:
        target = backend_for(request)
        logger.info("realtime proxy %s -> %s", request.remote or "?", target)
        backend = await session.ws_connect(target, max_msg_size=0, heartbeat=None)
    except Exception as exc:  # noqa: BLE001
        await client.close(code=1011, message=str(exc).encode())
        return client

    async def pump(src, dst, label: str):
        """把一侧的消息转发到另一侧，结束后关闭对端。

        单侧出错只结束这条连接，绝不让异常冒到 ``gather`` 之外：否则另一侧的
        转发协程会变成孤儿，表现为"Session 跑到一半就不动了"。
        """
        try:
            async for msg in src:
                if msg.type == WSMsgType.TEXT:
                    await dst.send_str(msg.data)
                elif msg.type == WSMsgType.BINARY:
                    await dst.send_bytes(msg.data)
                elif msg.type == WSMsgType.ERROR:
                    logger.warning(
                        "realtime proxy %s: peer error %s", label, src.exception()
                    )
                    break
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.warning("realtime proxy %s stopped: %r", label, exc)
        finally:
            # 任意一侧结束都把对端关掉：对端的 async for 因此结束，gather 才能
            # 一起收尾（半关闭时同样通知）。
            if not dst.closed:
                try:
                    await dst.close()
                except Exception as exc:  # noqa: BLE001
                    logger.debug("realtime proxy %s: close peer failed: %r", label, exc)

    try:
        await asyncio.gather(
            pump(client, backend, "client->backend"),
            pump(backend, client, "backend->client"),
        )
    finally:
        if not backend.closed:
            await backend.close()
    return client


def backend_for(request: web.Request) -> str:
    """Pick the vLLM backend for this socket.

    ``?instance=2`` selects the optional second instance; every other value
    (including a missing/unknown one, or a second instance that is not
    configured) falls back to the primary backend so the default URL keeps
    working unchanged.
    """
    instance = (request.query.get("instance") or "").strip()
    if instance in ("", "1"):
        return BACKEND_WS
    if instance == "2" and BACKEND_WS_2:
        return BACKEND_WS_2
    logger.warning(
        "realtime proxy: unknown instance=%r (AUDIO8_BACKEND_WS_2 %s), using primary",
        instance,
        "configured" if BACKEND_WS_2 else "not configured",
    )
    return BACKEND_WS


async def make_app() -> web.Application:
    app = web.Application()

    async def on_startup(app: web.Application) -> None:
        app["client_session"] = ClientSession()

    async def on_cleanup(app: web.Application) -> None:
        await app["client_session"].close()

    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)

    async def index(_request: web.Request) -> web.FileResponse:
        return web.FileResponse(f"{ROOT}/index.html")

    app.router.add_get("/", index)
    app.router.add_get("/v1/realtime", realtime_proxy)
    app.router.add_static("/", ROOT)
    return app


def main() -> None:
    ssl_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ssl_ctx.load_cert_chain("/certs/server.crt", "/certs/server.key")
    app = asyncio.run(make_app())
    web.run_app(app, host="0.0.0.0", port=LISTEN_PORT, ssl_context=ssl_ctx)


if __name__ == "__main__":
    main()
