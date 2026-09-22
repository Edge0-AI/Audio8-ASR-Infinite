# Audio8 ASR Infinite 服务镜像（版本口径以本文件为准）。
# 基镜像已含 vllm 0.27.1 + torch 2.13.0+cu129 + flashinfer（python 3.12）。
# 版本口径：vllm 0.27.1 / transformers 5.13.0（需含 models.voxtral_realtime）/
#           torch 2.13.0+cu129（随基镜像）/ python 3.12。
# 维护点（升级 vLLM 前必读）：
#   - Audio8RollingScheduler 依赖原生 Scheduler._update_request_as_session 的签名与语义；
#   - realtime endpoint 插件依赖 0.27.1 的两个行为：fastapi 同路径「首个注册优先」
#     （故先移除原生路由再挂载）、init_state 里 openai_serving_realtime 的接线方式；
#     升级后须重验启动日志里 /v1/realtime 只有一条路由。
# 环境要点：
#   1. 基镜像 transformers 是 5.15.0，降到已验证的 5.13.0（voxtral_realtime 硬依赖口径）；
#   2. 额外安装 aiohttp（web_tls_proxy 反代）与 pybase64（realtime/connection.py 依赖）。
# 适配代码不在镜像内烘焙：仓库与插件 dist-info 通过 compose 挂载 + PYTHONPATH 在启动时引入
# （importlib.metadata 会扫描 PYTHONPATH 上的 dist-info，因此 vllm_serve/ 目录加入
# PYTHONPATH 即完成 vllm.general_plugins / vllm.endpoint_plugins 注册，无需复制进 site-packages）。
# build context 为仓库根目录（见 docker-compose.yml build.context）。
FROM docker.m.daocloud.io/vllm/vllm-openai:v0.27.1-cu129-ubuntu2404

# 默认走公共 PyPI；需要镜像时用 --build-arg PIP_INDEX_URL=... 覆盖
# （若是 http 源，请自行追加 --trusted-host）。
ARG PIP_INDEX_URL=https://pypi.org/simple/

RUN /usr/bin/python3.12 -m pip install --no-cache-dir "transformers==5.13.0" "aiohttp" "pybase64" \
      -i "${PIP_INDEX_URL}"

# Web 客户端（纯静态：index.html + config.js + pcm-worklet.js，浏览器直连 vLLM /v1/realtime）。
# 烘焙进镜像 /opt/audio8_client；运行时由 web 服务复制到 /srv 后按部署端口改写
# config.js 的 publicVllmPort 再用 python http.server 提供。
COPY client/ /opt/audio8_client/

# HTTPS 单端口入口（静态 + /v1/realtime wss 反代，docker/web_tls_proxy.py）
COPY docker/web_tls_proxy.py /opt/web_tls_proxy.py
