FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

# 回归测试与 pytest 配置（默认不安装测试依赖；需要在容器内跑测试时
# 执行 `pip install -r requirements-dev.txt` 后 `pytest -v`）
COPY tests ./tests
COPY pytest.ini requirements-dev.txt ./

# 采集进程健康检查：本容器注册的 worker 心跳未过期即为存活
COPY scripts/worker_healthcheck.sh /usr/local/bin/worker_healthcheck.sh
RUN chmod +x /usr/local/bin/worker_healthcheck.sh

EXPOSE 8080 8081
