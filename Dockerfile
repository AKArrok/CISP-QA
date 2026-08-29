FROM python:3.13-slim

WORKDIR /app

# 先装依赖再拷代码，利用 Docker 层缓存
COPY requirements.txt .
# torch 用 CPU 版，镜像从 ~2GB 降到 ~600MB
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

# 模型缓存目录挂载为卷，避免每次构建重新下载 embedding/精排模型
ENV HF_HOME=/app/.hf_cache \
    LOCAL_EMBEDDING_DEVICE=cpu \
    ENABLE_RERANKING=true
VOLUME ["/app/.hf_cache", "/app/data"]

EXPOSE 9528

# 容器内需挂载 data/（或先运行 data_ingest 脚本生成）
CMD ["python", "server.py"]
