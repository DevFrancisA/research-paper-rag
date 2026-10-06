FROM python:3.14-slim

COPY --from=ghcr.io/astral-sh/uv:0.12 /uv /uvx /bin/

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy \
    FASTEMBED_CACHE_PATH=/models HF_HUB_DISABLE_SYMLINKS_WARNING=1

# Dependencies first so code changes don't reinstall them.
COPY pyproject.toml uv.lock .python-version README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src
COPY main.py streamlit_app.py ./
RUN uv sync --frozen --no-dev

# Bake the BM25 and reranker models into the image so the first request doesn't download them.
ARG RERANKER_MODEL=BAAI/bge-reranker-base
RUN uv run python -c "from fastembed import SparseTextEmbedding; \
from fastembed.rerank.cross_encoder import TextCrossEncoder; \
SparseTextEmbedding('Qdrant/bm25'); TextCrossEncoder('${RERANKER_MODEL}')"

EXPOSE 8000
CMD ["uv", "run", "--no-sync", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
