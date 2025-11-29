FROM python:3.11-slim AS builder

WORKDIR /app

COPY requirements.txt /tmp/requirements.txt

# Install runtime dependencies into a staging prefix
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir --prefix=/install -r /tmp/requirements.txt

# Copy source to embed in final image
COPY . .


FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CUDA_VISIBLE_DEVICES=""

WORKDIR /app

# Bring in installed deps and project source
COPY --from=builder /install /usr/local
COPY . .

ENTRYPOINT ["python", "-m", "mcp_server.main"]
