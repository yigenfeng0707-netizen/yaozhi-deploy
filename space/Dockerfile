FROM python:3.10-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 fonts-noto-cjk \
    && rm -rf /var/lib/apt/lists/*

# torch/torchvision 走 CPU 源 + --no-deps（该源缺 flit_core 等构建依赖），
# 运行时依赖在下一步从 PyPI 补齐
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu --no-deps \
    "torch==2.3.1" "torchvision==0.18.1"

RUN pip install --no-cache-dir \
    "filelock>=3.10" "typing-extensions>=4.8" "sympy>=1.11" \
    "networkx>=2.8" "jinja2>=3" "fsspec>=2023.5" \
    "fastapi>=0.110,<0.120" \
    "uvicorn>=0.27,<0.35" \
    "python-multipart>=0.0.9" \
    "pydantic>=2.5,<3" \
    "numpy>=1.24,<2.0" \
    "opencv-python-headless>=4.8,<6" \
    "pillow>=10.0"

COPY . .

# 魔搭 Docker 创空间硬性要求：监听 0.0.0.0:7860
EXPOSE 7860
ENTRYPOINT ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
