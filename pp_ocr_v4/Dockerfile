# syntax=docker/dockerfile:1
FROM registry.baidubce.com/paddlepaddle/paddle:2.6.1-gpu-cuda12.0-cudnn8.9-trt8.6
ARG TEST_ENV

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=${PORT:-9090} \
    PIP_CACHE_DIR=/.cache \
    # pip镜像源
    PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
    WORKERS=1 \
    THREADS=8

# Update the base OS
#RUN --mount=type=cache,target="/var/cache/apt",sharing=locked \
#    --mount=type=cache,target="/var/lib/apt/lists",sharing=locked \
#    set -eux; \
#    apt-get update; \
#    apt-get upgrade -y; \
#    apt install --no-install-recommends -y  \
#        git; \
#    apt-get autoremove -y

# install base requirements
COPY requirements-base.txt .
RUN --mount=type=cache,target=${PIP_CACHE_DIR},sharing=locked \
    pip install -r requirements-base.txt

# install custom requirements
COPY requirements.txt .
RUN --mount=type=cache,target=${PIP_CACHE_DIR},sharing=locked \
    pip install -r requirements.txt

# install test requirements if needed
COPY requirements-test.txt .
# build only when TEST_ENV="true"
RUN --mount=type=cache,target=${PIP_CACHE_DIR},sharing=locked \
    if [ "$TEST_ENV" = "true" ]; then \
      pip install -r requirements-test.txt; \
    fi

COPY . .

EXPOSE 9090

# CMD gunicorn --preload --bind :$PORT --workers $WORKERS --threads $THREADS --timeout 0 _wsgi:app
CMD label-studio-ml start .
