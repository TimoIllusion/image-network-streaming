FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

# Build context should be the repo root (see README docker build command).
COPY . /app

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libsm6 libxext6 \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install .

# Control plane on 9000. Default transport (http_multipart) binds 8008 inside the container.
# Override with `--default <name>`; expose the matching port when running the image.
EXPOSE 9000 8008

CMD ["python", "serve.py"]