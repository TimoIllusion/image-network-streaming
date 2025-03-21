FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

COPY ../ /app

WORKDIR /app

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

RUN pip install --no-cache-dir .

CMD ["python", "backend_imagezmq.py"]