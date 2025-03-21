# image-network-streaming

A test for image transmission via multiple communication systems: FastAPI, ZeroMQ (ZMQ), ImageZMQ, grpc.

Result of experiments (latency from better to worse, for 1920x1080 images, using NVIDIA RTX 4090 and AMD 7800X3D):

1. ImageZMQ (ca. 60-70 fps)
2. ZeroMQ (ca. 25 fps)
3. grpc (quite fast, but not very consistent, about 20-30 fps)
4. FastAPI (very slow, around 2s or 0.5 fps)

## Setup

```bash
git clone https://github.com/TimoIllusion/image-network-streaming.git

cd image-network-streaming

conda create -p ./.conda python=3.11 -y

conda activate ./.conda

pip install -e .

```
Note: Install torch with the respective hardware acceleration, e.g. cuda to increase inference speed.

## Run

**Using FastAPI for communication**

```bash
python backend_fastapi.py

streamlit run frontend.py fastapi

http://127.0.0.1:8501
```

**Using ZMQ for communication**

Note: The order of starting is important!

```bash
python backend_zmq.py

streamlit run frontend.py zmq

http://127.0.0.1:8501
```

**Using ImageZMQ for communication**

Note: The order of starting is important!

```bash
python backend_imagezmq.py

streamlit run frontend.py imagezmq

http://127.0.0.1:8501
```

**Using GRPC for communication**

```bash
python backend_grpc.py

streamlit run frontend.py grpc

http://127.0.0.1:8501
```

Note: transfer speed for images can significantly boosted by resizing them before sending. This will usually not cause issues with the ai model, since most models need images of low input sizes like 224x224.

## Tests

```bash
pip install -e .[test]
```

```bash
pytest tests
```
## Build Docker

```bash
docker build -f ./docker/Dockerfile -t aiserver:latest .
```

```bash
# runs imagezmq backend by default
docker run -it --name aiserver1 --rm --shm-size=8g --gpus=all -p 5555:5555 aiserver:latest
```
## TODO

- [x] Dockerize ai server