from inference_streaming_benchmark.backend.grpc.ai_server import serve
from inference_streaming_benchmark.backend.sidecar import start_sidecar_in_thread

if __name__ == "__main__":
    start_sidecar_in_thread(transport="grpc", data_port=50051, sidecar_port=9004)
    serve()
