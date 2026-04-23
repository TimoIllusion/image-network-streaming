from inference_streaming_benchmark.backend.imagezmq.ai_server import main
from inference_streaming_benchmark.backend.sidecar import start_sidecar_in_thread

if __name__ == "__main__":
    start_sidecar_in_thread(transport="imagezmq", data_port=5555)
    main()
