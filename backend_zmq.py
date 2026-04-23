from inference_streaming_benchmark.backend.sidecar import start_sidecar_in_thread
from inference_streaming_benchmark.backend.zmq.ai_server import main

if __name__ == "__main__":
    start_sidecar_in_thread(transport="zmq", data_port=5555, sidecar_port=9002)
    main()
