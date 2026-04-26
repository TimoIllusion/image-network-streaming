"""Per-device client. One per machine — or several on one host with --port for multi-instance benchmarks.

Usage::

    python client.py                       # port 8501 if free, else auto; name = <hostname>-<port>
    python client.py --port 8502           # explicit port
    python client.py --port 0              # auto-pick a free port
    python client.py --name rpi-edge-1     # explicit name
    INFSB_UI_PORT=8503 python client.py    # env var still works (overridden by --port)
"""

import os

# Must precede any cv2 import in this process — disables MSMF HW transforms on Windows.
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

import argparse  # noqa: E402
import socket  # noqa: E402

import uvicorn  # noqa: E402

DEFAULT_PORT = 8501


def _find_free_port() -> int:
    """Ask the kernel for any available port (binds to :0, reads back, releases)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("", 0))
        return s.getsockname()[1]
    finally:
        s.close()


def _is_port_free(port: int) -> bool:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        s.bind(("", port))
        return True
    except OSError:
        return False
    finally:
        s.close()


def _resolve_port(cli_port: int | None) -> int:
    """Resolve port: CLI flag wins; else env; else 8501 if free; else auto."""
    if cli_port is not None:
        return cli_port if cli_port > 0 else _find_free_port()
    env_port = os.getenv("INFSB_UI_PORT")
    if env_port:
        return int(env_port)
    return DEFAULT_PORT if _is_port_free(DEFAULT_PORT) else _find_free_port()


def _resolve_name(cli_name: str | None, port: int) -> str:
    """Default to <hostname>-<port> so multiple clients on one host don't collide in the registry."""
    if cli_name:
        return cli_name
    if os.getenv("INFSB_CLIENT_NAME"):
        return os.getenv("INFSB_CLIENT_NAME")
    return f"{socket.gethostname()}-{port}"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--port",
        type=int,
        default=None,
        help=f"UI port (0 = auto-pick free; default {DEFAULT_PORT} or auto if taken)",
    )
    p.add_argument(
        "--name",
        type=str,
        default=None,
        help="client name shown in the central UI (default: <hostname>-<port>)",
    )
    args = p.parse_args()

    port = _resolve_port(args.port)
    name = _resolve_name(args.name, port)

    # Push resolutions into env BEFORE importing config-consuming modules, so config picks them up.
    os.environ["INFSB_UI_PORT"] = str(port)
    os.environ["INFSB_CLIENT_NAME"] = name

    from inference_streaming_benchmark.client.app import app  # noqa: E402, deferred import

    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
