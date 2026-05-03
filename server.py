"""Unified AI server with a hot-swappable transport.

Usage::

    python server.py                      # start with http_multipart active (default)
    python server.py --default zmq        # start with zmq active
    python server.py --default none       # start with no transport active — a client picks
"""

from __future__ import annotations

import argparse

from inference_streaming_benchmark.logging import setup_logging
from inference_streaming_benchmark.server import run
from inference_streaming_benchmark.transports import registry


def main() -> None:
    setup_logging()
    p = argparse.ArgumentParser()
    choices = sorted(registry.all_transports())
    p.add_argument(
        "--default",
        default="http_multipart",
        help=f"transport to start with. One of {choices + ['none']}. 'none' means idle until a client picks one.",
    )
    args = p.parse_args()
    default = None if args.default == "none" else args.default
    if default is not None and default not in registry.all_transports():
        raise SystemExit(f"unknown transport: {default!r} (have: {choices})")
    run(default=default)


if __name__ == "__main__":
    main()
