"""Unified AI server with a hot-swappable transport.

Usage::

    python serve.py                      # start with http_multipart active (default)
    python serve.py --default zmq        # start with zmq active
    python serve.py --default none       # start with no transport active — frontend picks
"""

from __future__ import annotations

import argparse

from inference_streaming_benchmark.server import run
from inference_streaming_benchmark.transports import registry


def main() -> None:
    p = argparse.ArgumentParser()
    choices = sorted(registry.all_transports())
    p.add_argument(
        "--default",
        default="http_multipart",
        help=f"transport to start with. One of {choices + ['none']}. 'none' means idle until the frontend picks one.",
    )
    args = p.parse_args()
    default = None if args.default == "none" else args.default
    if default is not None and default not in registry.all_transports():
        raise SystemExit(f"unknown transport: {default!r} (have: {choices})")
    run(default=default)


if __name__ == "__main__":
    main()
