from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import requests

from inference_streaming_benchmark.config import CONTROL_BASE, CONTROL_TIMEOUT_S


@dataclass(frozen=True)
class SweepConfig:
    transport: str
    batching_enabled: bool
    max_batch_size: int
    max_wait_ms: float
    inference_mode: str = "single"
    inference_instances: int = 1


def parse_csv_strings(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_csv_ints(value: str) -> list[int]:
    return [int(item) for item in parse_csv_strings(value)]


def parse_csv_floats(value: str) -> list[float]:
    return [float(item) for item in parse_csv_strings(value)]


def build_plan(
    transports: list[str],
    batch_modes: list[str],
    batch_sizes: list[int],
    batch_waits_ms: list[float],
    inference_modes: list[str] | None = None,
    inference_instances: list[int] | None = None,
) -> list[SweepConfig]:
    plan: list[SweepConfig] = []
    inference_modes = inference_modes or ["single"]
    inference_instances = inference_instances or [1]
    for inference_mode in inference_modes:
        mode_instances = inference_instances if inference_mode == "multi-instance" else [1]
        for instance_count in mode_instances:
            for transport in transports:
                if "off" in batch_modes:
                    plan.append(SweepConfig(transport, False, 1, 0.0, inference_mode, instance_count))
                if "on" in batch_modes:
                    for size in batch_sizes:
                        for wait_ms in batch_waits_ms:
                            plan.append(SweepConfig(transport, True, size, wait_ms, inference_mode, instance_count))
    return plan


def fetch_transport_names(session: requests.Session, control_base: str) -> list[str]:
    response = session.get(f"{control_base}/transports", timeout=CONTROL_TIMEOUT_S)
    response.raise_for_status()
    return [item["name"] for item in response.json()]


def run_sweep(
    plan: list[SweepConfig],
    *,
    control_base: str = CONTROL_BASE,
    duration_s: float = 10.0,
    warmup_s: float = 2.0,
    session: requests.Session | None = None,
    sleep=time.sleep,
    on_run_complete: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    owns_session = session is None
    session = session or requests.Session()
    runs = []
    try:
        for idx, config in enumerate(plan, start=1):
            inference_response = session.post(
                f"{control_base}/inference",
                json={
                    "mode": config.inference_mode,
                    "instances": config.inference_instances,
                },
                timeout=CONTROL_TIMEOUT_S,
            )
            inference_response.raise_for_status()

            batching_response = session.post(
                f"{control_base}/batching",
                json={
                    "enabled": config.batching_enabled,
                    "max_batch_size": config.max_batch_size,
                    "max_wait_ms": config.max_wait_ms,
                },
                timeout=CONTROL_TIMEOUT_S,
            )
            batching_response.raise_for_status()

            control_response = session.post(
                f"{control_base}/clients/control-all",
                json={"backend": config.transport, "inference": True},
                timeout=CONTROL_TIMEOUT_S,
            )
            control_response.raise_for_status()
            sleep(warmup_s)

            clear_response = session.post(f"{control_base}/clients/clear-all", timeout=CONTROL_TIMEOUT_S)
            clear_response.raise_for_status()
            sleep(duration_s)

            clients_response = session.get(f"{control_base}/clients", timeout=CONTROL_TIMEOUT_S)
            clients_response.raise_for_status()
            clients_payload = clients_response.json()
            run = {
                "index": idx,
                "config": asdict(config),
                "inference": inference_response.json(),
                "batching": batching_response.json(),
                "control_results": control_response.json(),
                "clients": clients_payload.get("clients", []),
                "active_transport": clients_payload.get("active_transport"),
            }
            runs.append(run)
            if on_run_complete is not None:
                on_run_complete(run)
    finally:
        if owns_session:
            session.close()
    return {"runs": runs}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep transport and batching configs against a running cluster.")
    parser.add_argument("--control-base", default=CONTROL_BASE, help=f"control plane URL (default: {CONTROL_BASE})")
    parser.add_argument("--transports", default="", help="comma-separated transport names; default fetches all")
    parser.add_argument(
        "--batch",
        default="off,on",
        help="comma-separated batch modes: off,on (default: off,on)",
    )
    parser.add_argument("--batch-sizes", default="1,2,4,8", help="comma-separated max batch sizes for batch=on")
    parser.add_argument("--batch-waits-ms", default="0,5,10,20", help="comma-separated max wait values for batch=on")
    parser.add_argument(
        "--inference-modes",
        default="single",
        help="comma-separated inference modes: single,unsafe-multi,multi-instance",
    )
    parser.add_argument("--inference-instances", default="1", help="comma-separated instance counts")
    parser.add_argument("--duration-s", type=float, default=10.0, help="measurement duration per run")
    parser.add_argument("--warmup-s", type=float, default=2.0, help="warmup duration before clearing stats")
    parser.add_argument("--output", default="benchmark-runs.json", help="JSON output path")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    batch_modes = parse_csv_strings(args.batch)
    bad_modes = sorted(set(batch_modes) - {"off", "on"})
    if bad_modes:
        raise SystemExit(f"unknown batch mode(s): {', '.join(bad_modes)}")

    with requests.Session() as session:
        transports = (
            parse_csv_strings(args.transports) if args.transports else fetch_transport_names(session, args.control_base)
        )
        plan = build_plan(
            transports=transports,
            batch_modes=batch_modes,
            batch_sizes=parse_csv_ints(args.batch_sizes),
            batch_waits_ms=parse_csv_floats(args.batch_waits_ms),
            inference_modes=parse_csv_strings(args.inference_modes),
            inference_instances=parse_csv_ints(args.inference_instances),
        )
        result = run_sweep(
            plan,
            control_base=args.control_base,
            duration_s=args.duration_s,
            warmup_s=args.warmup_s,
            session=session,
        )

    output = Path(args.output)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote {len(result['runs'])} runs to {output}")


if __name__ == "__main__":
    main()
