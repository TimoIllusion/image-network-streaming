from __future__ import annotations

import argparse
import json
import time
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
) -> list[SweepConfig]:
    plan: list[SweepConfig] = []
    for transport in transports:
        if "off" in batch_modes:
            plan.append(SweepConfig(transport, False, 1, 0.0))
        if "on" in batch_modes:
            for size in batch_sizes:
                for wait_ms in batch_waits_ms:
                    plan.append(SweepConfig(transport, True, size, wait_ms))
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
) -> dict[str, Any]:
    owns_session = session is None
    session = session or requests.Session()
    runs = []
    try:
        for idx, config in enumerate(plan, start=1):
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
            runs.append(
                {
                    "index": idx,
                    "config": asdict(config),
                    "batching": batching_response.json(),
                    "control_results": control_response.json(),
                    "clients": clients_payload.get("clients", []),
                    "active_transport": clients_payload.get("active_transport"),
                }
            )
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
