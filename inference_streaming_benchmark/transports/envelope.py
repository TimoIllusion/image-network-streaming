from __future__ import annotations

from typing import TypedDict

from .base import Detections, Timings

# gRPC has its own typed proto schema (DetectionResponse) and does not use this dict envelope.


class DetectionEnvelope(TypedDict):
    batched_detections: list[Detections]
    timings: Timings


def build(detections: list[Detections], timings: Timings) -> DetectionEnvelope:
    return {"batched_detections": detections, "timings": timings}


def unpack(payload: dict) -> tuple[Detections | None, Timings]:
    """Returns (first_detection_list_or_None, timings). Used by clients."""
    timings: Timings = dict(payload.get("timings", {}))
    batched = payload.get("batched_detections")
    if not batched:
        return None, timings
    return batched[0], timings
