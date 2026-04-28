from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import ClassVar
from uuid import uuid4

import numpy as np

Detections = list[dict]
Timings = dict[str, float]
CLIENT_RESPONSE_TIMEOUT_S = 5.0


@dataclass(frozen=True)
class InferenceRequest:
    image: np.ndarray
    client_name: str = "unknown"
    request_id: str = field(default_factory=lambda: uuid4().hex[:12])
    transport: str = "unknown"
    received_at: float = 0.0

    def __post_init__(self) -> None:
        if not self.request_id:
            object.__setattr__(self, "request_id", uuid4().hex[:12])


Handler = Callable[[InferenceRequest], tuple[Detections, Timings]]


class Transport(ABC):
    """A single protocol handling both the server-side listener and the client-side sender.

    An instance is typically used in one role at a time. Server-side state and client-side
    state coexist on the object but do not interact.
    """

    name: ClassVar[str]
    display_name: ClassVar[str]
    default_port: ClassVar[int]

    # Server role
    @abstractmethod
    def start(self, port: int, handler: Handler) -> None:
        """Begin listening. Non-blocking — listener runs in its own thread."""

    @abstractmethod
    def stop(self) -> None:
        """Stop listening. Blocks until the listener thread has returned."""

    # Client role
    @abstractmethod
    def connect(self, host: str, port: int) -> None: ...

    @abstractmethod
    def send(
        self,
        frame: np.ndarray,
        *,
        client_name: str = "unknown",
        request_id: str | None = None,
    ) -> tuple[Detections | None, Timings]: ...

    @abstractmethod
    def disconnect(self) -> None: ...
