from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import ClassVar

import numpy as np

Detections = list[dict]
Timings = dict[str, float]
Handler = Callable[[np.ndarray], tuple[Detections, Timings]]


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
    def send(self, frame: np.ndarray) -> tuple[Detections | None, Timings]: ...

    @abstractmethod
    def disconnect(self) -> None: ...
