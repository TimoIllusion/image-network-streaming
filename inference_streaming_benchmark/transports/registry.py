from __future__ import annotations

from .base import Transport

_REGISTRY: dict[str, type[Transport]] = {}


def register(cls: type[Transport]) -> None:
    if cls.name in _REGISTRY:
        raise ValueError(f"Transport {cls.name!r} already registered")
    _REGISTRY[cls.name] = cls


def get(name: str) -> type[Transport]:
    if name not in _REGISTRY:
        raise KeyError(f"Transport {name!r} not registered (have: {sorted(_REGISTRY)})")
    return _REGISTRY[name]


def all_transports() -> dict[str, type[Transport]]:
    return dict(_REGISTRY)
