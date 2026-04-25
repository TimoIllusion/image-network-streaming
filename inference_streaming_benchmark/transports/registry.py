from __future__ import annotations

from .base import Transport

_REGISTRY: dict[str, type[Transport]] = {}


def register(
    cls: type[Transport],
    *,
    name: str | None = None,
    display_name: str | None = None,
    port: int | None = None,
    raw: bool = False,
) -> None:
    """Register a transport. Overrides let the same class register multiple variants (e.g. JPEG + raw)."""
    overrides: dict = {}
    if name is not None:
        overrides["name"] = name
    if display_name is not None:
        overrides["display_name"] = display_name
    if port is not None:
        overrides["default_port"] = port
    if raw:
        overrides["RAW"] = True

    if overrides:
        cls = type(f"{cls.__name__}_{overrides.get('name', cls.name)}", (cls,), overrides)

    if cls.name in _REGISTRY:
        raise ValueError(f"Transport {cls.name!r} already registered")
    _REGISTRY[cls.name] = cls


def get(name: str) -> type[Transport]:
    if name not in _REGISTRY:
        raise KeyError(f"Transport {name!r} not registered (have: {sorted(_REGISTRY)})")
    return _REGISTRY[name]


def all_transports() -> dict[str, type[Transport]]:
    return dict(_REGISTRY)
