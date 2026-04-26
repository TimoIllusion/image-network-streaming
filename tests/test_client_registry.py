import time

from inference_streaming_benchmark.client_registry import ClientRegistry


def test_register_creates_record():
    reg = ClientRegistry()
    rec = reg.register("rpi-1", "http://10.0.0.5:8501", "1.0")
    assert rec.name == "rpi-1"
    assert rec.ui_url == "http://10.0.0.5:8501"
    assert rec.last_heartbeat_at == rec.registered_at
    assert reg.get("rpi-1") is rec


def test_register_replaces_existing_entry():
    """A reconnecting client (same name) should replace the old record without leaking state."""
    reg = ClientRegistry()
    reg.register("rpi-1", "http://10.0.0.5:8501", "1.0")
    reg.heartbeat("rpi-1", {"fps": 30.0})

    rec2 = reg.register("rpi-1", "http://10.0.0.6:8501", "1.1")
    assert rec2.ui_url == "http://10.0.0.6:8501"
    # Stats from the old record must not bleed through.
    assert rec2.stats == {}


def test_heartbeat_updates_stats_and_timestamp():
    reg = ClientRegistry()
    rec = reg.register("rpi-1", "http://10.0.0.5:8501", "1.0")
    t0 = rec.last_heartbeat_at
    time.sleep(0.01)
    reg.heartbeat("rpi-1", {"fps": 25.5})
    rec = reg.get("rpi-1")
    assert rec.stats == {"fps": 25.5}
    assert rec.last_heartbeat_at > t0


def test_heartbeat_unknown_client_raises():
    reg = ClientRegistry()
    try:
        reg.heartbeat("nope", {})
    except KeyError:
        pass
    else:
        raise AssertionError("expected KeyError for unknown client")


def test_list_active_filters_stale():
    """Stale clients (no heartbeat past cutoff) age out lazily on list_active."""
    reg = ClientRegistry()
    reg.register("fresh", "http://x", "")
    rec = reg.register("stale", "http://y", "")
    rec.last_heartbeat_at = time.time() - 30  # forced stale

    active = reg.list_active(stale_after_s=5.0)
    names = {r.name for r in active}
    assert names == {"fresh"}
    # Stale entry is also dropped from the registry so a re-register is clean.
    assert reg.get("stale") is None


def test_remove_returns_true_only_if_present():
    reg = ClientRegistry()
    reg.register("rpi-1", "http://x", "")
    assert reg.remove("rpi-1") is True
    assert reg.remove("rpi-1") is False


def test_to_dict_includes_age_s():
    reg = ClientRegistry()
    rec = reg.register("rpi-1", "http://10.0.0.5:8501", "1.0")
    d = rec.to_dict()
    assert d["name"] == "rpi-1"
    assert d["ui_url"] == "http://10.0.0.5:8501"
    assert "age_s" in d and d["age_s"] >= 0
