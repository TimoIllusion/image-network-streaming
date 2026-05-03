import importlib
from pathlib import Path

from loguru import logger


def test_importing_logging_module_does_not_create_log_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from inference_streaming_benchmark import logging as logging_module

    importlib.reload(logging_module)
    assert not (tmp_path / "logs").exists()


def test_setup_logging_creates_dir_and_writes_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from inference_streaming_benchmark import logging as logging_module

    importlib.reload(logging_module)
    log_dir = tmp_path / "custom-logs"
    logging_module.setup_logging(log_dir=str(log_dir))
    logger.info("hello")
    logger.complete()
    files = list(log_dir.glob("*.log"))
    assert len(files) == 1
    assert "hello" in Path(files[0]).read_text()


def test_setup_logging_is_idempotent(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from inference_streaming_benchmark import logging as logging_module

    importlib.reload(logging_module)
    log_dir = tmp_path / "logs-idem"
    logging_module.setup_logging(log_dir=str(log_dir))
    logging_module.setup_logging(log_dir=str(log_dir))
    logging_module.setup_logging(log_dir=str(log_dir))
    files = list(log_dir.glob("*.log"))
    assert len(files) == 1
