"""Tests for per-task progress tracking, its /queue payload, and the web UI."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

import subgen
from subgen import ProgressHandler, TranscriptionCancelled, app


@pytest.fixture(autouse=True)
def reset_state():
    def clear():
        subgen.task_progress.clear()
        subgen.cancel_events.clear()
        subgen.cancelled_paths.clear()

    clear()
    yield
    clear()


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


class TestProgressStore:
    def test_update_records_progress(self):
        subgen.update_progress("/a.mkv", "a.mkv", "whisper", 30, 120)
        entry = subgen.get_progress_snapshot()["/a.mkv"]
        assert entry["seek"] == 30.0
        assert entry["total"] == 120.0
        assert entry["engine"] == "whisper"
        assert entry["name"] == "a.mkv"

    def test_started_kept_across_updates(self):
        subgen.update_progress("/a.mkv", "a.mkv", "whisper", 0, 120)
        first = subgen.get_progress_snapshot()["/a.mkv"]
        subgen.update_progress("/a.mkv", "a.mkv", "whisper", 60, 120)
        second = subgen.get_progress_snapshot()["/a.mkv"]
        assert second["started"] == first["started"]
        assert second["seek"] == 60.0
        assert second["updated"] >= second["started"]

    def test_clear_removes_entry(self):
        subgen.update_progress("/a.mkv", "a.mkv", "whisper", 1, 2)
        subgen.clear_progress("/a.mkv")
        assert "/a.mkv" not in subgen.get_progress_snapshot()

    def test_clear_unknown_path_is_noop(self):
        subgen.clear_progress("/missing.mkv")

    def test_snapshot_is_a_copy(self):
        subgen.update_progress("/a.mkv", "a.mkv", "whisper", 1, 2)
        subgen.get_progress_snapshot()["/a.mkv"]["seek"] = 99
        assert subgen.get_progress_snapshot()["/a.mkv"]["seek"] == 1.0


class TestProgressHandler:
    def test_call_records_progress(self):
        ProgressHandler("a.mkv", cancel_path="/a.mkv")(10, 100)
        entry = subgen.get_progress_snapshot()["/a.mkv"]
        assert (entry["seek"], entry["total"], entry["engine"]) == (10.0, 100.0, "whisper")

    def test_call_without_cancel_path_records_nothing(self):
        ProgressHandler("a.mkv")(10, 100)
        assert subgen.get_progress_snapshot() == {}

    def test_cancelled_task_raises(self):
        subgen.cancel_task("/a.mkv")
        with pytest.raises(TranscriptionCancelled):
            ProgressHandler("a.mkv", cancel_path="/a.mkv")(10, 100)


class _Audio:
    """Stands in for the numpy waveform chain in asr_task_worker (numpy is mocked)."""

    def __init__(self, seconds):
        self.samples = int(seconds * 16000)

    def flatten(self):
        return self

    def astype(self, _dtype):
        return self

    def __truediv__(self, _other):
        return self

    def __len__(self):
        return self.samples


class _Segment:
    def __init__(self, start, end, text):
        self.start = start
        self.end = end
        self.text = text


class _Model:
    def __init__(self, segments, on_yield=None):
        self.segments = segments
        self.on_yield = on_yield
        self.yielded = 0

    def recognize(self, _audio, sample_rate):
        for seg in self.segments:
            self.yielded += 1
            yield seg
            if self.on_yield:
                self.on_yield(self.yielded)


class TestParakeetWorkerProgress:
    PATH = "/movies/a.mkv"

    @pytest.fixture(autouse=True)
    def parakeet_env(self, monkeypatch):
        monkeypatch.setattr(subgen, "PARAKEET_ENABLED", True)
        monkeypatch.setattr(subgen, "delete_model", lambda: None)
        monkeypatch.setattr(subgen.np, "frombuffer", lambda *_a, **_k: _Audio(60))

    def _task(self, container):
        return {
            "path": self.PATH,
            "task": "transcribe",
            "language": "en",
            "video_file": self.PATH,
            "audio_content": b"",
            "encode": False,
            "output_format": "srt",
            "result_container": container,
        }

    def test_reports_progress_per_segment(self, monkeypatch):
        seen = []
        model = _Model(
            [_Segment(0, 10, "one"), _Segment(10, 25, "two")],
            on_yield=lambda _n: seen.append(subgen.get_progress_snapshot()[self.PATH]["seek"]),
        )
        monkeypatch.setattr(subgen, "get_parakeet_model", lambda: model)
        container = MagicMock()

        subgen.asr_task_worker(self._task(container))

        entry = subgen.get_progress_snapshot()[self.PATH]
        assert seen == [10.0, 25.0]
        assert (entry["total"], entry["engine"], entry["name"]) == (60.0, "parakeet", "a.mkv")
        container.set_result.assert_called_once()
        assert "two" in container.set_result.call_args[0][0]

    def test_cancel_stops_mid_stream(self, monkeypatch):
        model = _Model(
            [_Segment(0, 10, "one"), _Segment(10, 20, "two"), _Segment(20, 30, "three")],
            on_yield=lambda n: subgen.cancel_task(self.PATH) if n == 1 else None,
        )
        monkeypatch.setattr(subgen, "get_parakeet_model", lambda: model)
        container = MagicMock()

        subgen.asr_task_worker(self._task(container))

        assert model.yielded == 2
        container.set_error.assert_called_once_with("Cancelled by user")
        container.set_result.assert_not_called()


class TestQueueEndpoint:
    def test_includes_progress(self, client):
        subgen.update_progress("/a.mkv", "a.mkv", "parakeet", 5, 50)
        body = client.get("/queue").json()
        assert body["progress"]["/a.mkv"]["seek"] == 5.0
        assert body["progress"]["/a.mkv"]["engine"] == "parakeet"

    def test_keeps_existing_keys(self, client):
        body = client.get("/queue").json()
        assert isinstance(body["processing"], list)
        assert isinstance(body["queued"], list)
        assert isinstance(body["now"], float)


class TestWebUI:
    def test_serves_html(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/html")
        assert "Subgen queue" in resp.text

    def test_uses_relative_api_paths(self, client):
        text = client.get("/").text
        assert "fetch('queue'" in text
        assert "fetch('/queue" not in text
