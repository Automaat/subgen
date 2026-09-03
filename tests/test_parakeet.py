"""Tests for the Parakeet language-routed fast path."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import subgen
from subgen import should_use_parakeet, parakeet_segments_to_output, _srt_timestamp


class _Segment:
    def __init__(self, start, end, text):
        self.start = start
        self.end = end
        self.text = text


class TestShouldUseParakeet:
    def setup_method(self):
        subgen.PARAKEET_ENABLED = True

    def test_supported_language_transcribe(self):
        assert should_use_parakeet("transcribe", "es", False, "srt") is True

    def test_case_insensitive_language(self):
        assert should_use_parakeet("transcribe", "EN", False, "srt") is True

    def test_translate_never_routes(self):
        assert should_use_parakeet("translate", "es", False, "srt") is False

    def test_unsupported_language_falls_back(self):
        assert should_use_parakeet("transcribe", "ja", False, "srt") is False

    def test_encoded_audio_falls_back(self):
        assert should_use_parakeet("transcribe", "es", True, "srt") is False

    def test_missing_language_falls_back(self):
        assert should_use_parakeet("transcribe", None, False, "srt") is False

    def test_verbose_json_falls_back(self):
        assert should_use_parakeet("transcribe", "es", False, "verbose_json") is False

    def test_disabled_falls_back(self):
        subgen.PARAKEET_ENABLED = False
        assert should_use_parakeet("transcribe", "es", False, "srt") is False


class TestSrtTimestamp:
    def test_zero(self):
        assert _srt_timestamp(0) == "00:00:00,000"

    def test_sub_second(self):
        assert _srt_timestamp(1.234) == "00:00:01,234"

    def test_hours(self):
        assert _srt_timestamp(3661.5) == "01:01:01,500"

    def test_negative_clamps_to_zero(self):
        assert _srt_timestamp(-1) == "00:00:00,000"


class TestParakeetSegmentsToOutput:
    def setup_method(self):
        self.segments = [
            _Segment(0.0, 1.5, "Hello there"),
            _Segment(2.0, 3.25, "General Kenobi"),
        ]

    def test_srt_format(self):
        out = parakeet_segments_to_output(self.segments, "srt")
        assert out.splitlines()[:4] == [
            "1",
            "00:00:00,000 --> 00:00:01,500",
            "Hello there",
            "",
        ]
        assert "General Kenobi" in out

    def test_text_format(self):
        assert parakeet_segments_to_output(self.segments, "text") == "Hello there General Kenobi"

    def test_json_format(self):
        assert json.loads(parakeet_segments_to_output(self.segments, "json")) == {
            "text": "Hello there General Kenobi"
        }

    def test_vtt_format(self):
        out = parakeet_segments_to_output(self.segments, "vtt")
        assert out.startswith("WEBVTT\n")
        assert "00:00:00.000 --> 00:00:01.500" in out

    def test_empty_segments(self):
        assert parakeet_segments_to_output([], "text") == ""
