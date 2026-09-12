from __future__ import annotations

import numpy as np
import pytest

from altur.types import AudioExample, Segmentation, Turn

SR = 8000


@pytest.fixture
def fake_webrtcvad(monkeypatch):
    class Vad:
        def __init__(self, mode: int):
            assert mode == 2

        def is_speech(self, frame: bytes, sample_rate: int) -> bool:
            assert sample_rate == SR
            return any(frame)

    monkeypatch.setitem(__import__("sys").modules, "webrtcvad", type("Module", (), {"Vad": Vad}))


def _example(
    ch0: np.ndarray, ch1: np.ndarray | None = None, seg: Segmentation | None = None
) -> AudioExample:
    return AudioExample(
        ch0=ch0.astype(np.float32),
        ch1=None if ch1 is None else ch1.astype(np.float32),
        seg=seg,
    )


def _tone(seconds: float, amplitude: float = 0.2) -> np.ndarray:
    samples = np.arange(round(seconds * SR))
    return amplitude * np.sin(2 * np.pi * 180 * samples / SR)


def test_pre_registered_segmenters_have_frozen_parameters():
    import altur.seg  # noqa: F401
    from altur.registry import turn_sources

    assert turn_sources.meta("energy@1") == {
        "is_oracle": False,
        "frame_ms": 20,
        "hop_ms": 10,
        "threshold_db_above_noise_floor": 12,
        "pad_ms": 30,
        "merge_gap_ms": 450,
        "min_length_ms": 200,
    }
    assert turn_sources.meta("webrtc@1") == {
        "is_oracle": False,
        "mode": 2,
        "frame_ms": 20,
        "padding_ms": 300,
        "hysteresis": 0.90,
    }
    assert turn_sources.meta("oracle@1") == {"is_oracle": True}


@pytest.mark.parametrize("name", ["energy", "webrtc"])
def test_vads_return_no_turns_for_silence(name: str, fake_webrtcvad):
    from altur.seg import energy, webrtc

    segment = {"energy": energy, "webrtc": webrtc}[name]
    result = segment(_example(np.zeros(SR), np.zeros(SR)))

    assert result.source == f"{name}@1"
    assert result.turns == ()


@pytest.mark.parametrize("name", ["energy", "webrtc"])
def test_vads_handle_audio_shorter_than_one_frame(name: str, fake_webrtcvad):
    from altur.seg import energy, webrtc

    segment = {"energy": energy, "webrtc": webrtc}[name]
    result = segment(_example(np.zeros(10)))

    assert result.turns == ()


def test_energy_segments_both_channels_independently_and_deterministically():
    from altur.seg import energy

    ch0 = np.zeros(3 * SR)
    ch1 = np.zeros(3 * SR)
    ch0[SR : 2 * SR] = _tone(1)
    ch1[2 * SR : 3 * SR] = _tone(1)
    example = _example(ch0, ch1)

    first = energy(example)
    second = energy(example)

    assert first == second
    assert len(first.by_channel(0)) == 1
    assert len(first.by_channel(1)) == 1
    assert first.by_channel(0)[0].start <= 1.0
    assert first.by_channel(0)[0].end >= 2.0
    assert first.by_channel(1)[0].start <= 2.0
    assert first.by_channel(1)[0].end == pytest.approx(3.0)


def test_energy_discards_speech_shorter_than_minimum_length():
    from altur.seg import energy

    ch0 = np.zeros(SR)
    ch0[2000:2800] = _tone(0.1)

    assert energy(_example(ch0)).turns == ()


def test_webrtc_segments_both_channels_independently_and_deterministically(fake_webrtcvad):
    from altur.seg import webrtc

    ch0 = np.zeros(2 * SR)
    ch1 = np.zeros(2 * SR)
    ch0[:SR] = _tone(1)
    ch1[SR:] = _tone(1)
    example = _example(ch0, ch1)

    first = webrtc(example)
    second = webrtc(example)

    assert first == second
    assert len(first.by_channel(0)) == 1
    assert len(first.by_channel(1)) == 1
    assert first.by_channel(0)[0].start == 0.0
    # El padding retrospectivo de 300 ms conserva el frame silencioso precedente.
    assert first.by_channel(1)[0].start == pytest.approx(0.98)


def test_webrtc_missing_dependency_explains_how_to_install(monkeypatch):
    import builtins

    from altur.seg import webrtc

    real_import = builtins.__import__

    def missing_webrtcvad(name, *args, **kwargs):
        if name == "webrtcvad":
            raise ImportError("missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing_webrtcvad)

    with pytest.raises(RuntimeError, match=r"pip install '.*\[train\]'?"):
        webrtc(_example(np.zeros(SR)))


def test_oracle_only_wraps_an_oracle_already_on_audio_example():
    from altur.seg import oracle

    expected = Segmentation((Turn(0, 0.2, 0.8),), source="oracle@1", params={"provider": "test"})
    assert oracle(_example(np.zeros(SR), seg=expected)) is expected


@pytest.mark.parametrize(
    "seg",
    [None, Segmentation((Turn(0, 0.2, 0.8),), source="energy@1")],
)
def test_oracle_rejects_missing_or_non_oracle_segmentation(seg: Segmentation | None):
    from altur.seg import oracle

    with pytest.raises(ValueError, match="oracle"):
        oracle(_example(np.zeros(SR), seg=seg))
