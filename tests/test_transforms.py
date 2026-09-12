from __future__ import annotations

import numpy as np
import pytest

from altur.types import AudioExample, Segmentation, Turn


def _example(ch0: np.ndarray, ch1: np.ndarray | None = None) -> AudioExample:
    if ch1 is None:
        ch1 = np.linspace(-0.2, 0.2, len(ch0), dtype=np.float32)
    return AudioExample(ch0=ch0, ch1=ch1, sr=8000, seg=Segmentation(
        (Turn(0, 0.1, 0.4), Turn(1, 0.5, 0.8)), "vad@fixture"
    ))


@pytest.fixture(scope="module", autouse=True)
def _load_transforms():
    import altur.transforms  # noqa: F401


def _tone(hz: float, seconds: float = 1.0) -> np.ndarray:
    t = np.arange(int(seconds * 8000), dtype=np.float64) / 8000
    return (0.5 * np.sin(2 * np.pi * hz * t)).astype(np.float32)


@pytest.mark.parametrize("ref", ["lowpass_3400@1", "highpass_300@1", "mulaw_roundtrip@1"])
def test_transform_metadata_is_diagnostic_holdout_and_does_not_change_shape(ref: str):
    from altur.registry import transforms

    meta = transforms.meta(ref)
    assert meta["use"] == "holdout"
    assert meta["purpose"] == "diagnostic"
    assert meta["changes_length"] is False
    assert meta["shifts_timing"] is False


@pytest.mark.parametrize("name", ["lowpass_3400", "highpass_300", "mulaw_roundtrip"])
def test_transform_preserves_length_sr_segmentation_and_does_not_mutate_input(name: str):
    from altur.registry import transforms

    ch0 = _tone(700)
    ch1 = _tone(1100)
    ex = _example(ch0, ch1)
    before0, before1 = ex.ch0.copy(), ex.ch1.copy()
    out = transforms.resolve(f"{name}@1")(ex, np.random.default_rng(4))

    assert out is not ex
    assert out.sr == ex.sr
    assert out.n_samples == ex.n_samples
    assert out.seg == ex.seg
    np.testing.assert_array_equal(ex.ch0, before0)
    np.testing.assert_array_equal(ex.ch1, before1)
    np.testing.assert_array_equal(out.ch1, ex.ch1)
    assert out.ch0.flags.writeable is False


def test_lowpass_and_highpass_have_expected_approximate_frequency_response():
    from altur.registry import transforms

    lowpass = transforms.resolve("lowpass_3400@1")
    highpass = transforms.resolve("highpass_300@1")
    low = lowpass(_example(_tone(500)), None).ch0
    high = lowpass(_example(_tone(3800)), None).ch0
    hp_low = highpass(_example(_tone(100)), None).ch0
    hp_high = highpass(_example(_tone(1200)), None).ch0

    def rms(x: np.ndarray) -> float:
        return float(np.sqrt(np.mean(x[1000:-1000] ** 2)))

    assert rms(low) > 0.2
    assert rms(high) < 0.15
    assert rms(hp_high) > 3 * rms(hp_low)


def test_mulaw_roundtrip_is_deterministic_and_quantized():
    from altur.registry import transforms

    fn = transforms.resolve("mulaw_roundtrip@1")
    ex = _example(_tone(440))
    first = fn(ex, np.random.default_rng(99))
    second = fn(ex, np.random.default_rng(99))
    np.testing.assert_array_equal(first.ch0, second.ch0)
    assert np.unique(first.ch0).size < np.unique(ex.ch0).size
    np.testing.assert_array_equal(first.ch1, ex.ch1)


def test_audio_transforms_reject_invalid_rng_without_using_global_randomness():
    from altur.registry import transforms

    with pytest.raises(TypeError):
        transforms.resolve("lowpass_3400@1")(_example(_tone(440)), object())
