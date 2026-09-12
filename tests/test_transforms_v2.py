from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest
import yaml

from altur.types import AudioExample, Segmentation, Turn

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg no disponible")
V2 = Path(__file__).resolve().parent.parent / "configs" / "perturbations" / "v2.yaml"


@pytest.fixture(scope="module", autouse=True)
def _load():
    import altur.transforms
    import altur.transforms_v2  # noqa: F401


def _tone(hz: float, seconds: float = 2.0, amp: float = 0.4) -> np.ndarray:
    t = np.arange(int(seconds * 8000), dtype=np.float64) / 8000
    return (amp * np.sin(2 * np.pi * hz * t)).astype(np.float32)


def _ex(ch0: np.ndarray, ch1: np.ndarray | None = None) -> AudioExample:
    ch1 = _tone(1100, len(ch0) / 8000, 0.2) if ch1 is None else ch1
    seg = Segmentation((Turn(0, 0.2, 1.0), Turn(1, 1.1, 1.8)), "oracle@1")
    return AudioExample(ch0, ch1, sr=8000, seg=seg)


def _peak_hz(x: np.ndarray) -> float:
    spectrum = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    return float(np.fft.rfftfreq(len(x), 1 / 8000)[int(np.argmax(spectrum))])


def test_v2_mapea_las_seis_condiciones_a_refs_registradas_y_holdout():
    from altur.registry import transforms

    spec = yaml.safe_load(V2.read_text())
    assert set(spec["conditions"]) == {
        "clean", "noise_snr10", "pitch_up1st", "timestretch_0.9x", "lowpass_3400hz", "opus_16kbps",
    }
    for refs in spec["conditions"].values():
        for ref in refs:
            assert transforms.meta(ref)["use"] == "holdout"


def test_noise_exige_rng_y_no_usa_el_estado_global():
    from altur.registry import transforms

    fn = transforms.resolve("noise_snr10@1")
    ex = _ex(_tone(440))
    with pytest.raises(ValueError, match="rng"):
        fn(ex, None)
    np.random.seed(1)
    a = fn(ex, np.random.default_rng(7)).ch0
    np.random.seed(999)
    b = fn(ex, np.random.default_rng(7)).ch0
    c = fn(ex, np.random.default_rng(8)).ch0
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_noise_llega_al_snr_declarado_y_no_toca_el_agente():
    from altur.registry import transforms

    ex = _ex(_tone(440))
    out = transforms.resolve("noise_snr10@1")(ex, np.random.default_rng(3))
    noise = out.ch0.astype(np.float64) - ex.ch0
    snr = 10 * np.log10(np.mean(ex.ch0.astype(np.float64) ** 2) / np.mean(noise**2))
    assert abs(snr - 10.0) < 0.5
    np.testing.assert_array_equal(out.ch1, ex.ch1)
    assert out.seg == ex.seg


@needs_ffmpeg
def test_pitch_sube_un_semitono_conserva_longitud_y_es_determinista():
    from altur.registry import transforms

    fn = transforms.resolve("pitch_up1st@1")
    ex = _ex(_tone(440))
    out = fn(ex, None)
    assert out.n_samples == ex.n_samples
    assert abs(_peak_hz(out.ch0[2000:-2000]) - 440 * 2 ** (1 / 12)) < 6
    np.testing.assert_array_equal(out.ch0, fn(ex, None).ch0)
    np.testing.assert_array_equal(out.ch1, ex.ch1)


@needs_ffmpeg
def test_timestretch_alarga_los_dos_canales_conserva_tono_y_reescala_turnos():
    from altur.registry import transforms

    fn = transforms.resolve("timestretch_0_9x@1")
    ex = _ex(_tone(440))
    out = fn(ex, None)
    assert out.n_samples == round(ex.n_samples / 0.9)
    assert out.ch1 is not None and len(out.ch1) == out.n_samples
    assert abs(_peak_hz(out.ch0[2000:-2000]) - 440) < 6
    assert out.seg is not None and out.seg.is_oracle
    first = out.seg.turns[0]
    assert first.start == pytest.approx(0.2 / 0.9) and first.end == pytest.approx(1.0 / 0.9)
    assert out.seg.params["time_scale"] == pytest.approx(1 / 0.9)


@needs_ffmpeg
def test_opus_conserva_longitud_es_determinista_y_si_degrada():
    from altur.registry import transforms

    fn = transforms.resolve("opus_16kbps@1")
    ex = _ex(_tone(440))
    out = fn(ex, None)
    assert out.n_samples == ex.n_samples
    assert np.all(np.isfinite(out.ch0))
    assert not np.array_equal(out.ch0, ex.ch0)
    ratio = np.sqrt(np.mean(out.ch0.astype(np.float64) ** 2) / np.mean(ex.ch0.astype(np.float64) ** 2))
    assert 0.5 < ratio < 2.0
    np.testing.assert_array_equal(out.ch0, fn(ex, None).ch0)


@needs_ffmpeg
@pytest.mark.parametrize("ref", ["noise_snr10@1", "pitch_up1st@1", "timestretch_0_9x@1", "opus_16kbps@1"])
def test_no_mutan_la_entrada_y_la_salida_es_de_solo_lectura(ref: str):
    from altur.registry import transforms

    ex = _ex(_tone(440))
    before = ex.ch0.copy()
    out = transforms.resolve(ref)(ex, np.random.default_rng(0))
    assert out is not ex
    np.testing.assert_array_equal(ex.ch0, before)
    assert out.ch0.flags.writeable is False


def test_la_imagen_de_detect_no_importa_transforms_v2():
    from altur import serving

    assert "transforms_v2" not in Path(serving.__file__).read_text()
