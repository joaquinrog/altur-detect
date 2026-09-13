"""El port a NumPy de LFCC y del VAD del repo del modelo reproduce al original.

La referencia (`fixtures/spectral_factory_reference_golden.json`) se generó UNA vez con el
código original de `cd028c7` (librosa 1.0.0, SciPy 1.18.1) sobre el WAV dorado sintético. No
hay audio del dataset aquí.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from altur.features import spectral_factory_lfcc as lfcc
from altur.io import decode
from altur.registry import audit_product_safety, extractors, turn_sources
from altur.seg import spectral_factory_vad as sfvad
from altur.types import AudioExample

REF = Path(__file__).parent / "fixtures" / "spectral_factory_reference_golden.json"


@pytest.fixture(scope="module")
def ref() -> dict:
    return json.loads(REF.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def golden_example(golden_bytes) -> AudioExample:
    return decode(golden_bytes).example


def test_vad_reproduce_los_turnos_del_original(golden_example, ref):
    seg = sfvad.energy_adaptive(golden_example)
    assert [[t.start, t.end] for t in seg.by_channel(0)] == ref["turns_ch0"]
    assert seg.by_channel(1) == ()


def test_lfcc_reproduce_las_features_del_original(golden_example, ref):
    seg = sfvad.energy_adaptive(golden_example)
    ex = AudioExample(golden_example.ch0, golden_example.ch1, seg=seg)
    feats, diag = lfcc.extract(ex)
    esperado = np.array([ref["features"][k.removeprefix(f"{lfcc.NAMESPACE}.")] for k in feats])
    np.testing.assert_allclose(np.array(list(feats.values())), esperado, rtol=1e-5, atol=1e-6)
    assert not diag["vad_fallback_full_channel"] and not diag["insufficient_audio"]


def test_el_orden_es_el_del_modelo_medido(ref):
    assert list(lfcc.FEATURE_ORDER) == [f"{lfcc.NAMESPACE}.{k}" for k in ref["features"]]


@pytest.mark.parametrize("n_frames", [3, 4, 5, 9, 10, 57])
@pytest.mark.parametrize("order", [1, 2])
def test_delta_igual_a_savgol_de_scipy(n_frames, order):
    signal = pytest.importorskip("scipy.signal")
    width = lfcc.DELTA_WIDTH
    if width >= n_frames:
        width = max(3, n_frames - 1 if (n_frames - 1) % 2 == 1 else n_frames - 2)
    data = np.random.default_rng(n_frames).normal(size=(20, n_frames))
    esperado = signal.savgol_filter(data, width, polyorder=order, deriv=order, axis=-1, mode="interp")
    np.testing.assert_allclose(lfcc.delta(data, width, order), esperado, rtol=1e-9, atol=1e-9)


def test_registro_apto_para_producto():
    ref = "spectral_factory.lfcc@1"
    meta = extractors.meta(ref)
    assert meta["needs_seg"] and tuple(meta["channels"]) == (0,)
    assert audit_product_safety([ref]) == []
    assert turn_sources.meta("spectral_factory.energy_adaptive@1")["is_oracle"] is False


def test_sin_habla_usa_el_canal_completo_y_lo_declara():
    ruido = (np.random.default_rng(0).standard_normal(16000) * 0.01).astype(np.float32)
    ex = AudioExample(ruido, ruido.copy())
    ex = AudioExample(ex.ch0, ex.ch1, seg=sfvad.energy_adaptive(ex))
    feats, diag = lfcc.extract(ex)
    assert len(feats) == 120 and all(np.isfinite(list(feats.values())))
    assert diag["vad_fallback_full_channel"] is True


def test_audio_sin_un_frame_no_revienta():
    corto = np.zeros(100, dtype=np.float32)
    ex = AudioExample(corto, corto.copy())
    ex = AudioExample(ex.ch0, ex.ch1, seg=sfvad.energy_adaptive(ex))
    feats, diag = lfcc.extract(ex)
    assert diag["insufficient_audio"] is True
    assert set(feats.values()) == {0.0}
