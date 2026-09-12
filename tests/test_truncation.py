"""Curva de truncación y perfil de latencia (A3.7)."""

import numpy as np
import pytest

from altur.models import ModelError
from altur.truncation import (
    HORIZONS_S,
    Timing,
    confidence,
    format_curve,
    measure_horizon,
    truncate,
)
from altur.types import AudioExample


def _ejemplo(segundos: float = 30.0, sr: int = 8000, seed: int = 0) -> AudioExample:
    rng = np.random.default_rng(seed)
    n = int(segundos * sr)
    return AudioExample(
        ch0=(rng.standard_normal(n) * 0.1).astype(np.float32),
        ch1=(rng.standard_normal(n) * 0.1).astype(np.float32),
    )


def test_truncar_corta_los_dos_canales_al_mismo_largo():
    ex = _ejemplo(30.0)
    corto = truncate(ex, 5.0)
    assert corto.duration_s == pytest.approx(5.0)
    assert corto.n_samples == 5 * ex.sr
    assert np.asarray(corto.ch1).shape == np.asarray(corto.ch0).shape


def test_truncar_conserva_el_prefijo_exacto():
    ex = _ejemplo(10.0)
    corto = truncate(ex, 3.0)
    assert np.array_equal(np.asarray(corto.ch0), np.asarray(ex.ch0)[: 3 * ex.sr])


def test_none_devuelve_la_llamada_completa():
    ex = _ejemplo(10.0)
    assert truncate(ex, None) is ex


def test_una_llamada_mas_corta_que_el_horizonte_entra_completa_sin_rellenar():
    """Rellenar con silencio inventaría audio que el sistema nunca tuvo."""
    ex = _ejemplo(4.0)
    salida = truncate(ex, 60.0)
    assert salida is ex
    assert salida.duration_s == pytest.approx(4.0)


@pytest.mark.parametrize("mal", [0.0, -1.0])
def test_horizonte_no_positivo(mal):
    with pytest.raises(ModelError, match="positivo"):
        truncate(_ejemplo(), mal)


def test_los_horizontes_declarados_terminan_en_full():
    assert HORIZONS_S[-1] is None
    numericos = [h for h in HORIZONS_S if h is not None]
    assert numericos == sorted(numericos), "los horizontes van de menor a mayor"


# --- confianza y tiempos -----------------------------------------------------------------


def test_la_confianza_es_distancia_al_umbral_neutro():
    c = confidence([0.5, 1.0, 0.0, 0.75])
    assert c[0] == pytest.approx(0.0)
    assert c[1] == pytest.approx(1.0)
    assert c[2] == pytest.approx(1.0)
    assert c[3] == pytest.approx(0.5)


def test_timing_resume_percentiles():
    t = Timing.from_samples("extract", [1.0, 2.0, 3.0, 100.0])
    assert t.n == 4
    assert t.max_ms == 100.0
    assert t.p50_ms < t.p95_ms <= t.max_ms


def test_timing_sin_muestras_no_revienta():
    t = Timing.from_samples("vacio", [])
    assert t.n == 0 and np.isnan(t.p50_ms)


def test_measure_horizon_separa_audio_consumido_de_tiempo_de_computo():
    """Son dos magnitudes distintas y una cifra única las haría ver igual."""
    ejemplos = {f"call_{i}": _ejemplo(30.0, seed=i) for i in range(4)}

    def extract_fn(ex):
        return {"f.dur": ex.duration_s}

    def predict_fn(filas):
        return np.array([min(r["f.dur"] / 60.0, 1.0) for r in filas])

    punto, por_id, scores = measure_horizon(ejemplos, 10.0, extract_fn, predict_fn)
    assert punto.audio_seconds_used_median == pytest.approx(10.0)
    assert punto.n == 4 and len(por_id) == 4 and scores.shape == (4,)
    assert punto.extract["p50_ms"] >= 0.0
    assert punto.total["p50_ms"] >= punto.model["p50_ms"]
    assert any("magnitudes distintas" in n for n in punto.notes)


def test_measure_horizon_cuenta_y_declara_las_llamadas_cortas():
    ejemplos = {"a": _ejemplo(5.0), "b": _ejemplo(90.0)}
    punto, _, _ = measure_horizon(
        ejemplos, 60.0, lambda ex: {"f.x": ex.duration_s}, lambda fs: np.zeros(len(fs))
    )
    assert punto.calls_shorter_than_horizon == 1
    assert any("no se rellenan" in n.lower() for n in punto.notes)


def test_la_tabla_muestra_horizonte_audio_y_tiempo_por_separado():
    ejemplos = {"a": _ejemplo(30.0)}
    punto, _, _ = measure_horizon(
        ejemplos, 5.0, lambda ex: {"f.x": 1.0}, lambda fs: np.array([0.9])
    )
    texto = format_curve([punto])
    cabecera = texto.splitlines()[0]
    assert "audio med" in cabecera and "extract p50" in cabecera
    assert "conf" in cabecera
