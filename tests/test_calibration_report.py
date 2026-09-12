"""Bloque de calibración de A3.4.

El test que da sentido al módulo es `test_la_identidad_de_brier_no_se_cumple_con_bins`: el
plan v1 afirmaba `BS = REL − RES + UNC` como identidad exacta con 3 bins, y no lo es.
"""

import numpy as np
import pytest

from altur.calibration_report import (
    brier,
    build_report,
    decompose_brier,
    grouped_brier,
    log_loss,
    reliability,
    score_histogram,
    shift_prior,
    wilson_interval,
)
from altur.models import ModelError


def _datos(n=300, seed=0):
    rng = np.random.default_rng(seed)
    y = np.r_[np.zeros(n, int), np.ones(n, int)]
    s = np.r_[rng.beta(2, 5, n), rng.beta(5, 2, n)]
    return y, s


# --- descomposición ----------------------------------------------------------------------


def test_la_identidad_de_brier_no_se_cumple_con_bins():
    """La corrección al plan v1. El residual se calcula, no se asume cero."""
    y, s = _datos()
    d = decompose_brier(y, s, n_bins=3)
    assert not d.identity_holds
    assert abs(d.residual) > 1e-4
    # Y el residual es exactamente lo que falta para cerrar la cuenta.
    assert d.brier == pytest.approx(d.reliability - d.resolution + d.uncertainty + d.residual)
    # No es un detalle numérico: aquí pesa más del 10 % del Brier.
    assert abs(d.residual) / d.brier > 0.10


def test_con_pronosticos_constantes_por_bin_la_identidad_si_se_cumple():
    """La teoría dice que el residual nace de la dispersión DENTRO del bin. Sin dispersión, cero."""
    y, s = _datos()
    constante = np.where(s > np.median(s), 0.8, 0.2)
    d = decompose_brier(y, constante, n_bins=2)
    assert d.identity_holds
    assert abs(d.residual) < 1e-12


def test_uncertainty_solo_depende_de_la_prevalencia():
    y = np.r_[np.zeros(70, int), np.ones(30, int)]
    rng = np.random.default_rng(1)
    d = decompose_brier(y, rng.random(100), n_bins=3)
    assert d.uncertainty == pytest.approx(0.3 * 0.7)


def test_un_modelo_perfecto_tiene_reliability_cero_y_resolution_maxima():
    y = np.r_[np.zeros(50, int), np.ones(50, int)]
    s = y.astype(float)
    d = decompose_brier(y, s, n_bins=2)
    assert d.brier == pytest.approx(0.0)
    assert d.reliability == pytest.approx(0.0)
    assert d.resolution == pytest.approx(d.uncertainty)


def test_la_serializacion_lleva_la_advertencia_del_residual():
    y, s = _datos(n=50)
    d = decompose_brier(y, s).to_dict()
    assert "residual" in d and "identity_holds" in d
    assert "NO es cero" in d["note"]


# --- Wilson y reliability ----------------------------------------------------------------


def test_wilson_es_asimetrico_con_n_chico_y_contiene_la_proporcion():
    lo, hi = wilson_interval(3, 10)
    assert lo < 0.3 < hi
    assert (0.3 - lo) != pytest.approx(hi - 0.3), "con n=10 el intervalo no es simétrico"
    # Con n grande se acerca a simétrico y se estrecha.
    lo2, hi2 = wilson_interval(300, 1000)
    assert (hi2 - lo2) < (hi - lo)


@pytest.mark.parametrize("k,n", [(0, 10), (10, 10), (0, 1)])
def test_wilson_en_los_extremos_no_se_sale_de_cero_uno(k, n):
    lo, hi = wilson_interval(k, n)
    assert 0.0 <= lo <= hi <= 1.0


def test_wilson_sin_datos_es_nan():
    lo, hi = wilson_interval(0, 0)
    assert np.isnan(lo) and np.isnan(hi)


def test_reliability_cubre_todas_las_filas_y_trae_intervalo():
    y, s = _datos(n=100)
    bins = reliability(y, s, n_bins=3)
    assert sum(b.n for b in bins) == len(y)
    for b in bins:
        lo, hi = b.wilson95
        assert lo <= b.empirical_rate <= hi
        assert b.lo <= b.mean_score <= b.hi


def test_el_histograma_acompaña_a_la_reliability():
    """Sin él, un bin con 3 casos y uno con 200 se ven igual de convincentes."""
    _, s = _datos(n=100)
    edges, counts = score_histogram(s, n_bins=20)
    assert len(edges) == 21 and len(counts) == 20
    assert sum(counts) == len(s)


# --- prior shift -------------------------------------------------------------------------


def test_mover_el_prior_al_mismo_valor_no_cambia_nada():
    _, s = _datos(n=50)
    assert np.allclose(shift_prior(s, prior_from=0.6, prior_to=0.6), s)


def test_bajar_el_prior_baja_todos_los_scores():
    _, s = _datos(n=50)
    bajado = shift_prior(s, prior_from=0.599, prior_to=0.3)
    assert np.all(bajado < s)
    subido = shift_prior(s, prior_from=0.599, prior_to=0.8)
    assert np.all(subido > s)


def test_el_prior_shift_es_monotono_y_no_mueve_el_ranking():
    _, s = _datos(n=80)
    movido = shift_prior(s, prior_from=0.599, prior_to=0.479)
    assert np.array_equal(np.argsort(s), np.argsort(movido))


def test_prior_shift_es_reversible():
    _, s = _datos(n=40)
    ida = shift_prior(s, prior_from=0.599, prior_to=0.3)
    vuelta = shift_prior(ida, prior_from=0.3, prior_to=0.599)
    assert np.allclose(vuelta, s, atol=1e-12)


@pytest.mark.parametrize("kw", [{"prior_from": 0.0}, {"prior_to": 1.0}, {"prior_from": -0.1}])
def test_priors_fuera_de_rango(kw):
    args = {"prior_from": 0.5, "prior_to": 0.5} | kw
    with pytest.raises(ModelError, match="tiene que estar en"):
        shift_prior([0.5], **args)


# --- Brier agrupado y reporte ------------------------------------------------------------


def test_brier_agrupado_coincide_con_el_original_si_cada_fila_es_su_grupo():
    """Y que coincidan no es una validación: es el síntoma de no tener llave de unión."""
    y, s = _datos(n=40)
    unidades = [f"g{i}" for i in range(len(y))]
    assert grouped_brier(y, s, unidades) == pytest.approx(brier(y, s))


def test_brier_agrupado_pesa_las_unidades_no_las_filas():
    y = np.array([0, 0, 0, 1])
    s = np.array([0.0, 0.0, 0.0, 0.0])  # perfecto en g0, pésimo en g1
    unidades = ["g0", "g0", "g0", "g1"]
    assert brier(y, s) == pytest.approx(0.25)
    assert grouped_brier(y, s, unidades) == pytest.approx(0.5)


def test_log_loss_castiga_la_confianza_equivocada():
    assert log_loss([1], [0.99]) < log_loss([1], [0.5]) < log_loss([1], [0.01])
    assert np.isfinite(log_loss([1], [0.0])), "un 0.0 no puede dar infinito"


def test_el_reporte_completo_declara_sus_limitaciones():
    y, s = _datos(n=120)
    unidades = [f"g{i}" for i in range(len(y))]
    r = build_report(y, s, unidades, conservative_groups=True)
    assert r.n == len(y)
    assert r.observed_prior == pytest.approx(0.5)
    assert len(r.prior_map) == 6
    assert {p["prior"] for p in r.prior_map} >= {0.479, 0.5, 0.599}
    texto = " ".join(r.notes)
    assert "D-A1.2" in texto
    assert "residual" in texto
    assert "no es transportable" in texto.lower() or "No se afirma" in texto


def test_scores_fuera_de_cero_uno_se_rechazan():
    with pytest.raises(ModelError, match="probabilidades"):
        brier([0, 1], [0.5, 1.4])


def test_longitudes_distintas_fallan():
    with pytest.raises(ModelError, match="scores"):
        brier([0, 1, 1], [0.5, 0.5])
