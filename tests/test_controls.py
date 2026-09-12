"""Controles de confound de A3.6.

Lo que estos tests protegen: que la familia de hipótesis no se pueda ampliar después de
mirar los datos, que ubicación y escala se prueben juntas, y que un control positivo diga
qué claims invalida en vez de esconderse en un número.
"""

import numpy as np
import pytest

from altur.controls import (
    Hypothesis,
    HypothesisFamily,
    benjamini_hochberg,
    format_controls,
    median_distance_auc,
    run_controls,
)
from altur.models import ModelError


def _familia():
    return HypothesisFamily(
        name="test_v1",
        hypotheses=(
            Hypothesis("ch1", "¿separa el canal del agente?", ("El eje mide la voz.",)),
            Hypothesis("silencio", "¿separa el silencio?", ("El eje mide fisiología.",)),
        ),
    )


# --- la familia se declara antes ----------------------------------------------------------


def test_una_hipotesis_nueva_despues_de_mirar_los_datos_es_error():
    """Corregir sobre las pruebas que sobrevivieron a la mirada es el sesgo que BH evita."""
    with pytest.raises(ModelError, match="no está en la familia declarada"):
        _familia().get("la_que_se_me_ocurrio_al_ver_el_grafico")


def test_una_observacion_que_falta_no_se_omite_en_silencio():
    rng = np.random.default_rng(0)
    y = np.r_[np.zeros(30, int), np.ones(30, int)]
    with pytest.raises(ModelError, match="no llegaron observaciones"):
        run_controls(_familia(), {"ch1": (y, rng.random(60))}, n_boot=20)


def test_familia_vacia_o_con_repetidas():
    with pytest.raises(ModelError, match="familia vacía"):
        HypothesisFamily(name="x", hypotheses=())
    h = Hypothesis("a", "?", ())
    with pytest.raises(ModelError, match="repetidas"):
        HypothesisFamily(name="x", hypotheses=(h, h))


def test_la_familia_registra_cuando_se_declaro():
    assert _familia().declared_at.endswith("+00:00")


# --- ubicación y escala ------------------------------------------------------------------


def test_el_test_de_escala_ve_lo_que_el_de_ubicacion_no():
    """El punto ciego de CP0: misma mediana, dispersión muy distinta."""
    rng = np.random.default_rng(1)
    concentrada = np.full(200, 0.5) + rng.normal(0, 0.001, 200)
    dispersa = 0.5 + rng.normal(0, 0.3, 200)
    y = np.r_[np.ones(200, int), np.zeros(200, int)]
    x = np.r_[concentrada, dispersa]

    from altur.controls import auc_score

    assert abs(auc_score(y, x) - 0.5) < 0.10, "ubicación no debería ver nada"
    assert abs(median_distance_auc(y, x) - 0.5) > 0.35, "escala sí tiene que verlo"


def test_auc_de_distancia_a_la_mediana_en_un_caso_sin_diferencia():
    rng = np.random.default_rng(2)
    x = rng.normal(0, 1, 400)
    y = np.r_[np.zeros(200, int), np.ones(200, int)]
    assert abs(median_distance_auc(y, x) - 0.5) < 0.15


# --- Benjamini-Hochberg ------------------------------------------------------------------


def test_bh_es_monotono_y_no_supera_uno():
    p = [0.001, 0.01, 0.03, 0.2, 0.9]
    q, _ = benjamini_hochberg(p)
    assert np.all(q <= 1.0)
    assert np.all(np.diff(q[np.argsort(p)]) >= -1e-12)
    assert np.all(q >= np.array(p) - 1e-12), "q nunca es menor que su p"


def test_bh_con_todo_nulo_no_rechaza_casi_nada():
    rng = np.random.default_rng(3)
    q, corte = benjamini_hochberg(rng.uniform(0, 1, 200))
    assert (q <= 0.05).sum() <= 5
    assert corte <= 0.05


def test_bh_es_menos_conservador_que_bonferroni():
    p = [0.001] * 5 + [0.5] * 15
    q, _ = benjamini_hochberg(p)
    assert (q <= 0.05).sum() == 5
    assert all(pi * 20 > 0.05 for pi in [0.004]), "Bonferroni rechazaría menos en el margen"


def test_p_fuera_de_rango():
    with pytest.raises(ModelError, match=r"\[0, 1\]"):
        benjamini_hochberg([0.5, 1.4])


def test_bh_vacio():
    q, corte = benjamini_hochberg([])
    assert q.size == 0 and corte == 0.0


# --- resultado y reporte -----------------------------------------------------------------


def _observaciones(separa_ch1: bool):
    rng = np.random.default_rng(4)
    y = np.r_[np.zeros(120, int), np.ones(120, int)]
    ruido = rng.normal(0, 1, 240)
    señal = np.r_[rng.normal(0, 1, 120), rng.normal(3.0, 1, 120)]
    return {"ch1": (y, señal if separa_ch1 else ruido), "silencio": (y, señal)}


def test_un_control_positivo_enumera_los_claims_que_invalida():
    res = run_controls(_familia(), _observaciones(separa_ch1=True), n_boot=50)
    por_nombre = {r.name: r for r in res}
    assert por_nombre["ch1"].positive
    assert por_nombre["ch1"].invalidated_claims == ["El eje mide la voz."]


def test_un_control_negativo_se_reporta_igual():
    """Un control que no separa es evidencia, no ausencia de resultado."""
    res = run_controls(_familia(), _observaciones(separa_ch1=False), n_boot=50)
    ch1 = next(r for r in res if r.name == "ch1")
    assert not ch1.positive
    assert ch1.invalidated_claims == []
    assert any("se reporta igual" in n.lower() for n in ch1.notes)
    assert np.isfinite(ch1.auc), "el AUC se reporta aunque el control salga negativo"


def test_todos_los_controles_de_la_familia_aparecen_en_el_reporte():
    res = run_controls(_familia(), _observaciones(separa_ch1=False), n_boot=50)
    assert {r.name for r in res} == set(_familia().names)


def test_un_positivo_sin_claims_se_marca_como_referencia_no_como_confound():
    familia = HypothesisFamily(
        name="ref_v1",
        hypotheses=(Hypothesis("referencia", "¿separa el habla?", ()),),
    )
    rng = np.random.default_rng(5)
    y = np.r_[np.zeros(100, int), np.ones(100, int)]
    s = np.r_[rng.normal(0, 1, 100), rng.normal(3, 1, 100)]
    res = run_controls(familia, {"referencia": (y, s)}, n_boot=50)
    assert res[0].positive and res[0].invalidated_claims == []
    texto = format_controls(res)
    assert "ref" in texto and "🔴" not in texto


def test_el_reporte_lleva_la_nota_de_grupos_conservadores():
    res = run_controls(_familia(), _observaciones(separa_ch1=False), n_boot=50)
    assert any("D-A1.2" in n for r in res for n in r.notes)


def test_longitudes_que_no_cuadran():
    y = np.r_[np.zeros(10, int), np.ones(10, int)]
    obs = {"ch1": (y, np.zeros(19)), "silencio": (y, np.zeros(20))}
    with pytest.raises(ModelError, match="no cuadran"):
        run_controls(_familia(), obs, n_boot=10)
