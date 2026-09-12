"""Regla de desacuerdo (A3.3) y ablación OOF con incertidumbre.

Lo que estos tests protegen: que la regla haga lo que dice —conservar la decisión del
backbone y bajar la confianza— y que ningún número de ablación pueda salir sin intervalo.
"""

import numpy as np
import pytest

from altur.ablation import auc, bootstrap_ci, brier, format_table
from altur.crossfit import DisagreementFusion, LogisticFusion
from altur.models import ModelError

RAMAS = ("acoustic", "behavioral")


def _entrenada(**kw):
    rng = np.random.default_rng(0)
    y = np.r_[np.zeros(200, int), np.ones(200, int)]
    S = np.c_[
        np.r_[rng.beta(2, 6, 200), rng.beta(6, 2, 200)],
        np.r_[rng.beta(3, 4, 200), rng.beta(4, 3, 200)],
    ]
    return DisagreementFusion(**kw).fit(S, y, RAMAS), S, y


def test_cuando_las_ramas_concuerdan_la_regla_no_interviene():
    f, _, _ = _entrenada()
    acuerdo = np.array([[0.9, 0.88], [0.1, 0.12]])
    assert np.allclose(f.transform(acuerdo), f._inner.transform(acuerdo))


def test_cuando_discrepan_gana_el_backbone_y_baja_la_confianza():
    f, _, _ = _entrenada(shrink=0.5)
    # El backbone dice sintético con fuerza; el booster dice lo contrario.
    fila = np.array([[0.97, 0.03]])
    out = float(f.transform(fila)[0])
    assert out > 0.5, "la decisión tiene que seguir siendo la del backbone"
    assert out < 0.97, "y la confianza tiene que bajar"
    assert out == pytest.approx(0.5 + 0.5 * (0.97 - 0.5))


def test_shrink_cero_lleva_la_confianza_a_la_duda_total():
    f, _, _ = _entrenada(shrink=0.0)
    assert float(f.transform(np.array([[0.99, 0.01]]))[0]) == pytest.approx(0.5)


def test_el_umbral_de_desacuerdo_se_aprende_no_se_inventa():
    f, S, _ = _entrenada(quantile=0.90)
    esperado = np.quantile(np.abs(S[:, 1] - S[:, 0]), 0.90)
    assert f.delta_ == pytest.approx(esperado)
    # Un cuantil más alto interviene menos.
    g, _, _ = _entrenada(quantile=0.99)
    assert g.delta_ > f.delta_


def test_la_serializacion_declara_que_shrink_no_se_optimiza():
    f, _, _ = _entrenada()
    d = f.to_dict()
    assert d["backbone"] == "acoustic"
    assert "cuantil" in d["delta_source"]
    assert d["shrink_source"] == "declarado, no optimizado"
    assert d["inner"]["branches"] == list(RAMAS)


def test_backbone_inexistente_o_una_sola_rama_fallan():
    rng = np.random.default_rng(1)
    y = np.r_[np.zeros(20, int), np.ones(20, int)]
    S = rng.random((40, 2))
    with pytest.raises(ModelError, match="no está entre las ramas"):
        DisagreementFusion(backbone="prosodic").fit(S, y, RAMAS)
    with pytest.raises(ModelError, match="al menos dos ramas"):
        DisagreementFusion(backbone="acoustic").fit(S[:, :1], y, ("acoustic",))


@pytest.mark.parametrize("kw", [{"quantile": 0.0}, {"quantile": 1.0}, {"shrink": -0.1}, {"shrink": 1.5}])
def test_parametros_fuera_de_rango(kw):
    with pytest.raises(ModelError):
        DisagreementFusion(**kw)


def test_fusion_no_ajustada_falla():
    with pytest.raises(ModelError, match="no está ajustada"):
        DisagreementFusion().transform(np.zeros((2, 2)))


# --- métricas e intervalos ----------------------------------------------------------------


def test_auc_y_brier_en_casos_conocidos():
    y = np.array([0, 0, 1, 1])
    assert auc(y, np.array([0.1, 0.2, 0.8, 0.9])) == pytest.approx(1.0)
    assert auc(y, np.array([0.9, 0.8, 0.2, 0.1])) == pytest.approx(0.0)
    assert auc(y, np.array([0.5, 0.5, 0.5, 0.5])) == pytest.approx(0.5)
    assert brier(y, np.array([0.0, 0.0, 1.0, 1.0])) == pytest.approx(0.0)


def test_el_bootstrap_remuestrea_unidades_no_filas():
    """Con todas las filas en UNA unidad, el remuestreo no puede variar nada."""
    rng = np.random.default_rng(2)
    y = np.r_[np.zeros(50, int), np.ones(50, int)]
    s = np.r_[rng.beta(2, 5, 50), rng.beta(5, 2, 50)]
    lo, hi = bootstrap_ci(y, s, ["g0"] * 100, n_boot=200)
    assert lo == pytest.approx(hi), "una sola unidad no deja variabilidad que muestrear"

    lo2, hi2 = bootstrap_ci(y, s, [f"g{i}" for i in range(100)], n_boot=400)
    assert lo2 < hi2


def test_el_intervalo_contiene_al_estimador_puntual():
    rng = np.random.default_rng(3)
    y = np.r_[np.zeros(80, int), np.ones(80, int)]
    s = np.r_[rng.beta(2, 5, 80), rng.beta(5, 2, 80)]
    punto = auc(y, s)
    lo, hi = bootstrap_ci(y, s, [f"g{i}" for i in range(160)], n_boot=500)
    assert lo <= punto <= hi


def test_el_bootstrap_es_reproducible():
    rng = np.random.default_rng(4)
    y = np.r_[np.zeros(40, int), np.ones(40, int)]
    s = rng.random(80)
    u = [f"g{i}" for i in range(80)]
    assert bootstrap_ci(y, s, u, n_boot=200) == bootstrap_ci(y, s, u, n_boot=200)


def test_la_tabla_siempre_lleva_el_intervalo_pegado_al_numero():
    from altur.ablation import ScenarioResult

    r = ScenarioResult(
        name="solo_acoustic", branches=["acoustic"], fusion="logistic", n=282,
        auc=0.9777, auc_ci95=(0.961, 0.9902), brier=0.0535, brier_ci95=(0.03, 0.08),
        threshold_mean=0.488,
    )
    texto = format_table([r])
    assert "0.9777" in texto and "[0.9610, 0.9902]" in texto
    assert "IC95" in texto.splitlines()[0]


def test_fusion_logistica_con_una_rama_es_monotona():
    rng = np.random.default_rng(5)
    y = np.r_[np.zeros(60, int), np.ones(60, int)]
    S = np.r_[rng.beta(2, 5, 60), rng.beta(5, 2, 60)].reshape(-1, 1)
    f = LogisticFusion().fit(S, y, ("acoustic",))
    out = f.transform(S)
    assert np.array_equal(np.argsort(out.ravel()), np.argsort(S.ravel()))
