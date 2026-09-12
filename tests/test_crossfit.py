"""Cross-fitting anidado de A3.2.

El test que justifica el módulo es `test_espia_*`: comprueba, leyendo la auditoría de accesos,
que ninguna fila de `OUT_k` participa en el fit, la fusión, la calibración ni el umbral.
Lo demás — determinismo e idempotencia, y la caída bajo permutación — son las otras dos
condiciones de aceptación que pide `LONG_SCOPE` §7 para A3.2.
"""

import numpy as np
import pytest

from altur.crossfit import (
    Branch,
    FeatureTable,
    LogisticFusion,
    balanced_threshold,
    make_nested_fit_predict,
)
from altur.models import ModelError, logreg
from altur.models.calibration import IdentityCalibrator, PlattCalibrator

ACU = ("a2.acoustic.ch0.uno", "a2.acoustic.ch0.dos")
CON = ("behavioral.lat_med",)


def _corpus(n_por_clase: int = 40, seed: int = 7, señal: float = 1.4):
    rng = np.random.default_rng(seed)
    features, labels, ids = {}, {}, []
    for clase in (0, 1):
        for i in range(n_por_clase):
            cid = f"call_{clase}_{i:03d}"
            ids.append(cid)
            desp = señal if clase == 1 else 0.0
            features[cid] = {
                ACU[0]: float(rng.normal(desp, 1.0)),
                ACU[1]: float(rng.normal(desp * 0.5, 1.0)),
                CON[0]: float(rng.normal(desp * 0.3, 1.0)),
            }
            labels[cid] = clase
    return ids, features, labels


def _ramas():
    return [
        Branch("acoustic", ACU, logreg),
        Branch("behavioral", CON, logreg),
    ]


def _partir(ids):
    """Mitad y mitad, alternando clases para que ningún lado quede monoclase."""
    ceros = [i for i in ids if i.startswith("call_0")]
    unos = [i for i in ids if i.startswith("call_1")]
    tr = tuple(ceros[: len(ceros) // 2] + unos[: len(unos) // 2])
    te = tuple(ceros[len(ceros) // 2 :] + unos[len(unos) // 2 :])
    return tr, te


def _auc(y, s):
    y, s = np.asarray(y), np.asarray(s)
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    ss = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and ss[j + 1] == ss[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    n1 = float(y.sum())
    n0 = float(len(y) - n1)
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n0 * n1))


# --- el candado ---------------------------------------------------------------------------


def test_espia_ninguna_fila_externa_participa_en_el_ajuste():
    """A3.2: `OUT_k` no toca fit, fusión, calibración ni umbral."""
    ids, features, labels = _corpus()
    tabla = FeatureTable(features, labels, audit=True)
    tr, te = _partir(ids)
    g_in = np.array(tr)  # grupos conservadores: 1 llamada = 1 grupo (D-A1.2)

    fp = make_nested_fit_predict(tabla, _ramas(), calibrator_factory=PlattCalibrator)
    fp(tr, te, g_in)

    externos = set(te)
    for fase in ("inner", "refit"):
        fugas = tabla.ids_in_phase(fase) & externos
        assert not fugas, f"{len(fugas)} filas de OUT_k se leyeron en la fase {fase!r}"

    # Y la fase de predicción toca OUT_k y nada más.
    assert tabla.ids_in_phase("predict") == externos


def test_espia_la_fase_interna_solo_ve_el_fold_de_entrenamiento():
    ids, features, labels = _corpus()
    tabla = FeatureTable(features, labels, audit=True)
    tr, te = _partir(ids)
    fp = make_nested_fit_predict(tabla, _ramas(), calibrator_factory=PlattCalibrator)
    fp(tr, te, np.array(tr))
    assert tabla.ids_in_phase("inner") <= set(tr)


def test_el_umbral_sale_de_los_scores_internos_no_de_los_externos():
    """Si el umbral viniera de OUT_k, cambiar OUT_k lo movería. No debe moverse."""
    ids, features, labels = _corpus()
    tr, te = _partir(ids)
    fp = make_nested_fit_predict(
        FeatureTable(features, labels), _ramas(), calibrator_factory=PlattCalibrator
    )
    _, thr_a, _ = fp(tr, te, np.array(tr))

    # Mismo IN_k, OUT_k reducido a la mitad.
    _, thr_b, _ = fp(tr, te[::2], np.array(tr))
    assert thr_a == pytest.approx(thr_b)


# --- determinismo y permutación -----------------------------------------------------------


def test_repetir_con_la_misma_semilla_da_scores_identicos():
    ids, features, labels = _corpus()
    tr, te = _partir(ids)
    g_in = np.array(tr)
    fp = make_nested_fit_predict(
        FeatureTable(features, labels), _ramas(), calibrator_factory=PlattCalibrator
    )
    s1, t1, _ = fp(tr, te, g_in)
    s2, t2, _ = fp(tr, te, g_in)
    assert np.array_equal(s1, s2), "bit a bit, no solo allclose"
    assert t1 == t2


def test_con_señal_separa_y_con_etiquetas_permutadas_cae_hacia_azar():
    ids, features, labels = _corpus(n_por_clase=60, señal=2.0)
    tr, te = _partir(ids)
    g_in = np.array(tr)
    y_te = np.array([labels[i] for i in te])

    fp = make_nested_fit_predict(
        FeatureTable(features, labels), _ramas(), calibrator_factory=PlattCalibrator
    )
    real, _, _ = fp(tr, te, g_in)
    assert _auc(y_te, real) > 0.85

    rng = np.random.default_rng(0)
    revueltas = dict(zip(ids, rng.permutation([labels[i] for i in ids]), strict=True))
    fp_perm = make_nested_fit_predict(
        FeatureTable(features, revueltas), _ramas(), calibrator_factory=PlattCalibrator
    )
    perm, _, _ = fp_perm(tr, te, g_in)
    y_perm = np.array([revueltas[i] for i in te])
    assert 0.25 < _auc(y_perm, perm) < 0.75, "con etiquetas permutadas debería estar en azar"


# --- fusión y calibración -----------------------------------------------------------------


def test_una_sola_rama_no_cambia_el_ranking():
    """La fusión no debe inventar señal donde hay una sola fuente."""
    ids, features, labels = _corpus()
    tr, te = _partir(ids)
    tabla = FeatureTable(features, labels)
    fp = make_nested_fit_predict(
        tabla, [Branch("acoustic", ACU, logreg)], calibrator_factory=IdentityCalibrator
    )
    scores, _, _ = fp(tr, te, np.array(tr))

    m = logreg(ACU).fit(tabla.matrix(tr, ACU), tabla.y(tr))
    crudo = m.p_synthetic(tabla.matrix(te, ACU))
    assert np.array_equal(np.argsort(scores), np.argsort(crudo))


def test_la_auditoria_registra_de_que_se_ajusto_cada_pieza():
    ids, features, labels = _corpus()
    tr, te = _partir(ids)
    fp = make_nested_fit_predict(
        FeatureTable(features, labels), _ramas(), calibrator_factory=PlattCalibrator
    )
    _, _, extra = fp(tr, te, np.array(tr))
    audit = extra["audit"]
    assert audit.n_train == len(tr) and audit.n_test == len(te)
    assert set(audit.branch_inner_auc) == {"acoustic", "behavioral"}
    assert audit.fusion["branches"] == ["acoustic", "behavioral"]
    assert audit.calibrator["name"] == "platt@1"
    assert audit.threshold_source == "inner_oof_balanced_accuracy"


def test_platt_no_mueve_el_ranking_pero_si_las_probabilidades():
    rng = np.random.default_rng(3)
    y = np.r_[np.zeros(200, int), np.ones(200, int)]
    crudo = np.r_[rng.beta(2, 5, 200), rng.beta(5, 2, 200)]
    cal = PlattCalibrator().fit(crudo, y).transform(crudo)
    assert _auc(y, cal) == pytest.approx(_auc(y, crudo))
    assert not np.allclose(cal, crudo)


def test_platt_serializa_y_vuelve():
    rng = np.random.default_rng(4)
    y = np.r_[np.zeros(50, int), np.ones(50, int)]
    s = np.r_[rng.beta(2, 5, 50), rng.beta(5, 2, 50)]
    c = PlattCalibrator().fit(s, y)
    vuelto = PlattCalibrator.from_dict(c.to_dict())
    assert np.allclose(vuelto.transform(s), c.transform(s))


# --- umbral y validaciones ----------------------------------------------------------------


def test_umbral_balanceado_no_se_casa_con_la_prevalencia():
    """D-A1.5: train 59.9 % sintético, val 47.9 %. El umbral no puede depender de eso."""
    rng = np.random.default_rng(5)
    s_neg = rng.normal(0.3, 0.08, 500)
    s_pos = rng.normal(0.7, 0.08, 500)
    equilibrado = balanced_threshold(
        np.r_[s_neg, s_pos], np.r_[np.zeros(500, int), np.ones(500, int)]
    )
    # Mismas distribuciones, prevalencia muy distinta.
    sesgado = balanced_threshold(
        np.r_[s_neg, s_pos[:50]], np.r_[np.zeros(500, int), np.ones(50, int)]
    )
    assert abs(equilibrado - sesgado) < 0.08


def test_ramas_con_nombre_repetido_fallan():
    with pytest.raises(ModelError, match="repetidos"):
        make_nested_fit_predict(
            FeatureTable({}, {}),
            [Branch("a", ACU, logreg), Branch("a", CON, logreg)],
            calibrator_factory=PlattCalibrator,
        )


def test_sin_ramas_falla():
    with pytest.raises(ModelError, match="al menos una rama"):
        make_nested_fit_predict(
            FeatureTable({}, {}), [], calibrator_factory=PlattCalibrator
        )


def test_feature_faltante_falla_en_vez_de_imputar():
    ids, features, labels = _corpus(n_por_clase=6)
    del features[ids[0]][ACU[1]]
    with pytest.raises(ModelError, match="le falta la feature"):
        FeatureTable(features, labels).matrix(tuple(ids), ACU)


def test_id_sin_etiqueta_falla_al_construir_la_tabla():
    ids, features, labels = _corpus(n_por_clase=4)
    del labels[ids[0]]
    with pytest.raises(ModelError, match="sin etiqueta"):
        FeatureTable(features, labels)


def test_nan_en_la_tabla_falla():
    ids, features, labels = _corpus(n_por_clase=4)
    features[ids[0]][ACU[0]] = float("nan")
    with pytest.raises(ModelError, match="NaN o inf"):
        FeatureTable(features, labels).matrix(tuple(ids), ACU)


def test_fusion_sin_ajustar_falla():
    with pytest.raises(ModelError, match="no está ajustada"):
        LogisticFusion().transform(np.zeros((2, 2)))
