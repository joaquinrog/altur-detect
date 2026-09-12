"""Contrato de los modelos de A3.1.

Lo que estos tests protegen, en orden de cuánto dolería que se rompiera:

1. **La clase positiva.** Un modelo invertido produce un AUC que se lee igual de bien.
2. **El orden de features.** Un vector reordenado da un número plausible y falso.
3. **El aislamiento del preprocesamiento.** Si el escalado se ajusta fuera del fold, todo el
   cross-fitting anidado es decorativo.
4. **Que el camino de inferencia no necesite sklearn.**
"""

import json
import subprocess
import sys

import numpy as np
import pytest

from altur.models import (
    ConstantModel,
    LinearExport,
    ModelError,
    SklearnAdapter,
    ThresholdStump,
    acoustic_baseline,
    logreg,
    logreg_factory,
)

ORDER = ("f.a", "f.b", "f.c")


def _blobs(n: int = 60, seed: int = 0, shift: float = 1.5):
    rng = np.random.default_rng(seed)
    X = np.vstack([rng.normal(0.0, 1.0, (n, 3)), rng.normal(shift, 1.0, (n, 3))])
    y = np.r_[np.zeros(n, dtype=int), np.ones(n, dtype=int)]
    return X, y


def _auc(y: np.ndarray, s: np.ndarray) -> float:
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=np.float64)
    sorted_s = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    n1 = int(y.sum())
    n0 = len(y) - n1
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n0 * n1))


# --- 1. clase positiva -------------------------------------------------------------------


def test_synthetic_es_la_clase_positiva():
    X, y = _blobs()
    m = logreg(ORDER).fit(X, y)
    p = m.p_synthetic(X)
    assert p[y == 1].mean() > p[y == 0].mean()
    assert _auc(y, p) > 0.9


class _InvertedEstimator:
    """Estimador cuyo `classes_` está al revés. sklearn puede hacerlo; hay que sobrevivirlo."""

    def fit(self, X, y):
        self.classes_ = np.array([1, 0])
        self._mu = X[np.asarray(y) == 1].mean(axis=0)
        return self

    def predict_proba(self, X):
        d = np.linalg.norm(np.asarray(X) - self._mu, axis=1)
        p_pos = 1.0 / (1.0 + d)
        # columna 0 = clase 1, columna 1 = clase 0, según classes_
        return np.column_stack([p_pos, 1.0 - p_pos])


def test_columna_de_la_clase_positiva_se_localiza_no_se_supone():
    X, y = _blobs()
    m = SklearnAdapter("inv@1", _InvertedEstimator, feature_order=ORDER).fit(X, y)
    assert m._pos_col == 0
    p = m.p_synthetic(X)
    assert p[y == 1].mean() > p[y == 0].mean(), "el adaptador leyó la columna equivocada"
    assert np.allclose(m.predict_proba(X)[:, 1], p)


def test_classes_sin_la_clase_positiva_es_error():
    class _Rare:
        def fit(self, X, y):
            self.classes_ = np.array([2, 3])
            return self

        def predict_proba(self, X):  # pragma: no cover - no debe llegar
            raise AssertionError

    with pytest.raises(ModelError, match="clase positiva"):
        SklearnAdapter("rare@1", _Rare, feature_order=ORDER).fit(*_blobs())


@pytest.mark.parametrize(
    "bad, match",
    [
        (np.array([0, 1, 2]), "solo admite 0"),
        (np.zeros(6, dtype=int), "monoclase"),
        (np.array([True, False, True, False]), "bool"),
    ],
)
def test_etiquetas_invalidas(bad, match):
    X = np.zeros((len(bad), 3))
    with pytest.raises(ModelError, match=match):
        logreg(ORDER).fit(X, bad)


# --- 2. orden y dimensiones --------------------------------------------------------------


def test_numero_de_columnas_equivocado_falla():
    X, y = _blobs()
    m = logreg(ORDER).fit(X, y)
    with pytest.raises(ModelError, match="espera 3"):
        m.p_synthetic(np.zeros((5, 2)))


def test_feature_order_repetido_o_vacio():
    with pytest.raises(ModelError, match="repetidos"):
        logreg(("f.a", "f.a"))
    with pytest.raises(ModelError, match="vacío"):
        logreg(())


def test_orden_permutado_cambia_el_resultado():
    """Si permutar columnas no cambiara nada, el feature_order no sería un contrato."""
    X, y = _blobs()
    m = logreg(ORDER).fit(X, y)
    export = m.export_linear()
    fila = {"f.a": 2.0, "f.b": -1.0, "f.c": 0.5}
    directo = export.p_synthetic_from_mapping(fila)
    permutado = export.p_synthetic(np.array([[0.5, -1.0, 2.0]]))[0]
    assert not np.isclose(directo, permutado)


def test_mapping_rechaza_features_faltantes_y_de_mas():
    X, y = _blobs()
    export = logreg(ORDER).fit(X, y).export_linear()
    with pytest.raises(ModelError, match="faltan features"):
        export.p_synthetic_from_mapping({"f.a": 1.0, "f.b": 1.0})
    with pytest.raises(ModelError, match="no declaradas"):
        export.p_synthetic_from_mapping({"f.a": 1.0, "f.b": 1.0, "f.c": 1.0, "f.d": 1.0})


def test_nan_e_inf_se_rechazan():
    X, y = _blobs()
    m = logreg(ORDER).fit(X, y)
    for mal in (np.nan, np.inf):
        X_mal = np.zeros((2, 3))
        X_mal[0, 0] = mal
        with pytest.raises(ModelError, match="NaN o inf"):
            m.p_synthetic(X_mal)


# --- 3. export y save-load ---------------------------------------------------------------


def test_export_reproduce_al_adaptador():
    X, y = _blobs()
    m = logreg(ORDER).fit(X, y)
    assert np.allclose(m.export_linear().p_synthetic(X), m.p_synthetic(X))


def test_save_load_no_mueve_una_prediccion(tmp_path):
    X, y = _blobs()
    export = logreg(ORDER).fit(X, y).export_linear()
    ruta = export.save(tmp_path / "modelo.json")
    vuelto = LinearExport.load(ruta)
    assert vuelto.feature_order == export.feature_order
    assert np.array_equal(vuelto.p_synthetic(X), export.p_synthetic(X))


def test_export_corrupto_falla_al_cargar(tmp_path):
    X, y = _blobs()
    ruta = logreg(ORDER).fit(X, y).export_linear().save(tmp_path / "m.json")
    payload = json.loads(ruta.read_text())
    payload["coef"] = payload["coef"][:-1]
    ruta.write_text(json.dumps(payload))
    with pytest.raises(ModelError, match="shape"):
        LinearExport.load(ruta)

    payload["schema_version"] = 99
    ruta.write_text(json.dumps(payload))
    with pytest.raises(ModelError, match="schema_version"):
        LinearExport.load(ruta)


def test_modelo_no_lineal_no_se_puede_exportar():
    X, y = _blobs()
    m = SklearnAdapter("inv@1", _InvertedEstimator, feature_order=ORDER).fit(X, y)
    with pytest.raises(ModelError, match="no es lineal"):
        m.export_linear()


def test_export_de_un_modelo_invertido_conserva_la_direccion():
    """El signo se corrige en el export, no se arrastra al bundle."""

    class _LinearInverted:
        def fit(self, X, y):
            self.classes_ = np.array([1, 0])
            self.coef_ = np.array([[1.0, 0.0, 0.0]])
            self.intercept_ = np.array([0.0])
            return self

        def predict_proba(self, X):
            # Convención de sklearn: coef_ apunta a classes_[1], que aquí es `human`.
            z = np.asarray(X) @ self.coef_[0] + self.intercept_[0]
            p_de_classes_1 = 1.0 / (1.0 + np.exp(-z))
            return np.column_stack([1.0 - p_de_classes_1, p_de_classes_1])

    X, y = _blobs()
    m = SklearnAdapter("linv@1", _LinearInverted, feature_order=ORDER).fit(X, y)
    assert np.allclose(m.export_linear(verify_on=X).p_synthetic(X), m.p_synthetic(X))


def test_export_detecta_un_estimador_que_rompe_la_convencion_de_signo():
    """Sin verify_on el export confía en sklearn. Con verify_on, no confía en nadie."""

    class _ConvencionRota:
        def fit(self, X, y):
            self.classes_ = np.array([1, 0])
            self.coef_ = np.array([[1.0, 0.0, 0.0]])
            self.intercept_ = np.array([0.0])
            return self

        def predict_proba(self, X):
            z = np.asarray(X) @ self.coef_[0] + self.intercept_[0]
            p = 1.0 / (1.0 + np.exp(-z))
            return np.column_stack([p, 1.0 - p])  # al revés de lo que dice classes_

    X, y = _blobs()
    m = SklearnAdapter("roto@1", _ConvencionRota, feature_order=ORDER).fit(X, y)
    m.export_linear()  # sin verificar, pasa — y estaría invertido
    with pytest.raises(ModelError, match="no reproduce"):
        m.export_linear(verify_on=X)


def test_verify_on_vacio_es_error():
    X, y = _blobs()
    m = logreg(ORDER).fit(X, y)
    with pytest.raises(ModelError, match="vacío"):
        m.export_linear(verify_on=np.zeros((0, 3)))


# --- 4. aislamiento del preprocesamiento -------------------------------------------------


def test_el_escalado_se_ajusta_solo_con_lo_que_ve_fit():
    """El requisito de A3.1: ningún preprocesamiento se ajusta fuera de su fold."""
    X, y = _blobs(n=40, seed=3)
    tr = np.r_[np.arange(0, 30), np.arange(40, 70)]

    m = logreg(ORDER).fit(X[tr], y[tr])
    esperado_mean = X[tr].mean(axis=0)
    esperado_scale = X[tr].std(axis=0, ddof=0)
    assert np.allclose(m._mean, esperado_mean)
    assert np.allclose(m._scale, esperado_scale)

    # Reajustar con TODO cambia el escalado: prueba de que no venía de ahí.
    m_todo = logreg(ORDER).fit(X, y)
    assert not np.allclose(m_todo._mean, esperado_mean)


def test_feature_constante_en_el_fold_no_divide_por_cero():
    X, y = _blobs()
    X[:, 2] = 7.0
    m = logreg(ORDER).fit(X, y)
    assert np.isfinite(m.p_synthetic(X)).all()
    assert m._scale[2] == 1.0


def test_predecir_sin_entrenar_falla():
    with pytest.raises(ModelError, match="no está entrenado"):
        logreg(ORDER).p_synthetic(np.zeros((2, 3)))


# --- 5. baselines ------------------------------------------------------------------------


def test_constante_da_auc_exactamente_medio():
    X, y = _blobs()
    m = ConstantModel(ORDER).fit(X, y)
    p = m.p_synthetic(X)
    assert np.unique(p).size == 1
    assert m.p_ == pytest.approx(y.mean())
    assert _auc(y, p) == pytest.approx(0.5)


def test_stump_tiene_exactamente_dos_niveles_y_aprende_la_direccion():
    X, y = _blobs()
    m = ThresholdStump(ORDER, "f.a").fit(X, y)
    p = m.p_synthetic(X)
    assert np.unique(p).size == 2
    assert _auc(y, p) > 0.6
    assert 0.0 < min(m.p_low_, m.p_high_) and max(m.p_low_, m.p_high_) < 1.0

    # Con la señal invertida el stump sigue separando: la dirección se aprende.
    m2 = ThresholdStump(ORDER, "f.a").fit(-X, y)
    assert _auc(y, m2.p_synthetic(-X)) > 0.6


def test_stump_sobre_feature_inexistente_falla():
    with pytest.raises(ModelError, match="no está en feature_order"):
        ThresholdStump(ORDER, "f.z")


def test_baseline_acustico_selecciona_solo_su_familia():
    order = ("a2.acoustic.ch0.uno", "a2.acoustic.ch0.dos", "behavioral.lat_med")
    m = acoustic_baseline(order)
    assert m.feature_order == ("a2.acoustic.ch0.uno", "a2.acoustic.ch0.dos")
    with pytest.raises(ModelError, match="ninguna feature"):
        acoustic_baseline(("behavioral.lat_med",))


# --- 6. la frontera de dependencias -------------------------------------------------------


def test_el_camino_de_inferencia_no_importa_sklearn():
    """`/detect` corre en una imagen con cinco dependencias. base.py no puede romper eso."""
    codigo = (
        "import sys;"
        "import altur.models.base, altur.models.baselines;"
        "mods=[m for m in sys.modules if m.split('.')[0] in {'sklearn','scipy','pandas'}];"
        "print(mods)"
    )
    out = subprocess.run(
        [sys.executable, "-c", codigo], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "[]", f"el camino de inferencia arrastró: {out.stdout}"


def test_logreg_factory_produce_estimadores_independientes():
    make = logreg_factory()
    a, b = make(), make()
    assert a is not b
