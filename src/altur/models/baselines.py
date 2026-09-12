"""Baselines. NumPy puro — no importan sklearn.

No son candidatos a ganar: son **la prueba de que el arnés funciona**. Si el cross-fitting
está bien construido, el baseline constante tiene que dar AUC 0.5 exacto y el stump tiene que
dar algo mediocre. Un baseline que sale demasiado bien es un bug del protocolo, no una sorpresa
agradable.

Los baselines "acústico" y "conductual" de `LONG_SCOPE` A3.1 no son clases: son
`logreg(feature_order=...)` sobre la familia correspondiente. Ver `acoustic_baseline()` y
`behavioral_baseline()` en `__init__.py`.
"""

from typing import Any

import numpy as np

from altur.models.base import ModelError, _as_2d, check_feature_order, check_labels
from altur.registry import models


class ConstantModel:
    """Devuelve la prevalencia de `train` para todas las filas.

    Es el piso: AUC exactamente 0.5, y un Brier que equivale a no saber nada. Su valor está en
    el bloque de calibración — separa "el modelo discrimina" de "el modelo acierta al umbral",
    que con prevalencia inestable entre splits (D-A1.5) no es lo mismo.
    """

    def __init__(self, feature_order: Any, name: str = "constant@1") -> None:
        self.name = name
        self.feature_order = check_feature_order(feature_order)
        self.p_: float | None = None

    def fit(self, X: Any, y: Any) -> "ConstantModel":
        labels = check_labels(y)
        _as_2d(X, n_features=len(self.feature_order))
        self.p_ = float(labels.mean())
        return self

    def p_synthetic(self, X: Any) -> np.ndarray:
        if self.p_ is None:
            raise ModelError(f"{self.name} no está entrenado; llama fit() primero")
        arr = _as_2d(X, n_features=len(self.feature_order))
        return np.full(arr.shape[0], self.p_, dtype=np.float64)


class ThresholdStump:
    """Un corte sobre UNA feature, con dirección y hojas aprendidas dentro del fold.

    El score no es el valor de la feature: son las dos tasas empíricas de positivos, una por
    hoja. Eso lo mantiene deliberadamente tosco — un stump tiene exactamente dos niveles de
    score y su AUC está acotada por eso. Es la comparación correcta para decir "el modelo
    completo aporta esto por encima de mirar una sola variable".
    """

    def __init__(self, feature_order: Any, feature: str, name: str | None = None) -> None:
        self.feature_order = check_feature_order(feature_order)
        if feature not in self.feature_order:
            raise ModelError(f"{feature!r} no está en feature_order")
        self.feature = feature
        self._col = self.feature_order.index(feature)
        self.name = name or f"stump[{feature}]@1"
        self.threshold_: float | None = None
        self.p_high_: float | None = None
        self.p_low_: float | None = None

    def fit(self, X: Any, y: Any) -> "ThresholdStump":
        arr = _as_2d(X, n_features=len(self.feature_order))
        labels = check_labels(y)
        if arr.shape[0] != labels.shape[0]:
            raise ModelError(f"X tiene {arr.shape[0]} filas y y tiene {labels.shape[0]}")
        v = arr[:, self._col]

        # Candidatos: puntos medios entre valores únicos consecutivos. Determinista.
        uniq = np.unique(v)
        if uniq.size < 2:
            self.threshold_ = float(uniq[0])
            self.p_high_ = self.p_low_ = float(labels.mean())
            return self
        cuts = (uniq[:-1] + uniq[1:]) / 2.0

        n_pos = float(labels.sum())
        n_neg = float(labels.size - n_pos)
        best_score, best_cut = -1.0, float(cuts[0])
        for cut in cuts:
            high = v > cut
            tp = float(labels[high].sum())
            fp = float(high.sum() - tp)
            # Exactitud balanceada: no la sesga la prevalencia, que aquí no es estable.
            bal = 0.5 * (tp / n_pos + (n_neg - fp) / n_neg)
            bal = max(bal, 1.0 - bal)  # la dirección se aprende, no se supone
            if bal > best_score:
                best_score, best_cut = bal, float(cut)

        self.threshold_ = best_cut
        high = v > best_cut
        # Laplace: una hoja pura en el fold daría 0.0 o 1.0 y el log loss se iría a infinito.
        self.p_high_ = float((labels[high].sum() + 1.0) / (high.sum() + 2.0)) if high.any() else 0.5
        low = ~high
        self.p_low_ = float((labels[low].sum() + 1.0) / (low.sum() + 2.0)) if low.any() else 0.5
        return self

    def p_synthetic(self, X: Any) -> np.ndarray:
        if self.threshold_ is None:
            raise ModelError(f"{self.name} no está entrenado; llama fit() primero")
        arr = _as_2d(X, n_features=len(self.feature_order))
        v = arr[:, self._col]
        return np.where(v > self.threshold_, self.p_high_, self.p_low_).astype(np.float64)


@models.register("constant", version=1)
def constant(feature_order: Any, **_: Any) -> ConstantModel:
    """Piso del protocolo: la prevalencia de train. AUC 0.5 por construcción."""
    return ConstantModel(feature_order)


@models.register("stump", version=1)
def stump(feature_order: Any, feature: str = "behavioral.lat_med", **_: Any) -> ThresholdStump:
    """Corte sobre una sola feature. Por defecto la latencia mediana del caller."""
    return ThresholdStump(feature_order, feature)
