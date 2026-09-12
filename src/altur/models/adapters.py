"""Adaptadores de entrenamiento. Importan sklearn; el camino de inferencia no.

El adaptador es **genérico a propósito**: envuelve cualquier estimador con `fit` y
`predict_proba`, no solo `LogisticRegression`. El modelo real lo construye la otra mitad del
equipo (ver `docs/MODEL_HANDOFF.md`) y tiene que entrar sin tocar el arnés.

Dos invariantes que este módulo verifica en vez de suponer:

1. **Cuál columna de `predict_proba` es P(synthetic).** sklearn ordena por `classes_`, y con
   etiquetas 0/1 la columna 1 suele ser la correcta — pero "suele" no es una garantía, y un
   modelo invertido produce un AUC que se lee igual de bien. Se localiza la columna del 1.
2. **El escalado se ajusta dentro del `fit`.** Quien llame `fit` por fold obtiene, sin
   esfuerzo, preprocesamiento aislado por fold (requisito de A3.1).
"""

from collections.abc import Callable
from typing import Any

import numpy as np

from altur.models.base import (
    LinearExport,
    ModelError,
    _as_2d,
    check_feature_order,
    check_labels,
)
from altur.registry import models


class SklearnAdapter:
    """Envuelve un estimador de sklearn y expone el contrato `Model` del arnés."""

    def __init__(
        self,
        name: str,
        factory: Callable[[], Any],
        *,
        feature_order: Any,
        standardize: bool = True,
    ) -> None:
        self.name = str(name)
        self._factory = factory
        self.feature_order = check_feature_order(feature_order)
        self.standardize = bool(standardize)
        self._est: Any = None
        self._pos_col: int | None = None
        self._mean: np.ndarray | None = None
        self._scale: np.ndarray | None = None

    @property
    def n_features(self) -> int:
        return len(self.feature_order)

    @property
    def is_fitted(self) -> bool:
        return self._est is not None

    def _standardizer(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if not self.standardize:
            n = X.shape[1]
            return np.zeros(n, dtype=np.float64), np.ones(n, dtype=np.float64)
        mean = X.mean(axis=0)
        scale = X.std(axis=0, ddof=0)
        # Una feature constante dentro del fold no aporta y dividir por su std es dividir por
        # cero. Se deja pasar con escala 1: el coeficiente que aprenda será irrelevante.
        scale = np.where(scale > 0.0, scale, 1.0)
        return mean.astype(np.float64), scale.astype(np.float64)

    def fit(self, X: Any, y: Any) -> "SklearnAdapter":
        arr = _as_2d(X, n_features=self.n_features)
        labels = check_labels(y)
        if arr.shape[0] != labels.shape[0]:
            raise ModelError(f"X tiene {arr.shape[0]} filas y y tiene {labels.shape[0]}")

        self._mean, self._scale = self._standardizer(arr)
        est = self._factory()
        est.fit((arr - self._mean) / self._scale, labels)

        classes = np.asarray(getattr(est, "classes_", np.array([0, 1])))
        hits = np.flatnonzero(classes == 1)
        if hits.size != 1:
            raise ModelError(
                f"no se pudo localizar la clase positiva en classes_={classes!r}. "
                "synthetic = 1 es la clase positiva (D-A1.1)."
            )
        self._pos_col = int(hits[0])
        self._est = est
        return self

    def _require_fitted(self) -> None:
        if self._est is None:
            raise ModelError(f"{self.name} no está entrenado; llama fit() primero")

    def p_synthetic(self, X: Any) -> np.ndarray:
        self._require_fitted()
        arr = _as_2d(X, n_features=self.n_features)
        proba = np.asarray(self._est.predict_proba((arr - self._mean) / self._scale))
        if proba.ndim != 2 or proba.shape[0] != arr.shape[0]:
            raise ModelError(f"predict_proba devolvió shape {proba.shape}, inesperado")
        return proba[:, self._pos_col].astype(np.float64)

    def predict_proba(self, X: Any) -> np.ndarray:
        """(n, 2) con la columna 1 = P(synthetic), pase lo que pase con `classes_`."""
        p = self.p_synthetic(X)
        return np.column_stack([1.0 - p, p])

    def export_linear(self, verify_on: Any = None) -> LinearExport:
        """Coeficientes + preprocesamiento para el bundle. Sin sklearn en el otro lado.

        El signo depende de una convención de sklearn: `coef_` apunta siempre a `classes_[1]`,
        así que `predict_proba[:, 1] = sigmoid(Xw + b)`. Con `classes_ = [1, 0]` la clase
        positiva queda en la columna 0 y hay que negar coeficientes e intercepto.

        Confiar en esa convención en silencio es justo el tipo de error que produce un número
        plausible e invertido. **Pasa `verify_on` con las filas del fold** y el export se
        compara contra el estimador real antes de salir. Sin `verify_on` la convención se
        asume, y eso queda declarado aquí.
        """
        self._require_fitted()
        coef = getattr(self._est, "coef_", None)
        intercept = getattr(self._est, "intercept_", None)
        if coef is None or intercept is None:
            raise ModelError(
                f"{self.name} no es lineal: no tiene coef_/intercept_. "
                "Solo los modelos lineales se pueden empaquetar sin meter sklearn a la imagen."
            )
        coef = np.asarray(coef, dtype=np.float64)
        intercept = np.asarray(intercept, dtype=np.float64)
        if coef.shape != (1, self.n_features) or intercept.shape != (1,):
            raise ModelError(
                f"coef_ {coef.shape} / intercept_ {intercept.shape} no corresponden a un "
                "clasificador binario lineal sobre este feature_order"
            )
        sign = 1.0 if self._pos_col == 1 else -1.0
        export = LinearExport(
            feature_order=self.feature_order,
            mean=self._mean,
            scale=self._scale,
            coef=sign * coef[0],
            intercept=sign * float(intercept[0]),
            name=self.name,
        )
        if verify_on is not None:
            rows = _as_2d(verify_on, n_features=self.n_features)
            if rows.shape[0] == 0:
                raise ModelError("verify_on está vacío; no verifica nada")
            esperado = self.p_synthetic(rows)
            obtenido = export.p_synthetic(rows)
            peor = float(np.max(np.abs(esperado - obtenido)))
            if peor > 1e-9:
                raise ModelError(
                    f"el export lineal no reproduce a {self.name}: max|delta| = {peor:.3e}. "
                    "El estimador no sigue la convención de signo de sklearn; empaquetarlo "
                    "daría predicciones invertidas."
                )
        return export


def logreg_factory(**kwargs: Any) -> Callable[[], Any]:
    def make() -> Any:
        from sklearn.linear_model import LogisticRegression

        params = {"max_iter": 2000, "C": 1.0, "solver": "lbfgs"} | kwargs
        return LogisticRegression(**params)

    return make


@models.register("logreg", version=1)
def logreg(feature_order: Any, **kwargs: Any) -> SklearnAdapter:
    """Regresión logística estandarizada. Lineal, así que se exporta a NumPy puro."""
    return SklearnAdapter("logreg@1", logreg_factory(**kwargs), feature_order=feature_order)
