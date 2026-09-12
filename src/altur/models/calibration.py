"""Calibradores. Platt es el productivo; isotónica queda como comparador research-only.

Un calibrador se ajusta **solo** con scores internos OOF, nunca con el fold externo. Eso no
es una convención de estilo: `CalibratedClassifierCV` de sklearn no acepta `groups` — usa
`StratifiedKFold` interno y mete fuga de hablante en la calibración. Por eso el arnés
calibra a mano, con los splits agrupados de `protocol.grouped_inner_splits`.
"""

from typing import Any

import numpy as np

from altur.models.base import ModelError, _sigmoid, check_labels
from altur.registry import calibrators

_EPS = 1e-12


def _as_scores(s: Any) -> np.ndarray:
    arr = np.asarray(s, dtype=np.float64).ravel()
    if arr.size == 0:
        raise ModelError("no hay scores que calibrar")
    if not np.isfinite(arr).all():
        raise ModelError("los scores contienen NaN o inf")
    return arr


def _logit(p: np.ndarray) -> np.ndarray:
    q = np.clip(p, _EPS, 1.0 - _EPS)
    return np.log(q / (1.0 - q))


class PlattCalibrator:
    """Sigmoide de dos parámetros sobre el logit del score.

    Se ajusta con Newton amortiguado sobre la log-verosimilitud, en NumPy. No es por
    purismo: el calibrador **entra al bundle**, y el bundle no puede arrastrar sklearn.

    Usa los targets suavizados de Platt (1999), que evitan que un fold separable mande los
    parámetros a infinito.
    """

    name = "platt@1"

    def __init__(self, max_iter: int = 100, tol: float = 1e-10) -> None:
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.a_: float | None = None
        self.b_: float | None = None

    def fit(self, scores: Any, y: Any) -> "PlattCalibrator":
        s = _as_scores(scores)
        labels = check_labels(y)
        if s.shape[0] != labels.shape[0]:
            raise ModelError(f"hay {s.shape[0]} scores y {labels.shape[0]} etiquetas")
        z = _logit(s)

        n_pos = float(labels.sum())
        n_neg = float(labels.size - n_pos)
        hi = (n_pos + 1.0) / (n_pos + 2.0)
        lo = 1.0 / (n_neg + 2.0)
        t = np.where(labels == 1, hi, lo)

        a, b = 1.0, 0.0
        for _ in range(self.max_iter):
            p = _sigmoid(a * z + b)
            g_a = float(np.sum((p - t) * z))
            g_b = float(np.sum(p - t))
            w = p * (1.0 - p)
            h_aa = float(np.sum(w * z * z)) + 1e-10
            h_ab = float(np.sum(w * z))
            h_bb = float(np.sum(w)) + 1e-10
            det = h_aa * h_bb - h_ab * h_ab
            if abs(det) < 1e-15:
                break
            d_a = (h_bb * g_a - h_ab * g_b) / det
            d_b = (h_aa * g_b - h_ab * g_a) / det
            a, b = a - d_a, b - d_b
            if max(abs(d_a), abs(d_b)) < self.tol:
                break

        if not (np.isfinite(a) and np.isfinite(b)):
            raise ModelError("Platt divergió; revisa si el fold interno es separable")
        self.a_, self.b_ = float(a), float(b)
        return self

    def transform(self, scores: Any) -> np.ndarray:
        if self.a_ is None:
            raise ModelError("el calibrador no está ajustado")
        return _sigmoid(self.a_ * _logit(_as_scores(scores)) + self.b_)

    def to_dict(self) -> dict[str, Any]:
        if self.a_ is None:
            raise ModelError("el calibrador no está ajustado")
        return {"name": self.name, "a": self.a_, "b": self.b_}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PlattCalibrator":
        c = cls()
        c.a_, c.b_ = float(payload["a"]), float(payload["b"])
        return c


class IdentityCalibrator:
    """No calibra. Es la referencia contra la que se mide si Platt aporta algo."""

    name = "identity@1"

    def fit(self, scores: Any, y: Any) -> "IdentityCalibrator":
        _as_scores(scores)
        check_labels(y)
        return self

    def transform(self, scores: Any) -> np.ndarray:
        return _as_scores(scores)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "IdentityCalibrator":
        return cls()


@calibrators.register("platt", version=1)
def platt(**kwargs: Any) -> PlattCalibrator:
    """Calibrador por defecto. Product-safe: NumPy puro, entra al bundle."""
    return PlattCalibrator(**kwargs)


@calibrators.register("identity", version=1)
def identity(**_: Any) -> IdentityCalibrator:
    """Sin calibración. Referencia."""
    return IdentityCalibrator()
