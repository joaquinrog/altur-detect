"""Contrato de modelo y export lineal a NumPy puro.

Este módulo **no importa sklearn**, a propósito. `LinearExport` es lo que viaja dentro del
bundle y lo que `/detect` carga: coeficientes, centrado y escala, nada más. La imagen de
inferencia declara cinco dependencias (`pyproject.toml`) y ese mínimo es el argumento de
Feasibility del pitch — un `joblib` de sklearn lo rompería.

Los adaptadores que sí necesitan sklearn viven en `adapters.py` y solo se usan en
entrenamiento.
"""

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

SCHEMA_VERSION = 1


class ModelError(RuntimeError):
    """Contrato de modelo violado."""


@runtime_checkable
class Model(Protocol):
    """Lo que el cross-fitting necesita de un modelo. Deliberadamente pequeño."""

    name: str

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Model": ...

    def p_synthetic(self, X: np.ndarray) -> np.ndarray:
        """P(synthetic) por fila, en [0, 1]. `synthetic` = 1 es la clase positiva (D-A1.1)."""
        ...


def _as_2d(X: Any, *, n_features: int | None = None) -> np.ndarray:
    arr = np.asarray(X, dtype=np.float64)
    if arr.ndim != 2:
        raise ModelError(f"X debe ser 2-D (n_muestras, n_features); llegó ndim={arr.ndim}")
    if not np.isfinite(arr).all():
        raise ModelError("X contiene NaN o inf; nan_policy es reject en todo el arnés")
    if n_features is not None and arr.shape[1] != n_features:
        raise ModelError(
            f"X tiene {arr.shape[1]} columnas y el modelo espera {n_features}. "
            "Casi siempre es un feature_order que cambió entre corridas."
        )
    return arr


def _sigmoid(z: np.ndarray) -> np.ndarray:
    # Estable en ambas colas: evita exp() de un positivo grande.
    out = np.empty_like(z, dtype=np.float64)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


@dataclass(frozen=True, slots=True)
class LinearExport:
    """Un clasificador lineal calibrado, sin dependencias más allá de NumPy.

    `score = sigmoid(coef · (x - mean) / scale + intercept)` y ese score **es** P(synthetic).
    El `feature_order` viaja con los coeficientes porque un vector de features en otro orden
    produce un número perfectamente plausible y perfectamente falso.
    """

    feature_order: tuple[str, ...]
    mean: np.ndarray
    scale: np.ndarray
    coef: np.ndarray
    intercept: float
    name: str = "linear@1"

    def __post_init__(self) -> None:
        n = len(self.feature_order)
        if n == 0:
            raise ModelError("feature_order no puede estar vacío")
        if len(set(self.feature_order)) != n:
            raise ModelError("feature_order tiene nombres repetidos")
        for field_name in ("mean", "scale", "coef"):
            arr = np.asarray(getattr(self, field_name), dtype=np.float64)
            if arr.shape != (n,):
                raise ModelError(
                    f"{field_name} tiene shape {arr.shape} y feature_order tiene {n} nombres"
                )
            if not np.isfinite(arr).all():
                raise ModelError(f"{field_name} contiene valores no finitos")
            arr.setflags(write=False)
            object.__setattr__(self, field_name, arr)
        if np.any(self.scale == 0.0):
            raise ModelError("scale tiene ceros; una feature constante no se puede estandarizar")
        if not np.isfinite(self.intercept):
            raise ModelError("intercept no es finito")

    @property
    def n_features(self) -> int:
        return len(self.feature_order)

    def p_synthetic(self, X: np.ndarray) -> np.ndarray:
        arr = _as_2d(X, n_features=self.n_features)
        z = ((arr - self.mean) / self.scale) @ self.coef + self.intercept
        return _sigmoid(z)

    def p_synthetic_from_mapping(self, features: dict[str, float]) -> float:
        """Camino de `/detect`: ordena por contrato en vez de confiar en el orden del dict."""
        missing = [k for k in self.feature_order if k not in features]
        if missing:
            raise ModelError(f"faltan features requeridas: {missing[:5]}")
        extra = [k for k in features if k not in self.feature_order]
        if extra:
            raise ModelError(f"features no declaradas en el contrato: {extra[:5]}")
        row = np.array([[float(features[k]) for k in self.feature_order]], dtype=np.float64)
        return float(self.p_synthetic(row)[0])

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "name": self.name,
            "feature_order": list(self.feature_order),
            "mean": [float(v) for v in self.mean],
            "scale": [float(v) for v in self.scale],
            "coef": [float(v) for v in self.coef],
            "intercept": float(self.intercept),
        }

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self.to_dict(), ensure_ascii=True, indent=2, allow_nan=False)
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(payload + "\n", encoding="ascii")
        tmp.replace(p)
        return p

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LinearExport":
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise ModelError(
                f"schema_version {payload.get('schema_version')!r} no es {SCHEMA_VERSION}"
            )
        required = ("name", "feature_order", "mean", "scale", "coef", "intercept")
        missing = [k for k in required if k not in payload]
        if missing:
            raise ModelError(f"al export lineal le faltan claves: {missing}")
        return cls(
            feature_order=tuple(payload["feature_order"]),
            mean=np.asarray(payload["mean"], dtype=np.float64),
            scale=np.asarray(payload["scale"], dtype=np.float64),
            coef=np.asarray(payload["coef"], dtype=np.float64),
            intercept=float(payload["intercept"]),
            name=str(payload["name"]),
        )

    @classmethod
    def load(cls, path: str | Path) -> "LinearExport":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="ascii")))


def check_labels(y: Any) -> np.ndarray:
    """Valida el vector de etiquetas. `synthetic` = 1 es la clase positiva (D-A1.1)."""
    arr = np.asarray(y)
    if arr.ndim != 1:
        raise ModelError(f"y debe ser 1-D; llegó ndim={arr.ndim}")
    if arr.dtype == bool:
        raise ModelError("y no puede ser bool: obliga a adivinar cuál clase es la positiva")
    arr = arr.astype(np.int64)
    bad = set(np.unique(arr)) - {0, 1}
    if bad:
        raise ModelError(f"y solo admite 0 (human) y 1 (synthetic); llegaron {sorted(bad)}")
    if len(np.unique(arr)) < 2:
        raise ModelError("y es monoclase; un fold monoclase es un bug del protocolo, no un caso")
    return arr


def check_feature_order(feature_order: Sequence[str]) -> tuple[str, ...]:
    order = tuple(feature_order)
    if not order:
        raise ModelError("feature_order no puede estar vacío")
    if len(set(order)) != len(order):
        raise ModelError("feature_order tiene nombres repetidos")
    if not all(isinstance(k, str) and k for k in order):
        raise ModelError("cada nombre de feature debe ser un str no vacío")
    return order
