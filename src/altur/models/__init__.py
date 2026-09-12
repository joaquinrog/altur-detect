"""Modelos del arnés.

Frontera que este paquete existe para mantener:

- `base.py` y `baselines.py` son **NumPy puro**. `/detect` puede importarlos.
- `adapters.py` importa sklearn y solo se usa en entrenamiento.

Importar `altur.models` registra todo en el registro `models` sin que nadie edite
`registry.py`, que es de edición exclusiva del integrador.
"""

from typing import Any

from altur.models.adapters import SklearnAdapter, logreg, logreg_factory
from altur.models.base import LinearExport, Model, ModelError
from altur.models.baselines import ConstantModel, ThresholdStump, constant, stump

__all__ = [
    "ConstantModel",
    "LinearExport",
    "Model",
    "ModelError",
    "SklearnAdapter",
    "ThresholdStump",
    "acoustic_baseline",
    "behavioral_baseline",
    "constant",
    "logreg",
    "logreg_factory",
    "stump",
]


def _prefixed(feature_order: Any, prefix: str) -> tuple[str, ...]:
    order = tuple(feature_order)
    hit = tuple(k for k in order if k.startswith(prefix))
    if not hit:
        raise ModelError(f"ninguna feature de feature_order empieza con {prefix!r}")
    return hit


def acoustic_baseline(feature_order: Any, prefix: str = "a2.acoustic.") -> SklearnAdapter:
    """Logreg sobre la familia acústica y nada más. Baseline de rama, no candidato."""
    return SklearnAdapter(
        f"acoustic_baseline[{prefix}]@1",
        logreg_factory(),
        feature_order=_prefixed(feature_order, prefix),
    )


def behavioral_baseline(feature_order: Any, prefix: str = "behavioral.") -> SklearnAdapter:
    """Logreg sobre la familia conductual. D-A2.2 la dejó como diagnóstico, no productiva."""
    return SklearnAdapter(
        f"behavioral_baseline[{prefix}]@1",
        logreg_factory(),
        feature_order=_prefixed(feature_order, prefix),
    )
