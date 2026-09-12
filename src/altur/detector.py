"""El artefacto que sirve `/detect`.

`api.py` importa SOLO esto. Cambiar de enfoque = cambiar el bundle en disco, nunca editar
el endpoint. Es lo que permite congelar la API en la hora 2 y seguir iterando 28 horas.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .types import AudioExample, Prediction


@runtime_checkable
class Detector(Protocol):
    """Contrato mínimo de inferencia."""

    name: str

    def predict(self, ex: AudioExample) -> Prediction: ...


class ConstantDetector:
    """Detector de referencia: responde siempre lo mismo.

    No es un placeholder que se tira. Cumple tres funciones permanentes:
      1. Permite construir, desplegar y ensayar `/detect` antes de que exista un modelo.
      2. Es el piso contra el que se compara todo (predecir la clase mayoritaria).
      3. Es el fallback del runbook de failover si el bundle real no carga.
    """

    def __init__(self, is_synthetic: bool = True, confidence: float = 0.5,
                 name: str = "constant@1") -> None:
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence fuera de [0,1]")
        self.name = name
        self._is_synthetic = bool(is_synthetic)
        self._confidence = float(confidence)

    def predict(self, ex: AudioExample) -> Prediction:
        return Prediction(
            is_synthetic=self._is_synthetic,
            confidence=self._confidence,
            degraded="ch0_only" if ex.is_mono else None,
            diagnostics={"detector": self.name, "duration_s": round(ex.duration_s, 3)},
        )
