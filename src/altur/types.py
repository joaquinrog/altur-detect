"""Contratos del arnés. Este archivo es el candado anti-fuga.

REGLA DE INTEGRACIÓN: prohibido editar en paralelo (plan §11). Un solo integrador.

La separación entre `AudioExample` y `DatasetRecord` es la corrección P0.1 de la revisión.
Un extractor recibe SOLO `AudioExample`. Nunca ve id, etiqueta, split, grupos ni procedencia.

La fuga se REDUCE por interfaz, tests y revisión. No se elimina: un extractor en Python
puede leer un archivo global. Por eso existe `scripts/check_extractor.py`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

SAMPLE_RATE = 8000
Channel = Literal[0, 1]


def _freeze(a: np.ndarray) -> np.ndarray:
    """Copia read-only. Hace real la inmutabilidad que `frozen=True` NO da a un ndarray."""
    out = np.asarray(a, dtype=np.float32)
    out = out.copy() if out.flags.owndata else np.array(out, dtype=np.float32)
    out.setflags(write=False)
    return out


@dataclass(frozen=True, slots=True)
class Turn:
    channel: int
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass(frozen=True, slots=True)
class Segmentation:
    """Turnos por canal. `source` distingue el oráculo de lo recalculable.

    `oracle@1` son los `turns/` oficiales: SOLO entrenamiento, es cota superior.
    Nada que aspire a `/detect` puede usarlos — en inferencia solo llega el WAV.
    """

    turns: tuple[Turn, ...]
    source: str
    params: dict[str, Any] = field(default_factory=dict)

    @property
    def is_oracle(self) -> bool:
        return self.source.startswith("oracle")

    def by_channel(self, ch: int) -> tuple[Turn, ...]:
        return tuple(t for t in self.turns if t.channel == ch)

    def speech_mask(self, ch: int, n_samples: int, sr: int = SAMPLE_RATE) -> np.ndarray:
        """Máscara booleana de habla. Base de todo control silence-only."""
        m = np.zeros(n_samples, dtype=bool)
        for t in self.by_channel(ch):
            i0 = max(0, round(t.start * sr))
            i1 = min(n_samples, round(t.end * sr))
            if i1 > i0:
                m[i0:i1] = True
        return m


@dataclass(frozen=True, slots=True)
class AudioExample:
    """Lo ÚNICO que ve un extractor.

    Sin id, sin etiqueta, sin split, sin grupos, sin procedencia. Se construye desde
    bytes, así que entrenamiento e inferencia comparten el mismo objeto: sin train/serve skew.

    `ch1 is None` significa entrada mono. NO se rellena duplicando ch0: eso inventaría un
    canal de agente y corrompería toda feature conversacional (corrección P1.10).
    """

    ch0: np.ndarray
    ch1: np.ndarray | None
    sr: int = SAMPLE_RATE
    seg: Segmentation | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "ch0", _freeze(self.ch0))
        if self.ch1 is not None:
            object.__setattr__(self, "ch1", _freeze(self.ch1))
            if len(self.ch1) != len(self.ch0):
                raise ValueError(f"canales desalineados: {len(self.ch0)} vs {len(self.ch1)}")
        if self.sr != SAMPLE_RATE:
            raise ValueError(f"sample rate {self.sr}: el reto es 8 kHz")

    @property
    def is_mono(self) -> bool:
        return self.ch1 is None

    @property
    def n_samples(self) -> int:
        return len(self.ch0)

    @property
    def duration_s(self) -> float:
        return self.n_samples / self.sr

    def channel(self, ch: int) -> np.ndarray:
        if ch == 0:
            return self.ch0
        if ch == 1:
            if self.ch1 is None:
                raise ValueError("no hay canal 1: la entrada era mono")
            return self.ch1
        raise ValueError(f"canal inválido: {ch}")


@dataclass(frozen=True, slots=True)
class DatasetRecord:
    """Metadatos. NUNCA se entrega a un extractor.

    `groups` lleva las llaves con las que se construyen componentes conectados (plan §4):
    speaker_or_voice_id, donor_call_id, script_line_id, vendor, accent_country,
    capture_route, source_tier. Todo lo conectado cae en una sola partición.
    """

    example_id: str
    label: int | None = None
    split: str | None = None
    groups: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)

    @property
    def is_derived(self) -> bool:
        return bool(self.provenance.get("transforms"))


@dataclass(frozen=True, slots=True)
class Prediction:
    """Salida del detector.

    El contrato oficial exige `is_synthetic` y acepta `confidence`. Los campos extra son
    aditivos y se emiten solo cuando aplican; `to_response()` decide qué sale por la API.
    """

    is_synthetic: bool
    confidence: float
    seconds_used: float | None = None
    degraded: str | None = None          # p.ej. "ch0_only" si la entrada era mono
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_response(self, extras: bool = True) -> dict[str, Any]:
        out: dict[str, Any] = {
            "is_synthetic": bool(self.is_synthetic),
            "confidence": round(float(self.confidence), 6),
        }
        if extras:
            if self.seconds_used is not None:
                out["seconds_used"] = round(float(self.seconds_used), 3)
            if self.degraded is not None:
                out["degraded"] = self.degraded
        return out


# (features, diagnostics) — devolver diagnósticos es parte del contrato del extractor.
# R3: el fallo del extractor es una tercera clase disfrazada de señal.
ExtractorResult = tuple[dict[str, float], dict[str, Any]]
