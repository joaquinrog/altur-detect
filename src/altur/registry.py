"""Registro de piezas intercambiables.

REGLA DE INTEGRACIÓN: prohibido editar en paralelo (plan §11).

Añadir un enfoque = escribir una función con un decorador + un YAML de experimento.
Nada más del arnés cambia. Ese es todo el punto de la arquitectura.

Las entradas se direccionan `nombre@version`. Pedir `nombre` sin versión resuelve a la
más alta registrada, pero un spec de experimento DEBE fijar la versión: el `run_id` se
hashea del spec y una versión flotante lo vuelve irreproducible.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any


class RegistryError(RuntimeError):
    pass


@dataclass(frozen=True)
class Entry:
    name: str
    version: int
    obj: Any
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def ref(self) -> str:
        return f"{self.name}@{self.version}"

    def __repr__(self) -> str:  # pragma: no cover
        return f"<{self.ref}>"


class Registry:
    """Un registro por tipo de pieza. `kind` solo sirve para mensajes de error claros."""

    def __init__(self, kind: str, required_meta: tuple[str, ...] = ()) -> None:
        self.kind = kind
        self.required_meta = required_meta
        self._entries: dict[str, Entry] = {}

    def register(self, name: str, version: int = 1, **meta: Any) -> Callable[[Any], Any]:
        if "@" in name:
            raise RegistryError(f"el nombre no lleva '@': {name!r}")

        def deco(obj: Any) -> Any:
            ref = f"{name}@{version}"
            if ref in self._entries:
                raise RegistryError(f"{self.kind} duplicado: {ref}")
            missing = [k for k in self.required_meta if k not in meta]
            if missing:
                raise RegistryError(f"{ref}: falta metadata obligatoria {missing}")
            self._entries[ref] = Entry(name=name, version=version, obj=obj, meta=dict(meta))
            obj._altur_ref = ref
            obj._altur_meta = dict(meta)
            return obj

        return deco

    __call__ = register

    def get(self, ref: str) -> Entry:
        if "@" in ref:
            try:
                return self._entries[ref]
            except KeyError:
                raise RegistryError(
                    f"{self.kind} no registrado: {ref}. Disponibles: {sorted(self._entries)}"
                ) from None
        candidates = [e for e in self._entries.values() if e.name == ref]
        if not candidates:
            raise RegistryError(
                f"{self.kind} no registrado: {ref}. Disponibles: {sorted(self._entries)}"
            )
        return max(candidates, key=lambda e: e.version)

    def resolve(self, ref: str) -> Any:
        return self.get(ref).obj

    def meta(self, ref: str) -> dict[str, Any]:
        return self.get(ref).meta

    def __contains__(self, ref: str) -> bool:
        try:
            self.get(ref)
            return True
        except RegistryError:
            return False

    def __iter__(self) -> Iterator[Entry]:
        return iter(sorted(self._entries.values(), key=lambda e: e.ref))

    def __len__(self) -> int:
        return len(self._entries)

    def refs(self) -> list[str]:
        return sorted(self._entries)


# --------------------------------------------------------------------------------------
# Los registros. `required_meta` es lo que obliga a declarar lo que el arnés necesita
# para tomar decisiones automáticas (licencia, si puede ir a producto, presupuesto...).
# --------------------------------------------------------------------------------------

# extract(ex: AudioExample) -> (features, diagnostics)
#   channels      : canales que lee
#   needs_seg     : requiere segmentación
#   license       : SPDX o etiqueta; el empaquetado la audita
#   product_safe  : False => banco de experimentos, prohibido congelarlo en /detect
#   budget_ms     : presupuesto declarado para 150 s de audio
extractors = Registry(
    "extractor", required_meta=("channels", "needs_seg", "license", "product_safe")
)

# segment(ex: AudioExample) -> Segmentation
turn_sources = Registry("turn_source", required_meta=("is_oracle",))

# apply(ex, rng) -> AudioExample
#   use            : "augment" | "holdout"  — CONGELADO antes de correr (plan §9)
#   changes_length : True invalida PESQ sin realineación
#   shifts_timing  : True obliga a recalcular la segmentación
transforms = Registry(
    "transform", required_meta=("use", "changes_length", "shifts_timing")
)

models = Registry("model")
fusions = Registry("fusion")
calibrators = Registry("calibrator")

# Fuentes de voz del corpus ampliado
#   kind: "human" | "synthetic"
voice_sources = Registry("voice_source", required_meta=("kind",))

ALL: dict[str, Registry] = {
    "extractor": extractors,
    "turn_source": turn_sources,
    "transform": transforms,
    "model": models,
    "fusion": fusions,
    "calibrator": calibrators,
    "voice_source": voice_sources,
}


def audit_product_safety(refs: list[str]) -> list[str]:
    """Devuelve los extractores que NO pueden congelarse en un bundle de producción.

    openSMILE es research-only (licencia audEERING, incluye el uso indirecto vía features)
    y Parselmouth es GPL-3. Sirven como banco de experimentos; el empaquetado los rechaza.
    """
    bad = []
    for ref in refs:
        m = extractors.meta(ref)
        if not m.get("product_safe", False):
            bad.append(f"{extractors.get(ref).ref} (licencia: {m.get('license', '?')})")
    return bad


def summary() -> str:  # pragma: no cover
    lines = []
    for kind, reg in ALL.items():
        lines.append(f"{kind:14s} {len(reg):3d}  {', '.join(reg.refs()) or '—'}")
    return "\n".join(lines)
