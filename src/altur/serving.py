"""El detector que sirve un bundle entrenado. NumPy puro, sin sklearn.

Este módulo es el otro extremo del contrato de `models/base.py`: ahí se **exporta** un
clasificador lineal a coeficientes; aquí se **ejecuta**. Entre los dos no hay ningún
`joblib`, ningún pickle y ninguna dependencia que no esté en las cinco de
`pyproject.toml`. Ese mínimo es el argumento de Feasibility del pitch, y es una propiedad
que se rompe sola en cuanto alguien importa sklearn aquí: no lo hagas.

Tres cosas que este detector se niega a hacer, cada una por una decisión medida:

1. **No para temprano** (D-A3.8). A 5 s de audio el AUC OOF es 0.4012 con IC95
   [0.328, 0.460] — invertido, no indeciso — y la confianza media es la MÁS ALTA de toda
   la curva. Un detector que corta cuando "ya está seguro" cortaría justo donde se
   equivoca con más convicción. Se consume el audio completo, siempre.
2. **No inventa el canal del agente.** Entrada mono es `degraded="ch0_only"`, no ch0
   duplicado. Duplicar fabricaría un canal 1 y corrompería toda feature conversacional.
3. **No sirve un extractor `product_safe=False`.** Se rechaza al construir, no al primer
   request: un bundle con licencia no apta no debe llegar a estar vivo.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .features import acoustic_minimal, behavioral, spectral_factory_lfcc
from .models.base import LinearExport, ModelError
from .registry import RegistryError, audit_product_safety, extractors, turn_sources
from .seg import spectral_factory_vad, vad
from .types import AudioExample, Prediction

# Los plugins que un bundle puede servir. El import es EXPLÍCITO, igual que en `runner.py`:
# la registración no puede depender de descubrimiento por filesystem, o un despliegue
# acabaría sirviendo lo que hubiera en el directorio en vez de lo que declara el manifest.
# Todos son NumPy puro. `seg/vad.py` importa `webrtcvad` dentro de la función `webrtc`, no
# al importar el módulo, así que `energy@1` no arrastra la dependencia a la imagen.
_SERVABLE_PLUGINS = (acoustic_minimal, behavioral, spectral_factory_lfcc, vad, spectral_factory_vad)

# Política de confianza. `confidence` es la probabilidad calibrada DE LA CLASE REPORTADA,
# no P(synthetic): si el veredicto es `human`, devolver P(synthetic)=0.02 como "confianza"
# leería 2 % de confianza en una decisión tomada con 98 %. La pregunta 2 a Altur sigue
# abierta (UNK) — si responden que quieren P(synthetic) crudo, se cambia AQUÍ y el bundle
# lo declara; `api.py` no se toca.
CONFIDENCE_POLICY = "calibrated_probability_of_reported_class"


class ServingError(RuntimeError):
    """El bundle no se puede servir. Nunca se degrada en silencio a una constante."""


def _resolve_extractors(refs: Sequence[str]) -> list[Any]:
    """Resuelve los extractores del bundle y audita su licencia ANTES de servir."""
    if not refs:
        raise ServingError("el bundle no declara ningún extractor")
    for ref in refs:
        if "@" not in ref:
            raise ServingError(
                f"referencia flotante de extractor: {ref!r}. Un bundle fija name@version; "
                "sin versión, dos despliegues del mismo manifest pueden servir código distinto."
            )
    try:
        entries = [extractors.get(ref) for ref in refs]
    except RegistryError as exc:
        raise ServingError(str(exc)) from exc
    no_aptos = audit_product_safety(list(refs))
    if no_aptos:
        raise ServingError(
            f"extractores no aptos para producto en el bundle: {no_aptos}. "
            "openSMILE y Parselmouth son banco de experimentos, no despliegue."
        )
    return entries


class BundledDetector:
    """Extractores → vector ordenado → lineal → calibrador → umbral → `Prediction`.

    El `feature_order` del `LinearExport` es la autoridad sobre el orden del vector. No se
    confía en el orden de iteración de un dict: un vector permutado produce un número
    perfectamente plausible y perfectamente falso (por eso existe
    `LinearExport.p_synthetic_from_mapping`, que ordena por contrato y rechaza faltantes
    y sobrantes).
    """

    def __init__(
        self,
        *,
        name: str,
        export: LinearExport,
        extractor_refs: Sequence[str],
        segmenter_ref: str | None,
        calibrator: Any | None = None,
        threshold: float = 0.5,
        min_seconds_declared: float | None = None,
    ) -> None:
        if not 0.0 < float(threshold) < 1.0:
            raise ServingError(f"umbral fuera de (0, 1): {threshold!r}")
        self.name = str(name)
        self._export = export
        self._extractor_refs = tuple(extractor_refs)
        self._entries = _resolve_extractors(self._extractor_refs)
        self._calibrator = calibrator
        self.threshold = float(threshold)
        self.min_seconds_declared = (
            None if min_seconds_declared is None else float(min_seconds_declared)
        )

        self._needs_seg = any(e.meta.get("needs_seg") for e in self._entries)
        self._segmenter = None
        if self._needs_seg:
            if not segmenter_ref or "@" not in segmenter_ref:
                raise ServingError(
                    "hay extractores con needs_seg=True y el bundle no fija un segmentador "
                    "name@version"
                )
            try:
                entry = turn_sources.get(segmenter_ref)
            except RegistryError as exc:
                raise ServingError(str(exc)) from exc
            if entry.meta.get("is_oracle"):
                raise ServingError(
                    f"segmentador oráculo en un bundle: {segmenter_ref}. En inferencia solo "
                    "llega el WAV; no hay turns/. Un bundle con oráculo no es desplegable."
                )
            self._segmenter = entry.obj
        self.segmenter_ref = segmenter_ref if self._needs_seg else None

        # Que los canales que el bundle necesita estén declarados evita descubrir en el
        # primer request de un juez que el modelo exige ch1 y la entrada venía mono.
        canales: set[int] = set()
        for e in self._entries:
            canales.update(int(c) for c in e.meta.get("channels", ()))
        self.required_channels = tuple(sorted(canales))

    @property
    def feature_order(self) -> tuple[str, ...]:
        return self._export.feature_order

    def features(self, ex: AudioExample) -> dict[str, float]:
        """Vector de features del ejemplo. Separado de `predict` para poder auditarlo."""
        if self._segmenter is not None and ex.seg is None:
            ex = AudioExample(ex.ch0, ex.ch1, sr=ex.sr, seg=self._segmenter(ex))
        merged: dict[str, float] = {}
        for entry in self._entries:
            result = entry.obj(ex)
            if not isinstance(result, tuple) or len(result) != 2:
                raise ServingError(f"{entry.ref} no devolvió (features, diagnostics)")
            feats = result[0]
            if not isinstance(feats, Mapping):
                raise ServingError(f"{entry.ref} devolvió features que no son un mapping")
            chocan = set(feats) & set(merged)
            if chocan:
                raise ServingError(
                    f"{entry.ref} reusa claves de otro extractor: {sorted(chocan)[:3]}. "
                    "Cada extractor escribe bajo su propio prefijo."
                )
            merged.update({str(k): float(v) for k, v in feats.items()})
        return merged

    def predict(self, ex: AudioExample) -> Prediction:
        if 1 in self.required_channels and ex.is_mono:
            raise ServingError(
                "el bundle necesita el canal del agente y la entrada es mono; "
                "duplicar ch0 fabricaría un canal 1"
            )
        try:
            feats = self.features(ex)
            p = self._export.p_synthetic_from_mapping(feats)
        except ModelError as exc:
            raise ServingError(str(exc)) from exc
        if self._calibrator is not None:
            p = float(self._calibrator.transform([p])[0])

        is_syn = p > self.threshold
        # Confianza = probabilidad calibrada de lo que se está afirmando, no P(synthetic).
        conf = p if is_syn else 1.0 - p

        diagnostics: dict[str, Any] = {
            "detector": self.name,
            "duration_s": round(ex.duration_s, 3),
            "p_synthetic": round(p, 6),
            "threshold": self.threshold,
        }
        degraded = "ch0_only" if ex.is_mono else None
        # D-A3.8: el audio corto NO cambia el veredicto ni corta la inferencia. Se MARCA,
        # porque a 5 s el IC95 del AUC es [0.328, 0.460] y no toca 0.5 — está invertido.
        if self.min_seconds_declared is not None and ex.duration_s < self.min_seconds_declared:
            diagnostics["below_declared_horizon"] = True
            degraded = degraded or "short_audio"

        return Prediction(
            is_synthetic=bool(is_syn),
            confidence=float(min(max(conf, 0.0), 1.0)),
            seconds_used=round(ex.duration_s, 3),   # SIEMPRE el audio completo (D-A3.8)
            degraded=degraded,
            diagnostics=diagnostics,
        )
