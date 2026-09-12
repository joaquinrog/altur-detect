"""Indice del dataset oficial. El lado de metadatos del candado anti-fuga.

REGLA: `load()` devuelve `AudioExample`. `record()` devuelve `DatasetRecord`. **Son dos
llamadas distintas a proposito.** El runner las asocia FUERA del extractor; un extractor
recibe lo que devuelve `load()` y nada mas. Si alguna vez esta clase devuelve los dos
juntos en un solo objeto, el candado de `types.py` deja de servir para nada.

Estructura en disco (FACT, notas/01 §3):

    data/manifest.csv        anon_id, label (human|synthetic), split (train|val), duration_s
    data/audio/<id>.wav      estereo, 8 kHz, 16-bit PCM. ch0 = caller, ch1 = agente
    data/turns/<id>.json     {"turns": [{"channel": 0, "start": 12.4, "end": 15.1}, ...]}

Sobre `turns/`, textual del README de Altur: *"Derived automatically from the audio; use
them as a starting point."* Se cargan con `source="oracle@1"` y `is_oracle == True`.

🔴 **En inferencia solo llega el WAV. No hay `turns/`.** Cualquier feature basada en turnos
se recalcula con VAD propio, o no puede llegar a `/detect`. El oraculo sirve como cota
superior en experimentos: dice cuanto se pierde por segmentar mal, no cuanto se gana.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .io import decode
from .types import AudioExample, DatasetRecord, Segmentation, Turn

# 1 = synthetic es la clase positiva: `/detect` responde `is_synthetic`, y una metrica
# que mide lo contrario de lo que el contrato pregunta se lee bien y esta al reves.
LABELS: dict[str, int] = {"human": 0, "synthetic": 1}
LABEL_NAMES: dict[int, str] = {v: k for k, v in LABELS.items()}

SPLITS = ("train", "val")
EXPECTED_CALLS = 353

DEFAULT_ROOT = Path(__file__).resolve().parent.parent.parent / "data"


class DatasetError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class ManifestRow:
    """Una fila cruda del manifest, ya tipada y validada."""

    anon_id: str
    label: int
    split: str
    duration_s: float


def read_manifest(path: Path) -> list[ManifestRow]:
    """Lee y valida el manifest. Falla fuerte: un manifest raro envenena todo lo demas."""
    if not path.exists():
        raise DatasetError(
            f"no existe {path}. Corre `python scripts/download_data.py` primero."
        )
    rows: list[ManifestRow] = []
    seen: set[str] = set()
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"anon_id", "label", "split", "duration_s"}
        got = set(reader.fieldnames or ())
        if not required <= got:
            raise DatasetError(f"manifest.csv: faltan columnas {sorted(required - got)}")
        for i, raw in enumerate(reader, start=2):  # 2 = primera fila de datos
            anon_id = (raw["anon_id"] or "").strip()
            if not anon_id:
                raise DatasetError(f"manifest.csv linea {i}: anon_id vacio")
            if anon_id in seen:
                raise DatasetError(f"manifest.csv linea {i}: anon_id duplicado {anon_id!r}")
            seen.add(anon_id)

            label_s = (raw["label"] or "").strip().lower()
            if label_s not in LABELS:
                raise DatasetError(f"manifest.csv linea {i}: label {label_s!r} desconocida")
            split = (raw["split"] or "").strip().lower()
            if split not in SPLITS:
                raise DatasetError(f"manifest.csv linea {i}: split {split!r} desconocido")
            try:
                duration = float(raw["duration_s"])
            except (TypeError, ValueError) as e:
                raise DatasetError(
                    f"manifest.csv linea {i}: duration_s {raw['duration_s']!r} no es numero"
                ) from e

            rows.append(
                ManifestRow(anon_id=anon_id, label=LABELS[label_s], split=split,
                            duration_s=duration)
            )
    return rows


def read_turns(path: Path) -> Segmentation:
    """`turns/<id>.json` -> `Segmentation` marcada como oraculo."""
    data = json.loads(path.read_text(encoding="utf-8"))
    raw = data["turns"] if isinstance(data, dict) else data
    turns = tuple(
        Turn(channel=int(t["channel"]), start=float(t["start"]), end=float(t["end"]))
        for t in raw
    )
    # Orden estable: dos corridas deben producir la misma segmentacion byte a byte,
    # porque entra en la clave de cache.
    turns = tuple(sorted(turns, key=lambda t: (t.start, t.channel, t.end)))
    return Segmentation(turns=turns, source="oracle@1", params={"provider": "altur"})


class Dataset:
    """Indice perezoso sobre `data/`. Los metadatos se leen al construir; el audio, al pedirlo.

    Perezoso a proposito: 353 llamadas estereo a 8 kHz son ~600 MB de WAV y ~1.2 GB como
    float32 en RAM. La maquina tiene 15 GB y tambien tiene que sostener el modelo.
    """

    def __init__(self, root: Path | str = DEFAULT_ROOT, *, require_audio: bool = True) -> None:
        self.root = Path(root)
        self.audio_dir = self.root / "audio"
        self.turns_dir = self.root / "turns"
        self.manifest_path = self.root / "manifest.csv"

        self._rows: dict[str, ManifestRow] = {}
        for r in read_manifest(self.manifest_path):
            self._rows[r.anon_id] = r
        self._ids: tuple[str, ...] = tuple(sorted(self._rows))

        if require_audio:
            missing = [i for i in self._ids if not (self.audio_dir / f"{i}.wav").exists()]
            if missing:
                raise DatasetError(
                    f"faltan {len(missing)} WAV (p.ej. {missing[:3]}). "
                    "Corre `python scripts/download_data.py`."
                )

    # -- metadatos -------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._ids)

    def __contains__(self, example_id: object) -> bool:
        return example_id in self._rows

    @property
    def ids(self) -> tuple[str, ...]:
        """Orden determinista (alfabetico). No depende del orden del manifest."""
        return self._ids

    def ids_for(self, split: str) -> tuple[str, ...]:
        if split not in SPLITS:
            raise DatasetError(f"split desconocido: {split!r}")
        return tuple(i for i in self._ids if self._rows[i].split == split)

    def record(self, example_id: str) -> DatasetRecord:
        """Metadatos. 🔴 NUNCA se le pasa esto a un extractor.

        `groups` lleva solo lo que el dataset oficial permite saber. **OBS: el manifest
        trae `anon_id`, `label`, `split` y `duration_s` — y nada mas.** No hay
        `speaker_or_voice_id`, ni `donor_call_id`, ni vendor, ni linea de guion. Asi que
        aqui la unica llave real es la llamada; `protocol.py` construye los componentes
        conectados con lo que haya y, si no hay evidencia de agrupacion por hablante,
        el fallback conservador es grupo = llamada, declarado. Ver plan §4.
        """
        row = self._require(example_id)
        return DatasetRecord(
            example_id=row.anon_id,
            label=row.label,
            split=row.split,
            groups={"call_id": row.anon_id},
            provenance={
                "source": "altur_official_v1",
                "manifest_duration_s": row.duration_s,
                "transforms": [],
            },
        )

    def records(self, split: str | None = None) -> Iterator[DatasetRecord]:
        ids = self._ids if split is None else self.ids_for(split)
        for i in ids:
            yield self.record(i)

    # -- audio -----------------------------------------------------------------

    def load(self, example_id: str, *, with_turns: bool = False) -> AudioExample:
        """Lo UNICO que puede ver un extractor.

        `with_turns=False` por defecto, y es deliberado: el camino comodo debe ser el que
        se parece a produccion. Pedir el oraculo es un acto explicito que se lee en el diff.
        """
        row = self._require(example_id)
        raw = (self.audio_dir / f"{row.anon_id}.wav").read_bytes()
        ex = decode(raw, allow_mono=False, allow_resample=False).example
        if not with_turns:
            return ex
        turns_path = self.turns_dir / f"{row.anon_id}.json"
        if not turns_path.exists():
            raise DatasetError(f"no hay turns/ para {row.anon_id}")
        return AudioExample(ch0=ex.ch0, ch1=ex.ch1, sr=ex.sr, seg=read_turns(turns_path))

    def audio_sha256(self, example_id: str) -> str:
        """Hash de los BYTES del WAV. Entra en la clave de cache (plan §5)."""
        row = self._require(example_id)
        h = hashlib.sha256()
        with (self.audio_dir / f"{row.anon_id}.wav").open("rb") as f:
            while chunk := f.read(1 << 20):
                h.update(chunk)
        return h.hexdigest()

    def fingerprint(self) -> str:
        """Hash del dataset entero: identidad + etiqueta + split, en orden fijo.

        Va en cada corrida del libro mayor. Si dos corridas dicen numeros distintos con el
        mismo fingerprint, el culpable es el codigo; si el fingerprint cambio, son datos
        distintos y no se comparan.
        """
        h = hashlib.sha256()
        for i in self._ids:
            r = self._rows[i]
            h.update(f"{r.anon_id}|{r.label}|{r.split}|{r.duration_s:.6f}\n".encode())
        return h.hexdigest()

    # -- resumen ---------------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        out: dict[str, Any] = {"n": len(self._ids), "fingerprint": self.fingerprint()[:16]}
        for split in SPLITS:
            ids = self.ids_for(split)
            n_syn = sum(self._rows[i].label == 1 for i in ids)
            durs = [self._rows[i].duration_s for i in ids]
            out[split] = {
                "n": len(ids),
                "synthetic": n_syn,
                "human": len(ids) - n_syn,
                "prevalence": round(n_syn / len(ids), 4) if ids else None,
                "duration_s": {
                    "total": round(sum(durs), 1),
                    "min": round(min(durs), 2) if durs else None,
                    "max": round(max(durs), 2) if durs else None,
                },
            }
        return out

    def _require(self, example_id: str) -> ManifestRow:
        try:
            return self._rows[example_id]
        except KeyError:
            raise DatasetError(f"no existe en el manifest: {example_id!r}") from None
