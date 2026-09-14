"""Construye el corpus privado de C3 D-A8.3 sin copiar audio ni leer ``val``.

Las tablas autoritativas son ``pairs.csv`` para B y ``voices_map.csv`` para B2. El
índice y los grupos contienen identificadores privados, por lo que solo se congelan bajo
``.private/c3_bc_v1``. El bundle recibe únicamente agregados y fingerprints.
"""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import subprocess
import sys
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any, NamedTuple

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent / "cc_hc_hackmty26"
DEFAULT_OFFICIAL = ROOT / "data"
DEFAULT_B = WORKSPACE / "bloque_b"
DEFAULT_B2 = WORKSPACE / "bloque_b2"
PRIVATE_ROOT = ROOT / ".private" / "c3_bc_v1"
EXPECTED_COUNTS = {"total": 372, "human": 128, "synthetic": 244}
EXPECTED_SOURCE_COUNTS = {"official_train": 282, "bloque_b": 30, "bloque_b2": 60}

sys.path.insert(0, str(ROOT / "src"))

from altur.dataset import LABELS
from altur.io import decode
from altur.provenance import effective_code_digest
from altur.runner import run_experiment
from altur.types import DatasetRecord


class C3CorpusError(RuntimeError):
    pass


class CorpusRow(NamedTuple):
    internal_id: str
    call_id: str
    label: int
    source: str
    audio_path: Path
    audio_sha256: str
    group_id: str


class CorpusIndex(NamedTuple):
    rows: tuple[CorpusRow, ...]
    fingerprint: str


class C3Corpus(NamedTuple):
    features: dict[str, dict[str, float]]
    labels: dict[str, int]
    groups: dict[str, str]
    fingerprint: str


def _read_csv(path: Path, required: set[str]) -> list[dict[str, str]]:
    if not path.is_file():
        raise C3CorpusError(f"no existe tabla requerida: {path}")
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise C3CorpusError(f"{path.name}: faltan columnas {sorted(missing)}")
        return [{key: (value or "").strip() for key, value in row.items()} for row in reader]


def _manifest(root: Path) -> tuple[list[dict[str, str]], dict[str, dict[str, str]]]:
    rows = _read_csv(root / "manifest.csv", {"anon_id", "label", "split", "duration_s"})
    by_id: dict[str, dict[str, str]] = {}
    for row in rows:
        call_id = row["anon_id"]
        if not call_id or call_id in by_id:
            raise C3CorpusError(f"{root / 'manifest.csv'}: anon_id vacío o duplicado")
        if row["label"] not in LABELS:
            raise C3CorpusError(f"label inválida en {root / 'manifest.csv'}")
        by_id[call_id] = row
    return rows, by_id


def _sha256(path: Path) -> str:
    if not path.is_file():
        raise C3CorpusError(f"falta WAV de audio seleccionado: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def add(self, value: str) -> None:
        self.parent.setdefault(value, value)

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        a, b = self.find(left), self.find(right)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def build_corpus_index(
    *,
    official_root: Path = DEFAULT_OFFICIAL,
    block_b_root: Path = DEFAULT_B,
    block_b2_root: Path = DEFAULT_B2,
    expected_counts: Mapping[str, int] | None = None,
) -> CorpusIndex:
    strict = expected_counts is None
    expected = dict(expected_counts or EXPECTED_COUNTS)
    candidates: list[dict[str, Any]] = []

    official_rows, _ = _manifest(official_root)
    for row in official_rows:
        if row["split"] == "train":
            candidates.append({
                "internal_id": f"official:{row['anon_id']}",
                "call_id": row["anon_id"],
                "label": LABELS[row["label"]],
                "source": "official_train",
                "audio_path": official_root / "audio" / f"{row['anon_id']}.wav",
                "relations": (f"call:{row['anon_id']}",),
            })
        elif row["split"] != "val":
            raise C3CorpusError("el manifest oficial contiene un split distinto de train/val")

    b_manifest_rows, b_manifest = _manifest(block_b_root)
    pairs = _read_csv(
        block_b_root / "pairs.csv",
        {"guion", "variante", "id_sintetica", "id_humana", "voz", "persona"},
    )
    joined_b: set[str] = set()
    for pair in pairs:
        if pair["variante"] not in {"plana", "g711"}:
            raise C3CorpusError("pairs.csv contiene una variante distinta de plana/g711")
        for field, label, identity in (
            ("id_sintetica", "synthetic", f"voice:{pair['voz']}"),
            ("id_humana", "human", f"persona:{pair['persona']}"),
        ):
            call_id = pair[field]
            if call_id in joined_b:
                raise C3CorpusError("join B no es 1:1: anon_id duplicado")
            joined_b.add(call_id)
            manifest_row = b_manifest.get(call_id)
            if manifest_row is None or manifest_row["label"] != label:
                raise C3CorpusError("join B no es 1:1 o la etiqueta no coincide")
            if pair["variante"] == "plana":
                candidates.append({
                    "internal_id": f"b:{call_id}", "call_id": call_id,
                    "label": LABELS[label], "source": "bloque_b",
                    "audio_path": block_b_root / "audio" / f"{call_id}.wav",
                    "relations": (f"guion:{pair['guion']}", identity),
                })
    if joined_b != set(b_manifest) or len(b_manifest_rows) != len(joined_b):
        raise C3CorpusError("join B no es 1:1 con manifest.csv")

    b2_manifest_rows, b2_manifest = _manifest(block_b2_root)
    voices = _read_csv(
        block_b2_root / "voices_map.csv",
        {"anon_id", "slot", "guion", "variante", "voz"},
    )
    joined_b2: set[str] = set()
    for voice in voices:
        call_id = voice["anon_id"]
        if call_id in joined_b2:
            raise C3CorpusError("join B2 no es 1:1: anon_id duplicado")
        joined_b2.add(call_id)
        manifest_row = b2_manifest.get(call_id)
        if manifest_row is None or manifest_row["label"] != "synthetic":
            raise C3CorpusError("join B2 no es 1:1 o contiene etiqueta no sintética")
        if voice["variante"] not in {"plana", "g711"}:
            raise C3CorpusError("voices_map.csv contiene una variante distinta de plana/g711")
        if voice["variante"] == "plana":
            candidates.append({
                "internal_id": f"b2:{call_id}", "call_id": call_id,
                "label": LABELS["synthetic"], "source": "bloque_b2",
                "audio_path": block_b2_root / "audio" / f"{call_id}.wav",
                "relations": (f"guion:{voice['guion']}", f"voice:{voice['voz']}"),
            })
    if joined_b2 != set(b2_manifest) or len(b2_manifest_rows) != len(joined_b2):
        raise C3CorpusError("join B2 no es 1:1 con manifest.csv")

    ids = [row["internal_id"] for row in candidates]
    call_ids = [row["call_id"] for row in candidates]
    if len(set(ids)) != len(ids) or len(set(call_ids)) != len(call_ids):
        raise C3CorpusError("una llamada aparece más de una vez en el corpus C3")

    counts = {
        "total": len(candidates),
        "human": sum(row["label"] == 0 for row in candidates),
        "synthetic": sum(row["label"] == 1 for row in candidates),
    }
    if counts != expected:
        raise C3CorpusError(f"conteos C3 inválidos: {counts}; se esperaba {expected}")
    source_counts = {source: sum(row["source"] == source for row in candidates)
                     for source in EXPECTED_SOURCE_COUNTS}
    if strict and source_counts != EXPECTED_SOURCE_COUNTS:
        raise C3CorpusError(f"conteos por fuente inválidos: {source_counts}")
    if strict and (len(official_rows) != 353 or len(b_manifest_rows) != 60 or len(b2_manifest_rows) != 120):
        raise C3CorpusError("los manifests no tienen tamaños oficiales 353/60/120")

    uf = _UnionFind()
    relation_owner: dict[str, str] = {}
    for row in candidates:
        internal_id = row["internal_id"]
        uf.add(internal_id)
        for relation in row["relations"]:
            owner = relation_owner.setdefault(relation, internal_id)
            uf.union(internal_id, owner)
    components: dict[str, list[str]] = defaultdict(list)
    for internal_id in ids:
        components[uf.find(internal_id)].append(internal_id)
    group_by_id = {
        internal_id: "group:" + hashlib.sha256("\n".join(sorted(members)).encode()).hexdigest()[:16]
        for members in components.values()
        for internal_id in members
    }

    rows = tuple(
        CorpusRow(
            internal_id=row["internal_id"], call_id=row["call_id"], label=row["label"],
            source=row["source"], audio_path=row["audio_path"],
            audio_sha256=_sha256(row["audio_path"]), group_id=group_by_id[row["internal_id"]],
        )
        for row in sorted(candidates, key=lambda item: item["internal_id"])
    )
    fingerprint_payload = [
        [row.internal_id, row.label, row.source, row.audio_sha256, row.group_id] for row in rows
    ]
    fingerprint = hashlib.sha256(
        json.dumps(fingerprint_payload, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()
    return CorpusIndex(rows=rows, fingerprint=fingerprint)


class CombinedDataset:
    def __init__(self, index: CorpusIndex) -> None:
        self.index = index
        self._rows = {row.internal_id: row for row in index.rows}

    def ids_for(self, split: str) -> tuple[str, ...]:
        if split != "train":
            raise C3CorpusError("C3 solo permite el split train combinado")
        return tuple(self._rows)

    def record(self, example_id: str) -> DatasetRecord:
        row = self._rows[example_id]
        return DatasetRecord(
            example_id=example_id, label=row.label, split="train",
            groups={"group_id": row.group_id}, provenance={"source": row.source, "transforms": []},
        )

    def load(self, example_id: str, *, with_turns: bool = False):
        if with_turns:
            raise C3CorpusError("C3 no permite segmentación oráculo")
        return decode(
            self._rows[example_id].audio_path.read_bytes(),
            allow_mono=False, allow_resample=False,
        ).example

    def audio_sha256(self, example_id: str) -> str:
        return self._rows[example_id].audio_sha256

    def fingerprint(self) -> str:
        return self.index.fingerprint


def freeze_private_index(index: CorpusIndex, path: Path = PRIVATE_ROOT / "corpus.csv") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["internal_id", "call_id", "label", "source", "group_id", "audio_sha256"])
        for row in index.rows:
            writer.writerow([
                row.internal_id, row.call_id, row.label, row.source, row.group_id, row.audio_sha256
            ])
    return path


def _commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def extract_corpus(
    index: CorpusIndex,
    spec: Mapping[str, Any],
    *,
    private_root: Path,
    feature_order: tuple[str, ...],
) -> C3Corpus:
    """Extrae un orden LFCC completo sobre un índice ya validado y congelado."""
    freeze_private_index(index, private_root / "corpus.csv")
    dataset = CombinedDataset(index)
    extraction_spec = dict(spec)
    extraction_spec["feature_order"] = list(feature_order)
    result = run_experiment(
        extraction_spec,
        dataset=dataset,
        cache_root=private_root / "cache",
        artifact_root=private_root / "artifacts",
        ledger_root=private_root / "experiments" / "runs",
        code_digest=effective_code_digest(ROOT),
        commit=_commit(),
        environment={
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "package_version": "0.1.0",
        },
        code_version="0.1.0",
    )
    artifacts = private_root / "artifacts" / result.run_id
    features: dict[str, dict[str, float]] = {}
    labels, groups = {}, {}
    for row in index.rows:
        artifact = artifacts / f"{hashlib.sha256(row.internal_id.encode()).hexdigest()}.json"
        payload = json.loads(artifact.read_text(encoding="utf-8"))
        features[row.internal_id] = payload["features"]
        labels[row.internal_id] = row.label
        groups[row.internal_id] = row.group_id
    return C3Corpus(features, labels, groups, index.fingerprint)


def load_corpus(spec: Mapping[str, Any], *, private_root: Path = PRIVATE_ROOT) -> C3Corpus:
    """Congela el índice y extrae LFCC 120-dim; el builder selecciona las 100 de C3."""
    from altur.features.spectral_factory_lfcc import FEATURE_ORDER

    return extract_corpus(
        build_corpus_index(), spec, private_root=private_root, feature_order=FEATURE_ORDER
    )
