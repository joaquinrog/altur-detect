"""Bundle de despliegue: un directorio versionado, no un `joblib` suelto.

Corrección P1.3 de la revisión. Un pickle suelto rompe en cuanto cambia una versión de
librería o entra una dependencia nativa, y no lleva consigo lo que hace falta para
auditarlo. El bundle lleva manifest + hashes + lockfile + schema y ORDEN de features.

Prueba obligatoria antes de congelar: construir, guardar, cargar y predecir en un
contenedor limpio (`make bundle-test`).
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import UTC
from pathlib import Path
from typing import Any

MANIFEST = "manifest.json"
SCHEMA_VERSION = 1


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], capture_output=True, text=True, timeout=5, check=False,
            cwd=Path(__file__).resolve().parents[2],
        ).stdout.strip()
    except Exception:  # noqa: BLE001 — sin git disponible el bundle sigue siendo válido
        return ""


def git_commit() -> str:
    return _git("rev-parse", "HEAD") or "unknown"


def git_dirty() -> bool:
    return bool(_git("status", "--porcelain"))


def sha256_file(path: str | os.PathLike) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class BundleManifest:
    """Todo lo necesario para auditar qué está sirviendo el endpoint."""

    detector_name: str
    schema_version: int = SCHEMA_VERSION
    created_utc: str = ""
    git_commit: str = field(default_factory=git_commit)
    git_dirty: bool = field(default_factory=git_dirty)
    protocol_id: str | None = None
    run_id: str | None = None
    feature_order: list[str] = field(default_factory=list)   # ORDEN, no solo nombres
    extractor_refs: list[str] = field(default_factory=list)
    threshold: float = 0.5
    confidence_policy: str = "calibrated_probability"
    files: dict[str, str] = field(default_factory=dict)      # ruta relativa -> sha256
    metrics: dict[str, Any] = field(default_factory=dict)
    limitations: list[str] = field(default_factory=list)     # se declaran, no se ocultan

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, ensure_ascii=False, sort_keys=True)


def save_manifest(bundle_dir: str | os.PathLike, manifest: BundleManifest) -> Path:
    from datetime import datetime

    d = Path(bundle_dir)
    d.mkdir(parents=True, exist_ok=True)
    if not manifest.created_utc:
        manifest.created_utc = datetime.now(UTC).isoformat(timespec="seconds")
    manifest.files = {
        str(p.relative_to(d)): sha256_file(p)
        for p in sorted(d.rglob("*"))
        if p.is_file() and p.name != MANIFEST
    }
    path = d / MANIFEST
    path.write_text(manifest.to_json(), encoding="utf-8")
    return path


def load_manifest(bundle_dir: str | os.PathLike) -> BundleManifest:
    d = Path(bundle_dir)
    data = json.loads((d / MANIFEST).read_text(encoding="utf-8"))
    if data.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"schema de bundle {data.get('schema_version')} != {SCHEMA_VERSION} soportado"
        )
    known = {f for f in BundleManifest.__dataclass_fields__}
    return BundleManifest(**{k: v for k, v in data.items() if k in known})


def verify(bundle_dir: str | os.PathLike) -> list[str]:
    """Devuelve la lista de problemas. Vacía = el bundle es íntegro."""
    d = Path(bundle_dir)
    problems: list[str] = []
    if not (d / MANIFEST).exists():
        return [f"falta {MANIFEST}"]
    try:
        m = load_manifest(d)
    except Exception as e:  # noqa: BLE001 — verify() reporta problemas, nunca lanza
        return [f"manifest ilegible: {e}"]
    for rel, expected in m.files.items():
        p = d / rel
        if not p.exists():
            problems.append(f"falta el archivo {rel}")
        elif sha256_file(p) != expected:
            problems.append(f"hash distinto en {rel}")
    on_disk = {
        str(p.relative_to(d)) for p in d.rglob("*") if p.is_file() and p.name != MANIFEST
    }
    for extra in sorted(on_disk - set(m.files)):
        problems.append(f"archivo no declarado en el manifest: {extra}")
    return problems


def load_detector(bundle_dir: str | os.PathLike | None):
    """Carga el detector del bundle. Sin bundle o bundle roto -> `ConstantDetector`.

    Nunca se carga un artefacto de origen no confiable: el bundle lo produce este repo.
    """
    from .detector import ConstantDetector

    if not bundle_dir or not Path(bundle_dir).exists():
        return ConstantDetector(name="constant@1"), ["sin bundle: usando ConstantDetector"]
    problems = verify(bundle_dir)
    if problems:
        return ConstantDetector(name="constant@1"), problems
    m = load_manifest(bundle_dir)
    if m.detector_name.startswith("constant"):
        return ConstantDetector(name=m.detector_name), []
    # Los detectores entrenados se registran aquí conforme existan (Fase A3+).
    return ConstantDetector(name="constant@1"), [
        f"detector '{m.detector_name}' no implementado todavía"
    ]
