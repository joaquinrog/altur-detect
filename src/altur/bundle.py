"""Bundle de despliegue: un directorio versionado, no un `joblib` suelto.

Corrección P1.3 de la revisión. Un pickle suelto rompe en cuanto cambia una versión de
librería o entra una dependencia nativa, y no lleva consigo lo que hace falta para
auditarlo. El bundle lleva manifest + hashes + lockfile + schema y ORDEN de features.

Qué contiene un bundle entrenado:

    manifest.json       identidad, procedencia, refs con versión, hashes de todo lo demás
    model.json          `LinearExport`: feature_order + mean/scale/coef/intercept
    calibrator.json     `PlattCalibrator`: a, b
    requirements.lock   las dependencias de inferencia, fijadas

Nada de eso necesita sklearn para cargarse, y esa es la propiedad que hace la imagen chica.

Lo que el bundle **rechaza**, cada cosa por una razón:

- Extractores `product_safe=False` — openSMILE es research-only y Parselmouth es GPL-3.
- Referencias flotantes (`nombre` sin `@version`) — dos despliegues del mismo manifest
  podrían servir código distinto sin que el manifest cambie.
- Mismatch de `schema_version` entre manifest, export y calibrador.
- Un `feature_order` en el manifest distinto del que lleva el modelo.
- Rutas locales, secretos o `anon_id` dentro del manifest. El manifest se publica.

Prueba obligatoria antes de congelar: construir, guardar, cargar y predecir en un
contenedor limpio (`make bundle-test`).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import UTC
from pathlib import Path
from typing import Any

MANIFEST = "manifest.json"
MODEL_FILE = "model.json"
CALIBRATOR_FILE = "calibrator.json"
REQUIREMENTS_FILE = "requirements.lock"
SCHEMA_VERSION = 2

# El manifest se publica: va dentro de la imagen y lo sirve `/version`. Estos patrones son
# lo que NUNCA debe salir por ahí. `anon_id` está primero a propósito — es el que de verdad
# tiene consecuencias legales (los términos de Altur prohíben redistribuir).
_ANON_ID = re.compile(r"call_[0-9a-f]{8,}")
_ABS_PATH = re.compile(r"(^|[\s\"'=])(/home/|/Users/|/root/|/tmp/|[A-Za-z]:\\)")
_SECRET_KEY = re.compile(r"(token|secret|password|passwd|credential|api_?key|private_?key)", re.IGNORECASE)


class BundleError(RuntimeError):
    """El bundle no se puede construir o no se puede confiar en él."""


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


def _versioned(ref: str, what: str) -> str:
    """Una referencia de bundle SIEMPRE lleva versión. Sin excepciones."""
    if not isinstance(ref, str) or ref.count("@") != 1:
        raise BundleError(
            f"{what} debe ser una referencia explícita nombre@version; llegó {ref!r}"
        )
    base, version = ref.rsplit("@", 1)
    if not base or not version.isdigit():
        raise BundleError(f"{what} debe ser nombre@version; llegó {ref!r}")
    return ref


def scrub_problems(manifest_json: str) -> list[str]:
    """Qué hay en el manifest que no debería publicarse. Vacío = se puede publicar."""
    problems: list[str] = []
    if _ANON_ID.search(manifest_json):
        ids = sorted(set(_ANON_ID.findall(manifest_json)))
        problems.append(f"el manifest contiene {len(ids)} anon_id; no se publica ninguno")
    if _ABS_PATH.search(manifest_json):
        problems.append("el manifest contiene una ruta local absoluta")
    try:
        data = json.loads(manifest_json)
    except json.JSONDecodeError:
        problems.append("el manifest no es JSON válido")
        return problems

    def walk(node: Any, path: str) -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                if _SECRET_KEY.search(str(k)):
                    problems.append(f"clave con pinta de secreto en el manifest: {path}{k}")
                walk(v, f"{path}{k}.")
        elif isinstance(node, list):
            for item in node:
                walk(item, path)

    walk(data, "")
    return problems


@dataclass
class BundleManifest:
    """Todo lo necesario para auditar qué está sirviendo el endpoint.

    Cada campo existe porque su ausencia produce un modo de falla concreto: sin
    `feature_order` el vector se puede permutar, sin `data_fingerprint` no se sabe contra
    qué datos se midió, sin `code_digest` dos corridas del mismo commit con el árbol sucio
    son indistinguibles.
    """

    detector_name: str
    schema_version: int = SCHEMA_VERSION
    created_utc: str = ""
    git_commit: str = field(default_factory=git_commit)
    git_dirty: bool = field(default_factory=git_dirty)
    code_digest: str | None = None
    data_fingerprint: str | None = None
    protocol_id: str | None = None
    run_id: str | None = None

    # El contrato de features. ORDEN, no solo nombres.
    feature_order: list[str] = field(default_factory=list)
    extractor_refs: list[str] = field(default_factory=list)
    segmenter_ref: str | None = None
    transform_refs: list[str] = field(default_factory=list)

    # Modelo y política de decisión.
    model_file: str = MODEL_FILE
    calibrator_file: str | None = CALIBRATOR_FILE
    calibrator_ref: str | None = None
    threshold: float = 0.5
    confidence_policy: str = "calibrated_probability_of_reported_class"
    min_seconds_declared: float | None = None

    files: dict[str, str] = field(default_factory=dict)      # ruta relativa -> sha256
    metrics: dict[str, Any] = field(default_factory=dict)
    limitations: list[str] = field(default_factory=list)     # se declaran, no se ocultan

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, ensure_ascii=False, sort_keys=True)

    def validate(self) -> None:
        """Lo que se comprueba al construir Y al cargar. Fail-closed en ambos lados."""
        if self.schema_version != SCHEMA_VERSION:
            raise BundleError(
                f"schema de bundle {self.schema_version} != {SCHEMA_VERSION} soportado"
            )
        if not self.detector_name:
            raise BundleError("el bundle no declara detector_name")
        if self.detector_name.startswith("constant"):
            return  # el bundle de referencia no lleva modelo ni features
        if not self.feature_order:
            raise BundleError("el bundle no declara feature_order")
        if len(set(self.feature_order)) != len(self.feature_order):
            raise BundleError("feature_order tiene nombres repetidos")
        if not self.extractor_refs:
            raise BundleError("el bundle no declara extractores")
        for ref in self.extractor_refs:
            _versioned(ref, "extractor_ref")
        for ref in self.transform_refs:
            _versioned(ref, "transform_ref")
        if self.segmenter_ref is not None:
            _versioned(self.segmenter_ref, "segmenter_ref")
        if self.calibrator_ref is not None:
            _versioned(self.calibrator_ref, "calibrator_ref")
        if not 0.0 < float(self.threshold) < 1.0:
            raise BundleError(f"umbral fuera de (0, 1): {self.threshold!r}")


def save_manifest(bundle_dir: str | os.PathLike, manifest: BundleManifest) -> Path:
    """Sella el bundle: hashea todo lo que hay dentro y escribe el manifest.

    Se hace AL FINAL, cuando los demás artefactos ya están en disco: el manifest es la
    lista de lo que hay, así que escribirlo antes produciría una lista incompleta que
    `verify()` aceptaría.
    """
    from datetime import datetime

    d = Path(bundle_dir)
    d.mkdir(parents=True, exist_ok=True)
    manifest.validate()
    if not manifest.created_utc:
        manifest.created_utc = datetime.now(UTC).isoformat(timespec="seconds")
    manifest.files = {
        str(p.relative_to(d)): sha256_file(p)
        for p in sorted(d.rglob("*"))
        if p.is_file() and p.name != MANIFEST
    }
    payload = manifest.to_json()
    problems = scrub_problems(payload)
    if problems:
        raise BundleError(f"el manifest no se puede publicar: {problems}")
    path = d / MANIFEST
    path.write_text(payload, encoding="utf-8")
    return path


def load_manifest(bundle_dir: str | os.PathLike) -> BundleManifest:
    d = Path(bundle_dir)
    data = json.loads((d / MANIFEST).read_text(encoding="utf-8"))
    if data.get("schema_version") != SCHEMA_VERSION:
        raise BundleError(
            f"schema de bundle {data.get('schema_version')} != {SCHEMA_VERSION} soportado"
        )
    known = set(BundleManifest.__dataclass_fields__)
    unknown = set(data) - known
    if unknown:
        raise BundleError(f"el manifest trae campos desconocidos: {sorted(unknown)}")
    m = BundleManifest(**{k: v for k, v in data.items() if k in known})
    m.validate()
    return m


def verify(bundle_dir: str | os.PathLike) -> list[str]:
    """Devuelve la lista de problemas. Vacía = el bundle es íntegro.

    Nunca lanza: el llamador decide qué hacer con un bundle roto, y en `/detect` esa
    decisión es 503, no una excepción que tumbe el arranque.
    """
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
    if problems:
        return problems

    # Integridad de contenido, no solo de bytes: el manifest y el modelo tienen que estar
    # de acuerdo sobre el orden de las features. Un manifest correcto apuntando a un
    # modelo de otro experimento es un bundle invertido con todos sus hashes en verde.
    if not m.detector_name.startswith("constant"):
        try:
            from .models.base import LinearExport

            export = LinearExport.load(d / m.model_file)
        except Exception as e:  # noqa: BLE001
            return [f"no se pudo cargar {m.model_file}: {e}"]
        if list(export.feature_order) != list(m.feature_order):
            problems.append(
                "el feature_order del manifest no coincide con el del modelo: "
                f"{len(m.feature_order)} vs {len(export.feature_order)} nombres"
            )
    return problems


def build_bundle(
    bundle_dir: str | os.PathLike,
    *,
    export: Any,
    manifest: BundleManifest,
    calibrator: Any | None = None,
    requirements: str | None = None,
) -> Path:
    """Escribe un bundle completo y lo sella. Falla antes de escribir si algo no cuadra."""
    # Importar `serving` registra los plugins servibles. Es la coupling correcta: solo se
    # puede empaquetar lo que se puede servir, y auditar la licencia contra un registro
    # vacío daría un verde que no significa nada.
    from . import serving  # noqa: F401
    from .registry import audit_product_safety

    d = Path(bundle_dir)
    manifest.validate()

    if list(export.feature_order) != list(manifest.feature_order):
        raise BundleError(
            "el feature_order del manifest no coincide con el del export lineal; "
            "casi siempre es un manifest copiado de otro experimento"
        )
    no_aptos = audit_product_safety(list(manifest.extractor_refs))
    if no_aptos:
        raise BundleError(
            f"extractores no aptos para producto: {no_aptos}. "
            "El empaquetado rechaza congelarlos en /detect."
        )

    d.mkdir(parents=True, exist_ok=True)
    export.save(d / manifest.model_file)

    if calibrator is not None:
        if manifest.calibrator_file is None:
            raise BundleError("hay calibrador pero el manifest no declara calibrator_file")
        (d / manifest.calibrator_file).write_text(
            json.dumps(calibrator.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    else:
        manifest.calibrator_file = None
        manifest.calibrator_ref = None

    if requirements is not None:
        (d / REQUIREMENTS_FILE).write_text(requirements, encoding="utf-8")

    return save_manifest(d, manifest)


def load_detector(bundle_dir: str | os.PathLike | None, *, strict: bool = True):
    """Carga el detector del bundle.

    `strict=True` (el default y lo que usa `/detect`): si se PIDIÓ un bundle y no se puede
    servir, devuelve `(None, problemas)`. El endpoint lo traduce a readiness 503. Nunca se
    cae en silencio a predicciones constantes — una constante que se lee como un modelo es
    peor que un 503, porque el 503 se ve y la constante no.

    `strict=False` solo existe para el modo de emergencia explícito del runbook.
    """
    from .detector import ConstantDetector

    if not bundle_dir:
        # Sin bundle configurado no hay nada que fallar: es el arranque de A0, el endpoint
        # de pie antes de que exista un modelo.
        return ConstantDetector(name="constant@1"), ["sin bundle: usando ConstantDetector"]

    d = Path(bundle_dir)
    if not d.exists():
        problems = ["el bundle configurado no existe"]
        return (None, problems) if strict else (ConstantDetector(name="constant@1"), problems)

    problems = verify(d)
    if problems:
        return (None, problems) if strict else (ConstantDetector(name="constant@1"), problems)

    try:
        m = load_manifest(d)
        if m.detector_name.startswith("constant"):
            return ConstantDetector(name=m.detector_name), []

        from .models.base import LinearExport
        from .serving import BundledDetector

        export = LinearExport.load(d / m.model_file)
        calibrator = None
        if m.calibrator_file:
            from .models.calibration import PlattCalibrator

            payload = json.loads((d / m.calibrator_file).read_text(encoding="utf-8"))
            calibrator = PlattCalibrator.from_dict(payload)

        det = BundledDetector(
            name=m.detector_name,
            export=export,
            extractor_refs=m.extractor_refs,
            segmenter_ref=m.segmenter_ref,
            calibrator=calibrator,
            threshold=m.threshold,
            min_seconds_declared=m.min_seconds_declared,
        )
        return det, []
    except Exception as e:  # noqa: BLE001 — frontera: un bundle malo es 503, no un crash
        problems = [f"el bundle no se pudo cargar: {type(e).__name__}: {e}"]
        return (None, problems) if strict else (ConstantDetector(name="constant@1"), problems)
