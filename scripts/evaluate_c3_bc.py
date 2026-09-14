"""Criterios D-A8.3 para C3 B+C.

Este módulo evalúa predicciones ya producidas sobre pares de WAV idénticos. B/B2 son un
diagnóstico quemado; B3 es una única comparación confirmatoria y nunca se fabrica aquí.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
import time
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from altur.features.spectral_factory_lfcc import FEATURE_ORDER
from altur.io import decode

FULL_FEATURE_ORDER = tuple(FEATURE_ORDER)
C3_FEATURE_ORDER = tuple(
    name for name in FULL_FEATURE_ORDER
    if not (".static" in name and name.endswith("_mean"))
)
EXPECTED_COUNTS = {"total": 372, "human": 128, "synthetic": 244}
DEFAULT_WORKSPACE = ROOT.parent / "cc_hc_hackmty26"


class C3EvaluationError(ValueError):
    """Entrada o fase incompatible con el preregistro D-A8.3."""


def select_flat_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Conserva exclusivamente B/B2 ``plana``; no duplica ni repondera filas."""
    selected = [dict(row) for row in rows if row.get("variant") == "plana"]
    if any(row.get("variant") not in {"plana", "g711"} for row in rows):
        raise C3EvaluationError("variante desconocida; se esperaba plana o g711")
    return selected


def validate_corpus_rows(rows: Sequence[Mapping[str, Any]], expected_counts: Mapping[str, int] | None = None) -> None:
    """Comprueba conteos y que cada fila pertenezca a un único grupo."""
    counts = {"total": len(rows), "human": sum(int(row["label"]) == 0 for row in rows),
              "synthetic": sum(int(row["label"]) == 1 for row in rows)}
    expected = dict(expected_counts or EXPECTED_COUNTS)
    for key, value in expected.items():
        if counts.get(key) != value:
            raise C3EvaluationError(f"conteo {key}={counts.get(key)}; se esperaba {value}")
    ids = [row.get("id", row.get("wav_id")) for row in rows]
    if any(not item for item in ids) or len(set(ids)) != len(ids):
        raise C3EvaluationError("cada llamada debe tener un id único")
    groups = [row.get("group_id") for row in rows]
    if any(not group for group in groups):
        raise C3EvaluationError("cada fila debe declarar group_id")


def phase_for_rows(rows: Sequence[Mapping[str, Any]]) -> str:
    sources = {str(row.get("source", "")) for row in rows}
    if sources and sources <= {"bloque_b", "bloque_b2", "B", "B2"}:
        return "diagnostic_b_b2_burned"
    if "b3" in {source.lower() for source in sources}:
        raise C3EvaluationError("B3 requiere manifest y paths explícitos; no se ejecuta por filas implícitas")
    raise C3EvaluationError("fuente no reconocida para D-A8.3")


def require_confirmatory_inputs(manifest: str | Path | None, paths: Sequence[str | Path] | None) -> None:
    if manifest is None or paths is None or not paths:
        raise C3EvaluationError("B3 confirmatorio requiere manifest y paths explícitos")
    if not Path(manifest).is_file() or any(not Path(path).is_file() for path in paths):
        raise C3EvaluationError("manifest y cada path de B3 deben existir y ser explícitos")


def reject_production_path(path: str | Path) -> None:
    if "models/current" in str(Path(path)).replace("\\", "/"):
        raise C3EvaluationError("D-A8.3 no permite models/current")


def ensure_report_path(path: str | Path) -> Path:
    reject_production_path(path)
    return Path(path)


def score_records(
    records: Sequence[Mapping[str, Any]], c2_detector: Any, c3_detector: Any
) -> list[dict[str, Any]]:
    """Puntúa exactamente los mismos bytes con ambos bundles y elimina las rutas del resultado."""
    rows = []
    for record in records:
        raw = Path(record["audio_path"]).read_bytes()
        example = decode(raw, allow_mono=False, allow_resample=False).example
        scored = {}
        for name, detector in (("c2", c2_detector), ("c3", c3_detector)):
            started = time.perf_counter()
            prediction = detector.predict(example)
            scored[f"{name}_latency_s"] = time.perf_counter() - started
            scored[f"{name}_score"] = float(prediction.diagnostics["p_synthetic"])
        rows.append({
            "wav_id": record["wav_id"],
            "label": int(record["label"]),
            "condition": str(record["condition"]),
            **scored,
        })
    return rows


def _csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise C3EvaluationError(f"no existe {path}")
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def diagnostic_records(
    block_b_root: Path = DEFAULT_WORKSPACE / "bloque_b",
    block_b2_root: Path = DEFAULT_WORKSPACE / "bloque_b2",
) -> list[dict[str, Any]]:
    records = []
    b_labels = {row["anon_id"]: int(row["label"] == "synthetic")
                for row in _csv_rows(block_b_root / "manifest.csv")}
    for pair in _csv_rows(block_b_root / "pairs.csv"):
        for field in ("id_sintetica", "id_humana"):
            call_id = pair[field]
            if call_id not in b_labels:
                raise C3EvaluationError("pairs.csv no enlaza 1:1 con el manifest de B")
            records.append({
                "wav_id": hashlib.sha256(f"B:{call_id}".encode()).hexdigest(),
                "label": b_labels[call_id],
                "condition": f"B/{pair['variante']}",
                "audio_path": block_b_root / "audio" / f"{call_id}.wav",
            })
    b2_labels = {row["anon_id"]: int(row["label"] == "synthetic")
                 for row in _csv_rows(block_b2_root / "manifest.csv")}
    for mapped in _csv_rows(block_b2_root / "voices_map.csv"):
        call_id = mapped["anon_id"]
        if call_id not in b2_labels:
            raise C3EvaluationError("voices_map.csv no enlaza 1:1 con el manifest de B2")
        records.append({
            "wav_id": hashlib.sha256(f"B2:{call_id}".encode()).hexdigest(),
            "label": b2_labels[call_id],
            "condition": f"B2/{mapped['variante']}",
            "audio_path": block_b2_root / "audio" / f"{call_id}.wav",
        })
    if len(records) != 180 or any(not Path(row["audio_path"]).is_file() for row in records):
        raise C3EvaluationError("diagnóstico requiere exactamente 180 WAV existentes de B/B2")
    return records


def evaluate_diagnostic(
    c2_bundle: Path,
    c3_bundle: Path,
    *,
    block_b_root: Path = DEFAULT_WORKSPACE / "bloque_b",
    block_b2_root: Path = DEFAULT_WORKSPACE / "bloque_b2",
) -> dict[str, Any]:
    from altur import bundle as bundle_mod

    c2_detector, c2_problems = bundle_mod.load_detector(c2_bundle)
    c3_detector, c3_problems = bundle_mod.load_detector(c3_bundle)
    if c2_detector is None or c3_detector is None or c2_problems or c3_problems:
        raise C3EvaluationError(
            f"bundles inválidos: C2={c2_problems or []}, C3={c3_problems or []}"
        )
    rows = score_records(
        diagnostic_records(block_b_root, block_b2_root), c2_detector, c3_detector
    )
    report = evaluate_paired(
        rows,
        c2_threshold=bundle_mod.load_manifest(c2_bundle).threshold,
        c3_threshold=bundle_mod.load_manifest(c3_bundle).threshold,
    )
    report["phase"] = "diagnostic_b_b2_burned"
    report["limitations"] = [
        "B/B2 entraron al ajuste de C3; estas métricas son diagnóstico quemado, no validación.",
        "B3 no se crea, consulta ni ejecuta desde este modo.",
    ]
    return report


def _wilson(successes: int, n: int, z: float = 1.959963984540054) -> list[float]:
    if not n:
        return [None, None]
    p = successes / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return [max(0.0, center - half), min(1.0, center + half)]


def _rate(y: Sequence[int], predicted: Sequence[bool], positive: bool) -> dict[str, Any]:
    eligible = [i for i, label in enumerate(y) if bool(label) is positive]
    successes = sum(bool(predicted[i]) is positive for i in eligible)
    return {"value": successes / len(eligible) if eligible else None, "n": len(eligible),
            "successes": successes, "ci95": _wilson(successes, len(eligible))}


def _percentile(values: Sequence[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1 - weight) + ordered[upper] * weight)


def _paired_bootstrap(
    y: Sequence[int],
    a: Sequence[bool],
    b: Sequence[bool],
    positive: bool,
    *,
    repetitions: int = 10_000,
    seed: int = 20260912,
) -> list[float | None]:
    values = [int(bool(b[i]) is positive) - int(bool(a[i]) is positive)
              for i, label in enumerate(y) if bool(label) is positive]
    if not values:
        return [None, None]
    rng = random.Random(seed)
    means = [sum(rng.choices(values, k=len(values))) / len(values) for _ in range(repetitions)]
    return [_percentile(means, 0.025), _percentile(means, 0.975)]


def mcnemar_exact(c2_pred: Sequence[bool], c3_pred: Sequence[bool], y: Sequence[int] | None = None) -> dict[str, Any]:
    """McNemar bilateral exact p-value, pooled over the supplied paired observations."""
    if len(c2_pred) != len(c3_pred) or (y is not None and len(y) != len(c2_pred)):
        raise C3EvaluationError("McNemar requiere vectores pareados del mismo tamaño")
    if y is not None:
        c2_pred = [bool(p) == bool(label) for p, label in zip(c2_pred, y)]
        c3_pred = [bool(p) == bool(label) for p, label in zip(c3_pred, y)]
    b = sum(a and not c for a, c in zip(c2_pred, c3_pred))
    c = sum(not a and c for a, c in zip(c2_pred, c3_pred))
    n = b + c
    tail = sum(math.comb(n, k) for k in range(min(b, c) + 1)) / (2**n) if n else 1.0
    return {"c2_only": b, "c3_only": c, "discordant_total": n, "p_value": min(1.0, 2 * tail)}


def _condition(
    rows: Sequence[Mapping[str, Any]], c2_threshold: float, c3_threshold: float
) -> dict[str, Any]:
    y = [int(row["label"]) for row in rows]
    c2 = [float(row["c2_score"]) for row in rows]
    c3 = [float(row["c3_score"]) for row in rows]
    p2 = [score > c2_threshold for score in c2]
    p3 = [score > c3_threshold for score in c3]
    out: dict[str, Any] = {"n": len(rows), "c2": {}, "c3": {}, "difference": {}}
    for name, scores, pred in (("c2", c2, p2), ("c3", c3, p3)):
        out[name] = {"tpr": _rate(y, pred, True), "tnr": _rate(y, pred, False),
                     "brier": mean((score - label) ** 2 for score, label in zip(scores, y))}
    for metric, positive in (("tpr", True), ("tnr", False)):
        left = out["c2"][metric]["value"]
        right = out["c3"][metric]["value"]
        out["difference"][metric] = {"value": None if left is None or right is None else right - left,
                                      "ci95": _paired_bootstrap(y, p2, p3, positive)}
    out["calibration"] = {name: {"mean_score": mean(scores), "mean_label": mean(y)}
                          for name, scores in (("c2", c2), ("c3", c3))}
    out["mcnemar"] = mcnemar_exact(p2, p3, y)
    latencies = {name: [float(row[name]) for row in rows if name in row]
                 for name in ("c2_latency_s", "c3_latency_s")}
    out["latency_s"] = {name: {"p50": _percentile(values, .5),
                               "p95": _percentile(values, .95), "max": max(values)}
                         for name, values in latencies.items() if values}
    return out


def evaluate_paired(
    rows: Sequence[Mapping[str, Any]],
    *,
    c2_threshold: float = 0.5,
    c3_threshold: float = 0.5,
) -> dict[str, Any]:
    """Evalúa condiciones pareadas y devuelve métricas descriptivas sin cambiar política."""
    if not rows:
        raise C3EvaluationError("no hay predicciones pareadas")
    required = {"wav_id", "label", "condition", "c2_score", "c3_score"}
    for row in rows:
        if not required <= set(row):
            raise C3EvaluationError(f"faltan columnas pareadas: {sorted(required - set(row))}")
    ids = [(str(row["condition"]), row["wav_id"]) for row in rows]
    if len(set(ids)) != len(ids):
        raise C3EvaluationError("cada WAV debe aparecer una sola vez por condición")
    conditions = {str(row["condition"]): _condition(
        [r for r in rows if str(r["condition"]) == str(row["condition"])],
        c2_threshold, c3_threshold,
    )
                   for row in rows}
    return {"phase": "confirmatory_or_explicit_input",
            "thresholds": {"c2": c2_threshold, "c3": c3_threshold},
            "conditions": conditions,
            "mcnemar": mcnemar_exact(
                [float(row["c2_score"]) > c2_threshold for row in rows],
                [float(row["c3_score"]) > c3_threshold for row in rows],
                [int(row["label"]) for row in rows]),
            "limitations": ["B3 no se crea ni se ejecuta; métricas requieren manifest y paths externos"]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("diagnostic", "confirmatory"), required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--wav", action="append", type=Path, default=[])
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--c2-threshold", type=float, default=0.5)
    parser.add_argument("--c3-threshold", type=float, default=0.5)
    parser.add_argument("--c2-bundle", type=Path, default=ROOT / "models" / "current")
    parser.add_argument(
        "--c3-bundle", type=Path, default=ROOT / "models" / "candidates" / "c3_bc_v1"
    )
    parser.add_argument("--block-b-root", type=Path, default=DEFAULT_WORKSPACE / "bloque_b")
    parser.add_argument("--block-b2-root", type=Path, default=DEFAULT_WORKSPACE / "bloque_b2")
    args = parser.parse_args(argv)
    try:
        if args.phase == "confirmatory":
            require_confirmatory_inputs(args.manifest, args.wav)
        if args.output:
            ensure_report_path(args.output)
        if args.predictions:
            data = json.loads(args.predictions.read_text(encoding="utf-8"))
            report = evaluate_paired(
                data["rows"] if isinstance(data, dict) else data,
                c2_threshold=args.c2_threshold,
                c3_threshold=args.c3_threshold,
            )
        elif args.phase == "diagnostic":
            report = evaluate_diagnostic(
                args.c2_bundle,
                args.c3_bundle,
                block_b_root=args.block_b_root,
                block_b2_root=args.block_b2_root,
            )
        else:
            report = {"phase": "confirmatory",
                      "status": "B3_not_available",
                      "message": "No se inventa B3 ni se ejecuta sin predicciones pareadas explícitas."}
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))
        return 0
    except (C3EvaluationError, OSError, json.JSONDecodeError) as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
