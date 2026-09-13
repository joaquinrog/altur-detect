"""Leak-resistant, deterministic experiment runner for versioned plugins."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from . import transforms as builtin_transforms
from . import transforms_v2 as builtin_transforms_v2
from .cache import FeatureCache, cache_key
from .features import acoustic_minimal, behavioral, spectral_factory_lfcc
from .ledger import write_aggregate, write_run
from .registry import RegistryError, extractors, transforms, turn_sources
from .seg import spectral_factory_vad, vad
from .types import AudioExample

# Imports are deliberately explicit: registration must never depend on filesystem discovery.
_BUILTIN_PLUGINS = (
    acoustic_minimal, behavioral, spectral_factory_lfcc, vad, spectral_factory_vad,
    builtin_transforms, builtin_transforms_v2,
)


class RunnerError(RuntimeError):
    """Invalid experiment spec or plugin output."""


@dataclass(frozen=True)
class RunResult:
    run_id: str
    summary: dict[str, Any]
    ledger_path: Path


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise RunnerError("spec must be strict JSON") from exc


def _versioned_ref(value: Any, name: str) -> str:
    if not isinstance(value, str) or value.count("@") != 1:
        raise RunnerError(f"{name} must be an explicit name@version reference")
    base, version = value.rsplit("@", 1)
    if not base or not version.isdigit():
        raise RunnerError(f"{name} must be an explicit name@version reference")
    return value


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    data = _canonical_bytes(dict(payload)) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _validate_features(
    features: Any, diagnostics: Any, *, namespace: str, order: tuple[str, ...]
) -> tuple[dict[str, float], dict[str, Any]]:
    if not isinstance(features, dict) or tuple(features) != order:
        raise RunnerError("extractor feature keys must exactly match feature_order")
    prefix = f"{namespace}."
    if not all(name.startswith(prefix) for name in features):
        raise RunnerError(f"feature keys must use namespace {namespace!r}")
    normalized: dict[str, float] = {}
    for name, value in features.items():
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
        ):
            raise RunnerError(f"feature {name!r} must be finite and numeric")
        normalized[name] = float(value)
    if not isinstance(diagnostics, dict):
        raise RunnerError("extractor diagnostics must be a dict")
    try:
        _canonical_bytes(diagnostics)
    except RunnerError as exc:
        raise RunnerError("extractor diagnostics must be finite strict JSON") from exc
    return normalized, dict(diagnostics)


def validate_experiment_spec(spec: Mapping[str, Any]) -> None:
    """Fail closed on the A2 experiment contract before touching dataset audio."""
    if not isinstance(spec, Mapping):
        raise RunnerError("spec must be a mapping")
    required = {
        "schema_version", "extractor", "segmenter", "feature_order", "dtype",
        "nan_policy", "split", "random_seed", "transforms",
    }
    allowed = required | {"extractor_params"}
    missing = required - set(spec)
    unknown = set(spec) - allowed
    if missing:
        raise RunnerError(f"spec is missing fields: {sorted(missing)}")
    if unknown:
        raise RunnerError(f"spec has unknown fields: {sorted(unknown)}")
    if spec["schema_version"] != 1:
        raise RunnerError("unsupported spec schema_version")
    if spec["split"] != "train":
        raise RunnerError("A2 split must be exactly 'train'")
    _versioned_ref(spec["extractor"], "extractor")
    _versioned_ref(spec["segmenter"], "segmenter")
    if not isinstance(spec["transforms"], Sequence) or isinstance(
        spec["transforms"], (str, bytes, bytearray)
    ):
        raise RunnerError("transforms must be an ordered sequence")
    transform_refs = tuple(
        _versioned_ref(ref, f"transforms[{index}]")
        for index, ref in enumerate(spec["transforms"])
    )
    if spec["dtype"] != "float64" or spec["nan_policy"] != "reject":
        raise RunnerError("A2 requires dtype 'float64' and nan_policy 'reject'")
    if not isinstance(spec["random_seed"], int) or isinstance(spec["random_seed"], bool):
        raise RunnerError("random_seed must be an integer")
    if "extractor_params" in spec and not isinstance(spec["extractor_params"], Mapping):
        raise RunnerError("extractor_params must be a mapping")
    if not isinstance(spec["feature_order"], Sequence) or isinstance(
        spec["feature_order"], (str, bytes, bytearray)
    ):
        raise RunnerError("feature_order must be an ordered sequence")
    feature_order = tuple(spec["feature_order"])
    if not feature_order or not all(isinstance(name, str) and name for name in feature_order):
        raise RunnerError("feature_order must contain non-empty strings")
    if len(set(feature_order)) != len(feature_order):
        raise RunnerError("feature_order must not contain duplicates")
    # This also rejects non-finite values and unsupported YAML/JSON structures early.
    _canonical_bytes(dict(spec))

    try:
        extractor_entry = extractors.get(spec["extractor"])
        turn_sources.get(spec["segmenter"])
        transform_entries = tuple(transforms.get(ref) for ref in transform_refs)
    except RegistryError as exc:
        raise RunnerError(str(exc)) from exc
    if any(entry.meta["use"] != "holdout" for entry in transform_entries):
        raise RunnerError("A2 diagnostic transforms must be registered as holdout")
    if not all(name.startswith(f"{extractor_entry.name}.") for name in feature_order):
        raise RunnerError("feature_order does not match the extractor namespace")


def run_experiment(
    spec: Mapping[str, Any],
    *,
    dataset: Any,
    cache_root: str | Path,
    artifact_root: str | Path,
    ledger_root: str | Path,
    code_digest: str,
    commit: str,
    environment: Mapping[str, Any],
    code_version: str,
) -> RunResult:
    """Extract one split while keeping dataset metadata outside every plugin call."""
    validate_experiment_spec(spec)
    extractor_ref = spec["extractor"]
    segmenter_ref = spec["segmenter"]
    feature_order = tuple(spec["feature_order"])

    try:
        extractor_entry = extractors.get(extractor_ref)
        segmenter_entry = turn_sources.get(segmenter_ref)
        transform_entries = tuple(transforms.get(ref) for ref in spec["transforms"])
    except RegistryError as exc:
        raise RunnerError(str(exc)) from exc
    transform_components = [
        {
            "id": entry.name,
            "version": entry.version,
            "params": dict(entry.meta),
        }
        for entry in transform_entries
    ]
    dataset_fingerprint = dataset.fingerprint()
    identity = {
        "schema_version": 1,
        "spec": dict(spec),
        "effective_plugins": {
            "extractor": {
                "id": extractor_entry.name,
                "version": extractor_entry.version,
                "params": dict(spec.get("extractor_params", {})),
            },
            "segmenter": {
                "id": segmenter_entry.name,
                "version": segmenter_entry.version,
                "params": dict(segmenter_entry.meta),
            },
            "transforms": transform_components,
        },
        "dataset_fingerprint": dataset_fingerprint,
        "code_digest": code_digest,
        "commit": commit,
    }
    run_id = f"sha256-v1-{hashlib.sha256(_canonical_bytes(identity)).hexdigest()}"
    cache = FeatureCache(cache_root)
    rows: list[dict[str, float]] = []
    artifact_directory = Path(artifact_root) / run_id

    for example_id in dataset.ids_for(spec["split"]):
        record = dataset.record(example_id)
        if record.split != spec["split"]:
            raise RunnerError("dataset returned a record from the wrong split")
        with_turns = bool(segmenter_entry.meta["is_oracle"])
        loaded = dataset.load(example_id, with_turns=with_turns)
        if type(loaded) is not AudioExample:
            raise RunnerError("dataset.load must return exactly AudioExample")

        audio_hash = dataset.audio_sha256(example_id)
        try:
            audio_identity = bytes.fromhex(audio_hash)
        except (TypeError, ValueError) as exc:
            raise RunnerError("dataset audio_sha256 must be a hexadecimal digest") from exc
        seed_material = _canonical_bytes(
            {"audio_sha256": audio_hash, "random_seed": spec["random_seed"]}
        )
        rng_seed = int.from_bytes(hashlib.sha256(seed_material).digest(), "big")
        rng = np.random.default_rng(rng_seed)
        transformed = loaded
        for entry in transform_entries:
            transformed = entry.obj(transformed, rng)
            if type(transformed) is not AudioExample:
                raise RunnerError("transform must return exactly AudioExample")
        segmentation = segmenter_entry.obj(transformed)
        example = AudioExample(
            transformed.ch0, transformed.ch1, sr=transformed.sr, seg=segmentation
        )
        key = cache_key(
            audio_bytes=audio_identity,
            segmenter={
                "id": segmenter_entry.name,
                "version": segmenter_entry.version,
                "params": dict(segmenter_entry.meta),
            },
            transforms=transform_components,
            random_seed=spec["random_seed"],
            extractor={
                "id": extractor_entry.name,
                "version": extractor_entry.version,
                "params": dict(spec.get("extractor_params", {})),
                "schema": {"names": list(feature_order)},
            },
            feature_order=feature_order,
            nan_policy=spec["nan_policy"],
            dtype=spec["dtype"],
            code_digest=code_digest,
        )
        cached = cache.read(
            key,
            feature_order=feature_order,
            dtype=spec["dtype"],
            nan_policy=spec["nan_policy"],
        )
        if cached is None:
            started = time.perf_counter()
            result = extractor_entry.obj(example)
            if not isinstance(result, tuple) or len(result) != 2:
                raise RunnerError("extractor must return (features, diagnostics)")
            features, diagnostics = _validate_features(
                result[0], result[1], namespace=extractor_entry.name, order=feature_order
            )
            diagnostics["runner.extraction_ms"] = (time.perf_counter() - started) * 1000
            cache.write(
                key,
                features=features,
                diagnostics=diagnostics,
                feature_order=feature_order,
                dtype=spec["dtype"],
                nan_policy=spec["nan_policy"],
            )
        else:
            features, diagnostics = cached
        rows.append(features)
        private_name = hashlib.sha256(example_id.encode("utf-8")).hexdigest()
        _atomic_json(
            artifact_directory / f"{private_name}.json",
            {
                "schema_version": 1,
                "example_id": example_id,
                "label": record.label,
                "features": features,
                "diagnostics": diagnostics,
            },
        )

    summary = {
        "schema_version": 1,
        "n": len(rows),
        "split": spec["split"],
        "feature_order": list(feature_order),
        "feature_means": {
            name: sum(row[name] for row in rows) / len(rows) if rows else None
            for name in feature_order
        },
    }
    ledger_payload = {
        "commit": commit,
        "data_hash": dataset_fingerprint,
        "environment": dict(environment),
        "code_version": code_version,
        "code_digest": code_digest,
        "spec": dict(spec),
        "results": summary,
    }
    ledger_path = write_run(run_id, ledger_payload, root=ledger_root)
    write_aggregate(root=ledger_root)
    return RunResult(run_id=run_id, summary=summary, ledger_path=ledger_path)
