"""FACT: cache identities must include every input that can alter features."""

import base64
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


def _canonical(value: Any) -> Any:
    if value is None:
        return ["null"]
    if isinstance(value, bool):
        return ["bool", value]
    if isinstance(value, int):
        return ["int", str(value)]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite floats are not valid cache inputs")
        return ["float", value.hex()]
    if isinstance(value, str):
        return ["str", value]
    if isinstance(value, bytes):
        return ["bytes", base64.b64encode(value).decode("ascii")]
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("cache mappings require string keys")
        return ["mapping", [[key, _canonical(value[key])] for key in sorted(value)]]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return ["sequence", [_canonical(item) for item in value]]
    raise TypeError(f"unsupported cache input type: {type(value).__name__}")


def _component(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    required = {"id", "version", "params"}
    if name == "extractor":
        required.add("schema")
    if set(value) != required:
        raise ValueError(f"{name} must contain exactly {sorted(required)}")
    if not isinstance(value["id"], str) or not value["id"]:
        raise ValueError(f"{name}.id must be a non-empty string")
    if not isinstance(value["version"], int) or isinstance(value["version"], bool) or value["version"] < 0:
        raise ValueError(f"{name}.version must be a non-negative integer")
    return dict(value)


def cache_key(
    *,
    audio_bytes: bytes,
    segmenter: Mapping[str, Any],
    transforms: Sequence[Mapping[str, Any]],
    random_seed: int | None = None,
    random_bytes: bytes | None = None,
    extractor: Mapping[str, Any],
    feature_order: Sequence[str],
    nan_policy: str,
    dtype: str,
    code_digest: str,
) -> str:
    """Return a deterministic, versioned SHA-256 identity for one feature computation."""
    if not isinstance(audio_bytes, bytes):
        raise TypeError("audio_bytes must be bytes")
    if not isinstance(transforms, Sequence) or isinstance(transforms, (str, bytes, bytearray)):
        raise TypeError("transforms must be an ordered sequence")
    if (random_seed is None) == (random_bytes is None):
        raise ValueError("provide exactly one of random_seed or random_bytes")
    if random_seed is not None and (not isinstance(random_seed, int) or isinstance(random_seed, bool)):
        raise TypeError("random_seed must be an integer")
    if random_bytes is not None and not isinstance(random_bytes, bytes):
        raise TypeError("random_bytes must be bytes")
    if not isinstance(feature_order, Sequence) or isinstance(feature_order, (str, bytes, bytearray)):
        raise TypeError("feature_order must be an ordered sequence")
    if not all(isinstance(item, str) and item for item in feature_order):
        raise ValueError("feature_order must contain non-empty strings")
    if not isinstance(nan_policy, str) or not nan_policy:
        raise ValueError("nan_policy must be a non-empty string")
    if not isinstance(dtype, str) or not dtype:
        raise ValueError("dtype must be a non-empty string")
    if not isinstance(code_digest, str) or not code_digest:
        raise ValueError("code_digest must be a non-empty string")
    document = {
        "schema_version": 1,
        "audio": audio_bytes,
        "segmenter": _component(segmenter, "segmenter"),
        "transforms": [_component(item, "transform") for item in transforms],
        "randomness": {"seed": random_seed, "bytes": random_bytes},
        "extractor": _component(extractor, "extractor"),
        "feature_order": list(feature_order),
        "nan_policy": nan_policy,
        "dtype": dtype,
        "code_digest": code_digest,
    }
    encoded = json.dumps(_canonical(document), ensure_ascii=True, separators=(",", ":"), allow_nan=False).encode()
    return f"sha256:v1:{hashlib.sha256(encoded).hexdigest()}"


def _strict_json(value: Any, name: str) -> None:
    try:
        json.dumps(value, ensure_ascii=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain finite strict-JSON values") from exc


class FeatureCache:
    """Persistent feature rows with atomic publication and strict schema checks."""

    def __init__(self, root: str | Path = "cache/features") -> None:
        self.root = Path(root)

    def path_for(self, key: str) -> Path:
        prefix = "sha256:v1:"
        if not isinstance(key, str) or not key.startswith(prefix):
            raise ValueError("cache key must be a versioned SHA-256 identity")
        digest = key.removeprefix(prefix)
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError("cache key must contain a lowercase SHA-256 digest")
        return self.root / digest[:2] / f"{digest}.json"

    def write(
        self,
        key: str,
        *,
        features: Mapping[str, float],
        diagnostics: Mapping[str, Any],
        feature_order: Sequence[str],
        dtype: str,
        nan_policy: str,
    ) -> Path:
        order = tuple(feature_order)
        if tuple(features) != order:
            raise ValueError("features must exactly match feature_order")
        if not all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            for value in features.values()
        ):
            raise ValueError("features must contain finite numeric values")
        if not isinstance(diagnostics, Mapping):
            raise TypeError("diagnostics must be a mapping")
        _strict_json(diagnostics, "diagnostics")
        payload = {
            "schema_version": 1,
            "key": key,
            "feature_order": list(order),
            "dtype": dtype,
            "nan_policy": nan_policy,
            "features": [[name, float(features[name])] for name in order],
            "diagnostics": dict(diagnostics),
        }
        data = (
            json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False)
            + "\n"
        ).encode("utf-8")
        destination = self.path_for(key)
        destination.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
        finally:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
        return destination

    def read(
        self,
        key: str,
        *,
        feature_order: Sequence[str],
        dtype: str,
        nan_policy: str,
    ) -> tuple[dict[str, float], dict[str, Any]] | None:
        try:
            payload = json.loads(self.path_for(key).read_text(encoding="utf-8"))
            order = list(feature_order)
            if (
                not isinstance(payload, dict)
                or set(payload) != {
                    "schema_version", "key", "feature_order", "dtype", "nan_policy",
                    "features", "diagnostics",
                }
                or payload["schema_version"] != 1
                or payload["key"] != key
                or payload["feature_order"] != order
                or payload["dtype"] != dtype
                or payload["nan_policy"] != nan_policy
                or not isinstance(payload["diagnostics"], dict)
                or not isinstance(payload["features"], list)
                or len(payload["features"]) != len(order)
            ):
                return None
            features: dict[str, float] = {}
            for expected, item in zip(order, payload["features"], strict=True):
                if (
                    not isinstance(item, list)
                    or len(item) != 2
                    or item[0] != expected
                    or not isinstance(item[1], (int, float))
                    or isinstance(item[1], bool)
                    or not math.isfinite(float(item[1]))
                ):
                    return None
                features[expected] = float(item[1])
            _strict_json(payload["diagnostics"], "diagnostics")
            return features, payload["diagnostics"]
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError):
            return None
