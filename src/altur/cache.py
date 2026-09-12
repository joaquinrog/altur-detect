"""FACT: cache identities must include every input that can alter features."""

import base64
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
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
    }
    encoded = json.dumps(_canonical(document), ensure_ascii=True, separators=(",", ":"), allow_nan=False).encode()
    return f"sha256:v1:{hashlib.sha256(encoded).hexdigest()}"
