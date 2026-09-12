"""FACT: run records are immutable files; the aggregate is a derived artifact."""

import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any


class LedgerError(ValueError):
    """Invalid ledger input."""


_REQUIRED = ("commit", "data_hash", "environment", "code_version")


def _validate_run_id(run_id: str) -> None:
    if not isinstance(run_id, str) or not run_id or run_id in {".", "..", "aggregate"}:
        raise LedgerError("run_id must be a single non-empty path component")
    if "/" in run_id or "\\" in run_id or Path(run_id).name != run_id:
        raise LedgerError("run_id must not contain path separators")


def _json_bytes(value: Any) -> bytes:
    try:
        return (json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    except (TypeError, ValueError) as exc:
        raise LedgerError("payload must be strict JSON") from exc


def _atomic_replace(data: bytes, destination: Path, *, exclusive: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            os.fchmod(stream.fileno(), 0o644)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        if exclusive:
            os.link(temporary, destination)
        else:
            os.replace(temporary, destination)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def write_run(run_id: str, payload: Mapping[str, Any], *, root: str | Path = "experiments/runs") -> Path:
    """Create one immutable run record and return its path."""
    _validate_run_id(run_id)
    if not isinstance(payload, Mapping):
        raise LedgerError("payload must be a mapping")
    missing = [field for field in _REQUIRED if field not in payload]
    if missing:
        raise LedgerError(f"missing required fields: {', '.join(missing)}")
    for field in ("commit", "data_hash", "code_version"):
        if not isinstance(payload[field], str) or not payload[field]:
            raise LedgerError(f"{field} must be a non-empty string")
    if not isinstance(payload["environment"], Mapping):
        raise LedgerError("environment must be a mapping")
    data = _json_bytes(dict(payload))
    destination = Path(root) / f"{run_id}.json"
    _atomic_replace(data, destination, exclusive=True)
    return destination


def write_aggregate(*, root: str | Path = "experiments/runs") -> Path:
    """Atomically regenerate the deterministic aggregate from run files only."""
    directory = Path(root)
    records = []
    for path in sorted(directory.glob("*.json"), key=lambda item: item.name):
        if path.name == "aggregate.json":
            continue
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise LedgerError(f"invalid run file: {path.name}") from exc
        records.append({"run_id": path.stem, "record": record})
    aggregate = _json_bytes({"schema_version": 1, "runs": records})
    destination = directory / "aggregate.json"
    _atomic_replace(aggregate, destination, exclusive=False)
    return destination
