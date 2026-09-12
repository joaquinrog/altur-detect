"""Deterministic identities for effective code and local immutable checkpoints."""

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path

_INCLUDED_DIRECTORIES = ("src", "scripts", "configs")
_INCLUDED_FILES = ("pyproject.toml",)
_EXCLUDED_PARTS = {
    ".git",
    ".checkpoints",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
}
_PRIVATE_PROTOCOL_FILES = {
    Path("configs/protocol/folds_v1.json"),
    Path("configs/protocol/groups_v1.csv"),
}


def _hash_bytes(hasher, value: bytes) -> None:
    hasher.update(len(value).to_bytes(8, "big"))
    hasher.update(value)


def _included_paths(root: Path) -> list[Path]:
    paths: list[Path] = []
    for name in _INCLUDED_FILES:
        path = root / name
        if path.is_file():
            paths.append(path)
    for name in _INCLUDED_DIRECTORIES:
        directory = root / name
        if not directory.is_dir():
            continue
        for path in directory.rglob("*"):
            if not path.is_file():
                continue
            relative = path.relative_to(root)
            if any(part in _EXCLUDED_PARTS for part in relative.parts):
                continue
            if relative in _PRIVATE_PROTOCOL_FILES:
                continue
            paths.append(path)
    return sorted(paths, key=lambda path: path.relative_to(root).as_posix())


def effective_code_digest(root: str | Path = ".") -> str:
    """Hash effective source/spec/config bytes, including uncommitted files."""
    directory = Path(root).resolve()
    if not directory.is_dir():
        raise ValueError("root must be an existing directory")
    hasher = hashlib.sha256()
    _hash_bytes(hasher, b"altur-effective-code-v1")
    for path in _included_paths(directory):
        if path.is_symlink():
            raise ValueError(f"effective code cannot contain symlinks: {path}")
        _hash_bytes(hasher, path.relative_to(directory).as_posix().encode("utf-8"))
        _hash_bytes(hasher, path.read_bytes())
    return f"sha256:v1:{hasher.hexdigest()}"


def _safe_component(value: str, name: str) -> None:
    if not isinstance(value, str) or not value or value in {".", ".."}:
        raise ValueError(f"{name} must be a non-empty path component")
    if "/" in value or "\\" in value or Path(value).name != value:
        raise ValueError(f"{name} must not contain path separators")


def _artifact_record(name: str, path: str | Path) -> dict[str, str | int]:
    _safe_component(name, "artifact name")
    artifact = Path(path)
    if artifact.is_symlink() or not artifact.is_file():
        raise ValueError(f"artifact must be a regular file: {name}")
    data = artifact.read_bytes()
    return {
        "name": name,
        "sha256": f"sha256:{hashlib.sha256(data).hexdigest()}",
        "bytes": len(data),
    }


def write_checkpoint(
    gate: str,
    checkpoint: str,
    artifacts: Mapping[str, str | Path],
    *,
    code_digest: str,
    root: str | Path = ".checkpoints",
) -> Path:
    """Create an immutable local checkpoint index record grouped by gate."""
    _safe_component(gate, "gate")
    _safe_component(checkpoint, "checkpoint")
    if not isinstance(artifacts, Mapping) or not artifacts:
        raise ValueError("artifacts must be a non-empty mapping")
    if not isinstance(code_digest, str) or not code_digest:
        raise ValueError("code_digest must be a non-empty string")

    records = [_artifact_record(name, artifacts[name]) for name in sorted(artifacts)]
    payload = {
        "schema_version": 1,
        "gate": gate,
        "checkpoint": checkpoint,
        "code_digest": code_digest,
        "artifacts": records,
    }
    data = (
        json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    destination = Path(root) / gate / f"{checkpoint}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, destination)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
    return destination
