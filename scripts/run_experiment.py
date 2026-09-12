"""Run one validated A2 feature spec without printing dataset identifiers."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from altur.dataset import Dataset
from altur.provenance import effective_code_digest
from altur.runner import RunnerError, run_experiment, validate_experiment_spec


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RunnerError(f"spec contains duplicate key: {key!r}")
        result[key] = value
    return result


def load_spec(path: str | Path) -> dict[str, Any]:
    """Load one JSON or YAML mapping while rejecting ambiguous duplicate keys."""
    source = Path(path)
    try:
        text = source.read_text(encoding="utf-8")
    except OSError as exc:
        raise RunnerError(f"cannot read spec: {source}") from exc
    if source.suffix == ".json":
        try:
            value = json.loads(
                text,
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=lambda value: (_ for _ in ()).throw(
                    RunnerError(f"JSON constant is not allowed: {value}")
                ),
            )
        except json.JSONDecodeError as exc:
            raise RunnerError("spec is not valid JSON") from exc
    elif source.suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise RunnerError("YAML specs require PyYAML; install '.[train]'") from exc

        class StrictLoader(yaml.SafeLoader):
            pass

        def construct_mapping(loader, node, deep=False):
            pairs = []
            for key_node, value_node in node.value:
                key = loader.construct_object(key_node, deep=deep)
                if not isinstance(key, str):
                    raise RunnerError("YAML mapping keys must be strings")
                pairs.append((key, loader.construct_object(value_node, deep=deep)))
            return _reject_duplicate_keys(pairs)

        StrictLoader.add_constructor(
            yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, construct_mapping
        )
        try:
            value = yaml.load(text, Loader=StrictLoader)
        except yaml.YAMLError as exc:
            raise RunnerError("spec is not valid YAML") from exc
    else:
        raise RunnerError("spec must use a .json, .yaml, or .yml extension")
    if not isinstance(value, Mapping):
        raise RunnerError("spec must be a mapping")
    return dict(value)


def _commit(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RunnerError("a Git commit is required for reproducible runs") from exc


def _environment() -> dict[str, str]:
    try:
        version = importlib.metadata.version("altur-detect")
    except importlib.metadata.PackageNotFoundError:
        version = "uninstalled"
    return {
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "package_version": version,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spec", type=Path)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--private-root", type=Path, default=Path(".private"))
    args = parser.parse_args(argv)

    spec = load_spec(args.spec)
    validate_experiment_spec(spec)  # Must run before Dataset can load any audio.
    root = Path(__file__).resolve().parents[1]
    environment = _environment()
    result = run_experiment(
        spec,
        dataset=Dataset() if args.data_root is None else Dataset(args.data_root),
        cache_root=args.private_root / "cache",
        artifact_root=args.private_root / "artifacts",
        ledger_root=args.private_root / "experiments" / "runs",
        code_digest=effective_code_digest(root),
        commit=_commit(root),
        environment=environment,
        code_version=environment["package_version"],
    )
    print(json.dumps(result.summary, ensure_ascii=True, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
