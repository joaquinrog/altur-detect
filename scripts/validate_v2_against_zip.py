#!/usr/bin/env python3
"""Compara nuestras perturbaciones v2 contra el set de robustez original (ZIP de Ricardo).

Cierra los UNK de `configs/perturbations/v2.yaml` (D-A5.4). No evalúa ningún detector: compara
señal contra señal. Para cada condición toma la llamada `clean` del ZIP, le aplica nuestra cadena
v2 y la compara con la versión del ZIP en longitud, RMS, fracción de energía sobre 3400 Hz y
distancia log-espectral **en 100–3400 Hz**. Fuera de esa banda, el piso de cuantización int16
domina el espectro logarítmico e inflaría la distancia en bins sin contenido; lo que pasa arriba
de 3400 Hz lo mide `over3400_*`.

Reglas que respeta:
- **Solo `train`** (regla 4). Las llamadas de `val` del ZIP se saltan, sin excepción.
- **Streaming**: lee cada WAV del ZIP a memoria, nunca descomprime a disco (12 GB).
- **Solo agregados**: no imprime ni escribe un solo identificador de llamada.

Uso:
    python scripts/validate_v2_against_zip.py --zip "robustness_phase1 (2).zip" --limit 30
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
import zipfile
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from altur import transforms as _v1  # noqa: F401  (registra los transforms v1)
from altur import transforms_v2 as _v2  # noqa: F401  (registra los transforms v2)
from altur.io import decode
from altur.registry import transforms
from altur.types import AudioExample

V2_YAML = ROOT / "configs" / "perturbations" / "v2.yaml"
_FRAME, _HOP, _SR = 256, 128, 8000


def load_conditions(path: Path = V2_YAML) -> dict[str, list[str]]:
    return dict(yaml.safe_load(path.read_text())["conditions"])


def train_ids(manifest: Path) -> list[str]:
    with manifest.open(newline="") as fh:
        return sorted(row["anon_id"] for row in csv.DictReader(fh) if row["split"] == "train")


def find_prefix(names: list[str]) -> str:
    """Directorio que contiene `clean/`. Se detecta en vez de fijarse: el ZIP cambió de nombre una vez."""
    for name in names:
        head, sep, _ = name.partition("/clean/")
        if sep and name.endswith(".wav"):
            return head
    raise ValueError("el ZIP no tiene una carpeta clean/ con WAVs")


def _psd(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    if len(x) < _FRAME:
        x = np.pad(x, (0, _FRAME - len(x)))
    n = 1 + (len(x) - _FRAME) // _HOP
    idx = np.arange(_FRAME)[None, :] + _HOP * np.arange(n)[:, None]
    spec = np.abs(np.fft.rfft(x[idx] * np.hanning(_FRAME), axis=1)) ** 2
    return spec.mean(axis=0) + 1e-12


def compare(ours: np.ndarray, theirs: np.ndarray) -> dict[str, float]:
    p_ours, p_theirs = _psd(ours), _psd(theirs)
    freqs = np.fft.rfftfreq(_FRAME, 1 / _SR)
    over = freqs > 3400
    band = (freqs >= 100) & (freqs <= 3400)

    def rms(x: np.ndarray) -> float:
        return float(np.sqrt(np.mean(x.astype(np.float64) ** 2)) + 1e-12)

    return {
        "len_ratio": len(ours) / max(len(theirs), 1),
        "rms_ratio": rms(ours) / rms(theirs),
        "over3400_ours": float(p_ours[over].sum() / p_ours.sum()),
        "over3400_theirs": float(p_theirs[over].sum() / p_theirs.sum()),
        "lsd_db": float(np.sqrt(np.mean((10 * np.log10(p_ours[band]) - 10 * np.log10(p_theirs[band])) ** 2))),
    }


def _rng(raw: bytes, seed: int) -> np.random.Generator:
    material = hashlib.sha256(raw).hexdigest().encode() + str(seed).encode()
    return np.random.default_rng(int.from_bytes(hashlib.sha256(material).digest(), "big"))


def run(zip_path: Path, manifest: Path, limit: int, seed: int) -> list[dict[str, object]]:
    conditions = load_conditions()
    wanted = set(train_ids(manifest))
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        prefix = find_prefix(names)
        present = {n for n in names if n.endswith(".wav")}
        in_zip = sorted(
            Path(n).stem for n in present if n.startswith(f"{prefix}/clean/") and Path(n).stem in wanted
        )
        order = np.random.default_rng(seed).permutation(len(in_zip))
        sample = [in_zip[i] for i in order[:limit]]

        rows: list[dict[str, object]] = []
        for condition, refs in conditions.items():
            metrics: list[dict[str, float]] = []
            for call in sample:
                entry = f"{prefix}/{condition}/{call}.wav"
                if entry not in present:
                    continue
                clean_raw = zf.read(f"{prefix}/clean/{call}.wav")
                ex: AudioExample = decode(clean_raw).example
                rng = _rng(clean_raw, seed)
                for ref in refs:
                    ex = transforms.resolve(ref)(ex, rng)
                theirs = decode(zf.read(entry)).example
                metrics.append(compare(ex.ch0, theirs.ch0))
            row: dict[str, object] = {"condition": condition, "n": len(metrics)}
            for key in ("len_ratio", "rms_ratio", "over3400_ours", "over3400_theirs", "lsd_db"):
                row[key] = round(float(np.median([m[key] for m in metrics])), 4) if metrics else None
            rows.append(row)
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--zip", required=True, type=Path)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data" / "manifest.csv")
    parser.add_argument("--limit", type=int, default=30, help="llamadas de train por condición")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=Path, help="CSV de agregados (opcional)")
    args = parser.parse_args(argv)

    rows = run(args.zip, args.manifest, args.limit, args.seed)
    cols = ["condition", "n", "len_ratio", "rms_ratio", "over3400_ours", "over3400_theirs", "lsd_db"]
    print("  ".join(f"{c:>16s}" for c in cols))
    for row in rows:
        print("  ".join(f"{row[c]!s:>16s}" for c in cols))
    if args.out:
        with args.out.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=cols)
            writer.writeheader()
            writer.writerows(rows)
    print("\nMediana por condición, sin identificadores. len_ratio≈1 y lsd_db bajo ⇒ la reimplementación "
          "coincide con el set original; si no, el UNK correspondiente de v2.yaml sigue abierto.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
