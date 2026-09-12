"""
Corre la extraccion de features espectrales (LFCC 60-dim, referencia/placeholder
-- ver README.md de esta carpeta) sobre TODAS las llamadas del dataset y escribe
persona3_prosodic/data/latents_spectral.csv -- el contrato que fusion_ablation.py
ya espera (ver persona3_prosodic/CONTRATOS.md seccion 3).

Uso:
    python -m persona2_reference_spectral.build_spectral_dataset [--limit N] [--workers K]
"""

import argparse
import time
from pathlib import Path

import pandas as pd
import soundfile as sf

from persona2_reference_spectral.lfcc_features import extract_spectral_latent
from persona3_prosodic.prosodic_features import _load_turns_channel0
from persona3_prosodic.splits import load_splits

REPO_ROOT = Path(__file__).parent.parent
AUDIO_DIR = REPO_ROOT / "audio"
TURNS_DIR = REPO_ROOT / "turns"
# escribe directo al data/ compartido de persona3_prosodic -- ahi es donde
# fusion_ablation.py busca latents_spectral.csv (ver CONTRATOS.md seccion 3)
OUT_DIR = REPO_ROOT / "persona3_prosodic" / "data"
OUT_PATH = OUT_DIR / "latents_spectral.csv"


def process_one(anon_id):
    wav_path = AUDIO_DIR / f"{anon_id}.wav"
    turns_path = TURNS_DIR / f"{anon_id}.json"
    try:
        audio, sr = sf.read(wav_path, always_2d=True)
        ch0 = audio[:, 0].astype(float)
        turns = _load_turns_channel0(turns_path)
        latent = extract_spectral_latent(anon_id, ch0, sr, turns)
        if latent is None:
            return anon_id, None, "insuficiente_voz_estimable"
        return anon_id, latent.features, None
    except Exception as e:
        return anon_id, None, f"error: {e}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    OUT_DIR.mkdir(exist_ok=True)
    splits_df = load_splits()
    if args.limit:
        splits_df = splits_df.head(args.limit)

    anon_ids = splits_df["anon_id"].tolist()
    print(f"Procesando {len(anon_ids)} llamadas con {args.workers} workers (LFCC placeholder)...")

    t0 = time.time()
    results, failures = [], []
    import multiprocessing as mp
    with mp.Pool(args.workers) as pool:
        for i, (anon_id, feats, err) in enumerate(pool.imap_unordered(process_one, anon_ids)):
            if err is not None:
                failures.append((anon_id, err))
            else:
                results.append((anon_id, feats))
            if (i + 1) % 25 == 0:
                print(f"  {i+1}/{len(anon_ids)} ({time.time()-t0:.0f}s)")

    print(f"Total: {time.time()-t0:.0f}s. OK={len(results)}, fallos={len(failures)}")
    for anon_id, err in failures:
        print(f"  [WARN] {anon_id}: {err}")

    meta = splits_df.set_index("anon_id")
    rows = []
    for anon_id, feats in results:
        row = {"anon_id": anon_id, "label": meta.loc[anon_id, "label"], "fold": meta.loc[anon_id, "fold"]}
        row.update(feats)
        rows.append(row)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT_PATH, index=False)
    print(f"Guardado en {OUT_PATH} ({len(out_df)} filas, {out_df.shape[1]-3} features)")


if __name__ == "__main__":
    main()
