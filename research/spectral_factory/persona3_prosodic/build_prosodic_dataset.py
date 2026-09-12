"""
Corre la extraccion de features prosodicas (Tareas 3.1+3.2+pooling) sobre TODAS
las llamadas del dataset y escribe data/latents_prosodic.csv -- ver CONTRATOS.md
seccion 3.

Uso:
    python -m persona3_prosodic.build_prosodic_dataset [--limit N] [--workers K]
"""

import argparse
import time
from pathlib import Path

import pandas as pd
import soundfile as sf

from persona3_prosodic.prosodic_features import extract_prosodic_latent, _load_turns_channel0
from persona3_prosodic.splits import load_splits

REPO_ROOT = Path(__file__).parent.parent
AUDIO_DIR = REPO_ROOT / "audio"
TURNS_DIR = REPO_ROOT / "turns"
DATA_DIR = Path(__file__).parent / "data"
OUT_PATH = DATA_DIR / "latents_prosodic.csv"


def process_one(anon_id):
    """Ejecuta en un proceso worker: aislado, sin estado compartido, para que
    multiprocessing.Pool pueda paralelizar sin problemas de pickling raros."""
    wav_path = AUDIO_DIR / f"{anon_id}.wav"
    turns_path = TURNS_DIR / f"{anon_id}.json"
    try:
        audio, sr = sf.read(wav_path, always_2d=True)
        ch0 = audio[:, 0].astype(float)
        turns = _load_turns_channel0(turns_path)
        latent = extract_prosodic_latent(anon_id, ch0, sr, turns)
        if latent is None:
            return anon_id, None, "insuficiente_voz_estimable"
        return anon_id, latent.features, None
    except Exception as e:
        return anon_id, None, f"error: {e}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Procesar solo N llamadas (debug)")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    DATA_DIR.mkdir(exist_ok=True)
    splits_df = load_splits()
    if args.limit:
        splits_df = splits_df.head(args.limit)

    anon_ids = splits_df["anon_id"].tolist()
    print(f"Procesando {len(anon_ids)} llamadas con {args.workers} workers...")

    t0 = time.time()
    results = []
    failures = []
    import multiprocessing as mp
    with mp.Pool(args.workers) as pool:
        for i, (anon_id, feats, err) in enumerate(pool.imap_unordered(process_one, anon_ids)):
            if err is not None:
                failures.append((anon_id, err))
            else:
                results.append((anon_id, feats))
            if (i + 1) % 25 == 0:
                elapsed = time.time() - t0
                print(f"  {i+1}/{len(anon_ids)} procesadas ({elapsed:.0f}s)")

    print(f"Total: {time.time()-t0:.0f}s. OK={len(results)}, fallos={len(failures)}")
    for anon_id, err in failures:
        print(f"  [WARN] {anon_id}: {err}")

    rows = []
    meta = splits_df.set_index("anon_id")
    for anon_id, feats in results:
        row = {"anon_id": anon_id, "label": meta.loc[anon_id, "label"], "fold": meta.loc[anon_id, "fold"]}
        row.update(feats)
        rows.append(row)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT_PATH, index=False)
    print(f"Guardado en {OUT_PATH} ({len(out_df)} filas, {out_df.shape[1]-3} features)")

    if failures:
        fail_path = DATA_DIR / "latents_prosodic_failures.csv"
        pd.DataFrame(failures, columns=["anon_id", "error"]).to_csv(fail_path, index=False)
        print(f"Detalle de fallos en {fail_path}")


if __name__ == "__main__":
    main()
