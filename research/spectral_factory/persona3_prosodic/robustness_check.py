"""
Tarea 4: verificar si Shimmer sigue siendo discriminativo DESPUES de que el audio
pasa por el modulo de corrupcion de canal (G.711 + paso-banda). Ver prompt maestro
Persona 3, punto 4, y CONTRATOS.md seccion 2.

Metodologia: sobre una muestra estratificada, extraemos el vector latente
prosodico dos veces por clip -- version limpia y version corrompida
(deterministicamente, forzando p=1.0 con seed fijo por clip para que el resultado
sea reproducible) -- y comparamos, para cada feature, el AUC de separacion
human-vs-synthetic (mismo metodo Mann-Whitney U del analisis forense original)
antes y despues de la corrupcion. Si el AUC cae mucho, shimmer dependia en parte
de artefactos de canal y hay que discutirlo con el equipo ANTES de la Fase 2
(instruccion explicita del prompt maestro).
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from scipy import stats as spstats

from persona3_prosodic.corruption import TelephonyAugmenter
from persona3_prosodic.prosodic_features import extract_prosodic_latent, _load_turns_channel0
from persona3_prosodic.splits import load_splits

REPO_ROOT = Path(__file__).parent.parent
AUDIO_DIR = REPO_ROOT / "audio"
TURNS_DIR = REPO_ROOT / "turns"
DATA_DIR = Path(__file__).parent / "data"
OUT_PATH = DATA_DIR / "robustness_check.csv"

N_PER_CLASS = 25  # muestra mas chica que el dataset completo -- este experimento
                   # se corre dos veces por clip (limpio + corrompido), asi que se
                   # mantiene acotado a proposito


def mannwhitney_auc(df, col):
    a = df[df["label"] == "human"][col].dropna().values
    b = df[df["label"] == "synthetic"][col].dropna().values
    if len(a) < 2 or len(b) < 2:
        return np.nan, np.nan
    stat, pval = spstats.mannwhitneyu(a, b, alternative="two-sided")
    n1, n2 = len(a), len(b)
    auc = (n1 * n2 - stat) / (n1 * n2)
    return pval, max(auc, 1 - auc)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_per_class", type=int, default=N_PER_CLASS)
    args = parser.parse_args()

    DATA_DIR.mkdir(exist_ok=True)
    splits_df = load_splits()
    rng = np.random.default_rng(123)
    sample = []
    for label in ["human", "synthetic"]:
        subset = splits_df[splits_df["label"] == label]
        n = min(args.n_per_class, len(subset))
        idx = rng.choice(subset.index.values, size=n, replace=False)
        sample.append(subset.loc[idx])
    sample_df = pd.concat(sample).reset_index(drop=True)
    print(f"Muestra: {sample_df.groupby('label').size().to_dict()}")

    rows = []
    t0 = time.time()
    for i, row in sample_df.iterrows():
        anon_id, label = row["anon_id"], row["label"]
        audio, sr = sf.read(AUDIO_DIR / f"{anon_id}.wav", always_2d=True)
        ch0 = audio[:, 0].astype(float)
        turns = _load_turns_channel0(TURNS_DIR / f"{anon_id}.json")

        clean_latent = extract_prosodic_latent(anon_id, ch0, sr, turns)

        # corrupcion FORZADA (p=1.0) y determinista (seed por clip) para que el
        # experimento sea reproducible -- en entrenamiento real la corrupcion es
        # estocastica (P=0.5), pero aqui queremos medir el peor caso, no el promedio
        seed = abs(hash(anon_id)) % (2**32)
        # families=("channel",) explicito: este experimento prueba especificamente
        # robustez a codec/canal, no a pitch/time-stretch (que ahora tambien
        # existen en TelephonyAugmenter desde la Fase 2, ver corruption.py) --
        # sin esto, el default (las 3 familias) mezclaria ambos ejes y dejaria
        # de medir lo que este script dice medir.
        augmenter = TelephonyAugmenter(p=1.0, families=("channel",), seed=seed)
        corrupted_audio, meta = augmenter(ch0, sr)
        corrupted_latent = extract_prosodic_latent(anon_id, corrupted_audio, sr, turns)

        if clean_latent is None or corrupted_latent is None:
            print(f"  [WARN] {anon_id}: insuficiente voz estimable en limpio o corrompido, se omite")
            continue

        clean_row = {"anon_id": anon_id, "label": label, "corrupted": False}
        clean_row.update(clean_latent.features)
        corrupted_row = {"anon_id": anon_id, "label": label, "corrupted": True, "codec": meta.get("codec")}
        corrupted_row.update(corrupted_latent.features)
        rows.append(clean_row)
        rows.append(corrupted_row)

        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(sample_df)} clips ({time.time()-t0:.0f}s)")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_PATH, index=False)
    print(f"Guardado en {OUT_PATH}")

    feature_cols = [c for c in df.columns if c.startswith(("shimmer_", "dshimmer_", "ddshimmer_"))]
    summary_rows = []
    for col in feature_cols:
        clean_df = df[df["corrupted"] == False]
        corrupted_df = df[df["corrupted"] == True]
        p_clean, auc_clean = mannwhitney_auc(clean_df, col)
        p_corr, auc_corr = mannwhitney_auc(corrupted_df, col)
        summary_rows.append({
            "feature": col,
            "auc_limpio": auc_clean,
            "auc_corrompido": auc_corr,
            "delta_auc": auc_corr - auc_clean if pd.notna(auc_clean) and pd.notna(auc_corr) else np.nan,
            "p_value_limpio": p_clean,
            "p_value_corrompido": p_corr,
        })

    summary_df = pd.DataFrame(summary_rows).sort_values("auc_limpio", ascending=False)
    summary_path = DATA_DIR / "robustness_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print("\n" + summary_df.to_string(index=False))
    print(f"\nResumen en {summary_path}")

    mean_drop = (summary_df["auc_limpio"] - summary_df["auc_corrompido"]).mean()
    if mean_drop > 0.10:
        print(
            f"\n[ALERTA] El AUC promedio de las features de shimmer cae {mean_drop:.3f} "
            "tras la corrupcion de canal. Esto sugiere que shimmer SI dependia en parte "
            "de artefactos de canal, no solo de calidad vocal real -- discutir con el "
            "equipo antes de construir sobre esta rama en la Fase 2."
        )
    else:
        print(
            f"\n[OK] El AUC de las features de shimmer se mantiene estable tras la "
            f"corrupcion (caida promedio {mean_drop:.3f}). La rama prosodica parece ser "
            "una senal razonablemente ortogonal al canal de grabacion."
        )


if __name__ == "__main__":
    main()
