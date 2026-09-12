"""
Fase 2 -- valida la recomendacion concreta del equipo: entrenar la rama
prosodica CON pitch-shift (+-1 semitono) y time-stretch (0.9x-1.1x) en el
train fold, para ver si el AUC bajo esas condiciones sube de ~0.62 a >0.80
como se hipotetizo.

La prueba HONESTA no es reproducir nuestro propio pitch-shift/time-stretch y
evaluarnos a nosotros mismos (séria circular) -- es entrenar con NUESTRA
propia corrupcion (TelephonyAugmenter) y validar contra el audio YA
corrompido por Ricardo en su ZIP (pitch_up1st, timestretch_0.9x), que usa su
propia implementacion independiente. Si el AUC sube ahi, es evidencia real de
que la augmentacion generaliza, no que aprendimos a deshacer nuestra propia
transformacion.

Reglas de Oro respetadas (igual que persona2_reference_spectral/train_isolated_spectral.py):
  - Aislamiento estricto de folds: el fold de validacion (tanto el clean
    interno como TODO el ZIP de Ricardo) se evalua SIEMPRE con el modelo de
    los otros 4 folds -- nunca con uno que haya visto esa llamada, ni en
    limpio.
  - Conserva de muestras limpias: el train set de cada fold es clean UNION
    augmented (no se reemplaza clean por corrompido).
"""

import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

from persona3_prosodic.corruption import TelephonyAugmenter, rescale_turns
from persona3_prosodic.prosodic_features import extract_prosodic_latent, _load_turns_channel0
from persona3_prosodic.train_isolated_branch import compute_eer
from phase1_stress_test.eval_stress_test import load_call_from_zip

REPO_ROOT = Path(__file__).parent.parent
AUDIO_DIR = REPO_ROOT / "audio"
TURNS_DIR = REPO_ROOT / "turns"
DATA_DIR = REPO_ROOT / "persona3_prosodic" / "data"
OUT_DIR = Path(__file__).parent / "data"

# condiciones del ZIP de Ricardo que usamos como examen final (independiente
# de nuestra propia implementacion de pitch/time-stretch)
ZIP_CONDITIONS = ["clean", "pitch_up1st", "timestretch_0.9x"]


def build_augmented_latents(splits_df, seed=777):
    """Genera, para cada llamada, UNA version aumentada (pitch_shift o
    time_stretch, elegido random por llamada) a partir de su audio limpio, y
    extrae shimmer -- con los turns reescalados si hubo time-stretch."""
    rows = []
    t0 = time.time()
    for i, row in splits_df.iterrows():
        anon_id, label, fold = row["anon_id"], row["label"], row["fold"]
        audio, sr = sf.read(AUDIO_DIR / f"{anon_id}.wav", always_2d=True)
        ch0 = audio[:, 0].astype(float)
        turns = _load_turns_channel0(TURNS_DIR / f"{anon_id}.json")

        call_seed = (seed + abs(hash(anon_id))) % (2**32)
        augmenter = TelephonyAugmenter(p=1.0, families=("pitch_shift", "time_stretch"), seed=call_seed)
        corrupted_audio, meta = augmenter(ch0, sr)
        adj_turns = rescale_turns(turns, meta["time_scale"])

        latent = extract_prosodic_latent(anon_id, corrupted_audio, sr, adj_turns)
        if latent is None:
            continue
        r = {"anon_id": anon_id, "label": label, "fold": fold, "aug_family": meta["family"]}
        r.update(latent.features)
        rows.append(r)
        if (i + 1) % 50 == 0:
            print(f"  augmentado: {i+1}/{len(splits_df)} ({time.time()-t0:.0f}s)")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "latents_prosodic_augmented.csv", index=False)
    print(f"Latentes aumentados guardados ({len(df)} filas)")
    return df


def fit_fold_models(clean_df, augmented_df, feats, folds_unique):
    """Por fold: entrena SOLO con clean+augmented de los OTROS folds."""
    models = {}
    for fold in folds_unique:
        train = pd.concat([
            clean_df[clean_df["fold"] != fold],
            augmented_df[augmented_df["fold"] != fold][["anon_id", "label", "fold"] + feats],
        ])
        X = train[feats].values
        y = (train["label"] == "synthetic").astype(int).values
        scaler = StandardScaler().fit(X)
        model = LogisticRegression(max_iter=1000, class_weight="balanced")
        model.fit(scaler.transform(X), y)
        models[fold] = (scaler, model)
    return models


def score_call(features_dict, scaler, model, feats):
    vec = np.array([[features_dict[f] for f in feats]])
    return float(model.predict_proba(scaler.transform(vec))[0, 1])


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", required=True, help="Ruta al zip robustness_phase1 (2).zip")
    args = parser.parse_args()

    OUT_DIR.mkdir(exist_ok=True)
    splits_df = pd.read_csv(DATA_DIR / "splits.csv")
    clean_df = pd.read_csv(DATA_DIR / "latents_prosodic.csv").dropna()
    feats = [c for c in clean_df.columns if c not in ("anon_id", "label", "fold")]

    print("Generando features aumentadas (pitch-shift / time-stretch) para las 353 llamadas...")
    augmented_df = build_augmented_latents(splits_df)
    augmented_df = augmented_df.dropna(subset=feats)

    folds_unique = sorted(splits_df["fold"].unique())
    print(f"\nEntrenando modelos leave-one-fold-out CON augmentacion (clean UNION augmented, {len(folds_unique)} folds)...")
    models_by_fold = fit_fold_models(clean_df, augmented_df, feats, folds_unique)

    fold_by_anon = splits_df.set_index("anon_id")["fold"].to_dict()
    label_by_anon = splits_df.set_index("anon_id")["label"].to_dict()

    print(f"\nValidando contra el ZIP REAL de Ricardo ({args.zip})...")
    zf = zipfile.ZipFile(args.zip)
    rows = []
    t0 = time.time()
    for condition in ZIP_CONDITIONS:
        print(f"  condicion: {condition}")
        for anon_id in splits_df["anon_id"]:
            try:
                ch0, sr, turns_ch0 = load_call_from_zip(zf, condition, anon_id)
                latent = extract_prosodic_latent(anon_id, ch0, sr, turns_ch0)
                if latent is None:
                    continue
                fold = fold_by_anon[anon_id]
                scaler, model = models_by_fold[fold]
                score = score_call(latent.features, scaler, model, feats)
                rows.append({"condicion": condition, "anon_id": anon_id,
                             "label": label_by_anon[anon_id], "fold": fold, "score": score})
            except Exception as e:
                print(f"    [WARN] {condition}/{anon_id}: {e}")
    zf.close()
    print(f"Listo en {time.time()-t0:.0f}s")

    scores_df = pd.DataFrame(rows)
    scores_df.to_csv(OUT_DIR / "scores_prosodica_augmentada_vs_zip_real.csv", index=False)

    print("\n=== Comparacion: rama prosodica SIN vs. CON augmentacion (pitch/time-stretch), sobre ZIP REAL de Ricardo ===")
    baseline = {"clean": 0.8283, "pitch_up1st": 0.6250, "timestretch_0.9x": 0.6208}  # de la Fase 1, sin augmentacion
    results = []
    for condition in ZIP_CONDITIONS:
        sub = scores_df[scores_df["condicion"] == condition]
        y = (sub["label"] == "synthetic").astype(int).values
        auc = roc_auc_score(y, sub["score"].values)
        eer, _ = compute_eer(y, sub["score"].values)
        row = {
            "condicion": condition,
            "auc_sin_augmentacion": baseline[condition],
            "auc_con_augmentacion": round(auc, 4),
            "delta": round(auc - baseline[condition], 4),
            "eer_con_augmentacion_pct": round(eer * 100, 2),
        }
        results.append(row)
        print(f"  {condition:<18} sin_aug={baseline[condition]:.4f}  con_aug={auc:.4f}  "
              f"delta={auc-baseline[condition]:+.4f}  EER={eer*100:.2f}%")

    results_df = pd.DataFrame(results)
    results_path = OUT_DIR / "comparacion_augmentacion_prosodica.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\nGuardado en {results_path}")

    hipotesis_confirmada = all(
        r["condicion"] == "clean" or r["auc_con_augmentacion"] > 0.80
        for r in results
    )
    print(
        f"\n[VEREDICTO] Hipotesis del equipo (AUC de 0.62 a >0.80 bajo pitch/time-stretch "
        f"tras augmentacion): {'CONFIRMADA' if hipotesis_confirmada else 'NO CONFIRMADA -- ver numeros arriba'}."
    )


if __name__ == "__main__":
    main()
