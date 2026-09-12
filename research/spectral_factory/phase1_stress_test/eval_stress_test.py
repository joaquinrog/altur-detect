"""
Fase 1: Stress Test de Robustez y Ortogonalidad sobre el ZIP de Persona 2.

Evalua las 3 configuraciones (Espectral sola, Prosodica sola, Fusion tardia)
por separado sobre cada una de las 6 condiciones del ZIP de Ricardo (clean,
noise_snr10, pitch_up1st, timestretch_0.9x, lowpass_3400hz, opus_16kbps), SIN
mezclar condiciones en una sola metrica.

Decision metodologica critica (para no repetir el error de fuga de datos que
ya nos costo caro en la rama espectral placeholder, ver HALLAZGOS_LOG.md):
las 353 llamadas del ZIP de Ricardo son las MISMAS 353 llamadas que usamos
para entrenar nuestros modelos (el "validation set" de su ZIP no es un
held-out nuevo, es el dataset completo). Si evaluaramos con el modelo final
(entrenado con las 353), estariamos evaluando cada llamada con un modelo que
YA la vio en su version limpia durante entrenamiento -- optimista, no
confiable. Por eso aqui se re-entrena un modelo POR FOLD (GroupKFold ya
existente en data/splits.csv) y cada llamada se evalua SIEMPRE con el modelo
de los otros 4 folds, nunca con uno que la haya visto (ni siquiera en su
version limpia).

No se extrae el ZIP a disco (12GB descomprimido, solo 17GB libres en el
momento de escribir esto): se lee cada .wav/.json directamente desde el ZIP
en memoria via el modulo zipfile + un buffer, y se descarta tras extraer
features.
"""

import argparse
import io
import json
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

from persona3_prosodic.prosodic_features import extract_prosodic_latent
from persona3_prosodic.train_isolated_branch import compute_eer
from persona2_reference_spectral.lfcc_features import extract_spectral_latent

REPO_ROOT = Path(__file__).parent.parent
DATA_DIR = REPO_ROOT / "persona3_prosodic" / "data"
OUT_DIR = Path(__file__).parent / "data"

ZIP_ENTRY_PREFIX = "robustness_phase1/robustness/data/validation_corrupted"
CONDITIONS = [
    "clean",
    "noise_snr10",
    "pitch_up1st",
    "timestretch_0.9x",
    "lowpass_3400hz",
    "opus_16kbps",
]


def _turns_channel0_from_dict(data):
    turns = [t for t in data["turns"] if t["channel"] == 0]
    turns.sort(key=lambda t: t["start"])
    return turns


def load_call_from_zip(zf, condition, anon_id):
    """Lee wav+json de UNA llamada bajo UNA condicion, directamente del zip,
    sin tocar disco. Devuelve (audio_ch0, sr, turns_ch0)."""
    wav_entry = f"{ZIP_ENTRY_PREFIX}/{condition}/{anon_id}.wav"
    json_entry = f"{ZIP_ENTRY_PREFIX}/{condition}/{anon_id}.json"

    wav_bytes = zf.read(wav_entry)
    audio, sr = sf.read(io.BytesIO(wav_bytes), always_2d=True)
    ch0 = audio[:, 0].astype(float)

    turns_data = json.loads(zf.read(json_entry))
    turns_ch0 = _turns_channel0_from_dict(turns_data)
    return ch0, sr, turns_ch0


def fit_fold_models(prosodic_df, spectral_df, merged_df, folds_unique):
    """Entrena, para cada fold, un modelo de cada una de las 3 configuraciones
    SOBRE LOS OTROS 4 folds -- devuelve dict fold -> {"prosodic": (scaler,model,feats), ...}."""
    prosodic_feats = [c for c in prosodic_df.columns if c not in ("anon_id", "label", "fold")]
    spectral_feats = [c for c in spectral_df.columns if c not in ("anon_id", "label", "fold")]
    fusion_feats = [c for c in merged_df.columns if c not in ("anon_id", "label", "fold")]

    def fit(df, feats, fold):
        train = df[df["fold"] != fold]
        X = train[feats].values
        y = (train["label"] == "synthetic").astype(int).values
        scaler = StandardScaler().fit(X)
        model = LogisticRegression(max_iter=1000, class_weight="balanced")
        model.fit(scaler.transform(X), y)
        return scaler, model

    models_by_fold = {}
    for fold in folds_unique:
        models_by_fold[fold] = {
            "prosodic": (*fit(prosodic_df, prosodic_feats, fold), prosodic_feats),
            "spectral": (*fit(spectral_df, spectral_feats, fold), spectral_feats),
            "fusion": (*fit(merged_df, fusion_feats, fold), fusion_feats),
        }
    return models_by_fold


def score_with_model(features_dict, scaler, model, feat_names):
    vec = np.array([[features_dict[f] for f in feat_names]])
    return float(model.predict_proba(scaler.transform(vec))[0, 1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", required=True, help="Ruta al zip robustness_phase1 (2).zip")
    parser.add_argument("--limit", type=int, default=None, help="Procesar solo N llamadas (debug)")
    args = parser.parse_args()

    OUT_DIR.mkdir(exist_ok=True)

    splits_df = pd.read_csv(DATA_DIR / "splits.csv")
    prosodic_df = pd.read_csv(DATA_DIR / "latents_prosodic.csv").dropna()
    spectral_df = pd.read_csv(DATA_DIR / "latents_spectral.csv").dropna()
    merged_df = prosodic_df.merge(spectral_df, on=["anon_id", "label", "fold"], how="inner",
                                   suffixes=("_pros", "_spec"))

    folds_unique = sorted(splits_df["fold"].unique())
    print(f"Entrenando modelos por fold ({len(folds_unique)} folds, leave-one-fold-out)...")
    models_by_fold = fit_fold_models(prosodic_df, spectral_df, merged_df, folds_unique)

    fold_by_anon = splits_df.set_index("anon_id")["fold"].to_dict()
    label_by_anon = splits_df.set_index("anon_id")["label"].to_dict()

    anon_ids = splits_df["anon_id"].tolist()
    if args.limit:
        anon_ids = anon_ids[:args.limit]

    print(f"Abriendo {args.zip} ({len(anon_ids)} llamadas x {len(CONDITIONS)} condiciones)...")
    zf = zipfile.ZipFile(args.zip)

    rows = []
    failures = []
    t0 = time.time()
    for cond_idx, condition in enumerate(CONDITIONS):
        print(f"\n[{cond_idx+1}/{len(CONDITIONS)}] Condicion: {condition}")
        for i, anon_id in enumerate(anon_ids):
            try:
                ch0, sr, turns_ch0 = load_call_from_zip(zf, condition, anon_id)
                pros_latent = extract_prosodic_latent(anon_id, ch0, sr, turns_ch0)
                spec_latent = extract_spectral_latent(anon_id, ch0, sr, turns_ch0)
                if pros_latent is None or spec_latent is None:
                    failures.append((condition, anon_id, "insuficiente_voz_estimable"))
                    continue

                fold = fold_by_anon[anon_id]
                fmodels = models_by_fold[fold]

                pros_score = score_with_model(pros_latent.features, *fmodels["prosodic"])
                spec_score = score_with_model(spec_latent.features, *fmodels["spectral"])
                fusion_feats_dict = {**pros_latent.features, **spec_latent.features}
                fusion_score = score_with_model(fusion_feats_dict, *fmodels["fusion"])

                rows.append({
                    "condicion": condition,
                    "anon_id": anon_id,
                    "label": label_by_anon[anon_id],
                    "fold": fold,
                    "score_espectral": spec_score,
                    "score_prosodica": pros_score,
                    "score_fusion": fusion_score,
                })
            except Exception as e:
                failures.append((condition, anon_id, f"error: {e}"))

            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{len(anon_ids)} ({time.time()-t0:.0f}s)")

    zf.close()
    print(f"\nTotal: {time.time()-t0:.0f}s. Filas OK={len(rows)}, fallos={len(failures)}")

    scores_df = pd.DataFrame(rows)
    scores_path = OUT_DIR / "scores_por_condicion.csv"
    scores_df.to_csv(scores_path, index=False)
    print(f"Scores crudos guardados en {scores_path}")

    if failures:
        fail_df = pd.DataFrame(failures, columns=["condicion", "anon_id", "error"])
        fail_path = OUT_DIR / "phase1_eval_failures.csv"
        fail_df.to_csv(fail_path, index=False)
        print(f"Fallos guardados en {fail_path}")

    # matriz de resiliencia: AUC/EER por condicion x configuracion
    results = []
    for condition in CONDITIONS:
        sub = scores_df[scores_df["condicion"] == condition]
        if len(sub) == 0:
            continue
        y = (sub["label"] == "synthetic").astype(int).values
        row = {"condicion": condition, "n": len(sub)}
        for branch, col in [("espectral", "score_espectral"), ("prosodica", "score_prosodica"),
                             ("fusion", "score_fusion")]:
            auc = roc_auc_score(y, sub[col].values)
            eer, _ = compute_eer(y, sub[col].values)
            row[f"auc_{branch}"] = round(auc, 4)
            row[f"eer_{branch}"] = round(eer * 100, 2)
        results.append(row)

    matrix_df = pd.DataFrame(results)
    matrix_path = OUT_DIR / "matriz_resiliencia_corrupcion.csv"
    matrix_df.to_csv(matrix_path, index=False)

    print("\n=== Matriz de Resiliencia (out-of-fold, leave-one-fold-out) ===")
    print(matrix_df.to_string(index=False))
    print(f"\nGuardada en {matrix_path}")

    # analisis: caida vs clean, por rama
    clean_row = matrix_df[matrix_df["condicion"] == "clean"].iloc[0]
    print("\n=== Delta de AUC vs. clean, por condicion y rama ===")
    deltas = []
    for _, row in matrix_df.iterrows():
        if row["condicion"] == "clean":
            continue
        d = {
            "condicion": row["condicion"],
            "delta_auc_espectral": round(row["auc_espectral"] - clean_row["auc_espectral"], 4),
            "delta_auc_prosodica": round(row["auc_prosodica"] - clean_row["auc_prosodica"], 4),
            "delta_auc_fusion": round(row["auc_fusion"] - clean_row["auc_fusion"], 4),
        }
        deltas.append(d)
    deltas_df = pd.DataFrame(deltas).sort_values("delta_auc_espectral")
    print(deltas_df.to_string(index=False))
    deltas_path = OUT_DIR / "delta_vs_clean.csv"
    deltas_df.to_csv(deltas_path, index=False)
    print(f"\nGuardado en {deltas_path}")

    worst_for_spectral = deltas_df.iloc[0]
    print(
        f"\n[DIAGNOSTICO] La condicion que mas dana a la rama espectral es "
        f"'{worst_for_spectral['condicion']}' (delta AUC={worst_for_spectral['delta_auc_espectral']:+.4f}). "
        "Esa es la corrupcion mas urgente para activar en TelephonyAugmenter durante el entrenamiento "
        "de Fase 2, siguiendo la logica del equipo: si degrada mucho el AUC espectral, es evidencia de "
        "que la rama SI dependia de un atajo de canal en esa dimension especifica."
    )


if __name__ == "__main__":
    main()
