"""
Tarea 5 del prompt de Persona 1 (Regi): benchmark de latencia y memoria en CPU
para una llamada completa, desglosado por etapa (no solo el total) -- el
objetivo del equipo es < 1 segundo en la version final.

Mide, para UNA llamada real (~150s de duracion, elegida del dataset):
  1. Extraccion de features prosodicas (Shimmer CS3, persona3_prosodic)
  2. Extraccion de features espectrales (LFCC placeholder, persona2_reference_spectral)
  3. Inferencia de cada rama (regresion logistica ya entrenada)
  4. Fusion tardia (concat + regresion logistica)
  5. Calibracion de Platt
  6. Total end-to-end

El fit de los modelos (entrenamiento) NO se cronometra -- eso pasa una sola
vez offline, nunca en el camino de inferencia real. Solo se cronometra lo que
correria en produccion por cada llamada nueva.

IMPORTANTE -- cold start vs. warm (hallazgo de este mismo script): la PRIMERA
llamada a librosa.stft/scipy.fft en un proceso Python paga un costo de
inicializacion (~30-40x mas lento que llamadas subsecuentes) que no se repite
mientras el proceso siga vivo. Un servidor real (POST /detect) arranca una vez
y atiende muchas llamadas, asi que el numero que importa para el dia a dia es
el WARM, no el de la primera peticion. Por eso este script mide ambos: hace un
"warm-up" completo (descartado) antes de la medicion real, y reporta el costo
de cold-start aparte para que el equipo sepa si les importa (si el endpoint
corre en un contenedor que se reinicia seguido / serverless, si les importa;
si es un servidor que se queda prendido, no).
"""

import json
import time
import tracemalloc
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from persona3_prosodic.prosodic_features import extract_prosodic_latent, _load_turns_channel0
from persona2_reference_spectral.lfcc_features import extract_spectral_latent
from persona1_reference_calibration.format_detect_response import load_platt_params, to_detect_response

REPO_ROOT = Path(__file__).parent.parent
AUDIO_DIR = REPO_ROOT / "audio"
TURNS_DIR = REPO_ROOT / "turns"
DATA_DIR = REPO_ROOT / "persona3_prosodic" / "data"
OUT_DIR = Path(__file__).parent / "data"

TARGET_DURATION_S = 150  # el "clip de 150 segundos" de referencia del plan del equipo


def pick_reference_call():
    """Elige la llamada del manifest cuya duracion este mas cerca de 150s, para
    que el benchmark sea representativo del caso de uso real, no del caso mas
    corto/facil del dataset."""
    manifest = pd.read_csv(REPO_ROOT / "manifest.csv")
    manifest["dist"] = (manifest["duration_s"] - TARGET_DURATION_S).abs()
    row = manifest.sort_values("dist").iloc[0]
    return row["anon_id"], row["duration_s"]


def _fit_final_models():
    """Entrena (sin cronometrar -- esto pasa offline UNA vez) los modelos
    finales de cada rama sobre TODOS los datos disponibles, para poder medir
    la latencia de INFERENCIA real, no de entrenamiento."""
    prosodic_df = pd.read_csv(DATA_DIR / "latents_prosodic.csv").dropna()
    spectral_df = pd.read_csv(DATA_DIR / "latents_spectral.csv").dropna()

    prosodic_feats = [c for c in prosodic_df.columns if c not in ("anon_id", "label", "fold")]
    spectral_feats = [c for c in spectral_df.columns if c not in ("anon_id", "label", "fold")]

    def fit(df, feats):
        X = df[feats].values
        y = (df["label"] == "synthetic").astype(int).values
        scaler = StandardScaler().fit(X)
        model = LogisticRegression(max_iter=1000, class_weight="balanced")
        model.fit(scaler.transform(X), y)
        return scaler, model

    pros_scaler, pros_model = fit(prosodic_df, prosodic_feats)
    spec_scaler, spec_model = fit(spectral_df, spectral_feats)

    merged = prosodic_df.merge(spectral_df, on=["anon_id", "label", "fold"], how="inner",
                                suffixes=("_pros", "_spec"))
    fusion_feats = [c for c in merged.columns if c not in ("anon_id", "label", "fold")]
    fus_scaler, fus_model = fit(merged, fusion_feats)

    return {
        "prosodic": (pros_scaler, pros_model, prosodic_feats),
        "spectral": (spec_scaler, spec_model, spectral_feats),
        "fusion": (fus_scaler, fus_model, fusion_feats),
    }


def run_inference_once(anon_id, wav_path, turns_path, models, platt_params):
    """Corre el pipeline completo de inferencia UNA vez sobre una llamada,
    devuelve (timings_dict, detect_response). Se llama dos veces desde main():
    una de warm-up (descartada) y una cronometrada de verdad."""
    timings = {}

    t0 = time.perf_counter()
    audio, sr = sf.read(wav_path, always_2d=True)
    ch0 = audio[:, 0].astype(float)
    turns = _load_turns_channel0(turns_path)
    timings["io_carga_audio_turns"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    pros_latent = extract_prosodic_latent(anon_id, ch0, sr, turns)
    timings["extraccion_prosodica_shimmer"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    spec_latent = extract_spectral_latent(anon_id, ch0, sr, turns)
    timings["extraccion_espectral_lfcc"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    pros_scaler, pros_model, pros_feats = models["prosodic"]
    pros_vec = np.array([[pros_latent.features[f] for f in pros_feats]])
    pros_model.predict_proba(pros_scaler.transform(pros_vec))
    timings["inferencia_rama_prosodica"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    spec_scaler, spec_model, spec_feats = models["spectral"]
    spec_vec = np.array([[spec_latent.features[f] for f in spec_feats]])
    spec_model.predict_proba(spec_scaler.transform(spec_vec))
    timings["inferencia_rama_espectral"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    fus_scaler, fus_model, fus_feats = models["fusion"]
    all_feats = {**pros_latent.features, **spec_latent.features}
    fus_vec = np.array([[all_feats[f] for f in fus_feats]])
    fus_score = fus_model.predict_proba(fus_scaler.transform(fus_vec))[0, 1]
    timings["fusion_tardia"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    detect_response = to_detect_response(fus_score, platt_params)
    timings["calibracion_platt_y_formato"] = time.perf_counter() - t0

    timings["TOTAL"] = sum(v for k, v in timings.items() if k != "TOTAL")
    return timings, detect_response


def main():
    OUT_DIR.mkdir(exist_ok=True)
    anon_id, duration = pick_reference_call()
    print(f"Llamada de referencia: {anon_id} ({duration:.0f}s)")

    print("Entrenando modelos finales (offline, no se cronometra)...")
    models = _fit_final_models()
    platt_params = load_platt_params("fusion")

    wav_path = AUDIO_DIR / f"{anon_id}.wav"
    turns_path = TURNS_DIR / f"{anon_id}.json"

    print("Warm-up (descartado -- absorbe el costo de cold-start de FFT/BLAS)...")
    tracemalloc.start()
    cold_timings, _ = run_inference_once(anon_id, wav_path, turns_path, models, platt_params)
    _, peak_cold = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    tracemalloc.start()
    warm_timings, detect_response = run_inference_once(anon_id, wav_path, turns_path, models, platt_params)
    _, peak_warm = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print("\nDesglose de latencia -- PRIMERA llamada del proceso (cold start):")
    for stage, t in cold_timings.items():
        marker = " <<<" if stage == "TOTAL" else ""
        print(f"  {stage:<35} {t*1000:>8.1f} ms{marker}")

    print("\nDesglose de latencia -- llamadas SUBSECUENTES (warm, lo que importa en un servidor vivo):")
    for stage, t in warm_timings.items():
        marker = " <<<" if stage == "TOTAL" else ""
        print(f"  {stage:<35} {t*1000:>8.1f} ms{marker}")

    print(f"\nMemoria pico (cold): {peak_cold / 1024 / 1024:.1f} MB   (warm): {peak_warm / 1024 / 1024:.1f} MB")
    print(f"Respuesta POST /detect: {json.dumps(detect_response)}")

    objetivo_cumplido_warm = warm_timings["TOTAL"] < 1.0
    objetivo_cumplido_cold = cold_timings["TOTAL"] < 1.0
    print(f"\nObjetivo <1s (warm, servidor vivo): {'CUMPLIDO' if objetivo_cumplido_warm else 'NO CUMPLIDO'} "
          f"({warm_timings['TOTAL']:.3f}s)")
    print(f"Objetivo <1s (cold, primera peticion tras arrancar): {'CUMPLIDO' if objetivo_cumplido_cold else 'NO CUMPLIDO'} "
          f"({cold_timings['TOTAL']:.3f}s)")

    result = {
        "anon_id": anon_id,
        "duration_s": float(duration),
        "cold_start_timings_ms": {k: v * 1000 for k, v in cold_timings.items()},
        "warm_timings_ms": {k: v * 1000 for k, v in warm_timings.items()},
        "peak_memory_mb_cold": peak_cold / 1024 / 1024,
        "peak_memory_mb_warm": peak_warm / 1024 / 1024,
        "objetivo_1s_cumplido_warm": objetivo_cumplido_warm,
        "objetivo_1s_cumplido_cold": objetivo_cumplido_cold,
        "detect_response": detect_response,
    }
    out_path = OUT_DIR / "latency_benchmark.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResultado guardado en {out_path}")


if __name__ == "__main__":
    main()
