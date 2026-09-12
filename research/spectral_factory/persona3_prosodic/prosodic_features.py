"""
Rama prosodica — Tareas 3.1 y 3.2: Shimmer CS3(DeltaDelta) como serie temporal,
via F0, con enmascaramiento VAD (turns.json como VAD oracle) ANTES del pooling.
Ver prompt maestro Persona 3 y CONTRATOS.md.

Decision de diseno — F0 tracker (REVISADA, ver HALLAZGOS_LOG.md / benchmark de
latencia): se reemplazo librosa.pyin por parselmouth (Praat, autocorrelacion en
C). Motivo: el benchmark de latencia end-to-end (persona1_reference_calibration)
encontro que pYIN tardaba 7.4s sobre una llamada de 150s -- el 99% del tiempo
total del pipeline, muy por encima del objetivo de <1s del equipo. La capa
probabilistica/Viterbi de pYIN existe para decidir voiced/unvoiced con
incertidumbre, pero esa decision YA es redundante aqui: usamos turns.json como
VAD oracle para restringir el analisis a tramos de habla del canal 0 (ver
seccion 2 de este mismo modulo), asi que pagar el costo de la capa probabilistica
de pYIN no aporta nada que no tuvieramos ya. parselmouth expone el pitch
tracker de Praat (autocorrelacion, implementado en C), con la misma convencion
de "0 Hz = unvoiced" que necesitamos, y corre ~680x mas rapido en la practica
(7.4s -> ~0.01s en la misma llamada de referencia).

Rango de F0 y hop -- AJUSTADOS DOS VECES (ver HALLAZGOS_LOG.md):
  1ra pasada (fix de latencia): 80-350 Hz, hop 20ms. Bajo la latencia a 0.159s
  total (de 1s de presupuesto), pero el AUC aislado de la rama cayo de 0.842 a
  0.803 -- costo real, no gratis.
  2da pasada (recuperar AUC, dado que sobraba presupuesto de latencia: 159ms de
  1000ms usados): se ensancho a 60-450 Hz (cubre mejor voces graves masculinas
  con creaky voice <80Hz y picos agudos femeninos >350Hz) y se afino el hop a
  10ms (mas resolucion temporal para el calculo de Shimmer CS3 ciclo a ciclo).
  Resultado de este segundo ajuste documentado en HALLAZGOS_LOG.md.

Decision de diseno — cadena de pulsos (pitch-synchronous), no Shimmer por frame:
  Shimmer es una medida de perturbacion de amplitud PERIODO A PERIODO, no una
  metrica de ventana STFT. Por eso, a partir del track de F0 por frame
  reconstruimos una marca de pulsos glotales aproximada (como el PointProcess de
  Praat) y medimos amplitud pulso-a-pulso. CS3 (APQ3, "amplitude perturbation
  quotient" de 3 puntos) se calcula sobre ventanas deslizantes de pulsos, dando
  una serie temporal de shimmer (no un solo numero por llamada) sobre la que
  luego se calculan Delta y DeltaDelta.
"""

from dataclasses import dataclass

import numpy as np
import parselmouth

F0_MIN_HZ = 60.0
F0_MAX_HZ = 450.0
F0_HOP_MS = 10             # afinado de 20ms -> 10ms tras confirmar margen de
                           # sobra en el presupuesto de latencia (159/1000ms)
F0_HOP_SAMPLES_8K = 80     # equivalente en muestras a 8kHz (10ms), usado para
                           # acotar el final de un tramo voiced (ver _mark_pulses)

PULSES_PER_WINDOW = 10     # tamano de ventana deslizante para CS3 (en # de pulsos glotales)
PULSE_WINDOW_HOP = 5       # salto de la ventana deslizante (50% overlap)
MIN_PULSES_FOR_APQ3 = 3    # APQ3 necesita vecino izquierdo y derecho


def _load_turns_channel0(turns_path):
    import json
    with open(turns_path) as f:
        data = json.load(f)
    turns = [t for t in data["turns"] if t["channel"] == 0]
    turns.sort(key=lambda t: t["start"])
    return turns


def estimate_f0_track(audio, sr):
    """Pitch tracker de Praat (via parselmouth) sobre el canal completo.
    Devuelve (f0, voiced_flag, times) a resolucion de frame (20ms hop). f0=0.0
    en frames unvoiced, misma convencion que pyworld -- se mantiene el mismo
    contrato de salida (f0, voiced_flag, frame_times) que la version anterior
    basada en librosa.pyin, asi que el resto del pipeline (_voiced_runs_in_range,
    _mark_pulses, etc.) no necesito ni necesita cambiar."""
    snd = parselmouth.Sound(audio.astype(np.float64), sampling_frequency=sr)
    pitch = snd.to_pitch(time_step=F0_HOP_MS / 1000.0, pitch_floor=F0_MIN_HZ, pitch_ceiling=F0_MAX_HZ)
    f0 = pitch.selected_array["frequency"]  # 0.0 = unvoiced (convencion Praat)
    times = pitch.xs()
    voiced_flag = f0 > 0
    return f0, voiced_flag, times


def _voiced_runs_in_range(voiced_flag, frame_times, t_start, t_end):
    """Tramos [start_frame, end_frame) contiguos con voiced_flag=True, restringidos
    a la ventana temporal [t_start, t_end) de un turno del canal 0 (VAD oracle)."""
    in_range = (frame_times >= t_start) & (frame_times < t_end)
    idx = np.where(in_range & voiced_flag)[0]
    if len(idx) == 0:
        return []
    runs = []
    run_start = idx[0]
    prev = idx[0]
    for i in idx[1:]:
        if i != prev + 1:
            runs.append((run_start, prev + 1))
            run_start = i
        prev = i
    runs.append((run_start, prev + 1))
    return runs


def _mark_pulses(audio, sr, f0, frame_times, frame_lo, frame_hi):
    """Marca pulsos glotales aproximados (pitch-synchronous) dentro de un tramo
    voiced contiguo [frame_lo, frame_hi) de f0_track. Devuelve indices de muestra
    de cada pulso."""
    t0 = frame_times[frame_lo]
    sample_start = int(t0 * sr)
    sample_end = min(int(frame_times[frame_hi - 1] * sr) + F0_HOP_SAMPLES_8K, len(audio))
    if sample_end - sample_start < 8:
        return np.array([], dtype=int)

    def f0_at_sample(s):
        t = s / sr
        frame_idx = np.searchsorted(frame_times[frame_lo:frame_hi], t) + frame_lo
        frame_idx = np.clip(frame_idx, frame_lo, frame_hi - 1)
        return f0[frame_idx]

    # alinea el primer pulso a un pico local cerca del inicio del tramo
    search_w = max(4, int(sr / f0_at_sample(sample_start) / 4))
    seg = audio[sample_start:min(sample_start + 2 * search_w, sample_end)]
    if len(seg) == 0:
        return np.array([], dtype=int)
    pulses = [sample_start + int(np.argmax(np.abs(seg)))]

    while True:
        cur = pulses[-1]
        f0_cur = f0_at_sample(cur)
        if not np.isfinite(f0_cur) or f0_cur <= 0:
            break
        period = sr / f0_cur
        next_expected = cur + period
        if next_expected >= sample_end:
            break
        window = max(2, int(0.25 * period))
        lo, hi = int(next_expected - window), int(next_expected + window)
        lo, hi = max(sample_start, lo), min(sample_end, hi)
        if hi <= lo:
            break
        local = audio[lo:hi]
        next_pulse = lo + int(np.argmax(np.abs(local)))
        if next_pulse <= cur:
            break
        pulses.append(next_pulse)

    return np.array(pulses, dtype=int)


def _pulse_amplitudes(audio, sr, pulses, half_window_ms=1.5):
    """Amplitud de pico local alrededor de cada pulso, en vez de solo la muestra
    exacta, para tolerar pequenos errores de marcado del pulso."""
    hw = max(1, int(sr * half_window_ms / 1000))
    amps = []
    for p in pulses:
        lo, hi = max(0, p - hw), min(len(audio), p + hw + 1)
        amps.append(np.max(np.abs(audio[lo:hi])))
    return np.array(amps, dtype=np.float64)


def _apq3_sliding(amplitudes, window=PULSES_PER_WINDOW, hop=PULSE_WINDOW_HOP):
    """CS3 (APQ3) sobre ventanas deslizantes de pulsos -> serie temporal de shimmer,
    en vez de un solo numero por tramo voiced."""
    n = len(amplitudes)
    if n < max(window, MIN_PULSES_FOR_APQ3 + 2):
        if n < MIN_PULSES_FOR_APQ3:
            return np.array([])
        window = n  # ventana unica si el tramo es corto pero utilizable
    values = []
    start = 0
    while start + window <= n or (start == 0 and window > n):
        w = amplitudes[start:start + window] if start + window <= n else amplitudes[start:]
        if len(w) >= MIN_PULSES_FOR_APQ3:
            local_perturb = [
                abs(w[i] - np.mean(w[i - 1:i + 2])) for i in range(1, len(w) - 1)
            ]
            mean_amp = np.mean(w)
            if mean_amp > 0:
                values.append(100.0 * np.mean(local_perturb) / mean_amp)
        if start + window >= n:
            break
        start += hop
    return np.array(values)


def shimmer_series_for_call(audio, sr, turns_ch0):
    """Serie temporal completa de shimmer CS3 para un clip, respetando el VAD
    oracle (turns_ch0) y sin puentear tramos unvoiced entre turnos o dentro de
    ellos. Devuelve un array 1D (puede ser vacio si no hay suficiente voz)."""
    f0, voiced_flag, frame_times = estimate_f0_track(audio, sr)

    all_shimmer = []
    for turn in turns_ch0:
        runs = _voiced_runs_in_range(voiced_flag, frame_times, turn["start"], turn["end"])
        for frame_lo, frame_hi in runs:
            pulses = _mark_pulses(audio, sr, f0, frame_times, frame_lo, frame_hi)
            if len(pulses) < MIN_PULSES_FOR_APQ3:
                continue
            amps = _pulse_amplitudes(audio, sr, pulses)
            shimmer_vals = _apq3_sliding(amps)
            if len(shimmer_vals) > 0:
                all_shimmer.append(shimmer_vals)

    if not all_shimmer:
        return np.array([])
    return np.concatenate(all_shimmer)


@dataclass
class ProsodicLatent:
    anon_id: str
    n_shimmer_frames: int
    features: dict  # nombre_columna -> valor


def pool_shimmer_series(shimmer_series):
    """Pooling estadistico (mean, std, p10, p90) sobre shimmer, Delta y DeltaDelta.

    Decision de diseno (Tarea 3.3 del prompt): con solo 353 llamadas de
    entrenamiento, un pooling atencional (aprendido) tiene alto riesgo de
    sobreajuste -- necesitaria su propio set de parametros entrenables sin
    suficientes datos para regularizarlo bien. Global Average Pooling (solo la
    media) es la opcion mas simple y la que mejor generaliza con pocos datos,
    pero descarta precisamente la dispersion, que es donde la literatura
    (Li et al., Boenninghoff 2021) reporta que aparecen las diferencias entre voz
    real y sintetica. Pooling estadistico (media+std+percentiles) es el punto
    medio: sigue siendo de muy baja dimensionalidad (12 features fijas
    independientemente de la duracion de la llamada) pero conserva informacion
    de dispersion. Se eligio este ultimo.
    """
    if len(shimmer_series) < MIN_PULSES_FOR_APQ3:
        return None

    d1 = np.diff(shimmer_series) if len(shimmer_series) > 1 else np.array([])
    d2 = np.diff(d1) if len(d1) > 1 else np.array([])

    def stats(x, prefix):
        if len(x) == 0:
            return {f"{prefix}_mean": np.nan, f"{prefix}_std": np.nan,
                    f"{prefix}_p10": np.nan, f"{prefix}_p90": np.nan}
        return {
            f"{prefix}_mean": float(np.mean(x)),
            f"{prefix}_std": float(np.std(x)),
            f"{prefix}_p10": float(np.percentile(x, 10)),
            f"{prefix}_p90": float(np.percentile(x, 90)),
        }

    feats = {}
    feats.update(stats(shimmer_series, "shimmer"))
    feats.update(stats(d1, "dshimmer"))
    feats.update(stats(d2, "ddshimmer"))
    return feats


def extract_prosodic_latent(anon_id, audio, sr, turns_ch0):
    """Pipeline completo Tarea 3.1 + 3.2 + pooling: audio+turns -> vector latente."""
    series = shimmer_series_for_call(audio, sr, turns_ch0)
    feats = pool_shimmer_series(series)
    if feats is None:
        return None
    return ProsodicLatent(anon_id=anon_id, n_shimmer_frames=len(series), features=feats)
