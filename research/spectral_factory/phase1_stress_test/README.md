# Fase 1: Stress Test de Robustez sobre el ZIP de Persona 2

Evalúa las 3 configuraciones (espectral sola, prosódica sola, fusión) contra
las 6 condiciones del ZIP de robustez de Ricardo (`robustness_phase1 (2).zip`):
`clean`, `noise_snr10`, `pitch_up1st`, `timestretch_0.9x`, `lowpass_3400hz`,
`opus_16kbps`. No mezcla condiciones en una sola métrica — cada una se reporta
por separado.

## Decisión metodológica crítica: leave-one-fold-out, no el modelo final

Las 353 llamadas del ZIP de Ricardo son las MISMAS 353 llamadas que usamos
para entrenar nuestros modelos — su "validation set" no es un held-out nuevo.
Si evaluáramos con el modelo final (entrenado con las 353), cada llamada se
evaluaría con un modelo que ya la vio (en su versión limpia) durante
entrenamiento — optimista, no confiable, el mismo tipo de error que ya nos
costó caro con la rama espectral placeholder (ver `HALLAZGOS_LOG.md`).

Por eso este script entrena un modelo POR FOLD (los 5 folds de
`persona3_prosodic/data/splits.csv`) y cada llamada se evalúa SIEMPRE con el
modelo de los otros 4 folds — nunca con uno que la haya visto, ni siquiera en
su versión limpia.

## Sin extraer el ZIP a disco

El ZIP descomprimido pesa ~12GB; la máquina solo tenía ~17GB libres al momento
de correr esto. En vez de descomprimir, `eval_stress_test.py` lee cada
`.wav`/`.json` directamente del ZIP a un buffer en memoria (`zipfile` +
`io.BytesIO`), extrae features, y descarta — nunca toca disco con los datos
corrompidos.

## Uso

```bash
python -m phase1_stress_test.eval_stress_test --zip "/ruta/a/robustness_phase1 (2).zip"
```

## Resultados (ver `data/matriz_resiliencia_corrupcion.csv` y `data/delta_vs_clean.csv`)

| Condición | AUC espectral | ΔAUC espectral | AUC prosódica | ΔAUC prosódica | AUC fusión |
|---|---|---|---|---|---|
| clean | 1.0000 | — | 0.8283 | — | 0.9999 |
| noise_snr10 | 0.9714 | -0.0286 | 0.8005 | -0.0278 | 0.9756 |
| pitch_up1st | 0.9745 | -0.0255 | 0.6250 | **-0.2033** | 0.9652 |
| timestretch_0.9x | 0.9893 | -0.0107 | 0.6208 | **-0.2075** | 0.9845 |
| lowpass_3400hz | 0.9951 | -0.0049 | 0.8370 | +0.0087 | 0.9960 |
| opus_16kbps | 0.9985 | -0.0015 | 0.7480 | -0.0803 | 0.9990 |

## ⚠️ Esto CONTRADICE la hipótesis del equipo — leer antes de decidir Fase 2

La hipótesis original (ver el plan de experimentos) era: *"la rama espectral
debería caer con lowpass/opus (atajo de canal); la rama prosódica debería
mantenerse estable (señal ortogonal)"*. Los datos dicen lo contrario en ambos
lados:

1. **La rama espectral es la MÁS robusta a lowpass/opus, y la MENOS robusta a
   ruido** — exactamente al revés de lo esperado. Si el atajo de canal
   (ancho de banda telefónico) fuera lo que domina esta rama, lowpass/opus
   (que atacan directamente esa dimensión) deberían destruirla más que el
   ruido gaussiano, no menos. Esto es evidencia ADICIONAL a favor de la
   hipótesis abierta en `HALLAZGOS_LOG.md` (posible fuga de motor TTS / pocas
   voces sintéticas) — ninguna de estas 5 corrupciones logra tumbar el AUC de
   forma significativa, lo cual es preocupante, no tranquilizador.
2. **La rama prosódica SÍ tiene un punto débil real y no anticipado**: pitch
   shift y time-stretch la destruyen (ΔAUC ≈ -0.20, caída de 0.83 a ~0.62).
   Esto tiene sentido técnico — Shimmer se calcula pitch-synchronous (períodos
   glóticos detectados vía F0), así que alterar directamente el pitch o la
   duración ataca la base misma del cálculo. La robustez a corrupción de canal
   que reportamos antes (`persona3_prosodic/README.md`, `robustness_check.py`)
   sigue siendo válida — shimmer SÍ es robusto a códec/ruido/paso-bajo — pero
   **no es robusto a pitch-shift ni a time-stretch**, un eje de vulnerabilidad
   distinto que no se había probado hasta ahora.

## Recomendación para Fase 2 (TelephonyAugmenter, P=0.5)

Con esta evidencia, activar `pitch_up1st` y `timestretch_0.9x` en el
`DataLoader` de entrenamiento tiene más justificación empírica que activar
`lowpass`/`opus` — son las corrupciones que de verdad mueven el comportamiento
del modelo (via la rama prosódica y, en menor medida, la fusión), mientras que
lowpass/opus apenas mueven nada en ninguna rama. `noise_snr10` es la que más
daña a la espectral (aunque poco, -0.029) y vale la pena incluirla también por
generalización. La pregunta de fondo (¿por qué la rama espectral es casi
perfecta bajo TODAS las condiciones?) sigue sin resolverse y apunta de nuevo a
verificar diversidad de motores TTS en el dataset, no a un problema de
selección de corrupciones de entrenamiento.


## Validación de la Fase 2 propuesta (`train_augmented_prosodic.py`)

Se implementó la recomendación concreta (activar pitch-shift ±1 semitono y
time-stretch 0.9x-1.1x en `TelephonyAugmenter`) y se validó **contra el ZIP
real de Ricardo**, no contra nuestra propia reproducción del corruption (para
que la prueba no fuera circular).

| Condición | Sin augmentación | Con augmentación | Δ |
|---|---|---|---|
| clean | 0.8283 | 0.8088 | -0.0195 |
| pitch_up1st | 0.6250 | 0.7078 | **+0.0828** |
| timestretch_0.9x | 0.6208 | 0.6969 | **+0.0761** |

**Veredicto honesto**: la augmentación SÍ ayuda de forma real y medible, pero
**no confirma la hipótesis completa** de subir a >0.80 — se queda a medio
camino (~0.70), con un costo pequeño en clean (-0.02). Hipótesis de por qué:
este experimento generó UNA sola copia aumentada por llamada para un
clasificador clásico, un régimen mucho más débil que el que tendría un
DataLoader real con augmentación estocástica por época (muchos valores
aleatorios de pitch/rate por llamada a lo largo del entrenamiento). La
dirección es correcta; el placeholder simplemente no puede demostrar la
recuperación completa. Ver `HALLAZGOS_LOG.md` adenda 4 para el detalle
completo.
