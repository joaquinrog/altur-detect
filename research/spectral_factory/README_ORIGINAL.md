# Análisis forense de canal — dataset de detección human vs synthetic caller

Pipeline exploratorio (no un clasificador) que audita, antes de entrenar nada, si un
dataset de detección "humano vs. sintético" en llamadas telefónicas se puede resolver
con un atajo del canal de grabación/transmisión en vez de una diferencia real de
síntesis de voz.

## Hipótesis que se descarta

Si el audio humano y el audio sintético se hubieran inyectado por canales de
grabación distintos (p. ej. humano vía línea telefónica real con su ruido
característico, sintético inyectado directo desde el pipeline de texto a voz sin
pasar por ese mismo canal), cualquier modelo aprendería a distinguir el canal de
grabación, no la síntesis en sí. Funcionaría perfecto en train/val y se caería en
cualquier set de evaluación oculto con voces y grabaciones nunca vistas.

## Qué hace `forensic_analysis.py`

Sobre una muestra estratificada del split de entrenamiento (40 clips humanos + 40
sintéticos), separa silencio y habla del canal del llamante usando los segmentos de
turno provistos, y compara ocho características de bajo nivel del canal —ninguna
relacionada con el contenido lingüístico ni con la calidad de síntesis en sí—:

1. Piso de ruido espectral (Welch PSD sobre silencio)
2. Espectro promedio de largo plazo (LTAS) sobre tramos con habla
3. Ancho de banda ocupado (energía dentro/fuera de la banda telefónica clásica 300–3400 Hz)
4. Espectrogramas de ejemplo (inspección visual — no incluidos en este repo, ver nota abajo)
5. RMS / loudness de los tramos con habla
6. Zero-crossing rate
7. Spectral flatness sobre silencio (artefactos de cuantización/códec)
8. Patrones tipo "peine" en el silencio + hum de línea eléctrica (50/60 Hz y armónicos)

Para cada característica numérica corre un test de Mann-Whitney U y calcula el AUC de
un clasificador de una sola variable. Un AUC > 0.85 con una sola característica de
canal se marca como "sospechoso" — bandera roja de dataset amañado.

## Resultado

Ver [`analisis_forense/resumen_estadistico.md`](analisis_forense/resumen_estadistico.md)
y [`analisis_forense/resumen_interpretativo.md`](analisis_forense/resumen_interpretativo.md).
En resumen: ninguna característica cruza el umbral de alarma (todas con AUC ≤ 0.85);
el ancho de banda telefónico es la más alta (AUC ≈ 0.82) y queda marcada como
"borderline" a vigilar, junto con el score de patrones tipo peine y el zero-crossing
rate (~0.79 cada una).

## Nota sobre los datos originales

El dataset fuente no se redistribuye aquí: este repo solo contiene el script de
análisis y estadísticas agregadas (histogramas, boxplots, PSD promediadas por clase,
tablas resumen) sobre 40+40 clips. Se excluyó deliberadamente cualquier gráfica que
mostrara espectrogramas de llamadas individuales identificables por su id anónimo,
para respetar la cláusula de "no redistribuir" de los términos del dataset original.

## Uso

```bash
pip install librosa scipy numpy pandas matplotlib soundfile requests tabulate
python forensic_analysis.py
```

Requiere `manifest.csv`, `turns/<anon_id>.json` y `audio/<anon_id>.wav` (estéreo,
8 kHz, canal 0 = llamante) en la raíz, provistos por separado por el dueño del dataset.
