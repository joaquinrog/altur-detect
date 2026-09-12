# Model card

## Estado

**UNK/TODO(A3):** el candidato de produccion no esta congelado. Esta card es el contrato que debe
completar cada bundle candidato; no describe un modelo ganador ni resultados finales.

| Campo | Valor |
|---|---|
| Nombre/version | `TODO(A3)` |
| Bundle SHA-256 | `TODO(A4.1)` |
| Imagen por digest | `TODO(A4.3)` |
| Commit limpio | `TODO(A3)` |
| Digest de codigo efectivo | `TODO(A3)` |
| Protocolo | `TODO(A3)` |
| Semillas | `TODO(A3)` |
| Fecha UTC | `TODO(A3)` |
| Licencia del codigo | `UNK: decision de Joaquin` |

## Uso previsto

Clasificar si la voz del canal 0 de un WAV telefonico es sintetica. El canal 1 es contexto de
agente y control; no cambia la definicion de etiqueta. La salida es una decision binaria y una
confianza entre 0 y 1.

No esta previsto para identificar personas, autenticar hablantes, detectar replay/liveness,
atribuir un proveedor TTS ni tomar una decision bancaria automatizada sin controles adicionales.

## Entrada y salida

- Entrada objetivo: WAV estereo, 8 kHz; ch0 caller y ch1 agente.
- Entrada tolerada: **TODO(A4.2)** documentar mono/remuestreo finalmente habilitados.
- Salida: `{"is_synthetic": bool, "confidence": float}`.
- Clase positiva: `synthetic = 1`.
- Horizonte de audio usado: `TODO(A3)`.

## Arquitectura efectiva

| Componente | Referencia versionada |
|---|---|
| Segmentacion | `TODO(A3)` |
| Features acusticas | `TODO(A3)` |
| Features conductuales | `TODO(A3)` |
| Modelo(s) | `TODO(A3)` |
| Fusion/desacuerdo | `TODO(A3)` |
| Calibracion | `TODO(A3)` |
| Umbral | `TODO(A3)` |

Solo componentes `product_safe=True` pueden entrar al bundle. Parselmouth y openSMILE son
`product_safe=False` y quedan fuera de la imagen.

## Evaluacion

No completar una celda sin un artefacto reproducible.

| Condicion | Metrica | Resultado | n | Unidad | Split | Protocolo | Seed | Commit | Digest efectivo |
|---|---|---:|---:|---|---|---|---|---|---|
| OOF clean | `TODO` | `TODO(A3)` | `TODO` | llamada | train | `TODO` | `TODO` | `TODO` | `TODO` |
| OOF codec | `TODO` | `TODO(A3)` | `TODO` | llamada | train | `TODO` | `TODO` | `TODO` | `TODO` |
| OOF lowpass/highpass | `TODO` | `TODO(A3)` | `TODO` | llamada | train | `TODO` | `TODO` | `TODO` | `TODO` |
| canal 1 | `TODO` | `TODO(A3)` | `TODO` | llamada | train | `TODO` | `TODO` | `TODO` | `TODO` |
| silence-only | `TODO` | `TODO(A3)` | `TODO` | llamada | train | `TODO` | `TODO` | `TODO` | `TODO` |
| calibracion | Brier/descomposicion | `TODO(A3)` | `TODO` | llamada | train OOF | `TODO` | `TODO` | `TODO` | `TODO` |
| latencia CPU | p50/p95 | `TODO(A4.3)` | `TODO` | llamada | fixture sintetico | smoke versionado | no aplica | `TODO` | `TODO` |

**FACT:** `val` no se usa para iterar y no se completa aqui durante A3.

## Limitaciones

1. **D-A1.2:** no se puede afirmar generalizacion a hablante no visto sobre el dataset oficial;
   el manifest no trae llave de union y grupo = llamada es un fallback conservador.
2. **D-A1.5:** la prevalencia cambia: train 59.9 % sintetico frente a val 47.9 %. El prior oculto
   es `UNK`; la confianza puede trasladarse mal.
3. **D-A1.6:** las clases tienen cadenas de produccion distintas. El sintetico corta alrededor de
   3400 Hz y el humano llega a Nyquist; un AUC alto puede medir cadena, no voz. La causa depende de
   la pregunta 4 del booth y sigue `UNK`.
4. El dataset oficial no demuestra generalizacion cross-vendor.
5. La rama conductual mide un proxy de la etiqueta y puede fallar con replay humano, humanos
   lentos o agentes de voz rapidos.
6. **TODO(A3):** documentar fallos del extractor por clase y modos degradados del candidato.

Proveniencia D-A1.5: `n=282` train y `n=71` val; unidad = llamada; protocolo = conteo del manifest
v1.0; seed = no aplica; commit = `a5979a4`; digest efectivo = no aplica al conteo textual.
Proveniencia D-A1.6: `n=282`; unidad = llamada; split = train; protocolo = `official_v1`; seed de
submuestra y digest efectivo = `UNK`; commit = `a5979a4`. Por esa carencia no se cita su AUC.
Detalle en `ARCHITECTURE.md` y `DATA_PROTOCOL_CARD.md`.

## Privacidad y seguridad

La imagen objetivo no requiere egress; **TODO(A4.3)** verificarlo por digest. El bundle no debe
contener audio, IDs, manifest del dataset, rutas locales, secretos ni artefactos por llamada. La
card publica solo agregados y hashes no reversibles.
