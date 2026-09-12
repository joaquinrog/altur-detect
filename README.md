# altur-detect

**¿La voz que llama al banco es una persona o un sistema?**
HackMTY 2026 · track Altur · equipo Chorizos Circuits HC

`POST /detect` recibe una llamada telefónica en estéreo a 8 kHz —canal 0 quien llama,
canal 1 el agente del banco— y responde si **la voz del canal 0** es sintética, con una
confianza calibrada.

```json
{"is_synthetic": true, "confidence": 0.87}
```

---

## El enfoque en un párrafo

Altur nos dijo qué define la etiqueta: **solo importa si la voz es de IA o humana**, sin
importar si es una grabación. Eso decide la arquitectura. En este dataset lo conductual
—la latencia entre turnos— resuelve casi todo (VAL AUC 0.9968), pero funciona por
correlación, no por definición: el mismo proceso generó la voz TTS y la latencia del
pipeline. Un caller de IA sub-segundo, o una grabación humana reproducida, rompen ese
proxy. Por eso el backbone es **acústico**, lo conductual es un **refuerzo**, y cuando los
dos discrepan **manda el acústico y baja la confianza**.

La pieza central no es un modelo: es el **arnés** que permite cambiar de modelo. Extractores,
modelos, fusiones, calibradores y transformaciones son piezas registradas e intercambiables
que corren bajo un protocolo estadístico común. Añadir un enfoque es un archivo y un YAML.

## Por qué el arnés y no una receta

Con 353 llamadas, un detector puede sacar AUC 0.99 midiendo el códec en vez de la voz.
No es hipotético — es el modo de fallo documentado del campo: un detector con **0.22 % de
EER en su dominio cae a 38.57 % fuera de él**. El arnés existe para detectar eso antes que
un juez:

- **Cross-fitting anidado.** Modelos, fusión, calibrador y umbral se ajustan *dentro* de
  cada fold externo. Nada aprendido toca el fold que va a evaluar.
- **Grupos por componentes conectados.** Hablante, voz, llamada donante y derivados caen
  siempre en la misma partición.
- **Controles negativos.** El canal 1 es el mismo TTS en ambas clases: cualquier feature
  acústica que separe ahí mide plomería, no voz. El control **silence-only** prueba lo
  mismo sobre las regiones sin habla.
- **Gauntlet con corrupción realista.** La compresión daña más que el ruido, y daña aunque
  el audio suene igual.
- **`val` bajo llave.** Toda iteración se mide out-of-fold sobre `train`; cada consulta a
  `val` queda registrada en `experiments/val_looks.csv`.

## Desplegable, no un notebook

El contenedor es autocontenido: **sin GPU, sin dependencias nativas de audio, sin egress**,
con el modelo dentro de la imagen. Corre dentro del perímetro del banco — el audio de las
llamadas no sale a ningún lado.

```bash
make setup && make test      # 50 tests de contrato
make serve                   # http://127.0.0.1:8000
make smoke                   # golpea un servidor vivo con casos válidos e inválidos
make docker && make bundle-test
```

| Endpoint | Para qué |
|---|---|
| `POST /detect` | El veredicto |
| `GET /health` · `GET /health/ready` | Liveness y readiness, separados |
| `GET /version` | Commit, hash del bundle, protocolo y limitaciones declaradas |

## Estructura

```
src/altur/
  types.py       AudioExample · DatasetRecord   ← el candado anti-fuga
  registry.py    extractores · modelos · fusiones · transformaciones
  protocol.py    componentes conectados · cross-fitting anidado
  battery.py     gauntlet · controles · ablación · truncación
  api.py         POST /detect
configs/         protocolos, gauntlet, perturbaciones, experimentos
experiments/     un archivo por corrida + el registro de miradas a val
```

## Contribuir

Tres reglas que el arnés da por sentadas:

1. `types.py`, `registry.py`, `protocol.py` y `configs/protocol/` son contratos: no se
   editan en paralelo. Cada quien trabaja en su rama y un solo integrador hace merge.
2. Un extractor recibe `AudioExample` y nada más — sin id, etiqueta, split ni procedencia.
   Si necesitas metadatos para calcular una feature, la feature está mal planteada.
3. `val` no se mira para iterar. Todo se decide con out-of-fold sobre `train`, y cada
   consulta a `val` queda registrada.

## Honestidad

Los números acústicos vienen de una muestra de 60 llamadas y se están confirmando sobre las
353. `val` tiene 71 llamadas: diferencias menores a 0.02 AUC no son señal. Y `val` comparte
pipeline de TTS con `train`, así que **no prueba generalización a otro proveedor de voz** —
para eso construimos un corpus propio, cuyos límites se declaran donde se reportan sus
resultados.

## Datos

El dataset es de Altur, solo para HackMTY 2026, y **no se redistribuye**. Ni el audio ni
`manifest.csv` están en este repo. Las figuras son agregadas: nunca una llamada individual.

## Licencia

Pendiente — se fija tras auditar las licencias de las dependencias.
