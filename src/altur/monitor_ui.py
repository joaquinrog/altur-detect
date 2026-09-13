"""La página del Monitor, embebida como texto.

Va en un `.py` y no en un `.html` porque la imagen se construye con `pip install .` y
`setuptools.packages.find` solo empaqueta módulos: un `.html` suelto no llegaría al
contenedor y el monitor daría 404 justo en la mesa.

Sin dependencias externas a propósito: el contenedor no tiene salida a internet, así que
cualquier CDN dejaría la página en blanco. Todo el CSS y el JS van aquí.

Lo que la página pone al centro es el **presupuesto de 30 s por llamada**, no el conteo:
"Latency" es uno de los cinco criterios con los que Altur califica, y el juez lo está
midiendo mientras esto se ve en pantalla.
"""

from __future__ import annotations

PAGE = """<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>altur-detect · monitor</title>
<style>
  /* Identidad del equipo: blanco principal, rojos de marca, #cddade para estructura.
     El rojo se reserva para "sintética", que es la alerta; lo humano va en el azul
     apagado de la familia de #cddade, para que el color signifique algo y no decore.
     En esta paleta no hay verde: "listo" también usa el azul apagado. */
  :root {
    --bg: #ffffff; --panel: #ffffff; --soft: #f7fafa; --line: #cddade;
    --text: #16191c; --dim: #6d7f87;
    --synthetic: #c10404; --synthetic-soft: #de473a; --human: #4d6f80; --warn: #de473a;
    --accent: #c10404;
    --human-wash: rgba(77,111,128,.12); --synthetic-wash: rgba(193,4,4,.10);
    --warn-wash: rgba(222,71,58,.12);
    --mono: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; background: var(--bg); color: var(--text); font: 14px/1.5 system-ui, sans-serif;
    padding: 20px 16px 40px;
  }
  .wrap { max-width: 1100px; margin: 0 auto; }
  header { display: flex; flex-wrap: wrap; gap: 12px; align-items: baseline; margin-bottom: 18px; }
  h1 { font-size: 18px; margin: 0; font-weight: 650; letter-spacing: -0.01em; }
  .sub { color: var(--dim); font-family: var(--mono); font-size: 12px; }
  .dot { width: 9px; height: 9px; border-radius: 50%; display: inline-block; margin-right: 6px; }
  .live { background: var(--human); box-shadow: 0 0 0 3px var(--human-wash); }
  .down { background: var(--synthetic); box-shadow: 0 0 0 3px var(--synthetic-wash); }

  /* El presupuesto: lo primero que se ve. */
  .budget { border: 1px solid var(--line); border-radius: 12px; padding: 16px 18px; background: var(--soft); }
  .budget .k { color: var(--dim); font-size: 11px; text-transform: uppercase; letter-spacing: .06em; }
  .budget .headline { display: flex; flex-wrap: wrap; align-items: baseline; gap: 10px; margin: 6px 0 12px; }
  .budget .big { font-size: 34px; font-weight: 650; font-family: var(--mono); letter-spacing: -0.02em; }
  .budget .of { color: var(--dim); font-family: var(--mono); font-size: 15px; }
  .budget .x { margin-left: auto; font-family: var(--mono); font-size: 15px; color: var(--human); font-weight: 600; }
  .track { height: 14px; border-radius: 7px; background: var(--bg); border: 1px solid var(--line);
           overflow: hidden; display: flex; }
  .track > i { display: block; height: 100%; }
  .track > i.up { background: var(--human); }
  .track > i.dec { background: var(--line); }
  .track > i.inf { background: var(--accent); }
  .legend { display: flex; flex-wrap: wrap; gap: 14px; margin-top: 9px; color: var(--dim); font-size: 12px; }
  .legend b { font-weight: 600; font-family: var(--mono); color: var(--text); }
  .swatch { width: 9px; height: 9px; border-radius: 2px; display: inline-block; margin-right: 5px; }

  .cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 10px; margin-top: 14px; }
  .card { background: var(--panel); border: 1px solid var(--line); border-radius: 10px; padding: 12px 14px; }
  .card .k { color: var(--dim); font-size: 11px; text-transform: uppercase; letter-spacing: .06em; }
  .card .v { font-size: 24px; font-weight: 600; font-family: var(--mono); margin-top: 4px; }
  .card .v small { font-size: 13px; color: var(--dim); font-weight: 400; }
  section { margin-top: 22px; }
  h2 { font-size: 12px; text-transform: uppercase; letter-spacing: .08em; color: var(--dim); margin: 0 0 8px; }
  .scroll { overflow-x: auto; border: 1px solid var(--line); border-radius: 10px; }
  table { border-collapse: collapse; width: 100%; min-width: 780px; background: var(--panel); }
  th, td { text-align: left; padding: 9px 12px; border-bottom: 1px solid var(--line); white-space: nowrap; }
  th { color: var(--dim); font-weight: 500; font-size: 11px; text-transform: uppercase; letter-spacing: .06em; }
  tbody tr:last-child td { border-bottom: 0; }
  td.num { font-family: var(--mono); text-align: right; }
  .tag { font-family: var(--mono); font-size: 12px; padding: 2px 8px; border-radius: 999px; font-weight: 600; }
  .tag.s { color: var(--synthetic); background: var(--synthetic-wash); }
  .tag.h { color: var(--human); background: var(--human-wash); }
  .tag.e { color: var(--warn); background: var(--warn-wash); }
  .bar { height: 4px; border-radius: 2px; background: var(--line); overflow: hidden; min-width: 54px; }
  .bar > i { display: block; height: 100%; background: var(--accent); }
  .mini { height: 6px; border-radius: 3px; background: var(--bg); border: 1px solid var(--line);
          overflow: hidden; display: flex; min-width: 90px; }
  .mini > i { display: block; height: 100%; }
  .empty { padding: 26px 14px; color: var(--dim); text-align: center; }
  .note { color: var(--dim); font-size: 12px; margin-top: 10px; max-width: 78ch; }
  code { font-family: var(--mono); color: var(--accent); }
  @media (max-width: 520px) { .card .v { font-size: 20px; } .budget .big { font-size: 27px; } }
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>altur-detect</h1>
    <span class="sub" id="state"><span class="dot down"></span>conectando…</span>
    <span class="sub" id="bundle"></span>
  </header>

  <div class="budget">
    <div class="k">peor llamada contra el presupuesto del juez</div>
    <div class="headline">
      <span class="big" id="b-worst">—</span>
      <span class="of">de 30 s</span>
      <span class="x" id="b-x"></span>
    </div>
    <div class="track" id="b-track"></div>
    <div class="legend">
      <span><span class="swatch" style="background:var(--human)"></span>subida <b id="b-up">—</b></span>
      <span><span class="swatch" style="background:var(--line)"></span>decodificación <b id="b-dec">—</b></span>
      <span><span class="swatch" style="background:var(--accent)"></span>modelo <b id="b-inf">—</b></span>
      <span>contestadas <b id="b-ok">—</b></span>
      <span>errores <b id="b-err">—</b></span>
    </div>
  </div>

  <div class="cards">
    <div class="card"><div class="k">llamadas</div><div class="v" id="c-calls">—</div></div>
    <div class="card"><div class="k">sintéticas</div><div class="v" id="c-syn" style="color:var(--synthetic)">—</div></div>
    <div class="card"><div class="k">humanas</div><div class="v" id="c-hum" style="color:var(--human)">—</div></div>
    <div class="card"><div class="k">subida p50</div><div class="v" id="c-up50">—<small> s</small></div></div>
    <div class="card"><div class="k">modelo p50</div><div class="v" id="c-p50">—<small> ms</small></div></div>
    <div class="card"><div class="k">modelo p95</div><div class="v" id="c-p95">—<small> ms</small></div></div>
    <div class="card"><div class="k">cuerpo mayor</div><div class="v" id="c-mb">—<small> MB</small></div></div>
  </div>

  <section>
    <h2>Últimas llamadas</h2>
    <div class="scroll">
      <table>
        <thead><tr>
          <th>hora</th><th>llamada</th><th>veredicto</th><th>confianza</th>
          <th style="text-align:right">audio</th><th style="text-align:right">cuerpo</th>
          <th style="text-align:right">subida</th><th style="text-align:right">modelo</th>
          <th>de 30 s</th>
        </tr></thead>
        <tbody id="rows"><tr><td colspan="9" class="empty">Sin llamadas todavía.</td></tr></tbody>
      </table>
    </div>
    <p class="note">
      La llamada se parte en tres: <strong>subida</strong> (el cliente mandando el cuerpo,
      cronometrado en el servidor), <strong>decodificación</strong> (el JSON, el base64 y
      el WAV) y <strong>modelo</strong> (el detector, lo mismo que la cabecera
      <code>X-Inference-Ms</code>). La barra <em>de 30 s</em> es la suma de las tres.
      No incluye el handshake ni el viaje de vuelta (~2 RTT), así que queda un poco por
      debajo de lo que imprime <code>check_endpoint.py</code>. El umbral de decisión lo
      declara <code>/version</code>, junto con las limitaciones del bundle.
    </p>
  </section>
</div>

<script>
const $ = (id) => document.getElementById(id);
const pad = (n) => String(n).padStart(2, "0");
const hora = (ts) => { const d = new Date(ts * 1000);
  return pad(d.getHours()) + ":" + pad(d.getMinutes()) + ":" + pad(d.getSeconds()); };
const BUDGET_MS = 30000;

// El token viaja en la URL de la página; hay que reenviarlo en cada consulta.
const K = new URLSearchParams(location.search).get("k");
const CALLS_URL = "monitor/calls?limit=40" + (K ? "&k=" + encodeURIComponent(K) : "");

const segs = (ms) => (ms / 1000).toFixed(2) + " s";
// El total es el trabajo entero del servidor, no subida+modelo: entre esos dos está la
// decodificación del JSON/base64/WAV, que sobre 6 MB son decenas de ms.
const total = (c) => c.server_ms != null ? c.server_ms : (c.upload_ms || 0) + (c.ms || 0);
const decode = (c) => c.server_ms != null
  ? Math.max(0, c.server_ms - (c.upload_ms || 0) - (c.ms || 0)) : 0;
const pct = (ms) => Math.max(ms > 0 ? 0.4 : 0, Math.min(100, (ms / BUDGET_MS) * 100));

function veredicto(c) {
  if (c.status !== 200) return '<span class="tag e">' + (c.error || c.status) + "</span>";
  if (c.is_synthetic === true) return '<span class="tag s">sintética</span>';
  if (c.is_synthetic === false) return '<span class="tag h">humana</span>';
  return "—";
}

function confianza(c) {
  if (typeof c.confidence !== "number") return "—";
  const p = Math.round(c.confidence * 100);
  // La barra se tiñe del color del veredicto. Si no, una llamada humana saldría con una
  // barra roja llena y el rojo dejaría de significar "sintética" (notes/20).
  const col = c.is_synthetic === true ? "var(--synthetic)"
            : c.is_synthetic === false ? "var(--human)" : "var(--dim)";
  return '<div class="bar" title="' + c.confidence.toFixed(3) + '"><i style="width:' + p +
    "%;background:" + col + '"></i></div>';
}

function presupuesto(c) {
  const t = total(c);
  if (!t) return "—";
  return '<div class="mini" title="' + segs(t) + ' de 30 s">' +
    '<i style="width:' + pct(c.upload_ms || 0) + '%;background:var(--human)"></i>' +
    '<i style="width:' + pct(decode(c)) + '%;background:var(--line)"></i>' +
    '<i style="width:' + pct(c.ms || 0) + '%;background:var(--accent)"></i></div>';
}

async function tick() {
  try {
    const r = await fetch(CALLS_URL, {cache: "no-store"});
    if (!r.ok) throw new Error(r.status);
    const d = await r.json();

    const ready = d.ready === true;
    $("state").innerHTML = '<span class="dot ' + (ready ? "live" : "down") + '"></span>' +
      (ready ? "listo" : "no listo");
    $("bundle").textContent = [d.detector, d.threshold != null ? "umbral " + d.threshold.toFixed(4) : null]
      .filter(Boolean).join(" · ");

    const s = d.summary || {};
    const up = s.upload_ms || {}, ms = s.inference_ms || {};
    // El desglose es el de LA peor llamada. No se usan los máximos por etapa: salen de
    // llamadas distintas y no sumarían el titular.
    const w = s.worst;

    $("b-worst").textContent = w ? segs(w.total_ms) : "—";
    $("b-x").textContent = s.worst_headroom_x != null ? s.worst_headroom_x + "× de margen" : "";
    $("b-track").innerHTML = w
      ? '<i class="up" style="width:' + pct(w.upload_ms) + '%"></i>' +
        '<i class="dec" style="width:' + pct(w.decode_ms) + '%"></i>' +
        '<i class="inf" style="width:' + pct(w.inference_ms) + '%"></i>'
      : "";
    $("b-up").textContent = w ? segs(w.upload_ms) : "—";
    $("b-dec").textContent = w ? w.decode_ms.toFixed(0) + " ms" : "—";
    $("b-inf").textContent = w ? w.inference_ms.toFixed(0) + " ms" : "—";
    $("b-ok").textContent = s.ok ?? 0;
    $("b-err").textContent = s.errors ?? 0;

    $("c-calls").textContent = s.calls ?? 0;
    $("c-syn").textContent = s.synthetic ?? 0;
    $("c-hum").textContent = s.human ?? 0;
    $("c-up50").innerHTML = (up.p50 != null ? (up.p50 / 1000).toFixed(2) : "—") + "<small> s</small>";
    $("c-p50").innerHTML = (ms.p50 ?? "—") + "<small> ms</small>";
    $("c-p95").innerHTML = (ms.p95 ?? "—") + "<small> ms</small>";
    $("c-mb").innerHTML = (s.biggest_mb ?? "—") + "<small> MB</small>";

    const calls = d.calls || [];
    $("rows").innerHTML = calls.length ? calls.map((c) =>
      "<tr>" +
      "<td>" + hora(c.ts) + "</td>" +
      '<td class="num">' + (c.ref || "—") + "</td>" +
      "<td>" + veredicto(c) + "</td>" +
      "<td>" + confianza(c) + "</td>" +
      '<td class="num">' + (c.duration_s != null ? c.duration_s.toFixed(0) + " s" : "—") + "</td>" +
      '<td class="num">' + (c.mb != null ? c.mb.toFixed(2) + " MB" : "—") + "</td>" +
      '<td class="num">' + (c.upload_ms != null ? segs(c.upload_ms) : "—") + "</td>" +
      '<td class="num">' + (c.ms != null ? c.ms.toFixed(1) + " ms" : "—") + "</td>" +
      "<td>" + presupuesto(c) + "</td>" +
      "</tr>").join("")
      : '<tr><td colspan="9" class="empty">Sin llamadas todavía.</td></tr>';
  } catch (e) {
    $("state").innerHTML = '<span class="dot down"></span>sin respuesta';
  }
}

tick();
setInterval(tick, 1000);
</script>
</body>
</html>
"""
