"""La página del Monitor, embebida como texto.

Va en un `.py` y no en un `.html` porque la imagen se construye con `pip install .` y
`setuptools.packages.find` solo empaqueta módulos: un `.html` suelto no llegaría al
contenedor y el monitor daría 404 justo en la mesa.

Sin dependencias externas a propósito: el contenedor no tiene salida a internet, así que
cualquier CDN dejaría la página en blanco. Todo el CSS y el JS van aquí.
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
     apagado de la familia de #cddade, para que el color signifique algo y no decore. */
  :root {
    --bg: #ffffff; --panel: #ffffff; --soft: #f7fafa; --line: #cddade;
    --text: #16191c; --dim: #6d7f87;
    --synthetic: #c10404; --synthetic-soft: #de473a; --human: #4d6f80; --warn: #de473a;
    --accent: #c10404;
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
  .live { background: var(--human); box-shadow: 0 0 0 3px rgba(74,217,145,.18); }
  .down { background: var(--synthetic); box-shadow: 0 0 0 3px rgba(255,122,107,.18); }
  .cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; }
  .card { background: var(--panel); border: 1px solid var(--line); border-radius: 10px; padding: 12px 14px; }
  .card .k { color: var(--dim); font-size: 11px; text-transform: uppercase; letter-spacing: .06em; }
  .card .v { font-size: 24px; font-weight: 600; font-family: var(--mono); margin-top: 4px; }
  .card .v small { font-size: 13px; color: var(--dim); font-weight: 400; }
  section { margin-top: 22px; }
  h2 { font-size: 12px; text-transform: uppercase; letter-spacing: .08em; color: var(--dim); margin: 0 0 8px; }
  .scroll { overflow-x: auto; border: 1px solid var(--line); border-radius: 10px; }
  table { border-collapse: collapse; width: 100%; min-width: 640px; background: var(--panel); }
  th, td { text-align: left; padding: 9px 12px; border-bottom: 1px solid var(--line); white-space: nowrap; }
  th { color: var(--dim); font-weight: 500; font-size: 11px; text-transform: uppercase; letter-spacing: .06em; }
  tbody tr:last-child td { border-bottom: 0; }
  td.num { font-family: var(--mono); text-align: right; }
  .tag { font-family: var(--mono); font-size: 12px; padding: 2px 8px; border-radius: 999px; font-weight: 600; }
  .tag.s { color: var(--synthetic); background: rgba(255,122,107,.12); }
  .tag.h { color: var(--human); background: rgba(74,217,145,.12); }
  .tag.e { color: var(--warn); background: rgba(255,200,87,.12); }
  .bar { height: 4px; border-radius: 2px; background: var(--line); overflow: hidden; min-width: 54px; }
  .bar > i { display: block; height: 100%; background: var(--accent); }
  .empty { padding: 26px 14px; color: var(--dim); text-align: center; }
  .note { color: var(--dim); font-size: 12px; margin-top: 10px; }
  code { font-family: var(--mono); color: var(--accent); }
  @media (max-width: 520px) { .card .v { font-size: 20px; } }
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>altur-detect</h1>
    <span class="sub" id="state"><span class="dot down"></span>conectando…</span>
    <span class="sub" id="bundle"></span>
  </header>

  <div class="cards">
    <div class="card"><div class="k">llamadas</div><div class="v" id="c-calls">—</div></div>
    <div class="card"><div class="k">sintéticas</div><div class="v" id="c-syn" style="color:var(--synthetic)">—</div></div>
    <div class="card"><div class="k">humanas</div><div class="v" id="c-hum" style="color:var(--human)">—</div></div>
    <div class="card"><div class="k">errores</div><div class="v" id="c-err">—</div></div>
    <div class="card"><div class="k">inferencia p50</div><div class="v" id="c-p50">—<small> ms</small></div></div>
    <div class="card"><div class="k">inferencia p95</div><div class="v" id="c-p95">—<small> ms</small></div></div>
    <div class="card"><div class="k">cuerpo mayor</div><div class="v" id="c-mb">—<small> MB</small></div></div>
  </div>

  <section>
    <h2>Últimas llamadas</h2>
    <div class="scroll">
      <table>
        <thead><tr>
          <th>hora</th><th>llamada</th><th>veredicto</th><th>confianza</th>
          <th style="text-align:right">audio</th><th style="text-align:right">cuerpo</th>
          <th style="text-align:right">inferencia</th>
        </tr></thead>
        <tbody id="rows"><tr><td colspan="7" class="empty">Sin llamadas todavía.</td></tr></tbody>
      </table>
    </div>
    <p class="note">
      <strong>inferencia</strong> es el trabajo del modelo (lo mismo que la cabecera
      <code>X-Inference-Ms</code>); no incluye la subida, que es lo que domina el tiempo
      que mide el cliente del juez. El umbral de decisión lo declara <code>/version</code>.
    </p>
  </section>
</div>

<script>
const $ = (id) => document.getElementById(id);
const pad = (n) => String(n).padStart(2, "0");
const hora = (ts) => { const d = new Date(ts * 1000);
  return pad(d.getHours()) + ":" + pad(d.getMinutes()) + ":" + pad(d.getSeconds()); };

function veredicto(c) {
  if (c.status !== 200) return '<span class="tag e">' + (c.error || c.status) + "</span>";
  if (c.is_synthetic === true) return '<span class="tag s">sintética</span>';
  if (c.is_synthetic === false) return '<span class="tag h">humana</span>';
  return "—";
}

function confianza(c) {
  if (typeof c.confidence !== "number") return "—";
  const pct = Math.round(c.confidence * 100);
  return '<div class="bar" title="' + c.confidence.toFixed(3) + '"><i style="width:' + pct + '%"></i></div>';
}

async function tick() {
  try {
    const r = await fetch("monitor/calls?limit=40", {cache: "no-store"});
    if (!r.ok) throw new Error(r.status);
    const d = await r.json();

    const ready = d.ready === true;
    $("state").innerHTML = '<span class="dot ' + (ready ? "live" : "down") + '"></span>' +
      (ready ? "listo" : "no listo");
    $("bundle").textContent = [d.detector, d.threshold != null ? "umbral " + d.threshold.toFixed(4) : null]
      .filter(Boolean).join(" · ");

    const s = d.summary || {};
    $("c-calls").textContent = s.calls ?? 0;
    $("c-syn").textContent = s.synthetic ?? 0;
    $("c-hum").textContent = s.human ?? 0;
    $("c-err").textContent = s.errors ?? 0;
    const ms = s.inference_ms || {};
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
      '<td class="num">' + (c.ms != null ? c.ms.toFixed(1) + " ms" : "—") + "</td>" +
      "</tr>").join("")
      : '<tr><td colspan="7" class="empty">Sin llamadas todavía.</td></tr>';
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
