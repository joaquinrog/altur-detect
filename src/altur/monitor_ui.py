"""Página autocontenida del Monitor de la mesa.

Vive en un módulo porque la imagen se construye con ``pip install .`` y no empaqueta
HTML suelto. No usa CDN ni recursos remotos: el contenedor no tiene salida a internet.
"""

from __future__ import annotations

PAGE = """<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>altur-detect · monitor</title>
<style>
  :root {
    --paper: #ffffff;
    --ink: #202426;
    --muted: #60747d;
    --human: #4d6f80;
    --structure: #cddade;
    --field: #f2f5f6;
    --synthetic: #c10404;
    --confidence: #e87822;
    --mono: ui-monospace, "SFMono-Regular", Consolas, "Liberation Mono", monospace;
    --sans: "Arial Narrow", "Aptos Narrow", "Helvetica Neue", Arial, sans-serif;
  }

  * { box-sizing: border-box; }
  html { background: var(--paper); color-scheme: light; }
  body {
    margin: 0;
    min-width: 280px;
    background: var(--paper);
    color: var(--ink);
    font-family: var(--sans);
    font-variant-numeric: tabular-nums lining-nums;
  }
  ::selection { color: var(--paper); background: var(--human); }
  :focus-visible { outline: 3px solid var(--confidence); outline-offset: 3px; }
  [hidden] { display: none !important; }

  .page { width: min(100%, 1600px); margin: 0 auto; padding: 28px clamp(20px, 4vw, 64px) 56px; }
  .masthead {
    min-height: 62px;
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 24px;
  }
  .lockup { display: flex; align-items: center; gap: 16px; }
  .mark {
    width: 42px;
    height: 42px;
    display: grid;
    place-items: center;
    border: 1px solid var(--ink);
    color: var(--ink);
    font: 700 13px/1 var(--mono);
    letter-spacing: -.08em;
  }
  .product { display: grid; gap: 2px; }
  .product strong { font-size: 15px; letter-spacing: -.02em; }
  .product span { color: var(--muted); font: 10px/1.2 var(--mono); letter-spacing: .08em; text-transform: uppercase; }
  .connection {
    display: flex;
    align-items: center;
    gap: 10px;
    min-height: 42px;
    color: var(--human);
    font: 700 10px/1 var(--mono);
    letter-spacing: .1em;
    text-transform: uppercase;
  }
  .connection-dot { width: 8px; height: 8px; border-radius: 50%; background: currentColor; }
  .connection[data-state="connecting"] .connection-dot { animation: pulse 1.2s ease-in-out infinite; }
  .connection[data-state="error"] { color: var(--muted); }

  .stage {
    min-height: calc(100svh - 122px);
    display: grid;
    grid-template-rows: auto 1fr auto auto;
    gap: 0;
    padding: clamp(24px, 4vw, 52px);
    background: var(--field);
    border-top: 1px solid var(--structure);
  }
  .stage-head { display: flex; align-items: baseline; justify-content: space-between; gap: 24px; }
  .stage-head strong { font-size: 13px; font-weight: 650; letter-spacing: -.01em; }
  .call-ref { color: var(--muted); font: 11px/1.3 var(--mono); text-align: right; }

  .notice { align-self: center; max-width: 620px; padding: 54px 0; }
  .notice strong { display: block; font-size: clamp(38px, 6vw, 78px); line-height: .96; letter-spacing: -.04em; }
  .notice p { max-width: 52ch; margin: 22px 0 0; color: var(--muted); font-size: 15px; }
  .notice code { color: var(--human); font-family: var(--mono); }

  .readout {
    min-width: 0;
    display: grid;
    grid-template-columns: minmax(0, 1.35fr) minmax(250px, .65fr);
    align-items: center;
    gap: clamp(44px, 6vw, 96px);
    padding: clamp(32px, 5vh, 64px) 0;
  }
  .verdict-block { min-width: 0; }
  .verdict-word-wrap { position: relative; overflow: hidden; padding: .08em 0 .12em; }
  .verdict-word {
    margin: 0;
    width: min-content;
    max-width: 100%;
    color: var(--human);
    font-size: clamp(64px, 10vw, 148px);
    font-weight: 700;
    line-height: .82;
    letter-spacing: -.04em;
    text-transform: uppercase;
  }
  .verdict-ghost { position: absolute; inset: .08em auto auto 0; pointer-events: none; }
  .confidence-line { display: flex; align-items: baseline; gap: clamp(12px, 2vw, 24px); margin-top: clamp(28px, 4vh, 48px); }
  .confidence-value {
    color: var(--confidence);
    font-size: clamp(52px, 6vw, 88px);
    font-weight: 650;
    line-height: .85;
    letter-spacing: -.04em;
    white-space: nowrap;
  }
  .confidence-copy { max-width: 150px; color: var(--muted); font-size: 14px; line-height: 1.2; }
  .digit-roll { position: relative; display: inline-grid; overflow: hidden; vertical-align: bottom; }
  .digit-roll > span { grid-area: 1 / 1; }
  .digit-roll .old { animation: digit-old 520ms cubic-bezier(.2,.8,.2,1) forwards; }
  .digit-roll .new { animation: digit-new 520ms cubic-bezier(.2,.8,.2,1) forwards; }

  .latency-block { min-width: 0; padding-left: clamp(30px, 4vw, 66px); border-left: 1px solid var(--structure); }
  .metric-title { display: block; color: var(--muted); font: 700 10px/1.2 var(--mono); letter-spacing: .12em; text-transform: uppercase; }
  .latency-value { display: block; margin-top: 22px; font-size: clamp(62px, 7.4vw, 112px); font-weight: 650; line-height: .82; letter-spacing: -.04em; white-space: nowrap; }
  .latency-value small { color: var(--muted); font-size: .26em; font-weight: 500; letter-spacing: 0; }
  .latency-note { margin: 20px 0 0; color: var(--muted); font-size: 13px; }

  .budget { padding: 22px 0 10px; border-top: 1px solid var(--structure); }
  .budget-head { display: flex; justify-content: space-between; gap: 20px; color: var(--muted); font: 700 10px/1 var(--mono); letter-spacing: .1em; text-transform: uppercase; }
  .sausage-chain { display: grid; grid-template-columns: repeat(10, minmax(0, 1fr)); gap: 8px; margin-top: 18px; }
  .sausage-link {
    position: relative;
    height: clamp(28px, 3.2vw, 42px);
    overflow: hidden;
    border: 1px solid #aebfc5;
    border-radius: 999px;
    background: rgba(255,255,255,.72);
  }
  .sausage-link:not(:last-child)::after {
    content: "";
    position: absolute;
    z-index: 2;
    top: 50%;
    right: -7px;
    width: 10px;
    height: 10px;
    border: 1px solid #aebfc5;
    background: var(--field);
    transform: translateY(-50%) rotate(45deg);
  }
  .sausage-fill { position: absolute; inset: 0; background: var(--human); transform: scaleX(0); transform-origin: left; }
  .motion .sausage-fill { transition: transform 520ms cubic-bezier(.16,1,.3,1); }
  .budget-marker { margin-top: 10px; color: var(--ink); font: 700 11px/1 var(--mono); }

  .evidence {
    display: flex;
    align-items: flex-end;
    justify-content: space-between;
    gap: 28px;
    margin-top: 20px;
    padding-top: 18px;
    border-top: 1px solid rgba(77,111,128,.18);
    color: var(--muted);
    font-size: 11px;
    line-height: 1.55;
  }
  .evidence strong { color: var(--ink); font-weight: 650; }
  .evidence-detail { max-width: 78ch; }
  .rtt { max-width: 36ch; text-align: right; }

  .history { margin-top: clamp(52px, 8vw, 104px); }
  .history-head { display: flex; justify-content: space-between; align-items: baseline; gap: 24px; margin-bottom: 14px; }
  .history h2 { margin: 0; font-size: 21px; letter-spacing: -.025em; }
  .history-counts { color: var(--muted); font: 11px/1.3 var(--mono); text-align: right; }
  .table-wrap { overflow-x: auto; border-top: 1px solid var(--ink); scrollbar-color: var(--human) var(--field); }
  table { width: 100%; min-width: 760px; border-collapse: collapse; font-size: 12px; }
  th, td { padding: 11px 10px; border-bottom: 1px solid var(--structure); text-align: left; white-space: nowrap; }
  th { color: var(--muted); font: 700 9px/1 var(--mono); letter-spacing: .09em; text-transform: uppercase; }
  td.num { font-family: var(--mono); text-align: right; }
  .result { font-weight: 750; text-transform: uppercase; }
  .result.synthetic { color: var(--synthetic); }
  .result.human { color: var(--human); }
  .result.error { color: var(--muted); }
  .history-empty { padding: 36px 10px; color: var(--muted); text-align: center; }

  @keyframes pulse { 50% { opacity: .28; transform: scale(.72); } }
  @keyframes verdict-in {
    from { opacity: 0; clip-path: inset(100% 0 0); transform: translateY(16px); }
    to { opacity: 1; clip-path: inset(0); transform: translateY(0); }
  }
  @keyframes verdict-out { to { opacity: 0; clip-path: inset(0 0 100%); transform: translateY(-12px); } }
  @keyframes digit-old { to { opacity: 0; transform: translateY(-72%); } }
  @keyframes digit-new { from { opacity: 0; transform: translateY(72%); } to { opacity: 1; transform: translateY(0); } }

  @media (max-width: 760px) {
    .page { padding: 20px 16px 42px; }
    .masthead { min-height: 70px; }
    .product span { display: none; }
    .stage { min-height: calc(100svh - 102px); padding: 24px 20px; }
    .stage-head { align-items: flex-start; }
    .readout { grid-template-columns: 1fr; align-content: center; gap: 40px; padding: 34px 0; }
    .verdict-word { font-size: clamp(48px, 16vw, 78px); }
    .confidence-line { margin-top: 24px; }
    .latency-block { display: flex; flex-wrap: wrap; align-items: baseline; gap: 10px 18px; padding: 24px 0 0; border-left: 0; border-top: 1px solid var(--structure); }
    .latency-block .metric-title { flex-basis: 100%; }
    .latency-value { margin-top: 4px; font-size: clamp(56px, 20vw, 82px); }
    .latency-note { margin: 0; }
    .sausage-chain { gap: 3px; }
    .sausage-link { height: 28px; }
    .evidence { align-items: flex-start; flex-direction: column; gap: 9px; }
    .rtt { text-align: left; }
  }

  @media (max-width: 390px) {
    .connection { max-width: 118px; justify-content: flex-end; text-align: right; }
    .call-ref { max-width: 46%; overflow-wrap: anywhere; }
    .confidence-copy { max-width: 124px; font-size: 12px; }
  }

  @media (prefers-reduced-motion: reduce) {
    *, *::before, *::after { scroll-behavior: auto !important; animation: none !important; transition: none !important; }
  }
</style>
</head>
<body>
<div class="page">
  <header class="masthead">
    <div class="lockup" aria-label="Chorizos Circuits, Altur voice authenticity">
      <span class="mark" aria-hidden="true">CC</span>
      <span class="product"><strong>altur-detect</strong><span>voice authenticity monitor</span></span>
    </div>
    <div class="connection" id="connection" data-state="connecting" role="status" aria-live="polite">
      <i class="connection-dot" aria-hidden="true"></i><span id="connection-copy">Conectando</span>
    </div>
  </header>

  <main class="stage" id="stage">
    <div class="stage-head">
      <strong>Última llamada</strong>
      <span class="call-ref" id="call-ref">Esperando datos</span>
    </div>

    <div class="notice" id="notice">
      <strong id="notice-title">Conectando al detector</strong>
      <p id="notice-copy">Consultando <code>/monitor/calls</code>. Esta pantalla se actualizará automáticamente.</p>
    </div>

    <section class="readout" id="readout" aria-label="Resultado de la última llamada" aria-live="polite" aria-atomic="true" hidden>
      <div class="verdict-block">
        <span class="metric-title">Veredicto reportado</span>
        <div class="verdict-word-wrap" id="verdict-wrap">
          <h1 class="verdict-word" id="verdict">—</h1>
        </div>
        <div class="confidence-line">
          <strong class="confidence-value" id="confidence-value">—</strong>
          <span class="confidence-copy">probabilidad calibrada de este veredicto</span>
        </div>
      </div>
      <div class="latency-block">
        <span class="metric-title">Latencia total del servidor</span>
        <strong class="latency-value"><span id="latency-value">—</span> <small>s</small></strong>
        <p class="latency-note">Máximo permitido: 30 segundos</p>
      </div>
    </section>

    <section class="budget" id="budget" aria-label="Latencia usada del presupuesto de 30 segundos" hidden>
      <div class="budget-head"><span>Tiempo de respuesta</span><span>30 segundos</span></div>
      <div class="sausage-chain" id="sausage-chain" aria-hidden="true">
        <span class="sausage-link"><i class="sausage-fill"></i></span>
        <span class="sausage-link"><i class="sausage-fill"></i></span>
        <span class="sausage-link"><i class="sausage-fill"></i></span>
        <span class="sausage-link"><i class="sausage-fill"></i></span>
        <span class="sausage-link"><i class="sausage-fill"></i></span>
        <span class="sausage-link"><i class="sausage-fill"></i></span>
        <span class="sausage-link"><i class="sausage-fill"></i></span>
        <span class="sausage-link"><i class="sausage-fill"></i></span>
        <span class="sausage-link"><i class="sausage-fill"></i></span>
        <span class="sausage-link"><i class="sausage-fill"></i></span>
      </div>
      <div class="budget-marker" id="budget-marker">— de 30 s</div>
    </section>

    <div class="evidence" id="evidence" hidden>
      <span class="evidence-detail" id="worst">Peor llamada observada: —</span>
      <span class="rtt">El total no incluye handshake ni viaje de vuelta (~2 RTT).</span>
    </div>
  </main>

  <section class="history" aria-labelledby="history-title">
    <div class="history-head">
      <h2 id="history-title">Llamadas recientes</h2>
      <span class="history-counts" id="history-counts">0 llamadas</span>
    </div>
    <div class="table-wrap" tabindex="0" aria-label="Historial de llamadas, desplazable horizontalmente">
      <table>
        <thead><tr>
          <th>Hora</th><th>Referencia</th><th>Veredicto</th><th style="text-align:right">Confianza</th>
          <th style="text-align:right">Total</th><th style="text-align:right">Subida</th>
          <th style="text-align:right">Modelo</th><th style="text-align:right">Cuerpo</th>
        </tr></thead>
        <tbody id="rows"><tr><td colspan="8" class="history-empty">Sin llamadas todavía.</td></tr></tbody>
      </table>
    </div>
  </section>
</div>

<script>
const $ = (id) => document.getElementById(id);
const BUDGET_MS = 30000;
const K = new URLSearchParams(location.search).get("k");
const CALLS_URL = "monitor/calls?limit=40" + (K ? "&k=" + encodeURIComponent(K) : "");
const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
let lastCallKey = null;
let initialLoad = true;
let pollInFlight = false;
let currentConfidence = "—";
let currentLatency = null;

const pad = (n) => String(n).padStart(2, "0");
const hora = (ts) => {
  const d = new Date(ts * 1000);
  return Number.isFinite(d.getTime()) ? pad(d.getHours()) + ":" + pad(d.getMinutes()) + ":" + pad(d.getSeconds()) : "—";
};
const total = (c) => c.server_ms != null ? c.server_ms : (c.upload_ms || 0) + (c.ms || 0);
const decode = (c) => c.server_ms != null ? Math.max(0, c.server_ms - (c.upload_ms || 0) - (c.ms || 0)) : 0;
const seconds = (ms) => (ms / 1000).toFixed(2);
const esc = (value) => String(value).replace(/[&<>"']/g, (ch) => ({
  "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
})[ch]);

function setConnection(kind, text) {
  $("connection").dataset.state = kind;
  $("connection-copy").textContent = text;
}

function showNotice(kind) {
  $("readout").hidden = true;
  $("budget").hidden = true;
  $("evidence").hidden = true;
  $("notice").hidden = false;
  if (kind === "empty") {
    $("notice-title").textContent = "Esperando la primera llamada";
    $("notice-copy").textContent = "Cuando llegue una llamada, aquí aparecerán su veredicto, confianza y latencia.";
    $("call-ref").textContent = "Sin llamadas";
  } else if (kind === "error") {
    $("notice-title").textContent = "No pudimos actualizar el monitor";
    $("notice-copy").textContent = "La conexión se reintentará automáticamente en un segundo.";
    $("call-ref").textContent = "Sin respuesta";
  }
}

function verdictData(c) {
  if (c.status !== 200) return {text: "Error", color: "var(--human)", cls: "error"};
  if (c.is_synthetic === true) return {text: "Sintética", color: "var(--synthetic)", cls: "synthetic"};
  if (c.is_synthetic === false) return {text: "Humana", color: "var(--human)", cls: "human"};
  return {text: "Sin veredicto", color: "var(--human)", cls: "error"};
}

function setVerdict(next, animate) {
  const el = $("verdict");
  const wrap = $("verdict-wrap");
  if (animate && el.textContent !== "—") {
    const ghost = el.cloneNode(true);
    ghost.removeAttribute("id");
    ghost.classList.add("verdict-ghost");
    wrap.appendChild(ghost);
    ghost.style.animation = "verdict-out 520ms cubic-bezier(.2,.8,.2,1) forwards";
    window.setTimeout(() => ghost.remove(), 540);
  }
  el.textContent = next.text;
  el.style.color = next.color;
  if (animate) {
    el.style.animation = "none";
    void el.offsetWidth;
    el.style.animation = "verdict-in 520ms cubic-bezier(.16,1,.3,1) both";
  } else {
    el.style.animation = "none";
  }
}

function setConfidence(value, animate) {
  const next = typeof value === "number" ? Math.round(value * 100) + "%" : "—";
  const el = $("confidence-value");
  el.setAttribute("aria-label", next === "—" ? "Probabilidad calibrada no disponible" : next + " de probabilidad calibrada para este veredicto");
  if (!animate || currentConfidence === "—" || next === "—") {
    el.textContent = next;
    currentConfidence = next;
    return;
  }
  const length = Math.max(currentConfidence.length, next.length);
  el.innerHTML = Array.from({length}, (_, i) => {
    const oldChar = currentConfidence.padStart(length)[i];
    const newChar = next.padStart(length)[i];
    if (oldChar === newChar) return esc(newChar);
    return '<span class="digit-roll"><span class="old">' + esc(oldChar) +
      '</span><span class="new">' + esc(newChar) + "</span></span>";
  }).join("");
  currentConfidence = next;
  window.setTimeout(() => { el.textContent = next; }, 540);
}

function setLatency(ms, animate) {
  const el = $("latency-value");
  if (!(ms > 0)) {
    el.textContent = "—";
    currentLatency = null;
    return;
  }
  if (!animate || currentLatency == null) {
    el.textContent = seconds(ms);
    currentLatency = ms;
    return;
  }
  const start = currentLatency;
  const started = performance.now();
  const frame = (now) => {
    const p = Math.min(1, (now - started) / 520);
    const eased = 1 - Math.pow(1 - p, 4);
    el.textContent = seconds(start + (ms - start) * eased);
    if (p < 1) requestAnimationFrame(frame);
  };
  requestAnimationFrame(frame);
  currentLatency = ms;
}

function setSausage(ms, animate) {
  document.documentElement.classList.toggle("motion", animate);
  const links = document.querySelectorAll(".sausage-fill");
  const used = Math.max(0, Math.min(10, (ms / BUDGET_MS) * 10));
  links.forEach((link, i) => { link.style.transform = "scaleX(" + Math.max(0, Math.min(1, used - i)) + ")"; });
  $("budget-marker").textContent = ms > 0 ? seconds(ms) + " s de 30 s" : "Latencia no disponible";
  window.setTimeout(() => document.documentElement.classList.remove("motion"), 560);
}

function renderLatest(c, animate) {
  const motion = animate && !reducedMotion.matches;
  const verdict = verdictData(c);
  const latency = total(c);
  $("notice").hidden = true;
  $("readout").hidden = false;
  $("budget").hidden = false;
  $("evidence").hidden = false;
  $("call-ref").textContent = (c.ref || "Sin referencia") + " · " + hora(c.ts);
  setVerdict(verdict, motion);
  setConfidence(c.status === 200 ? c.confidence : null, motion);
  setLatency(latency, motion);
  setSausage(latency || 0, motion);
}

function renderSummary(summary) {
  const s = summary || {};
  const w = s.worst;
  $("history-counts").textContent = (s.calls ?? 0) + " llamadas · " + (s.ok ?? 0) +
    " contestadas · " + (s.errors ?? 0) + " errores";
  $("worst").innerHTML = w
    ? "<strong>Peor llamada observada: " + seconds(w.total_ms) + " s</strong> · subida " +
      seconds(w.upload_ms) + " s · decodificación " + w.decode_ms.toFixed(0) +
      " ms · modelo " + w.inference_ms.toFixed(0) + " ms" +
      (s.worst_headroom_x != null ? " · " + esc(s.worst_headroom_x) + "× de margen" : "")
    : "Peor llamada observada: —";
}

function renderRows(calls) {
  $("rows").innerHTML = calls.length ? calls.map((c) => {
    const verdict = verdictData(c);
    const confidence = c.status === 200 && typeof c.confidence === "number"
      ? Math.round(c.confidence * 100) + "%" : "—";
    const latency = total(c);
    return "<tr><td>" + hora(c.ts) + "</td><td>" + esc(c.ref || "—") +
      '</td><td><span class="result ' + verdict.cls + '">' + esc(verdict.text) +
      '</span></td><td class="num">' + confidence + '</td><td class="num">' +
      (latency > 0 ? seconds(latency) + " s" : "—") + '</td><td class="num">' +
      (c.upload_ms != null ? seconds(c.upload_ms) + " s" : "—") + '</td><td class="num">' +
      (c.ms != null ? Number(c.ms).toFixed(1) + " ms" : "—") + '</td><td class="num">' +
      (c.mb != null ? Number(c.mb).toFixed(2) + " MB" : "—") + "</td></tr>";
  }).join("") : '<tr><td colspan="8" class="history-empty">Sin llamadas todavía.</td></tr>';
}

async function tick() {
  if (pollInFlight) return;
  pollInFlight = true;
  try {
    const response = await fetch(CALLS_URL, {cache: "no-store"});
    if (!response.ok) throw new Error(response.status);
    const data = await response.json();
    const ready = data.ready === true;
    const calls = data.calls || [];
    setConnection(ready ? "ready" : "error", ready ? "Detector listo" : "Detector no listo");
    renderSummary(data.summary);
    renderRows(calls);

    if (!calls.length) {
      lastCallKey = null;
      showNotice("empty");
    } else {
      const c = calls[0];
      const key = JSON.stringify([c.ts, c.ref, c.status, c.is_synthetic, c.confidence, c.server_ms, c.ms, c.upload_ms]);
      if (key !== lastCallKey) {
        renderLatest(c, !initialLoad);
        lastCallKey = key;
      } else if ($("readout").hidden) {
        // Una reconexión restaura el dato existente sin fingir que llegó otra llamada.
        renderLatest(c, false);
      }
    }
    initialLoad = false;
  } catch (_error) {
    setConnection("error", "Sin respuesta");
    showNotice("error");
  } finally {
    pollInFlight = false;
  }
}

tick();
setInterval(tick, 1000);
</script>
</body>
</html>
"""
