"""Página autocontenida del Monitor de la mesa.

Vive en un módulo porque la imagen se construye con ``pip install .`` y no empaqueta
HTML suelto. No usa CDN ni recursos remotos: el contenedor no tiene salida a internet,
así que la fuente y los logos llegan como ``data:`` desde ``monitor_assets``.

Diseño D5 (sesión 16): la presentación del equipo es la referencia visual. Rojo = solo
"sintética", naranja = solo "confianza", coral = marca y "pasó el límite". La única
animación continua es el latido del punto LIVE; lo demás se mueve solo cuando llega una
llamada nueva, y nada se mueve con ``prefers-reduced-motion``.
"""

from __future__ import annotations

from . import monitor_assets as _assets

_TEMPLATE = """<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>altur-detect · monitor</title>
<style>
  @font-face {
    font-family: "Urbanist";
    src: url(data:font/woff2;base64,__FONT__) format("woff2");
    font-weight: 100 900;
    font-display: block;
  }
  :root {
    --paper: #ffffff;
    --ink: #323232;
    --ink-strong: #1a1a1a;
    --muted: #5b6b72;
    --faint: #7d8d94;
    --structure: #cddade;
    --synthetic: #c10404;
    --brand: #de473a;
    --human: #4d6f80;
    --confidence: #e87822;
    --ease: cubic-bezier(.32,.72,0,1);
    --bounce: cubic-bezier(.34,1.56,.64,1);
  }
  * { box-sizing: border-box; }
  [hidden] { display: none !important; }
  html, body { background: var(--paper); color-scheme: light; }
  body {
    margin: 0;
    min-height: 100dvh;
    overflow-x: hidden;
    color: var(--ink);
    font-family: "Urbanist", system-ui, sans-serif;
    -webkit-font-smoothing: antialiased;
  }
  :focus-visible { outline: 3px solid var(--confidence); outline-offset: 3px; }

  /* Atmósfera de la presentación: detrás del panel, nunca encima de un dato. */
  .blob {
    position: fixed; z-index: 0; pointer-events: none;
    top: -22vmax; right: -18vmax; width: 62vmax; height: 62vmax; border-radius: 50%;
    background:
      radial-gradient(closest-side at 38% 58%, rgba(222,71,58,.34), transparent 70%),
      radial-gradient(closest-side at 62% 40%, rgba(214,120,190,.26), transparent 72%),
      radial-gradient(closest-side at 50% 50%, rgba(190,160,230,.20), transparent 78%);
    filter: blur(38px);
  }
  .grain {
    position: fixed; inset: 0; z-index: 0; pointer-events: none; opacity: .55;
    background-image: url(data:image/png;base64,__GRAIN__); mix-blend-mode: multiply;
    -webkit-mask-image: radial-gradient(60vmax 60vmax at 100% 0%, #000 0%, transparent 70%);
    mask-image: radial-gradient(60vmax 60vmax at 100% 0%, #000 0%, transparent 70%);
  }
  .page {
    position: relative; z-index: 1; width: min(100%, 1480px); margin: 0 auto;
    padding: 26px clamp(16px, 4vw, 56px) 64px;
  }

  .mast { display: flex; align-items: center; justify-content: space-between; gap: 16px; margin-bottom: 22px; }
  .lockup { display: flex; align-items: center; gap: 14px; min-width: 0; }
  .lockup img { display: block; width: auto; }
  .logo-altur { height: 30px; }
  .logo-chori { height: 34px; }
  .lockup .x { font-size: 28px; font-weight: 500; line-height: 1; color: var(--ink-strong); }

  /* LIVE como en la lámina: punto separado y píldora redondeada a la derecha. */
  .live { display: inline-flex; align-items: center; gap: 9px; flex: none; }
  .live-dot { width: 13px; height: 13px; border-radius: 50%; background: var(--brand); }
  .live-pill {
    display: inline-flex; align-items: center; height: 34px; padding: 0 18px 0 14px;
    border-radius: 4px 99px 99px 4px; background: var(--ink); color: #fff;
    font-weight: 800; font-size: 17px; letter-spacing: .02em; white-space: nowrap;
  }
  .live[data-state="live"] .live-dot { animation: live 2s ease-in-out infinite; }
  .live[data-state="connecting"] .live-dot, .live[data-state="off"] .live-dot { background: var(--faint); }
  .live[data-state="connecting"] .live-pill, .live[data-state="off"] .live-pill {
    background: #e4eaec; color: var(--muted); font-weight: 700; letter-spacing: 0;
  }

  .shell { padding: 8px; border-radius: 34px; background: rgba(238,243,244,.88); box-shadow: inset 0 0 0 1px rgba(77,111,128,.10); }
  .core {
    position: relative; overflow: hidden; border-radius: 28px; background: #fff;
    padding: clamp(24px, 3.6vw, 52px);
    box-shadow: 0 0 0 1px rgba(26,26,26,.04), 0 30px 60px -38px rgba(35,60,72,.35), 0 12px 24px -18px rgba(35,60,72,.18);
  }
  .core-head { display: flex; justify-content: space-between; align-items: baseline; gap: 16px; font-size: 16px; color: var(--muted); }
  .core-head strong { color: var(--ink-strong); font-weight: 700; }

  .notice { padding: clamp(40px, 9vh, 110px) 0 clamp(30px, 6vh, 70px); }
  .notice h1 { margin: 0; font-size: clamp(44px, 6vw, 92px); line-height: 1; font-weight: 700; letter-spacing: -.025em; color: var(--ink-strong); }
  .notice p { margin: 20px 0 0; font-size: 20px; color: var(--muted); max-width: 52ch; }

  .readout {
    display: grid; grid-template-columns: minmax(0, 1.45fr) minmax(290px, .8fr);
    gap: clamp(28px, 5vw, 80px); align-items: end; margin-top: clamp(24px, 4vh, 52px);
  }
  .k { display: block; font-size: 17px; font-weight: 600; color: var(--muted); margin-bottom: 12px; }

  /* El espacio de arriba es para la tilde de SINTÉTICA: la caja oculta el rebote sin cortarla. */
  .vwrap { position: relative; overflow: hidden; padding-top: .5em; margin-top: -.5em; padding-bottom: .08em; }
  .verdict {
    margin: 0; font-size: clamp(60px, 10vw, 160px); line-height: .9; font-weight: 800;
    letter-spacing: -.025em; text-transform: uppercase; color: var(--synthetic);
  }
  .verdict.human { color: var(--human); }
  .verdict.none { color: var(--ink-strong); font-size: clamp(46px, 6.4vw, 96px); text-transform: none; letter-spacing: -.02em; }
  .verdict.in { animation: verdict-in 560ms var(--bounce) both; }
  .verdict.ghost { position: absolute; left: 0; right: 0; bottom: .08em; pointer-events: none; animation: verdict-out 420ms cubic-bezier(.4,0,.2,1) forwards; }

  .conf { display: flex; align-items: center; gap: 18px; margin-top: clamp(22px, 3.4vh, 38px); }
  .conf-num { font-size: clamp(50px, 5.4vw, 82px); font-weight: 700; letter-spacing: -.03em; line-height: .9; color: var(--confidence); white-space: nowrap; }
  .conf-copy { display: grid; gap: 2px; }
  .conf-copy b { font-size: clamp(22px, 1.8vw, 26px); font-weight: 700; color: var(--ink-strong); }
  .conf-copy span { font-size: 16px; color: var(--muted); }
  .why { margin: 18px 0 0; font-size: 18px; color: var(--muted); max-width: 40ch; }

  .lat { padding: 26px 28px 24px; border-radius: 22px; background: linear-gradient(180deg, #f7fafb, #f1f5f6); box-shadow: inset 0 0 0 1px rgba(77,111,128,.10); }
  .lat-num { display: flex; align-items: baseline; gap: 12px; margin-top: 2px; }
  .lat-num > b { font-size: clamp(62px, 6.6vw, 104px); font-weight: 700; letter-spacing: -.035em; line-height: .88; color: var(--ink-strong); white-space: nowrap; }
  .lat-num > span { font-size: clamp(21px, 1.8vw, 26px); font-weight: 600; }
  .lat p { margin: 14px 0 0; font-size: 17px; color: var(--muted); }
  .lat p b { color: var(--ink-strong); }
  .pill {
    display: inline-flex; align-items: center; gap: 8px; margin-top: 14px; padding: 7px 14px 7px 9px;
    border-radius: 99px; background: #fff; box-shadow: 0 0 0 1px rgba(77,111,128,.18);
    font-size: 16px; font-weight: 700; color: var(--human);
  }
  .pill svg { width: 18px; height: 18px; flex: none; }
  .pill.over { color: #fff; background: var(--brand); box-shadow: none; }

  /* Dígitos que ruedan. El que sale va en absoluto: la caja mide solo el dígito nuevo,
     así "segundos" no se corre durante el giro ni salta al terminar. */
  .roll { position: relative; display: inline-block; overflow: hidden; vertical-align: bottom; padding: .06em 0; margin: -.06em 0; }
  .roll .o { position: absolute; left: 0; top: .06em; animation: roll-out 420ms cubic-bezier(.4,0,.2,1) forwards; }
  .roll .n { display: inline-block; animation: roll-in 560ms var(--bounce) both; }

  .rule { width: min(240px, 40%); height: 2px; border-radius: 2px; background: var(--structure); margin: clamp(28px, 4vh, 44px) auto 0; }

  .budget { margin-top: clamp(20px, 3vh, 32px); }
  .budget-top { display: flex; justify-content: space-between; align-items: baseline; gap: 12px; font-size: 16px; color: var(--muted); }
  .budget-top b { color: var(--ink-strong); }
  .sausage-chain { display: grid; grid-template-columns: repeat(10, minmax(0, 1fr)); gap: 10px; margin-top: 14px; }
  .sausage-link {
    position: relative; height: clamp(18px, 1.8vw, 24px); border-radius: 99px; background: #f2f6f7; overflow: hidden;
    box-shadow: inset 0 0 0 1px rgba(77,111,128,.16), inset 0 2px 3px rgba(35,60,72,.06);
  }
  .sausage-link:not(:last-child)::after {
    content: ""; position: absolute; right: -9px; top: 50%; width: 8px; height: 2px;
    background: rgba(77,111,128,.28); transform: translateY(-50%);
  }
  .sausage-fill { position: absolute; inset: 0; border-radius: inherit; background: linear-gradient(180deg, #5f8394, #4d6f80); transform-origin: left; transform: scaleX(0); }
  .over-state .sausage-fill { background: var(--brand); }
  .motion .sausage-fill { transition: transform 520ms var(--bounce); }
  .ticks { position: relative; height: 20px; margin-top: 9px; font-size: 15px; color: var(--faint); }
  .ticks span { position: absolute; top: 0; transform: translateX(-50%); white-space: nowrap; }
  .ticks span:first-child { transform: none; }
  .ticks span:last-child { transform: translateX(-100%); }

  .more { margin-top: 24px; padding-top: 22px; border-top: 1px solid rgba(77,111,128,.12); }
  .toggle {
    appearance: none; border: 0; cursor: pointer; display: inline-flex; align-items: center; gap: 12px;
    padding: 6px 6px 6px 18px; border-radius: 99px; background: #f2f6f7; box-shadow: inset 0 0 0 1px rgba(77,111,128,.14);
    font: 600 16px/1 "Urbanist", system-ui, sans-serif; color: var(--ink-strong); transition: transform .5s var(--ease);
  }
  .toggle:active { transform: scale(.98); }
  .knob { width: 30px; height: 30px; border-radius: 50%; background: #fff; display: grid; place-items: center; box-shadow: 0 0 0 1px rgba(77,111,128,.14), 0 4px 10px -6px rgba(35,60,72,.4); }
  .knob svg { width: 14px; height: 14px; transition: transform .6s var(--bounce); }
  .toggle[aria-expanded="true"] .knob svg { transform: rotate(180deg); }
  .drawer { display: grid; grid-template-rows: 0fr; transition: grid-template-rows .7s var(--ease); }
  .drawer.open { grid-template-rows: 1fr; }
  .drawer > div { overflow: hidden; }
  .split { display: flex; height: 14px; margin: 22px 0 16px; gap: 3px; }
  .split i { display: block; border-radius: 99px; min-width: 6px; }
  .legend { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 14px; margin: 0; }
  .legend div { padding: 14px 16px; border-radius: 16px; background: #f7fafb; box-shadow: inset 0 0 0 1px rgba(77,111,128,.10); }
  .legend dt { display: flex; align-items: center; gap: 8px; font-size: 16px; color: var(--muted); }
  .legend dt i { width: 10px; height: 10px; border-radius: 3px; }
  .legend dd { margin: 6px 0 0; font-size: 26px; font-weight: 700; color: var(--ink-strong); }
  .legend small { display: block; font-size: 15px; color: var(--muted); font-weight: 500; margin-top: 2px; }
  .foot { margin: 16px 0 4px; font-size: 16px; color: var(--muted); max-width: 78ch; }

  .hist { margin-top: 56px; }
  .hist-head { display: flex; justify-content: space-between; align-items: baseline; gap: 16px; margin-bottom: 14px; }
  .hist h2 { margin: 0; font-size: clamp(22px, 2.2vw, 28px); font-weight: 700; white-space: nowrap; color: var(--ink-strong); }
  .hist-head span { font-size: 16px; color: var(--muted); text-align: right; }
  .tw { overflow-x: auto; }
  table { width: 100%; min-width: 720px; border-collapse: collapse; font-size: 17px; }
  th { font-size: 15px; font-weight: 600; color: var(--muted); text-align: left; padding: 0 12px 12px; }
  td { padding: 14px 12px; border-top: 1px solid rgba(77,111,128,.12); white-space: nowrap; }
  td.n, th.n { text-align: right; }
  .dot { display: inline-flex; align-items: center; gap: 9px; font-weight: 700; }
  .dot::before { content: ""; width: 8px; height: 8px; border-radius: 50%; background: currentColor; }
  .syn { color: var(--synthetic); }
  .hum { color: var(--human); }
  .err { color: var(--muted); }
  .late { color: var(--brand); font-weight: 700; }
  .history-empty { color: var(--muted); padding: 28px 12px; }

  @keyframes live { 0%, 100% { opacity: 1; transform: scale(1); } 50% { opacity: .25; transform: scale(.72); } }
  @keyframes verdict-in { from { opacity: 0; transform: translateY(70%); } to { opacity: 1; transform: none; } }
  @keyframes verdict-out { to { opacity: 0; transform: translateY(-70%); } }
  @keyframes roll-out { to { opacity: 0; transform: translateY(-90%); } }
  @keyframes roll-in { from { opacity: 0; transform: translateY(90%); } to { opacity: 1; transform: none; } }

  @media (max-width: 900px) {
    .readout { grid-template-columns: 1fr; align-items: start; }
    .legend { grid-template-columns: 1fr; }
  }
  @media (prefers-reduced-motion: reduce) {
    *, *::before, *::after { animation: none !important; transition: none !important; }
  }
</style>
</head>
<body>
<div class="blob" aria-hidden="true"></div><div class="grain" aria-hidden="true"></div>
<div class="page">
  <header class="mast">
    <div class="lockup" aria-label="altur X CHORI">
      <img class="logo-altur" alt="altur" src="data:image/png;base64,__ALTUR__">
      <span class="x" aria-hidden="true">X</span>
      <img class="logo-chori" alt="CHORI, Calibrador Holístico Ortogonal de Respuesta Inmediata" src="data:image/png;base64,__CHORI__">
    </div>
    <div class="live" id="live" data-state="connecting" role="status" aria-live="polite">
      <i class="live-dot" aria-hidden="true"></i><span class="live-pill" id="live-text">Conectando</span>
    </div>
  </header>

  <div class="shell"><main class="core" id="core">
    <div class="core-head"><strong>Última llamada</strong><span id="call-ref">Esperando datos</span></div>

    <section class="notice" id="notice">
      <h1 id="notice-title">Conectando al detector</h1>
      <p id="notice-copy">Esta pantalla se actualiza sola cada segundo.</p>
    </section>

    <div id="call" hidden>
      <section class="readout" aria-label="Resultado de la última llamada" aria-live="polite" aria-atomic="true">
        <div>
          <span class="k">Veredicto</span>
          <div class="vwrap" id="vwrap"><h1 class="verdict" id="verdict"></h1></div>
          <div class="conf" id="conf">
            <strong class="conf-num" id="confidence-value"></strong>
            <span class="conf-copy"><b>Confianza</b><span>en esta llamada, no el accuracy del modelo</span></span>
          </div>
          <p class="why" id="why" hidden>La petición no se pudo procesar, así que no hay veredicto ni confianza que mostrar.</p>
        </div>
        <div class="lat">
          <span class="k">Tardó en responder</span>
          <div class="lat-num"><b id="latency-value"></b><span>segundos</span></div>
          <p>El juez da <b>30 segundos</b> por llamada.</p>
          <span class="pill" id="pill"></span>
        </div>
      </section>

      <div class="rule" aria-hidden="true"></div>

      <section class="budget" aria-label="Latencia usada del límite de 30 segundos">
        <div class="budget-top"><span id="budget-copy"></span><span>Límite</span></div>
        <div class="sausage-chain" aria-hidden="true">
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
        <div class="ticks" aria-hidden="true">
          <span style="left:0">0 s</span><span style="left:16.66%">5</span><span style="left:33.33%">10</span>
          <span style="left:50%">15</span><span style="left:66.66%">20</span><span style="left:83.33%">25</span>
          <span style="left:100%">30 s</span>
        </div>
      </section>

      <section class="more">
        <button class="toggle" id="toggle" type="button" aria-expanded="false" aria-controls="drawer">Ver en qué se fue el tiempo
          <span class="knob" aria-hidden="true"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"><path d="m6 9 6 6 6-6"/></svg></span></button>
        <div class="drawer" id="drawer"><div>
          <div class="split" id="split" aria-hidden="true"></div>
          <dl class="legend" id="legend"></dl>
          <p class="foot"><span id="worst">La llamada más lenta todavía no tiene desglose.</span>
            El juez ve entre 0.1 y 0.25 s más que esta pantalla, porque abrir la conexión y el viaje de
            vuelta no pasan por el servidor (~2 RTT).</p>
        </div></div>
      </section>
    </div>
  </main></div>

  <section class="hist" aria-labelledby="history-title">
    <div class="hist-head"><h2 id="history-title">Llamadas recientes</h2><span id="history-counts">0 llamadas</span></div>
    <div class="tw" tabindex="0" aria-label="Historial de llamadas, desplazable horizontalmente">
      <table>
        <thead><tr>
          <th>Hora</th><th>Veredicto</th><th class="n">Confianza</th><th class="n">Tardó</th>
          <th class="n">Subida</th><th class="n">Modelo</th><th class="n">Audio</th>
        </tr></thead>
        <tbody id="rows"><tr><td colspan="7" class="history-empty">Sin llamadas todavía.</td></tr></tbody>
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
const BOLT = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" ' +
  'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M13 3 5 14h6l-1 7 8-11h-6l1-7Z"/></svg>';
let lastCallKey = null;
let initialLoad = true;
let pollInFlight = false;

const pad = (n) => String(n).padStart(2, "0");
const hora = (ts) => {
  const d = new Date(ts * 1000);
  return Number.isFinite(d.getTime()) ? pad(d.getHours()) + ":" + pad(d.getMinutes()) + ":" + pad(d.getSeconds()) : "—";
};
const total = (c) => c.server_ms != null ? c.server_ms : (c.upload_ms || 0) + (c.ms || 0);
const decode = (c) => c.server_ms != null ? Math.max(0, c.server_ms - (c.upload_ms || 0) - (c.ms || 0)) : null;
const seconds = (ms) => (ms / 1000).toFixed(2);
const times = (ms) => { const x = BUDGET_MS / ms; return x >= 10 ? String(Math.round(x)) : x.toFixed(1); };
const esc = (value) => String(value).replace(/[&<>"']/g, (ch) => ({
  "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
})[ch]);

function setLive(state, text) {
  $("live").dataset.state = state;
  $("live-text").textContent = text;
}

function showNotice(kind) {
  $("call").hidden = true;
  $("notice").hidden = false;
  if (kind === "empty") {
    $("notice-title").textContent = "Esperando la primera llamada";
    $("notice-copy").textContent = "En cuanto el juez mande un audio, aquí aparecen su veredicto, la confianza y cuánto tardó.";
    $("call-ref").textContent = "Sin llamadas todavía";
  } else {
    $("notice-title").textContent = "Sin conexión con el detector";
    $("notice-copy").textContent = "Reintentando cada segundo. Los datos vuelven solos en cuanto responda.";
    $("call-ref").textContent = "Sin respuesta";
  }
}

// Solo cambian los caracteres distintos; el que sale sube y el que entra rebota.
function rollTo(el, next, animate) {
  const prev = el.dataset.v || "";
  el.dataset.v = next;
  window.clearTimeout(el._settle);
  if (!animate || !prev || !next || prev === next) {
    el.textContent = next;
    return;
  }
  const length = Math.max(prev.length, next.length);
  const a = prev.padStart(length);
  const b = next.padStart(length);
  el.innerHTML = Array.from({length}, (_, i) => {
    if (a[i] === b[i]) return esc(b[i]);
    return '<span class="roll"><span class="o">' + esc(a[i]) + '</span><span class="n">' + esc(b[i]) + "</span></span>";
  }).join("").replace(/^ +/, "");
  el._settle = window.setTimeout(() => { el.textContent = next; }, 600);
}

function setVerdict(c, animate) {
  const el = $("verdict");
  const ok = c.status === 200 && typeof c.is_synthetic === "boolean";
  const text = !ok ? "Sin veredicto" : c.is_synthetic ? "Sintética" : "Humana";
  const cls = "verdict" + (!ok ? " none" : c.is_synthetic ? "" : " human");
  if (animate && el.textContent && el.textContent !== text) {
    const ghost = el.cloneNode(true);
    ghost.removeAttribute("id");
    ghost.className = el.className.replace(" in", "") + " ghost";
    $("vwrap").appendChild(ghost);
    window.setTimeout(() => ghost.remove(), 460);
  }
  el.className = cls;
  el.textContent = text;
  if (animate) { void el.offsetWidth; el.classList.add("in"); }
  return ok;
}

function setBreakdown(c) {
  const parts = [
    ["Subida del audio", c.upload_ms, "#5f8394", "Depende de la red del juez"],
    ["Lectura del audio", decode(c), "#b9cad0", "Abrir y decodificar el archivo"],
    ["Modelo", c.ms, "#1a1a1a", "La decisión en sí"],
  ];
  $("split").innerHTML = parts.map((p) => '<i style="flex:' + Math.max(Number(p[1]) || 0, 1) + ";background:" + p[2] + '"></i>').join("");
  $("legend").innerHTML = parts.map((p) => '<div><dt><i style="background:' + p[2] + '"></i>' + p[0] +
    "</dt><dd>" + (p[1] != null ? seconds(p[1]) + " s" : "—") + "<small>" + p[3] + "</small></dd></div>").join("");
}

function renderLatest(c, animate) {
  const motion = animate && !reducedMotion.matches;
  const latency = total(c);
  const over = latency > BUDGET_MS;
  $("notice").hidden = true;
  $("call").hidden = false;
  $("call-ref").textContent = "Referencia " + (c.ref || "sin referencia") + ", " + hora(c.ts);

  const ok = setVerdict(c, motion);
  const hasConfidence = ok && typeof c.confidence === "number";
  $("conf").hidden = !hasConfidence;
  $("why").hidden = ok;
  const confidence = hasConfidence ? Math.round(c.confidence * 100) + "%" : "";
  rollTo($("confidence-value"), confidence, motion);
  $("confidence-value").setAttribute("aria-label", hasConfidence ? confidence + " de confianza en esta llamada" : "Sin confianza");

  rollTo($("latency-value"), latency > 0 ? seconds(latency) : "—", motion);
  const pill = $("pill");
  pill.hidden = !(latency > 0);
  pill.className = "pill" + (over ? " over" : "");
  pill.innerHTML = BOLT + "<span>" + (over
    ? "Pasó el límite del juez por " + seconds(latency - BUDGET_MS) + " s"
    : times(latency) + " veces más rápido que el límite") + "</span>";

  const core = $("core");
  core.classList.toggle("over-state", over);
  core.classList.toggle("motion", motion);
  $("budget-copy").innerHTML = latency > 0
    ? "<b>" + seconds(latency) + " s</b>" + (over ? ", más que los 30 s disponibles" : " de los 30 s disponibles")
    : "Latencia no disponible";
  const used = Math.max(0, Math.min(10, (latency / BUDGET_MS) * 10));
  document.querySelectorAll(".sausage-fill").forEach((fill, i) => {
    fill.style.transform = "scaleX(" + Math.max(0, Math.min(1, used - i)) + ")";
  });
  setBreakdown(c);
}

function renderSummary(summary) {
  const s = summary || {};
  const w = s.worst;
  $("history-counts").textContent = (s.calls ?? 0) + " llamadas, " + (s.ok ?? 0) +
    " contestadas, " + (s.errors ?? 0) + " errores";
  $("worst").textContent = w
    ? "La llamada más lenta hasta ahora tardó " + seconds(w.total_ms) + " s: subida " + seconds(w.upload_ms) +
      " s, lectura " + seconds(w.decode_ms) + " s y modelo " + seconds(w.inference_ms) + " s" +
      (s.worst_headroom_x != null ? ", " + s.worst_headroom_x + " veces bajo el límite." : ".")
    : "La llamada más lenta todavía no tiene desglose.";
}

function renderRows(calls) {
  $("rows").innerHTML = calls.length ? calls.map((c) => {
    const ok = c.status === 200 && typeof c.is_synthetic === "boolean";
    const verdict = !ok ? '<span class="dot err">Sin veredicto</span>'
      : c.is_synthetic ? '<span class="dot syn">Sintética</span>' : '<span class="dot hum">Humana</span>';
    const latency = total(c);
    const late = latency > BUDGET_MS ? ' <span class="late">pasó el límite</span>' : "";
    return "<tr><td>" + hora(c.ts) + "</td><td>" + verdict + late + '</td><td class="n">' +
      (ok && typeof c.confidence === "number" ? Math.round(c.confidence * 100) + "%" : "—") + '</td><td class="n">' +
      (latency > 0 ? seconds(latency) + " s" : "—") + '</td><td class="n">' +
      (c.upload_ms != null ? seconds(c.upload_ms) + " s" : "—") + '</td><td class="n">' +
      (ok && c.ms != null ? Number(c.ms).toFixed(0) + " ms" : "—") + '</td><td class="n">' +
      (c.mb != null ? Number(c.mb).toFixed(2) + " MB" : "—") + "</td></tr>";
  }).join("") : '<tr><td colspan="7" class="history-empty">Sin llamadas todavía.</td></tr>';
}

function setOpen(open) {
  $("toggle").setAttribute("aria-expanded", String(open));
  $("drawer").classList.toggle("open", open);
}
$("toggle").addEventListener("click", () => setOpen($("toggle").getAttribute("aria-expanded") !== "true"));

async function tick() {
  if (pollInFlight) return;
  pollInFlight = true;
  try {
    const response = await fetch(CALLS_URL, {cache: "no-store"});
    if (!response.ok) throw new Error(response.status);
    const data = await response.json();
    const ready = data.ready === true;
    const calls = data.calls || [];
    setLive(ready ? "live" : "off", ready ? "LIVE" : "Detector no listo");
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
      } else if ($("call").hidden) {
        // Una reconexión restaura el dato existente sin fingir que llegó otra llamada.
        renderLatest(c, false);
      }
    }
    initialLoad = false;
  } catch (_error) {
    setLive("off", "Sin conexión");
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

PAGE = (
    _TEMPLATE.replace("__FONT__", _assets.FONT_WOFF2)
    .replace("__GRAIN__", _assets.GRAIN_PNG)
    .replace("__ALTUR__", _assets.ALTUR_PNG)
    .replace("__CHORI__", _assets.CHORI_PNG)
)
