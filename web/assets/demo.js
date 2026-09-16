/* Live demo page: /stream, /replay and the race. */
"use strict";

const PRESETS = [
  { d: "structured", p: "Return a JSON object describing a book: title, author, year, tags." },
  { d: "code", p: "Write a Python function that checks whether a string is a palindrome, ignoring case and spaces." },
  { d: "math", p: "Solve step by step: a train travels 180 km in 2.5 hours. What is its average speed in km/h and m/s?" },
  { d: "reasoning", p: "A deploy failed only on Fridays. List the most likely causes and how you would confirm each one." },
  { d: "prose", p: "Write a short, warm paragraph welcoming new volunteers to a community garden." },
  { d: "translation", p: "Translate into French: The meeting has been moved to Thursday afternoon because the room is unavailable." },
  { d: "mixed", p: "In two sentences, explain what a hash map is and why lookups are fast. Then write a short Python class HashMap with put and get methods using separate chaining." },
];

const state = {
  mode: "spec", regime: "graphed",
  es: null, busy: false,
  rounds: [], counts: { accept: 0, correct: 0, bonus: 0 },
  t0: 0, tokens: 0,
  // A speedup is only meaningful against a baseline measured on the SAME
  // prompt in the SAME regime; graphed and eager differ ~2x on their own.
  baselineRef: null,     // {tps, regime, prompt}
  currentRun: null,      // {regime, prompt, mode} or {replay: true}
};

const el = (id) => document.getElementById(id);

/* ------------------------------------------------------------------ controls */
function segmented(id, onChange) {
  const root = el(id);
  root.addEventListener("click", (e) => {
    const b = e.target.closest("button");
    if (!b || state.busy) return;
    $$("button", root).forEach((x) => x.classList.toggle("on", x === b));
    onChange(b.dataset.v);
  });
}

function gammaHint() {
  const g = +el("gamma").value;
  el("gamma-val").textContent = g;
  let html;
  if (state.mode !== "spec") html = "";
  else if (g === 0) html = `<span class="tag">off</span> γ=0 is plain decoding — no draft involved.`;
  else if (g <= 2) html = `<span class="tag good">✓ sweet spot</span> γ=${g} stays below the verify-cost step. Best single γ overall.`;
  else if (g <= 4) html = `<span class="tag warn">⚠ pessimal</span> γ=${g} verifies ${g + 1} tokens: pays the whole <span class="term" data-term="verifystep">cost step</span> without amortising it.`;
  else if (g <= 8) html = `<span class="tag info">↗ bold</span> Pays off on math/structured/code; hurts prose and translation.`;
  else html = `<span class="tag bad">✗ aggressive</span> Most long guesses get rejected; usually slower than baseline.`;
  el("gamma-hint").innerHTML = html;
}

function modeHint() {
  const hints = {
    spec: "Draft guesses γ tokens each round; target verifies them.",
    baseline: "Target alone, one token per step. Run this first to get a speedup reading.",
    adaptive: "γ is chosen live by the controller from observed acceptance.",
  };
  el("mode-hint").textContent = hints[state.mode];
  el("gamma-field").style.opacity = state.mode === "spec" ? 1 : 0.4;
  el("gamma").disabled = state.mode !== "spec";
  gammaHint();
}

function setBusy(busy, label) {
  state.busy = busy;
  ["go", "race", "replay"].forEach((id) => { el(id).disabled = busy; });
  el("stop").style.display = busy ? "" : "none";
  el("live").classList.toggle("on", busy);
  el("live-text").textContent = busy ? (label || "streaming") : "idle";
}

/* ------------------------------------------------------------------ tokens */
const tokcard = () => el("tokcard");
function describeToken(t, round) {
  const show = (s) => `<code>${escapeHTML(JSON.stringify(s))}</code>`;
  if (t.origin === "accept")
    return `<b style="color:var(--accept)">✓ accepted</b><br>Draft guessed ${show(t.text)} and the target agreed.<br><span class="faint">round ${round.round} · γ=${round.gamma}</span>`;
  if (t.origin === "correct")
    return `<b style="color:var(--correct)">✎ corrected</b><br>Draft guessed ${t.draft_guessed != null ? show(t.draft_guessed) : "something else"}, target chose ${show(t.text)}. Later guesses this round were discarded.<br><span class="faint">round ${round.round} · γ=${round.gamma}</span>`;
  return `<b style="color:var(--bonus)">★ bonus</b><br>All guesses accepted, so the target's verify pass gave ${show(t.text)} for free.<br><span class="faint">round ${round.round} · γ=${round.gamma}</span>`;
}

function addTokens(container, round) {
  const empty = $(".empty", container);
  if (empty) empty.remove();
  (round.tokens || []).forEach((t) => {
    const s = document.createElement("span");
    s.className = "tok " + t.origin;
    s.textContent = t.text;
    s._info = [t, round];
    container.appendChild(s);
  });
  container.scrollTop = container.scrollHeight;
}

function wireTokenHover(container) {
  container.addEventListener("mousemove", (e) => {
    const s = e.target.closest(".tok");
    const card = tokcard();
    if (!s || !s._info) { card.classList.remove("show"); return; }
    card.innerHTML = describeToken(...s._info);
    card.classList.add("show");
    const x = Math.min(window.innerWidth - card.offsetWidth - 10, e.clientX + 14);
    const y = e.clientY + 18 + card.offsetHeight > window.innerHeight ? e.clientY - card.offsetHeight - 12 : e.clientY + 18;
    card.style.left = x + "px"; card.style.top = y + "px";
  });
  container.addEventListener("mouseleave", () => tokcard().classList.remove("show"));
}

/* ------------------------------------------------------------------ stats */
function updateCounts() {
  const c = state.counts, total = c.accept + c.correct + c.bonus || 1;
  ["accept", "correct", "bonus"].forEach((k) => {
    el("c-" + k).textContent = c[k];
    el("sb-" + k).style.width = (100 * c[k] / total) + "%";
  });
}

function resetRun() {
  state.rounds = []; state.counts = { accept: 0, correct: 0, bonus: 0 };
  state.tokens = 0; state.t0 = performance.now();
  el("stream").innerHTML = "";
  ["k-gamma", "k-acc", "k-tpr", "k-tps", "k-speed"].forEach((id) => { el(id).textContent = "–"; el(id).dataset.cur = "0"; });
  el("k-speed").style.color = "";
  updateCounts(); drawCharts();
}

function onRound(d) {
  state.rounds.push(d);
  (d.tokens || []).forEach((t) => { state.counts[t.origin] = (state.counts[t.origin] || 0) + 1; });
  state.tokens += (d.tokens || []).length;
  addTokens(el("stream"), d);
  updateCounts();
  el("k-gamma").textContent = d.gamma;
  const drafted = state.rounds.reduce((a, r) => a + r.gamma, 0);
  const accepted = state.rounds.reduce((a, r) => a + r.accepted, 0);
  el("k-acc").textContent = drafted ? (accepted / drafted).toFixed(2) : "–";
  el("k-tpr").textContent = (state.tokens / state.rounds.length).toFixed(2);
  const secs = (performance.now() - state.t0) / 1000;
  if (secs > 0.3) el("k-tps").textContent = (state.tokens / secs).toFixed(1);
  drawCharts();
}

function onDone(d) {
  const s = d.stats || {};
  if (s.tokens_per_round != null) countUp(el("k-tpr"), s.tokens_per_round, { decimals: 2, ms: 600 });
  if (s.acceptance_rate != null) countUp(el("k-acc"), s.acceptance_rate, { decimals: 2, ms: 600 });
  const run = state.currentRun;
  let note = "";
  if (s.tok_per_s != null) {
    countUp(el("k-tps"), s.tok_per_s, { decimals: 1, ms: 700 });
    let ratio = null;
    if (d.speedup != null) {
      ratio = d.speedup;       // recorded traces carry the speedup measured at recording time
      note = `${d.speedup.toFixed(2)}× its recorded baseline (${(d.baseline_tok_per_s || 0).toFixed(1)} tok/s)`;
    } else if (run && !run.replay && run.mode === "baseline") {
      state.baselineRef = { tps: s.tok_per_s, regime: run.regime, prompt: run.prompt };
      note = "baseline saved — now run Speculative on this prompt to see the speedup";
      el("k-speed").textContent = "1.00×";
    } else if (run && state.baselineRef && !run.replay &&
               state.baselineRef.regime === run.regime && state.baselineRef.prompt === run.prompt) {
      ratio = s.tok_per_s / state.baselineRef.tps;
      note = `vs ${run.regime} baseline ${state.baselineRef.tps.toFixed(1)} tok/s on this prompt`;
    } else if (run && !run.replay) {
      note = `no ${run.regime} baseline for this prompt yet — run Baseline or 🏁 Race`;
    }
    if (ratio != null) {
      countUp(el("k-speed"), ratio, { decimals: 2, suffix: "×", ms: 900 });
      el("k-speed").style.color = ratio >= 1 ? "var(--emerald)" : "var(--rose)";
      if (!REDUCED && ratio >= 1.1) celebrate(el("k-speed"));
    }
  }
  addStatus(`<span class="tag ${s.tok_per_s ? "good" : ""}">done</span> ${s.tokens ?? state.tokens} tokens${note ? " · " + note : ""}`);
}

function addStatus(html) { el("status").innerHTML += (el("status").innerHTML ? " " : "") + html; }

function celebrate(anchor) {
  const r = anchor.getBoundingClientRect();
  const colors = [COLORS.emerald, COLORS.cyan, COLORS.violet, COLORS.amber];
  for (let i = 0; i < 18; i++) {
    const p = document.createElement("i");
    Object.assign(p.style, {
      position: "fixed", left: r.left + r.width / 2 + "px", top: r.top + r.height / 2 + "px",
      width: "7px", height: "7px", borderRadius: "2px", background: colors[i % 4],
      zIndex: 300, pointerEvents: "none",
      transition: "transform .9s cubic-bezier(.2,.8,.2,1), opacity .9s",
    });
    document.body.append(p);
    requestAnimationFrame(() => {
      const a = Math.random() * Math.PI * 2, dist = 40 + Math.random() * 70;
      p.style.transform = `translate(${Math.cos(a) * dist}px, ${Math.sin(a) * dist}px) rotate(${Math.random() * 360}deg)`;
      p.style.opacity = "0";
    });
    setTimeout(() => p.remove(), 1000);
  }
}

/* ------------------------------------------------------------------ charts */
function drawCharts() {
  // acceptance sparkline
  const { ctx, w, h } = setupCanvas(el("spark"), 120);
  const data = state.rounds.filter((r) => r.gamma > 0).slice(-60).map((r) => r.rate);
  ctx.clearRect(0, 0, w, h);
  ctx.strokeStyle = COLORS.grid; ctx.lineWidth = 1;
  [0.25, 0.5, 0.75].forEach((f) => { ctx.beginPath(); ctx.moveTo(0, h - f * (h - 10) - 5); ctx.lineTo(w, h - f * (h - 10) - 5); ctx.stroke(); });
  ctx.fillStyle = COLORS.faint; ctx.font = "10px system-ui";
  ctx.fillText("1.0", 2, 12); ctx.fillText("0.5", 2, h / 2 - 2);
  if (data.length >= 2) {
    const X = (i) => 22 + (i / (data.length - 1)) * (w - 28);
    const Y = (v) => h - 5 - v * (h - 10);
    const grad = ctx.createLinearGradient(0, 0, 0, h);
    grad.addColorStop(0, "rgba(52,211,153,.35)"); grad.addColorStop(1, "rgba(52,211,153,0)");
    ctx.beginPath(); ctx.moveTo(X(0), h);
    data.forEach((v, i) => ctx.lineTo(X(i), Y(v)));
    ctx.lineTo(X(data.length - 1), h); ctx.closePath(); ctx.fillStyle = grad; ctx.fill();
    ctx.beginPath(); data.forEach((v, i) => (i ? ctx.lineTo(X(i), Y(v)) : ctx.moveTo(X(i), Y(v))));
    ctx.strokeStyle = COLORS.emerald; ctx.lineWidth = 2; ctx.stroke();
    const last = data[data.length - 1];
    ctx.fillStyle = COLORS.emerald; ctx.beginPath(); ctx.arc(X(data.length - 1), Y(last), 4, 0, Math.PI * 2); ctx.fill();
  } else {
    ctx.fillStyle = COLORS.faint; ctx.fillText("waiting for speculative rounds…", 24, h / 2 + 16);
  }

  // gamma timeline
  const t = setupCanvas(el("gtimeline"), 120);
  const rounds = state.rounds.slice(-60);
  t.ctx.clearRect(0, 0, t.w, t.h);
  const gmax = Math.max(8, ...rounds.map((r) => r.gamma));
  const bw = (t.w - 4) / 60;
  rounds.forEach((r, i) => {
    const x = 2 + i * bw, full = (r.gamma / gmax) * (t.h - 16), acc = (r.accepted / gmax) * (t.h - 16);
    t.ctx.fillStyle = "rgba(124,124,255,.25)";
    t.ctx.fillRect(x, t.h - full, Math.max(1, bw - 2), full);
    t.ctx.fillStyle = r.accepted === r.gamma ? COLORS.violet : COLORS.cyan;
    t.ctx.fillRect(x, t.h - acc, Math.max(1, bw - 2), acc);
    if (r.gamma === 0) { t.ctx.fillStyle = COLORS.rose; t.ctx.fillRect(x, t.h - 3, Math.max(1, bw - 2), 3); }
  });
  if (!rounds.length) { t.ctx.fillStyle = COLORS.faint; t.ctx.font = "10px system-ui"; t.ctx.fillText("no rounds yet", 6, t.h / 2); }
}

/* ------------------------------------------------------------------ streaming */
function closeStream() { if (state.es) { state.es.close(); state.es = null; } setBusy(false); }

function openStream(url, label) {
  closeStream();
  resetRun();
  el("status").innerHTML = `<span class="spinner"></span> connecting…`;
  setBusy(true, label);
  state.es = new EventSource(url);
  state.es.onmessage = (ev) => {
    const d = JSON.parse(ev.data);
    if (d.type === "start") {
      state.t0 = performance.now();
      el("status").innerHTML = `<span class="tag info">${d.regime || ""}</span> <span class="tag">${d.mode}</span>` +
        (d.mode !== "baseline" ? ` <span class="tag">${typeof d.gamma === "number" ? "γ=" + d.gamma : "γ chosen live"}</span>` : "") +
        ` <span class="faint">${d.prompt_tokens ?? "?"} prompt tokens${d.replay_of ? " · replay of " + d.replay_of : ""}</span>`;
      if (d.mode === "baseline") {
        el("stream").innerHTML = `<div class="empty"><div><span class="spinner"></span><br><br>Baseline generates one token per step with no rounds to show — text appears when it finishes.</div></div>`;
      }
    } else if (d.type === "round") {
      onRound(d);
    } else if (d.type === "done") {
      if (!state.rounds.length && d.text) {
        el("stream").innerHTML = "";
        const s = document.createElement("span"); s.textContent = d.text; s.className = "tok";
        el("stream").append(s);
      }
      onDone(d); closeStream();
    } else if (d.type === "error") {
      addStatus(`<span class="tag bad">error</span> ${escapeHTML(d.message)}`); closeStream();
    }
  };
  state.es.onerror = () => { addStatus(`<span class="tag warn">connection closed</span>`); closeStream(); };
}

function runOnce() {
  const prompt = el("prompt").value.trim();
  if (!prompt) { toast("Type a prompt first"); return; }
  const q = new URLSearchParams({ prompt, mode: state.mode, gamma: el("gamma").value,
                                  regime: state.regime, max_new_tokens: el("mnt").value });
  state.currentRun = { regime: state.regime, prompt, mode: state.mode };
  openStream("/stream?" + q, state.mode === "baseline" ? "baseline" : "streaming");
}

function replay() {
  state.currentRun = { replay: true };
  openStream("/replay?file=results/mixed_trace.json&speed=1", "replaying");
}

/* ------------------------------------------------------------------ race */
function streamRun(params, container, onTok) {
  return new Promise((resolve) => {
    const src = new EventSource("/stream?" + new URLSearchParams(params));
    state.es = src;
    let text = "", tps = null, rounds = 0;
    src.onmessage = (ev) => {
      const d = JSON.parse(ev.data);
      if (d.type === "round") { rounds++; addTokens(container, d); onTok && onTok(); }
      else if (d.type === "done") {
        text = d.text || ""; tps = (d.stats || {}).tok_per_s;
        if (!rounds) container.textContent = text;
        src.close(); resolve({ text, tps, ok: true });
      } else if (d.type === "error") {
        container.innerHTML = `<span style="color:var(--rose)">${escapeHTML(d.message)}</span>`;
        src.close(); resolve({ text: "", tps: null, ok: false });
      }
    };
    src.onerror = () => { src.close(); resolve({ text, tps, ok: false }); };
  });
}

async function race() {
  const prompt = el("prompt").value.trim();
  if (!prompt) { toast("Type a prompt first"); return; }
  const regime = state.regime;
  const gamma = Math.max(1, +el("gamma").value);
  setBusy(true, "racing");
  el("race-card").scrollIntoView({ behavior: REDUCED ? "auto" : "smooth", block: "center" });
  el("race-base").innerHTML = `<div class="empty"><span class="spinner"></span></div>`;
  el("race-spec").innerHTML = `<div class="empty">waiting for baseline…</div>`;
  ["rb-bar", "rs-bar"].forEach((id) => { el(id).style.width = "0"; });
  el("rb-tps").textContent = "–"; el("rs-tps").textContent = "–";
  el("rs-gamma").textContent = `γ=${gamma}`;
  const v = el("race-verdict");
  v.className = "verdict-box"; v.innerHTML = `<span class="spinner"></span> running baseline (${regime})…`;

  const base = await streamRun({ prompt, mode: "baseline", gamma: 0, regime, max_new_tokens: el("mnt").value }, el("race-base"));
  if (!state.busy) return;
  if (base.tps) {
    state.baselineRef = { tps: base.tps, regime, prompt };
    el("rb-tps").textContent = base.tps.toFixed(1) + " tok/s";
  }
  el("race-spec").innerHTML = "";
  v.innerHTML = `<span class="spinner"></span> running speculative γ=${gamma}…`;
  const spec = await streamRun({ prompt, mode: "spec", gamma, regime, max_new_tokens: el("mnt").value }, el("race-spec"));
  if (!state.busy) return;
  if (spec.tps) el("rs-tps").textContent = spec.tps.toFixed(1) + " tok/s";

  if (base.tps && spec.tps) {
    const max = Math.max(base.tps, spec.tps);
    requestAnimationFrame(() => {
      el("rb-bar").style.width = (100 * base.tps / max) + "%";
      el("rs-bar").style.width = (100 * spec.tps / max) + "%";
    });
  }
  let same = 0;
  const n = Math.min(base.text.length, spec.text.length);
  while (same < n && base.text[same] === spec.text[same]) same++;
  const identical = base.ok && spec.ok && base.text === spec.text;
  const ratio = base.tps && spec.tps ? spec.tps / base.tps : null;
  const speedTxt = ratio ? `<b style="font-size:22px;color:${ratio >= 1 ? "var(--emerald)" : "var(--rose)"}">${ratio.toFixed(2)}×</b> throughput` : "throughput unavailable";
  if (!base.ok || !spec.ok) {
    v.className = "verdict-box bad";
    v.innerHTML = `A run did not finish, so there is nothing to compare.`;
  } else if (identical) {
    v.className = "verdict-box ok";
    v.innerHTML = `<div style="font-size:20px;font-weight:700">✓ Identical output — ${base.text.length} / ${base.text.length} characters</div><div style="margin-top:6px">${speedTxt} · ${regime} regime</div>`;
    if (ratio && ratio >= 1.05 && !REDUCED) celebrate(v);
  } else {
    v.className = "verdict-box bad";
    v.innerHTML = `<div style="font-size:18px;font-weight:700">Diverged at character ${same} of ${base.text.length}</div>
      <div class="muted" style="margin-top:6px">${speedTxt}. Usually an <span class="term" data-term="fp16tie">fp16 tie</span>: the target's top two tokens scored the same, so the pick depends on float rounding.</div>`;
  }
  state.es = null;
  setBusy(false);
}

/* ------------------------------------------------------------------ init */
document.addEventListener("DOMContentLoaded", () => {
  const presets = el("presets");
  PRESETS.forEach((p, i) => {
    const b = document.createElement("button");
    b.className = "chip" + (i === 0 ? " active" : "");
    const color = DOMAIN_COLORS[p.d] || COLORS.indigo;
    b.innerHTML = `<i class="dot" style="background:${color}"></i>${DOMAIN_ICONS[p.d] || "🔀"} ${p.d}`;
    b.onclick = () => {
      if (state.busy) return;
      el("prompt").value = p.p;
      $$(".chip", presets).forEach((c) => c.classList.toggle("active", c === b));
    };
    presets.append(b);
  });

  segmented("mode", (v) => { state.mode = v; modeHint(); });
  segmented("regime", (v) => { state.regime = v; });
  el("gamma").oninput = gammaHint;
  el("mnt").oninput = () => { el("mnt-val").textContent = el("mnt").value; };
  el("go").onclick = runOnce;
  el("replay").onclick = replay;
  el("race").onclick = race;
  el("stop").onclick = () => {
    closeStream();
    addStatus(`<span class="tag warn">stopped watching</span>`);
    toast("Stopped watching. A generation already on the GPU finishes in the background.");
  };
  el("prompt").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey) && !state.busy) runOnce();
  });
  wireTokenHover(el("stream"));
  wireTokenHover(el("race-spec"));
  modeHint();
  drawCharts();
  window.addEventListener("resize", drawCharts);

  fetchJSON("/health").then((h) => {
    if (!h.models_loaded) addStatus(`<span class="tag warn">models not loaded yet</span> <span class="faint">first Run will load them · Replay works now</span>`);
    else addStatus(`<span class="tag good">models ready</span>`);
  }).catch(() => addStatus(`<span class="tag bad">server unreachable</span>`));
});
