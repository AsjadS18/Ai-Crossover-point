/* How-it-works page: round simulator (real trace) and crossover explorer. */
"use strict";

const byId = (id) => document.getElementById(id);
const sleep = (ms) => new Promise((ok) => setTimeout(ok, REDUCED ? 0 : ms));
const show = (s) => JSON.stringify(s);

/* ------------------------------------------------------------ simulator */
const sim = { rounds: [], i: 0, playing: false, speed: 1, busy: false, tokens: 0, gen: 0 };

function slot(text, cls, badge) {
  const d = document.createElement("div");
  d.className = "slot " + cls;
  d.textContent = text === null ? "?" : text.replace(/\n/g, "⏎");
  if (badge) {
    const b = document.createElement("span");
    b.className = "badge"; b.textContent = badge[0]; b.style.background = badge[1];
    d.append(b);
  }
  return d;
}

async function playRound() {
  if (sim.busy || sim.i >= sim.rounds.length) return false;
  sim.busy = true;
  const gen = sim.gen;
  const r = sim.rounds[sim.i];
  const wait = (ms) => sleep(ms / sim.speed);
  const draftLane = byId("lane-draft"), targetLane = byId("lane-target"), narr = byId("narration");
  draftLane.innerHTML = ""; targetLane.innerHTML = "";
  byId("sim-round").textContent = `round ${r.round} / ${sim.rounds[sim.rounds.length - 1].round}`;
  byId("sim-gamma").textContent = r.gamma;

  const toks = r.tokens || [];
  const hasFix = toks.some((t) => t.origin === "correct");

  // 1. draft proposes gamma tokens
  if (r.gamma === 0) {
    narr.innerHTML = `γ=0 this round: <b>no guessing</b>. The target simply generates one token itself.`;
  } else {
    narr.innerHTML = `The draft quickly guesses <b>${r.gamma}</b> token${r.gamma > 1 ? "s" : ""}…`;
  }
  const guesses = [];
  for (let k = 0; k < r.gamma; k++) {
    const t = toks[k];
    let text, known = true;
    if (t && (t.origin === "accept" || t.origin === "correct")) text = t.draft_guessed;
    else { text = "…"; known = false; }      // discarded after a wrong guess, never recorded
    const s = slot(text, "draft" + (known ? "" : " ghost"));
    draftLane.append(s); guesses.push(s);
    await wait(260);
    if (gen !== sim.gen) return false;
  }
  await wait(350);

  // 2. target verifies in ONE pass
  if (r.gamma > 0) {
    narr.innerHTML = `The target checks <b>all ${r.gamma}</b> guesses in a <b>single</b> forward pass…`;
    guesses.forEach((g) => { const sc = document.createElement("i"); sc.className = "scan"; g.append(sc); });
    await wait(800);
    if (gen !== sim.gen) return false;
  }

  // 3. verdict per position
  for (let k = 0; k < toks.length; k++) {
    const t = toks[k];
    if (t.origin === "accept") {
      guesses[k].className = "slot ok";
      guesses[k].append(Object.assign(document.createElement("span"), { className: "badge", textContent: "✓", style: "background:var(--accept)" }));
      targetLane.append(slot(t.text, "ok"));
    } else if (t.origin === "correct") {
      guesses[k].className = "slot no";
      guesses[k].append(Object.assign(document.createElement("span"), { className: "badge", textContent: "✗", style: "background:var(--rose)" }));
      for (let j = k + 1; j < guesses.length; j++) guesses[j].className = "slot no ghost";
      targetLane.append(slot(t.text, "fix", ["✎", "var(--correct)"]));
    } else {
      targetLane.append(slot(t.text, "bonus", ["★", "var(--bonus)"]));
    }
    await wait(300);
    if (gen !== sim.gen) return false;
  }

  if (r.gamma === 0) narr.innerHTML = `Plain step: <b>1</b> token.`;
  else if (hasFix) {
    const fix = toks.find((t) => t.origin === "correct");
    narr.innerHTML = `<b style="color:var(--accept)">${r.accepted}</b> guess${r.accepted === 1 ? "" : "es"} accepted, then the draft said <code>${escapeHTML(show(fix.draft_guessed))}</code> but the target wanted <code>${escapeHTML(show(fix.text))}</code> — <span style="color:var(--correct)">corrected</span>. Round yields <b>${toks.length}</b> token${toks.length > 1 ? "s" : ""}.`;
  } else {
    narr.innerHTML = `All <b style="color:var(--accept)">${r.accepted}</b> guesses accepted, so the target's pass also gives a free <span style="color:var(--bonus)">bonus token</span>. <b>${toks.length}</b> tokens from one target pass!`;
  }

  // 4. append to output
  const out = byId("sim-out");
  toks.forEach((t) => {
    const s = document.createElement("span");
    s.className = "tok " + t.origin; s.textContent = t.text; out.append(s);
  });
  out.scrollTop = out.scrollHeight;
  sim.tokens += toks.length; sim.i++;
  byId("sim-tokens").textContent = sim.tokens;
  byId("sim-rounds").textContent = sim.i;
  byId("sim-tpr").textContent = (sim.tokens / sim.i).toFixed(2);
  await wait(900);
  sim.busy = false;
  return gen === sim.gen;
}

async function playLoop() {
  while (sim.playing) {
    const ok = await playRound();
    if (!ok || sim.i >= sim.rounds.length) break;
  }
  sim.playing = false;
  byId("sim-play").textContent = sim.i >= sim.rounds.length ? "↺ Replay" : "▶ Play";
}

function resetSim() {
  sim.gen++; sim.playing = false; sim.busy = false; sim.i = 0; sim.tokens = 0;
  ["lane-draft", "lane-target", "sim-out"].forEach((id) => { byId(id).innerHTML = ""; });
  byId("narration").textContent = "Press ▶ Play to start.";
  byId("sim-tokens").textContent = "0"; byId("sim-rounds").textContent = "0";
  byId("sim-tpr").textContent = "–"; byId("sim-play").textContent = "▶ Play";
  if (sim.rounds.length) byId("sim-round").textContent = `round – / ${sim.rounds[sim.rounds.length - 1].round}`;
}

/* ------------------------------------------------------------ explorer */
const ex = { alpha: 0.85, r: 0.248, model: "linear", shown: null, target: null, summary: null, domain: null, anim: 0 };
const GMAX = 12;

function expectedTokens(alpha, gamma) {
  if (gamma <= 0) return 1;
  if (alpha >= 1) return gamma + 1;
  return (1 - Math.pow(alpha, gamma + 1)) / (1 - alpha);
}

function verifyMs(width) {
  const w = ex.summary && ex.summary.verify_width && ex.summary.verify_width.widths;
  if (!w) return null;
  const row = w.find((x) => x.width === width);
  return row ? row.ms : w[w.length - 1].ms;
}

function costOf(gamma) {
  if (ex.model === "linear" || verifyMs(1) == null) return gamma * ex.r + 1;
  // measured: gamma draft steps (r x a width-1 target step) + a width-(gamma+1) verify
  const w1 = verifyMs(1);
  return (gamma * ex.r * w1 + verifyMs(gamma + 1)) / w1;
}

function computeCurve() {
  const vals = [];
  for (let g = 0; g <= GMAX; g++) vals.push(expectedTokens(ex.alpha, g) / costOf(g));
  return vals;
}

function drawCurve() {
  const target = computeCurve();
  const from = ex.shown || target;
  const t0 = performance.now(), dur = REDUCED ? 0 : 380;
  cancelAnimationFrame(ex.anim);
  const frame = (t) => {
    const p = dur ? Math.min(1, (t - t0) / dur) : 1;
    const e = 1 - Math.pow(1 - p, 3);
    ex.shown = target.map((v, i) => from[i] + (v - from[i]) * e);
    paint(ex.shown);
    if (p < 1) ex.anim = requestAnimationFrame(frame);
  };
  ex.anim = requestAnimationFrame(frame);
  updateText(target);
}

function paint(vals) {
  const { ctx, w, h } = setupCanvas(byId("curve"), 330);
  const padL = 44, padB = 34, padT = 16, padR = 14;
  const measured = measuredDots();
  const peak = Math.max(...vals, ...(measured || []).map((m) => m.v));
  const top = Math.max(1.4, Math.ceil(peak * 5) / 5 + 0.1);
  const X = (g) => padL + (g / GMAX) * (w - padL - padR);
  const Y = (v) => h - padB - (v / top) * (h - padB - padT);
  ctx.clearRect(0, 0, w, h);

  // grid
  ctx.font = "11px system-ui"; ctx.fillStyle = COLORS.faint; ctx.strokeStyle = COLORS.grid; ctx.lineWidth = 1;
  for (let v = 0; v <= top + 1e-9; v += 0.2) {
    ctx.beginPath(); ctx.moveTo(padL, Y(v)); ctx.lineTo(w - padR, Y(v)); ctx.stroke();
    ctx.fillText(v.toFixed(1) + "×", 6, Y(v) + 4);
  }
  for (let g = 0; g <= GMAX; g++) ctx.fillText(String(g), X(g) - 3, h - padB + 16);
  ctx.fillText("γ  (guesses per round)", w / 2 - 50, h - 4);

  // pessimal band for measured model
  if (ex.model === "measured") {
    ctx.fillStyle = "rgba(251,191,36,.07)";
    ctx.fillRect(X(2.5), padT, X(4.5) - X(2.5), h - padB - padT);
    ctx.fillStyle = "rgba(251,191,36,.8)"; ctx.fillText("step", X(3.5) - 12, padT + 12);
  }

  // area above / below break-even
  ctx.save();
  ctx.beginPath(); ctx.moveTo(X(0), Y(1));
  vals.forEach((v, g) => ctx.lineTo(X(g), Y(v)));
  ctx.lineTo(X(GMAX), Y(1)); ctx.closePath(); ctx.clip();
  ctx.fillStyle = "rgba(52,211,153,.14)"; ctx.fillRect(padL, padT, w, Y(1) - padT);
  ctx.fillStyle = "rgba(251,113,133,.14)"; ctx.fillRect(padL, Y(1), w, h - padB - Y(1));
  ctx.restore();

  // break-even
  ctx.strokeStyle = COLORS.rose; ctx.setLineDash([6, 5]); ctx.lineWidth = 1.5;
  ctx.beginPath(); ctx.moveTo(padL, Y(1)); ctx.lineTo(w - padR, Y(1)); ctx.stroke(); ctx.setLineDash([]);
  ctx.fillStyle = COLORS.rose; ctx.fillText("break-even", w - padR - 64, Y(1) - 6);

  // curve
  const grad = ctx.createLinearGradient(padL, 0, w, 0);
  grad.addColorStop(0, COLORS.indigo); grad.addColorStop(0.5, COLORS.cyan); grad.addColorStop(1, COLORS.emerald);
  ctx.strokeStyle = grad; ctx.lineWidth = 3; ctx.lineJoin = "round";
  ctx.beginPath(); vals.forEach((v, g) => (g ? ctx.lineTo(X(g), Y(v)) : ctx.moveTo(X(g), Y(v)))); ctx.stroke();
  vals.forEach((v, g) => { ctx.fillStyle = v >= 1 ? COLORS.cyan : COLORS.rose; ctx.beginPath(); ctx.arc(X(g), Y(v), 3, 0, Math.PI * 2); ctx.fill(); });

  // peak
  let best = 0; vals.forEach((v, g) => { if (v > vals[best]) best = g; });
  if (vals[best] > 1) {
    ctx.fillStyle = COLORS.emerald;
    ctx.beginPath(); ctx.arc(X(best), Y(vals[best]), 7, 0, Math.PI * 2); ctx.fill();
    ctx.strokeStyle = "rgba(52,211,153,.35)"; ctx.lineWidth = 6;
    ctx.beginPath(); ctx.arc(X(best), Y(vals[best]), 12, 0, Math.PI * 2); ctx.stroke();
  }

  // crossover marker: first gamma >= 1 after the peak where speedup < 1
  const cross = vals.findIndex((v, g) => g > best && v < 1);
  if (cross > 0) {
    ctx.strokeStyle = COLORS.amber; ctx.lineWidth = 1; ctx.setLineDash([2, 4]);
    ctx.beginPath(); ctx.moveTo(X(cross), padT); ctx.lineTo(X(cross), h - padB); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle = COLORS.amber; ctx.fillText("crossover", X(cross) + 4, padT + 26);
  }

  // measured dots for the selected domain
  if (measured) {
    measured.forEach((m) => {
      ctx.fillStyle = "#fff"; ctx.strokeStyle = DOMAIN_COLORS[ex.domain] || COLORS.cyan; ctx.lineWidth = 3;
      ctx.beginPath(); ctx.arc(X(m.g), Y(m.v), 5, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
    });
  }
}

function measuredDots() {
  const g = ex.summary && ex.summary.graphed;
  if (!ex.domain || !g) return null;
  return g.gammas.filter((x) => x <= GMAX)
    .map((x) => ({ g: x, v: g.speedup[ex.domain][String(x)] }))
    .filter((m) => m.v != null);
}

function updateText(vals) {
  byId("v-alpha").textContent = ex.alpha.toFixed(2);
  byId("v-r").textContent = ex.r.toFixed(3);
  let best = 0; vals.forEach((v, g) => { if (v > vals[best]) best = g; });
  const box = byId("best-box");
  if (vals[best] > 1.0001) {
    box.style.borderColor = "rgba(52,211,153,.45)"; box.style.background = "rgba(52,211,153,.07)";
    const cross = vals.findIndex((v, g) => g > best && v < 1);
    box.innerHTML = `🏆 Best <b>γ = ${best}</b> → <b style="color:var(--emerald)">${vals[best].toFixed(2)}×</b><br>
      <span class="muted" style="font-size:13px">${cross > 0 ? `From γ = ${cross} on, guessing makes it slower than no speculation.` : "No crossover within γ ≤ 12."}</span>`;
  } else {
    box.style.borderColor = "rgba(251,113,133,.45)"; box.style.background = "rgba(251,113,133,.07)";
    box.innerHTML = `🛑 <b style="color:var(--rose)">No γ pays.</b><br><span class="muted" style="font-size:13px">At this acceptance and cost, switch speculation off (γ = 0).</span>`;
  }
  const f = byId("formula");
  const costLine = ex.model === "linear"
    ? `<span class="k">cost</span>    = γ·r + 1`
    : `<span class="k">cost</span>    = (γ·r·t₁ + verify_ms(γ+1)) / t₁   <span class="faint">t₁ = ${verifyMs(1) ? verifyMs(1).toFixed(1) : "?"} ms</span>`;
  f.innerHTML = `<span class="k">E</span>       = (1 − α<sup>γ+1</sup>) / (1 − α)   <span class="faint">expected tokens per round</span><br>${costLine}<br><span class="k">speedup</span> = E / cost`;
  byId("dots-legend").style.display = ex.domain ? "" : "none";
}

/* ------------------------------------------------------------ init */
document.addEventListener("DOMContentLoaded", async () => {
  // simulator
  try {
    const t = await fetchJSON("/api/trace?limit=500");
    sim.rounds = t.rounds || [];
    resetSim();
  } catch (e) {
    byId("narration").textContent = "Recorded trace unavailable (results/mixed_trace.json).";
  }
  byId("sim-play").onclick = () => {
    if (!sim.rounds.length) return;
    if (sim.i >= sim.rounds.length) resetSim();
    sim.playing = !sim.playing;
    byId("sim-play").textContent = sim.playing ? "⏸ Pause" : "▶ Play";
    if (sim.playing && !sim.busy) playLoop();
  };
  byId("sim-step").onclick = () => { sim.playing = false; byId("sim-play").textContent = "▶ Play"; playRound(); };
  byId("sim-reset").onclick = resetSim;
  byId("sim-speed").addEventListener("click", (e) => {
    const b = e.target.closest("button"); if (!b) return;
    $$("#sim-speed button").forEach((x) => x.classList.toggle("on", x === b));
    sim.speed = +b.dataset.v;
  });
  onVisible(byId("sim"), () => { if (sim.rounds.length && !sim.playing && sim.i === 0) byId("sim-play").click(); });

  // explorer
  byId("a-alpha").oninput = (e) => { ex.alpha = +e.target.value; ex.domain = null; markDomain(); drawCurve(); };
  byId("a-r").oninput = (e) => { ex.r = +e.target.value; drawCurve(); };
  $$("[data-r]").forEach((b) => { b.onclick = () => { ex.r = +b.dataset.r; byId("a-r").value = ex.r; drawCurve(); }; });
  byId("cost-model").addEventListener("click", (e) => {
    const b = e.target.closest("button"); if (!b) return;
    $$("#cost-model button").forEach((x) => x.classList.toggle("on", x === b));
    ex.model = b.dataset.v; drawCurve();
  });
  window.addEventListener("resize", () => ex.shown && paint(ex.shown));
  drawCurve();

  try { ex.summary = await fetchJSON("/api/summary"); } catch (e) { return; }
  const g = ex.summary.graphed;
  if (!g) return;
  const chips = byId("dom-chips");
  g.domains.forEach((d) => {
    const b = document.createElement("button");
    b.className = "chip"; b.dataset.dom = d;
    b.innerHTML = `<i class="dot" style="background:${DOMAIN_COLORS[d]}"></i>${d}`;
    b.onclick = () => {
      const a = g.acceptance[d]["1"];
      if (a == null) return;
      ex.domain = d; ex.alpha = Math.min(0.99, a); ex.r = 0.248;
      byId("a-alpha").value = ex.alpha; byId("a-r").value = ex.r;
      markDomain(); drawCurve();
      toast(`${d}: α = ${a.toFixed(3)} (acceptance at γ=1), white dots = measured speedups`);
    };
    chips.append(b);
  });
  drawCurve();
});

function markDomain() {
  $$("#dom-chips .chip").forEach((c) => c.classList.toggle("active", c.dataset.dom === ex.domain));
}
