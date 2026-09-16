/* Results page: every number from /api/summary. */
"use strict";

const R = { s: null, regime: "graphed", metric: "speedup", hist: null };
const gid = (id) => document.getElementById(id);

const CHARTS = [
  ["speedup_graphed.png", "Speedup by domain and γ — CUDA graphs"],
  ["speedup_eager.png", "Speedup by domain and γ — eager"],
  ["acceptance_graphed.png", "Acceptance rate by domain — graphed"],
  ["acceptance_eager.png", "Acceptance rate by domain — eager"],
  ["verify_width.png", "Verify forward cost by width — the step"],
  ["controller_comparison.png", "Controller vs fixed γ vs oracle"],
  ["regime_comparison.png", "Eager vs graphed regimes"],
  ["model_vs_measured_graphed.png", "Cost model vs measurement — graphed"],
  ["model_vs_measured_eager.png", "Cost model vs measurement — eager"],
];

const CONFIG_LABELS = {
  fixed_gamma_1: "fixed γ=1", fixed_gamma_2: "fixed γ=2", fixed_gamma_3: "fixed γ=3",
  fixed_gamma_5: "fixed γ=5", fixed_gamma_8: "fixed γ=8",
  adaptive_linear_cost: "adaptive · linear cost", adaptive_measured_cost: "adaptive · measured cost",
};

function segmentedR(id, fn) {
  gid(id).addEventListener("click", (e) => {
    const b = e.target.closest("button"); if (!b) return;
    $$("button", gid(id)).forEach((x) => x.classList.toggle("on", x === b));
    fn(b.dataset.v);
  });
}

/* ------------------------------------------------------------ heatmap */
function renderHeat() {
  const sw = R.s[R.regime];
  const table = gid("heat");
  if (!sw) { table.innerHTML = `<tr><td class="muted">No ${R.regime} sweep in results/.</td></tr>`; return; }
  const isSpeed = R.metric === "speedup";
  gid("heat-note").innerHTML = isSpeed
    ? `baseline ${sw.baseline_tok_per_s.toFixed(1)} tok/s · ${sw.prompts} prompts · hover a cell`
    : `fraction of draft guesses the target accepted`;
  let html = `<tr><th class="row">domain</th>${sw.gammas.map((g) => `<th>γ=${g}</th>`).join("")}<th>best</th></tr>`;
  sw.domains.forEach((d, row) => {
    html += `<tr><th class="row"><i class="sw" style="background:${DOMAIN_COLORS[d]};margin-right:8px"></i>${DOMAIN_ICONS[d]} ${d}</th>`;
    sw.gammas.forEach((g, col) => {
      const v = (isSpeed ? sw.speedup : sw.acceptance)[d][String(g)];
      const best = isSpeed && sw.best[d].gamma === g;
      const bg = isSpeed ? speedupColor(v) : `rgba(34,211,238,${v == null ? 0.05 : 0.08 + 0.7 * v})`;
      const tip = isSpeed
        ? `${d}, γ=${g}: ${v == null ? "n/a" : v.toFixed(3) + "× — " + (v >= 1 ? `${((v - 1) * 100).toFixed(1)}% faster` : `${((1 - v) * 100).toFixed(1)}% slower`)}`
        : `${d}, γ=${g}: ${v == null ? "n/a" : (100 * v).toFixed(1) + "% of guesses accepted"}`;
      html += `<td class="${best ? "best" : ""}" title="${tip}" style="background:${bg};opacity:0;animation:pop .45s ${(row * 6 + col) * 22}ms both">${v == null ? "–" : isSpeed ? v.toFixed(2) : v.toFixed(2)}</td>`;
    });
    const b = sw.best[d];
    html += `<td style="background:rgba(255,255,255,.04);font-size:13px">${isSpeed ? (b.gamma === 0 ? `<span style="color:var(--rose)">off</span>` : `γ=${b.gamma}`) : ""}</td></tr>`;
  });
  table.innerHTML = html;

  const cards = gid("best-cards");
  cards.innerHTML = sw.domains.map((d) => {
    const b = sw.best[d];
    const off = b.gamma === 0;
    return `<div class="best-card" style="border-color:${DOMAIN_COLORS[d]}55">
      <div class="ico">${DOMAIN_ICONS[d]}</div><div class="s">${d}</div>
      <div class="g" style="color:${off ? "var(--rose)" : DOMAIN_COLORS[d]}">${off ? "don't" : "γ = " + b.gamma}</div>
      <div class="s">${off ? "no γ pays" : b.speedup.toFixed(2) + "×"}</div></div>`;
  }).join("");
}

/* ------------------------------------------------------------ verify width */
function renderVerifyWidth() {
  const vw = R.s.verify_width && R.s.verify_width.widths;
  if (!vw) { gid("vw-note").textContent = "results/verify_width.json not found."; return; }
  const { ctx, w, h } = setupCanvas(gid("vw"), 260);
  const padL = 40, padB = 30, padT = 14;
  const max = Math.ceil(Math.max(...vw.map((x) => x.ms)) / 10) * 10;
  const bw = (w - padL - 10) / vw.length;
  const Y = (ms) => h - padB - (ms / max) * (h - padB - padT);
  ctx.font = "11px system-ui";
  for (let m = 0; m <= max; m += 10) {
    ctx.strokeStyle = COLORS.grid; ctx.beginPath(); ctx.moveTo(padL, Y(m)); ctx.lineTo(w, Y(m)); ctx.stroke();
    ctx.fillStyle = COLORS.faint; ctx.fillText(m + "ms", 2, Y(m) + 4);
  }
  const t0 = performance.now(), dur = REDUCED ? 0 : 900;
  const frame = (t) => {
    const p = dur ? Math.min(1, (t - t0) / dur) : 1;
    ctx.clearRect(padL, 0, w - padL, h - padB + 2);
    for (let m = 0; m <= max; m += 10) { ctx.strokeStyle = COLORS.grid; ctx.beginPath(); ctx.moveTo(padL, Y(m)); ctx.lineTo(w, Y(m)); ctx.stroke(); }
    vw.forEach((x, i) => {
      const e = 1 - Math.pow(1 - Math.min(1, Math.max(0, p * 1.6 - i * 0.05)), 3);
      const top = Y(x.ms * e);
      const color = x.width <= 3 ? COLORS.cyan : x.width === 4 ? COLORS.amber : COLORS.violet;
      ctx.fillStyle = color;
      const bx = padL + i * bw + 3;
      ctx.beginPath();
      if (ctx.roundRect) ctx.roundRect(bx, top, bw - 6, h - padB - top, [5, 5, 0, 0]); else ctx.rect(bx, top, bw - 6, h - padB - top);
      ctx.fill();
      ctx.fillStyle = COLORS.muted; ctx.fillText(String(x.width), bx + bw / 2 - 7, h - padB + 15);
    });
    if (p < 1) requestAnimationFrame(frame);
    else {
      const w3 = vw.find((x) => x.width === 3), w4 = vw.find((x) => x.width === 4);
      if (w3 && w4) {
        const i4 = vw.indexOf(w4), x4 = padL + i4 * bw + bw / 2;
        ctx.fillStyle = COLORS.amber; ctx.font = "bold 12px system-ui";
        ctx.fillText(`+${(w4.ms - w3.ms).toFixed(1)} ms`, x4 - 22, Y(w4.ms) - 8);
      }
      ctx.fillStyle = COLORS.faint; ctx.font = "11px system-ui";
      ctx.fillText("verify width = γ + 1", w - 120, h - 2);
    }
  };
  onVisible(gid("vw"), () => requestAnimationFrame(frame));

  const first = vw[0], w4 = vw.find((x) => x.width === 4), last = vw[vw.length - 1];
  if (w4) gid("vw-note").innerHTML =
    `Width ${w4.width} costs <b>${w4.ms.toFixed(1)} ms</b>; width ${last.width} costs <b>${last.ms.toFixed(1)} ms</b> — only ${(last.ms - w4.ms).toFixed(1)} ms more for ${last.width - w4.width} extra tokens. ` +
    `<span style="color:var(--amber)">γ=3</span> verifies 4 tokens, so it pays the whole step for little gain. Rule here: <b>γ ≤ 2 or γ ≥ 5</b>.`;
}

/* ------------------------------------------------------------ losslessness */
function renderLossless() {
  const parts = ["graphed", "eager"].map((k) => R.s[k]).filter(Boolean);
  const checked = parts.reduce((a, p) => a + p.checked, 0);
  const same = parts.reduce((a, p) => a + p.identical, 0);
  if (!checked) return;
  const frac = same / checked;
  const c = gid("donut");
  const draw = (p) => {
    const { ctx, w, h } = setupCanvas(c, 180);
    const cx = w / 2, cy = h / 2, rad = Math.min(w, h) / 2 - 12;
    ctx.clearRect(0, 0, w, h);
    ctx.lineWidth = 18; ctx.lineCap = "round";
    ctx.strokeStyle = "rgba(251,191,36,.35)";
    ctx.beginPath(); ctx.arc(cx, cy, rad, 0, Math.PI * 2); ctx.stroke();
    ctx.strokeStyle = COLORS.emerald;
    ctx.beginPath(); ctx.arc(cx, cy, rad, -Math.PI / 2, -Math.PI / 2 + Math.PI * 2 * frac * p); ctx.stroke();
    ctx.fillStyle = "#fff"; ctx.font = "800 28px system-ui"; ctx.textAlign = "center";
    ctx.fillText((100 * frac * p).toFixed(1) + "%", cx, cy + 6);
    ctx.fillStyle = COLORS.muted; ctx.font = "11px system-ui"; ctx.fillText("identical", cx, cy + 24);
  };
  onVisible(c, () => {
    const t0 = performance.now(), dur = REDUCED ? 0 : 1200;
    const f = (t) => { const p = dur ? Math.min(1, (t - t0) / dur) : 1; draw(1 - Math.pow(1 - p, 3)); if (p < 1) requestAnimationFrame(f); };
    requestAnimationFrame(f);
  });
  gid("lossless-text").innerHTML = parts.map((p, i) => {
    const name = i === 0 && R.s.graphed ? "graphed" : "eager";
    return `<div style="margin-bottom:10px"><span class="tag ${name === "graphed" ? "info" : ""}">${name}</span>
      <b style="font-size:18px;margin-left:6px;white-space:nowrap">${p.identical} / ${p.checked}</b>
      <span class="muted">(${(100 * p.identical / p.checked).toFixed(1)}%)</span></div>`;
  }).join("") + `<div class="legend"><span><i class="sw" style="background:var(--emerald)"></i>byte-identical</span><span><i class="sw" style="background:rgba(251,191,36,.6)"></i>differs</span></div>`;

  const d = R.s.divergence;
  if (d) {
    const rest = d.mismatches - d.numerically_ambiguous;
    const dnr = d.did_not_reproduce != null
      ? `${d.did_not_reproduce} did not reproduce${d.did_not_reproduce_ids && d.did_not_reproduce_ids.length ? ` (${d.did_not_reproduce_ids.join(", ")} were edited after the sweep)` : ""}.`
      : rest > 0 ? `The other ${rest} could not be re-measured on the current prompt set.` : "";
    gid("div-text").innerHTML = `Re-measured the ${d.mismatches} eager mismatches: <b style="color:var(--text)">${d.numerically_ambiguous}</b> sit exactly on an <span class="term" data-term="fp16tie">fp16 tie</span> (top-2 gap of 0 or 1 <span class="term" data-term="ulp">ULP</span>) and <b style="color:var(--emerald)">${d.decoder_bugs} are decoder bugs</b>. ${dnr}`;
  } else {
    gid("div-text").textContent = "results/divergence_analysis.json not found.";
  }
}

/* ------------------------------------------------------------ controller */
function renderController() {
  const c = R.s.controller;
  if (!c) { gid("ctrl-verdict").textContent = "results/controller_comparison.json not found."; return; }
  const entries = Object.entries(c.configs).sort((a, b) => b[1].mean_speedup - a[1].mean_speedup);
  const lo = 0.8, hi = Math.max(c.oracle || 1.2, ...entries.map((e) => e[1].mean_speedup)) + 0.03;
  const pct = (v) => (100 * (v - lo) / (hi - lo));
  const bestFixed = entries.find(([k]) => k.startsWith("fixed_"));
  let html = "";
  if (c.oracle) html += row("🔮 oracle (hindsight)", c.oracle, "linear-gradient(90deg,#7c7cff,#a78bfa)", true);
  entries.forEach(([k, v]) => {
    const adaptive = k.startsWith("adaptive");
    const color = adaptive ? "linear-gradient(90deg,#fb7185,#fbbf24)"
      : k === (bestFixed && bestFixed[0]) ? "var(--grad)" : "rgba(96,165,250,.7)";
    html += row((adaptive ? "🤖 " : "📌 ") + (CONFIG_LABELS[k] || k), v.mean_speedup, color);
  });
  function row(label, v, color, oracle) {
    return `<div class="ctrl-row"><div${oracle ? ' style="color:var(--violet)"' : ""}>${label}</div>
      <div class="track"><i data-w="${pct(v).toFixed(1)}" style="background:${color}"></i>
        <span class="mark" style="left:${pct(1)}%;background:var(--rose)"></span>
        ${c.oracle ? `<span class="mark" style="left:${pct(c.oracle)}%;border-left:2px dashed var(--violet);background:none"></span>` : ""}</div>
      <div class="v" style="color:${v >= 1 ? "var(--text)" : "var(--rose)"}">${v.toFixed(3)}×</div></div>`;
  }
  gid("ctrl-rows").innerHTML = html;
  onVisible(gid("ctrl-card"), () => $$("#ctrl-rows .track > i").forEach((i, k) => setTimeout(() => { i.style.width = i.dataset.w + "%"; }, k * 90)));

  const bestAdaptive = entries.find(([k]) => k.startsWith("adaptive"));
  if (bestFixed && c.oracle) {
    gid("ctrl-verdict").innerHTML = `<b style="color:var(--text)">${CONFIG_LABELS[bestFixed[0]]}</b> reaches ${(100 * bestFixed[1].mean_speedup / c.oracle).toFixed(0)}% of the oracle, leaving only <b style="color:var(--text)">${(100 * (c.oracle / bestFixed[1].mean_speedup - 1)).toFixed(1)}%</b> headroom for any adaptive policy.` +
      (bestAdaptive ? ` The best controller (${CONFIG_LABELS[bestAdaptive[0]]}, ${bestAdaptive[1].mean_speedup.toFixed(3)}×) loses because its lag and time spent at γ=3–4 cost more than that.` : "");
  }

  // histograms
  const withHist = entries.filter(([, v]) => v.gamma_histogram);
  const pick = gid("hist-pick");
  pick.innerHTML = withHist.map(([k], i) => `<button data-v="${k}" class="${i === 0 ? "on" : ""}">${k.includes("linear") ? "linear" : "measured"}</button>`).join("");
  if (withHist.length) R.hist = withHist[0][0];
  segmentedR("hist-pick", (k) => { R.hist = k; renderHist(k); });
  if (withHist.length) onVisible(gid("hist"), () => renderHist(R.hist));
}

function renderHist(key) {
  const h = R.s.controller.configs[key].gamma_histogram;
  const total = Object.values(h).reduce((a, b) => a + b, 0);
  const max = Math.max(...Object.values(h));
  const gs = Array.from({ length: 9 }, (_, i) => i);
  gid("hist").innerHTML = gs.map((g) => {
    const n = h[String(g)] || 0;
    const color = g === 0 ? COLORS.rose : g === 3 || g === 4 ? COLORS.amber : g <= 2 ? COLORS.cyan : COLORS.violet;
    return `<div title="γ=${g}: ${n} rounds (${(100 * n / total).toFixed(1)}%)"><span>${n ? (100 * n / total).toFixed(0) + "%" : ""}</span><i data-h="${(100 * n / max).toFixed(1)}" style="background:${color}"></i><span>γ${g}</span></div>`;
  }).join("");
  requestAnimationFrame(() => requestAnimationFrame(() => $$("#hist i").forEach((i) => { i.style.height = `max(${+i.dataset.h > 0 ? 3 : 0}px, ${(i.dataset.h * 0.72).toFixed(1)}%)`; })));
  const share = (keys) => keys.reduce((a, g) => a + (h[String(g)] || 0), 0) / total * 100;
  gid("hist-note").innerHTML = key.includes("linear")
    ? `Spends <b style="color:var(--amber)">${share([3, 4]).toFixed(0)}%</b> of rounds at γ=3–4 — the pessimal band. The linear cost model cannot see the verify step.`
    : `Spends <b style="color:var(--rose)">${share([0]).toFixed(0)}%</b> of rounds with speculation off (γ=0). While off it collects no acceptance evidence, so it recovers only through periodic probes.`;
}

/* ------------------------------------------------------------ gallery */
function renderGallery() {
  const g = gid("gallery");
  g.innerHTML = CHARTS.map(([f, cap]) => `<figure class="card hover-lift" style="margin:0;padding:12px">
    <img loading="lazy" src="/charts/${f}" alt="${cap}" data-full="/charts/${f}" onerror="this.closest('figure').remove()">
    <figcaption>${cap}</figcaption></figure>`).join("");
  const lb = gid("lightbox"), scrim = gid("lb-scrim"), img = gid("lb-img");
  const close = () => { lb.classList.remove("show"); scrim.classList.remove("show"); };
  g.addEventListener("click", (e) => {
    const i = e.target.closest("img"); if (!i) return;
    img.src = i.dataset.full; img.alt = i.alt;
    lb.classList.add("show"); scrim.classList.add("show");
  });
  lb.onclick = close; scrim.onclick = close;
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") close(); });
}

/* ------------------------------------------------------------ init */
document.addEventListener("DOMContentLoaded", async () => {
  renderGallery();
  try { R.s = await fetchJSON("/api/summary"); } catch (e) { toast("Could not load results: " + e.message, 5000); return; }
  const runs = ["graphed", "eager"].reduce((a, k) => a + (R.s[k] ? R.s[k].prompts * (R.s[k].gammas.length + 1) : 0), 0);
  gid("n-runs").textContent = runs.toLocaleString();
  segmentedR("regime", (v) => { R.regime = v; renderHeat(); });
  segmentedR("metric", (v) => { R.metric = v; renderHeat(); });
  renderHeat();
  renderVerifyWidth();
  renderLossless();
  renderController();
  window.addEventListener("resize", () => { renderVerifyWidth(); });
});
