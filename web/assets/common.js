/* The Crossover Point -- shared site behaviour: nav, help drawer, glossary
   tooltips, count-up numbers, scroll reveal. Vanilla JS, no dependencies. */
"use strict";

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));
const REDUCED = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

const COLORS = {
  accept: "#34d399", correct: "#fbbf24", bonus: "#a78bfa",
  indigo: "#7c7cff", cyan: "#22d3ee", emerald: "#34d399", amber: "#fbbf24",
  rose: "#fb7185", violet: "#a78bfa", sky: "#60a5fa",
  muted: "#98a3bb", faint: "#5f6a82", grid: "rgba(255,255,255,0.07)",
};
const DOMAIN_COLORS = {
  structured: "#60a5fa", code: "#34d399", math: "#a78bfa",
  reasoning: "#fbbf24", prose: "#fb7185", translation: "#22d3ee",
};
const DOMAIN_ICONS = {
  structured: "🗂️", code: "💻", math: "➗", reasoning: "🧠", prose: "✍️", translation: "🌐",
};

/* ------------------------------------------------------------------ glossary */
const GLOSSARY = [
  // --- the idea
  { id: "specdec", cat: "The idea", term: "Speculative decoding",
    def: "A way to make a big language model answer faster without changing what it says. A small, fast model guesses the next few words; the big model checks all the guesses at once and keeps the ones it agrees with.",
    ex: "Like a junior writer drafting a sentence and a senior editor approving it in one read, instead of the editor writing every word." },
  { id: "target", cat: "The idea", term: "Target model (7B)",
    def: "The big, accurate model whose answer we want: Qwen2.5-Coder-7B, compressed to 4-bit. It has the final say on every token.",
    ex: "It is the authority. If it disagrees with a guess, its choice wins." },
  { id: "draft", cat: "The idea", term: "Draft model (0.5B)",
    def: "The small, fast model that proposes tokens: Qwen2.5-Coder-0.5B. It never decides anything; it only guesses. It shares the exact same vocabulary as the target.",
  },
  { id: "token", cat: "The idea", term: "Token",
    def: "The unit a language model reads and writes: a word, part of a word, a space or a symbol. \" hash\" and \" map\" are two tokens.",
  },
  { id: "gamma", cat: "The idea", term: "γ (gamma)",
    def: "How many tokens the draft guesses per round. γ=0 means no speculation at all (plain decoding). Larger γ can win bigger but wastes more work when a guess is wrong.",
    ex: "γ=2: draft guesses 2 tokens, target checks 3 positions in one pass." },
  { id: "round", cat: "The idea", term: "Round",
    def: "One cycle of: draft guesses γ tokens → target verifies them in one forward pass → keep the agreed prefix + one token from the target. Every round produces at least one token.",
  },
  // --- token colours
  { id: "accept", cat: "Token colours", term: "Accepted token", color: "accept",
    def: "Green. The draft guessed this token and the target agreed. This is where speed comes from: the target confirmed it without generating it one step at a time.",
  },
  { id: "correct", cat: "Token colours", term: "Corrected token", color: "correct",
    def: "Amber. The draft guessed something else; the target overrode it with its own choice. Everything the draft guessed after this point is thrown away. Hover one in the demo to see what the draft guessed.",
  },
  { id: "bonus", cat: "Token colours", term: "Bonus token", color: "bonus",
    def: "Violet. When the target accepts ALL γ guesses, its verify pass already computed the next token too, so we get one extra token for free.",
  },
  // --- measurements
  { id: "acceptance", cat: "Measurements", term: "Acceptance rate (α)",
    def: "The fraction of draft guesses the target agreed with. 0.85 means 85% of guessed tokens were kept. Predictable text (JSON, code, maths) scores high; free prose and translation score low.",
  },
  { id: "tpr", cat: "Measurements", term: "Tokens per round",
    def: "Average tokens produced by each round. Plain decoding gives exactly 1. Speculation with γ=2 gives between 1 and 3.",
  },
  { id: "tps", cat: "Measurements", term: "tok/s",
    def: "Tokens generated per second, end to end (prompt processing + generation). Higher is faster.",
  },
  { id: "baseline", cat: "Measurements", term: "Baseline",
    def: "Ordinary greedy decoding with the target model alone, no draft. Every speedup is measured against the baseline on the SAME prompt in the SAME regime.",
  },
  { id: "speedup", cat: "Measurements", term: "Speedup (×)",
    def: "Speculative tok/s divided by baseline tok/s. 1.20× means 20% faster. Below 1.00× means speculation made it SLOWER.",
    ex: "Only compare within one regime: graphed and eager differ ~2× by themselves." },
  { id: "domain", cat: "Measurements", term: "Domain",
    def: "The kind of prompt. The evaluation uses 210 prompts, 35 in each of six domains: structured (JSON/YAML/tables), code, math, reasoning, prose and translation.",
  },
  // --- economics
  { id: "r", cat: "The economics", term: "Cost ratio r",
    def: "How expensive one draft step is compared with one target step. r = 0.25 means the draft costs a quarter of the target. The smaller r is, the more guessing pays.",
    ex: "Measured here: r = 0.64 eager, r = 0.248 with CUDA graphs." },
  { id: "crossover", cat: "The economics", term: "Crossover point",
    def: "The γ beyond which guessing more tokens costs more time than it saves. Past it, speculation is net-negative. The whole project is about finding where that point sits, per domain and per regime.",
  },
  { id: "breakeven", cat: "The economics", term: "Break-even (1.0×)",
    def: "The line where speculation and plain decoding are equally fast. Above it speculation helps; below it you should switch speculation off.",
  },
  { id: "verifystep", cat: "The economics", term: "Verify-cost step",
    def: "Measured on this GPU: verifying 1–3 tokens grows steeply, jumps at 4 tokens, then stays almost flat up to 13. So γ=3 (4 tokens) pays the whole jump but only checks 4 tokens — the worst choice.",
    ex: "Rule of thumb here: use γ ≤ 2 or γ ≥ 5, never 3 or 4." },
  { id: "expected", cat: "The economics", term: "Expected tokens E",
    def: "Tokens per round predicted from α and γ: E = (1 − α^(γ+1)) / (1 − α). It saturates: past a point, extra guesses almost never survive.",
  },
  // --- engineering
  { id: "regime", cat: "Engineering", term: "Regime (graphed / eager)",
    def: "How the model is executed. Eager runs each GPU operation one by one from Python. Graphed replays a pre-recorded CUDA graph. Same output, very different speed.",
  },
  { id: "cudagraphs", cat: "Engineering", term: "CUDA graphs",
    def: "Record a model's ~940 GPU kernel launches once, then replay them with a single call. On this Windows machine each launch costs ~13 µs, so graphs make the baseline 1.85× faster with no speculation at all.",
  },
  { id: "hostbound", cat: "Engineering", term: "Host-bound",
    def: "When the GPU sits idle waiting for the CPU to send it work. Measured here: a 0.5B draft step spends ~5 ms on GPU and ~24 ms in launch overhead, so a small model is not much cheaper than a big one.",
  },
  { id: "kvcache", cat: "Engineering", term: "KV cache",
    def: "Memory of what the model has already read, so it does not re-process the whole text each step. When a guess is rejected, the cache is rolled back to erase it.",
  },
  { id: "controller", cat: "Engineering", term: "Adaptive controller",
    def: "Tunes γ live from the acceptance it observes, and can switch speculation off (γ=0). Honest result: on this workload it did NOT beat a fixed γ=2 (1.102× vs 1.117×).",
  },
  { id: "oracle", cat: "Engineering", term: "Oracle",
    def: "A cheating reference that knows each prompt's domain in advance and picks that domain's best fixed γ with hindsight (1.182×). No real controller can beat it; it shows the maximum possible gain from adapting.",
  },
  // --- correctness
  { id: "lossless", cat: "Correctness", term: "Lossless",
    def: "Speculation must produce the exact same text as the target alone. Checked on every one of 2,520 sweep runs: 96.7–96.8% byte-identical, and every checked difference is an fp16 tie, not a bug.",
  },
  { id: "fp16tie", cat: "Correctness", term: "fp16 tie",
    def: "When the target's top two candidate tokens have scores equal to within one step of 16-bit float precision. There, even plain decoding's pick depends on the order the GPU adds numbers, so 'the correct output' is not well defined.",
    ex: "Example: '   ' vs ' ' in TOML indentation, both scored 26.1094." },
  { id: "ulp", cat: "Correctness", term: "ULP",
    def: "Unit in the last place: the smallest representable difference between two floating-point numbers at a given size. At magnitude 26, one fp16 ULP is 0.015625.",
  },
  // --- demo
  { id: "replay", cat: "Using the demo", term: "Replay",
    def: "Plays back a real recorded generation (prose + code, 1.40× over its baseline, output identical) without using the GPU. Works even when the models are not loaded.",
  },
  { id: "race", cat: "Using the demo", term: "Race",
    def: "Runs the baseline, then speculation, on the same prompt, and compares the two outputs character by character plus their speed. They run one after the other because there is one GPU.",
  },
];
const GLOSSARY_BY_ID = Object.fromEntries(GLOSSARY.map((g) => [g.id, g]));

/* -------------------------------------------------------------- page chrome */
function buildChrome() {
  const page = document.body.dataset.page || "";
  const links = [["/", "Home", "home"], ["/demo", "Live demo", "demo"],
                 ["/how", "How it works", "how"], ["/results", "Results", "results"]];

  const nav = document.createElement("nav");
  nav.className = "nav";
  nav.innerHTML = `<div class="container">
    <a class="brand" href="/"><span class="brand-mark">⚡</span><span>The Crossover Point</span></a>
    <div class="nav-links">${links.map(([href, label, key]) =>
      `<a href="${href}" class="${key === page ? "active" : ""}">${label}</a>`).join("")}</div>
    <button class="btn ghost sm" id="help-open" title="Glossary (press ?)">? Help</button>
  </div>`;
  document.body.prepend(nav);

  const bg = document.createElement("div");
  bg.className = "bg-orbs";
  bg.innerHTML = '<div class="orb a"></div><div class="orb b"></div><div class="orb c"></div>';
  document.body.prepend(bg);

  const footer = document.createElement("footer");
  footer.innerHTML = `<div class="container">
    <span>The Crossover Point · speculative decoding measured on one RTX 3060 · Windows</span>
    <span>Speculative decoding is published work (Leviathan et al. 2022, Chen et al. 2023). This project measures it.</span>
  </div>`;
  document.body.append(footer);

  // help drawer
  const scrim = document.createElement("div");
  scrim.className = "scrim";
  const drawer = document.createElement("aside");
  drawer.className = "drawer";
  drawer.setAttribute("aria-label", "Glossary");
  drawer.innerHTML = `
    <div class="drawer-head">
      <button class="btn ghost sm close" aria-label="Close">✕</button>
      <div class="eyebrow">Help</div>
      <h3 style="font-size:22px;margin:4px 0 6px">Every term on this site</h3>
      <p class="muted" style="margin:0 0 12px;font-size:14px">Hover any <span class="info-i">i</span> or dotted term for a quick definition. Press <kbd>?</kbd> anytime to open this.</p>
      <input class="input" id="gl-search" placeholder="Search terms… e.g. gamma, bonus, tie" autocomplete="off">
    </div>
    <div class="drawer-body" id="gl-list"></div>`;
  const fab = document.createElement("button");
  fab.className = "help-fab";
  fab.textContent = "?";
  fab.title = "What do these terms mean?";
  document.body.append(scrim, drawer, fab);

  const pop = document.createElement("div");
  pop.className = "pop";
  document.body.append(pop);

  const toast = document.createElement("div");
  toast.className = "toast";
  document.body.append(toast);

  const renderList = (q = "") => {
    q = q.trim().toLowerCase();
    let html = "", lastCat = "";
    for (const g of GLOSSARY) {
      const hay = (g.term + " " + g.def + " " + (g.ex || "")).toLowerCase();
      if (q && !hay.includes(q)) continue;
      if (g.cat !== lastCat) { html += `<div class="gl-cat">${g.cat}</div>`; lastCat = g.cat; }
      const sw = g.color ? `<i class="sw" style="background:${COLORS[g.color]}"></i>` : "";
      html += `<div class="glossary-item" id="gl-${g.id}"><h4>${sw}${g.term}</h4><p>${g.def}</p>${g.ex ? `<div class="ex">💡 ${g.ex}</div>` : ""}</div>`;
    }
    $("#gl-list").innerHTML = html || `<p class="muted">No term matches “${q}”.</p>`;
  };
  renderList();

  const open = (focusId) => {
    scrim.classList.add("show"); drawer.classList.add("show");
    if (focusId) {
      $("#gl-search").value = ""; renderList();
      const el = $("#gl-" + focusId);
      if (el) {
        el.scrollIntoView({ block: "center", behavior: REDUCED ? "auto" : "smooth" });
        el.style.background = "rgba(34,211,238,.08)";
        setTimeout(() => { el.style.background = ""; }, 1600);
      }
    } else {
      setTimeout(() => $("#gl-search").focus(), 200);
    }
  };
  const close = () => { scrim.classList.remove("show"); drawer.classList.remove("show"); };
  window.openHelp = open;
  $("#help-open").onclick = () => open();
  fab.onclick = () => open();
  scrim.onclick = close;
  $(".close", drawer).onclick = close;
  $("#gl-search").oninput = (e) => renderList(e.target.value);
  document.addEventListener("keydown", (e) => {
    const typing = /INPUT|TEXTAREA|SELECT/.test(document.activeElement.tagName);
    if (e.key === "Escape") close();
    if (e.key === "?" && !typing) { e.preventDefault(); open(); }
  });

  window.toast = (msg, ms = 2600) => {
    toast.textContent = msg; toast.classList.add("show");
    clearTimeout(toast._t); toast._t = setTimeout(() => toast.classList.remove("show"), ms);
  };

  // term tooltips: any element with data-term="id"
  const showPop = (el) => {
    const g = GLOSSARY_BY_ID[el.dataset.term];
    if (!g) return;
    pop.innerHTML = `<b>${g.term}</b>${g.def}${g.ex ? `<span class="ex">💡 ${g.ex}</span>` : ""}<span class="ex" style="color:var(--cyan)">click for full glossary</span>`;
    const r = el.getBoundingClientRect();
    pop.classList.add("show");
    const pw = pop.offsetWidth, ph = pop.offsetHeight;
    let x = Math.min(window.innerWidth - pw - 12, Math.max(12, r.left + r.width / 2 - pw / 2));
    let y = r.top - ph - 10;
    if (y < 70) y = r.bottom + 10;
    pop.style.left = x + "px"; pop.style.top = y + "px";
  };
  document.addEventListener("mouseover", (e) => {
    const el = e.target.closest("[data-term]");
    if (el) showPop(el);
  });
  document.addEventListener("mouseout", (e) => {
    if (e.target.closest("[data-term]")) pop.classList.remove("show");
  });
  document.addEventListener("click", (e) => {
    const el = e.target.closest("[data-term]");
    if (el && !el.closest("button, a")) { pop.classList.remove("show"); open(el.dataset.term); }
  });
}

/* Replace <i data-info="id"></i> placeholders with the small ⓘ badge. */
function decorateInfo(root = document) {
  $$("[data-info]", root).forEach((el) => {
    el.classList.add("info-i");
    el.dataset.term = el.dataset.info;
    el.textContent = "i";
    el.setAttribute("tabindex", "0");
    el.removeAttribute("data-info");
  });
}

/* ------------------------------------------------------------- animations */
function countUp(el, to, { decimals = 2, suffix = "", prefix = "", ms = 1400 } = {}) {
  if (to == null || isNaN(to)) { el.textContent = "–"; return; }
  if (REDUCED) { el.textContent = prefix + to.toFixed(decimals) + suffix; return; }
  const from = parseFloat(el.dataset.cur || "0");
  const t0 = performance.now();
  const step = (t) => {
    const p = Math.min(1, (t - t0) / ms);
    const e = 1 - Math.pow(1 - p, 3);
    const v = from + (to - from) * e;
    el.textContent = prefix + v.toFixed(decimals) + suffix;
    if (p < 1) requestAnimationFrame(step); else el.dataset.cur = String(to);
  };
  requestAnimationFrame(step);
}

function setupReveal() {
  const els = $$(".reveal");
  if (REDUCED || !("IntersectionObserver" in window)) {
    els.forEach((el) => el.classList.add("in"));
    return;
  }
  const io = new IntersectionObserver((entries) => {
    entries.forEach((en) => {
      if (en.isIntersecting) {
        en.target.classList.add("in");
        en.target.dispatchEvent(new CustomEvent("reveal"));
        io.unobserve(en.target);
      }
    });
  }, { threshold: 0.15 });
  els.forEach((el) => io.observe(el));
  $$(".stagger").forEach((p) => Array.from(p.children).forEach((c, i) => c.style.setProperty("--i", i)));
}

/* Run fn once when el scrolls into view (or immediately without observers). */
function onVisible(el, fn) {
  if (!el) return;
  if (!("IntersectionObserver" in window)) { fn(); return; }
  const io = new IntersectionObserver((ents) => {
    if (ents.some((e) => e.isIntersecting)) { io.disconnect(); fn(); }
  }, { threshold: 0.2 });
  io.observe(el);
}

/* HiDPI canvas setup; returns {ctx, w, h} in CSS pixels. */
function setupCanvas(canvas, cssHeight) {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth || canvas.parentElement.clientWidth;
  const h = cssHeight || canvas.clientHeight || 200;
  canvas.style.height = h + "px";
  canvas.width = Math.round(w * dpr);
  canvas.height = Math.round(h * dpr);
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, w, h };
}

async function fetchJSON(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url} → HTTP ${res.status}`);
  return res.json();
}

/* Colour a speedup value: red below 1, amber near 1, green/cyan above. */
function speedupColor(v, alpha = 1) {
  if (v == null) return `rgba(255,255,255,${0.05 * alpha})`;
  const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
  if (v < 1) {
    const t = clamp((1 - v) / 0.4, 0, 1);
    return `rgba(251,113,133,${(0.15 + 0.65 * t) * alpha})`;
  }
  const t = clamp((v - 1) / 0.45, 0, 1);
  return `rgba(52,211,153,${(0.12 + 0.7 * t) * alpha})`;
}

function escapeHTML(s) {
  return String(s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

document.addEventListener("DOMContentLoaded", () => {
  buildChrome();
  decorateInfo();
  setupReveal();
});
