/* QEC Primitives Benchmark — renders site/data/results.json (built by scripts/build_site_data.py). No dependencies. */
(function () {
  "use strict";
  const $ = (s, el = document) => el.querySelector(s);
  const NS = "http://www.w3.org/2000/svg";
  const SLOTS = ["var(--s1)", "var(--s2)", "var(--s3)"];
  const MAX_PICK = 3;
  const FRAME_COLOR = { Z: "var(--s1)", X: "var(--s2)", XZ: "var(--s3)" };
  const BASIS = { Z: { label: "|0⟩L", color: "var(--s1)" }, X: { label: "|+⟩L", color: "var(--s2)" } };

  /* ---------- theme ---------- */
  const root = document.documentElement;
  try { const t = localStorage.getItem("theme"); if (t) root.dataset.theme = t; } catch (e) {}
  $("#themeBtn").addEventListener("click", () => {
    const dark = root.dataset.theme ? root.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
    root.dataset.theme = dark ? "light" : "dark";
    try { localStorage.setItem("theme", root.dataset.theme); } catch (e) {}
  });

  /* ---------- formatting ---------- */
  const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
  const fDate = (s) => { if (!s) return "–"; const [y, m, d] = s.split("-").map(Number); return `${d} ${MONTHS[m - 1]} ${y}`; };
  const fShort = (s) => { const [, m, d] = s.split("-").map(Number); return `${d} ${MONTHS[m - 1]}`; };
  const fInt = (n) => (n == null ? "–" : Math.round(n).toLocaleString("en-US"));
  const f3 = (n) => (n == null ? "–" : n.toFixed(3));
  const f4 = (n) => (n == null ? "–" : n < 0.01 ? n.toFixed(4) : n.toFixed(3));
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const ok = (v) => v == null ? `<span class="st na">– no reference</span>` : v > 0 ? `<span class="st ok">✓ above random + 3σ</span>` : `<span class="st no">✗ not above random + 3σ</span>`;
  const sittingTime = (s) => (s.length >= 13 ? `${s.slice(9, 11)}:${s.slice(11, 13)}` : "");
  const codeLabel = (c) => {
    const m = c.match(/^(surface|color)_d(\d+)$/);
    return m ? `${m[1] === "surface" ? "Surface" : "Colour"} d = ${m[2]}` : c;
  };

  /* ---------- tooltip ---------- */
  const tip = $("#tip");
  function showTip(html, x, y) {
    tip.innerHTML = html; tip.hidden = false;
    const w = tip.offsetWidth, h = tip.offsetHeight;
    let left = x + 14, top = y + 14;
    if (left + w > innerWidth - 8) left = x - w - 14;
    if (top + h > innerHeight - 8) top = y - h - 14;
    tip.style.left = Math.max(8, left) + "px"; tip.style.top = Math.max(8, top) + "px";
  }
  const hideTip = () => { tip.hidden = true; };

  /* ---------- scales & ticks ---------- */
  function niceTicks(a, b, n = 5) {
    const step0 = (b - a) / n, mag = 10 ** Math.floor(Math.log10(step0)), e = step0 / mag;
    const step = (e >= 7.5 ? 10 : e >= 3.5 ? 5 : e >= 1.5 ? 2 : 1) * mag, out = [];
    for (let v = Math.ceil(a / step - 1e-9) * step; v <= b + 1e-9; v += step) out.push(+v.toFixed(10));
    return out;
  }
  function logTicks(a, b) {
    const lo = Math.floor(Math.log10(a)), hi = Math.ceil(Math.log10(b)), mults = hi - lo <= 2 ? [1, 2, 5] : [1, 3], out = [];
    for (let e = lo; e <= hi; e++) for (const m of mults) { const v = m * 10 ** e; if (v >= a * 0.999 && v <= b * 1.001) out.push(v); }
    return out;
  }
  const el = (tag, attrs = {}, parent) => {
    const n = document.createElementNS(NS, tag);
    for (const k in attrs) n.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(n);
    return n;
  };

  /* ---------- generic line chart ----------
     series: [{ key, label, color (null = context), dash, band, pts: [{x, y, lo, hi, hollow, ...}] }]
     cfg: xLog, yLog, xTicks (explicit), xFmt, yFmt, yDomain, refs [{y, label}], tip(s, p), onPoint(s, p, ev) */
  function lineChart(host, cfg) {
    const draw = () => {
      host.innerHTML = "";
      const W = host.clientWidth, H = host.clientHeight;
      if (!W || !H) return;
      const m = { t: 12, r: 18, b: 44, l: 56 };
      const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": cfg.aria || "" }, host);
      const all = cfg.series.flatMap((s) => s.pts.filter((p) => p.y != null));
      if (!all.length) { el("text", { x: W / 2, y: H / 2, "text-anchor": "middle", class: "empty" }, svg).textContent = "No runs to show"; return; }
      const ys = all.flatMap((p) => [p.y, p.lo, p.hi]).filter((v) => v != null && (!cfg.yLog || v > 0));
      let [x0, x1] = cfg.xDomain || [Math.min(...all.map((p) => p.x)), Math.max(...all.map((p) => p.x))];
      let [y0, y1] = cfg.yDomain || [Math.min(...ys), Math.max(...ys)];
      (cfg.refs || []).forEach((r) => { y0 = Math.min(y0, r.y); y1 = Math.max(y1, r.y); });
      if (cfg.xLog) { x0 /= 1.15; x1 *= 1.15; } else { const pad = (x1 - x0 || 1) * 0.05; x0 -= pad; x1 += pad; }
      if (cfg.yLog) { y0 /= 1.3; y1 *= 1.3; } else { const yPad = (y1 - y0 || 0.1) * 0.08; y0 -= yPad; y1 += yPad; }
      const lin = (v, a, b, A, B) => A + (v - a) / (b - a) * (B - A);
      const X = cfg.xLog ? (v) => lin(Math.log10(v), Math.log10(x0), Math.log10(x1), m.l, W - m.r) : (v) => lin(v, x0, x1, m.l, W - m.r);
      const Y = cfg.yLog ? (v) => lin(Math.log10(Math.max(v, y0)), Math.log10(y0), Math.log10(y1), H - m.b, m.t) : (v) => lin(v, y0, y1, H - m.b, m.t);

      const grid = el("g", { class: "grid" }, svg), axis = el("g", { class: "axis" }, svg);
      const yt = cfg.yLog ? logTicks(y0 * 1.3, y1 / 1.3) : niceTicks(y0, y1, Math.max(3, Math.round((H - m.t - m.b) / 60)));
      yt.forEach((v) => {
        el("line", { x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v) }, grid);
        el("text", { x: m.l - 8, y: Y(v) + 4, "text-anchor": "end" }, axis).textContent = cfg.yFmt ? cfg.yFmt(v) : (cfg.yLog ? String(+v.toPrecision(2)) : v.toFixed(2));
      });
      let xt = cfg.xTicks ? cfg.xTicks.filter((v) => v >= x0 && v <= x1) : (cfg.xLog ? logTicks(x0 * 1.15, x1 / 1.15) : niceTicks(x0, x1, Math.max(3, Math.round((W - m.l - m.r) / 80))));
      if (cfg.xInteger) xt = xt.filter((v) => Number.isInteger(v));
      xt.forEach((v) => {
        el("line", { x1: X(v), x2: X(v), y1: m.t, y2: H - m.b }, grid);
        el("text", { x: X(v), y: H - m.b + 17, "text-anchor": "middle" }, axis).textContent = cfg.xFmt ? cfg.xFmt(v) : fInt(v);
      });
      el("text", { x: (m.l + W - m.r) / 2, y: H - 6, "text-anchor": "middle", class: "title" }, axis).textContent = cfg.xLabel;
      el("text", { x: -(m.t + H - m.b) / 2, y: 14, transform: "rotate(-90)", "text-anchor": "middle", class: "title" }, axis).textContent = cfg.yLabel;
      (cfg.refs || []).forEach((r) => el("line", { x1: m.l, x2: W - m.r, y1: Y(r.y), y2: Y(r.y), class: "ref" }, svg));
      const labels = el("g", {}, svg);

      const ordered = [...cfg.series.filter((s) => !s.color), ...cfg.series.filter((s) => s.color)];
      const hit = [];
      ordered.forEach((s) => {
        const g = el("g", {}, svg), emph = !!s.color, col = s.color || "var(--ctx)";
        const pts = s.pts.filter((p) => p.y != null).sort((a, b) => a.x - b.x);
        if (s.band) {
          const b = pts.filter((p) => p.lo != null && p.hi != null);
          if (b.length > 1) el("path", { class: "iqr", fill: col, d: b.map((p, i) => `${i ? "L" : "M"}${X(p.x)},${Y(p.hi)}`).join("") + [...b].reverse().map((p) => `L${X(p.x)},${Y(p.lo)}`).join("") + "Z" }, g);
        }
        if (pts.length > 1) el("path", { d: pts.map((p, i) => `${i ? "L" : "M"}${X(p.x).toFixed(1)},${Y(p.y).toFixed(1)}`).join(""), fill: "none", stroke: col, "stroke-width": emph ? 2 : 1.2, "stroke-dasharray": s.dash === true ? "5 4" : s.dash || "none", "stroke-linejoin": "round", "stroke-linecap": "round" }, g);
        pts.forEach((p) => {
          const cx = X(p.x), cy = Y(p.y);
          if ((emph && !s.noDots) || pts.length === 1) {
            el("circle", { cx, cy, r: emph ? 4.5 : 3, fill: p.hollow ? "var(--card)" : col, stroke: p.hollow ? col : "var(--card)", "stroke-width": 2 }, g);
          }
          if (!s.noHit) hit.push({ cx, cy, s, p, emph });
        });
      });
      svg.appendChild(labels);
      (cfg.refs || []).forEach((r) => { el("text", { x: m.l + 6, y: Y(r.y) - 6, class: "ref-label" }, labels).textContent = r.label; });
      const ring = el("circle", { r: 7, fill: "none", stroke: "var(--ink)", "stroke-width": 1.5, visibility: "hidden" }, svg);
      const nearest = (ev) => {
        const b = svg.getBoundingClientRect(), mx = (ev.clientX - b.left) * W / b.width, my = (ev.clientY - b.top) * H / b.height;
        let best = null, bd = 26 * 26;
        for (const h of hit) { const d = (h.cx - mx) ** 2 + (h.cy - my) ** 2 - (h.emph ? 60 : 0); if (d < bd) { bd = d; best = h; } }
        return best;
      };
      svg.addEventListener("pointermove", (ev) => {
        const h = nearest(ev);
        if (!h) { ring.setAttribute("visibility", "hidden"); hideTip(); svg.style.cursor = ""; return; }
        ring.setAttribute("cx", h.cx); ring.setAttribute("cy", h.cy); ring.setAttribute("visibility", "visible");
        svg.style.cursor = cfg.onPoint ? "pointer" : "";
        showTip(cfg.tip(h.s, h.p), ev.clientX, ev.clientY);
      });
      svg.addEventListener("pointerleave", () => { ring.setAttribute("visibility", "hidden"); hideTip(); });
      svg.addEventListener("click", (ev) => {
        const h = nearest(ev);
        if (h && cfg.onPoint) { ev.stopPropagation(); hideTip(); cfg.onPoint(h.s, h.p, ev); }
      });
    };
    draw();
    if (!host._observed) {
      let w = host.clientWidth;
      new ResizeObserver(() => { if (host.clientWidth !== w) { w = host.clientWidth; host._redraw && host._redraw(); } }).observe(host);
      host._observed = true;
    }
    host._redraw = draw;
  }

  /* ---------- selection with stable colors (color follows the entity) ---------- */
  function Picker(onChange) {
    const slot = new Map();
    return {
      has: (k) => slot.has(k),
      color: (k) => (slot.has(k) ? SLOTS[slot.get(k)] : null),
      keys: () => [...slot.keys()],
      set(keys) { slot.clear(); keys.slice(0, MAX_PICK).forEach((k, i) => slot.set(k, i)); },
      toggle(k) {
        if (slot.has(k)) slot.delete(k);
        else {
          if (slot.size >= MAX_PICK) slot.delete(slot.keys().next().value);
          const used = new Set(slot.values()); let i = 0; while (used.has(i)) i++;
          slot.set(k, i);
        }
        onChange();
      },
    };
  }
  function renderChips(host, items, picker) {
    host.innerHTML = "";
    items.forEach(({ key, label }) => {
      const b = document.createElement("button");
      b.type = "button"; b.className = "chip"; b.setAttribute("aria-pressed", picker.has(key));
      b.innerHTML = `<i style="background:${picker.color(key) || "var(--ctx)"}"></i>${esc(label)}`;
      b.addEventListener("click", () => picker.toggle(key));
      host.appendChild(b);
    });
  }
  function seg(host, options, value, onPick) {
    host.innerHTML = options.map((o) => `<button type="button" role="tab" aria-selected="${o.key === value}" data-k="${esc(o.key)}">${esc(o.label)}</button>`).join("");
    host.querySelectorAll("button").forEach((b) => b.addEventListener("click", () => onPick(b.dataset.k)));
  }
  function guard(sel, fn) {
    try { fn(); } catch (err) {
      console.error(err);
      const note = document.createElement("p");
      note.className = "note";
      note.textContent = "This section could not be drawn. Reload the page (Shift + reload) to fetch the latest version.";
      $(sel + " .wrap").appendChild(note);
    }
  }

  /* ---------- detail popup ---------- */
  const pop = $("#pop");
  const closeDetail = () => { if (!pop.hidden) { pop.hidden = true; hideTip(); } };
  $("#popClose").addEventListener("click", closeDetail);
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeDetail(); });
  document.addEventListener("pointerdown", (e) => { if (!pop.contains(e.target)) closeDetail(); });
  addEventListener("scroll", closeDetail, { passive: true });
  function openPop(cx, cy) {
    pop.hidden = false;
    const w = pop.offsetWidth, h = pop.offsetHeight;
    let left = cx + 18 + w > innerWidth - 8 ? cx - w - 18 : cx + 18;
    if (innerWidth < 560) left = (innerWidth - w) / 2;
    pop.style.left = Math.max(8, left) + "px";
    pop.style.top = Math.max(8, Math.min(innerHeight - h - 8, cy - h / 2)) + "px";
  }

  /* one LR-QAOA run: r vs p for its best placement and the median, with the noiseless curve and random + 3σ */
  function curveSeries(run, color, which) {
    const rows = which === "median" ? run.median : run.best;
    return rows.map((q) => ({ x: q.p, y: q.r, r_eff: q.r_eff, r_ovl: q.r_ovl, threshold: q.threshold, hollow: q.r_eff != null && q.r_eff <= 0, run }));
  }
  function noiseless(run) {
    const pts = run.median.filter((q) => q.r_ideal != null).map((q) => ({ x: q.p, y: q.r_ideal }));
    return pts.length ? [{ key: "noiseless", label: "noiseless", color: "var(--faint)", dash: true, noDots: true, noHit: true, pts }] : [];
  }
  const runTip = (label, pt) => `<b>${esc(label)}</b> · p = ${pt.x}<br><span class="k">r =</span> ${f3(pt.y)}`
    + (pt.r_ovl != null ? ` · <span class="k">r<sub>ovl</sub> =</span> ${f3(pt.r_ovl)}` : "")
    + `<br><span class="k">r<sub>eff</sub> =</span> ${f3(pt.r_eff)} ${ok(pt.r_eff)}`;

  /* ---------- main ---------- */
  // data/results.js (loaded by index.html) sets window.QEC_RESULTS, so the page works opened from disk;
  // served over http without it, the JSON is fetched instead
  (window.QEC_RESULTS ? Promise.resolve(window.QEC_RESULTS)
    : fetch("data/results.json", { cache: "no-cache" }).then((r) => { if (!r.ok) throw new Error(r.status); return r.json(); }))
    .then(render)
    .catch((e) => { $("#kSrc").textContent = "Could not load results (" + e.message + "). Run python scripts/build_site_data.py first."; });

  function render(data) {
    const DEV = data.devices;
    const dev = (k) => DEV[k] || { name: k, kind: "qpu", vendor: "unknown" };
    const isQpu = (r) => dev(r.device).kind === "qpu";
    const modColor = (k) => (dev(k).modality === "trapped ion" ? "var(--ion)" : "var(--sc)");

    /* key figures */
    const used = new Set([...data.chains, ...data.qec, ...data.memory].map((r) => r.device).filter((d) => dev(d).kind === "qpu"));
    $("#kQpu").textContent = used.size;
    const longest = data.chains.filter((r) => r.kind === "mcm" && r.nq > 2 && isQpu(r) && r.best_r_eff > 0).sort((a, b) => b.nq - a.nq)[0];
    if (longest) { $("#kChain").textContent = longest.nq + " spins"; $("#kChainLbl").textContent = `longest MCM chain above random (${dev(longest.device).name})`; }
    const codes = new Set(data.qec.filter(isQpu).map((r) => r.code));
    $("#kCodes").textContent = codes.size;
    $("#kCodesLbl").textContent = `QEC structures benchmarked on QPUs (${new Set(data.qec.filter((r) => isQpu(r)).map((r) => r.frame)).size} frames)`;
    $("#kMem").textContent = data.memory.length;
    $("#kMemLbl").textContent = `surface-code memory sittings (${new Set(data.memory.map((r) => r.device)).size} device${new Set(data.memory.map((r) => r.device)).size > 1 ? "s" : ""})`;
    const newest = [...data.chains, ...data.qec, ...data.memory].map((r) => r.date).sort().pop();
    $("#kSrc").textContent = `${data.chains.length} chain runs, ${data.qec.length} QEC-structure runs, ${data.memory.length} memory sittings · newest result ${fDate(newest)}`;
    $("#buildInfo").textContent = `Built ${data.generated.replace("T", " ").slice(0, 16)} UTC` + (data.commit ? ` from commit ${data.commit}` : "");

    /* latest feed: one card per device and day */
    const groups = {};
    const add = (r, line) => { const k = r.device + "|" + r.date; (groups[k] = groups[k] || { device: r.device, date: r.date, lines: new Set() }).lines.add(line); };
    data.chains.filter(isQpu).forEach((r) => add(r, `${r.kind.toUpperCase()} chain · ${r.nq} spins`));
    data.qec.filter(isQpu).forEach((r) => add(r, `${codeLabel(r.code)} · ${r.frame} frame${r.kind === "direct" ? " (direct)" : ""}`));
    data.memory.forEach((r) => add(r, `memory · ${codeLabel(r.code)} · ${Object.keys(r.bases).map((b) => BASIS[b].label).join(", ")}`));
    $("#feed").innerHTML = Object.values(groups).sort((a, b) => b.date.localeCompare(a.date)).slice(0, 8).map((g) =>
      `<li><span class="d">${fDate(g.date)}</span><span class="n">${esc(dev(g.device).name)}</span>${[...g.lines].slice(0, 4).map((l) => `<span class="m">${esc(l)}</span>`).join("")}${g.lines.size > 4 ? `<span class="m">+ ${g.lines.size - 4} more</span>` : ""}</li>`).join("");

    guard("#chains", renderChains);
    guard("#chains", renderTriplets);
    guard("#qec", renderQec);
    guard("#memory", renderMemory);

    /* ---------- 1D chains ---------- */
    function renderChains() {
      const emu = $("#chainEmu"), direct = $("#chainDirect");
      const picker = Picker(() => draw());
      const pool = () => data.chains.filter((r) => r.nq > 2 && (isQpu(r) || (emu.checked && dev(r.device).kind === "emulator")));
      // per device, kind and length: the run whose best placement got furthest above random
      const bestOf = (runs) => {
        const out = {};
        runs.forEach((r) => { const k = `${r.device}|${r.kind}|${r.nq}`; if (!out[k] || (r.best_r_eff ?? -9) > (out[k].best_r_eff ?? -9)) out[k] = r; });
        return Object.values(out);
      };
      const count = (r) => data.chains.filter((x) => x.device === r.device && x.kind === r.kind && x.nq === r.nq).length;

      function showRun(run, color, cx, cy) {
        const series = [...noiseless(run),
          { key: "median", label: "median over placements", color: null, pts: curveSeries(run, null, "median") },
          { key: "best", label: "best placement", color, pts: curveSeries(run, color, "best") }];
        $("#popTitle").textContent = `${dev(run.device).name} · ${run.nq} spins · ${run.kind.toUpperCase()}`;
        $("#popSub").innerHTML = `${run.placements} placement${run.placements > 1 ? "s" : ""} × ${fInt(run.shots)} shots · ${fDate(run.date)}`;
        $("#popLegend").innerHTML = `<span><i style="background:${color}"></i>best placement</span>` + (run.placements > 1 ? `<span><i style="background:var(--ctx)"></i>median</span>` : "")
          + `<span><i class="noiseless"></i>noiseless</span><span><i class="hollow"></i>not above random + 3σ</span>`;
        $("#popFoot").textContent = `Dashed grey line: random + 3σ = ${f3(run.best[0].threshold)}. File: ${run.file}`;
        openPop(cx, cy);
        lineChart($("#popChart"), {
          series, xInteger: true, xLabel: "LR-QAOA layers p", yLabel: "approximation ratio r",
          refs: [{ y: run.best[0].threshold, label: "random + 3σ" }], aria: "r versus layers for one chain run",
          tip: (s, pt) => runTip(s.label, pt),
        });
      }

      function draw() {
        const runs = bestOf(pool());
        const mcm = runs.filter((r) => r.kind === "mcm");
        const byDev = {};
        mcm.forEach((r) => (byDev[r.device] = byDev[r.device] || []).push(r));
        const rows = Object.entries(byDev).map(([d, rs]) => {
          const pass = rs.filter((r) => r.best_r_eff > 0).sort((a, b) => b.nq - a.nq)[0];
          return { d, rs, pass, n: pass ? pass.nq : 0 };
        }).sort((a, b) => b.n - a.n || dev(a.d).name.localeCompare(dev(b.d).name));
        const maxN = Math.max(...rows.map((r) => r.n), 1);
        const lb = $("#chainLb"); lb.innerHTML = "";
        rows.forEach((row) => {
          const meta = dev(row.d), col = modColor(row.d);
          const div = document.createElement("div");
          div.className = "lb-row"; div.setAttribute("role", "listitem"); div.tabIndex = 0;
          div.innerHTML = `<span class="lb-name">${esc(meta.name)}${meta.kind !== "qpu" ? " <span class='kind-tag'>emulator</span>" : ""}</span><span class="lb-track">${row.n ? `<span class="lb-bar" style="width:${(row.n / maxN) * 82}%;background:${col}"></span>` : ""}<span class="lb-val">${row.n ? row.n + " spins" : "none above random"}</span></span>`;
          const r = row.pass;
          const html = `<b>${esc(meta.name)}</b><br>` + (r
            ? `<span class="k">Longest chain above random:</span> ${r.nq} spins<br><span class="k">Best r<sub>eff</sub>:</span> ${f3(r.best_r_eff)}<br><span class="k">Deepest p above random:</span> ${r.deepest_p_above_random ?? "–"}<br><span class="k">Shots:</span> ${fInt(r.shots)} · ${fDate(r.date)}<br><span class="k">Click for r vs p</span>`
            : `No length above random. Lengths tried: ${row.rs.map((x) => x.nq).sort((a, b) => a - b).join(", ")}`);
          div.addEventListener("pointermove", (ev) => showTip(html, ev.clientX, ev.clientY));
          div.addEventListener("pointerleave", hideTip);
          const target = row.pass || [...row.rs].sort((a, b) => b.nq - a.nq)[0];
          div.addEventListener("click", (ev) => { ev.stopPropagation(); hideTip(); showRun(target, col, ev.clientX, ev.clientY); });
          div.addEventListener("keydown", (ev) => { if (ev.key === "Enter" || ev.key === " ") { ev.preventDefault(); const b = div.getBoundingClientRect(); showRun(target, col, b.right, b.top + b.height / 2); } });
          lb.appendChild(div);
        });
        $("#chainLbLegend").innerHTML = `<span><i class="dot" style="background:var(--sc)"></i>superconducting</span><span><i class="dot" style="background:var(--ion)"></i>trapped ion</span>`;

        if (!picker.keys().length) picker.set(rows.filter((r) => r.n).slice(0, 3).map((r) => r.d));
        const shown = runs.filter((r) => r.kind === "mcm" || direct.checked);
        const series = [];
        Object.keys(byDev).forEach((d) => ["mcm", "direct"].forEach((kind) => {
          const rs = shown.filter((r) => r.device === d && r.kind === kind && r.best_r_eff != null);
          if (rs.length) series.push({ key: d, label: dev(d).name + (kind === "direct" ? " (direct)" : ""), color: picker.color(d), dash: kind === "direct",
            pts: rs.map((r) => ({ x: r.nq, y: r.best_r_eff, hollow: !(r.best_r_eff > 0), run: r })) });
        }));
        renderChips($("#chainChips"), rows.map((r) => ({ key: r.d, label: dev(r.d).name })), picker);
        $("#chainLegend").innerHTML = picker.keys().map((k) => `<span><i style="background:${picker.color(k)}"></i>${esc(dev(k).name)}</span>`).join("")
          + `<span><i style="background:var(--ctx)"></i>other devices</span><span><i class="hollow"></i>not above random + 3σ</span>` + (direct.checked ? `<span><i class="dash" style="border-top-color:var(--muted)"></i>direct reference</span>` : "");
        const sizes = [...new Set(shown.map((r) => r.nq))].sort((a, b) => a - b);
        lineChart($("#chainChart"), {
          series, xTicks: sizes, xLabel: "chain length N (spins)", yLabel: "best effective ratio r_eff",
          refs: [{ y: 0, label: "random + 3σ" }], aria: "Best effective approximation ratio versus chain length",
          onPoint: (s, p, ev) => showRun(p.run, picker.color(s.key) || "var(--s1)", ev.clientX, ev.clientY),
          tip: (s, p) => `<b>${esc(s.label)}</b> · N = ${p.x}<br><span class="k">best r<sub>eff</sub>:</span> ${f3(p.y)} ${ok(p.y)}<br><span class="k">deepest p above random:</span> ${p.run.deepest_p_above_random ?? "–"}<br><span class="k">placements:</span> ${p.run.placements} · <span class="k">shots:</span> ${fInt(p.run.shots)}<br><span class="k">${count(p.run)} run${count(p.run) > 1 ? "s" : ""} of this length; the best shown · ${fDate(p.run.date)}</span>`,
        });

        const all = data.chains.filter((r) => isQpu(r) || (emu.checked && dev(r.device).kind === "emulator")).sort((a, b) => b.nq - a.nq || (b.best_r_eff ?? -9) - (a.best_r_eff ?? -9));
        const table = $("#chainTable");
        table.innerHTML = `<thead><tr><th>Device</th><th>Kind</th><th>Spins N</th><th>Placements</th><th>Best r<sub>eff</sub></th><th>Deepest p above random</th><th>Depths</th><th>Shots</th><th>Date</th></tr></thead><tbody>`
          + all.map((r, i) => `<tr data-i="${i}"><td>${esc(dev(r.device).name)}</td><td>${r.kind.toUpperCase()}</td><td>${r.nq}</td><td>${r.placements}</td><td>${f3(r.best_r_eff)}</td><td>${r.deepest_p_above_random ?? "none"}</td><td>${r.best.map((q) => q.p).join(", ")}</td><td>${fInt(r.shots)}</td><td>${fDate(r.date)}</td></tr>`).join("") + "</tbody>";
        table.querySelectorAll("tbody tr").forEach((tr) => {
          tr.style.cursor = "pointer";
          tr.addEventListener("click", (ev) => { ev.stopPropagation(); const r = all[+tr.dataset.i]; showRun(r, picker.color(r.device) || "var(--s1)", ev.clientX, ev.clientY); });
        });
      }
      emu.addEventListener("change", draw);
      direct.addEventListener("change", draw);
      draw();
    }

    /* ---------- QEC structures, three frames ---------- */
    function renderQec() {
      const runs = data.qec;
      const order = (c) => { const m = c.match(/^(surface|color)_d(\d+)$/); return m ? (m[1] === "surface" ? 0 : 1) * 100 + +m[2] : 200 + (+c.replace(/\D/g, "") || 0); };
      const codes = [...new Set(runs.map((r) => r.code))].sort((a, b) => order(a) - order(b));
      const withFrames = codes.filter((c) => new Set(runs.filter((r) => r.code === c).map((r) => r.frame)).size > 1);
      let code = withFrames[0] || codes[0], agg = "median", sitting = null, mode = "best";
      const picker = Picker(() => draw());
      const sittingKey = (r) => `${r.device}|${r.sitting}`;
      // line style carries the frame (and the direct reference), colour carries the device
      const STYLE = { Z: null, X: "7 4", XZ: "2 3", direct: "9 3 2 3" };
      const styleOf = (r) => (r.kind === "direct" ? STYLE.direct : STYLE[r.frame]);
      const styleName = (r) => (r.kind === "direct" ? "direct (no ancillas)" : `${r.frame} frame`);
      const rowsOf = (r) => (agg === "best" ? r.best : r.median);
      const merit = (r) => { const v = rowsOf(r).map((q) => q.r_eff).filter((x) => x != null); return v.length ? v.reduce((a, b) => a + b) / v.length : -9; };
      const legendSwatch = (dash, color = "var(--muted)") => dash
        ? `<i style="height:0;background:none;border-radius:0;border-top:2px ${dash === STYLE.XZ ? "dotted" : "dashed"} ${color}"></i>`
        : `<i style="background:${color}"></i>`;

      function draw() {
        seg($("#qecCode"), codes.map((c) => ({ key: c, label: codeLabel(c) })), code, (k) => { code = k; sitting = null; draw(); });
        seg($("#qecMode"), [{ key: "best", label: "best run per device" }, { key: "sitting", label: "one sitting" }], mode, (k) => { mode = k; draw(); });
        const rs = runs.filter((r) => r.code === code);
        const multi = rs.some((r) => r.placements > 1);
        const aggHost = $("#qecAgg");
        aggHost.hidden = !multi;
        seg(aggHost, [{ key: "median", label: "median" }, { key: "best", label: "best placement" }], agg, (k) => { agg = k; draw(); });

        /* best run of each device, per frame and kind */
        const best = {};
        const rank = (r) => [Math.min(rowsOf(r).length, 3), merit(r)];          // a sweep of 3+ depths first
        const better = (a, b) => { const [x, y] = [rank(a), rank(b)]; return x[0] !== y[0] ? x[0] > y[0] : x[1] > y[1]; };
        rs.forEach((r) => { const k = `${r.device}|${r.frame}|${r.kind}`; if (!best[k] || better(r, best[k])) best[k] = r; });
        const bestRuns = Object.values(best);
        const devices = [...new Set(bestRuns.map((r) => r.device))]
          .sort((a, b) => Math.max(...bestRuns.filter((r) => r.device === b).map(merit)) - Math.max(...bestRuns.filter((r) => r.device === a).map(merit)));
        if (!picker.keys().some((k) => devices.includes(k))) {          // QPUs first; emulators only to fill
          const qpus = devices.filter((d) => dev(d).kind === "qpu");
          picker.set((qpus.length ? qpus : devices).slice(0, 3));
        }

        /* the sittings, for the one-sitting view */
        const sittings = {};
        rs.forEach((r) => { const k = sittingKey(r); (sittings[k] = sittings[k] || []).push(r); });
        const keys = Object.keys(sittings).sort((a, b) => sittings[b][0].sitting.localeCompare(sittings[a][0].sitting));
        if (!sitting || !sittings[sitting]) sitting = keys.find((k) => new Set(sittings[k].map((r) => r.frame)).size === 3) || keys[0];
        const sel = $("#qecSitting");
        sel.innerHTML = keys.map((k) => {
          const g = sittings[k], s = g[0];
          const what = [...new Set(g.map((r) => r.frame + (r.kind === "direct" ? " direct" : "")))].join(", ");
          return `<option value="${esc(k)}"${k === sitting ? " selected" : ""}>${esc(dev(s.device).name)} · ${fDate(s.date)} ${sittingTime(s.sitting)} · ${esc(what)}</option>`;
        }).join("");
        sel.onchange = () => { sitting = sel.value; draw(); };
        $("#qecSittingBox").hidden = mode !== "sitting";
        $("#qecChips").hidden = mode !== "best";

        let series, shown, ref;
        if (mode === "best") {
          renderChips($("#qecChips"), devices.map((d) => ({ key: d, label: dev(d).name + (dev(d).kind !== "qpu" ? " (emulator)" : "") })), picker);
          shown = bestRuns.filter((r) => picker.has(r.device));
          series = bestRuns.filter((r) => !picker.has(r.device)).map((r) => ({ key: r.id, label: `${dev(r.device).name} · ${styleName(r)}`, color: null, dash: styleOf(r), pts: curveSeries(r, null, agg) }));
          shown.forEach((r) => series.push({ key: r.id, label: `${dev(r.device).name} · ${styleName(r)}`, color: picker.color(r.device), dash: styleOf(r), pts: curveSeries(r, null, agg) }));
          ref = shown[0] || bestRuns[0];
          $("#qecSub").textContent = "best run of each device and frame: a full depth sweep, highest mean r_eff";
          const styles = [...new Map(bestRuns.map((r) => [styleName(r), styleOf(r)])).entries()];
          $("#qecLegend").innerHTML = picker.keys().filter((k) => devices.includes(k)).map((k) => `<span><i style="background:${picker.color(k)}"></i>${esc(dev(k).name)}</span>`).join("")
            + styles.map(([name, dash]) => `<span>${legendSwatch(dash)}${esc(name)}</span>`).join("")
            + `<span><i class="noiseless"></i>noiseless</span><span><i style="background:var(--ctx)"></i>other devices</span><span><i class="hollow"></i>not above random + 3σ</span>`;
        } else {
          shown = sittings[sitting];
          series = rs.filter((r) => !shown.includes(r)).map((r) => ({ key: r.id, label: `${dev(r.device).name} ${fShort(r.date)} · ${styleName(r)}`, color: null, dash: styleOf(r), pts: curveSeries(r, null, agg) }));
          shown.forEach((r) => series.push({ key: r.id, label: styleName(r), color: FRAME_COLOR[r.frame], dash: r.kind === "direct" ? STYLE.direct : null, pts: curveSeries(r, null, agg) }));
          ref = shown[0];
          $("#qecSub").textContent = `${dev(ref.device).name} · ${fDate(ref.date)} · ${ref.placements} placement${ref.placements > 1 ? "s" : ""} · ${fInt(ref.shots)} shots`;
          const frames = [...new Set(shown.map((r) => r.frame))];
          $("#qecLegend").innerHTML = frames.map((f) => `<span><i style="background:${FRAME_COLOR[f]}"></i>${f} frame</span>`).join("")
            + (shown.some((r) => r.kind === "direct") ? `<span>${legendSwatch(STYLE.direct)}direct (no ancillas)</span>` : "")
            + `<span><i class="noiseless"></i>noiseless</span><span><i style="background:var(--ctx)"></i>other sittings</span><span><i class="hollow"></i>not above random + 3σ</span>`;
        }
        series.unshift(...noiseless(ref));
        const thr = ref.median[0].threshold;
        const ps = [...new Set(rs.flatMap((r) => r.median.map((q) => q.p)))].sort((a, b) => a - b);
        lineChart($("#qecChart"), {
          series, xTicks: ps, xLabel: "LR-QAOA layers p (rounds of syndrome extraction)", yLabel: "approximation ratio r",
          refs: thr != null ? [{ y: thr, label: `random + 3σ (${fInt(ref.shots)} shots)` }] : [], aria: `r versus layers for ${codeLabel(code)}`,
          tip: (s, pt) => runTip(s.label, pt) + `<br><span class="k">${esc(dev(pt.run.device).name)} · ${fDate(pt.run.date)} ${sittingTime(pt.run.sitting)} · ${pt.run.placements} placement${pt.run.placements > 1 ? "s" : ""}</span>`,
        });

        /* r_eff at p = 3 (or the first depth) per sitting, per frame, for one device */
        const focus = mode === "best" ? (picker.keys().find((k) => devices.includes(k)) || devices[0]) : ref.device;
        const P = ps.includes(3) ? 3 : ps[0];
        $("#qecTimeSub").innerHTML = `${esc(dev(focus).name)} · r<sub>eff</sub> at p = ${P}, ${agg === "best" ? "best placement" : "median over placements"} · MCM`;
        const stampT = (r) => Date.UTC(+r.sitting.slice(0, 4), +r.sitting.slice(4, 6) - 1, +r.sitting.slice(6, 8), +r.sitting.slice(9, 11) || 0, +r.sitting.slice(11, 13) || 0);
        const tSeries = ["Z", "X", "XZ"].map((f) => ({
          key: f, label: `${f} frame`, color: FRAME_COLOR[f],
          pts: rs.filter((r) => r.frame === f && r.kind === "mcm" && r.device === focus).map((r) => {
            const q = rowsOf(r).find((x) => x.p === P);
            return q ? { x: stampT(r), y: q.r_eff, hollow: !(q.r_eff > 0), run: r } : null;
          }).filter(Boolean),
        })).filter((s) => s.pts.length);
        const xs = tSeries.flatMap((s) => s.pts.map((p) => p.x));
        const ticks = [];
        if (xs.length) { const lo = Math.min(...xs), hi = Math.max(...xs), step = Math.max(1, Math.ceil((hi - lo) / 864e5 / 6)) * 864e5; for (let v = Math.floor(lo / 864e5) * 864e5; v <= hi + 864e5; v += step) ticks.push(v); }
        $("#qecTimeLegend").innerHTML = tSeries.map((s) => `<span><i style="background:${s.color}"></i>${s.label}</span>`).join("") + `<span><i class="hollow"></i>not above random + 3σ</span><span class="kind-tag">click a point to show that sitting</span>`;
        lineChart($("#qecTime"), {
          series: tSeries, xTicks: ticks, xFmt: (v) => { const d = new Date(v); return `${d.getUTCDate()} ${MONTHS[d.getUTCMonth()]}`; },
          xLabel: "sitting", yLabel: `r_eff at p = ${P}`, refs: [{ y: 0, label: "random + 3σ" }],
          aria: `Effective ratio at p = ${P} for each frame over sittings, ${codeLabel(code)}`,
          tip: (s, pt) => `<b>${esc(s.label)}</b> · ${esc(dev(pt.run.device).name)}<br>${fDate(pt.run.date)} ${sittingTime(pt.run.sitting)}<br><span class="k">r<sub>eff</sub> =</span> ${f3(pt.y)} ${ok(pt.y)}<br><span class="k">click to show this sitting</span>`,
          onPoint: (s, pt) => { mode = "sitting"; sitting = sittingKey(pt.run); draw(); },
        });

        const table = $("#qecTable");
        const sorted = [...rs].sort((a, b) => b.sitting.localeCompare(a.sitting) || ["Z", "X", "XZ"].indexOf(a.frame) - ["Z", "X", "XZ"].indexOf(b.frame));
        table.innerHTML = `<thead><tr><th>Device</th><th>Sitting</th><th>Frame</th><th>Kind</th><th>Placements</th><th>Data qubits</th><th>r<sub>eff</sub> at p = ${P}</th><th>Best r<sub>eff</sub></th><th>Deepest p above random</th><th>Shots</th></tr></thead><tbody>`
          + sorted.map((r) => {
            const q = r.median.find((x) => x.p === P), on = shown.includes(r);
            const dot = on ? (mode === "best" ? picker.color(r.device) : FRAME_COLOR[r.frame]) : null;
            return `<tr class="${on ? "sel" : ""}" data-k="${esc(sittingKey(r))}"><td>${dot ? `<i style="display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:6px;background:${dot}"></i>` : ""}${esc(dev(r.device).name)}</td><td>${fDate(r.date)} ${sittingTime(r.sitting)}</td><td>${r.frame}</td><td>${r.kind.toUpperCase()}</td><td>${r.placements}</td><td>${r.n_data}</td><td>${f3(q ? q.r_eff : null)}</td><td>${f3(r.best_r_eff)}</td><td>${r.deepest_p_above_random ?? "none"}</td><td>${fInt(r.shots)}</td></tr>`;
          }).join("") + "</tbody>";
        table.querySelectorAll("tbody tr").forEach((tr) => { tr.style.cursor = "pointer"; tr.addEventListener("click", () => { mode = "sitting"; sitting = tr.dataset.k; draw(); }); });
      }
      draw();
    }

    /* ---------- triplets: distribution of r over every triple of a campaign ---------- */
    function renderTriplets() {
      const camps = data.chains.filter((r) => r.nq === 2 && r.every && isQpu(r));
      if (!camps.length) { $("#tripletPanel").hidden = true; return; }
      const kinds = [...new Set(camps.map((r) => r.kind))];
      const depths = [...new Set(camps.flatMap((r) => Object.keys(r.every).map(Number)))].sort((a, b) => a - b);
      let depth = depths.includes(3) ? 3 : depths[0], kind = "mcm";
      const host = $("#tripletChart");
      const pct = (v, q) => { const s = [...v].sort((a, b) => a - b), i = (s.length - 1) * q, lo = Math.floor(i); return s[lo] + (s[Math.min(lo + 1, s.length - 1)] - s[lo]) * (i - lo); };

      function draw() {
        seg($("#tripletDepth"), depths.map((p) => ({ key: String(p), label: `p = ${p}` })), String(depth), (k) => { depth = +k; draw(); });
        const kHost = $("#tripletKind");
        kHost.hidden = kinds.length < 2;
        seg(kHost, kinds.map((k) => ({ key: k, label: k === "mcm" ? "MCM" : "direct" })), kind, (k) => { kind = k; draw(); });
        const rows = camps.filter((r) => r.kind === kind && r.every[String(depth)]).map((r) => {
          const v = r.every[String(depth)].filter((x) => x != null), q = r.median.find((x) => x.p === depth) || {};
          return { r, v, med: pct(v, 0.5), q25: pct(v, 0.25), q75: pct(v, 0.75), lo: pct(v, 0.05), hi: pct(v, 0.95), thr: q.threshold, ideal: q.r_ideal };
        }).sort((a, b) => b.med - a.med);
        host.style.height = Math.max(220, 60 + rows.length * 44) + "px";
        host.innerHTML = "";
        const W = host.clientWidth, H = host.clientHeight;
        if (!W || !rows.length) return;
        const m = { t: 10, r: 18, b: 40, l: 170 };
        const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": `Distribution of r over every triplet at p = ${depth}` }, host);
        const all = rows.flatMap((x) => x.v).concat(rows.map((x) => x.thr), rows.map((x) => x.ideal)).filter((v) => v != null);
        let x0 = Math.min(...all), x1 = Math.max(...all); const pad = (x1 - x0) * 0.04; x0 -= pad; x1 += pad;
        const X = (v) => m.l + (v - x0) / (x1 - x0) * (W - m.l - m.r);
        const band = (H - m.t - m.b) / rows.length, Yc = (i) => m.t + band * (i + 0.5);
        const grid = el("g", { class: "grid" }, svg), axis = el("g", { class: "axis" }, svg);
        niceTicks(x0, x1, Math.max(3, Math.round((W - m.l - m.r) / 90))).forEach((v) => {
          el("line", { x1: X(v), x2: X(v), y1: m.t, y2: H - m.b }, grid);
          el("text", { x: X(v), y: H - m.b + 17, "text-anchor": "middle" }, axis).textContent = v.toFixed(2);
        });
        el("text", { x: (m.l + W - m.r) / 2, y: H - 6, "text-anchor": "middle", class: "title" }, axis).textContent = `approximation ratio r at p = ${depth}`;
        const thr = rows[0].thr, ideal = rows[0].ideal;
        if (thr != null) el("line", { x1: X(thr), x2: X(thr), y1: m.t, y2: H - m.b, class: "ref" }, svg);
        if (ideal != null) el("line", { x1: X(ideal), x2: X(ideal), y1: m.t, y2: H - m.b, stroke: "var(--faint)", "stroke-dasharray": "2 3", "stroke-width": 1.5 }, svg);
        const hits = [];
        rows.forEach((row, i) => {
          const y = Yc(i), col = modColor(row.r.device), h = Math.min(18, band * 0.45);
          el("text", { x: m.l - 10, y: y - 2, "text-anchor": "end", class: "row-label" }, svg).textContent = dev(row.r.device).name;
          el("text", { x: m.l - 10, y: y + 12, "text-anchor": "end", class: "row-sub" }, svg).textContent = `${fShort(row.r.date)} ${row.r.date.slice(0, 4)} · ${row.v.length} triplets`;
          row.v.forEach((v, j) => el("circle", { cx: X(v), cy: y + ((j * 7919) % 100 / 100 - 0.5) * h * 1.6, r: 2, fill: col, opacity: 0.35 }, svg));
          el("line", { x1: X(row.lo), x2: X(row.hi), y1: y, y2: y, stroke: "var(--ink)", "stroke-width": 1.2 }, svg);
          el("rect", { x: X(row.q25), y: y - h / 2, width: Math.max(1, X(row.q75) - X(row.q25)), height: h, fill: col, "fill-opacity": 0.25, stroke: col, class: "box", rx: 2 }, svg);
          el("line", { x1: X(row.med), x2: X(row.med), y1: y - h / 2, y2: y + h / 2, stroke: "var(--ink)", "stroke-width": 2.5 }, svg);
          hits.push({ y, row });
        });
        const labels = el("g", {}, svg);
        if (thr != null) el("text", { x: X(thr) + 4, y: m.t + 10, class: "ref-label" }, labels).textContent = "random + 3σ";
        if (ideal != null) el("text", { x: X(ideal) - 4, y: m.t + 10, "text-anchor": "end", class: "ref-label" }, labels).textContent = "noiseless";
        svg.addEventListener("pointermove", (ev) => {
          const b = svg.getBoundingClientRect(), my = (ev.clientY - b.top) * H / b.height;
          const h = hits.find((x) => Math.abs(x.y - my) < band / 2);
          if (!h) { hideTip(); return; }
          const { row } = h, above = row.thr != null ? row.v.filter((v) => v > row.thr).length : null;
          showTip(`<b>${esc(dev(row.r.device).name)}</b> · ${fDate(row.r.date)}<br><span class="k">${row.v.length} triplets at p = ${depth}, ${fInt(row.r.shots)} shots each</span>`
            + `<br><span class="k">median r:</span> ${f3(row.med)} · <span class="k">quartiles:</span> ${f3(row.q25)} – ${f3(row.q75)}`
            + `<br><span class="k">best triplet:</span> ${f3(Math.max(...row.v))}`
            + (above != null ? `<br><span class="k">above random + 3σ:</span> ${above} of ${row.v.length}` : ""), ev.clientX, ev.clientY);
        });
        svg.addEventListener("pointerleave", hideTip);
        $("#tripletLegend").innerHTML = `<span><i class="dot" style="background:var(--sc)"></i>superconducting</span><span><i class="dot" style="background:var(--ion)"></i>trapped ion</span>`
          + `<span>box: quartiles · bar: median · line: 5th–95th percentile · dots: one per triplet</span><span><i class="dash" style="border-top-color:var(--faint)"></i>random + 3σ</span><span><i class="dot2"></i>noiseless</span>`;
      }
      draw();
      let w = host.clientWidth;
      new ResizeObserver(() => { if (host.clientWidth !== w) { w = host.clientWidth; draw(); } }).observe(host);
    }

    /* ---------- surface-code memory ---------- */
    function renderMemory() {
      const runs = data.memory;
      if (!runs.length) { $("#memory").hidden = true; return; }
      const codes = [...new Set(runs.map((r) => r.code))].sort();
      let code = codes[0], pick = null;

      function draw() {
        seg($("#memCode"), codes.map((c) => ({ key: c, label: codeLabel(c) })), code, (k) => { code = k; pick = null; draw(); });
        const rs = runs.filter((r) => r.code === code).sort((a, b) => b.sitting.localeCompare(a.sitting) || a.device.localeCompare(b.device));
        if (!pick || !rs.find((r) => r.id === pick)) pick = (rs.find((r) => Object.keys(r.bases).length === 2 && r.bases.Z.R.length > 2) || rs[0]).id;
        const sel = $("#memSitting");
        sel.innerHTML = rs.map((r) => `<option value="${esc(r.id)}"${r.id === pick ? " selected" : ""}>${esc(dev(r.device).name)} · ${fDate(r.date)} ${sittingTime(r.sitting)} · ${r.patches} patches${r.history === "last" ? " · last-round record" : ""}</option>`).join("");
        sel.onchange = () => { pick = sel.value; draw(); };
        const mine = rs.find((r) => r.id === pick);

        const series = [];
        rs.filter((r) => r !== mine).forEach((r) => Object.entries(r.bases).forEach(([b, v]) => series.push({
          key: r.id + b, label: `${BASIS[b].label} · ${dev(r.device).name} ${fShort(r.date)}`, color: null,
          pts: v.R.map((R, i) => ({ x: R, y: v.median[i], run: r, basis: b, i })) })));
        Object.entries(mine.bases).forEach(([b, v]) => series.push({
          key: mine.id + b, label: BASIS[b].label, color: BASIS[b].color, band: true,
          pts: v.R.map((R, i) => ({ x: R, y: v.median[i], lo: v.q25[i], hi: v.q75[i], run: mine, basis: b, i })) }));
        $("#memLegend").innerHTML = Object.keys(mine.bases).map((b) => `<span><i style="background:${BASIS[b].color}"></i>${BASIS[b].label} (${b} memory)</span>`).join("")
          + `<span><i class="iqr" style="background:var(--muted)"></i>interquartile range over ${mine.patches} patches</span><span><i style="background:var(--ctx)"></i>other sittings</span>`;
        const Rs = [...new Set(rs.flatMap((r) => Object.values(r.bases).flatMap((v) => v.R)))].sort((a, b) => a - b);
        lineChart($("#memChart"), {
          series, xLog: Rs[Rs.length - 1] / Rs[0] >= 8, xTicks: Rs, yLog: true, xLabel: "rounds R", yLabel: "logical error probability P_L",
          aria: `Logical error probability versus rounds, ${codeLabel(code)}`,
          tip: (s, pt) => { const v = pt.run.bases[pt.basis]; return `<b>${esc(s.label)}</b> · R = ${pt.x}<br><span class="k">median P<sub>L</sub>:</span> ${f4(pt.y)}`
            + `<br><span class="k">quartiles:</span> ${f4(v.q25[pt.i])} – ${f4(v.q75[pt.i])}<br><span class="k">best patch:</span> ${f4(v.best[pt.i])}`
            + `<br><span class="k">${esc(dev(pt.run.device).name)} · ${fDate(pt.run.date)} · ${fInt(v.shots)} shots per patch</span>`; },
        });

        /* epsilon_L per sitting */
        const tOf = (r) => Date.UTC(+r.sitting.slice(0, 4), +r.sitting.slice(4, 6) - 1, +r.sitting.slice(6, 8), +r.sitting.slice(9, 11) || 0, +r.sitting.slice(11, 13) || 0);
        const tSeries = ["Z", "X"].map((b) => ({
          key: b, label: BASIS[b].label, color: BASIS[b].color,
          pts: rs.filter((r) => r.bases[b] && r.bases[b].eps_round != null && r.bases[b].R.length > 2).map((r) => ({ x: tOf(r), y: r.bases[b].eps_round, run: r, basis: b })),
        })).filter((s) => s.pts.length);
        const xs = tSeries.flatMap((s) => s.pts.map((p) => p.x));
        const ticks = [];
        if (xs.length) { const lo = Math.min(...xs), hi = Math.max(...xs), step = Math.max(1, Math.ceil((hi - lo) / 864e5 / 6)) * 864e5; for (let v = Math.floor(lo / 864e5) * 864e5; v <= hi + 864e5; v += step) ticks.push(v); }
        $("#memTimeLegend").innerHTML = tSeries.map((s) => `<span><i style="background:${s.color}"></i>${s.label}</span>`).join("") + `<span class="kind-tag">sittings with at least three round counts · click a point to show it</span>`;
        lineChart($("#memTime"), {
          series: tSeries, xTicks: ticks, xFmt: (v) => { const d = new Date(v); return `${d.getUTCDate()} ${MONTHS[d.getUTCMonth()]}`; },
          xLabel: "sitting", yLabel: "logical error per round ε_L", aria: `Logical error per round over sittings, ${codeLabel(code)}`,
          tip: (s, pt) => `<b>${esc(s.label)}</b> · ${esc(dev(pt.run.device).name)}<br>${fDate(pt.run.date)} ${sittingTime(pt.run.sitting)}<br><span class="k">ε<sub>L</sub> =</span> ${f4(pt.y)} per round`,
          onPoint: (s, pt) => { pick = pt.run.id; draw(); },
        });

        const at = (v, R) => (v && v.R.includes(R) ? v.median[v.R.indexOf(R)] : null);
        const table = $("#memTable");
        table.innerHTML = `<thead><tr><th>Device</th><th>Sitting</th><th>Patches</th><th>Rounds</th><th>P<sub>L</sub> |0⟩ at R = 3</th><th>P<sub>L</sub> |+⟩ at R = 3</th><th>ε<sub>L</sub> |0⟩</th><th>ε<sub>L</sub> |+⟩</th><th>Shots</th><th>Syndrome record</th></tr></thead><tbody>`
          + rs.map((r) => `<tr class="${r === mine ? "sel" : ""}" data-k="${esc(r.id)}"><td>${esc(dev(r.device).name)}</td><td>${fDate(r.date)} ${sittingTime(r.sitting)}</td><td>${r.patches}</td><td>${(r.bases.Z || r.bases.X).R.join(", ")}</td>`
            + `<td>${f4(at(r.bases.Z, 3))}</td><td>${f4(at(r.bases.X, 3))}</td><td>${f4(r.bases.Z && r.bases.Z.R.length > 2 ? r.bases.Z.eps_round : null)}</td><td>${f4(r.bases.X && r.bases.X.R.length > 2 ? r.bases.X.eps_round : null)}</td>`
            + `<td>${fInt((r.bases.Z || r.bases.X).shots)}</td><td>${r.history === "last" ? "last round only" : "every round"}</td></tr>`).join("") + "</tbody>";
        table.querySelectorAll("tbody tr").forEach((tr) => { tr.style.cursor = "pointer"; tr.addEventListener("click", () => { pick = tr.dataset.k; draw(); }); });
      }
      draw();
    }
  }
})();
