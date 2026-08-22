from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from profile_payload import ReportPayload, build_report_payload

DEVICE_SPECS_PATH = Path(__file__).with_name("device_specs.json")


def load_device_specs(path: Path = DEVICE_SPECS_PATH) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("devices"), dict):
        raise ValueError(f"{path}: device specs need a devices object")
    if not isinstance(value.get("models"), dict):
        raise ValueError(f"{path}: device specs need a models object")
    return value


def _payload_from_selection(
    records: list[dict[str, Any]],
    baseline_records: list[dict[str, Any]] | None,
    device_specs: dict[str, Any] | None,
) -> ReportPayload:
    selection = device_specs or load_device_specs()
    return build_report_payload(
        records,
        baseline_records,
        device_specs=selection,
        device_key=selection.get("selected_device"),
        baseline_device_key=selection.get("baseline_device"),
    )


def build_html(
    records: list[dict[str, Any]],
    baseline_records: list[dict[str, Any]] | None = None,
    device_specs: dict[str, Any] | None = None,
) -> str:
    payload = _payload_from_selection(records, baseline_records, device_specs)
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).replace("<", "\\u003c")
    return _HTML.replace("__REPORT_PAYLOAD__", encoded)


_HTML = r'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LuceGraph</title>
<style>
:root {
  color-scheme: dark;
  --bg: #001332;
  --bg-deep: #010b26;
  --panel: #06254b;
  --panel-soft: rgba(6, 37, 75, 0.74);
  --line: rgba(157, 178, 208, 0.2);
  --line-strong: rgba(248, 169, 3, 0.38);
  --text: #eaf2ff;
  --muted: #9db2d0;
  --dim: #5d7293;
  --accent: #f8a903;
  --accent-bright: #ffc34a;
  --success: #48d597;
  --danger: #ff5c70;
  --warn: #ffc34a;
  --shadow: rgba(0, 0, 0, 0.28);
}
* { box-sizing: border-box; }
body {
  min-width: 320px;
  margin: 0;
  background:
    radial-gradient(circle at 20% 12%, rgba(248, 169, 3, 0.08), transparent 28rem),
    radial-gradient(circle at 84% 4%, rgba(72, 213, 151, 0.08), transparent 24rem),
    linear-gradient(180deg, var(--bg), var(--bg-deep));
  color: var(--text);
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
body::before {
  position: fixed; inset: 0; z-index: -1; content: "";
  background-image:
    linear-gradient(rgba(157, 178, 208, 0.06) 1px, transparent 1px),
    linear-gradient(90deg, rgba(157, 178, 208, 0.06) 1px, transparent 1px);
  background-size: 44px 44px;
}
main { width: min(1440px, 100%); margin: 0 auto; padding: 22px; }
header { display: flex; flex-wrap: wrap; justify-content: space-between; gap: 18px; align-items: end; margin-bottom: 16px; }
h1, h2, h3, p { margin: 0; }
h1 { font-size: clamp(1.7rem, 3.6vw, 3.1rem); line-height: 0.95; }
h2 { font-size: 1rem; }
.eyebrow, .mono, label, select, button, th, td { font-family: "JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
.eyebrow { margin-bottom: 8px; color: var(--accent-bright); font-size: .72rem; letter-spacing: .12em; text-transform: uppercase; }
.subhead { margin-top: 10px; color: var(--muted); }
.chips, .controls { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
.chip { padding: 6px 9px; border: 1px solid var(--line); border-radius: 6px; background: var(--panel-soft); color: var(--muted); font: 700 .7rem "JetBrains Mono", ui-monospace, monospace; }
.chip.good { color: var(--success); border-color: rgba(72, 213, 151, .4); }
.chip.warn { color: var(--warn); border-color: rgba(255, 195, 74, .45); }
.chip.bad { color: var(--danger); border-color: rgba(255, 92, 112, .5); }
.panel { margin-top: 12px; overflow: hidden; border: 1px solid var(--line); border-radius: 9px; background: linear-gradient(180deg, rgba(6,37,75,.94), rgba(6,37,75,.68)); box-shadow: 0 12px 30px var(--shadow); }
.panel-head { display: flex; flex-wrap: wrap; justify-content: space-between; gap: 12px; align-items: center; min-height: 48px; padding: 12px 14px; border-bottom: 1px solid var(--line); }
.panel-body { padding: 14px; }
.walls { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(500px, 100%), 1fr)); gap: 12px; }
.wall-card { min-width: 0; border: 1px solid var(--line); border-radius: 7px; padding: 10px; background: rgba(1, 11, 38, .32); }
.wall-card h3 { margin-bottom: 8px; font-size: .78rem; color: var(--muted); text-transform: uppercase; letter-spacing: .08em; }
svg { display: block; width: 100%; overflow: visible; }
.wall-svg { min-height: 350px; }
.axis, .bar-label { fill: var(--muted); font: 11px "JetBrains Mono", ui-monospace, monospace; }
.bar-label.anchor-note { fill: var(--accent-bright); }
.segment-label { fill: rgba(1, 11, 38, .92); font: 700 9px "JetBrains Mono", ui-monospace, monospace; pointer-events: none; }
.gridline { stroke: var(--line); stroke-width: 1; }
.segment { stroke: rgba(234, 242, 255, .18); stroke-width: 1; cursor: pointer; }
.segment:focus { outline: none; stroke: var(--accent-bright); stroke-width: 3; }
.anchor { stroke: var(--accent-bright); stroke-width: 3; stroke-linecap: round; }
.legend { display: flex; flex-wrap: wrap; gap: 6px 14px; align-items: center; margin: 0 0 10px; color: var(--muted); font: 11px "JetBrains Mono", ui-monospace, monospace; }
.legend svg { display: inline-block; width: 12px; height: 12px; vertical-align: -2px; margin-right: 5px; border: 1px solid var(--line); border-radius: 2px; }
.legend .sw { display: inline-block; width: 12px; height: 12px; vertical-align: -2px; margin-right: 5px; border: 1px solid var(--line); border-radius: 2px; }
.strip { margin: 0 0 10px; color: var(--muted); font: 12px "JetBrains Mono", ui-monospace, monospace; }
select, button { min-height: 34px; padding: 6px 9px; border: 1px solid var(--line); border-radius: 5px; background: var(--bg-deep); color: var(--text); }
button[aria-pressed="true"] { border-color: var(--accent); color: var(--accent-bright); }
.notice { margin-bottom: 8px; padding: 9px 11px; border: 1px solid rgba(255,195,74,.4); border-radius: 6px; background: rgba(255,195,74,.07); color: var(--warn); }
.detail { min-height: 76px; padding: 12px; border: 1px solid var(--line); border-radius: 6px; background: rgba(1,11,38,.4); color: var(--muted); }
.detail strong { color: var(--text); }
.idle-strip { margin: 10px 0 12px; padding: 8px 10px; border-left: 4px solid var(--danger); background: rgba(255,92,112,.08); color: var(--muted); }
.waterfall { overflow-x: auto; }
.request.failed rect { opacity: .72; stroke: var(--danger); }
.request.incomplete rect { opacity: .35; stroke-dasharray: 3 2; }
.funnel { display: grid; grid-template-columns: repeat(8, minmax(90px, 1fr)); gap: 7px; overflow-x: auto; }
.funnel-step { min-width: 90px; padding: 10px; border: 1px solid var(--line-strong); border-radius: 6px; background: rgba(248,169,3,.07); }
.funnel-step b, .funnel-step small { display: block; }
.funnel-step b { margin-top: 5px; color: var(--accent-bright); font: 700 1rem "JetBrains Mono", ui-monospace, monospace; }
.funnel-step small { margin-top: 3px; color: var(--muted); }
.two-col { display: grid; grid-template-columns: 1.35fr .65fr; gap: 12px; }
table { width: 100%; border-collapse: collapse; font-size: .76rem; }
th, td { padding: 7px 8px; border-bottom: 1px solid var(--line); text-align: right; }
th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) { text-align: left; }
.positive { color: var(--danger); }
.negative { color: var(--success); }
#tooltip { position: fixed; z-index: 20; max-width: 360px; pointer-events: none; padding: 9px 10px; border: 1px solid var(--line-strong); border-radius: 6px; background: #010b26; color: var(--text); font: 12px/1.45 "JetBrains Mono", ui-monospace, monospace; box-shadow: 0 8px 24px var(--shadow); }
#tooltip[hidden] { display: none; }
.muted { color: var(--muted); }
@media (max-width: 850px) { main { padding: 12px; } .two-col { grid-template-columns: 1fr; } }
</style>
</head>
<body>
<main>
  <header>
    <div>
      <div class="eyebrow">Lucebox / offline profiler</div>
      <h1>LuceGraph</h1>
      <p class="subhead" id="identity"></p>
    </div>
    <div class="chips" id="chips"></div>
  </header>

  <section class="panel">
    <div class="panel-head">
      <h2>Phase-budget wall</h2>
      <div class="controls">
        <label>Normalize
          <select id="normalization">
            <option value="ns_per_token">ns / durable token</option>
            <option value="ns_per_serviced_token">ns / serviced token</option>
            <option value="ns_per_round">ns / round</option>
            <option value="wall_share">share of wall</option>
          </select>
        </label>
        <label>Path <select id="path"></select></label>
        <button type="button" id="split" aria-pressed="false">Split paths</button>
      </div>
    </div>
    <div class="panel-body">
      <div id="notices"></div>
      <div class="legend" id="wall-legend"></div>
      <div class="walls" id="walls"></div>
      <div class="idle-strip" id="idle"></div>
      <div class="detail" id="detail">Focus or select a segment to inspect its percentiles and roofline facts.</div>
    </div>
  </section>

  <section class="panel">
    <div class="panel-head"><h2>Request waterfall</h2><span class="muted mono" id="latency"></span><span class="muted mono" id="request-note"></span></div>
    <div class="panel-body waterfall"><div class="legend" id="wf-legend"></div><svg id="waterfall" aria-label="Request timing waterfall"></svg></div>
  </section>

  <section class="panel">
    <div class="panel-head"><h2>Speculation funnel</h2></div>
    <div class="panel-body">
      <div class="strip" id="decisions"></div>
      <div class="two-col">
        <div class="funnel" id="funnel"></div>
        <svg id="acceptance" aria-label="Acceptance by speculative position"></svg>
      </div>
    </div>
  </section>

  <section class="panel" id="diff-panel" hidden>
    <div class="panel-head"><h2>Baseline delta</h2><button type="button" id="delta-expand" hidden></button></div>
    <div class="panel-body">
      <div id="diff-warnings"></div>
      <div style="overflow:auto"><table>
        <thead><tr><th>Path</th><th>Cohort / phase</th><th>Baseline / durable token</th><th>Current / durable token</th><th>Δ / durable token</th><th>Δ %</th></tr></thead>
        <tbody id="delta-rows"></tbody>
      </table></div>
    </div>
  </section>

  <section class="panel">
    <div class="panel-head"><h2>Capture contract</h2></div>
    <div class="panel-body muted" id="footer"></div>
  </section>
</main>
<div id="tooltip" role="tooltip" hidden></div>
<script type="application/json" id="data">__REPORT_PAYLOAD__</script>
<script>
(function () {
  "use strict";
  const data = JSON.parse(document.getElementById("data").textContent);
  const current = data.current;
  const baseline = data.baseline;
  const state = { normalization: "ns_per_token", path: "all", split: false, diffExpanded: false };
  const ns = ["http:", "", "www.w3.org", "2000", "svg"].join("/");
  const $ = (id) => document.getElementById(id);
  const fmt = (value, digits = 2) => value == null ? "n/a" : Number(value).toLocaleString(undefined, { maximumFractionDigits: digits });
  const pct = (value) => value == null ? "n/a" : `${fmt(value * 100, 1)}%`;
  const fmtNs = (value, digits = 2) => {
    if (value == null) return "n/a";
    const abs = Math.abs(value);
    if (abs >= 1e9) return `${fmt(value / 1e9, digits)} s`;
    if (abs >= 1e6) return `${fmt(value / 1e6, digits)} ms`;
    if (abs >= 1e3) return `${fmt(value / 1e3, digits)} µs`;
    return `${fmt(value, 0)} ns`;
  };
  const PHASE_ABBR = {
    scheduler_plan: "plan", input_staging: "stage", draft_prepare: "d-prep",
    draft_compute: "draft", proposal_select: "select", target_graph_build: "build",
    metadata_upload: "upload", target_compute: "target", readback_sync: "readback",
    acceptance: "accept", state_promotion: "promote", sampling_commit: "sample",
    output_processing: "output", client_flush: "flush", unattributed: "unattr",
  };
  const abbrFor = (phase) => PHASE_ABBR[phase] || (phase.startsWith("overlap(") ? "overlap" : phase.slice(0, 8));
  const runLabel = (run) => `${run.run.model_name || "unknown model"} · ${run.device.name} · ${run.run.git_sha || "unknown SHA"}`;
  const isoStart = (run) => {
    const unixNs = Number(run.run.started_unix_ns);
    if (!Number.isFinite(unixNs) || unixNs <= 0) return "unknown";
    const started = new Date(unixNs / 1000000);
    return Number.isNaN(started.getTime()) ? "unknown" : started.toISOString();
  };

  function element(name, attrs = {}, text = null) {
    const node = document.createElementNS(ns, name);
    Object.entries(attrs).forEach(([key, value]) => node.setAttribute(key, value));
    if (text != null) node.textContent = text;
    return node;
  }

  function chips() {
    $("identity").textContent = baseline
      ? `Baseline ${runLabel(baseline)} · Current ${runLabel(current)}`
      : runLabel(current);
    const values = [
      [current.capture.complete ? "complete capture" : "incomplete capture", current.capture.complete ? "good" : "bad"],
      [`${current.capture.rounds} rounds`, ""],
      [`${current.capture.requests} requests`, ""],
      [`backend ${current.run.runtime_backend || "unknown"}`, ""],
      [`arch ${current.run.arch || "unknown"}`, ""],
      [`configured C=${current.run.max_concurrency ?? "unknown"}`, ""],
      [`started ${isoStart(current)}`, ""],
    ];
    const treeWidths = current.speculation.tree_widths || [];
    if (treeWidths.length) values.push([`tree width ${treeWidths.join("/")}`, ""]);
    const durableTokens = current.phase_groups.all.reduce((sum, group) => sum + group.durable_tokens, 0);
    const boundsNs = current.capture_bounds.duration_ns;
    if (durableTokens && boundsNs) {
      values.push([`${fmt(durableTokens / (boundsNs / 1e9), 0)} durable tok/s overall`, "good"]);
    }
    if (current.mixed_run_cohorts) values.push(["mixed-run cohorts", "warn"]);
    if (baseline) values.push(["baseline comparison", "good"]);
    if (data.diff && data.diff.warnings.length) values.push([`${data.diff.warnings.length} mismatch warning${data.diff.warnings.length === 1 ? "" : "s"}`, "bad"]);
    $("chips").replaceChildren(...values.map(([text, cls]) => {
      const node = document.createElement("span");
      node.className = `chip ${cls}`;
      node.textContent = text;
      return node;
    }));
  }

  function pathOptions() {
    const paths = new Set(Object.keys(current.phase_groups.paths));
    if (baseline) Object.keys(baseline.phase_groups.paths).forEach((path) => paths.add(path));
    $("path").replaceChildren(...["all", ...Array.from(paths).sort()].map((path) => {
      const option = document.createElement("option");
      option.value = path;
      option.textContent = path === "all" ? "all paths" : path;
      return option;
    }));
  }

  function groupsFor(run) {
    if (state.path !== "all") return run.phase_groups.paths[state.path] || [];
    if (!state.split) return run.phase_groups.all;
    return Object.values(run.phase_groups.paths).flat().sort((a, b) => a.cohort - b.cohort || a.path.localeCompare(b.path));
  }

  function metric(phase) {
    return phase[state.normalization];
  }

  function commonScale() {
    if (state.normalization === "wall_share") return 1;
    const runs = baseline ? [baseline, current] : [current];
    return Math.max(1, ...runs.flatMap((run) => groupsFor(run).map((group) =>
      group.phases.reduce((sum, phase) => sum + (metric(phase) || 0), 0)
    )));
  }

  function patternDefs(svg, prefix) {
    const defs = element("defs");
    const compute = element("pattern", { id: `${prefix}-compute`, width: 8, height: 8, patternUnits: "userSpaceOnUse", patternTransform: "rotate(35)" });
    compute.append(element("rect", { width: 8, height: 8, fill: "#f8a903" }));
    compute.append(element("line", { x1: 0, y1: 0, x2: 0, y2: 8, stroke: "#ffc34a", "stroke-width": 3 }));
    const idle = element("pattern", { id: `${prefix}-idle`, width: 7, height: 7, patternUnits: "userSpaceOnUse", patternTransform: "rotate(45)" });
    idle.append(element("rect", { width: 7, height: 7, fill: "#571f3a" }));
    idle.append(element("line", { x1: 0, y1: 0, x2: 0, y2: 7, stroke: "#ff5c70", "stroke-width": 2 }));
    const mixed = element("pattern", { id: `${prefix}-mixed`, width: 8, height: 8, patternUnits: "userSpaceOnUse", patternTransform: "rotate(-35)" });
    mixed.append(element("rect", { width: 8, height: 8, fill: "#5aa9e6" }));
    mixed.append(element("line", { x1: 0, y1: 0, x2: 0, y2: 8, stroke: "#f8a903", "stroke-width": 3 }));
    defs.append(compute, idle, mixed);
    svg.append(defs);
  }

  function fillFor(classification, prefix) {
    return {
      bandwidth: "#5aa9e6",
      compute: `url(#${prefix}-compute)`,
      overhead: "#5d7293",
      idle: `url(#${prefix}-idle)`,
      mixed: `url(#${prefix}-mixed)`,
      neutral: "#9db2d0",
    }[classification.class] || "#9db2d0";
  }

  function phaseText(group, phase) {
    const facts = phase.classification;
    const parts = [
      `${group.label} · ${group.path} · ${phase.phase}`,
      `${fmtNs(phase.total_ns)} total · ${fmtNs(phase.ns_per_round)}/round`,
      `${fmtNs(phase.ns_per_token)}/durable token · ${fmtNs(phase.ns_per_serviced_token)}/serviced token`,
      `${pct(phase.wall_share)} of cohort wall`,
      `${group.rounds} rounds · ${group.durable_tokens} durable + ${group.executed_prefill_tokens} prefill tokens`,
      `class ${facts.class}${facts.inherited_from ? ` (from ${facts.inherited_from})` : ""}`,
    ];
    if (facts.arithmetic_intensity_flops_per_byte != null) {
      parts.push(`AI ${fmt(facts.arithmetic_intensity_flops_per_byte)} FLOP/B · machine balance ${fmt(facts.machine_balance_flops_per_byte)} FLOP/B · headroom ${fmt(facts.headroom)}×`);
    }
    if (facts.per_path) {
      Object.entries(facts.per_path).forEach(([path, entry]) => {
        const intensity = entry.arithmetic_intensity_flops_per_byte;
        parts.push(`${path}: ${entry.class}${intensity != null ? ` · AI ${fmt(intensity)} FLOP/B` : ""}`);
      });
    }
    if (facts.padding_note) parts.push(facts.padding_note);
    return parts.join("\n");
  }

  function showTooltip(event, text) {
    const tip = $("tooltip");
    tip.textContent = text;
    tip.hidden = false;
    const rect = event.currentTarget.getBoundingClientRect();
    tip.style.left = `${Math.min(window.innerWidth - 380, Math.max(8, rect.left + rect.width + 8))}px`;
    tip.style.top = `${Math.min(window.innerHeight - tip.offsetHeight - 8, Math.max(8, rect.top))}px`;
  }

  function hideTooltip() { $("tooltip").hidden = true; }

  function showDetail(group, phase) {
    const facts = phase.classification;
    const roofline = facts.arithmetic_intensity_flops_per_byte == null
      ? facts.reason
      : `AI ${fmt(facts.arithmetic_intensity_flops_per_byte)} FLOP/B. Machine balance ${fmt(facts.machine_balance_flops_per_byte)} FLOP/B. Headroom ${fmt(facts.headroom)}×.`;
    const detail = $("detail");
    const heading = document.createElement("strong");
    heading.textContent = `${group.label} · ${group.path} · ${phase.phase}`;
    const distribution = `Zero-inclusive round distribution. p50 ${fmtNs(phase.p50_ns)}. p95 ${fmtNs(phase.p95_ns)}.`;
    const rooflineText = `${roofline}${facts.padding_note ? ` ${facts.padding_note}.` : ""}`;
    detail.replaceChildren(
      heading, document.createElement("br"), document.createTextNode(distribution),
      document.createElement("br"), document.createTextNode(rooflineText),
    );
  }

  function wallSvg(run, title, prefix, scale) {
    const groups = groupsFor(run);
    const width = Math.max(460, groups.length * 92 + 90);
    const height = 368;
    const plot = { left: 64, top: 18, width: width - 80, height: 275 };
    const svg = element("svg", { viewBox: `0 0 ${width} ${height}`, class: "wall-svg", role: "img", "aria-label": `${title} phase budget` });
    patternDefs(svg, prefix);
    for (let tick = 0; tick <= 4; tick += 1) {
      const y = plot.top + plot.height - plot.height * tick / 4;
      svg.append(element("line", { x1: plot.left, y1: y, x2: plot.left + plot.width, y2: y, class: "gridline" }));
      const value = scale * tick / 4;
      svg.append(element("text", { x: plot.left - 6, y: y + 4, "text-anchor": "end", class: "axis" }, state.normalization === "wall_share" ? pct(value) : fmtNs(value, 1)));
    }
    const slot = plot.width / Math.max(1, groups.length);
    const barWidth = Math.min(58, slot * .68);
    groups.forEach((group, index) => {
      const x = plot.left + slot * index + (slot - barWidth) / 2;
      let y = plot.top + plot.height;
      group.phases.forEach((phase) => {
        const value = metric(phase) || 0;
        if (value <= 0) return;
        const segmentHeight = plot.height * value / scale;
        y -= segmentHeight;
        const label = phaseText(group, phase);
        const rect = element("rect", {
          x, y, width: barWidth, height: Math.max(.8, segmentHeight),
          fill: fillFor(phase.classification, prefix),
          class: "segment",
          tabindex: 0, role: "button", "aria-label": label.replaceAll("\n", ". "),
        });
        rect.append(element("title", {}, label));
        rect.addEventListener("mouseenter", (event) => showTooltip(event, label));
        rect.addEventListener("focus", (event) => showTooltip(event, label));
        rect.addEventListener("mouseleave", hideTooltip);
        rect.addEventListener("blur", hideTooltip);
        const select = () => showDetail(group, phase);
        rect.addEventListener("click", select);
        rect.addEventListener("keydown", (event) => {
          if (event.key === "Enter" || event.key === " ") { event.preventDefault(); select(); }
        });
        svg.append(rect);
        if (segmentHeight >= 13) {
          svg.append(element("text", {
            x: x + barWidth / 2, y: y + segmentHeight / 2 + 3.5,
            "text-anchor": "middle", class: "segment-label",
          }, abbrFor(phase.phase)));
        }
      });
      const baselineY = plot.top + plot.height;
      if (group.cohort === 1) {
        svg.append(element("line", { x1: x, y1: baselineY + 6, x2: x + barWidth, y2: baselineY + 6, class: "anchor" }));
      }
      const tokRate = group.total_ns > 0 && group.durable_tokens > 0
        ? `${fmt(group.durable_tokens / (group.total_ns / 1e9), 0)} t/s`
        : "0 t/s";
      const lines = [group.label];
      if (group.path !== "all") lines.push(group.path);
      lines.push(`n=${group.rounds}`, tokRate);
      lines.forEach((line, lineIndex) => {
        svg.append(element("text", {
          x: x + barWidth / 2, y: baselineY + 22 + lineIndex * 13,
          "text-anchor": "middle",
          class: `bar-label ${group.cohort === 1 && lineIndex === 0 ? "anchor-note" : ""}`,
        }, line));
      });
    });
    return svg;
  }

  function renderWallLegend() {
    const host = $("wall-legend");
    const entries = [
      ["bandwidth-bound", "bandwidth"], ["compute-bound", "compute"],
      ["host overhead", "overhead"], ["idle / unattributed", "idle"],
      ["mixed paths", "mixed"], ["no device spec", "neutral"],
    ];
    host.replaceChildren(...entries.map(([label, cls], index) => {
      const item = document.createElement("span");
      const swatch = element("svg", { viewBox: "0 0 12 12" });
      patternDefs(swatch, `lg${index}`);
      swatch.append(element("rect", { width: 12, height: 12, fill: fillFor({ class: cls }, `lg${index}`) }));
      item.append(swatch, document.createTextNode(label));
      return item;
    }));
  }

  function renderWalls() {
    const scale = commonScale();
    const cards = [];
    const runs = baseline ? [[baseline, `Baseline · ${runLabel(baseline)}`, "baseline"], [current, `Current · ${runLabel(current)}`, "current"]] : [[current, runLabel(current), "current"]];
    runs.forEach(([run, title, prefix]) => {
      const card = document.createElement("div");
      card.className = "wall-card";
      const heading = document.createElement("h3");
      heading.textContent = title;
      card.append(heading, wallSvg(run, title, prefix, scale));
      cards.push(card);
    });
    $("walls").replaceChildren(...cards);
    const idle = current.phase_groups.inter_round_idle;
    $("idle").textContent = `Inter-round idle · ${fmtNs(idle.total_ns)} total · p50 ${fmtNs(idle.p50_ns)} · p95 ${fmtNs(idle.p95_ns)}. ${idle.note}`;
  }

  function renderNotices() {
    const messages = [...current.notices];
    if (baseline) baseline.notices.forEach((notice) => {
      messages.push({ ...notice, message: `Baseline. ${notice.message}` });
    });
    if (current.mixed_run_cohorts) messages.push({ message: "Mixed-run cohorts include admission and tail-drain states. Use dedicated fixed-C captures for honest cohort comparisons." });
    $("notices").replaceChildren(...messages.map((notice) => {
      const node = document.createElement("div");
      node.className = "notice";
      node.textContent = notice.message;
      return node;
    }));
  }

  function renderWaterfall() {
    const requests = current.requests;
    const rows = requests.rows.slice(0, requests.display_limit);
    const width = 1180;
    const rowHeight = 18;
    const labelWidth = 90;
    const axisHeight = 22;
    const plotWidth = width - labelWidth - 12;
    const duration = Math.max(1, requests.end_ns - requests.origin_ns);
    const svg = $("waterfall");
    svg.setAttribute("viewBox", `0 0 ${width} ${Math.max(60, rows.length * rowHeight + axisHeight + 12)}`);
    svg.replaceChildren();
    const colors = { queue_ns: "#5d7293", prefill_ns: "#5aa9e6", first_decode_ns: "#ffc34a", decode_ns: "#48d597" };
    const segmentNames = { queue_ns: "queue", prefill_ns: "prefill", first_decode_ns: "first token", decode_ns: "decode" };
    const bottom = axisHeight + rows.length * rowHeight;
    for (let tick = 0; tick <= 4; tick += 1) {
      const tickX = labelWidth + plotWidth * tick / 4;
      svg.append(element("line", { x1: tickX, y1: axisHeight - 6, x2: tickX, y2: bottom, class: "gridline" }));
      svg.append(element("text", { x: tickX, y: 10, "text-anchor": tick === 0 ? "start" : "middle", class: "axis" }, `${fmt(duration * tick / 4 / 1e9, 1)} s`));
    }
    rows.forEach((row, index) => {
      const group = element("g", { class: `request ${row.ok === false ? "failed" : ""} ${row.open_ended ? "incomplete" : ""}` });
      const y = axisHeight + index * rowHeight;
      const status = row.ok === false ? "failed" : row.open_ended ? "incomplete" : "ok";
      const spans = Object.keys(colors)
        .map((key) => `${segmentNames[key]} ${fmtNs(row[key])}`)
        .join(" · ");
      group.append(element("title", {}, `#${row.request_id} · ${status} · prompt ${row.prompt_tokens} tokens · output ${row.output_tokens} tokens\n${spans}`));
      group.append(element("text", { x: 0, y: y + 11, class: "axis" }, `#${row.request_id}`));
      let cursor = row.start_offset_ns;
      Object.keys(colors).forEach((key) => {
        const value = row[key];
        if (value == null) return;
        const x = labelWidth + plotWidth * cursor / duration;
        const w = Math.max(1, plotWidth * value / duration);
        group.append(element("rect", { x, y, width: w, height: 12, rx: 2, fill: colors[key] }));
        cursor += value;
      });
      svg.append(group);
    });
    const shown = Math.min(rows.length, requests.display_limit);
    $("request-note").textContent = `showing first ${shown} of ${requests.total}${requests.embedded < requests.total ? ` · ${requests.embedded} embedded` : ""}`;
    const legendNote = document.createElement("span");
    legendNote.textContent = "red outline = failed · dashed = incomplete";
    $("wf-legend").replaceChildren(...Object.keys(colors).map((key) => {
      const item = document.createElement("span");
      const swatch = document.createElement("span");
      swatch.className = "sw";
      swatch.style.background = colors[key];
      item.append(swatch, document.createTextNode(segmentNames[key]));
      return item;
    }), legendNote);
  }

  function renderLatency() {
    const latency = current.latency_ms;
    const pair = (values) => `${fmt(values.p50)}/${fmt(values.p95)}`;
    $("latency").textContent = `ms p50/p95 — queue ${pair(latency.queue)} · TTFT ${pair(latency.ttft)} · e2e ${pair(latency.end_to_end)} · inter-token ${pair(latency.inter_burst_per_token)}`;
  }

  function renderFunnel() {
    const speculation = current.speculation;
    const stages = [
      ["eligible", "spec_eligible_lanes"], ["reserved", "spec_reserved_lanes"],
      ["attempted", "spec_attempted_lanes"], ["proposed", "spec_proposed_draft_tokens"],
      ["verified", "spec_verified_draft_tokens"], ["accepted", "spec_accepted_draft_tokens"],
      ["durable", "spec_durable_draft_tokens"], ["consumed", "spec_scheduler_consumed_tokens"],
    ];
    const hasSpeculation = stages.some(([, key]) => speculation[key] > 0);
    if (!hasSpeculation) {
      const empty = document.createElement("p");
      empty.className = "muted";
      empty.textContent = "Speculation was disabled or no speculative work was captured.";
      $("funnel").replaceChildren(empty);
    } else {
      let previous = null;
      $("funnel").replaceChildren(...stages.map(([label, key]) => {
        const value = speculation[key];
        const step = document.createElement("div");
        step.className = "funnel-step";
        const conversion = label !== "proposed" && previous && previous.value ? value / previous.value : null;
        step.innerHTML = `<span>${label}</span><b>${fmt(value, 0)}</b><small>${conversion == null ? "anchor" : `${pct(conversion)} from prior`}</small>`;
        previous = { value };
        return step;
      }));
    }
    const decisions = Object.entries(speculation.decisions || {}).sort((a, b) => b[1] - a[1]);
    $("decisions").textContent = decisions.length
      ? `decode-lane speculation decisions: ${decisions.map(([name, count]) => `${name} ${fmt(count, 0)}`).join(" · ")}`
      : "";
    const svg = $("acceptance");
    // Index 0 is the draft root; real speculative positions start at 1.
    const values = (speculation.acceptance_by_position || []).slice(1);
    const width = 360;
    const height = 150;
    svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
    svg.replaceChildren(element("text", { x: 0, y: 12, class: "axis" }, "acceptance by position"));
    values.forEach((value, index) => {
      const barWidth = Math.max(8, (width - 20) / Math.max(1, values.length) - 5);
      const x = 10 + index * (barWidth + 5);
      const h = (value || 0) * 105;
      svg.append(element("rect", { x, y: 125 - h, width: barWidth, height: h, fill: "#f8a903", rx: 2 }));
      svg.append(element("text", { x: x + barWidth / 2, y: 142, "text-anchor": "middle", class: "axis" }, String(index + 1)));
    });
  }

  function renderDiff() {
    if (!data.diff) return;
    $("diff-panel").hidden = false;
    $("diff-warnings").replaceChildren(...data.diff.warnings.map((message) => {
      const node = document.createElement("div");
      node.className = "notice";
      node.textContent = message;
      return node;
    }));
    const rows = data.diff.rows;
    const visible = state.diffExpanded ? rows : rows.slice(0, 20);
    $("delta-rows").replaceChildren(...visible.map((row) => {
      const tr = document.createElement("tr");
      const values = [
        row.path, `C=${row.cohort} · ${row.phase}`,
        fmtNs(row.baseline_ns_per_token), fmtNs(row.current_ns_per_token),
        fmtNs(row.delta_ns_per_token), pct(row.delta_percent),
      ];
      values.forEach((value, index) => {
        const td = document.createElement("td");
        td.textContent = value;
        if (index >= 4 && row.delta_ns_per_token != null) td.className = row.delta_ns_per_token > 0 ? "positive" : row.delta_ns_per_token < 0 ? "negative" : "";
        tr.append(td);
      });
      return tr;
    }));
    const expand = $("delta-expand");
    if (rows.length > 20) {
      expand.hidden = false;
      expand.textContent = state.diffExpanded ? "Show top 20" : `Show all ${rows.length}`;
    } else {
      expand.hidden = true;
    }
  }

  function footer() {
    const capture = current.capture;
    const formatBounds = (run, label) => {
      const bounds = run.capture_bounds;
      const earliest = bounds.earliest_steady_ns == null ? "unknown" : `${fmt(bounds.earliest_steady_ns, 0)} ns`;
      const latest = bounds.latest_steady_ns == null ? "unknown" : `${fmt(bounds.latest_steady_ns, 0)} ns`;
      const duration = bounds.duration_ns == null ? "unknown" : fmtNs(bounds.duration_ns);
      return `${label} steady bounds ${earliest} to ${latest}. Duration ${duration}.`;
    };
    const boundsText = baseline
      ? `${formatBounds(baseline, "Baseline")} ${formatBounds(current, "Current")}`
      : formatBounds(current, "Current");
    $("footer").textContent = `Retention ${current.run.retention_policy || "unknown"}. Dropped steps ${capture.dropped_steps}, requests ${capture.dropped_requests}, token bursts ${capture.dropped_token_bursts}. Capture started at Unix ns ${current.run.started_unix_ns || "unknown"}. ${boundsText}`;
  }

  $("normalization").addEventListener("change", (event) => { state.normalization = event.target.value; renderWalls(); });
  $("path").addEventListener("change", (event) => { state.path = event.target.value; renderWalls(); });
  $("split").addEventListener("click", (event) => {
    state.split = !state.split;
    event.currentTarget.setAttribute("aria-pressed", String(state.split));
    renderWalls();
  });
  $("delta-expand").addEventListener("click", () => {
    state.diffExpanded = !state.diffExpanded;
    renderDiff();
  });

  chips();
  pathOptions();
  renderNotices();
  renderWallLegend();
  renderWalls();
  renderWaterfall();
  renderLatency();
  renderFunnel();
  renderDiff();
  footer();
}());
</script>
</body>
</html>
'''
