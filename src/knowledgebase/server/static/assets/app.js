// knowledgeBase atlas — lazy-loaded concept map.
// The browser never downloads the whole graph: it starts with the most important
// concepts (/api/graph/overview) and merges a neighbourhood in each time you open a
// concept (/api/graph/node/:id).

const $ = (sel) => document.querySelector(sel);
const prefersReducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
const ANIM = prefersReducedMotion ? 0 : 600;
const NEIGHBOURS_ON_MAP = 16;

// Atlas tints for regions (communities); paired with a darker ink for node fills.
const TINTS_LIGHT = ["#cfe0c3", "#f0d9a8", "#e6c7d3", "#c7d9e8", "#ddd0ea", "#c9e4dc", "#ead9c0", "#f1cdbd", "#d8dfb4", "#cbd3ea", "#e9c9c0", "#d2e6ef"];
const TINTS_DARK = ["#3e5a3a", "#6b5a2e", "#5e3c4b", "#33506a", "#4c3f63", "#2f5a52", "#5e4e36", "#6a4436", "#4f5a2f", "#3b4670", "#64403a", "#2f5866"];
const NODE_LIGHT = ["#5d8a4a", "#b2862a", "#a55878", "#3f7aa6", "#7a5fa6", "#3d8a78", "#9c7a4a", "#b56446", "#7d8a32", "#5468a8", "#a35b4f", "#3c8aa6"];
const NODE_DARK = ["#8fbf7a", "#e0b75c", "#d68aa9", "#7fb2dc", "#b49be0", "#76c4b0", "#d1ab78", "#e4977a", "#b3c066", "#93a5e6", "#da9184", "#7cc2dc"];

const state = {
  nodes: new Map(),          // id -> node (force-graph adds x/y)
  links: new Map(),          // "a|b" -> link
  communities: new Map(),    // id -> {id,label,size,top_nodes}
  selected: null,
  hover: null,
  neighbourhood: new Set(),
  route: { nodes: new Set(), links: new Set() },
  region: null,              // region filter id
  maxImportance: 1,
  theme: null,
  history: [],               // panel back-stack
};

// ------------------------------------------------------------------ utils
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const linkKey = (a, b) => (a < b ? `${a}|${b}` : `${b}|${a}`);
const idOf = (end) => (typeof end === "object" ? end.id : end);
const fmtMonth = (m) => {
  if (!m) return "";
  const [y, mo] = m.split("-");
  return new Date(Number(y), Number(mo) - 1, 1).toLocaleDateString(undefined, { month: "short", year: "numeric" });
};
const fmtDate = (d) => {
  if (!d || d === "unknown-date") return "undated";
  const dt = new Date(d + "T00:00:00");
  return isNaN(dt) ? d : dt.toLocaleDateString(undefined, { day: "numeric", month: "short", year: "numeric" });
};
const platformName = (p) => ({ chatgpt: "ChatGPT", claude: "Claude", gemini: "Gemini", "claude-code": "Claude Code" }[p] || p);

async function api(path) {
  const res = await fetch(path);
  if (res.status === 503) throw Object.assign(new Error("not indexed"), { code: 503 });
  if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || `Request failed (${res.status})`);
  return res.json();
}

function cssVar(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }
function isDark() {
  const forced = document.documentElement.dataset.theme;
  if (forced) return forced === "dark";
  return window.matchMedia("(prefers-color-scheme: dark)").matches;
}
function palette() {
  const dark = isDark();
  return {
    paper: cssVar("--paper"), ink: cssVar("--ink"), inkSoft: cssVar("--ink-soft"), route: cssVar("--route"),
    select: cssVar("--select"), survey: cssVar("--survey"), halo: cssVar("--halo"),
    tints: dark ? TINTS_DARK : TINTS_LIGHT, fills: dark ? NODE_DARK : NODE_LIGHT,
  };
}
let colors = palette();
const regionTint = (c) => (c == null || c < 0 ? colors.route : colors.tints[c % colors.tints.length]);
const regionFill = (c) => (c == null || c < 0 ? colors.route : colors.fills[c % colors.fills.length]);
function regionName(c) {
  const r = state.communities.get(c);
  return r ? r.label : "Unmapped";
}

// ------------------------------------------------------------------ graph
const mapEl = $("#map");
const Graph = ForceGraph()(mapEl)
  .backgroundColor("rgba(0,0,0,0)")
  .nodeId("id")
  .nodeRelSize(1)
  .nodeVal((n) => radius(n) ** 2)
  .nodeLabel(() => "")
  .linkSource("source")
  .linkTarget("target")
  .cooldownTicks(prefersReducedMotion ? 60 : 220)
  .d3VelocityDecay(0.32)
  .minZoom(0.2)
  .maxZoom(8)
  .onNodeClick((n) => openConcept(n.id))
  .onNodeHover((n) => { state.hover = n ? n.id : null; mapEl.style.cursor = n ? "pointer" : ""; })
  .onBackgroundClick(() => clearSelection())
  .linkColor(linkColor)
  .linkWidth(linkWidth)
  .linkLineDash((l) => (state.route.links.has(linkKey(idOf(l.source), idOf(l.target))) ? [5, 3] : null))
  .nodeCanvasObjectMode(() => "replace")
  .nodeCanvasObject(drawNode)
  .nodePointerAreaPaint((n, color, ctx) => {
    ctx.fillStyle = color; ctx.beginPath(); ctx.arc(n.x, n.y, radius(n) + 3, 0, 2 * Math.PI); ctx.fill();
  })
  .onRenderFramePre(drawTerritories)
  .onEngineStop(() => { if (fitPending) { fitPending = false; Graph.zoomToFit(ANIM, 90); } });
let fitPending = false;

Graph.d3Force("charge").strength(-180).distanceMax(380);

// Each region gets an anchor on a sunflower spiral (largest region in the middle) and a gentle
// pull towards it, so territories separate like countries instead of piling on top of each other.
const anchors = new Map();
function layoutAnchors(visibleNodes) {
  anchors.clear();
  // Only regions that are actually on the map get a slot, biggest (by visible places) in the middle.
  const counts = new Map();
  for (const n of visibleNodes) counts.set(n.community, (counts.get(n.community) || 0) + 1);
  const order = [...state.communities.values()]
    .filter((c) => counts.has(c.id))
    .sort((a, b) => counts.get(b.id) - counts.get(a.id) || b.size - a.size);
  const golden = Math.PI * (3 - Math.sqrt(5));
  order.forEach((c, i) => {
    const r = 175 * Math.sqrt(i);
    anchors.set(c.id, [r * Math.cos(i * golden), r * Math.sin(i * golden)]);
  });
}
function regionForce() {
  let nodes = [];
  const force = (alpha) => {
    for (const n of nodes) {
      const a = anchors.get(n.community);
      if (!a || n.x == null) continue;
      n.vx += (a[0] - n.x) * 0.16 * alpha;
      n.vy += (a[1] - n.y) * 0.16 * alpha;
    }
    // Collision: keep places (and room for their names) from overlapping.
    for (let i = 0; i < nodes.length; i++) {
      const a = nodes[i];
      if (a.x == null) continue;
      for (let j = i + 1; j < nodes.length; j++) {
        const b = nodes[j];
        if (b.x == null) continue;
        const dx = b.x - a.x, dy = b.y - a.y;
        const min = radius(a) + radius(b) + 24;
        const d2 = dx * dx + dy * dy;
        if (d2 >= min * min || d2 === 0) continue;
        const d = Math.sqrt(d2), push = ((min - d) / d) * 0.5;
        a.x -= dx * push; a.y -= dy * push; b.x += dx * push; b.y += dy * push;
      }
    }
  };
  force.initialize = (ns) => { nodes = ns; };
  return force;
}
Graph.d3Force("region", regionForce());
const sameRegion = (l) => {
  const a = typeof l.source === "object" ? l.source : state.nodes.get(l.source);
  const b = typeof l.target === "object" ? l.target : state.nodes.get(l.target);
  return a && b && a.community === b.community;
};
// Routes inside a territory are short and firm; routes between territories are long and loose.
Graph.d3Force("link")
  .distance((l) => (sameRegion(l) ? 55 : 130) + 30 * (1 - Math.min(1, (l.strength || 0) / 3)))
  .strength((l) => (sameRegion(l) ? 0.4 : 0.04));

function resize() { Graph.width(mapEl.clientWidth).height(mapEl.clientHeight); }
window.addEventListener("resize", resize);

function radius(n) {
  const imp = Math.max(0, n.importance || 0) / (state.maxImportance || 1);
  return 3.2 + 10 * Math.sqrt(imp);
}

function visibleByRegion(n) { return state.region == null || n.community === state.region; }

function drawNode(n, ctx, scale) {
  const r = radius(n);
  const isSel = n.id === state.selected;
  const inHood = state.neighbourhood.has(n.id) || n.id === state.hover;
  const onRoute = state.route.nodes.has(n.id);
  const dimmed = (state.selected && !inHood && !isSel && !onRoute) || !visibleByRegion(n);
  ctx.globalAlpha = dimmed ? 0.28 : 1;

  ctx.beginPath();
  ctx.arc(n.x, n.y, r, 0, 2 * Math.PI);
  ctx.fillStyle = regionFill(n.community);
  ctx.fill();
  ctx.lineWidth = 1.2 / scale;
  ctx.strokeStyle = colors.paper;
  ctx.stroke();
  if (isSel || onRoute) {
    ctx.beginPath();
    ctx.arc(n.x, n.y, r + 3.5 / scale + 1.5, 0, 2 * Math.PI);
    ctx.lineWidth = 2.2 / scale;
    ctx.strokeStyle = onRoute && !isSel ? colors.survey : colors.select;
    ctx.stroke();
  }

  // Map lettering: always for big places and the active neighbourhood, otherwise once zoomed in.
  const important = (n.importance || 0) / (state.maxImportance || 1) > 0.32;
  if (isSel || inHood || onRoute || important || scale > 1.6) {
    const size = (isSel ? 15 : important ? 13.5 : 12) / scale;
    ctx.font = `${isSel || important ? 700 : 500} ${size}px "Barlow Semi Condensed", "Arial Narrow", sans-serif`;
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    const y = n.y + r + 2.5 / scale;
    ctx.lineJoin = "round";
    ctx.lineWidth = 3.4 / scale;
    ctx.strokeStyle = colors.halo;
    ctx.strokeText(n.label, n.x, y);
    ctx.fillStyle = colors.ink;
    ctx.fillText(n.label, n.x, y);
  }
  ctx.globalAlpha = 1;
}

function linkColor(l) {
  const key = linkKey(idOf(l.source), idOf(l.target));
  if (state.route.links.has(key)) return colors.survey;
  const s = idOf(l.source), t = idOf(l.target);
  const focus = state.selected || state.hover;
  if (focus && (s === focus || t === focus)) return colors.ink;
  return hexAlpha(colors.route, state.selected ? 0.18 : sameRegion(l) ? 0.6 : 0.22);
}
function linkWidth(l) {
  const key = linkKey(idOf(l.source), idOf(l.target));
  if (state.route.links.has(key)) return 2.6;
  const focus = state.selected || state.hover;
  const base = 0.5 + Math.min(2.2, (l.strength || 0) * 0.7);
  return focus && (idOf(l.source) === focus || idOf(l.target) === focus) ? base + 0.6 : base;
}
function hexAlpha(hex, a) {
  const h = hex.replace("#", "");
  if (h.length !== 6) return hex;
  const [r, g, b] = [0, 2, 4].map((i) => parseInt(h.slice(i, i + 2), 16));
  return `rgba(${r},${g},${b},${a})`;
}

// Territories: a soft blob around each region's visible places + italic region lettering.
function drawTerritories(ctx, scale) {
  const groups = new Map();
  for (const n of state.nodes.values()) {
    if (n.x == null || n.community == null || n.community < 0) continue;
    if (!groups.has(n.community)) groups.set(n.community, []);
    groups.get(n.community).push(n);
  }
  for (const [c, members] of groups) {
    if (members.length < 3) continue;
    const pad = 22;
    const pts = [];
    for (const m of members) {
      const r = radius(m) + pad;
      for (let k = 0; k < 8; k++) {
        const a = (k / 8) * 2 * Math.PI;
        pts.push([m.x + r * Math.cos(a), m.y + r * Math.sin(a)]);
      }
    }
    const hull = convexHull(pts);
    if (hull.length < 3) continue;
    const dim = state.region != null && state.region !== c;
    ctx.globalAlpha = dim ? 0.18 : 0.55;
    ctx.fillStyle = regionTint(c);
    ctx.beginPath();
    smoothPath(ctx, hull);
    ctx.fill();
    ctx.globalAlpha = dim ? 0.15 : 0.5;
    ctx.lineWidth = 1 / scale;
    ctx.strokeStyle = regionFill(c);
    ctx.setLineDash([3 / scale, 3 / scale]);
    ctx.stroke();
    ctx.setLineDash([]);

    // Region name above the territory, in atlas italics, constant on-screen size.
    const top = hull.reduce((a, p) => (p[1] < a[1] ? p : a), hull[0]);
    const cx = members.reduce((s, m) => s + m.x, 0) / members.length;
    const size = 16 / scale;
    ctx.globalAlpha = dim ? 0.25 : 0.9;
    ctx.font = `italic 500 ${size}px Spectral, Georgia, serif`;
    ctx.textAlign = "center";
    ctx.textBaseline = "bottom";
    ctx.fillStyle = colors.inkSoft;
    ctx.fillText(regionName(c), cx, top[1] - 4 / scale);
    ctx.globalAlpha = 1;
  }
}

function convexHull(points) {
  const p = points.slice().sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  const cross = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const lower = [];
  for (const pt of p) { while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], pt) <= 0) lower.pop(); lower.push(pt); }
  const upper = [];
  for (let i = p.length - 1; i >= 0; i--) { const pt = p[i]; while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], pt) <= 0) upper.pop(); upper.push(pt); }
  upper.pop(); lower.pop();
  return lower.concat(upper);
}
function smoothPath(ctx, pts) {
  const mid = (a, b) => [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];
  const start = mid(pts[pts.length - 1], pts[0]);
  ctx.moveTo(start[0], start[1]);
  for (let i = 0; i < pts.length; i++) {
    const next = pts[(i + 1) % pts.length];
    const m = mid(pts[i], next);
    ctx.quadraticCurveTo(pts[i][0], pts[i][1], m[0], m[1]);
  }
  ctx.closePath();
}

// ------------------------------------------------------------------ data merge
function addNode(n, near) {
  const existing = state.nodes.get(n.id);
  if (existing) { Object.assign(existing, n, { x: existing.x, y: existing.y }); return existing; }
  const node = { ...n };
  const anchor = anchors.get(n.community);
  if (near && near.x != null) {
    const a = Math.random() * 2 * Math.PI;
    node.x = near.x + 30 * Math.cos(a);
    node.y = near.y + 30 * Math.sin(a);
  } else if (anchor) {
    node.x = anchor[0] + (Math.random() - 0.5) * 80;
    node.y = anchor[1] + (Math.random() - 0.5) * 80;
  }
  state.nodes.set(n.id, node);
  state.maxImportance = Math.max(state.maxImportance, n.importance || 0);
  return node;
}
function addLink(e) {
  if (!state.nodes.has(e.source) || !state.nodes.has(e.target)) return;
  const key = linkKey(e.source, e.target);
  if (!state.links.has(key)) state.links.set(key, { source: e.source, target: e.target, strength: e.strength });
}
function commit() {
  Graph.graphData({ nodes: [...state.nodes.values()], links: [...state.links.values()] });
}

async function loadOverview() {
  setStatus("Charting your most important concepts…");
  const data = await api("/api/graph/overview?limit=55");
  state.nodes.clear(); state.links.clear(); state.communities.clear();
  state.maxImportance = Math.max(...data.nodes.map((n) => n.importance || 0), 0.0001);
  data.communities.forEach((c) => state.communities.set(c.id, c));
  layoutAnchors(data.nodes);
  data.nodes.forEach((n) => addNode(n));  // each place starts inside its own territory
  data.edges.forEach(addLink);
  commit();
  renderRegions(data);
  setStatus(`${data.nodes.length} of ${data.total_nodes.toLocaleString()} concepts shown. Open one to see what it connects to.`);
  fitPending = true;  // fit once the layout has settled
  setTimeout(() => Graph.zoomToFit(ANIM, 90), prefersReducedMotion ? 50 : 700);
  return data;
}

function setStatus(text) { $("#map-status").textContent = text; }

// zoomToFit, but never closer than maxZoom (a 3-node route would otherwise fill the screen).
function fitTo(filter, padding = 120, maxZoom = 2.4) {
  Graph.zoomToFit(ANIM, padding, filter);
  setTimeout(() => { if (Graph.zoom() > maxZoom) Graph.zoom(maxZoom, ANIM / 2); }, ANIM + 30);
}

// ------------------------------------------------------------------ regions legend
function renderRegions(data) {
  const shown = new Set(data.nodes.map((n) => n.community));
  const list = $("#region-list");
  list.innerHTML = "";
  [...state.communities.values()]
    .filter((c) => shown.has(c.id))
    .slice(0, 14)
    .forEach((c) => {
      const li = document.createElement("li");
      li.innerHTML = `<button type="button" aria-pressed="false" data-region="${c.id}">
        <span class="swatch" style="background:${regionTint(c.id)}"></span><span>${esc(c.label)}</span>
        <span class="count">${c.size}</span></button>`;
      list.appendChild(li);
    });
}
$("#region-list").addEventListener("click", async (e) => {
  const btn = e.target.closest("button[data-region]");
  if (!btn) return;
  const id = Number(btn.dataset.region);
  const turningOff = state.region === id;
  state.region = turningOff ? null : id;
  document.querySelectorAll("#region-list button").forEach((b) => b.setAttribute("aria-pressed", String(Number(b.dataset.region) === state.region)));
  if (!turningOff) {
    const data = await api(`/api/graph/overview?limit=30&community=${id}`);
    if (!anchors.has(id)) layoutAnchors([...state.nodes.values(), ...data.nodes]);
    data.nodes.forEach((n) => addNode(n));
    data.edges.forEach(addLink);
    commit();
    const r = state.communities.get(id);
    setStatus(`${r ? r.label : "Region"}: ${r ? r.size : data.nodes.length} concepts in this region.`);
  }
  Graph.zoomToFit(ANIM, 90, (n) => state.region == null || n.community === state.region);
});

// ------------------------------------------------------------------ selection
function clearSelection() {
  state.selected = null;
  state.neighbourhood = new Set();
  state.route = { nodes: new Set(), links: new Set() };
  Graph.nodeColor(Graph.nodeColor()); // trigger redraw
}

async function ensureNode(id) {
  if (state.nodes.has(id)) return state.nodes.get(id);
  const detail = await api(`/api/graph/node/${encodeURIComponent(id)}?limit=8`);
  const node = addNode(detail.node);
  detail.neighbors.forEach((nb) => { if (state.nodes.has(nb.id)) addLink({ source: id, target: nb.id, strength: nb.edge.strength }); });
  commit();
  return node;
}

async function openConcept(id, { pushHistory = true } = {}) {
  setMode(null);
  showPanel(`<p class="loading">Opening the gazetteer…</p>`);
  let detail;
  try {
    detail = await api(`/api/graph/node/${encodeURIComponent(id)}?limit=30`);
  } catch (err) {
    showPanel(`<p>Couldn't open this concept: ${esc(err.message)}</p>`);
    return;
  }
  const centre = addNode(detail.node);
  detail.neighbors.slice(0, NEIGHBOURS_ON_MAP).forEach((nb) => addNode(nb, centre));
  detail.edges.forEach(addLink);
  commit();
  state.selected = detail.node.id;
  state.neighbourhood = new Set(detail.neighbors.slice(0, NEIGHBOURS_ON_MAP).map((n) => n.id));
  state.route = { nodes: new Set(), links: new Set() };
  if (pushHistory) state.history.push({ type: "concept", id });
  renderConcept(detail);
  setTimeout(() => {
    if (centre.x != null) { Graph.centerAt(centre.x, centre.y, ANIM); if (Graph.zoom() < 1.4) Graph.zoom(1.6, ANIM); }
  }, prefersReducedMotion ? 0 : 350);
  setStatus(`${detail.node.label}: ${detail.neighbors.length} connected concepts.`);
}

// ------------------------------------------------------------------ panel rendering
function showPanel(html) {
  const body = $("#panel-body");
  body.innerHTML = html;
  body.scrollTop = 0;
}
function backButton() {
  return state.history.length > 1 ? `<button type="button" class="back" data-action="back">Back</button>` : "";
}

function renderIntro(stats) {
  const range = stats.date_range && stats.date_range[0] ? `${fmtDate(stats.date_range[0])} to ${fmtDate(stats.date_range[1])}` : "";
  const platforms = Object.entries(stats.platforms || {}).map(([p, n]) => `${n} from ${platformName(p)}`).join(", ");
  showPanel(`<div class="intro">
    <h1>An atlas of your conversations</h1>
    <p>Every concept you have discussed with an AI assistant, placed by what it is connected to. The tinted regions are the themes your conversations keep returning to.</p>
    <p>Open a concept to see its neighbours and where you talked about it, or search above.</p>
    <dl class="facts">
      <dt>${(stats.nodes || 0).toLocaleString()}</dt><dd>concepts</dd>
      <dt>${(stats.conversations || 0).toLocaleString()}</dt><dd>conversations${platforms ? ` (${esc(platforms)})` : ""}</dd>
      <dt>${(stats.communities || 0).toLocaleString()}</dt><dd>regions</dd>
      ${range ? `<dt>Span</dt><dd>${esc(range)}</dd>` : ""}
    </dl>
  </div>`);
}

function fillMonths(timeline) {
  if (timeline.length < 2) return timeline;
  const out = [];
  const have = new Map(timeline.map((t) => [t.month, t.conversations]));
  let [y, m] = timeline[0].month.split("-").map(Number);
  const [ey, em] = timeline[timeline.length - 1].month.split("-").map(Number);
  while (y < ey || (y === ey && m <= em)) {
    const key = `${y}-${String(m).padStart(2, "0")}`;
    out.push({ month: key, conversations: have.get(key) || 0 });
    m += 1; if (m > 12) { m = 1; y += 1; }
  }
  return out;
}

function timelineHtml(timeline) {
  timeline = fillMonths(timeline);
  if (!timeline.length) return "";
  const max = Math.max(...timeline.map((t) => t.conversations));
  const bars = timeline.map((t) => `<span title="${esc(fmtMonth(t.month))}: ${t.conversations} conversation${t.conversations > 1 ? "s" : ""}" style="height:${t.conversations ? Math.max(10, (t.conversations / max) * 100) : 2}%"></span>`).join("");
  return `<h2>Over time</h2><div class="timeline" role="img" aria-label="Conversations per month">${bars}</div>
    <div class="timeline-axis"><span>${esc(fmtMonth(timeline[0].month))}</span><span>${esc(fmtMonth(timeline[timeline.length - 1].month))}</span></div>`;
}

function renderConcept(d) {
  const n = d.node;
  const span = n.first_seen ? (n.first_seen === n.last_seen ? ` on ${fmtDate(n.first_seen)}` : ` between ${fmtDate(n.first_seen)} and ${fmtDate(n.last_seen)}`) : "";
  const kind = (n.type && n.type !== "topic" ? n.type : n.kind === "topic" ? "Topic" : "Entity");
  const maxS = Math.max(...d.neighbors.map((x) => x.edge.strength), 0.0001);
  const neighbours = d.neighbors.map((nb) => `<li><button type="button" class="row-btn" data-concept="${esc(nb.id)}">
      <span class="row-title"><span class="swatch" style="background:${regionTint(nb.community)}"></span>${esc(nb.label)}
        ${nb.relations.length ? `<span class="tag">${esc(nb.relations.map(relWord).join(", "))}</span>` : ""}
        <span class="strength" title="connection strength"><i style="width:${(nb.edge.strength / maxS) * 100}%"></i></span></span>
      <span class="row-meta">${nb.edge.conversations || 0} shared conversation${nb.edge.conversations === 1 ? "" : "s"}</span>
    </button></li>`).join("");
  const relations = d.relations.slice(0, 12).map((r) => `<div class="relation">
      <button type="button" class="wikilink" data-concept="${esc(r.source)}">${esc(r.source_label)}</button>
      <span class="verb">${esc(relWord(r.relation))}</span>
      <button type="button" class="wikilink" data-concept="${esc(r.target)}">${esc(r.target_label)}</button>
      ${r.evidence && r.evidence[0] ? `<blockquote>${esc(r.evidence[0])}</blockquote>` : ""}</div>`).join("");
  const mentions = d.mentions.map((m) => `<li><button type="button" class="row-btn" data-conversation="${esc(m.conversation_id)}" data-chunk="${esc(m.chunk_id)}">
      <span class="row-title">${esc(m.title)}</span>
      <span class="row-meta">${esc(platformName(m.platform))}, ${esc(fmtDate(m.date))}${m.count > 1 ? `, ${m.count} passages` : ""}</span>
      <p class="row-snippet">${esc(m.snippet)}</p></button></li>`).join("");
  showPanel(`${backButton()}
    <h1>${esc(n.label)}</h1>
    <p class="lede">${esc(cap(kind))}, discussed in ${n.conversation_count} conversation${n.conversation_count === 1 ? "" : "s"}${esc(span)}.</p>
    ${d.community ? `<button type="button" class="region-chip" data-region-chip="${d.community.id}"><span class="swatch" style="background:${regionTint(d.community.id)}"></span>${esc(d.community.label)}</button>` : ""}
    ${timelineHtml(d.timeline)}
    ${neighbours ? `<h2>Connected to</h2><ul class="rows">${neighbours}</ul>` : `<p class="muted">No connections yet.</p>`}
    ${relations ? `<h2>How they relate</h2>${relations}` : ""}
    ${mentions ? `<h2>Where you discussed it</h2><ul class="rows">${mentions}</ul>` : ""}`);
}

const cap = (s) => (s ? s[0].toUpperCase() + s.slice(1) : s);
function relWord(r) {
  return ({ USES: "uses", DEPENDS_ON: "depends on", IMPLEMENTS: "implements", OPTIMIZES: "speeds up", CAUSES: "causes",
    COMPETES_WITH: "alternative to", PART_OF: "part of", REPLACES: "replaces", IS_A: "is a" }[r] || r.toLowerCase());
}

// ------------------------------------------------------------------ markdown-lite (for passages and the reader)
function renderText(md) {
  const blocks = [];
  md = md.replace(/```[\w-]*\n?([\s\S]*?)```/g, (_, code) => { blocks.push(code); return `\u0000${blocks.length - 1}\u0000`; });
  let html = esc(md)
    .replace(/\[\[([^\]|]+)\|([^\]]+)\]\]/g, (_, id, label) => `<button type="button" class="wikilink" data-concept="${id}">${label}</button>`)
    .replace(/\[\[([^\]]+)\]\]/g, (_, id) => `<button type="button" class="wikilink" data-concept="${id}">${id.replace(/_/g, " ")}</button>`)
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/`([^`]+)`/g, "<code>$1</code>");
  html = html.split(/\n{2,}/).map((p) => `<p>${p.replace(/\n/g, "<br>")}</p>`).join("");
  return html.replace(/<p>\u0000(\d+)\u0000<\/p>|\u0000(\d+)\u0000/g, (_, a, b) => `<pre><code>${esc(blocks[Number(a ?? b)])}</code></pre>`);
}

// ------------------------------------------------------------------ search
const input = $("#search-input");
const sugg = $("#suggestions");
let suggestTimer = null;
let suggestItems = [];
let activeIndex = -1;

input.addEventListener("input", () => {
  clearTimeout(suggestTimer);
  const q = input.value.trim();
  if (!q) { hideSuggestions(); return; }
  suggestTimer = setTimeout(async () => {
    const nodes = await api(`/api/nodes/search?q=${encodeURIComponent(q)}&limit=6`).catch(() => []);
    suggestItems = [...nodes.map((n) => ({ type: "concept", node: n })), { type: "ask", q }];
    activeIndex = -1;
    sugg.innerHTML = suggestItems.map((it, i) => it.type === "concept"
      ? `<li role="option" id="sugg-${i}" data-i="${i}"><span>${esc(it.node.label)}</span><span class="hint">${it.node.conversation_count} conversation${it.node.conversation_count === 1 ? "" : "s"}</span></li>`
      : `<li role="option" id="sugg-${i}" data-i="${i}" class="ask"><span>Search your conversations for “${esc(it.q)}”</span><span class="hint">Enter</span></li>`).join("");
    sugg.hidden = false;
    input.setAttribute("aria-expanded", "true");
  }, 140);
});
input.addEventListener("keydown", (e) => {
  if (sugg.hidden) return;
  if (e.key === "ArrowDown" || e.key === "ArrowUp") {
    e.preventDefault();
    activeIndex = (activeIndex + (e.key === "ArrowDown" ? 1 : -1) + suggestItems.length) % suggestItems.length;
    sugg.querySelectorAll("li").forEach((li, i) => li.setAttribute("aria-selected", String(i === activeIndex)));
    input.setAttribute("aria-activedescendant", `sugg-${activeIndex}`);
  } else if (e.key === "Escape") hideSuggestions();
});
sugg.addEventListener("mousedown", (e) => {
  const li = e.target.closest("li[data-i]");
  if (li) { e.preventDefault(); choose(suggestItems[Number(li.dataset.i)]); }
});
$("#search-form").addEventListener("submit", (e) => {
  e.preventDefault();
  const q = input.value.trim();
  if (!q) return;
  choose(activeIndex >= 0 ? suggestItems[activeIndex] : { type: "ask", q });
});
function hideSuggestions() { sugg.hidden = true; input.setAttribute("aria-expanded", "false"); activeIndex = -1; }
function choose(item) {
  hideSuggestions();
  if (!item) return;
  if (item.type === "concept") openConcept(item.node.id);
  else runSearch(item.q);
}

async function runSearch(q, { pushHistory = true } = {}) {
  setMode(null);
  showPanel(`<p class="loading">Searching your conversations…</p>`);
  const data = await api(`/api/search?q=${encodeURIComponent(q)}&k=10`);
  if (pushHistory) state.history.push({ type: "search", q });
  // Light up the matched concepts on the map.
  const matched = data.concepts.slice(0, 6);
  for (const c of matched) addNode(c);
  commit();
  state.selected = null;
  state.neighbourhood = new Set(matched.map((c) => c.id));
  state.route = { nodes: new Set(matched.map((c) => c.id)), links: new Set() };
  const results = data.results.map((r) => `<article class="passage">
      <div class="row-title">${esc(r.title)}</div>
      <div class="row-meta"><span>${esc(platformName(r.platform))}</span><span>${esc(fmtDate(r.date))}</span></div>
      <div class="text">${renderText(r.text)}</div>
      ${r.via.length ? `<div class="via">Connected through ${r.via.map((v) => `<button type="button" class="chip" data-concept="${esc(v)}">${esc(state.nodes.get(v)?.label || v.replace(/_/g, " "))}</button>`).join("")}</div>` : ""}
      <button type="button" class="open-conv" data-conversation="${esc(r.conversation_id)}" data-chunk="${esc(r.chunk_id)}">Read the conversation</button>
    </article>`).join("");
  showPanel(`${backButton()}<h1>${esc(q)}</h1>
    ${matched.length ? `<p class="lede">Concepts in your question: ${matched.map((c) => `<button type="button" class="chip" data-concept="${esc(c.id)}">${esc(c.label)}</button>`).join("")}</p>` : ""}
    ${results ? `<h2>Passages</h2>${results}` : `<p class="muted">Nothing in your conversations matches this yet. Try fewer or different words.</p>`}`);
  if (matched.length) setTimeout(() => fitTo((n) => state.route.nodes.has(n.id)), 400);
  setStatus(data.results.length ? `${data.results.length} passages for “${q}”.` : `No passages for “${q}”.`);
}

// ------------------------------------------------------------------ reader
async function openConversation(id, chunk, { pushHistory = true } = {}) {
  showPanel(`<p class="loading">Opening the conversation…</p>`);
  const data = await api(`/api/conversation/${encodeURIComponent(id)}`);
  if (pushHistory) state.history.push({ type: "conversation", id, chunk });
  const turns = data.chunks.map((c) => `<div class="turn ${c.chunk_id === chunk ? "focus" : ""}" id="chunk-${esc(c.chunk_id)}">${renderText(c.text.replace(/^\*\*(You|Assistant):\*\*\s*/gm, (_, who) => `**${who === "You" ? "You" : "Assistant"}:** `))}</div>`).join("");
  showPanel(`${backButton()}<div class="reader"><h1>${esc(data.title)}</h1>
    <p class="lede">${esc(platformName(data.platform))}, ${esc(fmtDate(data.date))}</p>${turns}</div>`);
  const focus = document.getElementById(`chunk-${chunk}`);
  if (focus) focus.scrollIntoView({ block: "center", behavior: prefersReducedMotion ? "auto" : "smooth" });
}

// ------------------------------------------------------------------ discover
const discoverBtn = $("#mode-discover");
function setMode(mode) { discoverBtn.setAttribute("aria-pressed", String(mode === "discover")); }
discoverBtn.addEventListener("click", async () => {
  if (discoverBtn.getAttribute("aria-pressed") === "true") { setMode(null); renderIntro(await api("/api/stats")); return; }
  setMode("discover");
  state.history.push({ type: "discover" });
  showPanel(`<p class="loading">Looking for connections you may have missed…</p>`);
  const d = await api("/api/discover");
  setStatus("Discover: open any pair to see the route between them on the map.");
  const pair = (a, al, b, bl, extra = "") => `<li><button type="button" class="row-btn" data-route="${esc(a)}|${esc(b)}">
      <span class="row-title">${esc(al)} and ${esc(bl)}</span>${extra}</button></li>`;
  const section = (title, explain, items) => items ? `<h2>${title}</h2><p class="explain">${explain}</p><ul class="rows">${items}</ul>` : "";
  showPanel(`<div class="discovery"><h1>Discover</h1>
    <p class="lede">Patterns in the structure of your map that are easy to miss when you read conversations one by one.</p>
    ${section("Hidden connections", "Never discussed together, yet they share many neighbours. Maybe an idea you haven't connected yet.",
      d.hidden_connections.slice(0, 8).map((h) => pair(h.source, h.source_label, h.target, h.target_label, `<span class="row-meta">both linked to ${esc(h.via_labels.slice(0, 3).join(", "))}</span>`)).join(""))}
    ${section("Unexpected links", "Concepts from different regions that keep turning up together.",
      d.unexpected_links.slice(0, 8).map((u) => pair(u.source, u.source_label, u.target, u.target_label, `<span class="row-meta">together in ${u.conversations} conversations</span>`)).join(""))}
    ${section("Bridges", "Concepts that connect otherwise separate regions of your thinking.",
      d.bridges.slice(0, 8).map((b) => `<li><button type="button" class="row-btn" data-concept="${esc(b.node)}"><span class="row-title">${esc(b.label)}</span>
        <span class="row-meta">links ${esc(b.community_labels.filter(Boolean).slice(0, 3).join(", "))}</span></button></li>`).join(""))}
    ${section("Not visited lately", "Important to you once, untouched for a while.",
      d.forgotten.slice(0, 8).map((f) => `<li><button type="button" class="row-btn" data-concept="${esc(f.node)}"><span class="row-title">${esc(f.label)}</span>
        <span class="row-meta">last discussed ${esc(fmtDate(f.last_seen))}</span></button></li>`).join(""))}
  </div>`);
});

async function showRoute(a, b) {
  const data = await api(`/api/graph/path?source=${encodeURIComponent(a)}&target=${encodeURIComponent(b)}`);
  if (!data.path.length) { showPanel(`${backButton()}<p>These two concepts are not connected yet.</p>`); return; }
  state.history.push({ type: "route", a, b });
  let prev = null;
  for (const p of data.path) { prev = addNode(p, prev || [...state.nodes.values()][0]); }
  data.hops.forEach((h) => addLink({ source: h.source, target: h.target, strength: h.strength || 0.5 }));
  commit();
  state.selected = null;
  state.neighbourhood = new Set(data.path.map((p) => p.id));
  state.route = { nodes: new Set(data.path.map((p) => p.id)), links: new Set(data.hops.map((h) => linkKey(h.source, h.target))) };
  const label = (id) => data.path.find((p) => p.id === id)?.label || id;
  const hops = data.hops.map((h) => `<div class="relation"><strong>${esc(label(h.source))}</strong> <span class="verb">to</span> <strong>${esc(label(h.target))}</strong>
      ${h.evidence ? `<blockquote>${esc(h.evidence.snippet)}</blockquote><div class="row-meta">${esc(h.evidence.title)}, ${esc(fmtDate(h.evidence.date))}</div>` : ""}</div>`).join("");
  showPanel(`${backButton()}<h1>${esc(label(data.source))} and ${esc(label(data.target))}</h1>
    <div class="path">${data.path.map((p) => `<button type="button" class="chip" data-concept="${esc(p.id)}">${esc(p.label)}</button>`).join('<span class="hop">/</span>')}</div>
    <h2>How the route runs</h2>${hops}`);
  setTimeout(() => fitTo((n) => state.route.nodes.has(n.id)), 500);
  setStatus(`Route from ${label(data.source)} to ${label(data.target)}: ${data.path.length - 1} steps.`);
}

// ------------------------------------------------------------------ panel events
$("#panel-body").addEventListener("click", async (e) => {
  const t = e.target.closest("[data-concept],[data-conversation],[data-route],[data-action],[data-region-chip]");
  if (!t) return;
  if (t.dataset.action === "back") {
    state.history.pop();
    const prev = state.history.pop();
    if (!prev) return;
    if (prev.type === "concept") openConcept(prev.id);
    else if (prev.type === "search") runSearch(prev.q);
    else if (prev.type === "conversation") openConversation(prev.id, prev.chunk);
    else if (prev.type === "route") showRoute(prev.a, prev.b);
    else if (prev.type === "discover") { setMode(null); discoverBtn.click(); }
    return;
  }
  if (t.dataset.concept) { await ensureNode(t.dataset.concept).catch(() => null); openConcept(t.dataset.concept); }
  else if (t.dataset.conversation) openConversation(t.dataset.conversation, t.dataset.chunk);
  else if (t.dataset.route) { const [a, b] = t.dataset.route.split("|"); showRoute(a, b); }
  else if (t.dataset.regionChip) document.querySelector(`#region-list button[data-region="${t.dataset.regionChip}"]`)?.click();
});

// ------------------------------------------------------------------ controls
$("#zoom-fit").addEventListener("click", () => Graph.zoomToFit(ANIM, 80));
$("#reset-map").addEventListener("click", async () => {
  state.region = null; state.history = []; clearSelection();
  await loadOverview();
  renderIntro(await api("/api/stats"));
});
$("#theme-toggle").addEventListener("click", () => {
  const next = isDark() ? "light" : "dark";
  document.documentElement.dataset.theme = next;
  try { localStorage.setItem("kb-theme", next); } catch { /* storage unavailable */ }
  colors = palette();
  document.querySelectorAll("#region-list .swatch").forEach((s) => {
    const id = Number(s.closest("button").dataset.region);
    s.style.background = regionTint(id);
  });
});
window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => { colors = palette(); });
document.addEventListener("keydown", (e) => {
  if (e.key === "/" && document.activeElement !== input) { e.preventDefault(); input.focus(); }
  if (e.key === "Escape" && document.activeElement !== input) clearSelection();
});

// ------------------------------------------------------------------ boot
(async function boot() {
  try { const saved = localStorage.getItem("kb-theme"); if (saved) document.documentElement.dataset.theme = saved; } catch { /* ignore */ }
  colors = palette();
  resize();
  try {
    const stats = await api("/api/stats");
    renderIntro(stats);
    await loadOverview();
    if (document.fonts) document.fonts.ready.then(() => Graph.nodeColor(Graph.nodeColor()));
  } catch (err) {
    if (err.code === 503) $("#empty-state").hidden = false;
    else showPanel(`<p>Couldn't reach the knowledge base server: ${esc(err.message)}. Is <code>kb serve</code> running?</p>`);
  }
})();
