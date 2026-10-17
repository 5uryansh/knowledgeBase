// knowledgeBase atlas — lazy-loaded concept map.
// The browser never downloads the whole graph: it starts with the most important
// concepts (/api/graph/overview) and merges a neighbourhood in each time you open a
// concept (/api/graph/node/:id).

const $ = (sel) => document.querySelector(sel);
const prefersReducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
const ANIM = prefersReducedMotion ? 0 : 750;
const GROW = 520;  // ms for a new place to grow in
const easeOut = (t) => 1 - (1 - t) ** 3;
const grown = (n) => (n.bornAt ? easeOut(Math.min(1, (performance.now() - n.bornAt) / GROW)) : 1);
const NEIGHBOURS_ON_MAP = 16;

// Region (community) colours for place symbols, one set per theme.
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
  capitals: new Map(),       // region id -> its capital's node id (fixed once chosen)
  townCut: null,             // importance a place needs to be a town (fixed at first chart)
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

// Static export (kb export): the same API contract, served from pre-computed JSON files.
const STATIC = window.KB_STATIC || null;
let nodeIndex = null;
async function staticApi(path) {
  const url = new URL(path, "http://x");
  const p = url.pathname, q = url.searchParams;
  const get = async (file) => {
    const res = await fetch(`data/${file}`);
    if (!res.ok) throw new Error(res.status === 404 ? "Not part of this export" : `Request failed (${res.status})`);
    return res.json();
  };
  if (p === "/api/stats") return get("stats.json");
  if (p === "/api/graph/overview") return get(q.get("community") != null ? `overview_c${q.get("community")}.json` : "overview.json");
  if (p.startsWith("/api/graph/node/")) {
    let id = decodeURIComponent(p.slice("/api/graph/node/".length));
    if (!/^[a-z0-9_]+$/.test(id)) {  // a label was passed: resolve it through the index
      nodeIndex = nodeIndex || await get("node_index.json");
      const hit = nodeIndex.find((n) => n.label.toLowerCase() === id.toLowerCase());
      if (hit) id = hit.id;
    }
    return get(`nodes/${id}.json`);
  }
  if (p === "/api/nodes/search") {
    nodeIndex = nodeIndex || await get("node_index.json");
    const needle = (q.get("q") || "").toLowerCase();
    const score = (n) => {
      const names = [n.label.toLowerCase(), ...(n.aliases || []).map((a) => a.toLowerCase())];
      if (names.includes(needle)) return 3;
      if (names.some((x) => x.startsWith(needle))) return 2;
      return names.some((x) => x.includes(needle)) ? 1 : 0;
    };
    return nodeIndex.map((n) => [score(n) + (n.importance || 0), n]).filter(([s]) => s >= 1)
      .sort((a, b) => b[0] - a[0]).slice(0, Number(q.get("limit") || 8)).map(([, n]) => n);
  }
  if (p === "/api/discover") return get("discover.json");
  if (p === "/api/graph/path") return get(`paths/${q.get("source")}__${q.get("target")}.json`);
  if (p.startsWith("/api/conversation/")) {
    if (!STATIC.conversations) throw new Error("Conversation text is not included in this export");
    return get(`conversations/${p.split("/").pop()}.json`);
  }
  throw new Error("Passage search needs the live server (kb serve)");
}

// The desktop app sets window.KB_API_BASE (default backend: http://127.0.0.1:3001); served by kb serve it is "".
const API_BASE = String(window.KB_API_BASE || "").replace(/\/$/, "");
async function api(path) {
  if (STATIC) return staticApi(path);
  const res = await fetch(API_BASE + path);
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
    fills: dark ? NODE_DARK : NODE_LIGHT,
  };
}
let colors = palette();
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
  // Settle the layout before painting it, then let it ease to rest. Places never fly
  // across the map; they appear where they belong.
  .warmupTicks(120)
  .cooldownTicks(prefersReducedMotion ? 0 : 70)
  .d3VelocityDecay(0.32)
  .minZoom(0.2)
  .maxZoom(8)
  .onNodeClick((n) => openConcept(n.id))
  .onNodeHover((n) => { state.hover = n ? n.id : null; mapEl.style.cursor = n ? "pointer" : ""; })
  .onBackgroundClick(() => clearSelection())
  .linkColor(linkColor)
  .linkWidth(linkWidth)
  .linkCurvature((l) => (sameRegion(l) ? 0 : 0.2))
  .linkLineDash((l) => (state.route.links.has(linkKey(idOf(l.source), idOf(l.target))) ? [5, 3] : null))
  .nodeCanvasObjectMode(() => "replace")
  .nodeCanvasObject(drawNode)
  .nodePointerAreaPaint((n, color, ctx) => {
    ctx.fillStyle = color; ctx.beginPath(); ctx.arc(n.x, n.y, radius(n) + 3, 0, 2 * Math.PI); ctx.fill();
  })
  .onRenderFramePre(() => { labelQueue = []; })
  .onRenderFramePost(drawLabels)
  .onEngineStop(() => {
    // Once settled, nothing needs holding: the next change pins whatever is on the map then.
    for (const n of state.nodes.values()) { delete n.fx; delete n.fy; }
    if (fitPending) { fitPending = false; Graph.zoomToFit(ANIM, 90); }
  });
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

const TIER_RADIUS = { capital: 8, town: 5.2, hamlet: 3.2 };
function radius(n) { return TIER_RADIUS[n.tier] || TIER_RADIUS.hamlet; }

function visibleByRegion(n) { return state.region == null || n.community === state.region; }

function drawNode(n, ctx, scale) {
  const g = grown(n);
  if (g >= 1 && n.bornAt) delete n.bornAt;
  const r = radius(n) * (0.35 + 0.65 * g);
  const isSel = n.id === state.selected;
  const inHood = state.neighbourhood.has(n.id) || n.id === state.hover;
  const onRoute = state.route.nodes.has(n.id);
  const dimmed = (state.selected && !inHood && !isSel && !onRoute) || !visibleByRegion(n);
  ctx.globalAlpha = (dimmed ? 0.28 : 1) * g;

  ctx.beginPath();
  ctx.arc(n.x, n.y, r, 0, 2 * Math.PI);
  ctx.fillStyle = regionFill(n.community);
  ctx.fill();
  if (n.tier === "capital") {
    // the cartographic capital: a ringed dot
    ctx.beginPath();
    ctx.arc(n.x, n.y, r * 0.42, 0, 2 * Math.PI);
    ctx.fillStyle = colors.paper;
    ctx.fill();
  }
  if (n.tier !== "hamlet") {
    ctx.lineWidth = 1.2 / scale;
    ctx.strokeStyle = colors.paper;
    ctx.stroke();
  }
  if (isSel || onRoute) {
    ctx.beginPath();
    ctx.arc(n.x, n.y, r + 3.5 / scale + 1.5, 0, 2 * Math.PI);
    ctx.lineWidth = 2.2 / scale;
    ctx.strokeStyle = onRoute && !isSel ? colors.survey : colors.select;
    ctx.stroke();
  }

  // Lettering is queued and placed after all places are drawn (see drawLabels).
  const rel = (n.importance || 0) / (state.maxImportance || 1);
  const rank = n.tier === "capital" ? 1.6 : n.tier === "town" ? 1 : 0;
  const priority = isSel ? 4 : onRoute ? 3 : inHood ? 2 + rank / 10 : rank + Math.min(rel, 0.99) / 2;
  labelQueue.push({ n, r, priority, dimmed, g, tier: isSel ? "selected" : n.tier || "hamlet" });
  ctx.globalAlpha = 1;
}

// Map lettering, like a cartographer would place it: most important names first, and a name is
// left out when it would collide with one already placed. Zooming in makes room for more.
let labelQueue = [];
function drawLabels(ctx, scale) {
  const placed = [];
  labelQueue.sort((a, b) => b.priority - a.priority);
  // Names sit to the right of their symbol, as on a printed map.
  const LETTER = {
    selected: { px: 15,   weight: 700, tone: "ink" },
    capital:  { px: 13.5, weight: 700, tone: "ink" },
    town:     { px: 12,   weight: 600, tone: "ink" },
    hamlet:   { px: 11,   weight: 500, tone: "inkSoft" },
  };
  ctx.textAlign = "left";
  ctx.textBaseline = "middle";
  ctx.lineJoin = "round";
  for (const { n, r, priority, dimmed, tier, g } of labelQueue) {
    const spec = LETTER[tier] || LETTER.hamlet;
    const size = spec.px / scale;
    ctx.font = `${spec.weight} ${size}px Overpass, "Arial Narrow", sans-serif`;
    const w = ctx.measureText(n.label).width;
    const x = n.x + r + 4 / scale;
    const box = [x - 1 / scale, n.y - size / 2 - 1 / scale, x + w + 2 / scale, n.y + size / 2 + 1 / scale];
    const collides = placed.some((b) => box[0] < b[2] && box[2] > b[0] && box[1] < b[3] && box[3] > b[1]);
    if (collides && priority < 3) continue;
    placed.push(box);
    ctx.globalAlpha = (dimmed ? 0.35 : 1) * (g ?? 1);
    ctx.lineWidth = 3 / scale;
    ctx.strokeStyle = colors.halo;
    ctx.strokeText(n.label, x, n.y);
    ctx.fillStyle = colors[spec.tone];
    ctx.fillText(n.label, x, n.y);
  }
  ctx.globalAlpha = 1;
  labelQueue = [];
}

function linkColor(l) {
  const key = linkKey(idOf(l.source), idOf(l.target));
  if (state.route.links.has(key)) return colors.survey;
  const s = idOf(l.source), t = idOf(l.target);
  const focus = state.selected || state.hover;
  if (focus && (s === focus || t === focus)) return colors.ink;
  if (sameRegion(l)) return hexAlpha(colors.route, state.selected ? 0.12 : 0.32);
  // Between regions: present, but not the subject of the map.
  return hexAlpha(colors.route, state.selected ? 0.11 : 0.2);
}
function linkWidth(l) {
  const key = linkKey(idOf(l.source), idOf(l.target));
  if (state.route.links.has(key)) return 1.8;
  const focus = state.selected || state.hover;
  const base = sameRegion(l) ? 0.35 + Math.min(0.4, (l.strength || 0) * 0.12) : 0.3;
  return focus && (idOf(l.source) === focus || idOf(l.target) === focus) ? base + 0.5 : base;
}
function hexAlpha(hex, a) {
  const h = hex.replace("#", "");
  if (h.length !== 6) return hex;
  const [r, g, b] = [0, 2, 4].map((i) => parseInt(h.slice(i, i + 2), 16));
  return `rgba(${r},${g},${b},${a})`;
}

// ------------------------------------------------------------------ data merge
function addNode(n, near) {
  const existing = state.nodes.get(n.id);
  if (existing) { Object.assign(existing, n, { x: existing.x, y: existing.y }); return existing; }
  const node = { ...n };
  const anchor = anchors.get(n.community);
  if (near && near.x != null) {
    // Even spacing around the place they came from (golden angle), not a random scatter.
    near._spawned = (near._spawned || 0) + 1;
    const a = near._spawned * 2.39996;
    node.x = near.x + 46 * Math.cos(a);
    node.y = near.y + 46 * Math.sin(a);
  } else if (anchor) {
    node.x = anchor[0] + (Math.random() - 0.5) * 80;
    node.y = anchor[1] + (Math.random() - 0.5) * 80;
  }
  if (!prefersReducedMotion && state.nodes.size) node.bornAt = performance.now();
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
  computeTiers();
  // The map never rearranges under you: everything already placed is held where it is,
  // and only newcomers find their spot.
  for (const n of state.nodes.values()) if (n.x != null && !n.bornAt) { n.fx = n.x; n.fy = n.y; }
  Graph.graphData({ nodes: [...state.nodes.values()], links: [...state.links.values()] });
}

// A place's tier is decided once, so nothing on the map changes size when neighbours arrive.
function computeTiers() {
  const fresh = [...state.nodes.values()].filter((n) => !n.tier)
    .sort((a, b) => (b.importance || 0) - (a.importance || 0));
  if (!fresh.length) return;
  if (state.townCut == null) {
    const k = Math.max(0, Math.round(fresh.length * 0.35) - 1);
    state.townCut = fresh[k] ? fresh[k].importance || 0 : 0;
  }
  for (const n of fresh) {
    const c = n.community;
    const names = namesakes(c);
    if (names.length ? names.includes(n.id) : c != null && c >= 0 && !state.capitals.has(c)) {
      state.capitals.set(c, n.id);
      n.tier = "capital";
    } else n.tier = (n.importance || 0) >= state.townCut ? "town" : "hamlet";
  }
}

// A region is named after its leading places ("SIGTERM & Bitbucket" = its first two
// top_nodes), so every place in the name gets the capital ring, not just the first.
function namesakes(c) {
  const r = state.communities.get(c);
  if (!r || !Array.isArray(r.top_nodes) || !r.label) return [];
  return r.top_nodes.slice(0, r.label.split(" & ").length);
}

async function loadOverview() {
  setStatus("Charting your most important concepts…");
  const data = await api("/api/graph/overview?limit=55");
  state.nodes.clear(); state.links.clear(); state.communities.clear();
  state.capitals.clear(); state.townCut = null;
  state.maxImportance = Math.max(...data.nodes.map((n) => n.importance || 0), 0.0001);
  data.communities.forEach((c) => state.communities.set(c.id, c));
  layoutAnchors(data.nodes);
  data.nodes.forEach((n) => addNode(n));  // each place starts inside its own territory
  data.edges.forEach(addLink);
  commit();
  renderRegions(data);
  setStatus(`${data.nodes.length} of ${data.total_nodes.toLocaleString()} concepts shown. Open one to see what it connects to.`);
  setTimeout(() => Graph.zoomToFit(ANIM, 90), 60);  // layout is pre-settled: one establishing move
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
  const visible = [...state.communities.values()].filter((c) => shown.has(c.id)).slice(0, 14);
  const largest = Math.max(...visible.map((c) => c.size), 1);
  visible.forEach((c) => {
      const li = document.createElement("li");
      // The bar carries relative size, which a bare number makes you compute.
      li.innerHTML = `<button type="button" aria-pressed="false" data-region="${c.id}" style="--fill:${(c.size / largest).toFixed(3)};--tone:${regionFill(c.id)}">
        <span class="swatch" style="background:${regionFill(c.id)}"></span><span class="region-name">${esc(c.label)}</span>
        <span class="count">${c.size.toLocaleString()}</span></button>`;
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
  showLoading("Opening the gazetteer…");
  let detail;
  try {
    detail = await api(`/api/graph/node/${encodeURIComponent(id)}?limit=30`);
  } catch (err) {
    showPanel(`<p>Couldn't open this concept: ${esc(err.message)}</p>`);
    return;
  }
  const centre = addNode(detail.node);
  delete centre.bornAt;                       // the place you opened is the fixed point
  if (centre.x != null) { centre.fx = centre.x; centre.fy = centre.y; }
  detail.neighbors.slice(0, NEIGHBOURS_ON_MAP).forEach((nb) => addNode(nb, centre));
  detail.edges.forEach(addLink);
  commit();
  state.selected = detail.node.id;
  state.neighbourhood = new Set(detail.neighbors.slice(0, NEIGHBOURS_ON_MAP).map((n) => n.id));
  state.route = { nodes: new Set(), links: new Set() };
  if (pushHistory) state.history.push({ type: "concept", id });
  renderConcept(detail);
  // One glide that frames the place and its neighbourhood together.
  setTimeout(() => fitTo((m) => m.id === centre.id || state.neighbourhood.has(m.id), 110, 2.2), 40);
  setStatus(`${detail.node.label}: ${detail.neighbors.length} connected concepts.`);
}

// ------------------------------------------------------------------ panel rendering
// Show a loading line only if the request is actually slow (avoids flashing on fast local calls).
let loadingTimer = null;
function showLoading(text) {
  clearTimeout(loadingTimer);
  loadingTimer = setTimeout(() => showPanel(`<p class="loading">${esc(text)}</p>`, true), 250);
}
function showPanel(html, isLoading = false) {
  if (!isLoading) clearTimeout(loadingTimer);
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
    ${STATIC ? `<p class="muted">This is a static snapshot: concepts and connections are browsable; passage search needs the live app (<code>kb serve</code>).</p>` : ""}
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

// A chart needs something to compare. One month, or a flat run of identical
// months, is a sentence — drawing it as a bar renders a grey slab that reads
// as a broken image.
function timelineHtml(timeline) {
  timeline = fillMonths(timeline);
  const active = timeline.filter((t) => t.conversations > 0);
  if (active.length < 2 || timeline.length < 3) return "";

  const max = Math.max(...timeline.map((t) => t.conversations));
  const total = timeline.reduce((a, t) => a + t.conversations, 0);
  const peak = timeline.find((t) => t.conversations === max);
  const bars = timeline.map((t) => {
    const label = `${esc(fmtMonth(t.month))}: ${t.conversations || "no"} conversation${t.conversations === 1 ? "" : "s"}`;
    return `<span class="${t.conversations === max ? "peak" : ""}" title="${label}" style="height:${t.conversations ? Math.max(8, (t.conversations / max) * 100) : 0}%"></span>`;
  }).join("");
  const mid = timeline[Math.floor(timeline.length / 2)];
  return `<h2>Over time <span class="count">${total}</span></h2>
    <p class="explain">Busiest in ${esc(fmtMonth(peak.month))}, with ${peak.conversations}.</p>
    <div class="timeline" role="img" aria-label="Conversations per month, ${timeline.length} months">${bars}</div>
    <div class="timeline-axis"><span>${esc(fmtMonth(timeline[0].month))}</span><span>${esc(fmtMonth(mid.month))}</span><span>${esc(fmtMonth(timeline[timeline.length - 1].month))}</span></div>`;
}

function renderConcept(d) {
  const n = d.node;
  const span = n.first_seen ? (n.first_seen === n.last_seen ? ` on ${fmtDate(n.first_seen)}` : ` between ${fmtDate(n.first_seen)} and ${fmtDate(n.last_seen)}`) : "";
  const kind = (n.type && n.type !== "topic" ? n.type : n.kind === "topic" ? "Topic" : "Entity");
  // Only draw a comparison when there is something to compare. Equal strengths
  // render as a row of identical full-width bars, which encode nothing; equal
  // conversation counts repeat one sentence down the whole list.
  const strengths = d.neighbors.map((x) => x.edge.strength || 0);
  const maxS = Math.max(...strengths, 0.0001);
  const varies = strengths.length > 2 && Math.max(...strengths) - Math.min(...strengths) > 1e-9;
  const convs = d.neighbors.map((nb) => nb.edge.conversations || 0);
  const sharedCount = convs.length > 1 && new Set(convs).size === 1 ? convs[0] : null;
  const neighbours = d.neighbors.map((nb) => `<li><button type="button" class="row-btn" data-concept="${esc(nb.id)}">
      <span class="row-title"><span class="swatch" style="background:${regionFill(nb.community)}"></span><span class="row-name">${esc(nb.label)}</span>
        ${nb.relations.length ? `<span class="tag">${esc(nb.relations.map(relWord).join(", "))}</span>` : ""}
        ${varies ? `<span class="strength" title="${nb.edge.conversations || 0} shared conversation${nb.edge.conversations === 1 ? "" : "s"}"><i style="width:${(nb.edge.strength / maxS) * 100}%"></i></span>` : ""}</span>
      ${sharedCount === null ? `<span class="row-meta">${nb.edge.conversations || 0} shared conversation${nb.edge.conversations === 1 ? "" : "s"}</span>` : ""}
    </button></li>`).join("");
  const relations = d.relations.slice(0, 12).map((r) => `<div class="relation">
      <button type="button" class="wikilink" data-concept="${esc(r.source)}">${esc(r.source_label)}</button>
      <span class="verb">${esc(relWord(r.relation))}</span>
      <button type="button" class="wikilink" data-concept="${esc(r.target)}">${esc(r.target_label)}</button>
      ${r.evidence && r.evidence[0] ? `<blockquote>${esc(r.evidence[0])}</blockquote>` : ""}</div>`).join("");
  const canRead = !STATIC || STATIC.conversations;
  const mentions = d.mentions.map((m) => `<li><button type="button" class="row-btn" ${canRead ? `data-conversation="${esc(m.conversation_id)}" data-chunk="${esc(m.chunk_id || "")}"` : "disabled"}>
      <span class="row-title">${esc(m.title)}</span>
      <span class="row-meta">${esc(platformName(m.platform))}, ${esc(fmtDate(m.date))}${m.count > 1 ? `, ${m.count} passages` : ""}</span>
      <p class="row-snippet">${esc(m.snippet)}</p></button></li>`).join("");
  const worth = (d.worth_connecting || []).map((w) => `<li><button type="button" class="row-btn" data-route="${esc(n.id)}|${esc(w.id)}">
      <span class="row-title"><span class="swatch" style="background:${regionFill(w.community)}"></span>${esc(w.label)}</span>
      <span class="row-meta">both linked to ${esc(w.via.map((v) => v.label).join(", "))}</span></button></li>`).join("");
  showPanel(`${backButton()}
    <h1>${esc(n.label)}</h1>
    <p class="lede">${esc(cap(kind))}, discussed in ${n.conversation_count} conversation${n.conversation_count === 1 ? "" : "s"}${esc(span)}.</p>
    ${d.community ? `<button type="button" class="region-chip" data-region-chip="${d.community.id}"><span class="swatch" style="background:${regionFill(d.community.id)}"></span>${esc(d.community.label)}</button>` : ""}
    ${d.summary ? `<h2>In short</h2><div class="summary">${renderText(d.summary.summary)}</div><p class="row-meta">Written by ${esc(d.summary.model)} from ${d.summary.sources.length} of your conversations.</p>` : ""}
    ${timelineHtml(d.timeline)}
    ${neighbours ? `<h2>Connected to <span class="count">${d.neighbors.length}</span></h2>${sharedCount !== null ? `<p class="explain">Each shares ${sharedCount} conversation${sharedCount === 1 ? "" : "s"} with ${esc(n.label)}.</p>` : ""}<ul class="rows ${sharedCount === null ? "" : "compact"}">${neighbours}</ul>` : `<p class="muted">Nothing connects to ${esc(n.label)} yet. It will link up as you discuss it alongside other ideas.</p>`}
    ${relations ? `<h2>How they relate <span class="count">${Math.min(d.relations.length, 12)}</span></h2>${relations}` : ""}
    ${worth && !STATIC ? `<h2>Worth connecting <span class="count">${(d.worth_connecting || []).length}</span></h2><p class="explain muted">Never discussed together with ${esc(n.label)}, but they share several neighbours.</p><ul class="rows">${worth}</ul>` : ""}
    ${mentions ? `<h2>Where you discussed it <span class="count">${d.mentions.length}</span></h2><ul class="rows">${mentions}</ul>` : ""}`);
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
  html = html.split(/\n{2,}/).map((p) => {
    const lines = p.split("\n");
    if (lines.length && lines.every((l) => /^\s*[-*•]\s+/.test(l))) {  // a bullet list
      return `<ul>${lines.map((l) => `<li>${l.replace(/^\s*[-*•]\s+/, "").replace(/\s*(\[[^\]]+\])\s*$/, ' <span class="cite">$1</span>')}</li>`).join("")}</ul>`;
    }
    return `<p>${p.replace(/\n/g, "<br>")}</p>`;
  }).join("");
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
    suggestItems = [...nodes.map((n) => ({ type: "concept", node: n })), ...(STATIC ? [] : [{ type: "ask", q }])];
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
  choose(activeIndex >= 0 ? suggestItems[activeIndex] : STATIC ? suggestItems[0] : { type: "ask", q });
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
  showLoading("Searching your conversations…");
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
    ${data.time_window ? `<p class="lede">Favouring conversations from ${esc(data.time_window)}.</p>` : ""}
    ${matched.length ? `<p class="lede">Concepts in your question: ${matched.map((c) => `<button type="button" class="chip" data-concept="${esc(c.id)}">${esc(c.label)}</button>`).join("")}</p>` : ""}
    ${results ? `<h2>Passages</h2>${results}` : `<p class="muted">Nothing in your conversations matches this yet. Try fewer or different words.</p>`}`);
  if (matched.length) setTimeout(() => fitTo((n) => state.route.nodes.has(n.id)), 400);
  setStatus(data.results.length ? `${data.results.length} passages for “${q}”.` : `No passages for “${q}”.`);
}

// ------------------------------------------------------------------ reader
async function openConversation(id, chunk, { pushHistory = true } = {}) {
  showLoading("Opening the conversation…");
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
  showLoading("Looking for connections you may have missed…");
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
  if (STATIC) document.querySelector("#search-input").placeholder = "Find a concept";
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
    s.style.background = regionFill(id);
    s.closest("button").style.setProperty("--tone", regionFill(id));
  });
});
window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => { colors = palette(); });
document.addEventListener("keydown", (e) => {
  if (e.key === "/" && document.activeElement !== input) { e.preventDefault(); input.focus(); }
  if (e.key === "Escape" && document.activeElement !== input) clearSelection();
});
// ------------------------------------------------------------------ your data: imports, output folder, build
const dataDialog = $("#data-dialog");
let platforms = [];
let buildTimer = null;

async function send(method, path, body) {
  const res = await fetch(API_BASE + path, {
    method, headers: body === undefined ? {} : { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw Object.assign(new Error(data.detail || `Request failed (${res.status})`), { status: res.status, data });
  return data;
}

function setResult(el, text, kind = "") {
  el.textContent = text;
  el.className = `result ${kind}`;
}

function rangeText(r) {
  return r && r[0] ? (r[0] === r[1] ? fmtDate(r[0]) : `${fmtDate(r[0])} to ${fmtDate(r[1])}`) : "";
}

function syncPlatformHint() {
  const p = platforms.find((x) => x.id === $("#import-platform").value);
  $("#import-hint").textContent = p ? p.hint : "";
  $("#import-replace-label").textContent = `Replace earlier ${p ? p.label : ""} imports`;
}

function renderWorkspace(ws) {
  const items = ws.imports.map((r) => `<li>
      <span><strong>${esc(r.label)}</strong> · ${r.conversations.toLocaleString()} conversations${rangeText(r.date_range) ? ` · ${esc(rangeText(r.date_range))}` : ""}
      <span class="muted">${esc(r.filename)}</span></span>
      <button type="button" class="tool" data-remove-import="${esc(r.id)}">Remove</button></li>`);
  const configured = ws.configured_sources.map((s) => `<li><span><strong>${esc(s.label)}</strong>
      <span class="muted">read from ${esc(s.path)}</span></span></li>`);
  $("#import-list").innerHTML = items.concat(configured).join("") || `<li class="muted">Nothing added yet.</li>`;
  if (document.activeElement !== $("#output-path")) $("#output-path").value = ws.output.path;
  setResult($("#output-result"), ws.output.note || (ws.output.has_index ? "This folder has a built knowledge base." : ""));
  showBuild(ws.build);
}

async function refreshWorkspace() {
  renderWorkspace(await api("/api/workspace"));
}

async function openDataDialog() {
  if (!platforms.length) {
    platforms = await api("/api/platforms");
    $("#import-platform").innerHTML = platforms.map((p) => `<option value="${esc(p.id)}">${esc(p.label)}</option>`).join("");
    syncPlatformHint();
  }
  if (!dataDialog.open) dataDialog.showModal();
  await refreshWorkspace().catch((err) => setResult($("#import-result"), err.message, "error"));
}

function upload(file, platform, replace, onProgress) {
  // XHR rather than fetch: exports can be gigabytes, and only XHR reports upload progress.
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    const q = new URLSearchParams({ platform, filename: file.name, replace: String(replace) });
    xhr.open("POST", `${API_BASE}/api/imports?${q}`);
    xhr.setRequestHeader("Content-Type", "application/octet-stream");
    xhr.upload.onprogress = (e) => e.lengthComputable && onProgress(e.loaded / e.total);
    xhr.onload = () => {
      let data = {};
      try { data = JSON.parse(xhr.responseText); } catch { /* not json */ }
      if (xhr.status >= 200 && xhr.status < 300) resolve(data);
      else reject(Object.assign(new Error(data.detail || `Upload failed (${xhr.status})`), { data }));
    };
    xhr.onerror = () => reject(new Error("Upload failed: the server could not be reached."));
    xhr.send(file);
  });
}

$("#import-platform").addEventListener("change", syncPlatformHint);
$("#import-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const file = $("#import-file").files[0];
  if (!file) return;
  const platform = $("#import-platform").value;
  const bar = $("#import-progress");
  const btn = $("#import-submit");
  btn.disabled = true; bar.hidden = false; bar.value = 0;
  setResult($("#import-result"), `Uploading ${file.name}…`);
  try {
    const rec = await upload(file, platform, $("#import-replace").checked, (f) => {
      bar.value = f;
      if (f >= 1) setResult($("#import-result"), "Checking the export…");
    });
    const range = rangeText(rec.date_range);
    setResult($("#import-result"), `Added ${rec.conversations.toLocaleString()} ${rec.label} conversations${range ? ` (${range})` : ""}. Build to add them to your atlas.`, "ok");
    $("#import-file").value = "";
    await refreshWorkspace();
  } catch (err) {
    const detected = err.data && err.data.detected;
    setResult($("#import-result"), err.message, "error");
    if (detected) { $("#import-platform").value = detected; syncPlatformHint(); }
  } finally {
    btn.disabled = false; bar.hidden = true;
  }
});

$("#import-list").addEventListener("click", async (e) => {
  const id = e.target.closest("[data-remove-import]")?.dataset.removeImport;
  if (!id) return;
  try { await send("DELETE", `/api/imports/${encodeURIComponent(id)}`); await refreshWorkspace(); setResult($("#import-result"), "Removed. Build again to drop those chats from your atlas.", "ok"); }
  catch (err) { setResult($("#import-result"), err.message, "error"); }
});

$("#output-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  try {
    const out = await send("PUT", "/api/settings/output", { path: $("#output-path").value });
    $("#output-path").value = out.path;
    setResult($("#output-result"), out.note || (out.has_index ? "Switched to the knowledge base in this folder." : "Saved. Build to write your notes and graph here."), "ok");
    if (out.has_index) await reloadAtlas();
  } catch (err) { setResult($("#output-result"), err.message, "error"); }
});

function showBuild(st) {
  const bar = $("#build-progress");
  const btn = $("#build-start");
  const running = st && st.status === "running";
  btn.disabled = running;
  bar.hidden = !running;
  clearTimeout(buildTimer);
  if (running) {
    bar.value = st.total ? st.done / st.total : 0;
    if (!st.total) bar.removeAttribute("value");
    const count = st.total > 1 ? ` (${st.done.toLocaleString()} of ${st.total.toLocaleString()})` : "";
    setResult($("#build-result"), `${st.label}${count}…`);
    buildTimer = setTimeout(pollBuild, 1000);
  } else if (st && st.status === "error") setResult($("#build-result"), st.error, "error");
  else if (st && st.status === "done") setResult($("#build-result"), `Built: ${st.report}`, "ok");
  else setResult($("#build-result"), "");
}

async function pollBuild() {
  const st = await api("/api/build").catch(() => null);
  if (!st) { buildTimer = setTimeout(pollBuild, 3000); return; }
  showBuild(st);
  if (st.status === "done" && state.lastBuild !== st.finished_at) {
    state.lastBuild = st.finished_at;
    await reloadAtlas();
  }
}

$("#build-start").addEventListener("click", async () => {
  try { showBuild(await send("POST", "/api/build")); }
  catch (err) { setResult($("#build-result"), err.message, "error"); }
});

async function reloadAtlas() {
  try {
    const stats = await api("/api/stats");
    $("#empty-state").hidden = true;
    state.region = null; state.history = []; clearSelection();
    await loadOverview();
    renderIntro(stats);
  } catch (err) {
    if (err.code === 503) $("#empty-state").hidden = false;
  }
}

$("#open-data").addEventListener("click", openDataDialog);
$("#empty-add").addEventListener("click", openDataDialog);
if (STATIC) { $("#open-data").hidden = true; $("#empty-add").hidden = true; }
else {
  // A build started earlier (or from another window) keeps reporting here, and reloads the map when it ends.
  api("/api/build").then((st) => { state.lastBuild = st.finished_at; showBuild(st); }).catch(() => {});
}


// ------------------------------------------------------------------ boot
(async function boot() {
  try { const saved = localStorage.getItem("kb-theme"); if (saved) document.documentElement.dataset.theme = saved; } catch { /* ignore */ }
  colors = palette();
  resize();
  if (document.fonts) {
    Promise.all(['600 12px Overpass', 'italic 600 15px "Libre Bodoni"'].map((f) => document.fonts.load(f)))
      .then(() => Graph.nodeColor(Graph.nodeColor())).catch(() => {});
  }
  if (STATIC) $("#search-input").placeholder = "Find a concept";
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
