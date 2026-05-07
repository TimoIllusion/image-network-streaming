// Operator console — data layer.
//
// Polls /clients, /transports, /batching, /inference, /multi-run/status on a
// 1s cadence; maintains a tiny pub-sub store and a few React hooks the JSX
// components consume. Sparkline ring buffers (per-client fpsSeries / latSeries)
// and the hero-chart history series live here, capped at SERIES_CAP /
// HISTORY_CAP. Mutators (applyTransport / applyBatching / applyInference /
// controlClient / clearAll / exportCsv / copyMd) hit the matching POST
// endpoints and use the pendingChanges TTL pattern so user-set values stay
// visible until the heartbeat catches up.

(function () {
  const POLL_MS = 1000;
  const PENDING_TTL_MS = 5000;
  const SERIES_CAP = 60;
  const HISTORY_CAP = 120;
  const STATUS_CLEAR_MS = 3000;
  const STAGE_KEYS = ["enc", "dec", "comms", "infer", "post", "wait"];

  const initial = {
    serverHost: window.location.host,
    bootAt: Date.now(),
    transports: [],
    activeTransport: null,
    batching: { enabled: false, max_batch_size: 8, max_wait_ms: 10 },
    inference: { mode: "single", instances: 1 },
    rawClients: [],
    seriesByClient: {},
    history: [],
    sweepStatus: null,
    pendingChanges: {},
    railStatus: "",
  };

  const Store = {
    state: initial,
    listeners: new Set(),
    subscribe(fn) { this.listeners.add(fn); return () => this.listeners.delete(fn); },
    notify() { for (const fn of this.listeners) fn(); },
    set(patch) { this.state = { ...this.state, ...patch }; this.notify(); },
  };

  // ── pendingChanges helpers ────────────────────────────────────────────
  const pendingKey = (name, action) => `${name}:${action}`;

  function effectiveValue(name, action, serverValue) {
    const k = pendingKey(name, action);
    const p = Store.state.pendingChanges[k];
    if (!p) return serverValue;
    if (Date.now() > p.expiresAt || p.value === serverValue) return serverValue;
    return p.value;
  }

  function setPending(name, action, value) {
    Store.set({
      pendingChanges: {
        ...Store.state.pendingChanges,
        [pendingKey(name, action)]: { value, expiresAt: Date.now() + PENDING_TTL_MS },
      },
    });
  }

  // ── Adapters (server payload → component-friendly shapes) ─────────────
  function parseUiUrl(ui_url) {
    try {
      const u = new URL(ui_url);
      return { ip: u.hostname, port: Number(u.port) || (u.protocol === "https:" ? 443 : 80) };
    } catch {
      return { ip: ui_url || "—", port: 0 };
    }
  }

  function parseMetric(v) {
    if (v == null) return 0;
    const n = Number.parseFloat(String(v).replace(",", "."));
    return Number.isFinite(n) ? n : 0;
  }

  function findCurrentRow(rows, backend) {
    if (!rows || !rows.length) return null;
    if (backend) {
      const matching = rows.filter((r) => r.Backend === backend);
      if (matching.length) return matching[matching.length - 1];
    }
    return rows[rows.length - 1];
  }

  function rowToTiming(row) {
    if (!row) {
      const empty = {};
      for (const k of STAGE_KEYS) empty[k] = 0;
      empty.total = 0;
      return empty;
    }
    return {
      enc: parseMetric(row["enc (ms)"]),
      dec: parseMetric(row["dec (ms)"]),
      comms: parseMetric(row["comms (ms)"]),
      infer: parseMetric(row["infer (ms)"]),
      post: parseMetric(row["post (ms)"]),
      wait: parseMetric(row["wait (ms)"]),
      total: parseMetric(row["total (ms)"]),
    };
  }

  function buildClient(record, seriesByClient, serverActiveTransport) {
    const stats = record.stats || {};
    const { ip, port } = parseUiUrl(record.ui_url);
    const transport = stats.backend || serverActiveTransport || null;
    const row = findCurrentRow(stats.bench_rows, transport);
    const fps = row ? parseMetric(row.FPS) : 0;
    const timing = rowToTiming(row);
    const series = seriesByClient[record.name] || { fpsSeries: [], latSeries: [] };

    return {
      name: record.name,
      ui_url: record.ui_url,
      transport: transport || "—",
      ip,
      port,
      inferenceOn: !!effectiveValue(record.name, "inference", !!stats.inference),
      mockCam: !!effectiveValue(record.name, "mock_camera", !!stats.mock_camera),
      ageMs: Math.round((record.age_s || 0) * 1000),
      frames: row ? Math.round(parseMetric(row.Frames)) : 0,
      fps,
      timing,
      fpsSeries: series.fpsSeries,
      latSeries: series.latSeries,
      batch: row ? parseMetric(row.batch) : 1,
    };
  }

  function buildAggregate(clients) {
    const active = clients.filter((c) => c.inferenceOn && c.timing.total > 0);
    const totalFps = active.reduce((a, c) => a + c.fps, 0);
    const totals = active.map((c) => c.timing.total);
    const avgLat = totals.length ? totals.reduce((a, v) => a + v, 0) / totals.length : 0;
    const minLat = totals.length ? Math.min(...totals) : 0;
    const maxLat = totals.length ? Math.max(...totals) : 0;
    const stageAvg = {};
    for (const k of STAGE_KEYS) {
      const vals = active.map((c) => c.timing[k]);
      stageAvg[k] = vals.length ? vals.reduce((a, v) => a + v, 0) / vals.length : 0;
    }
    return {
      totalFps, avgLat, minLat, maxLat, stageAvg,
      activeCount: active.length, totalCount: clients.length,
    };
  }

  function weightedAverage(samples) {
    let total = 0, weight = 0;
    for (const { value, weight: w } of samples) {
      if (!Number.isFinite(value) || !Number.isFinite(w) || w <= 0) continue;
      total += value * w;
      weight += w;
    }
    return weight > 0 ? total / weight : 0;
  }

  function aggregateRunClients(clients) {
    const byBackend = new Map();
    for (const c of clients || []) {
      const rows = (c.stats || {}).bench_rows || [];
      for (const row of rows) {
        const backend = row.Backend;
        if (!backend) continue;
        const item = byBackend.get(backend) || {
          backend, frames: 0, fps: 0,
          totalSamples: [], waitSamples: [], inferSamples: [], batchSamples: [],
        };
        const frames = parseMetric(row.Frames);
        item.frames += frames;
        item.fps += parseMetric(row.FPS);
        item.totalSamples.push({ value: parseMetric(row["total (ms)"]), weight: frames });
        item.waitSamples.push({ value: parseMetric(row["wait (ms)"]), weight: frames });
        item.inferSamples.push({ value: parseMetric(row["infer (ms)"]), weight: frames });
        item.batchSamples.push({ value: parseMetric(row.batch), weight: frames });
        byBackend.set(backend, item);
      }
    }
    return [...byBackend.values()].map((item) => ({
      backend: item.backend,
      frames: item.frames,
      fps: item.fps,
      totalMs: weightedAverage(item.totalSamples),
      waitMs: weightedAverage(item.waitSamples),
      inferMs: weightedAverage(item.inferSamples),
      batch: weightedAverage(item.batchSamples),
    })).sort((a, b) => b.fps - a.fps);
  }

  function adaptSweep(status) {
    if (!status) return { rows: [], completed: 0, total: 0, transports: [], running: false, raw: null };
    const plan = status.plan || [];
    const runs = status.result?.runs || [];
    const completed = runs.length;
    const total = plan.length;
    const rows = plan.map((cfg, idx) => {
      const run = runs[idx];
      let cellStatus = "queued";
      if (run) cellStatus = "done";
      else if (status.running && idx === completed) cellStatus = "running";
      const out = {
        id: idx,
        transport: cfg.transport,
        batch: {
          enabled: !!cfg.batching_enabled,
          size: Number(cfg.max_batch_size ?? 1),
          wait: Number(cfg.max_wait_ms ?? 0),
        },
        infer: {
          mode: cfg.inference_mode || "single",
          instances: Number(cfg.inference_instances ?? 1),
        },
        status: cellStatus,
        fps: null, total_ms: null, transport_ms: null,
        infer_ms: null, wait_ms: null, batch_size: 1,
      };
      if (run) {
        const aggs = aggregateRunClients(run.clients || []);
        const a = aggs[0];
        if (a) {
          out.fps = a.fps;
          out.total_ms = a.totalMs;
          out.transport_ms = Math.max(0, a.totalMs - a.inferMs - a.waitMs);
          out.infer_ms = a.inferMs;
          out.wait_ms = a.waitMs;
          out.batch_size = a.batch;
        }
      }
      return out;
    });
    return {
      rows, completed, total,
      transports: [...new Set(plan.map((c) => c.transport))],
      running: !!status.running,
      raw: status,
    };
  }

  // ── Polling ───────────────────────────────────────────────────────────
  async function refreshClients() {
    try {
      const r = await fetch("/clients");
      if (!r.ok) return;
      const payload = await r.json();
      const records = payload.clients || [];
      const seriesNext = {};
      const prevSeries = Store.state.seriesByClient;
      let totalFps = 0;
      for (const rec of records) {
        const stats = rec.stats || {};
        const transport = stats.backend || payload.active_transport || null;
        const row = findCurrentRow(stats.bench_rows, transport);
        const fps = row ? parseMetric(row.FPS) : 0;
        const total = row ? parseMetric(row["total (ms)"]) : 0;
        const inferOn = !!effectiveValue(rec.name, "inference", !!stats.inference);
        const prev = prevSeries[rec.name] || { fpsSeries: [], latSeries: [] };
        const fpsSeries = inferOn ? [...prev.fpsSeries, fps].slice(-SERIES_CAP) : prev.fpsSeries;
        const latSeries = inferOn ? [...prev.latSeries, total].slice(-SERIES_CAP) : prev.latSeries;
        seriesNext[rec.name] = { fpsSeries, latSeries };
        if (inferOn) totalFps += fps;
      }
      const history = [...Store.state.history, totalFps].slice(-HISTORY_CAP);
      Store.set({
        rawClients: records,
        activeTransport: payload.active_transport || Store.state.activeTransport,
        seriesByClient: seriesNext,
        history,
      });
    } catch { /* transient */ }
  }

  async function refreshTransports() {
    try {
      const r = await fetch("/transports");
      if (!r.ok) return;
      const items = await r.json();
      const active = items.find((t) => t.active);
      Store.set({ transports: items, activeTransport: active ? active.name : null });
    } catch { /* transient */ }
  }

  async function refreshBatching() {
    try {
      const r = await fetch("/batching");
      if (!r.ok) return;
      const s = await r.json();
      Store.set({
        batching: {
          enabled: !!s.enabled,
          max_batch_size: Number(s.max_batch_size),
          max_wait_ms: Number(s.max_wait_ms),
        },
      });
    } catch { /* transient */ }
  }

  async function refreshInference() {
    try {
      const r = await fetch("/inference");
      if (!r.ok) return;
      const s = await r.json();
      Store.set({ inference: { mode: s.mode, instances: Number(s.instances) } });
    } catch { /* transient */ }
  }

  async function refreshSweep() {
    try {
      const r = await fetch("/multi-run/status");
      if (!r.ok) return;
      const status = await r.json();
      Store.set({ sweepStatus: status });
    } catch { /* transient */ }
  }

  function start() {
    refreshTransports();
    refreshClients();
    refreshBatching();
    refreshInference();
    refreshSweep();
    setInterval(refreshClients, POLL_MS);
    setInterval(refreshTransports, POLL_MS * 3);
    setInterval(refreshBatching, POLL_MS * 3);
    setInterval(refreshInference, POLL_MS * 3);
    setInterval(refreshSweep, POLL_MS);
  }

  // ── Mutators ──────────────────────────────────────────────────────────
  function setRailStatus(text, autoclear = true) {
    Store.set({ railStatus: text });
    if (autoclear && text) {
      setTimeout(() => {
        if (Store.state.railStatus === text) Store.set({ railStatus: "" });
      }, STATUS_CLEAR_MS);
    }
  }

  async function applyTransport(name) {
    setRailStatus(`switching to ${name}…`, false);
    try {
      const r = await fetch("/switch", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name, cascade: true }),
      });
      if (!r.ok) {
        setRailStatus(`failed: ${await r.text()}`);
        return;
      }
      setRailStatus(`switched to ${name}`);
    } catch (e) {
      setRailStatus(`error: ${e}`);
    } finally {
      refreshTransports();
      refreshClients();
    }
  }

  async function applyBatching(patch) {
    Store.set({ batching: { ...Store.state.batching, ...patch } });
    try {
      const r = await fetch("/batching", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(patch),
      });
      if (r.ok) {
        const s = await r.json();
        Store.set({
          batching: {
            enabled: !!s.enabled,
            max_batch_size: Number(s.max_batch_size),
            max_wait_ms: Number(s.max_wait_ms),
          },
        });
      } else {
        setRailStatus(`batching failed: ${await r.text()}`);
      }
    } catch (e) {
      setRailStatus(`batching error: ${e}`);
    }
  }

  async function applyInference(patch) {
    Store.set({ inference: { ...Store.state.inference, ...patch } });
    try {
      const r = await fetch("/inference", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(patch),
      });
      if (r.ok) {
        const s = await r.json();
        Store.set({ inference: { mode: s.mode, instances: Number(s.instances) } });
      } else {
        setRailStatus(`inference failed: ${await r.text()}`);
      }
    } catch (e) {
      setRailStatus(`inference error: ${e}`);
    }
  }

  async function controlClient(name, body) {
    for (const [action, value] of Object.entries(body)) {
      if (action === "backend") continue;
      setPending(name, action, value);
    }
    try {
      const r = await fetch(`/clients/${encodeURIComponent(name)}/control`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!r.ok) {
        const txt = await r.text();
        console.warn(`control ${name} failed: ${txt}`);
        setRailStatus(`${name}: ${txt}`);
      }
    } catch (e) {
      console.warn(`control ${name} error:`, e);
    } finally {
      refreshClients();
    }
  }

  async function clearClient(name) {
    try {
      await fetch(`/clients/${encodeURIComponent(name)}/clear`, { method: "POST" });
    } finally {
      refreshClients();
    }
  }

  async function clearAll() {
    if (!confirm("Clear stats on all connected clients?")) return;
    try {
      await fetch("/clients/clear-all", { method: "POST" });
      setRailStatus("cleared stats on all clients");
    } finally {
      refreshClients();
    }
  }

  // ── Export helpers ────────────────────────────────────────────────────
  function flattenForExport() {
    const rows = [];
    const aggs = aggregateRunClients(Store.state.rawClients);
    for (const r of aggs) {
      rows.push({
        Scope: "aggregate", Client: "all",
        Backend: r.backend,
        Frames: r.frames,
        FPS: r.fps.toFixed(1),
        "total (ms)": r.totalMs.toFixed(1),
        "wait (ms)": r.waitMs.toFixed(1),
        "infer (ms)": r.inferMs.toFixed(1),
        batch: r.batch.toFixed(1),
      });
    }
    for (const c of Store.state.rawClients) {
      const benchRows = (c.stats || {}).bench_rows || [];
      for (const r of benchRows) {
        rows.push({ Scope: "client", Client: c.name, ...r });
      }
    }
    return rows;
  }

  function toMarkdown(rows) {
    if (!rows.length) return "";
    const cols = [...new Set(rows.flatMap((r) => Object.keys(r)))];
    const head = `| ${cols.join(" | ")} |`;
    const sep = `| ${cols.map(() => "---").join(" | ")} |`;
    const body = rows.map((r) => `| ${cols.map((c) => r[c] ?? "").join(" | ")} |`).join("\n");
    return [head, sep, body].join("\n");
  }

  function toCsv(rows) {
    if (!rows.length) return "";
    const cols = [...new Set(rows.flatMap((r) => Object.keys(r)))];
    const esc = (v) => {
      const s = v == null ? "" : String(v);
      return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
    };
    return [cols.join(","), ...rows.map((r) => cols.map((c) => esc(r[c])).join(","))].join("\n") + "\n";
  }

  async function copyMd() {
    const rows = flattenForExport();
    if (!rows.length) {
      setRailStatus("nothing to copy");
      return false;
    }
    try {
      await navigator.clipboard.writeText(toMarkdown(rows));
      setRailStatus("copied markdown");
      return true;
    } catch {
      setRailStatus("copy failed (clipboard blocked)");
      return false;
    }
  }

  function exportCsv() {
    const rows = flattenForExport();
    if (!rows.length) {
      setRailStatus("nothing to export");
      return false;
    }
    const stamp = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
    const blob = new Blob([toCsv(rows)], { type: "text/csv;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = Object.assign(document.createElement("a"), {
      href: url, download: `inference-benchmark-${stamp}.csv`,
    });
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
    setRailStatus("exported csv");
    return true;
  }

  // ── React hooks (built on a snapshot of state via useState/useEffect) ─
  function useStoreSnapshot() {
    const [, setTick] = React.useState(0);
    React.useEffect(() => Store.subscribe(() => setTick((t) => (t + 1) | 0)), []);
    return Store.state;
  }

  function useClients() {
    const state = useStoreSnapshot();
    return React.useMemo(() => {
      const clients = state.rawClients.map((rec) => buildClient(rec, state.seriesByClient, state.activeTransport));
      const aggregate = buildAggregate(clients);
      return { clients, aggregate };
    }, [state.rawClients, state.seriesByClient, state.pendingChanges, state.activeTransport]);
  }

  function useSweep() {
    const state = useStoreSnapshot();
    return React.useMemo(() => adaptSweep(state.sweepStatus), [state.sweepStatus]);
  }

  function useTransports() {
    const state = useStoreSnapshot();
    return state.transports;
  }

  function useBatching() {
    return useStoreSnapshot().batching;
  }

  function useInference() {
    return useStoreSnapshot().inference;
  }

  function useHistory() {
    return useStoreSnapshot().history;
  }

  function useMeta() {
    const state = useStoreSnapshot();
    const uptimeS = Math.floor((Date.now() - state.bootAt) / 1000);
    const hh = String(Math.floor(uptimeS / 3600)).padStart(2, "0");
    const mm = String(Math.floor((uptimeS % 3600) / 60)).padStart(2, "0");
    const ss = String(uptimeS % 60).padStart(2, "0");
    return {
      serverHost: state.serverHost,
      activeTransport: state.activeTransport,
      uptime: `${hh}:${mm}:${ss}`,
      railStatus: state.railStatus,
    };
  }

  // ── Public surface ────────────────────────────────────────────────────
  window.Data = {
    Store,
    start,
    useClients,
    useSweep,
    useTransports,
    useBatching,
    useInference,
    useHistory,
    useMeta,
  };
  window.Actions = {
    applyTransport,
    applyBatching,
    applyInference,
    controlClient,
    clearClient,
    clearAll,
    copyMd,
    exportCsv,
  };

  start();
})();
