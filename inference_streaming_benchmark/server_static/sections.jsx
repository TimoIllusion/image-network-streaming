// Section components for the operator dashboard.

const fmt = (v, p = 1) => (v == null || isNaN(v) ? "—" : v.toFixed(p));
const fmtInt = (v) => (v == null || isNaN(v) ? "—" : Math.round(v).toString());

const StatBlock = ({ label, value, unit, sub }) => (
  <div className="stat-block">
    <div className="stat-label">{label}</div>
    <div className="stat-value">
      <span>{value}</span>
      {unit && <span className="stat-unit">{unit}</span>}
    </div>
    {sub && <div className="stat-sub">{sub}</div>}
  </div>
);

// Numeric input that holds a local edit buffer and applies on blur / Enter.
// The legacy app used a "dirty bit" pattern to keep the polling loop from
// clobbering an in-progress edit; the equivalent here is: only sync from
// `value` while the user is not focused on the input.
const NumInput = ({ value, min, max, step, suffix, onApply, width }) => {
  const [draft, setDraft] = React.useState(String(value));
  const [focused, setFocused] = React.useState(false);
  React.useEffect(() => {
    if (!focused) setDraft(String(value));
  }, [value, focused]);
  const apply = () => {
    const n = Number.parseFloat(draft);
    if (Number.isFinite(n) && n !== value) onApply(n);
    else setDraft(String(value));
  };
  return (
    <input
      type="number"
      value={draft}
      min={min}
      max={max}
      step={step ?? 1}
      style={width ? { width } : undefined}
      onChange={(e) => setDraft(e.target.value)}
      onFocus={() => setFocused(true)}
      onBlur={() => { setFocused(false); apply(); }}
      onKeyDown={(e) => { if (e.key === "Enter") e.target.blur(); }}
    />
  );
};

// ── Top control rail ───────────────────────────────────────────────────
const ControlRail = ({ transports, activeTransport, batching, inference, railStatus }) => {
  const list = transports.length
    ? transports.map((t) => t.name)
    : ["imagezmq", "zmq_raw", "grpc", "websocket_raw", "http_multipart_raw", "http_multipart", "zmq", "websocket"];
  const startAll = () => {
    const body = { inference: true };
    if (activeTransport) body.backend = activeTransport;
    window.Actions.controlAll(body, "starting all clients...");
  };
  return (
    <div className="rail">
      <div className="rail-row">
        <div className="rail-label">transport</div>
        <div className="transport-selector">
          {list.map((t) => (
            <button
              key={t}
              className={`tx-chip ${activeTransport === t ? "active" : ""}`}
              onClick={() => window.Actions.applyTransport(t)}
            >
              {t}
            </button>
          ))}
        </div>
      </div>
      <div className="rail-row">
        <div className="rail-label">batching</div>
        <div className="rail-controls">
          <label className="switch">
            <input
              type="checkbox"
              checked={!!batching.enabled}
              onChange={(e) => window.Actions.applyBatching({ enabled: e.target.checked })}
            />
            <span className="switch-track"><span className="switch-thumb" /></span>
            <span>{batching.enabled ? "on" : "off"}</span>
          </label>
          <span className="rail-sep">/</span>
          <label className="num-input">
            size
            <NumInput
              value={batching.max_batch_size}
              min={1}
              max={64}
              onApply={(v) => window.Actions.applyBatching({ max_batch_size: Math.round(v) })}
            />
          </label>
          <label className="num-input">
            wait
            <NumInput
              value={batching.max_wait_ms}
              min={0}
              max={500}
              onApply={(v) => window.Actions.applyBatching({ max_wait_ms: v })}
            />
            <span className="suffix">ms</span>
          </label>
          <span className="rail-sep">/</span>
          <div className="rail-label">infer</div>
          <select
            className="select"
            value={inference.mode}
            onChange={(e) => window.Actions.applyInference({ mode: e.target.value })}
          >
            <option value="single">single</option>
            <option value="unsafe-multi">unsafe-multi</option>
            <option value="multi-instance">multi-instance</option>
          </select>
          {inference.mode === "multi-instance" && (
            <label className="num-input">
              n
              <NumInput
                value={inference.instances}
                min={1}
                max={8}
                onApply={(v) => window.Actions.applyInference({ instances: Math.max(1, Math.round(v)) })}
              />
            </label>
          )}
          <span className="rail-spacer" />
          {railStatus && <span className="rail-status mono small muted">{railStatus}</span>}
          <button className="btn" onClick={startAll}>start all</button>
          <button className="btn" onClick={() => window.Actions.controlAll({ inference: false }, "pausing all clients...")}>pause all</button>
          <button className="btn" onClick={() => window.Actions.controlAll({ mock_camera: true }, "enabling mock on all clients...")}>mock all</button>
          <button className="btn ghost" onClick={() => window.Actions.controlAll({ mock_camera: false }, "using real cameras on all clients...")}>real all</button>
          <button className="btn ghost" onClick={() => window.Actions.clearAll()}>clear stats</button>
          <button className="btn ghost" onClick={() => window.Actions.copyMd()}>copy md</button>
          <button className="btn ghost" onClick={() => window.Actions.exportCsv()}>export csv</button>
        </div>
      </div>
    </div>
  );
};

// ── Hero: aggregate live readout + waterfall + history ─────────────────
const Hero = ({ aggregate, history, clientCount, activeTransport, sweepProgress }) => {
  const sweepPct = sweepProgress.total > 0
    ? Math.round((100 * sweepProgress.completed) / sweepProgress.total)
    : 0;
  const sweepLabel = sweepProgress.total > 0
    ? `${sweepProgress.completed}/${sweepProgress.total}`
    : "—";
  return (
    <div className="hero card">
      <div className="hero-grid">
        <div className="hero-stats">
          <div className="hero-eyebrow">
            <span className="dot dot-live" />
            live · t = <span className="mono">{new Date().toISOString().slice(11, 19)}Z</span>
          </div>
          <div className="hero-headline">
            <span className="hero-fps">{fmtInt(aggregate.totalFps)}</span>
            <span className="hero-fps-unit">fps aggregate</span>
          </div>
          <div className="hero-substats">
            <StatBlock
              label="latency p50"
              value={fmt(aggregate.avgLat, 1)}
              unit="ms"
              sub={`p−min ${fmt(aggregate.minLat, 1)} · p−max ${fmt(aggregate.maxLat, 1)}`}
            />
            <StatBlock
              label="clients"
              value={`${aggregate.activeCount}/${clientCount}`}
              unit="active"
              sub={`transport · ${activeTransport}`}
            />
            <StatBlock
              label="sweep"
              value={sweepLabel}
              unit="runs"
              sub={sweepProgress.total > 0 ? `${sweepPct}% complete` : "idle"}
            />
          </div>
        </div>
        <div className="hero-chart">
          <div className="hero-chart-head">
            <span className="hero-chart-title">aggregate fps · last 120s</span>
            <span className="mono small muted">
              {history.length > 0
                ? `y · ${fmtInt(Math.min(...history))}–${fmtInt(Math.max(...history))} fps`
                : "y · — fps"}
            </span>
          </div>
          <AreaChart data={history} width={800} height={120} color="var(--accent)" />
          <div className="hero-chart-axis">
            <span>−120s</span><span>−90s</span><span>−60s</span><span>−30s</span><span>now</span>
          </div>
        </div>
      </div>

      <div className="hero-waterfall">
        <div className="hero-waterfall-head">
          <span className="hero-chart-title">
            latency breakdown · mean across {aggregate.activeCount} active client{aggregate.activeCount === 1 ? "" : "s"}
          </span>
          <span className="mono small muted">total {fmt(aggregate.avgLat, 1)} ms</span>
        </div>
        <div className="waterfall-stages">
          {STAGE_ORDER.map((k) => {
            const v = aggregate.stageAvg[k];
            const pct = aggregate.avgLat > 0 ? (v / aggregate.avgLat) * 100 : 0;
            return (
              <div key={k} className="stage-block" style={{ flex: pct || 1, minWidth: pct > 0.5 ? 0 : "auto" }}>
                <div className="stage-bar" style={{ background: STAGE_COLORS[k] }} />
                <div className="stage-label">
                  <span className="stage-name">{k}</span>
                  <span className="stage-value mono">{fmt(v, 1)}<span className="muted">ms</span></span>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
};

// ── Transport head-to-head comparison ──────────────────────────────────
const TransportComparison = ({ clients }) => {
  const byT = {};
  for (const c of clients) {
    if (!c.inferenceOn || c.timing.total <= 0) continue;
    if (!byT[c.transport]) byT[c.transport] = [];
    byT[c.transport].push(c);
  }
  const rows = Object.entries(byT).map(([t, cs]) => {
    const n = cs.length;
    const timing = { enc: 0, dec: 0, comms: 0, infer: 0, post: 0, wait: 0 };
    for (const c of cs) for (const k of STAGE_ORDER) timing[k] += c.timing[k];
    for (const k of STAGE_ORDER) timing[k] /= n;
    const total = STAGE_ORDER.reduce((a, k) => a + timing[k], 0);
    const fps = cs.reduce((a, c) => a + c.fps, 0) / n;
    return { transport: t, timing, total, fps, count: n };
  });
  rows.sort((a, b) => a.total - b.total);
  if (!rows.length) return null;
  const maxTotal = Math.max(...rows.map((r) => r.total)) || 1;
  const maxFps = Math.max(...rows.map((r) => r.fps)) || 1;

  return (
    <div className="card">
      <div className="card-head">
        <h2>transport · head-to-head</h2>
        <StageLegend />
      </div>
      <div className="comparison-grid">
        <div className="comparison-col">
          <div className="comparison-col-head mono small muted">total latency · stages stacked</div>
          {rows.map((r) => (
            <div key={r.transport} className="comparison-row">
              <div className="comparison-label mono">
                <span className="comparison-name">{r.transport}</span>
                <span className="muted">×{r.count}</span>
              </div>
              <StackedBar timing={r.timing} max={maxTotal} height={16} />
              <div className="comparison-value mono">{fmt(r.total, 1)}<span className="muted small">ms</span></div>
            </div>
          ))}
        </div>
        <div className="comparison-col">
          <div className="comparison-col-head mono small muted">throughput · per client</div>
          {rows.map((r) => (
            <BarRow key={r.transport} label={r.transport} value={r.fps} max={maxFps} suffix=" fps" color="var(--accent)" />
          ))}
        </div>
      </div>
    </div>
  );
};

Object.assign(window, { ControlRail, Hero, TransportComparison, fmt, fmtInt, StatBlock });
