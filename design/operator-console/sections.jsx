// Section components for the operator dashboard

const fmt = (v, p = 1) => (v == null || isNaN(v) ? "—" : v.toFixed(p));
const fmtInt = (v) => (v == null || isNaN(v) ? "—" : Math.round(v).toString());

const Pill = ({ children, tone = "default" }) => (
  <span className={`pill pill-${tone}`}>{children}</span>
);

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

// ── Top control rail ───────────────────────────────────────────────────
const ControlRail = ({ activeTransport, onTransport, batchOn, onBatchToggle, batchSize, batchWait, onBatchSize, onBatchWait, inferMode, onInferMode, inferInstances, onInferInstances, onClearAll }) => {
  return (
    <div className="rail">
      <div className="rail-row">
        <div className="rail-label">transport</div>
        <div className="transport-selector">
          {window.MOCK.TRANSPORTS.map((t) => (
            <button key={t}
              className={`tx-chip ${activeTransport === t ? "active" : ""}`}
              onClick={() => onTransport(t)}>
              {t}
            </button>
          ))}
        </div>
      </div>
      <div className="rail-row">
        <div className="rail-label">batching</div>
        <div className="rail-controls">
          <label className="switch">
            <input type="checkbox" checked={batchOn} onChange={(e) => onBatchToggle(e.target.checked)} />
            <span className="switch-track"><span className="switch-thumb" /></span>
            <span>{batchOn ? "on" : "off"}</span>
          </label>
          <span className="rail-sep">/</span>
          <label className="num-input">
            size
            <input type="number" min={1} max={64} value={batchSize} onChange={(e) => onBatchSize(+e.target.value)} />
          </label>
          <label className="num-input">
            wait
            <input type="number" min={0} max={500} value={batchWait} onChange={(e) => onBatchWait(+e.target.value)} />
            <span className="suffix">ms</span>
          </label>
          <span className="rail-sep">/</span>
          <div className="rail-label">infer</div>
          <select className="select" value={inferMode} onChange={(e) => onInferMode(e.target.value)}>
            <option value="single">single</option>
            <option value="unsafe-multi">unsafe-multi</option>
            <option value="multi-instance">multi-instance</option>
          </select>
          {inferMode === "multi-instance" && (
            <label className="num-input">
              n
              <input type="number" min={1} max={8} value={inferInstances} onChange={(e) => onInferInstances(+e.target.value)} />
            </label>
          )}
          <span className="rail-spacer" />
          <button className="btn ghost" onClick={onClearAll}>clear stats</button>
          <button className="btn ghost">copy md</button>
          <button className="btn ghost">export csv</button>
        </div>
      </div>
    </div>
  );
};

// ── Hero: aggregate live readout + waterfall + history ─────────────────
const Hero = ({ aggregate, history, clientCount, activeTransport, sweepProgress }) => {
  const stageMax = Math.max(...Object.values(aggregate.stageAvg)) || 1;
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
            <StatBlock label="latency p50" value={fmt(aggregate.avgLat, 1)} unit="ms" sub={`p−min ${fmt(aggregate.minLat, 1)} · p−max ${fmt(aggregate.maxLat, 1)}`} />
            <StatBlock label="clients" value={`${aggregate.activeCount}/${clientCount}`} unit="active" sub={`transport · ${activeTransport}`} />
            <StatBlock label="sweep" value={`${sweepProgress.completed}/${sweepProgress.total}`} unit="runs" sub={`${Math.round(100 * sweepProgress.completed / sweepProgress.total)}% · ETA 03:42`} />
          </div>
        </div>
        <div className="hero-chart">
          <div className="hero-chart-head">
            <span className="hero-chart-title">aggregate fps · last 120s</span>
            <span className="mono small muted">y · 200–340 fps</span>
          </div>
          <AreaChart data={history} width={800} height={120} color="var(--accent)" />
          <div className="hero-chart-axis">
            <span>−120s</span><span>−90s</span><span>−60s</span><span>−30s</span><span>now</span>
          </div>
        </div>
      </div>

      <div className="hero-waterfall">
        <div className="hero-waterfall-head">
          <span className="hero-chart-title">latency breakdown · mean across {aggregate.activeCount} active clients</span>
          <span className="mono small muted">total {fmt(aggregate.avgLat, 1)} ms</span>
        </div>
        <div className="waterfall-stages">
          {STAGE_ORDER.map((k) => {
            const v = aggregate.stageAvg[k];
            const pct = (v / aggregate.avgLat) * 100;
            return (
              <div key={k} className="stage-block" style={{ flex: pct, minWidth: pct > 0.5 ? 0 : "auto" }}>
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
  // group clients by transport, compute averages
  const byT = {};
  for (const c of clients) {
    if (!c.inferenceOn) continue;
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
  const maxTotal = Math.max(...rows.map(r => r.total));
  const maxFps = Math.max(...rows.map(r => r.fps));

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

Object.assign(window, { ControlRail, Hero, TransportComparison, fmt, fmtInt, StatBlock, Pill });
