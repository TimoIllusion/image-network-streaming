// Per-client cards + sweep results section.

const ClientCard = ({ client, activeTransport }) => {
  const {
    name, transport, ip, port, ui_url, inferenceOn, mockCam,
    frames, fps, timing, fpsSeries, latSeries, batch, ageMs,
  } = client;

  const onMock = () => window.Actions.controlClient(name, { mock_camera: !mockCam });
  const onPause = () => {
    if (inferenceOn) {
      window.Actions.controlClient(name, { inference: false });
    } else {
      const body = { inference: true };
      if (activeTransport) body.backend = activeTransport;
      window.Actions.controlClient(name, body);
    }
  };
  const onClear = () => window.Actions.clearClient(name);

  return (
    <div className={`client-card ${!inferenceOn ? "idle" : ""}`}>
      <div className="client-head">
        <div className="client-id">
          <span className={`status-dot ${inferenceOn ? "live" : "idle"}`} />
          {ui_url
            ? <a className="client-name mono" href={ui_url} target="_blank" rel="noopener">{name}</a>
            : <span className="client-name mono">{name}</span>}
          <span className="client-meta mono small muted">{ip}:{port}</span>
        </div>
        <div className="client-tags">
          <span className="tag tag-transport mono">{transport}</span>
          {mockCam && <span className="tag mono">mock</span>}
          {!inferenceOn && <span className="tag tag-warn mono">paused</span>}
          <span className="mono small muted">{ageMs}ms ago</span>
        </div>
      </div>

      <div className="client-body">
        <div className="client-metric">
          <div className="metric-label mono small muted">fps</div>
          <div className="metric-value mono">{fmt(fps, 1)}</div>
          <Spark data={fpsSeries} width={140} height={22} color="var(--accent)" fill />
        </div>
        <div className="client-metric">
          <div className="metric-label mono small muted">total ms</div>
          <div className="metric-value mono">{fmt(timing.total, 1)}</div>
          <Spark data={latSeries} width={140} height={22} color="oklch(0.55 0.14 280)" />
        </div>
        <div className="client-metric wide">
          <div className="metric-label mono small muted">latency stages</div>
          <Waterfall timing={timing} max={timing.total * 1.05 || 1} height={10} />
          <div className="stage-mini mono small">
            <span>enc <b>{fmt(timing.enc, 1)}</b></span>
            <span>dec <b>{fmt(timing.dec, 1)}</b></span>
            <span>comms <b>{fmt(timing.comms, 1)}</b></span>
            <span>infer <b>{fmt(timing.infer, 1)}</b></span>
            <span>post <b>{fmt(timing.post, 1)}</b></span>
          </div>
        </div>
        <div className="client-side">
          <div className="side-row mono small">
            <span className="muted">frames</span><span>{frames}</span>
          </div>
          <div className="side-row mono small">
            <span className="muted">batch</span><span>{fmt(batch, 1)}</span>
          </div>
          <div className="side-row mono small">
            <span className="muted">wait</span><span>{fmt(timing.wait, 1)}<span className="muted">ms</span></span>
          </div>
          <div className="client-actions">
            <button className={`btn xs ${mockCam ? "primary" : ""}`} onClick={onMock}>mock</button>
            <button className="btn xs" onClick={onPause}>{inferenceOn ? "pause" : "start"}</button>
            <button className="btn xs ghost" onClick={onClear} title="clear stats">clear</button>
          </div>
        </div>
      </div>
    </div>
  );
};

const ClientGrid = ({ clients, sortBy, onSortBy }) => {
  const sorted = [...clients].sort((a, b) => {
    if (sortBy === "fps") return b.fps - a.fps;
    if (sortBy === "latency") return a.timing.total - b.timing.total;
    if (sortBy === "transport") return a.transport.localeCompare(b.transport);
    return a.name.localeCompare(b.name);
  });
  const activeTransport = window.Data.useMeta().activeTransport;
  return (
    <div className="card">
      <div className="card-head">
        <h2>connected clients <span className="badge mono">{clients.length}</span></h2>
        <div className="sort-row mono small muted">
          <span>sort</span>
          {["name", "fps", "latency", "transport"].map((k) => (
            <button key={k} className={`sort-chip ${sortBy === k ? "active" : ""}`} onClick={() => onSortBy(k)}>{k}</button>
          ))}
        </div>
      </div>
      {clients.length === 0 ? (
        <p className="hint mono small muted" style={{ padding: "12px 4px" }}>
          No clients connected. Start a client with <span className="mono">INFSB_CONTROL_HOST</span> pointing here.
        </p>
      ) : (
        <div className="client-grid">
          {sorted.map((c) => <ClientCard key={c.name} client={c} activeTransport={activeTransport} />)}
        </div>
      )}
    </div>
  );
};

// ── Sweep progress + results ──────────────────────────────────────────
const SweepPanel = ({ sweep }) => {
  const { rows, completed, total, running } = sweep;
  if (!total) return null;
  const pct = (completed / total) * 100;

  const byTransport = {};
  for (const r of rows) {
    if (!byTransport[r.transport]) byTransport[r.transport] = [];
    byTransport[r.transport].push(r);
  }
  const allFps = rows.filter((r) => r.fps != null).map((r) => r.fps);
  const minF = allFps.length ? Math.min(...allFps) : 0;
  const maxF = allFps.length ? Math.max(...allFps) : 0;
  const fpsColor = (v) => {
    if (v == null) return "var(--track)";
    const t = (v - minF) / (maxF - minF || 1);
    return `oklch(${(0.92 - t * 0.30).toFixed(3)} ${(0.04 + t * 0.12).toFixed(3)} ${30 + t * 90})`;
  };

  const txCount = Object.keys(byTransport).length;
  const colCount = txCount ? Math.floor(rows.length / txCount) : 0;
  const headerCells = colCount > 0 ? rows.slice(0, colCount) : [];

  return (
    <div className="card">
      <div className="card-head">
        <h2>multi-run sweep</h2>
        <div className="sweep-status mono small">
          {running && <span className="dot dot-live" />}
          {running ? "running" : completed === total ? "complete" : "idle"} · <b>{completed}</b><span className="muted">/{total}</span>
        </div>
      </div>

      <div className="sweep-progress">
        <div className="progress-track">
          <div className="progress-fill" style={{ width: `${pct}%` }} />
          <div className="progress-marker" style={{ left: `${pct}%` }} />
        </div>
        <div className="progress-meta mono small muted">
          <span>{completed} done</span>
          <span>{total - completed} queued</span>
          <span>{Math.round(pct)}%</span>
        </div>
      </div>

      <div className="sweep-grid">
        {headerCells.length > 0 && (
          <div className="sweep-grid-head">
            <div className="mono small muted">transport</div>
            {headerCells.map((r, i) => (
              <div key={i} className="mono small muted sweep-cell-head">
                <div>b{r.batch.enabled ? r.batch.size : "○"}</div>
                <div className="muted">w{r.batch.wait}</div>
                <div className="muted">{r.infer.mode === "single" ? "s" : r.infer.mode === "unsafe-multi" ? "u" : `m×${r.infer.instances}`}</div>
              </div>
            ))}
          </div>
        )}
        {Object.entries(byTransport).map(([t, cells]) => (
          <div key={t} className="sweep-row">
            <div className="mono small sweep-tx">{t}</div>
            {cells.map((c) => (
              <div
                key={c.id}
                className={`sweep-cell ${c.status}`}
                style={c.status === "done" ? { background: fpsColor(c.fps) } : {}}
                title={c.status === "done"
                  ? `${t} · batch=${c.batch.enabled ? c.batch.size : "off"} · wait=${c.batch.wait}ms · ${c.infer.mode}\n${fmt(c.fps, 1)} fps · ${fmt(c.total_ms, 1)} ms`
                  : c.status}
              >
                {c.status === "done" && <span className="mono">{fmtInt(c.fps)}</span>}
                {c.status === "running" && <span className="mono blink">▸▸</span>}
                {c.status === "queued" && <span className="mono muted">·</span>}
              </div>
            ))}
          </div>
        ))}
      </div>

      {allFps.length > 0 && (
        <div className="sweep-legend mono small muted">
          <span>cell · fps · color = relative throughput</span>
          <span className="legend-gradient" />
          <span>{fmtInt(minF)} → {fmtInt(maxF)}</span>
          <span className="rail-spacer" />
          <span>b· batch size · w· wait ms · s/u/m· infer mode</span>
        </div>
      )}
    </div>
  );
};

Object.assign(window, { ClientCard, ClientGrid, SweepPanel });
