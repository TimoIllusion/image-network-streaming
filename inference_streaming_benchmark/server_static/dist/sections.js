// Section components for the operator dashboard.

const fmt = (v, p = 1) => v == null || isNaN(v) ? "—" : v.toFixed(p);
const fmtInt = v => v == null || isNaN(v) ? "—" : Math.round(v).toString();
const StatBlock = ({
  label,
  value,
  unit,
  sub
}) => /*#__PURE__*/React.createElement("div", {
  className: "stat-block"
}, /*#__PURE__*/React.createElement("div", {
  className: "stat-label"
}, label), /*#__PURE__*/React.createElement("div", {
  className: "stat-value"
}, /*#__PURE__*/React.createElement("span", null, value), unit && /*#__PURE__*/React.createElement("span", {
  className: "stat-unit"
}, unit)), sub && /*#__PURE__*/React.createElement("div", {
  className: "stat-sub"
}, sub));

// Numeric input that holds a local edit buffer and applies on blur / Enter.
// The legacy app used a "dirty bit" pattern to keep the polling loop from
// clobbering an in-progress edit; the equivalent here is: only sync from
// `value` while the user is not focused on the input.
const NumInput = ({
  value,
  min,
  max,
  step,
  suffix,
  onApply,
  width
}) => {
  const [draft, setDraft] = React.useState(String(value));
  const [focused, setFocused] = React.useState(false);
  React.useEffect(() => {
    if (!focused) setDraft(String(value));
  }, [value, focused]);
  const apply = () => {
    const n = Number.parseFloat(draft);
    if (Number.isFinite(n) && n !== value) onApply(n);else setDraft(String(value));
  };
  return /*#__PURE__*/React.createElement("input", {
    type: "number",
    value: draft,
    min: min,
    max: max,
    step: step ?? 1,
    style: width ? {
      width
    } : undefined,
    onChange: e => setDraft(e.target.value),
    onFocus: () => setFocused(true),
    onBlur: () => {
      setFocused(false);
      apply();
    },
    onKeyDown: e => {
      if (e.key === "Enter") e.target.blur();
    }
  });
};

// ── Top control rail ───────────────────────────────────────────────────
const ControlRail = ({
  transports,
  activeTransport,
  batching,
  inference,
  railStatus
}) => {
  const list = transports.length ? transports.map(t => t.name) : ["imagezmq", "zmq_raw", "grpc", "websocket_raw", "http_multipart_raw", "http_multipart", "zmq", "websocket"];
  return /*#__PURE__*/React.createElement("div", {
    className: "rail"
  }, /*#__PURE__*/React.createElement("div", {
    className: "rail-row"
  }, /*#__PURE__*/React.createElement("div", {
    className: "rail-label"
  }, "transport"), /*#__PURE__*/React.createElement("div", {
    className: "transport-selector"
  }, list.map(t => /*#__PURE__*/React.createElement("button", {
    key: t,
    className: `tx-chip ${activeTransport === t ? "active" : ""}`,
    onClick: () => window.Actions.applyTransport(t)
  }, t)))), /*#__PURE__*/React.createElement("div", {
    className: "rail-row"
  }, /*#__PURE__*/React.createElement("div", {
    className: "rail-label"
  }, "batching"), /*#__PURE__*/React.createElement("div", {
    className: "rail-controls"
  }, /*#__PURE__*/React.createElement("label", {
    className: "switch"
  }, /*#__PURE__*/React.createElement("input", {
    type: "checkbox",
    checked: !!batching.enabled,
    onChange: e => window.Actions.applyBatching({
      enabled: e.target.checked
    })
  }), /*#__PURE__*/React.createElement("span", {
    className: "switch-track"
  }, /*#__PURE__*/React.createElement("span", {
    className: "switch-thumb"
  })), /*#__PURE__*/React.createElement("span", null, batching.enabled ? "on" : "off")), /*#__PURE__*/React.createElement("span", {
    className: "rail-sep"
  }, "/"), /*#__PURE__*/React.createElement("label", {
    className: "num-input"
  }, "size", /*#__PURE__*/React.createElement(NumInput, {
    value: batching.max_batch_size,
    min: 1,
    max: 64,
    onApply: v => window.Actions.applyBatching({
      max_batch_size: Math.round(v)
    })
  })), /*#__PURE__*/React.createElement("label", {
    className: "num-input"
  }, "wait", /*#__PURE__*/React.createElement(NumInput, {
    value: batching.max_wait_ms,
    min: 0,
    max: 500,
    onApply: v => window.Actions.applyBatching({
      max_wait_ms: v
    })
  }), /*#__PURE__*/React.createElement("span", {
    className: "suffix"
  }, "ms")), /*#__PURE__*/React.createElement("span", {
    className: "rail-sep"
  }, "/"), /*#__PURE__*/React.createElement("div", {
    className: "rail-label"
  }, "infer"), /*#__PURE__*/React.createElement("select", {
    className: "select",
    value: inference.mode,
    onChange: e => window.Actions.applyInference({
      mode: e.target.value
    })
  }, /*#__PURE__*/React.createElement("option", {
    value: "single"
  }, "single"), /*#__PURE__*/React.createElement("option", {
    value: "unsafe-multi"
  }, "unsafe-multi"), /*#__PURE__*/React.createElement("option", {
    value: "multi-instance"
  }, "multi-instance")), inference.mode === "multi-instance" && /*#__PURE__*/React.createElement("label", {
    className: "num-input"
  }, "n", /*#__PURE__*/React.createElement(NumInput, {
    value: inference.instances,
    min: 1,
    max: 8,
    onApply: v => window.Actions.applyInference({
      instances: Math.max(1, Math.round(v))
    })
  })), /*#__PURE__*/React.createElement("span", {
    className: "rail-spacer"
  }), railStatus && /*#__PURE__*/React.createElement("span", {
    className: "rail-status mono small muted"
  }, railStatus), /*#__PURE__*/React.createElement("button", {
    className: "btn ghost",
    onClick: () => window.Actions.clearAll()
  }, "clear stats"), /*#__PURE__*/React.createElement("button", {
    className: "btn ghost",
    onClick: () => window.Actions.copyMd()
  }, "copy md"), /*#__PURE__*/React.createElement("button", {
    className: "btn ghost",
    onClick: () => window.Actions.exportCsv()
  }, "export csv"))));
};

// ── Hero: aggregate live readout + waterfall + history ─────────────────
const Hero = ({
  aggregate,
  history,
  clientCount,
  activeTransport,
  sweepProgress
}) => {
  const sweepPct = sweepProgress.total > 0 ? Math.round(100 * sweepProgress.completed / sweepProgress.total) : 0;
  const sweepLabel = sweepProgress.total > 0 ? `${sweepProgress.completed}/${sweepProgress.total}` : "—";
  return /*#__PURE__*/React.createElement("div", {
    className: "hero card"
  }, /*#__PURE__*/React.createElement("div", {
    className: "hero-grid"
  }, /*#__PURE__*/React.createElement("div", {
    className: "hero-stats"
  }, /*#__PURE__*/React.createElement("div", {
    className: "hero-eyebrow"
  }, /*#__PURE__*/React.createElement("span", {
    className: "dot dot-live"
  }), "live \xB7 t = ", /*#__PURE__*/React.createElement("span", {
    className: "mono"
  }, new Date().toISOString().slice(11, 19), "Z")), /*#__PURE__*/React.createElement("div", {
    className: "hero-headline"
  }, /*#__PURE__*/React.createElement("span", {
    className: "hero-fps"
  }, fmtInt(aggregate.totalFps)), /*#__PURE__*/React.createElement("span", {
    className: "hero-fps-unit"
  }, "fps aggregate")), /*#__PURE__*/React.createElement("div", {
    className: "hero-substats"
  }, /*#__PURE__*/React.createElement(StatBlock, {
    label: "latency p50",
    value: fmt(aggregate.avgLat, 1),
    unit: "ms",
    sub: `p−min ${fmt(aggregate.minLat, 1)} · p−max ${fmt(aggregate.maxLat, 1)}`
  }), /*#__PURE__*/React.createElement(StatBlock, {
    label: "clients",
    value: `${aggregate.activeCount}/${clientCount}`,
    unit: "active",
    sub: `transport · ${activeTransport}`
  }), /*#__PURE__*/React.createElement(StatBlock, {
    label: "sweep",
    value: sweepLabel,
    unit: "runs",
    sub: sweepProgress.total > 0 ? `${sweepPct}% complete` : "idle"
  }))), /*#__PURE__*/React.createElement("div", {
    className: "hero-chart"
  }, /*#__PURE__*/React.createElement("div", {
    className: "hero-chart-head"
  }, /*#__PURE__*/React.createElement("span", {
    className: "hero-chart-title"
  }, "aggregate fps \xB7 last 120s"), /*#__PURE__*/React.createElement("span", {
    className: "mono small muted"
  }, history.length > 0 ? `y · ${fmtInt(Math.min(...history))}–${fmtInt(Math.max(...history))} fps` : "y · — fps")), /*#__PURE__*/React.createElement(AreaChart, {
    data: history,
    width: 800,
    height: 120,
    color: "var(--accent)"
  }), /*#__PURE__*/React.createElement("div", {
    className: "hero-chart-axis"
  }, /*#__PURE__*/React.createElement("span", null, "\u2212120s"), /*#__PURE__*/React.createElement("span", null, "\u221290s"), /*#__PURE__*/React.createElement("span", null, "\u221260s"), /*#__PURE__*/React.createElement("span", null, "\u221230s"), /*#__PURE__*/React.createElement("span", null, "now")))), /*#__PURE__*/React.createElement("div", {
    className: "hero-waterfall"
  }, /*#__PURE__*/React.createElement("div", {
    className: "hero-waterfall-head"
  }, /*#__PURE__*/React.createElement("span", {
    className: "hero-chart-title"
  }, "latency breakdown \xB7 mean across ", aggregate.activeCount, " active client", aggregate.activeCount === 1 ? "" : "s"), /*#__PURE__*/React.createElement("span", {
    className: "mono small muted"
  }, "total ", fmt(aggregate.avgLat, 1), " ms")), /*#__PURE__*/React.createElement("div", {
    className: "waterfall-stages"
  }, STAGE_ORDER.map(k => {
    const v = aggregate.stageAvg[k];
    const pct = aggregate.avgLat > 0 ? v / aggregate.avgLat * 100 : 0;
    return /*#__PURE__*/React.createElement("div", {
      key: k,
      className: "stage-block",
      style: {
        flex: pct || 1,
        minWidth: pct > 0.5 ? 0 : "auto"
      }
    }, /*#__PURE__*/React.createElement("div", {
      className: "stage-bar",
      style: {
        background: STAGE_COLORS[k]
      }
    }), /*#__PURE__*/React.createElement("div", {
      className: "stage-label"
    }, /*#__PURE__*/React.createElement("span", {
      className: "stage-name"
    }, k), /*#__PURE__*/React.createElement("span", {
      className: "stage-value mono"
    }, fmt(v, 1), /*#__PURE__*/React.createElement("span", {
      className: "muted"
    }, "ms"))));
  }))));
};

// ── Transport head-to-head comparison ──────────────────────────────────
const TransportComparison = ({
  clients
}) => {
  const byT = {};
  for (const c of clients) {
    if (!c.inferenceOn || c.timing.total <= 0) continue;
    if (!byT[c.transport]) byT[c.transport] = [];
    byT[c.transport].push(c);
  }
  const rows = Object.entries(byT).map(([t, cs]) => {
    const n = cs.length;
    const timing = {
      enc: 0,
      dec: 0,
      comms: 0,
      infer: 0,
      post: 0,
      wait: 0
    };
    for (const c of cs) for (const k of STAGE_ORDER) timing[k] += c.timing[k];
    for (const k of STAGE_ORDER) timing[k] /= n;
    const total = STAGE_ORDER.reduce((a, k) => a + timing[k], 0);
    const fps = cs.reduce((a, c) => a + c.fps, 0) / n;
    return {
      transport: t,
      timing,
      total,
      fps,
      count: n
    };
  });
  rows.sort((a, b) => a.total - b.total);
  if (!rows.length) return null;
  const maxTotal = Math.max(...rows.map(r => r.total)) || 1;
  const maxFps = Math.max(...rows.map(r => r.fps)) || 1;
  return /*#__PURE__*/React.createElement("div", {
    className: "card"
  }, /*#__PURE__*/React.createElement("div", {
    className: "card-head"
  }, /*#__PURE__*/React.createElement("h2", null, "transport \xB7 head-to-head"), /*#__PURE__*/React.createElement(StageLegend, null)), /*#__PURE__*/React.createElement("div", {
    className: "comparison-grid"
  }, /*#__PURE__*/React.createElement("div", {
    className: "comparison-col"
  }, /*#__PURE__*/React.createElement("div", {
    className: "comparison-col-head mono small muted"
  }, "total latency \xB7 stages stacked"), rows.map(r => /*#__PURE__*/React.createElement("div", {
    key: r.transport,
    className: "comparison-row"
  }, /*#__PURE__*/React.createElement("div", {
    className: "comparison-label mono"
  }, /*#__PURE__*/React.createElement("span", {
    className: "comparison-name"
  }, r.transport), /*#__PURE__*/React.createElement("span", {
    className: "muted"
  }, "\xD7", r.count)), /*#__PURE__*/React.createElement(StackedBar, {
    timing: r.timing,
    max: maxTotal,
    height: 16
  }), /*#__PURE__*/React.createElement("div", {
    className: "comparison-value mono"
  }, fmt(r.total, 1), /*#__PURE__*/React.createElement("span", {
    className: "muted small"
  }, "ms"))))), /*#__PURE__*/React.createElement("div", {
    className: "comparison-col"
  }, /*#__PURE__*/React.createElement("div", {
    className: "comparison-col-head mono small muted"
  }, "throughput \xB7 per client"), rows.map(r => /*#__PURE__*/React.createElement(BarRow, {
    key: r.transport,
    label: r.transport,
    value: r.fps,
    max: maxFps,
    suffix: " fps",
    color: "var(--accent)"
  })))));
};
Object.assign(window, {
  ControlRail,
  Hero,
  TransportComparison,
  fmt,
  fmtInt,
  StatBlock
});