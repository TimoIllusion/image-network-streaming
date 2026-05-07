// Per-client cards + sweep results section.

const ClientCard = ({
  client,
  activeTransport
}) => {
  const {
    name,
    transport,
    ip,
    port,
    ui_url,
    inferenceOn,
    mockCam,
    frames,
    fps,
    timing,
    fpsSeries,
    latSeries,
    batch,
    ageMs
  } = client;
  const onMock = () => window.Actions.controlClient(name, {
    mock_camera: !mockCam
  });
  const onPause = () => {
    if (inferenceOn) {
      window.Actions.controlClient(name, {
        inference: false
      });
    } else {
      const body = {
        inference: true
      };
      if (activeTransport) body.backend = activeTransport;
      window.Actions.controlClient(name, body);
    }
  };
  const onClear = () => window.Actions.clearClient(name);
  return /*#__PURE__*/React.createElement("div", {
    className: `client-card ${!inferenceOn ? "idle" : ""}`
  }, /*#__PURE__*/React.createElement("div", {
    className: "client-head"
  }, /*#__PURE__*/React.createElement("div", {
    className: "client-id"
  }, /*#__PURE__*/React.createElement("span", {
    className: `status-dot ${inferenceOn ? "live" : "idle"}`
  }), ui_url ? /*#__PURE__*/React.createElement("a", {
    className: "client-name mono",
    href: ui_url,
    target: "_blank",
    rel: "noopener"
  }, name) : /*#__PURE__*/React.createElement("span", {
    className: "client-name mono"
  }, name), /*#__PURE__*/React.createElement("span", {
    className: "client-meta mono small muted"
  }, ip, ":", port)), /*#__PURE__*/React.createElement("div", {
    className: "client-tags"
  }, /*#__PURE__*/React.createElement("span", {
    className: "tag tag-transport mono"
  }, transport), mockCam && /*#__PURE__*/React.createElement("span", {
    className: "tag mono"
  }, "mock"), !inferenceOn && /*#__PURE__*/React.createElement("span", {
    className: "tag tag-warn mono"
  }, "paused"), /*#__PURE__*/React.createElement("span", {
    className: "mono small muted"
  }, ageMs, "ms ago"))), /*#__PURE__*/React.createElement("div", {
    className: "client-body"
  }, /*#__PURE__*/React.createElement("div", {
    className: "client-metric"
  }, /*#__PURE__*/React.createElement("div", {
    className: "metric-label mono small muted"
  }, "fps"), /*#__PURE__*/React.createElement("div", {
    className: "metric-value mono"
  }, fmt(fps, 1)), /*#__PURE__*/React.createElement(Spark, {
    data: fpsSeries,
    width: 140,
    height: 22,
    color: "var(--accent)",
    fill: true
  })), /*#__PURE__*/React.createElement("div", {
    className: "client-metric"
  }, /*#__PURE__*/React.createElement("div", {
    className: "metric-label mono small muted"
  }, "total ms"), /*#__PURE__*/React.createElement("div", {
    className: "metric-value mono"
  }, fmt(timing.total, 1)), /*#__PURE__*/React.createElement(Spark, {
    data: latSeries,
    width: 140,
    height: 22,
    color: "oklch(0.55 0.14 280)"
  })), /*#__PURE__*/React.createElement("div", {
    className: "client-metric wide"
  }, /*#__PURE__*/React.createElement("div", {
    className: "metric-label mono small muted"
  }, "latency stages"), /*#__PURE__*/React.createElement(Waterfall, {
    timing: timing,
    max: timing.total * 1.05 || 1,
    height: 10
  }), /*#__PURE__*/React.createElement("div", {
    className: "stage-mini mono small"
  }, /*#__PURE__*/React.createElement("span", null, "enc ", /*#__PURE__*/React.createElement("b", null, fmt(timing.enc, 1))), /*#__PURE__*/React.createElement("span", null, "dec ", /*#__PURE__*/React.createElement("b", null, fmt(timing.dec, 1))), /*#__PURE__*/React.createElement("span", null, "comms ", /*#__PURE__*/React.createElement("b", null, fmt(timing.comms, 1))), /*#__PURE__*/React.createElement("span", null, "infer ", /*#__PURE__*/React.createElement("b", null, fmt(timing.infer, 1))), /*#__PURE__*/React.createElement("span", null, "post ", /*#__PURE__*/React.createElement("b", null, fmt(timing.post, 1))))), /*#__PURE__*/React.createElement("div", {
    className: "client-side"
  }, /*#__PURE__*/React.createElement("div", {
    className: "side-row mono small"
  }, /*#__PURE__*/React.createElement("span", {
    className: "muted"
  }, "frames"), /*#__PURE__*/React.createElement("span", null, frames)), /*#__PURE__*/React.createElement("div", {
    className: "side-row mono small"
  }, /*#__PURE__*/React.createElement("span", {
    className: "muted"
  }, "batch"), /*#__PURE__*/React.createElement("span", null, fmt(batch, 1))), /*#__PURE__*/React.createElement("div", {
    className: "side-row mono small"
  }, /*#__PURE__*/React.createElement("span", {
    className: "muted"
  }, "wait"), /*#__PURE__*/React.createElement("span", null, fmt(timing.wait, 1), /*#__PURE__*/React.createElement("span", {
    className: "muted"
  }, "ms"))), /*#__PURE__*/React.createElement("div", {
    className: "client-actions"
  }, /*#__PURE__*/React.createElement("button", {
    className: `btn xs ${mockCam ? "primary" : ""}`,
    onClick: onMock
  }, "mock"), /*#__PURE__*/React.createElement("button", {
    className: "btn xs",
    onClick: onPause
  }, inferenceOn ? "pause" : "start"), /*#__PURE__*/React.createElement("button", {
    className: "btn xs ghost",
    onClick: onClear,
    title: "clear stats"
  }, "clear")))));
};
const ClientGrid = ({
  clients,
  sortBy,
  onSortBy
}) => {
  const sorted = [...clients].sort((a, b) => {
    if (sortBy === "fps") return b.fps - a.fps;
    if (sortBy === "latency") return a.timing.total - b.timing.total;
    if (sortBy === "transport") return a.transport.localeCompare(b.transport);
    return a.name.localeCompare(b.name);
  });
  const activeTransport = window.Data.useMeta().activeTransport;
  return /*#__PURE__*/React.createElement("div", {
    className: "card"
  }, /*#__PURE__*/React.createElement("div", {
    className: "card-head"
  }, /*#__PURE__*/React.createElement("h2", null, "connected clients ", /*#__PURE__*/React.createElement("span", {
    className: "badge mono"
  }, clients.length)), /*#__PURE__*/React.createElement("div", {
    className: "sort-row mono small muted"
  }, /*#__PURE__*/React.createElement("span", null, "sort"), ["name", "fps", "latency", "transport"].map(k => /*#__PURE__*/React.createElement("button", {
    key: k,
    className: `sort-chip ${sortBy === k ? "active" : ""}`,
    onClick: () => onSortBy(k)
  }, k)))), clients.length === 0 ? /*#__PURE__*/React.createElement("p", {
    className: "hint mono small muted",
    style: {
      padding: "12px 4px"
    }
  }, "No clients connected. Start a client with ", /*#__PURE__*/React.createElement("span", {
    className: "mono"
  }, "INFSB_CONTROL_HOST"), " pointing here.") : /*#__PURE__*/React.createElement("div", {
    className: "client-grid"
  }, sorted.map(c => /*#__PURE__*/React.createElement(ClientCard, {
    key: c.name,
    client: c,
    activeTransport: activeTransport
  }))));
};

// ── Sweep start form ──────────────────────────────────────────────────
const parseNumberList = (value, parser) => String(value).split(",").map(item => item.trim()).filter(Boolean).map(parser);
const RunForm = ({
  transports,
  sweep
}) => {
  const transportNames = transports.length ? transports.map(t => t.name) : ["imagezmq", "zmq_raw", "grpc", "websocket_raw", "http_multipart_raw", "http_multipart", "zmq", "websocket"];
  const [selectedTransports, setSelectedTransports] = React.useState([]);
  const [batchOff, setBatchOff] = React.useState(true);
  const [batchOn, setBatchOn] = React.useState(true);
  const [batchSizes, setBatchSizes] = React.useState("1,2,4,8");
  const [batchWaits, setBatchWaits] = React.useState("0,5,10,20");
  const [inferSingle, setInferSingle] = React.useState(true);
  const [inferUnsafe, setInferUnsafe] = React.useState(false);
  const [inferMulti, setInferMulti] = React.useState(false);
  const [inferInstances, setInferInstances] = React.useState("1,2");
  const [cameraMode, setCameraMode] = React.useState("current");
  const [warmupS, setWarmupS] = React.useState(2);
  const [durationS, setDurationS] = React.useState(10);
  const [submitting, setSubmitting] = React.useState(false);
  const [message, setMessage] = React.useState("");
  const [error, setError] = React.useState("");
  React.useEffect(() => {
    setSelectedTransports(prev => {
      if (prev.length) return prev.filter(name => transportNames.includes(name));
      return transportNames;
    });
  }, [transportNames.join("|")]);
  const toggleTransport = name => {
    setSelectedTransports(prev => prev.includes(name) ? prev.filter(item => item !== name) : [...prev, name]);
  };
  const setAllTransports = checked => {
    setSelectedTransports(checked ? transportNames : []);
  };
  const buildBody = () => {
    const batchModes = [];
    if (batchOff) batchModes.push("off");
    if (batchOn) batchModes.push("on");
    const inferenceModes = [];
    if (inferSingle) inferenceModes.push("single");
    if (inferUnsafe) inferenceModes.push("unsafe-multi");
    if (inferMulti) inferenceModes.push("multi-instance");
    const body = {
      transports: selectedTransports,
      batch_modes: batchModes,
      batch_sizes: parseNumberList(batchSizes, item => Number.parseInt(item, 10)),
      batch_waits_ms: parseNumberList(batchWaits, Number.parseFloat),
      inference_modes: inferenceModes,
      inference_instances: parseNumberList(inferInstances, item => Number.parseInt(item, 10)),
      warmup_s: Number.parseFloat(warmupS),
      duration_s: Number.parseFloat(durationS)
    };
    if (cameraMode === "mock") body.mock_camera = true;
    if (cameraMode === "real") body.mock_camera = false;
    return body;
  };
  const onSubmit = async event => {
    event.preventDefault();
    setSubmitting(true);
    setError("");
    setMessage("starting sweep...");
    try {
      const payload = await window.Actions.startSweep(buildBody());
      setMessage(`started ${payload.plan?.length || 0} runs`);
    } catch (e) {
      setError(e.message || String(e));
      setMessage("");
    } finally {
      setSubmitting(false);
    }
  };
  const disabled = submitting || !!sweep.running;
  const allSelected = selectedTransports.length === transportNames.length;
  return /*#__PURE__*/React.createElement("form", {
    className: "card run-form",
    onSubmit: onSubmit
  }, /*#__PURE__*/React.createElement("div", {
    className: "card-head"
  }, /*#__PURE__*/React.createElement("h2", null, "start sweep"), /*#__PURE__*/React.createElement("div", {
    className: "sweep-status mono small"
  }, sweep.running && /*#__PURE__*/React.createElement("span", {
    className: "dot dot-live"
  }), sweep.running ? "running" : message || "ready")), /*#__PURE__*/React.createElement("div", {
    className: "run-form-grid"
  }, /*#__PURE__*/React.createElement("fieldset", {
    className: "run-fieldset run-fieldset-wide"
  }, /*#__PURE__*/React.createElement("legend", null, "transports"), /*#__PURE__*/React.createElement("label", {
    className: "check-chip all"
  }, /*#__PURE__*/React.createElement("input", {
    type: "checkbox",
    checked: allSelected,
    onChange: e => setAllTransports(e.target.checked),
    disabled: disabled
  }), "all"), /*#__PURE__*/React.createElement("div", {
    className: "check-grid"
  }, transportNames.map(name => /*#__PURE__*/React.createElement("label", {
    key: name,
    className: "check-chip"
  }, /*#__PURE__*/React.createElement("input", {
    type: "checkbox",
    checked: selectedTransports.includes(name),
    onChange: () => toggleTransport(name),
    disabled: disabled
  }), name)))), /*#__PURE__*/React.createElement("fieldset", {
    className: "run-fieldset"
  }, /*#__PURE__*/React.createElement("legend", null, "batching"), /*#__PURE__*/React.createElement("div", {
    className: "inline-checks"
  }, /*#__PURE__*/React.createElement("label", {
    className: "check-chip"
  }, /*#__PURE__*/React.createElement("input", {
    type: "checkbox",
    checked: batchOff,
    onChange: e => setBatchOff(e.target.checked),
    disabled: disabled
  }), "off"), /*#__PURE__*/React.createElement("label", {
    className: "check-chip"
  }, /*#__PURE__*/React.createElement("input", {
    type: "checkbox",
    checked: batchOn,
    onChange: e => setBatchOn(e.target.checked),
    disabled: disabled
  }), "on")), /*#__PURE__*/React.createElement("label", {
    className: "text-input mono small"
  }, "sizes", /*#__PURE__*/React.createElement("input", {
    value: batchSizes,
    onChange: e => setBatchSizes(e.target.value),
    disabled: disabled
  })), /*#__PURE__*/React.createElement("label", {
    className: "text-input mono small"
  }, "waits ms", /*#__PURE__*/React.createElement("input", {
    value: batchWaits,
    onChange: e => setBatchWaits(e.target.value),
    disabled: disabled
  }))), /*#__PURE__*/React.createElement("fieldset", {
    className: "run-fieldset"
  }, /*#__PURE__*/React.createElement("legend", null, "inference"), /*#__PURE__*/React.createElement("div", {
    className: "inline-checks wrap"
  }, /*#__PURE__*/React.createElement("label", {
    className: "check-chip"
  }, /*#__PURE__*/React.createElement("input", {
    type: "checkbox",
    checked: inferSingle,
    onChange: e => setInferSingle(e.target.checked),
    disabled: disabled
  }), "single"), /*#__PURE__*/React.createElement("label", {
    className: "check-chip"
  }, /*#__PURE__*/React.createElement("input", {
    type: "checkbox",
    checked: inferUnsafe,
    onChange: e => setInferUnsafe(e.target.checked),
    disabled: disabled
  }), "unsafe-multi"), /*#__PURE__*/React.createElement("label", {
    className: "check-chip"
  }, /*#__PURE__*/React.createElement("input", {
    type: "checkbox",
    checked: inferMulti,
    onChange: e => setInferMulti(e.target.checked),
    disabled: disabled
  }), "multi-instance")), /*#__PURE__*/React.createElement("label", {
    className: "text-input mono small"
  }, "instances", /*#__PURE__*/React.createElement("input", {
    value: inferInstances,
    onChange: e => setInferInstances(e.target.value),
    disabled: disabled
  }))), /*#__PURE__*/React.createElement("fieldset", {
    className: "run-fieldset"
  }, /*#__PURE__*/React.createElement("legend", null, "camera + timing"), /*#__PURE__*/React.createElement("div", {
    className: "inline-checks wrap"
  }, [["current", "current"], ["mock", "mock"], ["real", "real"]].map(([value, label]) => /*#__PURE__*/React.createElement("label", {
    key: value,
    className: "check-chip"
  }, /*#__PURE__*/React.createElement("input", {
    type: "radio",
    name: "sweep-camera",
    checked: cameraMode === value,
    onChange: () => setCameraMode(value),
    disabled: disabled
  }), label))), /*#__PURE__*/React.createElement("label", {
    className: "text-input mono small"
  }, "warmup s", /*#__PURE__*/React.createElement("input", {
    type: "number",
    min: "0",
    step: "0.1",
    value: warmupS,
    onChange: e => setWarmupS(e.target.value),
    disabled: disabled
  })), /*#__PURE__*/React.createElement("label", {
    className: "text-input mono small"
  }, "duration s", /*#__PURE__*/React.createElement("input", {
    type: "number",
    min: "0.1",
    step: "0.1",
    value: durationS,
    onChange: e => setDurationS(e.target.value),
    disabled: disabled
  })), /*#__PURE__*/React.createElement("button", {
    className: "btn primary run-submit",
    type: "submit",
    disabled: disabled
  }, sweep.running ? "sweep running" : submitting ? "starting..." : "start sweep"))), error && /*#__PURE__*/React.createElement("div", {
    className: "run-message error mono small"
  }, error));
};

// ── Sweep progress + results ──────────────────────────────────────────
const SweepPanel = ({
  sweep
}) => {
  const {
    rows,
    completed,
    total,
    running
  } = sweep;
  if (!total) return null;
  const pct = completed / total * 100;
  const byTransport = {};
  for (const r of rows) {
    if (!byTransport[r.transport]) byTransport[r.transport] = [];
    byTransport[r.transport].push(r);
  }
  const allFps = rows.filter(r => r.fps != null).map(r => r.fps);
  const minF = allFps.length ? Math.min(...allFps) : 0;
  const maxF = allFps.length ? Math.max(...allFps) : 0;
  const fpsColor = v => {
    if (v == null) return "var(--track)";
    const t = (v - minF) / (maxF - minF || 1);
    return `oklch(${(0.92 - t * 0.30).toFixed(3)} ${(0.04 + t * 0.12).toFixed(3)} ${30 + t * 90})`;
  };
  const txCount = Object.keys(byTransport).length;
  const colCount = txCount ? Math.floor(rows.length / txCount) : 0;
  const headerCells = colCount > 0 ? rows.slice(0, colCount) : [];
  return /*#__PURE__*/React.createElement("div", {
    className: "card"
  }, /*#__PURE__*/React.createElement("div", {
    className: "card-head"
  }, /*#__PURE__*/React.createElement("h2", null, "multi-run sweep"), /*#__PURE__*/React.createElement("div", {
    className: "sweep-status mono small"
  }, running && /*#__PURE__*/React.createElement("span", {
    className: "dot dot-live"
  }), running ? "running" : completed === total ? "complete" : "idle", " \xB7 ", /*#__PURE__*/React.createElement("b", null, completed), /*#__PURE__*/React.createElement("span", {
    className: "muted"
  }, "/", total))), /*#__PURE__*/React.createElement("div", {
    className: "sweep-progress"
  }, /*#__PURE__*/React.createElement("div", {
    className: "progress-track"
  }, /*#__PURE__*/React.createElement("div", {
    className: "progress-fill",
    style: {
      width: `${pct}%`
    }
  }), /*#__PURE__*/React.createElement("div", {
    className: "progress-marker",
    style: {
      left: `${pct}%`
    }
  })), /*#__PURE__*/React.createElement("div", {
    className: "progress-meta mono small muted"
  }, /*#__PURE__*/React.createElement("span", null, completed, " done"), /*#__PURE__*/React.createElement("span", null, total - completed, " queued"), /*#__PURE__*/React.createElement("span", null, Math.round(pct), "%"))), /*#__PURE__*/React.createElement("div", {
    className: "sweep-grid"
  }, headerCells.length > 0 && /*#__PURE__*/React.createElement("div", {
    className: "sweep-grid-head"
  }, /*#__PURE__*/React.createElement("div", {
    className: "mono small muted"
  }, "transport"), headerCells.map((r, i) => /*#__PURE__*/React.createElement("div", {
    key: i,
    className: "mono small muted sweep-cell-head"
  }, /*#__PURE__*/React.createElement("div", null, "b", r.batch.enabled ? r.batch.size : "○"), /*#__PURE__*/React.createElement("div", {
    className: "muted"
  }, "w", r.batch.wait), /*#__PURE__*/React.createElement("div", {
    className: "muted"
  }, r.infer.mode === "single" ? "s" : r.infer.mode === "unsafe-multi" ? "u" : `m×${r.infer.instances}`), /*#__PURE__*/React.createElement("div", {
    className: "muted"
  }, r.camera === "mock" ? "cam m" : r.camera === "real" ? "cam r" : "cam -")))), Object.entries(byTransport).map(([t, cells]) => /*#__PURE__*/React.createElement("div", {
    key: t,
    className: "sweep-row"
  }, /*#__PURE__*/React.createElement("div", {
    className: "mono small sweep-tx"
  }, t), cells.map(c => /*#__PURE__*/React.createElement("div", {
    key: c.id,
    className: `sweep-cell ${c.status}`,
    style: c.status === "done" ? {
      background: fpsColor(c.fps)
    } : {},
    title: c.status === "done" ? `${t} · batch=${c.batch.enabled ? c.batch.size : "off"} · wait=${c.batch.wait}ms · ${c.infer.mode} · camera=${c.camera}\n${fmt(c.fps, 1)} fps · ${fmt(c.total_ms, 1)} ms` : c.status
  }, c.status === "done" && /*#__PURE__*/React.createElement("span", {
    className: "mono"
  }, fmtInt(c.fps)), c.status === "running" && /*#__PURE__*/React.createElement("span", {
    className: "mono blink"
  }, "\u25B8\u25B8"), c.status === "queued" && /*#__PURE__*/React.createElement("span", {
    className: "mono muted"
  }, "\xB7")))))), allFps.length > 0 && /*#__PURE__*/React.createElement("div", {
    className: "sweep-legend mono small muted"
  }, /*#__PURE__*/React.createElement("span", null, "cell \xB7 fps \xB7 color = relative throughput"), /*#__PURE__*/React.createElement("span", {
    className: "legend-gradient"
  }), /*#__PURE__*/React.createElement("span", null, fmtInt(minF), " \u2192 ", fmtInt(maxF)), /*#__PURE__*/React.createElement("span", {
    className: "rail-spacer"
  }), /*#__PURE__*/React.createElement("span", null, "b\xB7 batch size \xB7 w\xB7 wait ms \xB7 s/u/m\xB7 infer mode \xB7 cam m/r/-")));
};
Object.assign(window, {
  ClientCard,
  ClientGrid,
  RunForm,
  SweepPanel
});