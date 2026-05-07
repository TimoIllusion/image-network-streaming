// Main app — wires data and sections together

const { useState, useEffect, useMemo } = React;

const App = () => {
  const [seed, setSeed] = useState(7);
  const [tweaks, setTweak] = useTweaks(/*EDITMODE-BEGIN*/{
    "density": "default",
    "live": true,
    "showSweep": true,
    "showComparison": true,
    "accentHue": 30
  }/*EDITMODE-END*/);

  // Live re-seed for jitter
  const [tick, setTick] = useState(0);
  useEffect(() => {
    if (!tweaks.live) return;
    const id = setInterval(() => setTick((t) => t + 1), 1500);
    return () => clearInterval(id);
  }, [tweaks.live]);

  const clients = useMemo(() => MOCK.buildClients(seed + tick), [seed, tick]);
  const aggregate = useMemo(() => MOCK.buildAggregate(clients), [clients]);
  const history = useMemo(() => MOCK.buildAggregateHistory(seed + Math.floor(tick / 3)), [seed, tick]);
  const sweep = useMemo(() => MOCK.buildSweepResults(seed), [seed]);

  const [activeTransport, setActiveTransport] = useState("http_multipart_raw");
  const [batchOn, setBatchOn] = useState(true);
  const [batchSize, setBatchSize] = useState(8);
  const [batchWait, setBatchWait] = useState(10);
  const [inferMode, setInferMode] = useState("multi-instance");
  const [inferInstances, setInferInstances] = useState(2);
  const [sortBy, setSortBy] = useState("fps");

  // Update accent hue
  useEffect(() => {
    const root = document.documentElement;
    root.style.setProperty("--accent", `oklch(0.62 0.15 ${tweaks.accentHue})`);
    root.style.setProperty("--accent-strong", `oklch(0.50 0.16 ${tweaks.accentHue})`);
    root.style.setProperty("--accent-soft", `oklch(0.95 0.04 ${tweaks.accentHue})`);
  }, [tweaks.accentHue]);

  return (
    <div className="app">
      <header className="app-header">
        <h1><span className="accent">infsb</span> · central operator panel</h1>
        <span className="crumb">project · <b>inference-streaming-benchmark</b> · v0.4.2</span>
        <div className="meta">
          <span className="live-host"><span className="dot dot-live" style={{ marginRight: 6 }} />server <b>workstation-3090</b>:9000</span>
          <span>uptime 04:21:18</span>
          <span>build 84772d3b</span>
        </div>
      </header>

      <ControlRail
        activeTransport={activeTransport}
        onTransport={setActiveTransport}
        batchOn={batchOn}
        onBatchToggle={setBatchOn}
        batchSize={batchSize}
        onBatchSize={setBatchSize}
        batchWait={batchWait}
        onBatchWait={setBatchWait}
        inferMode={inferMode}
        onInferMode={setInferMode}
        inferInstances={inferInstances}
        onInferInstances={setInferInstances}
        onClearAll={() => setSeed(s => s + 100)}
      />

      <Hero
        aggregate={aggregate}
        history={history}
        clientCount={clients.length}
        activeTransport={activeTransport}
        sweepProgress={{ completed: sweep.completed, total: sweep.total }}
      />

      {tweaks.showComparison && <TransportComparison clients={clients} />}

      {tweaks.showSweep && <SweepPanel sweep={sweep} />}

      <ClientGrid clients={clients} density={tweaks.density} sortBy={sortBy} onSortBy={setSortBy} />

      <TweaksPanel title="Tweaks">
        <TweakSection label="Display" />
        <TweakRadio
          label="density"
          value={tweaks.density}
          options={["compact", "default", "spacious"]}
          onChange={(v) => setTweak("density", v)}
        />
        <TweakToggle label="live simulation" value={tweaks.live} onChange={(v) => setTweak("live", v)} />
        <TweakToggle label="comparison" value={tweaks.showComparison} onChange={(v) => setTweak("showComparison", v)} />
        <TweakToggle label="sweep panel" value={tweaks.showSweep} onChange={(v) => setTweak("showSweep", v)} />
        <TweakSection label="Theme" />
        <TweakRadio
          label="palette"
          value={document.documentElement.dataset.theme || "light"}
          options={["light", "blueprint", "dark"]}
          onChange={(v) => {
            if (v === "light") delete document.documentElement.dataset.theme;
            else document.documentElement.dataset.theme = v;
          }}
        />
        <TweakSlider label="accent hue" value={tweaks.accentHue} min={0} max={360} step={5} onChange={(v) => setTweak("accentHue", v)} />
      </TweaksPanel>
    </div>
  );
};

ReactDOM.createRoot(document.getElementById("root")).render(<App />);
