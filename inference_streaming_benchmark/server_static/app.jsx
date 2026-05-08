// Operator console — top-level wiring.
//
// Pulls live state from window.Data hooks and dispatches user input through
// window.Actions. No mock generators; no tweaks panel; no EDITMODE state.

const { useState, useEffect } = React;

const THEMES = ["light", "blueprint", "dark"];
const THEME_KEY = "infsb.theme";

function readInitialTheme() {
  try {
    const stored = localStorage.getItem(THEME_KEY);
    if (stored && THEMES.includes(stored)) return stored;
  } catch { /* localStorage may be blocked */ }
  return "light";
}

function applyTheme(theme) {
  const root = document.documentElement;
  if (theme === "light") delete root.dataset.theme;
  else root.dataset.theme = theme;
  try { localStorage.setItem(THEME_KEY, theme); } catch { /* ignore */ }
}

const App = () => {
  const { clients, aggregate } = window.Data.useClients();
  const sweep = window.Data.useSweep();
  const transportComparison = window.Data.useTransportComparison();
  const transports = window.Data.useTransports();
  const batching = window.Data.useBatching();
  const inference = window.Data.useInference();
  const history = window.Data.useHistory();
  const meta = window.Data.useMeta();

  const [sortBy, setSortBy] = useState("fps");
  const [theme, setTheme] = useState(readInitialTheme);

  useEffect(() => { applyTheme(theme); }, [theme]);

  const cycleTheme = () => setTheme((t) => THEMES[(THEMES.indexOf(t) + 1) % THEMES.length]);

  return (
    <div className="app">
      <header className="app-header">
        <h1><span className="accent">infsb</span> · central operator panel</h1>
        <span className="crumb">project · <b>inference-streaming-benchmark</b></span>
        <div className="meta">
          <span className="live-host">
            <span className="dot dot-live" style={{ marginRight: 6 }} />
            server <b>{meta.serverHost}</b>
          </span>
          <span>uptime {meta.uptime}</span>
          <button className="theme-toggle" onClick={cycleTheme} title={`theme: ${theme}`}>
            <span className="mono small">{theme}</span>
          </button>
        </div>
      </header>

      <ControlRail
        transports={transports}
        activeTransport={meta.activeTransport}
        batching={batching}
        inference={inference}
        railStatus={meta.railStatus}
      />

      <Hero
        aggregate={aggregate}
        history={history}
        clientCount={clients.length}
        activeTransport={meta.activeTransport || "—"}
        sweepProgress={{ completed: sweep.completed, total: sweep.total }}
      />

      {transportComparison.length > 0 && <TransportComparison rows={transportComparison} />}

      <RunForm transports={transports} sweep={sweep} />

      {sweep.total > 0 && <SweepPanel sweep={sweep} />}

      <ClientGrid clients={clients} sortBy={sortBy} onSortBy={setSortBy} />
    </div>
  );
};

ReactDOM.createRoot(document.getElementById("root")).render(<App />);
