// Operator console — top-level wiring.
//
// Pulls live state from window.Data hooks and dispatches user input through
// window.Actions. No mock generators; no tweaks panel; no EDITMODE state.

const {
  useState,
  useEffect
} = React;
const THEMES = ["light", "blueprint", "dark"];
const THEME_KEY = "infsb.theme";
function readInitialTheme() {
  try {
    const stored = localStorage.getItem(THEME_KEY);
    if (stored && THEMES.includes(stored)) return stored;
  } catch {/* localStorage may be blocked */}
  return "light";
}
function applyTheme(theme) {
  const root = document.documentElement;
  if (theme === "light") delete root.dataset.theme;else root.dataset.theme = theme;
  try {
    localStorage.setItem(THEME_KEY, theme);
  } catch {/* ignore */}
}
const App = () => {
  const {
    clients,
    aggregate
  } = window.Data.useClients();
  const sweep = window.Data.useSweep();
  const transports = window.Data.useTransports();
  const batching = window.Data.useBatching();
  const inference = window.Data.useInference();
  const history = window.Data.useHistory();
  const meta = window.Data.useMeta();
  const [sortBy, setSortBy] = useState("fps");
  const [theme, setTheme] = useState(readInitialTheme);
  useEffect(() => {
    applyTheme(theme);
  }, [theme]);
  const cycleTheme = () => setTheme(t => THEMES[(THEMES.indexOf(t) + 1) % THEMES.length]);
  return /*#__PURE__*/React.createElement("div", {
    className: "app"
  }, /*#__PURE__*/React.createElement("header", {
    className: "app-header"
  }, /*#__PURE__*/React.createElement("h1", null, /*#__PURE__*/React.createElement("span", {
    className: "accent"
  }, "infsb"), " \xB7 central operator panel"), /*#__PURE__*/React.createElement("span", {
    className: "crumb"
  }, "project \xB7 ", /*#__PURE__*/React.createElement("b", null, "inference-streaming-benchmark")), /*#__PURE__*/React.createElement("div", {
    className: "meta"
  }, /*#__PURE__*/React.createElement("span", {
    className: "live-host"
  }, /*#__PURE__*/React.createElement("span", {
    className: "dot dot-live",
    style: {
      marginRight: 6
    }
  }), "server ", /*#__PURE__*/React.createElement("b", null, meta.serverHost)), /*#__PURE__*/React.createElement("span", null, "uptime ", meta.uptime), /*#__PURE__*/React.createElement("button", {
    className: "theme-toggle",
    onClick: cycleTheme,
    title: `theme: ${theme}`
  }, /*#__PURE__*/React.createElement("span", {
    className: "mono small"
  }, theme)))), /*#__PURE__*/React.createElement(ControlRail, {
    transports: transports,
    activeTransport: meta.activeTransport,
    batching: batching,
    inference: inference,
    railStatus: meta.railStatus
  }), /*#__PURE__*/React.createElement(Hero, {
    aggregate: aggregate,
    history: history,
    clientCount: clients.length,
    activeTransport: meta.activeTransport || "—",
    sweepProgress: {
      completed: sweep.completed,
      total: sweep.total
    }
  }), clients.length > 0 && /*#__PURE__*/React.createElement(TransportComparison, {
    clients: clients
  }), /*#__PURE__*/React.createElement(RunForm, {
    transports: transports,
    sweep: sweep
  }), sweep.total > 0 && /*#__PURE__*/React.createElement(SweepPanel, {
    sweep: sweep
  }), /*#__PURE__*/React.createElement(ClientGrid, {
    clients: clients,
    sortBy: sortBy,
    onSortBy: setSortBy
  }));
};
ReactDOM.createRoot(document.getElementById("root")).render(/*#__PURE__*/React.createElement(App, null));