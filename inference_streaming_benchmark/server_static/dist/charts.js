// Tiny chart components — pure SVG, no deps.
// Used by the dashboard for sparklines, waterfall, and bar charts.

const Spark = ({
  data,
  width = 120,
  height = 28,
  color = "currentColor",
  fill = false,
  strokeWidth = 1.25
}) => {
  const plotWidth = Number.isFinite(Number(width)) ? Number(width) : 120;
  const svgWidth = typeof width === "string" ? width : plotWidth;
  if (!data || data.length < 2) {
    return /*#__PURE__*/React.createElement("svg", {
      width: svgWidth,
      height: height,
      viewBox: `0 0 ${plotWidth} ${height}`,
      preserveAspectRatio: "none",
      style: {
        display: "block"
      }
    });
  }
  const min = Math.min(...data);
  const max = Math.max(...data);
  const range = max - min || 1;
  const stepX = plotWidth / (data.length - 1);
  const points = data.map((v, i) => {
    const x = i * stepX;
    const y = height - (v - min) / range * (height - 2) - 1;
    return [x, y];
  });
  const path = "M" + points.map(([x, y]) => `${x.toFixed(2)},${y.toFixed(2)}`).join(" L");
  const fillPath = fill ? `${path} L${plotWidth},${height} L0,${height} Z` : null;
  const last = points[points.length - 1];
  return /*#__PURE__*/React.createElement("svg", {
    width: svgWidth,
    height: height,
    viewBox: `0 0 ${plotWidth} ${height}`,
    preserveAspectRatio: "none",
    style: {
      display: "block",
      overflow: "visible"
    }
  }, fillPath && /*#__PURE__*/React.createElement("path", {
    d: fillPath,
    fill: color,
    opacity: "0.12"
  }), /*#__PURE__*/React.createElement("path", {
    d: path,
    fill: "none",
    stroke: color,
    strokeWidth: strokeWidth,
    strokeLinecap: "round",
    strokeLinejoin: "round"
  }), /*#__PURE__*/React.createElement("circle", {
    cx: last[0],
    cy: last[1],
    r: 1.6,
    fill: color
  }));
};

// Stacked horizontal waterfall: enc | dec | comms | infer | post | wait
// Stage colors share lightness/chroma, vary hue.
const STAGE_COLORS = {
  enc: "oklch(0.74 0.10 60)",
  // warm yellow
  dec: "oklch(0.70 0.11 40)",
  // amber
  comms: "oklch(0.66 0.13 25)",
  // brick
  infer: "oklch(0.55 0.14 280)",
  // indigo
  post: "oklch(0.65 0.10 200)",
  // teal
  wait: "oklch(0.78 0.02 250)" // muted gray-blue
};
const STAGE_ORDER = ["enc", "dec", "comms", "infer", "post", "wait"];
const STAGE_LABELS = {
  enc: "enc",
  dec: "dec",
  comms: "comms",
  infer: "infer",
  post: "post",
  wait: "wait"
};
const Waterfall = ({
  timing,
  max,
  height = 12,
  showLabels = false,
  compact = false
}) => {
  const total = STAGE_ORDER.reduce((a, k) => a + (timing[k] || 0), 0);
  const scaleMax = max || total;
  let cursor = 0;
  return /*#__PURE__*/React.createElement("div", {
    style: {
      width: "100%"
    }
  }, /*#__PURE__*/React.createElement("div", {
    style: {
      display: "flex",
      width: "100%",
      height,
      background: "var(--track)",
      borderRadius: 2,
      overflow: "hidden",
      border: "1px solid var(--border)"
    }
  }, STAGE_ORDER.map(k => {
    const v = timing[k] || 0;
    const pct = v / scaleMax * 100;
    if (pct < 0.01) return null;
    const seg = /*#__PURE__*/React.createElement("div", {
      key: k,
      title: `${k}: ${v.toFixed(2)}ms`,
      style: {
        width: `${pct}%`,
        background: STAGE_COLORS[k],
        height: "100%"
      }
    });
    cursor += v;
    return seg;
  }), total < scaleMax && /*#__PURE__*/React.createElement("div", {
    style: {
      flex: 1,
      height: "100%"
    }
  })), showLabels && !compact && /*#__PURE__*/React.createElement("div", {
    style: {
      display: "flex",
      marginTop: 4,
      fontSize: 10,
      fontFamily: "var(--mono)",
      color: "var(--muted)",
      letterSpacing: "0.04em"
    }
  }, STAGE_ORDER.map(k => {
    const v = timing[k] || 0;
    const pct = v / scaleMax * 100;
    if (pct < 4) return null;
    return /*#__PURE__*/React.createElement("div", {
      key: k,
      style: {
        width: `${pct}%`,
        paddingLeft: 4
      }
    }, /*#__PURE__*/React.createElement("span", {
      style: {
        color: STAGE_COLORS[k]
      }
    }, "\u25A0"), " ", STAGE_LABELS[k], " ", v.toFixed(1));
  })));
};
const StageLegend = () => /*#__PURE__*/React.createElement("div", {
  style: {
    display: "flex",
    gap: 14,
    fontFamily: "var(--mono)",
    fontSize: 11,
    color: "var(--muted)",
    flexWrap: "wrap",
    letterSpacing: "0.02em"
  }
}, STAGE_ORDER.map(k => /*#__PURE__*/React.createElement("span", {
  key: k,
  style: {
    display: "inline-flex",
    alignItems: "center",
    gap: 6
  }
}, /*#__PURE__*/React.createElement("span", {
  style: {
    width: 10,
    height: 10,
    background: STAGE_COLORS[k],
    borderRadius: 2,
    display: "inline-block"
  }
}), STAGE_LABELS[k])));

// Horizontal grouped bar — for transport head-to-head comparison
const BarRow = ({
  label,
  value,
  max,
  color = "var(--accent)",
  width = 200,
  suffix = ""
}) => {
  const pct = max ? value / max * 100 : 0;
  return /*#__PURE__*/React.createElement("div", {
    style: {
      display: "grid",
      gridTemplateColumns: "140px 1fr 70px",
      alignItems: "center",
      gap: 10,
      fontSize: 11.5,
      fontFamily: "var(--mono)"
    }
  }, /*#__PURE__*/React.createElement("span", {
    style: {
      color: "var(--ink)",
      overflow: "hidden",
      textOverflow: "ellipsis",
      whiteSpace: "nowrap"
    }
  }, label), /*#__PURE__*/React.createElement("div", {
    style: {
      height: 8,
      background: "var(--track)",
      borderRadius: 1,
      overflow: "hidden",
      position: "relative"
    }
  }, /*#__PURE__*/React.createElement("div", {
    style: {
      width: `${pct}%`,
      height: "100%",
      background: color,
      transition: "width 400ms ease"
    }
  })), /*#__PURE__*/React.createElement("span", {
    style: {
      textAlign: "right",
      color: "var(--ink)"
    }
  }, value.toFixed(value > 100 ? 0 : 1), /*#__PURE__*/React.createElement("span", {
    style: {
      color: "var(--muted)",
      marginLeft: 2
    }
  }, suffix)));
};

// Multi-stage bar — used in transport comparison to show stage breakdown bars
const StackedBar = ({
  timing,
  max,
  height = 18
}) => {
  return /*#__PURE__*/React.createElement("div", {
    style: {
      width: "100%",
      height,
      display: "flex",
      background: "var(--track)",
      borderRadius: 1,
      overflow: "hidden",
      border: "1px solid var(--border)"
    }
  }, STAGE_ORDER.map(k => {
    const v = timing[k] || 0;
    const pct = v / max * 100;
    if (pct < 0.05) return null;
    return /*#__PURE__*/React.createElement("div", {
      key: k,
      title: `${k}: ${v.toFixed(2)}ms`,
      style: {
        width: `${pct}%`,
        background: STAGE_COLORS[k],
        height: "100%"
      }
    });
  }));
};

// Big aggregate FPS chart
const AreaChart = ({
  data,
  width,
  height,
  color = "var(--accent)"
}) => {
  if (!data || data.length < 2) {
    return /*#__PURE__*/React.createElement("svg", {
      width: "100%",
      viewBox: `0 0 ${width} ${height}`,
      preserveAspectRatio: "none",
      style: {
        display: "block"
      }
    }, /*#__PURE__*/React.createElement("line", {
      x1: 0,
      x2: width,
      y1: height / 2,
      y2: height / 2,
      stroke: "var(--border)",
      strokeDasharray: "2 4"
    }));
  }
  const min = Math.min(...data);
  const max = Math.max(...data);
  const padTop = 8;
  const range = max - min || 1;
  const stepX = width / (data.length - 1);
  const points = data.map((v, i) => {
    const x = i * stepX;
    const y = padTop + (height - padTop - 4) - (v - min) / range * (height - padTop - 4);
    return [x, y];
  });
  const path = "M" + points.map(([x, y]) => `${x.toFixed(1)},${y.toFixed(1)}`).join(" L");
  const fillPath = `${path} L${width},${height} L0,${height} Z`;
  // grid lines
  const gridY = [0.25, 0.5, 0.75].map(p => padTop + (height - padTop - 4) * p);
  return /*#__PURE__*/React.createElement("svg", {
    width: "100%",
    viewBox: `0 0 ${width} ${height}`,
    preserveAspectRatio: "none",
    style: {
      display: "block"
    }
  }, gridY.map((y, i) => /*#__PURE__*/React.createElement("line", {
    key: i,
    x1: 0,
    x2: width,
    y1: y,
    y2: y,
    stroke: "var(--border)",
    strokeDasharray: "2 4"
  })), /*#__PURE__*/React.createElement("path", {
    d: fillPath,
    fill: color,
    opacity: "0.10"
  }), /*#__PURE__*/React.createElement("path", {
    d: path,
    fill: "none",
    stroke: color,
    strokeWidth: 1.25,
    strokeLinecap: "round",
    strokeLinejoin: "round"
  }));
};
Object.assign(window, {
  Spark,
  Waterfall,
  StageLegend,
  BarRow,
  StackedBar,
  AreaChart,
  STAGE_COLORS,
  STAGE_ORDER,
  STAGE_LABELS
});