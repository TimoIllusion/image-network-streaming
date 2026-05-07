# Handoff: Operator Console redesign — `inference-streaming-benchmark`

## Overview

This is a redesign of the **Central Operator Panel** for the `inference-streaming-benchmark` project — the live ops UI served by `server.py` at `http://<server>:9000/`. It currently lives at `inference_streaming_benchmark/server_static/{index.html, app.js}`.

The redesign keeps the existing functionality (transport switching, dynamic batching, inference mode, multi-run sweeps, per-client controls, results export) but reimagines the layout as a **data-engineering / observability dashboard**: live aggregate readouts, latency-stage waterfall, head-to-head transport comparison, sweep heatmap, and per-client cards with sparklines.

## About the design files

The files in this folder (`Operator Console.html`, `style.css`, `*.jsx`, `mock-data.js`) are **design references created in HTML** — a high-fidelity prototype showing intended look, layout, and behavior. They are **not production code to ship verbatim**.

The task is to **recreate this design in the existing `server_static/` environment**: replace `index.html` and `app.js` with equivalents that produce the same UI, but driven by the real server endpoints (`/clients`, `/transport`, `/batching`, `/inference`, `/multi-run/...`) instead of the mock data the prototype uses.

The current `server_static/` is plain JS + plain HTML with no build step — keep that constraint. Either ship in-browser Babel (the prototype does this) or pre-compile JSX to JS once with `@babel/cli`. Pick whichever fits the project's existing posture better; there is no bundler today.

## Fidelity

**High-fidelity.** Final colors, typography, spacing, component shapes, and interaction states are all settled. Recreate pixel-perfectly. The only thing to swap is the data source.

## Target environment

- **Framework:** vanilla JS + small JSX components (the prototype uses React 18.3.1 via UMD + Babel standalone). The existing `app.js` is plain JS — you may either continue using vanilla DOM with the same HTML/CSS, or adopt React. The prototype is React-based because the component tree (cards, charts, sweep grid) benefits from it. Recommended: **adopt React** for the new UI; keep it loaded via `<script>` tags from unpkg, no bundler.
- **Static serving:** the existing FastAPI server in `inference_streaming_benchmark/server.py` mounts `server_static/` — extend that mount to serve the new files.
- **Polling cadence:** match the existing 1s poll (`POLL_MS = 1000` in current `app.js`). The prototype redraws sparklines every 1.5s with simulated jitter; production should redraw on each `/clients` poll.

## Layout

The dashboard is a vertical stack within a `max-width: 1480px` centered container, `padding: 24px 28px 80px`.

### 1. App header (top, 1-line)
- Brand `infsb · central operator panel` (mono, 14px, brick-orange `infsb`)
- Crumb: `project · inference-streaming-benchmark · v0.4.2`
- Right-aligned meta row: live host badge with pulsing green dot, uptime, build hash

### 2. Control rail — sticky control card
Two stacked rows, separated by a hairline border:
- **Row 1 — transport selector:** 8 chips, one per transport (`imagezmq, zmq_raw, grpc, websocket_raw, http_multipart_raw, http_multipart, zmq, websocket`). Active chip is filled black with paper-colored text.
- **Row 2 — batching + infer mode + global actions:** toggle switch (`batching on/off`), numeric inputs (`size`, `wait` ms), `infer` select (single / unsafe-multi / multi-instance), conditional `n` input when multi-instance, then ghost buttons (`clear stats`, `copy md`, `export csv`).

### 3. Hero card
Two-column grid (`1fr 1.4fr`):
- **Left — aggregate stats column**
  - Eyebrow: `live · t = 14:23:07Z` with pulsing green dot
  - Headline: 76px mono number = aggregate FPS, with `fps aggregate` label
  - 3-up substats (dashed-border-top divider): `latency p50` / `clients X/Y active` / `sweep N/M runs · ETA`
- **Right — aggregate FPS history**
  - Header: `aggregate fps · last 120s` + y-range
  - Area chart, accent fill at 10% opacity, accent stroke, 3 dashed gridlines
  - X-axis labels: `−120s, −90s, −60s, −30s, now`

Below the two columns, full-width:
- **Latency breakdown** — waterfall stages strip. 6 segments (enc/dec/comms/infer/post/wait), each a column with a top accent bar and a label block (stage name uppercase muted, mono ms value with `ms` suffix). Width of each segment = proportion of mean total latency. Header: `latency breakdown · mean across N active clients` + total ms.

### 4. Transport head-to-head card
Header: `transport · head-to-head` + stage legend (6 colored squares with labels).

Two-column body:
- **Left — total latency** stacked-stage bars, sorted ascending by total. Each row: `transport ×count` label · stacked bar · total ms.
- **Right — throughput** simple horizontal bars showing per-client FPS, accent-colored, sorted same as left.

### 5. Multi-run sweep card
Header: `multi-run sweep` · live status `running · 29/48 · ETA 03:42` (with pulsing dot).

- **Progress strip:** thin track, accent fill, dark dot marker at the leading edge, sub-line `warmup 2s · duration 10s · 60%`.
- **Sweep grid:** rows = transports (4), columns = (batchMode × inferMode) combinations (12). Header row shows column key (`b8 / w10 / m×2` etc.). Cells:
  - `done` → background colored by relative throughput (paper → orange → green-yellow gradient), mono integer FPS centered.
  - `running` → accent-soft background, accent border, blinking `▸▸`.
  - `queued` → dashed border, transparent, muted dot.
- **Legend strip:** `cell · fps · color = relative throughput` + gradient swatch + min→max range + key for `b/w/s/u/m`.

### 6. Connected clients card
Header: `connected clients [N]` + sort chips (`name | fps | latency | transport`).

Grid of client cards, 2 columns by default (3 in compact, 1 in spacious). Each card:
- **Header row:** status dot (green=live, faint=idle) · client name (mono) · `ip:port` muted · transport tag (accent-soft pill) · optional `mock` / `paused` tags · `Xms ago`.
- **Body** — 4-column grid (`110px 110px 1.3fr 110px`):
  1. `fps` label + 17px value + 140×22 sparkline (accent, filled)
  2. `total ms` label + value + sparkline (indigo, no fill)
  3. `latency stages` mini-waterfall (10px tall) + per-stage mono labels (`enc 1.2 / dec 0.0 / comms 3.4 ...`)
  4. Side stats (`frames`, `batch`, `wait`) + 3 xs buttons (`mock`, `pause/start`, `⋯`)

## Components

Each component below maps 1:1 to a JSX file in this folder.

### `charts.jsx`
- **`<Spark>`** — pure SVG sparkline. Props: `data, width, height, color, fill, strokeWidth`. Min/max auto-fit, line + dot at last point, optional area fill at 12%.
- **`<Waterfall>`** — horizontal stacked latency bar with optional per-stage labels.
- **`<StageLegend>`** — flex row of 6 colored swatches with stage names (mono, 11px, muted).
- **`<BarRow>`** — 3-column row (label / track / value) for the throughput list.
- **`<StackedBar>`** — multi-color stacked horizontal bar. Used in transport comparison.
- **`<AreaChart>`** — full-width SVG area chart with dashed gridlines for the hero history.
- **`STAGE_COLORS`** — exported map of stage → oklch color (used in waterfalls + legend).

### `sections.jsx`
- **`<ControlRail>`** — top sticky control card. Props are all controlled by parent state.
- **`<Hero>`** — aggregate readout + history chart + waterfall.
- **`<TransportComparison>`** — groups clients by transport, computes mean stages, renders two columns of bars.
- **`<StatBlock>`** — small reusable label/value/sub block.

### `clients-sweep.jsx`
- **`<ClientCard>`** — one client tile with all metrics + sparklines.
- **`<ClientGrid>`** — grid wrapper + sort chips.
- **`<SweepPanel>`** — progress strip + heatmap grid + legend.

### `app.jsx`
- **`<App>`** — top-level wiring. Owns active transport, batch, infer mode, sort, and tweak state. Polls / re-renders on a setInterval (the prototype uses `tick` for jitter; production should `fetch('/clients')` on the same cadence).

### `tweaks-panel.jsx`
**Discard for production.** This is a design-tool helper (palette / density / hue tweaks) and should not ship.

### `mock-data.js`
**Discard for production.** Replace with real fetch calls. See "Data contract" below.

## Design tokens

All tokens live in `:root` of `style.css`. Reproduce verbatim.

### Color (light / "paper" theme)
| Token | Value | Use |
|---|---|---|
| `--paper` | `oklch(0.985 0.005 95)` | Body background |
| `--paper-2` | `oklch(0.965 0.005 95)` | Sunken card background (client cards) |
| `--surface` | `oklch(1 0 0)` | Card background |
| `--ink` | `oklch(0.20 0.02 250)` | Primary text |
| `--ink-2` | `oklch(0.35 0.015 250)` | Secondary text |
| `--muted` | `oklch(0.55 0.01 250)` | Tertiary / labels |
| `--faint` | `oklch(0.78 0.005 250)` | Quaternary (idle dots, axis ticks) |
| `--border` | `oklch(0.90 0.005 250)` | Hairlines |
| `--border-strong` | `oklch(0.80 0.01 250)` | Hover borders |
| `--track` | `oklch(0.94 0.005 250)` | Bar / progress track |
| `--accent` | `oklch(0.62 0.15 30)` | Brick-orange brand accent |
| `--accent-strong` | `oklch(0.50 0.16 30)` | Hover state |
| `--accent-soft` | `oklch(0.95 0.04 30)` | Tag backgrounds, accent-soft fills |
| `--good` | `oklch(0.65 0.13 150)` | Live status |
| `--warn` | `oklch(0.70 0.15 80)` | Paused / warning |
| `--bad` | `oklch(0.58 0.18 25)` | Errors |

Stage colors (used in waterfall, legend, sweep cells):
| Stage | Color |
|---|---|
| `enc` | `oklch(0.74 0.10 60)` warm yellow |
| `dec` | `oklch(0.70 0.11 40)` amber |
| `comms` | `oklch(0.66 0.13 25)` brick |
| `infer` | `oklch(0.55 0.14 280)` indigo |
| `post` | `oklch(0.65 0.10 200)` teal |
| `wait` | `oklch(0.78 0.02 250)` muted gray-blue |

The prototype also includes `[data-theme="dark"]` and `[data-theme="blueprint"]` overrides — keep them, optional toggle.

### Typography
- **Mono:** `JetBrains Mono` (Google Fonts), weights 400/500/600 — used for *all* numbers, labels, transport names, button text. `font-feature-settings: "tnum"`.
- **Sans:** `Inter`, weights 400–700 — only for prose.
- **Sizes:** 76px (hero number) · 22px (substats) · 17px (client metric values) · 14px (body) · 12.5px (client names) · 12px (h2 small caps) · 11.5px (controls, body mono) · 11px (small) · 10.5px (eyebrows, axis) · 10px (labels). Tabular numerals everywhere.

### Spacing / radius / shadow
- Card padding: `16px 18px` (hero: `20px 22px`)
- Card gap: `16px`
- Card radius: `--radius-lg: 6px`; control / button radius: `--radius: 4px` (chips: 3px)
- Shadow: minimal — `0 1px 0 oklch(0.92 0.005 250)`. The aesthetic is flat with hairline borders.

### Motion
- Hover transitions: `100–150ms` ease
- Progress bar fill: `400ms ease`
- Pulse dot: `1.6s` infinite
- Blink (`▸▸` on running sweep cell): `1s` infinite

## Data contract

The current server already exposes the right endpoints; the new components need data shaped like this. Build an adapter that turns the existing `/clients` response into this shape (or change the endpoint shape — easier).

```ts
type Client = {
  name: string;          // hostname-port
  transport: string;     // one of TRANSPORTS
  ip: string;
  port: number;
  inferenceOn: boolean;
  mockCam: boolean;
  ageMs: number;         // ms since last heartbeat
  frames: number;
  fps: number;
  timing: {
    enc: number; dec: number; comms: number;
    infer: number; post: number; wait: number;
    total: number;       // sum of stages
  };
  fpsSeries: number[];   // ring buffer, ~60 points
  latSeries: number[];   // ring buffer, ~60 points
  batch: number;         // median batch size
};

type Aggregate = {
  totalFps: number;      // sum of active client fps
  avgLat: number;        // mean total across active
  minLat: number; maxLat: number;
  stageAvg: { enc, dec, comms, infer, post, wait };
  activeCount: number; totalCount: number;
};

type SweepRow = {
  id: number;
  transport: string;
  batch: { enabled: boolean; size: number; wait: number };
  infer: { mode: 'single'|'unsafe-multi'|'multi-instance'; instances: number };
  status: 'queued' | 'running' | 'done';
  fps: number | null;
  total_ms: number | null;
  transport_ms: number | null;
  infer_ms: number | null;
  wait_ms: number | null;
  batch_size: number;
};
```

Recommendation: keep the ring buffers (`fpsSeries`, `latSeries`) **on the client side** — append to a fixed-length array each poll. The server only needs to return current `fps` and `timing.total`. Same applies to the hero `history` array.

## Interactions & behavior

| Action | What happens | Endpoint to call |
|---|---|---|
| Click transport chip | Switch all clients to that transport | `POST /transport {name}` (existing) |
| Toggle batching | Apply globally | `POST /batching {enabled, max_batch_size, max_wait_ms}` |
| Change infer mode / instances | Apply globally | `POST /inference {mode, instances}` |
| Click `clear stats` | Reset all client stats | existing endpoint in `app.js` |
| Click client `mock` / `pause` button | Per-client toggle | existing endpoints |
| Sort chip click | Re-sort client grid (client-side) | n/a |
| Sweep cell hover | Tooltip with full config + result | n/a |

Optimistic toggles: keep the `pendingChanges` pattern from the existing `app.js` so user-set values stay visible until the heartbeat catches up. The new UI should respect the same TTL.

## Files in this bundle

| File | Purpose | Ship? |
|---|---|---|
| `Operator Console.html` | Entry point, loads everything | Yes — replaces `server_static/index.html` |
| `style.css` | All styling, design tokens | Yes — replaces inline `<style>` |
| `mock-data.js` | Fake data generator | **No** — replace with real polling |
| `tweaks-panel.jsx` | Design-tool palette/density tweaks | **No** — design only |
| `charts.jsx` | Spark / Waterfall / AreaChart / etc. | Yes |
| `sections.jsx` | ControlRail / Hero / TransportComparison | Yes |
| `clients-sweep.jsx` | ClientCard / ClientGrid / SweepPanel | Yes |
| `app.jsx` | Top-level wiring | Yes — but rewrite the data layer to fetch from real endpoints |

## Implementation steps (recommended order)

1. **Static drop-in.** Copy `Operator Console.html` (rename to `index.html`), `style.css`, and the four shipping `.jsx` files into `inference_streaming_benchmark/server_static/`. Confirm the FastAPI static mount serves them with correct MIME types. The page should render with mock data.
2. **Replace `mock-data.js` with `data.js`.** Write a real polling layer that calls `/clients` and `/multi-run/status` on a 1s interval and exposes `useClients()` / `useSweep()` hooks (or plain pub-sub) returning the shapes above.
3. **Wire control rail.** Replace local state setters in `<App>` with calls to `/transport`, `/batching`, `/inference`. Carry over the `pendingChanges` optimistic-toggle logic from existing `app.js`.
4. **Per-client actions.** Wire `mock` / `pause` buttons in `<ClientCard>` to existing per-client endpoints.
5. **Sparkline ring buffers.** Maintain `fpsSeries` / `latSeries` arrays in client-side state, push on each poll, cap at 60 points.
6. **Drop tweaks-panel** and the EDITMODE comment block in `app.jsx`. Remove the `live`, `density`, `palette` etc. state — pin density to `default` (or expose density via a real settings menu if desired).
7. **Build step (optional).** Pre-compile JSX → JS with `npx babel ...jsx --presets=@babel/preset-react -d static/dist/` and drop the in-browser Babel script tag.
8. **Theme toggle (optional).** Light / dark / blueprint themes are already in `style.css` via `[data-theme]`. Wire to a header button.

## Notes

- The existing `app.js` is ~940 lines of vanilla JS — useful as a reference for endpoint shapes, optimistic-toggle TTL, and the multi-run sweep status loop. Don't port it verbatim; the new component model is cleaner.
- Stage colors and the `enc + dec + comms + infer + post + wait = total` decomposition match what the README documents — preserve the math exactly (don't reorder the stack).
- Design works at 1100px+ width; below that, the hero collapses to a single column. Test responsively.
