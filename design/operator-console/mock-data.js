// Mock data generator — runs in window scope, not Babel.
// Provides the live, jittered numbers the dashboard renders.

(function () {
  const TRANSPORTS = [
    "imagezmq", "zmq_raw", "grpc", "websocket_raw",
    "http_multipart_raw", "http_multipart", "zmq", "websocket",
  ];

  // Baseline timing (ms) per transport — pulled from the README table.
  const BASELINE = {
    imagezmq:           { enc: 0.0, dec: 0.0, comms: 3.1, infer: 26.0, post: 1.3, fps: 33.0 },
    zmq_raw:            { enc: 0.2, dec: 0.0, comms: 3.6, infer: 26.1, post: 1.2, fps: 31.8 },
    grpc:               { enc: 0.2, dec: 0.2, comms: 6.6, infer: 22.4, post: 0.9, fps: 29.4 },
    websocket_raw:      { enc: 0.3, dec: 0.0, comms: 7.2, infer: 25.0, post: 1.3, fps: 29.1 },
    http_multipart_raw: { enc: 0.2, dec: 0.0, comms: 7.6, infer: 24.0, post: 1.2, fps: 29.3 },
    http_multipart:     { enc: 5.6, dec: 8.6, comms: 3.8, infer: 23.9, post: 1.1, fps: 22.4 },
    zmq:                { enc: 6.6, dec: 11.0, comms: 0.9, infer: 24.3, post: 1.4, fps: 22.5 },
    websocket:          { enc: 6.6, dec: 11.0, comms: 1.0, infer: 23.9, post: 1.3, fps: 22.6 },
  };

  // Simple seeded PRNG so renders are stable across reloads if seed fixed.
  function mulberry32(seed) {
    return function () {
      seed |= 0; seed = (seed + 0x6d2b79f5) | 0;
      let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  const HOSTS = [
    "rpi-edge-01", "rpi-edge-02", "rpi-edge-03", "rpi-edge-04",
    "jetson-nano-a", "jetson-nano-b",
    "macbook-m2-pro", "macbook-air-m1",
    "workstation-3090", "workstation-4090",
    "thinkpad-x1", "synology-ds923",
  ];

  function jitter(rng, base, pct) {
    return base * (1 + (rng() - 0.5) * 2 * pct);
  }

  // Build the static "snapshot" payload the dashboard renders.
  function buildClients(seed = 7) {
    const rng = mulberry32(seed);
    const clients = HOSTS.map((host, i) => {
      const transport = TRANSPORTS[i % TRANSPORTS.length];
      const b = BASELINE[transport];
      const enc = jitter(rng, b.enc, 0.25);
      const dec = jitter(rng, b.dec, 0.25);
      const comms = jitter(rng, b.comms, 0.25);
      const infer = jitter(rng, b.infer, 0.12);
      const post = jitter(rng, b.post, 0.30);
      const wait = transport.includes("zmq") ? 0 : jitter(rng, 1.2, 0.6);
      const total = enc + dec + comms + infer + post + wait;
      const fps = jitter(rng, b.fps, 0.08);
      const frames = Math.floor(jitter(rng, 320, 0.4));
      const inferenceOn = rng() > 0.15;
      const mockCam = rng() > 0.55;

      // Sparkline series: 60 points of FPS jitter
      const fpsSeries = [];
      const latSeries = [];
      let lastFps = fps;
      let lastLat = total;
      for (let k = 0; k < 60; k++) {
        lastFps = lastFps + (fps - lastFps) * 0.15 + (rng() - 0.5) * 2;
        lastLat = lastLat + (total - lastLat) * 0.15 + (rng() - 0.5) * 1.5;
        fpsSeries.push(Math.max(0, lastFps));
        latSeries.push(Math.max(0, lastLat));
      }

      return {
        name: host,
        transport,
        port: 8501 + i,
        ip: `10.0.${1 + Math.floor(i / 4)}.${10 + i}`,
        inferenceOn,
        mockCam,
        ageMs: Math.floor(rng() * 800),
        frames,
        fps,
        timing: { enc, dec, comms, infer, post, wait, total },
        fpsSeries,
        latSeries,
        batch: transport.includes("zmq") ? 1 : 1 + Math.floor(rng() * 6),
      };
    });
    return clients;
  }

  // Sweep results — partial progress, mid-run.
  function buildSweepResults(seed = 13) {
    const rng = mulberry32(seed);
    const transports = ["http_multipart_raw", "grpc", "websocket_raw", "imagezmq"];
    const batchModes = [
      { enabled: false, size: 1, wait: 0 },
      { enabled: true, size: 4, wait: 5 },
      { enabled: true, size: 8, wait: 10 },
      { enabled: true, size: 8, wait: 20 },
    ];
    const inferModes = [
      { mode: "single", instances: 1 },
      { mode: "unsafe-multi", instances: 1 },
      { mode: "multi-instance", instances: 2 },
    ];

    const total = transports.length * batchModes.length * inferModes.length;
    const completed = Math.floor(total * 0.62);
    const rows = [];
    let idx = 0;
    for (const t of transports) {
      for (const bm of batchModes) {
        for (const im of inferModes) {
          const done = idx < completed;
          const inProgress = idx === completed;
          const b = BASELINE[t];
          const inflate = bm.enabled ? 1 + bm.wait * 0.05 : 1;
          const inferDelta = im.mode === "multi-instance" ? 0.92 : im.mode === "unsafe-multi" ? 0.96 : 1;
          const total_ms = (b.enc + b.dec + b.comms + b.infer * inferDelta + b.post) * inflate;
          rows.push({
            id: idx,
            transport: t,
            batch: bm,
            infer: im,
            status: done ? "done" : inProgress ? "running" : "queued",
            fps: done ? jitter(rng, b.fps / inflate, 0.05) : null,
            total_ms: done ? jitter(rng, total_ms, 0.06) : null,
            transport_ms: done ? jitter(rng, b.enc + b.dec + b.comms, 0.08) : null,
            infer_ms: done ? jitter(rng, b.infer * inferDelta, 0.06) : null,
            wait_ms: done ? (bm.enabled ? jitter(rng, bm.wait * 0.6, 0.3) : 0) : null,
            batch_size: done && bm.enabled ? jitter(rng, bm.size * 0.85, 0.1) : 1,
          });
          idx++;
        }
      }
    }
    return { rows, total, completed, transports, batchModes, inferModes };
  }

  // For aggregate hero stats
  function buildAggregate(clients) {
    const totalFps = clients.reduce((a, c) => a + (c.inferenceOn ? c.fps : 0), 0);
    const active = clients.filter(c => c.inferenceOn);
    const avgLat = active.length ? active.reduce((a, c) => a + c.timing.total, 0) / active.length : 0;
    const minLat = Math.min(...active.map(c => c.timing.total));
    const maxLat = Math.max(...active.map(c => c.timing.total));
    // Stage averages across active
    const stages = ["enc", "dec", "comms", "infer", "post", "wait"];
    const stageAvg = {};
    for (const s of stages) {
      stageAvg[s] = active.length ? active.reduce((a, c) => a + c.timing[s], 0) / active.length : 0;
    }
    return { totalFps, avgLat, minLat, maxLat, stageAvg, activeCount: active.length, totalCount: clients.length };
  }

  // 60-second history of aggregate FPS for the hero chart
  function buildAggregateHistory(seed = 21) {
    const rng = mulberry32(seed);
    const points = 120;
    const series = [];
    let v = 280;
    for (let i = 0; i < points; i++) {
      v = v + (300 - v) * 0.05 + (rng() - 0.5) * 18;
      series.push(Math.max(0, v));
    }
    return series;
  }

  window.MOCK = {
    TRANSPORTS,
    BASELINE,
    buildClients,
    buildSweepResults,
    buildAggregate,
    buildAggregateHistory,
    mulberry32,
  };
})();
