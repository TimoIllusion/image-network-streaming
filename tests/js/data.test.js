import { describe, it, expect } from "vitest";
import { adaptSweep } from "../../inference_streaming_benchmark/server_static/data.js";

const planEntry = (overrides = {}) => ({
  transport: "grpc",
  batching_enabled: false,
  max_batch_size: 1,
  max_wait_ms: 0,
  inference_mode: "single",
  inference_instances: 1,
  ...overrides,
});

const runEntry = (clients = []) => ({ clients });

describe("adaptSweep", () => {
  it("returns an empty shape when status is null", () => {
    expect(adaptSweep(null)).toEqual({
      rows: [], completed: 0, total: 0, transports: [], running: false, raw: null,
    });
  });

  it("marks the in-flight row as 'running' and later rows as 'queued'", () => {
    const status = {
      running: true,
      plan: [planEntry({ transport: "grpc" }), planEntry({ transport: "websocket" }), planEntry({ transport: "http_multipart" })],
      result: { runs: [runEntry()] },
    };
    const out = adaptSweep(status);
    expect(out.running).toBe(true);
    expect(out.completed).toBe(1);
    expect(out.total).toBe(3);
    expect(out.rows.map((r) => r.status)).toEqual(["done", "running", "queued"]);
    expect(out.transports).toEqual(["grpc", "websocket", "http_multipart"]);
  });

  it("marks all rows 'queued' when nothing has run yet", () => {
    const status = {
      running: false,
      plan: [planEntry(), planEntry()],
      result: { runs: [] },
    };
    expect(adaptSweep(status).rows.map((r) => r.status)).toEqual(["queued", "queued"]);
  });

  it("normalises camera mode: undefined → current, true → mock, false → real", () => {
    const status = {
      plan: [
        planEntry({ mock_camera: undefined }),
        planEntry({ mock_camera: true }),
        planEntry({ mock_camera: false }),
      ],
      result: { runs: [] },
    };
    expect(adaptSweep(status).rows.map((r) => r.camera)).toEqual(["current", "mock", "real"]);
  });

  it("derives transport_ms by subtracting infer + wait from total, clamped at zero", () => {
    const clients = [{
      stats: {
        bench_rows: [{
          Backend: "grpc",
          Frames: "100",
          FPS: "30",
          "total (ms)": "20",
          "infer (ms)": "8",
          "wait (ms)": "5",
          batch: "4",
        }],
      },
    }];
    const status = {
      plan: [planEntry({ transport: "grpc" })],
      result: { runs: [runEntry(clients)] },
    };
    const row = adaptSweep(status).rows[0];
    expect(row.status).toBe("done");
    expect(row.fps).toBe(30);
    expect(row.total_ms).toBe(20);
    expect(row.transport_ms).toBe(7);
    expect(row.infer_ms).toBe(8);
    expect(row.wait_ms).toBe(5);
    expect(row.batch_size).toBe(4);
  });

  it("clamps transport_ms to 0 when infer + wait exceed total (clock skew)", () => {
    const clients = [{
      stats: {
        bench_rows: [{
          Backend: "grpc",
          Frames: "10",
          FPS: "5",
          "total (ms)": "5",
          "infer (ms)": "8",
          "wait (ms)": "2",
          batch: "1",
        }],
      },
    }];
    const status = { plan: [planEntry()], result: { runs: [runEntry(clients)] } };
    expect(adaptSweep(status).rows[0].transport_ms).toBe(0);
  });

  it("leaves done rows with null metrics when the run has no clients", () => {
    const status = { plan: [planEntry()], result: { runs: [runEntry([])] } };
    const row = adaptSweep(status).rows[0];
    expect(row.status).toBe("done");
    expect(row.fps).toBeNull();
    expect(row.total_ms).toBeNull();
  });

  it("dedupes the transports list while preserving order", () => {
    const status = {
      plan: [
        planEntry({ transport: "grpc" }),
        planEntry({ transport: "websocket" }),
        planEntry({ transport: "grpc" }),
      ],
      result: { runs: [] },
    };
    expect(adaptSweep(status).transports).toEqual(["grpc", "websocket"]);
  });
});
