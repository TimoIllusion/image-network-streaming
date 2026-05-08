import { describe, it, expect } from "vitest";
import { aggregateClientsByTransport } from "../../inference_streaming_benchmark/server_static/data.js";

const benchRow = (overrides = {}) => ({
  Backend: "grpc",
  Frames: "100",
  FPS: "30",
  "enc (ms)": "1",
  "dec (ms)": "1",
  "comms (ms)": "5",
  "infer (ms)": "10",
  "post (ms)": "1",
  "wait (ms)": "0",
  "total (ms)": "18",
  batch: "1",
  ...overrides,
});

const client = (name, rows) => ({ name, stats: { bench_rows: rows } });

describe("aggregateClientsByTransport", () => {
  it("returns empty array when there are no clients", () => {
    expect(aggregateClientsByTransport([])).toEqual([]);
    expect(aggregateClientsByTransport(undefined)).toEqual([]);
  });

  it("retains rows for transports the client previously used (the head-to-head bug fix)", () => {
    const clients = [client("c1", [
      benchRow({ Backend: "grpc", FPS: "30", "total (ms)": "18" }),
      benchRow({ Backend: "websocket", FPS: "20", "total (ms)": "30" }),
    ])];
    const rows = aggregateClientsByTransport(clients);
    expect(rows.map((r) => r.transport).sort()).toEqual(["grpc", "websocket"]);
  });

  it("sorts rows by ascending total latency", () => {
    const clients = [client("c1", [
      benchRow({ Backend: "slow", "total (ms)": "50" }),
      benchRow({ Backend: "fast", "total (ms)": "10" }),
      benchRow({ Backend: "mid", "total (ms)": "25" }),
    ])];
    expect(aggregateClientsByTransport(clients).map((r) => r.transport)).toEqual(["fast", "mid", "slow"]);
  });

  it("sums FPS across clients on the same transport", () => {
    const clients = [
      client("c1", [benchRow({ FPS: "30" })]),
      client("c2", [benchRow({ FPS: "25" })]),
    ];
    expect(aggregateClientsByTransport(clients)[0].fps).toBe(55);
  });

  it("counts unique clients per transport, not row count", () => {
    const clients = [
      client("c1", [benchRow({ Backend: "grpc" }), benchRow({ Backend: "grpc", "Batch config": "alt" })]),
      client("c2", [benchRow({ Backend: "grpc" })]),
    ];
    expect(aggregateClientsByTransport(clients)[0].count).toBe(2);
  });

  it("weights stage timings by Frames", () => {
    const clients = [
      client("c1", [benchRow({ Frames: "100", "infer (ms)": "10" })]),
      client("c2", [benchRow({ Frames: "300", "infer (ms)": "20" })]),
    ];
    expect(aggregateClientsByTransport(clients)[0].timing.infer).toBe(17.5);
  });

  it("returns full stage breakdown per transport", () => {
    const rows = aggregateClientsByTransport([client("c1", [benchRow()])]);
    expect(Object.keys(rows[0].timing).sort()).toEqual(["comms", "dec", "enc", "infer", "post", "wait"]);
  });

  it("filters out transports whose total latency is zero (no usable samples yet)", () => {
    const clients = [client("c1", [
      benchRow({ Backend: "grpc", "total (ms)": "18" }),
      benchRow({ Backend: "empty", Frames: "0", "total (ms)": "0" }),
    ])];
    expect(aggregateClientsByTransport(clients).map((r) => r.transport)).toEqual(["grpc"]);
  });

  it("ignores rows without a Backend key", () => {
    const clients = [client("c1", [{ Frames: "10", FPS: "5", "total (ms)": "100" }])];
    expect(aggregateClientsByTransport(clients)).toEqual([]);
  });
});
