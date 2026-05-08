import { describe, it, expect } from "vitest";
import { buildSweepBody } from "../../inference_streaming_benchmark/server_static/data.js";

const baseState = (overrides = {}) => ({
  selectedTransports: ["grpc", "websocket"],
  batchOff: true,
  batchOn: true,
  batchSizes: "2,4,8",
  batchWaits: "0,5,10",
  inferSingle: true,
  inferUnsafe: false,
  inferMulti: false,
  inferInstances: "1,2",
  cameraMode: "current",
  warmupS: 2,
  durationS: 10,
  ...overrides,
});

describe("buildSweepBody", () => {
  it("builds a baseline payload with no mock_camera key when cameraMode='current'", () => {
    const body = buildSweepBody(baseState());
    expect(body).toEqual({
      transports: ["grpc", "websocket"],
      batch_modes: ["off", "on"],
      batch_sizes: [2, 4, 8],
      batch_waits_ms: [0, 5, 10],
      inference_modes: ["single"],
      inference_instances: [1, 2],
      warmup_s: 2,
      duration_s: 10,
    });
    expect("mock_camera" in body).toBe(false);
  });

  it("sets mock_camera=true when cameraMode='mock'", () => {
    expect(buildSweepBody(baseState({ cameraMode: "mock" })).mock_camera).toBe(true);
  });

  it("sets mock_camera=false when cameraMode='real'", () => {
    expect(buildSweepBody(baseState({ cameraMode: "real" })).mock_camera).toBe(false);
  });

  it("omits unselected batch_modes", () => {
    expect(buildSweepBody(baseState({ batchOff: false, batchOn: true })).batch_modes).toEqual(["on"]);
    expect(buildSweepBody(baseState({ batchOff: true, batchOn: false })).batch_modes).toEqual(["off"]);
    expect(buildSweepBody(baseState({ batchOff: false, batchOn: false })).batch_modes).toEqual([]);
  });

  it("collects every selected inference mode in canonical order", () => {
    const body = buildSweepBody(baseState({
      inferSingle: true, inferUnsafe: true, inferMulti: true,
    }));
    expect(body.inference_modes).toEqual(["single", "unsafe-multi", "multi-instance"]);
  });

  it("trims whitespace and skips empty entries in numeric lists", () => {
    const body = buildSweepBody(baseState({
      batchSizes: " 2 , , 4, 8 ",
      batchWaits: "0, ,5",
      inferInstances: "1,,2,",
    }));
    expect(body.batch_sizes).toEqual([2, 4, 8]);
    expect(body.batch_waits_ms).toEqual([0, 5]);
    expect(body.inference_instances).toEqual([1, 2]);
  });

  it("parses batch_waits_ms with float precision", () => {
    expect(buildSweepBody(baseState({ batchWaits: "0,2.5,10" })).batch_waits_ms).toEqual([0, 2.5, 10]);
  });

  it("coerces numeric warmup/duration via parseFloat", () => {
    const body = buildSweepBody(baseState({ warmupS: "3.5", durationS: "15" }));
    expect(body.warmup_s).toBe(3.5);
    expect(body.duration_s).toBe(15);
  });

  it("returns empty transports[] when none selected", () => {
    expect(buildSweepBody(baseState({ selectedTransports: [] })).transports).toEqual([]);
  });

  it("handles missing selectedTransports gracefully", () => {
    expect(buildSweepBody(baseState({ selectedTransports: undefined })).transports).toEqual([]);
  });
});
