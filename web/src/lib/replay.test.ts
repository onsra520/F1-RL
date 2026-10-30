import { describe, expect, it } from 'vitest';
import { parseReplay, replayAt, replayModelId } from './replay';

const fixture = {
  schema: 'f1rl_racing_sandbox_replay_v1',
  label: 'TRAIN smoke',
  source: { split: 'TRAIN', model_id: 'C-seed0' },
  frames: [
    { time_s: 0, position: [0,0,0], heading_rad: 0, speed_mps: 10 },
    { time_s: 2, position: [0,0,20], heading_rad: Math.PI/2, speed_mps: 20 },
  ],
};
describe('read-only replay', () => {
  it('reads TRAIN frames and interpolates position/time', () => {
    const replay = parseReplay(fixture);
    const state = replayAt(replay,1);
    expect(state.position[2]).toBe(10);
    expect(state.speed_mps).toBe(15);
    expect(state.heading_rad).toBeCloseTo(Math.PI/4);
  });
  it('blocks held-out, missing, duplicated, and invalid timestamps', () => {
    expect(() => parseReplay({...fixture,source:{split:'HELD-OUT',model_id:'x'}})).toThrow();
    expect(() => parseReplay({...fixture,frames:fixture.frames.map(f=>({...f,time_s:0}))})).toThrow();
    expect(() => parseReplay({...fixture,frames:[...fixture.frames,{...fixture.frames[1],time_s:NaN}]})).toThrow();
  });

  it('validates and preserves version 2 authoritative TRAIN telemetry and provenance', () => {
    const hash = 'a'.repeat(64);
    const runtimeFrame = (tick: number, time_s: number) => ({
      schema_version: 'f1rl_runtime_frame_v1', tick, time_s, dt_s: 0.1,
      position: [0, 0, tick], velocity: [0, 0, 10], acceleration_mps2: [0, 0, 0],
      orientation_wxyz: [1, 0, 0, 0], heading_rad: 0, speed_mps: 10, grounded: true,
      action: { steering: 0.1, throttle: 1, brake: 0 }, route_progress_m: tick,
      route_progress_fraction: 0.5, route_highwater_m: tick, next_checkpoint_index: 1,
      checkpoint_count: 2, completed_checkpoint_ids: [], checkpoint_crossings: [],
      collision_count: 0, collision_ids: [], physical_event_types: [], outcome: null,
    });
    const replay = parseReplay({
      schema: 'f1rl_racing_sandbox_replay_v2', label: 'A-seed0 · straight · seed 7',
      provenance: { dataset_split: 'TRAIN', runtime_mode: 'development', source_sha256: hash,
        model_id: 'A-seed0', model_sha256: hash, track_id: 'straight-sprint-v1', track_sha256: hash, seed: 7 },
      result: { status: 'finished', valid_sample: true, successful_finish: true, reason: null,
        steps: 1, duration_s: 0.1, trace_sha256: hash },
      frames: [runtimeFrame(0, 0), runtimeFrame(1, 0.1)],
    });
    expect(replay.schema).toBe('f1rl_racing_sandbox_replay_v2');
    expect(replayModelId(replay)).toBe('A-seed0');
    expect(replay.frames[1].action?.throttle).toBe(1);
    expect(replayAt(replay, 0.05).position[2]).toBeCloseTo(0.5);
    if (replay.schema !== 'f1rl_racing_sandbox_replay_v2') throw new Error('expected runtime replay v2');
    const heldout = JSON.parse(JSON.stringify(replay)) as { provenance: Record<string, unknown> };
    heldout.provenance.dataset_split = 'HELD-OUT';
    expect(() => parseReplay(heldout)).toThrow();
  });
});
