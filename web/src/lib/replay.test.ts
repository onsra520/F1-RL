import { describe, expect, it } from 'vitest';
import { parseReplay, replayAt, replayModelId } from './replay';
import { DEFAULT_AUTO_CONFIG } from './autoDriveConfig';

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

  it('accepts authoritative Auto Drive replay provenance', () => {
    const hash = 'a'.repeat(64);
    const replay = parseReplay({
      schema: 'f1rl_racing_sandbox_replay_v2', label: 'Auto Drive · custom-track · seed 412',
      provenance: { dataset_split: 'DEVELOPMENT', runtime_mode: 'development', source_sha256: hash,
        model_id: 'auto-drive', model_sha256: null, track_id: 'custom-track', track_sha256: hash, seed: 412 },
      result: { status: 'finished', valid_sample: true, successful_finish: true, reason: null,
        steps: 1, duration_s: 0.1, trace_sha256: hash },
      frames: [
        { time_s: 0, position: [0, 0, 0], heading_rad: 0, speed_mps: 0 },
        { time_s: 0.1, position: [0, 0, 1], heading_rad: 0, speed_mps: 10 },
      ],
    });

    expect(replayModelId(replay)).toBe('auto-drive');
    expect(replayAt(replay, 0.05).position[2]).toBe(0.5);
  });

  it('accepts backend Auto Drive configuration with a fractional speed', () => {
    const hash = 'a'.repeat(64);
    const replay = parseReplay({ schema: 'f1rl_racing_sandbox_replay_v2', label: 'auto',
      provenance: { dataset_split: 'DEVELOPMENT', runtime_mode: 'development', source_sha256: hash,
        model_id: 'auto-drive', model_sha256: null, track_id: 'track', track_sha256: hash, seed: 1 },
      result: { status: 'finished', valid_sample: false, successful_finish: true, reason: null,
        steps: 1, duration_s: 2, trace_sha256: hash }, frames: fixture.frames,
      auto_drive_config: { ...DEFAULT_AUTO_CONFIG, target_max_speed_kmh: 0.5 } });
    expect(replay.schema === 'f1rl_racing_sandbox_replay_v2' && replay.auto_drive_config?.target_max_speed_kmh).toBe(0.5);
  });

  it('rejects malformed and zero-length orientations at parsing', () => {
    for (const orientation of [[0, 0, 0, 0], [NaN, 0, 0, 1], [Infinity, 0, 0, 1], [1, 0, 0], [ , 0, 0, 1], 'invalid']) {
      expect(() => parseReplay({ ...fixture, frames: fixture.frames.map(f => ({ ...f, orientation_wxyz: orientation })) })).toThrow(/orientation/i);
    }
  });

  it('normalizes valid orientations before interpolation', () => {
    const replay = parseReplay({ ...fixture, frames: fixture.frames.map((f, i) => ({ ...f, orientation_wxyz: i ? [0, 0, 2, 0] : [2, 0, 0, 0] })) });
    expect(replay.frames[0].orientation_wxyz).toEqual([1, 0, 0, 0]);
    expect(Math.hypot(...replayAt(replay, 1).orientation_wxyz!)).toBeCloseTo(1);
    for (const value of [1e-200, 1e308]) {
      const scaled = parseReplay({ ...fixture, frames: fixture.frames.map(f => ({ ...f, orientation_wxyz: [value, value, 0, 0] })) });
      expect(Math.hypot(...scaled.frames[0].orientation_wxyz!)).toBeCloseTo(1);
    }
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
      track_preview: {
        track_id: 'straight-sprint-v1', track_sha256: hash,
        preview: { schema: 'f1rl_racing_sandbox_draft_v1', id: 'straight-sprint-v1', name: 'Episode track', width: 12,
          future_preview_field: { retained: true },
          points: [{ x: 0, z: 0, future_point_field: 'retained' }, { x: 0, z: 20 }],
          props: [
            { id: 'ramp', kind: 'ramp', t: 0.25, offset: 0, rotation: 0, future_prop_field: 'retained' },
            { id: 'finish-gate', kind: 'checkpoint', t: 0.5, offset: 0, rotation: 0 },
          ] },
        future_preview_metadata: { retained: true },
        road_samples: [
          { x: 0, y: 0, z: 0, t: 0, heading: 0, nx: 1, nz: 0 },
          { x: 0, y: 0, z: 20, t: 1, heading: 0, nx: 1, nz: 0 },
        ],
      },
      frames: [runtimeFrame(0, 0), runtimeFrame(1, 0.1)],
    });
    expect(replay.schema).toBe('f1rl_racing_sandbox_replay_v2');
    expect(replayModelId(replay)).toBe('A-seed0');
    expect(replay.frames[1].action?.throttle).toBe(1);
    if (replay.schema !== 'f1rl_racing_sandbox_replay_v2') throw new Error('expected runtime replay v2');
    expect(replay.track_preview?.preview.props.map((prop) => prop.kind)).toEqual(['ramp', 'checkpoint']);
    expect(replay.track_preview?.road_samples[1]?.z).toBe(20);
    expect(replay.track_preview?.preview).toMatchObject({ future_preview_field: { retained: true } });
    expect(replay.track_preview?.preview.points[0]).toMatchObject({ future_point_field: 'retained' });
    expect(replay.track_preview?.preview.props[0]).toMatchObject({ future_prop_field: 'retained' });
    expect(replay.track_preview).toMatchObject({ future_preview_metadata: { retained: true } });
    expect(replayAt(replay, 0.05).position[2]).toBeCloseTo(0.5);
    const heldout = JSON.parse(JSON.stringify(replay)) as { provenance: Record<string, unknown> };
    heldout.provenance.dataset_split = 'HELD-OUT';
    expect(() => parseReplay(heldout)).toThrow();
    const wrongTrack = JSON.parse(JSON.stringify(replay)) as { track_preview: { track_sha256: string } };
    wrongTrack.track_preview.track_sha256 = 'b'.repeat(64);
    expect(() => parseReplay(wrongTrack)).toThrow(/does not match replay provenance/);
  });
});


describe('Physics contact metadata in replay interpolation', () => {
  it('never reports airborne before the certified incoming Physics frame', () => {
    const r = parseReplay({...fixture,frames:[
      {time_s:0,position:[0,.5,0],heading_rad:0,speed_mps:0,grounded:true,physical_event_types:[]},
      {time_s:.1,position:[0,.8,0],heading_rad:0,speed_mps:2,grounded:false,physical_event_types:['takeoff']},
      {time_s:.2,position:[0,.5,0],heading_rad:0,speed_mps:0,grounded:true,physical_event_types:['landing']},
    ]});
    expect(replayAt(r,.05).grounded).toBe(true);
    expect(replayAt(r,.05).physical_event_types).toEqual([]);
    expect(replayAt(r,.1).grounded).toBe(false);
    expect(replayAt(r,.15).grounded).toBe(false);
    expect(replayAt(r,.15).physical_event_types).toEqual(['takeoff']);
    expect(replayAt(r,.2).grounded).toBe(true);
  });
});

describe('rounded replay timeline vs authoritative floating-point event tick', () => {
  it('shows a certified landing at slider 0.35 for a frame time 0.35000000000000003', () => {
    const r = parseReplay({...fixture,frames:[
      {time_s:0.34,position:[0,.61,0],heading_rad:0,speed_mps:1,grounded:false,physical_event_types:[]},
      {time_s:0.35000000000000003,position:[0,.58,0],heading_rad:0,speed_mps:1,grounded:true,physical_event_types:['landing']},
      {time_s:0.36,position:[0,.55,0],heading_rad:0,speed_mps:1,grounded:true,physical_event_types:[]},
    ]});
    expect(replayAt(r,0.3499).grounded).toBe(false);
    expect(replayAt(r,0.35).grounded).toBe(true);
    expect(replayAt(r,0.35).physical_event_types).toEqual(['landing']);
  });
});
