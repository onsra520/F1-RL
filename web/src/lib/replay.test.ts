import { describe, expect, it } from 'vitest';
import { parseReplay, replayAt } from './replay';

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
});
