import { describe, expect, it } from 'vitest';
import { createTrack, nearestRoad, parseTrack, pointAt, roadHeight, roadLength, sampleRoad } from './track';

describe('racing sandbox track draft', () => {
  it('creates editable, nonempty 3D road centerline', () => {
    const track = createTrack();
    const samples = sampleRoad(track);
    expect(samples).toHaveLength(241);
    expect(roadLength(samples)).toBeGreaterThan(100);
    expect(samples.every((p) => Number.isFinite(p.x + p.y + p.z))).toBe(true);
  });
  it('snaps props to the route with signed lateral offset', () => {
    const track = createTrack();
    const pt = pointAt(track, 0.43, 3);
    const snap = nearestRoad(track, pt.x, pt.z);
    expect(snap.t).toBeCloseTo(0.43, 2);
    expect(snap.offset).toBeCloseTo(3, 1);
  });
  it('creates a smooth bridge bump and returns to ground', () => {
    const track = createTrack();
    const props = [...track.props, { id: 'b', kind: 'bridge' as const, t: 0.5, offset: 0, rotation: 0 }];
    expect(roadHeight(0.5, props)).toBe(5);
    expect(roadHeight(0.35, props)).toBe(0);
    expect(roadHeight(0.6, props)).toBeGreaterThan(0);
  });
  it('validates JSON round trip without accepting arbitrary schema', () => {
    const t = createTrack();
    expect(parseTrack(JSON.parse(JSON.stringify(t)))).toEqual(t);
    expect(() => parseTrack({ ...t, schema: 'other' })).toThrow();
    expect(() => parseTrack({ ...t, points: [{x:0,z:0}] })).toThrow();
  });
  it('rejects bad prop values and duplicate IDs', () => {
    const t = createTrack();
    expect(() => parseTrack({...t,props:[{...t.props[0],t:Infinity}]})).toThrow();
    expect(() => parseTrack({...t,props:[t.props[0],t.props[0]]})).toThrow();
  });
});
