import { describe, expect, it } from 'vitest';
import { createT3V4JumpTrack, createTrack, jumpHeight, nearestRoad, parseTrack, pointAt, roadHeight, roadLength, sampleRoad, visibleRoadSections } from './track';

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
  it('cuts true gaps in asphalt, shoulder, solid sidewalls and rails at exact authored bounds', () => {
    const base = createTrack();
    const straight = { ...base, points: [{x:0,z:0},{x:0,z:86}], props: [
      { id:'jump-1',kind:'gap' as const,t:.52,offset:4,rotation:0 },
    ] };
    const samples = sampleRoad(straight, 86);
    const sections = visibleRoadSections(straight,samples);
    expect(sections).toHaveLength(2);
    const leftLip = sections[0].at(-1)!;
    const rightLip = sections[1][0];
    expect(leftLip.t).toBeCloseTo(.52 - 2/86, 6);
    expect(rightLip.t).toBeCloseTo(.52 + 2/86, 6);
    expect(rightLip.z-leftLip.z).toBeCloseTo(4,3); // visual spline sampling is sub-millimetre
    expect(sections.every(section => section.length > 2)).toBe(true);
    expect(visibleRoadSections({...straight,props:[]},samples)).toEqual([samples]);
    expect(() => parseTrack({...straight,props:[{...straight.props[0],offset:1}]})).toThrow();
    expect(() => parseTrack({...straight,props:[{...straight.props[0],rotation:0.5}]})).toThrow();
  });

  it('mirrors the versioned Python T3 launch and downhill receiving trough exactly', () => {
    const map = createT3V4JumpTrack();
    expect(parseTrack(map)).toEqual(map);
    expect(map.props).toHaveLength(8);
    expect(map.props[0]).toEqual({ id: 'jump-landing-001', kind: 'jump', t: .5, offset: 3.7, rotation: 0 });
    expect(map.props.filter((p) => p.kind === 'booster').map((p) => Math.round(p.t * 90)))
      .toEqual([17,21,25,29,33,37,41]);
    for (const [z, height] of [
      [0,0], [27.15,0], [43.15,.8], [45,.6],
      [46.85,.4], [48.35,.4], [52.35,.18],
      [57.35,.18], [73.35,0], [90,0],
    ]) {
      expect(jumpHeight(z,45)).toBeCloseTo(height,8);
      expect(roadHeight(z/90,map.props,[],90)).toBeCloseTo(height,8);
    }
    const sections=visibleRoadSections(map,sampleRoad(map,360));
    expect(sections).toHaveLength(2);
    expect(sections[0].at(-1)!.z).toBeCloseTo(43.15,2);
    expect(sections[1][0].z).toBeCloseTo(46.85,2);
    expect(sections[0].at(-1)!.y).toBeCloseTo(.8,2);
    expect(sections[1][0].y).toBeCloseTo(.4,2);
    expect(() => parseTrack({ ...map, props: [{...map.props[0],offset:4}] })).toThrow();
    expect(() => parseTrack({ ...map, props: [{...map.props[0],rotation:1}] })).toThrow();
  });

  it('rejects bad prop values and duplicate IDs', () => {
    const t = createTrack();
    expect(() => parseTrack({...t,props:[{...t.props[0],t:Infinity}]})).toThrow();
    expect(() => parseTrack({...t,props:[t.props[0],t.props[0]]})).toThrow();
  });
});
