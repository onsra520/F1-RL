import { describe, expect, it } from 'vitest';
import { detectAutoBridges, bridgeElevationAt } from './elevation';
import { createTrack, getAutoBridges, pointAt, sampleRoad } from './track';

describe('automatic road overpasses', () => {
  it('lifts the later leg of a self-crossing spline at least 6m above the earlier leg', () => {
    const track = {
      ...createTrack(),
      points: [
        { x: -75, z: -42 }, { x: 75, z: 42 }, { x: 75, z: -42 },
        { x: -75, z: 42 },
      ],
      props: [],
    };
    const bridges = getAutoBridges(track);
    expect(bridges.length).toBeGreaterThan(0);
    const road = sampleRoad(track, 400);
    const crossing = bridges[0];
    const lower = pointAt(track, crossing.underT);
    const upper = pointAt(track, crossing.overT);
    expect(Math.hypot(lower.x - upper.x, lower.z - upper.z)).toBeLessThan(0.5);
    expect(upper.y - lower.y).toBeGreaterThanOrEqual(5.9);
    expect(road[0].y).toBeCloseTo(0, 6);
    expect(road[road.length - 1].y).toBeCloseTo(0, 6);
  });

  it('does not invent a bridge for straight or smooth non-crossing road', () => {
    const points = [
      { x: -80, z: 0 }, { x: -20, z: 0 }, { x: 20, z: 0 }, { x: 80, z: 0 },
    ];
    expect(detectAutoBridges(points, 160)).toEqual([]);
  });

  it('provides smooth, nonnegative approach and exit elevation', () => {
    const bridges = [{
      x: 0, z: 0, underT: 0.25, overT: 0.75, rise: 6, plateauMeters: 6, outerMeters: 36,
    }];
    expect(bridgeElevationAt(0.75, bridges, 200)).toBeCloseTo(6);
    expect(bridgeElevationAt(0, bridges, 200)).toBe(0);
    expect(bridgeElevationAt(1, bridges, 200)).toBe(0);
    expect(bridgeElevationAt(0.65, bridges, 200)).toBeGreaterThan(0);
  });
});
