import { describe, expect, it } from 'vitest';
import { Vector3 } from 'three';
import { shouldReanchorAfterRenderStall, smoothFollowVector } from './cameraFollow';

describe('camera follow smoothing', () => {
  it('bounds distance travelled by camera during abrupt pose updates', () => {
    const current = new Vector3(0, 10, -20);
    const target = new Vector3(80, 10, -20);
    smoothFollowVector(current, target, 1 / 240);
    expect(current.distanceTo(new Vector3(0, 10, -20))).toBeLessThanOrEqual(20 / 240 + 1e-6);
    expect(current.x).toBeGreaterThan(0);
  });
  it('caps a large stalled frame and converges without overshooting', () => {
    const current = new Vector3();
    const desired = new Vector3(25, 0, 0);
    smoothFollowVector(current, desired, 0.5);
    expect(current.x).toBeLessThanOrEqual(1.01);
    for (let i = 0; i < 2000; i++) smoothFollowVector(current, desired, 1 / 120);
    expect(current.x).toBeCloseTo(25, 4);
  });
  it('is timestep-invariant in unclamped normal motion', () => {
    const target = new Vector3(1, 0, 0);
    const a = new Vector3();
    const b = new Vector3();
    smoothFollowVector(a, target, .02);
    smoothFollowVector(b, target, .01);
    smoothFollowVector(b, target, .01);
    expect(a.x).toBeCloseTo(b.x, 6);
  });
  it('does not mutate targets or teleport on zero or nonfinite delta', () => {
    const current = new Vector3(1, 2, 3);
    const target = new Vector3(100, 200, 300);
    smoothFollowVector(current, target, 0);
    smoothFollowVector(current, target, Number.NaN);
    expect(current.toArray()).toEqual([1, 2, 3]);
    expect(target.toArray()).toEqual([100, 200, 300]);
  });
});

describe('real browser render suspension camera recovery', () => {
  it('never reanchors on ordinary 60–240 Hz animation or jitter', () => {
    for (const gap of [0, 4.2, 8.3, 16.7, 50, 200, 500, 750]) {
      expect(shouldReanchorAfterRenderStall(gap)).toBe(false);
    }
  });
  it('reanchors only after a genuine no-render interval', () => {
    expect(shouldReanchorAfterRenderStall(751)).toBe(true);
    expect(shouldReanchorAfterRenderStall(9823.5)).toBe(true);
    expect(shouldReanchorAfterRenderStall(Number.NaN)).toBe(false);
    expect(shouldReanchorAfterRenderStall(Infinity)).toBe(false);
  });
});
