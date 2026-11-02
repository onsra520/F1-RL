import { describe, expect, it } from 'vitest';
import { isLivePoseFrameInOrder, isRuntimePoseTerminal, LivePoseInterpolator, type LivePoseFrame } from './livePose';

function frame(
  tick: number,
  time_s: number,
  x: number,
  heading_rad = 0,
): LivePoseFrame {
  return {
    tick, time_s, dt_s: 0.1, position: [x, 0, 0], heading_rad, speed_mps: Math.abs(x),
  };
}

describe('LivePoseInterpolator', () => {
  it('rejects queued old frames during reset until the new tick zero arrives', () => {
    expect(isLivePoseFrameInOrder(null, { tick: 8, time_s: 0.8 }, true)).toBe(false);
    expect(isLivePoseFrameInOrder(null, { tick: 0, time_s: 0 }, true)).toBe(true);
    expect(isLivePoseFrameInOrder({ tick: 4, time_s: 0.4 }, { tick: 5, time_s: 0.5 })).toBe(true);
    expect(isLivePoseFrameInOrder({ tick: 4, time_s: 0.4 }, { tick: 3, time_s: 0.3 })).toBe(false);
  });

  it('interpolates position, heading, and speed between received frames', () => {
    const poses = new LivePoseInterpolator();
    poses.push(frame(1, 0, 0), 1000);
    poses.push(frame(2, 0.1, 10, 1), 1100);

    expect(poses.sample(1150)).toEqual({ x: 5, y: 0, z: 0, heading: 0.5, speed: 5 });
  });

  it('takes the shortest heading path across the pi boundary', () => {
    const poses = new LivePoseInterpolator();
    poses.push(frame(1, 0, 0, Math.PI - 0.1), 1000);
    poses.push(frame(2, 0.1, 1, -Math.PI + 0.1), 1100);

    expect(Math.abs(Math.abs(poses.sample(1150)!.heading) - Math.PI)).toBeLessThan(1e-9);
    expect(poses.sample(1200)?.heading).toBe(-Math.PI + 0.1);
  });

  it('bridges a long frame gap without teleportation and never extrapolates', () => {
    const poses = new LivePoseInterpolator();
    poses.push(frame(1, 0, 0), 1000);
    poses.push(frame(2, 0.1, 1), 1100);
    poses.push(frame(3, 0.2, 9), 1500);

    expect(poses.sample(1500)).toEqual({ x: 1, y: 0, z: 0, heading: 0, speed: 1 });
    expect(poses.sample(1550)?.x).toBeGreaterThan(1);
    expect(poses.sample(1550)?.x).toBeLessThan(9);
    expect(poses.sample(10000)).toEqual({ x: 9, y: 0, z: 0, heading: 0, speed: 9 });
  });

  it('ignores duplicate and out-of-order frames', () => {
    const poses = new LivePoseInterpolator();
    poses.push(frame(1, 0, 0), 1000);
    poses.push(frame(2, 0.1, 2), 1100);

    expect(poses.push(frame(2, 0.1, 200), 1110)).toBe(false);
    expect(poses.push(frame(1, 0, 100), 1120)).toBe(false);
    expect(poses.sample(1200)).toEqual({ x: 2, y: 0, z: 0, heading: 0, speed: 2 });
  });

  it('holds on pause and resumes from the held pose', () => {
    const poses = new LivePoseInterpolator();
    poses.push(frame(1, 0, 0), 1000);
    poses.push(frame(2, 0.1, 10), 1100);
    poses.pause(1150);

    expect(poses.sample(5000)).toEqual({ x: 5, y: 0, z: 0, heading: 0, speed: 5 });
    poses.resume();
    poses.push(frame(3, 0.2, 15), 5000);
    expect(poses.sample(5000)).toEqual({ x: 5, y: 0, z: 0, heading: 0, speed: 5 });
    expect(poses.sample(5050)).toEqual({ x: 10, y: 0, z: 0, heading: 0, speed: 10 });
  });

  it.each(['finished', 'failed', 'stopped', 'cancelled'] as const)(
    'completes interpolation to the final authoritative pose on %s', (status) => {
      const poses = new LivePoseInterpolator();
      poses.push(frame(1, 0, 0), 1000);
      poses.push(frame(2, 0.1, 10, 1), 1100);
      expect(isRuntimePoseTerminal(status)).toBe(true);

      poses.finish(1125);

      expect(poses.sample(1125)).toEqual({ x: 2.5, y: 0, z: 0, heading: 0.25, speed: 2.5 });
      expect(poses.sample(1175)).toEqual({ x: 7.5, y: 0, z: 0, heading: 0.75, speed: 7.5 });
      expect(poses.sample(1200)).toEqual({ x: 10, y: 0, z: 0, heading: 1, speed: 10 });
      expect(poses.sample(10000)).toEqual({ x: 10, y: 0, z: 0, heading: 1, speed: 10 });
      expect(poses.push(frame(3, 0.2, 20), 1210)).toBe(false);
    },
  );

  it('uses the final pose immediately when its interpolation interval already elapsed', () => {
    const poses = new LivePoseInterpolator();
    poses.push(frame(1, 0, 0), 1000);
    poses.push(frame(2, 0.1, 10), 1100);
    poses.finish(1250);

    expect(poses.sample(1250)).toEqual({ x: 10, y: 0, z: 0, heading: 0, speed: 10 });
    expect(poses.push(frame(3, 0.2, 20), 1300)).toBe(false);
  });

  it('clears a terminal transition on reset', () => {
    const poses = new LivePoseInterpolator();
    poses.push(frame(1, 0, 0), 1000);
    poses.push(frame(2, 0.1, 10), 1100);
    poses.finish(1125);
    expect(poses.sample(1125)?.x).toBe(2.5);

    poses.reset();
    expect(poses.sample(5000)).toBeNull();
    poses.push(frame(0, 0, -4), 6000);
    expect(poses.sample(6000)).toEqual({ x: -4, y: 0, z: 0, heading: 0, speed: 4 });
  });

  it('treats stopped and failed runtime statuses as terminal poses', () => {
    expect(isRuntimePoseTerminal('stopped')).toBe(true);
    expect(isRuntimePoseTerminal('cancelled')).toBe(true);
    expect(isRuntimePoseTerminal('failed')).toBe(true);
    expect(isRuntimePoseTerminal('ready')).toBe(false);
    expect(isRuntimePoseTerminal('running')).toBe(false);
    expect(isRuntimePoseTerminal('paused')).toBe(false);
    expect(isRuntimePoseTerminal('resetting')).toBe(false);
  });
});

describe('3D quaternion and bounded WebSocket jitter interpolation', () => {
  it('interpolates authoritative pitch and roll with shortest-arc SLERP', () => {
    const poses = new LivePoseInterpolator();
    const a = { ...frame(0, 0, 0), orientation_wxyz: [1, 0, 0, 0] as [number,number,number,number] };
    const b = { ...frame(1, 0.1, 4), orientation_wxyz: [Math.SQRT1_2, Math.SQRT1_2, 0, 0] as [number,number,number,number] };
    poses.push(a, 1000); poses.push(b, 1100);
    expect(poses.sample(1150)?.orientation_wxyz?.[0]).toBeCloseTo(Math.cos(Math.PI / 8), 6);
    expect(poses.sample(1150)?.orientation_wxyz?.[1]).toBeCloseTo(Math.sin(Math.PI / 8), 6);
    expect(poses.sample(1200)?.orientation_wxyz).toEqual(b.orientation_wxyz);
  });
  it('smooths bounded fast packet arrival without extrapolating or altering final physics state', () => {
    const poses = new LivePoseInterpolator();
    poses.push(frame(0, 0, 0), 1000);
    poses.push(frame(1, .1, 10), 1030);
    expect(poses.sample(1055)?.x).toBeGreaterThan(0);
    expect(poses.sample(1055)?.x).toBeLessThan(10);
    expect(poses.sample(2000)?.x).toBe(10);
  });
});


describe('live WebSocket position continuity', () => {
  it('preserves the exact visible pose at each incoming frame when the previous blend is incomplete', () => {
    const pose = new LivePoseInterpolator();
    pose.push(frame(0, 0, 0), 1000);
    pose.push(frame(1, .1, 1), 1100);
    const visible = pose.sample(1150)!;
    expect(visible.x).toBeCloseTo(.5, 6);
    pose.push(frame(2, .2, 2), 1150);
    expect(pose.sample(1150)?.x).toBeCloseTo(visible.x, 9);
    expect(pose.sample(1200)?.x).toBeGreaterThan(visible.x);
    expect(pose.sample(1250)?.x).toBeLessThanOrEqual(2);
    expect(pose.sample(1600)?.x).toBe(2);
  });

  it('preserves continuity through fast burst arrivals, including quaternion pitch', () => {
    const pose = new LivePoseInterpolator();
    const p = (tick: number, x: number): LivePoseFrame => ({
      ...frame(tick, tick * .1, x), orientation_wxyz: [Math.cos(tick / 12), Math.sin(tick / 12), 0, 0],
    });
    pose.push(p(0, 0), 1000);
    pose.push(p(1, 1), 1030);
    for (let i = 2; i <= 12; i++) {
      const now = 1030 + i * 15;
      const before = pose.sample(now)!;
      expect(pose.push(p(i, i), now)).toBe(true);
      const after = pose.sample(now)!;
      expect(after.x).toBeCloseTo(before.x, 9);
      expect(after.orientation_wxyz?.[0]).toBeCloseTo(before.orientation_wxyz![0], 8);
    }
    expect(pose.sample(10000)?.x).toBe(12);
  });

  it('does not snap backward upon a delayed packet, pause, resume, or terminal transition', () => {
    const pose = new LivePoseInterpolator();
    pose.push(frame(0, 0, 0), 1000);
    pose.push(frame(1, .1, 1), 1100);
    const before = pose.sample(1160)!;
    pose.push(frame(2, .2, 2), 1800);
    expect(pose.sample(1800)?.x).toBeCloseTo(1, 9);
    pose.pause(1830);
    const held = pose.sample(5000)!.x;
    pose.resume();
    pose.push(frame(3, .3, 3), 5000);
    expect(pose.sample(5000)?.x).toBeCloseTo(held, 9);
    pose.finish(5020);
    const atFinish = pose.sample(5020)!.x;
    expect(atFinish).toBeGreaterThanOrEqual(held);
    expect(pose.sample(10000)?.x).toBe(3);
    expect(before.x).toBeGreaterThan(0);
  });

  it('keeps the newest physics pose authoritative without extrapolating through packet stalls', () => {
    const pose = new LivePoseInterpolator();
    pose.push(frame(0, 0, 0), 1000);
    pose.push(frame(1, .1, 2), 1100);
    expect(pose.sample(1400)?.x).toBe(2);
    const old = pose.sample(2000)!.x;
    pose.push(frame(2, .2, 4), 2000);
    expect(pose.sample(2000)?.x).toBe(old);
    expect(pose.sample(10000)?.x).toBe(4);
  });
});

describe('configured speed limit for displayed pose', () => {
  it('does not exceed 15 m/s even under bursty valid authoritative frames', () => {
    const poses = new LivePoseInterpolator(15);
    poses.push(frame(0, 0, 0), 1000);
    for (let k = 1; k <= 30; k++) {
      const at = 1000 + k * 60;
      const before = poses.sample(at)!;
      expect(poses.push(frame(k, k * .1, k * .8), at)).toBe(true);
      expect(poses.sample(at)!.x).toBeCloseTo(before.x, 9);
      let last = poses.sample(at)!.x;
      for (let t = 5; t <= 55; t += 5) {
        const next = poses.sample(at + t)!.x;
        expect(Math.abs(next - last)).toBeLessThanOrEqual(15 * .005 + 1e-7);
        last = next;
      }
    }
    expect(poses.sample(10000)?.x).toBeCloseTo(24);
  });
});

describe('simulation playback multiplier visual cap', () => {
  it('allows realtime and accelerated play without packet teleports', () => {
    const pose = new LivePoseInterpolator(15);
    pose.push(frame(0, 0, 0), 1000);
    pose.push(frame(1, .1, 1), 1100);
    const p1 = pose.sample(1120)!.x;
    pose.setMaxVisualSpeedMps(15 * 16);
    expect(pose.sample(1120)?.x).toBe(p1);
    pose.push(frame(2, .2, 5), 1120);
    expect(pose.sample(1120)?.x).toBeCloseTo(p1);
    expect(pose.sample(1200)?.x).toBe(5);
    pose.setMaxVisualSpeedMps(15);
    expect(pose.sample(2000)?.x).toBe(5);
  });
});
