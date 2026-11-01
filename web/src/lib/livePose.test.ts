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

  it('snaps to the authoritative pose after a long frame gap and never extrapolates', () => {
    const poses = new LivePoseInterpolator();
    poses.push(frame(1, 0, 0), 1000);
    poses.push(frame(2, 0.1, 1), 1100);
    poses.push(frame(3, 0.2, 9), 1500);

    expect(poses.sample(1500)).toEqual({ x: 9, y: 0, z: 0, heading: 0, speed: 9 });
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

  it('freezes terminal state and clears old samples on reset', () => {
    const poses = new LivePoseInterpolator();
    poses.push(frame(1, 0, 0), 1000);
    poses.push(frame(2, 0.1, 10), 1100);
    poses.finish(1150);

    expect(poses.sample(5000)).toEqual({ x: 5, y: 0, z: 0, heading: 0, speed: 5 });
    expect(poses.push(frame(3, 0.2, 20), 1200)).toBe(false);
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
  });
});
