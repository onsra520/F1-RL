import type { RuntimeFrame, RuntimeSessionStatus } from './runtime';

export type LivePose = { x: number; y: number; z: number; heading: number; speed: number };
export type LivePoseFrame = Pick<RuntimeFrame, 'tick' | 'time_s' | 'dt_s' | 'position' | 'heading_rad' | 'speed_mps'>;
type OrderedFrame = Pick<LivePoseFrame, 'tick' | 'time_s'>;

type BufferedPose = { tick: number; time_s: number; receivedAt: number; pose: LivePose };
type TerminalInterpolation = { from: LivePose; startedAt: number; durationMs: number };

const MAX_FRAME_GAP_MS = 250;

export function isLivePoseFrameInOrder(previous: OrderedFrame | null, next: OrderedFrame, resetPending = false): boolean {
  if (resetPending && next.tick !== 0) return false;
  return !previous || next.tick > previous.tick && next.time_s > previous.time_s;
}

export function isRuntimePoseTerminal(status: RuntimeSessionStatus['status'] | null): boolean {
  return status !== null && status !== 'ready' && status !== 'running' && status !== 'paused';
}

function toPose(frame: LivePoseFrame): LivePose {
  return {
    x: frame.position[0], y: frame.position[1], z: frame.position[2],
    heading: frame.heading_rad, speed: frame.speed_mps,
  };
}

function interpolatePose(from: LivePose, to: LivePose, amount: number): LivePose {
  const headingDelta = Math.atan2(Math.sin(to.heading - from.heading), Math.cos(to.heading - from.heading));
  return {
    x: from.x + (to.x - from.x) * amount,
    y: from.y + (to.y - from.y) * amount,
    z: from.z + (to.z - from.z) * amount,
    heading: from.heading + headingDelta * amount,
    speed: from.speed + (to.speed - from.speed) * amount,
  };
}

export class LivePoseInterpolator {
  private previous: BufferedPose | null = null;
  private latest: BufferedPose | null = null;
  private heldPose: LivePose | null = null;
  private interpolationMs = 0;
  private resumeFromHeldPose = false;
  private terminal = false;
  private terminalInterpolation: TerminalInterpolation | null = null;

  push(frame: LivePoseFrame, receivedAt: number): boolean {
    if (this.terminal || !Number.isFinite(receivedAt) || this.heldPose && !this.resumeFromHeldPose) return false;
    if (!isLivePoseFrameInOrder(this.latest, frame)) return false;

    const next: BufferedPose = { tick: frame.tick, time_s: frame.time_s, receivedAt, pose: toPose(frame) };
    const origin = this.latest;
    const originPose = this.heldPose ?? origin?.pose;
    if (!originPose) {
      this.latest = next;
      this.previous = null;
      this.interpolationMs = 0;
      return true;
    }

    const arrivalGap = origin ? receivedAt - origin.receivedAt : frame.dt_s * 1000;
    const duration = this.resumeFromHeldPose ? frame.dt_s * 1000 : arrivalGap;
    if (!Number.isFinite(duration) || duration <= 0 || duration > MAX_FRAME_GAP_MS) {
      this.previous = null;
      this.latest = next;
      this.interpolationMs = 0;
      this.heldPose = null;
      this.resumeFromHeldPose = false;
      return true;
    }

    this.previous = origin ? { ...origin, pose: originPose } : next;
    this.latest = next;
    this.interpolationMs = duration;
    this.heldPose = null;
    this.resumeFromHeldPose = false;
    return true;
  }

  sample(now: number): LivePose | null {
    if (this.terminal) {
      if (!this.latest) return null;
      const transition = this.terminalInterpolation;
      if (!transition || transition.durationMs <= 0) return this.latest.pose;
      const amount = Math.min(1, Math.max(0, (now - transition.startedAt) / transition.durationMs));
      if (amount === 0) return transition.from;
      if (amount === 1) return this.latest.pose;
      return interpolatePose(transition.from, this.latest.pose, amount);
    }
    if (this.heldPose) return this.heldPose;
    if (!this.latest) return null;
    if (!this.previous || this.interpolationMs <= 0) return this.latest.pose;
    const amount = Math.min(1, Math.max(0, (now - this.latest.receivedAt) / this.interpolationMs));
    if (amount === 0) return this.previous.pose;
    if (amount === 1) return this.latest.pose;
    return interpolatePose(this.previous.pose, this.latest.pose, amount);
  }

  pause(now: number): void {
    if (this.terminal) return;
    this.heldPose = this.sample(now) ?? this.latest?.pose ?? null;
    this.previous = null;
    this.resumeFromHeldPose = false;
  }

  resume(): void {
    if (!this.terminal && this.heldPose) this.resumeFromHeldPose = true;
  }

  finish(now: number): void {
    if (this.terminal) return;
    if (!this.latest) { this.terminal = true; return; }
    const finalPose = this.latest.pose;
    const from = this.heldPose ?? this.sample(now) ?? finalPose;
    const remainingMs = this.interpolationMs > 0
      ? Math.max(0, this.latest.receivedAt + this.interpolationMs - now)
      : 0;
    this.terminalInterpolation = remainingMs > 0
      ? { from, startedAt: now, durationMs: Math.min(remainingMs, MAX_FRAME_GAP_MS) }
      : null;
    this.heldPose = null;
    this.previous = null;
    this.resumeFromHeldPose = false;
    this.terminal = true;
  }

  reset(): void {
    this.previous = null;
    this.latest = null;
    this.heldPose = null;
    this.interpolationMs = 0;
    this.resumeFromHeldPose = false;
    this.terminal = false;
    this.terminalInterpolation = null;
  }
}
