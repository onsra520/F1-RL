import type { RuntimeFrame, RuntimeSessionStatus } from './runtime';
import { slerpWxyz, type Wxyz } from './orientation';

export type LivePose = { x: number; y: number; z: number; heading: number; speed: number; orientation_wxyz?: Wxyz };
export type LivePoseFrame = Pick<RuntimeFrame, 'tick' | 'time_s' | 'dt_s' | 'position' | 'heading_rad' | 'speed_mps'> & { orientation_wxyz?: Wxyz };
type OrderedFrame = Pick<LivePoseFrame, 'tick' | 'time_s'>;
type RuntimePoseStatus = RuntimeSessionStatus['status'] | 'resetting';

type BufferedPose = { tick: number; time_s: number; receivedAt: number; pose: LivePose };
type TerminalInterpolation = { from: LivePose; startedAt: number; durationMs: number };

const MAX_FRAME_GAP_MS = 1000;

export function isLivePoseFrameInOrder(previous: OrderedFrame | null, next: OrderedFrame, resetPending = false): boolean {
  if (resetPending && next.tick !== 0) return false;
  return !previous || next.tick > previous.tick && next.time_s > previous.time_s;
}

export function isRuntimePoseTerminal(status: RuntimePoseStatus | null): boolean {
  return status === 'finished' || status === 'failed' || status === 'cancelled' || status === 'stopped';
}

function toPose(frame: LivePoseFrame): LivePose {
  return {
    x: frame.position[0], y: frame.position[1], z: frame.position[2],
    heading: frame.heading_rad, speed: frame.speed_mps,
    ...(frame.orientation_wxyz ? { orientation_wxyz: [...frame.orientation_wxyz] as Wxyz } : {}),
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
    ...(from.orientation_wxyz && to.orientation_wxyz
      ? { orientation_wxyz: slerpWxyz(from.orientation_wxyz, to.orientation_wxyz, amount) } : {}),
  };
}

export class LivePoseInterpolator {
  constructor(private maxVisualSpeedMps: number = Number.POSITIVE_INFINITY) {}

  private playbackMultiplier = 1;

  setPlaybackMultiplier(multiplier: number): void {
    if (!Number.isFinite(multiplier) || multiplier <= 0) throw new Error('Invalid playback multiplier');
    if (multiplier !== this.playbackMultiplier) this.previousArrivalGapMs = 0;
    this.playbackMultiplier = multiplier;
  }

  setMaxVisualSpeedMps(limit: number, now?: number): void {
    if (!Number.isFinite(limit) || limit <= 0) throw new Error('Invalid visual speed limit');
    if (now !== undefined && Number.isFinite(now) && this.previous && this.latest && !this.heldPose && !this.terminal) {
      const visible = this.sample(now)!;
      const remainingMs = Math.max(0, this.interpolationStartedAt + this.interpolationMs - now);
      const distance = Math.hypot(this.latest.pose.x - visible.x, this.latest.pose.y - visible.y,
                                  this.latest.pose.z - visible.z);
      this.previous = { ...this.previous, pose: visible };
      this.interpolationStartedAt = now;
      this.interpolationMs = Math.max(remainingMs, distance / limit * 1000);
    }
    this.maxVisualSpeedMps = limit;
  }
  private previous: BufferedPose | null = null;
  private latest: BufferedPose | null = null;
  private heldPose: LivePose | null = null;
  private interpolationMs = 0;
  private interpolationStartedAt = 0;
  private previousArrivalGapMs = 0;
  private resumeFromHeldPose = false;
  private terminal = false;
  private terminalInterpolation: TerminalInterpolation | null = null;

  push(frame: LivePoseFrame, receivedAt: number): boolean {
    if (this.terminal || !Number.isFinite(receivedAt) || this.heldPose && !this.resumeFromHeldPose) return false;
    if (!isLivePoseFrameInOrder(this.latest, frame)) return false;

    // Evaluate the pose that is visible at the instant the new authoritative
    // packet arrives. Starting from the previous *packet* rather than the
    // currently rendered pose causes a backward jump whenever packets are
    // faster than the previous interpolation interval.
    const visiblePose = this.heldPose ?? this.sample(receivedAt);
    const next: BufferedPose = { tick: frame.tick, time_s: frame.time_s, receivedAt, pose: toPose(frame) };
    const origin = this.latest;
    if (!visiblePose) {
      this.latest = next;
      this.previous = null;
      this.interpolationMs = 0;
      return true;
    }

    const expectedMs = frame.dt_s * 1000 / this.playbackMultiplier;
    const arrivalGap = origin ? receivedAt - origin.receivedAt : expectedMs;
    // Never extrapolate beyond the newest authoritative packet. A delayed
    // packet is blended from the held visible pose, not teleported into place.
    // Minimum duration prevents an oversized position step on a burst.
    const distance = Math.hypot(
      next.pose.x - visiblePose.x, next.pose.y - visiblePose.y, next.pose.z - visiblePose.z,
    );
    // An explicitly configured visual speed budget bounds packet-burst catch-up.
    // Physics remains authoritative and the optional default preserves generic unit uses.
    const minimumMs = distance / this.maxVisualSpeedMps * 1000;
    let nominalMs = this.resumeFromHeldPose ? expectedMs
      : Math.max(expectedMs * 0.75, Math.min(arrivalGap, Math.max(expectedMs * 1.5, MAX_FRAME_GAP_MS)));
    if (!this.resumeFromHeldPose && this.previousArrivalGapMs > expectedMs * 1.2) {
      nominalMs = Math.max(nominalMs, Math.min(MAX_FRAME_GAP_MS, this.previousArrivalGapMs * 1.25));
    }
    const duration = Math.max(1, nominalMs, minimumMs);
    if (!Number.isFinite(duration) || !Number.isFinite(arrivalGap) || arrivalGap < 0) {
      return false;
    }

    this.previous = { ...(origin ?? next), pose: visiblePose };
    this.latest = next;
    this.interpolationMs = duration;
    this.interpolationStartedAt = receivedAt;
    this.previousArrivalGapMs = this.resumeFromHeldPose ? 0 : arrivalGap;
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
    const amount = Math.min(1, Math.max(0, (now - this.interpolationStartedAt) / this.interpolationMs));
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
    const pendingMs = this.interpolationMs > 0
      ? Math.max(0, this.interpolationStartedAt + this.interpolationMs - now)
      : 0;
    const distance = Math.hypot(finalPose.x - from.x, finalPose.y - from.y, finalPose.z - from.z);
    const remainingMs = Math.max(pendingMs, distance / this.maxVisualSpeedMps * 1000);
    this.terminalInterpolation = remainingMs > 0
      ? { from, startedAt: now, durationMs: remainingMs }
      : null;
    this.heldPose = null;
    this.previous = null;
    this.resumeFromHeldPose = false;
    this.terminal = true;
  }

  /**
   * A user-requested Reset changes the authoritative physics epoch to tick 0.
   * Preserve the last *visible* pose until tick 0 arrives, then blend back
   * at the configured visual-speed budget rather than teleporting in one rAF.
   */
  prepareRestart(now: number): void {
    const visible = this.sample(now);
    this.reset();
    if (visible) {
      this.heldPose = visible;
      this.resumeFromHeldPose = true;
    }
  }

  reset(): void {
    this.previous = null;
    this.latest = null;
    this.heldPose = null;
    this.interpolationMs = 0;
    this.interpolationStartedAt = 0;
    this.previousArrivalGapMs = 0;
    this.resumeFromHeldPose = false;
    this.terminal = false;
    this.terminalInterpolation = null;
  }
}
