import type { Vector3 } from 'three';

/**
 * Smooth visual follow motion only; does not alter the simulated car pose.
 *
 * Limits camera movement per rendered frame so an occasional WebSocket or
 * frame-time spike cannot cause a visible camera teleport. The follow target
 * itself stays physical; delayed camera recovery is purely presentational.
 */
export function smoothFollowVector(
  current: Vector3,
  desired: Vector3,
  elapsedSeconds: number,
  responseHz = 4,
  maxSpeedMps = 20,
): void {
  if (!Number.isFinite(elapsedSeconds) || elapsedSeconds <= 0) return;
  const dt = Math.min(elapsedSeconds, 0.05);
  const fraction = 1 - Math.exp(-responseHz * dt);
  const dx = (desired.x - current.x) * fraction;
  const dy = (desired.y - current.y) * fraction;
  const dz = (desired.z - current.z) * fraction;
  const distance = Math.hypot(dx, dy, dz);
  const maxStep = maxSpeedMps * dt;
  const scale = distance > maxStep ? maxStep / distance : 1;
  current.set(current.x + dx * scale, current.y + dy * scale, current.z + dz * scale);
}
