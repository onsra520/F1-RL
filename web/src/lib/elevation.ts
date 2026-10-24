import type { Point2 } from './track';

export type AutoBridge = {
  x: number; z: number;
  underT: number; overT: number;
  rise: number;
  plateauMeters: number;
  outerMeters: number;
};

/** Non-adjacent centerline intersections, ignoring shared endpoints. */
function segmentCross(a: Point2, b: Point2, c: Point2, d: Point2) {
  const rx = b.x - a.x, rz = b.z - a.z;
  const sx = d.x - c.x, sz = d.z - c.z;
  const det = rx * sz - rz * sx;
  if (Math.abs(det) < 1e-8) return null;
  const qx = c.x - a.x, qz = c.z - a.z;
  const u = (qx * sz - qz * sx) / det;
  const v = (qx * rz - qz * rx) / det;
  if (u <= 0.0001 || u >= 0.9999 || v <= 0.0001 || v >= 0.9999) return null;
  return { u, v, x: a.x + u * rx, z: a.z + u * rz };
}

export function bumpAt(distance: number, center: number, plateau: number, outer: number) {
  const d = Math.abs(distance - center);
  if (d <= plateau) return 1;
  if (d >= outer) return 0;
  const u = (d - plateau) / (outer - plateau);
  return 1 - u * u * (3 - 2 * u);
}

export function bridgeElevationAt(t: number, bridges: AutoBridge[], length: number) {
  const distance = t * length;
  return bridges.reduce((sum, bridge) =>
    sum + bridge.rise * bumpAt(distance, bridge.overT * length, bridge.plateauMeters, bridge.outerMeters), 0);
}

/** A later traversal goes above an earlier traversal. 
 * Heights are inferred for the VISUAL track editor, not certified road contact physics.
 */
export function detectAutoBridges(samples: Point2[], totalLength: number, minimumClearance = 6): AutoBridge[] {
  if (samples.length < 6 || totalLength <= 0) return [];
  const distances = [0];
  for (let i = 1; i < samples.length; i++) {
    distances.push(distances[i - 1] + Math.hypot(samples[i].x - samples[i - 1].x, samples[i].z - samples[i - 1].z));
  }
  const intersections: { x: number; z: number; under: number; over: number }[] = [];
  for (let i = 0; i < samples.length - 1; i++) {
    for (let j = i + 3; j < samples.length - 1; j++) {
      if (distances[j] - distances[i + 1] < 16) continue;
      const hit = segmentCross(samples[i], samples[i + 1], samples[j], samples[j + 1]);
      if (!hit) continue;
      const under = distances[i] + hit.u * (distances[i + 1] - distances[i]);
      const over = distances[j] + hit.v * (distances[j + 1] - distances[j]);
      if (intersections.some((a) => Math.hypot(a.x - hit.x, a.z - hit.z) < 1.5)) continue;
      intersections.push({ x: hit.x, z: hit.z, under, over });
    }
  }

  intersections.sort((a, b) => a.over - b.over);
  const result: AutoBridge[] = [];
  for (const cross of intersections) {
    const remaining = Math.min(cross.over, totalLength - cross.over);
    // Avoid unbounded height profiles at the route's start/finish.
    if (remaining < 10) continue;
    const plateauMeters = 6;
    const outerMeters = Math.min(42, remaining - 0.1);
    if (outerMeters <= plateauMeters + 1) continue;
    const underT = cross.under / totalLength, overT = cross.over / totalLength;
    const alreadyAbove = bridgeElevationAt(overT, result, totalLength);
    const lowerRoad = bridgeElevationAt(underT, result, totalLength);
    const rise = Math.max(0, minimumClearance - (alreadyAbove - lowerRoad));
    if (rise < 0.01) continue;
    result.push({ x: cross.x, z: cross.z, underT, overT, rise, plateauMeters, outerMeters });
  }
  return result;
}
