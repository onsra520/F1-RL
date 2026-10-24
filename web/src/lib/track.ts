import { CatmullRomCurve3, Vector3 } from 'three';

export type Point2 = { x: number; z: number };
export type PropKind = 'bridge' | 'ramp' | 'barrier' | 'cone' | 'booster' | 'checkpoint';
export type TrackProp = { id: string; kind: PropKind; t: number; offset: number; rotation: number };
export type TrackDocument = {
  schema: 'f1rl_racing_sandbox_draft_v1';
  id: string;
  name: string;
  width: number;
  points: Point2[];
  props: TrackProp[];
};
export type RoadSample = Point2 & { y: number; t: number; heading: number; nx: number; nz: number };
export const PROP_KINDS: PropKind[] = ['bridge', 'ramp', 'barrier', 'cone', 'booster', 'checkpoint'];
export const EXAMPLE_POINTS: Point2[] = [
  { x: -90, z: 35 }, { x: -65, z: 33 }, { x: -41, z: 14 }, { x: -9, z: -28 },
  { x: 22, z: -29 }, { x: 35, z: 5 }, { x: 54, z: 28 }, { x: 87, z: 12 },
];

export function createTrack(): TrackDocument {
  return {
    schema: 'f1rl_racing_sandbox_draft_v1',
    id: 'sandbox-' + Math.random().toString(36).slice(2, 9),
    name: 'My first circuit',
    width: 11,
    points: EXAMPLE_POINTS.map((p) => ({ ...p })),
    props: [
      { id: 'sample-boost', kind: 'booster', t: 0.20, offset: 0, rotation: 0 },
      { id: 'sample-cone', kind: 'cone', t: 0.55, offset: 2.8, rotation: 0 },
    ],
  };
}

const smoothStep = (a: number, b: number, t: number) => {
  const u = Math.min(1, Math.max(0, (t - a) / (b - a)));
  return u * u * (3 - 2 * u);
};

export function roadHeight(t: number, props: TrackProp[]): number {
  let y = 0;
  for (const p of props) {
    const d = Math.abs(t - p.t);
    if (p.kind === 'bridge') {
      const lift = 5 * (1 - smoothStep(0.055, 0.125, d));
      y = Math.max(y, lift);
    } else if (p.kind === 'ramp') {
      const lift = 2.8 * (1 - smoothStep(0, 0.075, d));
      y = Math.max(y, lift);
    }
  }
  return y;
}

export function sampleRoad(track: TrackDocument, count = 240): RoadSample[] {
  if (track.points.length < 2) return [];
  const spline = new CatmullRomCurve3(
    track.points.map((p) => new Vector3(p.x, 0, p.z)),
    false,
    'centripetal',
  );
  const samples: RoadSample[] = [];
  for (let i = 0; i <= count; i++) {
    const t = i / count;
    const pos = spline.getPointAt(t);
    const tangent = spline.getTangentAt(t).normalize();
    const nx = -tangent.z;
    const nz = tangent.x;
    samples.push({
      x: pos.x, z: pos.z, y: roadHeight(t, track.props), t,
      heading: Math.atan2(tangent.x, tangent.z), nx, nz,
    });
  }
  return samples;
}

export function roadLength(samples: RoadSample[]): number {
  let length = 0;
  for (let i = 1; i < samples.length; i++) {
    length += Math.hypot(samples[i].x - samples[i - 1].x, samples[i].z - samples[i - 1].z);
  }
  return length;
}

export function pointAt(track: TrackDocument, t: number, offset = 0): RoadSample {
  const arr = sampleRoad(track, 400);
  if (!arr.length) return { x: 0, y: 0, z: 0, t: 0, heading: 0, nx: 1, nz: 0 };
  const u = Math.max(0, Math.min(1, t)) * (arr.length - 1);
  const a = arr[Math.floor(u)];
  const b = arr[Math.min(arr.length - 1, Math.ceil(u))];
  const k = u - Math.floor(u);
  const x = a.x + (b.x - a.x) * k;
  const z = a.z + (b.z - a.z) * k;
  const y = a.y + (b.y - a.y) * k;
  const heading = Math.atan2(Math.sin(a.heading) * (1 - k) + Math.sin(b.heading) * k,
                            Math.cos(a.heading) * (1 - k) + Math.cos(b.heading) * k);
  const nx = -Math.cos(heading), nz = Math.sin(heading);
  return { x: x + offset * nx, z: z + offset * nz, y, t, heading, nx, nz };
}

export function nearestRoad(track: TrackDocument, x: number, z: number): { t: number; offset: number; distance: number } {
  const samples = sampleRoad(track, 400);
  if (samples.length < 2) return { t: 0, offset: 0, distance: Infinity };
  let best = { t: 0, offset: 0, distance: Infinity };
  for (let i = 1; i < samples.length; i++) {
    const a = samples[i - 1], b = samples[i];
    const dx = b.x - a.x, dz = b.z - a.z;
    const lengthSq = dx * dx + dz * dz;
    if (lengthSq < 1e-9) continue;
    const u = Math.max(0, Math.min(1, ((x - a.x) * dx + (z - a.z) * dz) / lengthSq));
    const px = a.x + u * dx, pz = a.z + u * dz;
    const distance = Math.hypot(x - px, z - pz);
    if (distance < best.distance) {
      const nx = -dz / Math.sqrt(lengthSq), nz = dx / Math.sqrt(lengthSq);
      best = { t: a.t + u * (b.t - a.t), offset: (x - px) * nx + (z - pz) * nz, distance };
    }
  }
  return best;
}

const finite = (n: unknown): n is number => typeof n === 'number' && Number.isFinite(n);
export function parseTrack(input: unknown): TrackDocument {
  if (!input || typeof input !== 'object') throw new Error('Invalid map file');
  const d = input as Partial<TrackDocument>;
  if (d.schema !== 'f1rl_racing_sandbox_draft_v1') throw new Error('Unsupported map schema');
  if (typeof d.id !== 'string' || !d.id || d.id.length > 100 ||
      typeof d.name !== 'string' || !d.name.trim() || d.name.length > 100 ||
      !finite(d.width) || d.width < 4 || d.width > 30 ||
      !Array.isArray(d.points) || d.points.length < 2 || d.points.length > 512 ||
      !d.points.every((p) => p && finite(p.x) && finite(p.z) && Math.abs(p.x) <= 10000 && Math.abs(p.z) <= 10000) ||
      !Array.isArray(d.props) || d.props.length > 200 ||
      !d.props.every((p) => p && typeof p.id === 'string' && p.id.length > 0 &&
        p.id.length <= 100 && PROP_KINDS.includes(p.kind) && finite(p.t) && p.t >= 0 && p.t <= 1 &&
        finite(p.offset) && Math.abs(p.offset) <= 30 && finite(p.rotation) && Math.abs(p.rotation) <= 1000)) {
    throw new Error('Map geometry or props are invalid');
  }
  if (new Set(d.props.map((p) => p.id)).size !== d.props.length) throw new Error('Duplicate prop IDs');
  if (d.points.every((p) => Math.hypot(p.x - d.points![0].x, p.z - d.points![0].z) < 0.01)) {
    throw new Error('Map must have a route with nonzero length');
  }
  return {
    schema: d.schema, id: d.id, name: d.name, width: d.width,
    points: d.points.map((p) => ({ x: p.x, z: p.z })),
    props: d.props.map((p) => ({ id: p.id, kind: p.kind, t: p.t, offset: p.offset, rotation: p.rotation })),
  };
}
