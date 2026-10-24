import { useEffect, useMemo } from 'react';
import { BufferAttribute, BufferGeometry, DoubleSide } from 'three';
import type { RoadSample, TrackDocument } from '../lib/track';

// The road is a solid strip, not an infinitely thin plane. All faces follow
// the SAME sampled elevation profile as the rendered surface (visual draft only).
function sideGeometry(samples: RoadSample[], width: number, thickness: number) {
  const vertices: number[] = [], indices: number[] = [];
  const half = width / 2;
  const addQuad = (a: number[], b: number[], c: number[], d: number[]) => {
    const start = vertices.length / 3;
    vertices.push(...a, ...b, ...c, ...d);
    indices.push(start, start + 1, start + 2, start, start + 2, start + 3);
  };
  const sides = (p: RoadSample) => {
    const left = [p.x - p.nx * half, p.y - 0.11, p.z - p.nz * half];
    const right = [p.x + p.nx * half, p.y - 0.11, p.z + p.nz * half];
    return { left, right,
      leftBottom: [left[0], left[1] - thickness, left[2]],
      rightBottom: [right[0], right[1] - thickness, right[2]] };
  };
  for (let i = 1; i < samples.length; i++) {
    const a = sides(samples[i - 1]), b = sides(samples[i]);
    addQuad(a.left, b.left, b.leftBottom, a.leftBottom);
    addQuad(a.rightBottom, b.rightBottom, b.right, a.right);
    addQuad(a.leftBottom, b.leftBottom, b.rightBottom, a.rightBottom);
  }
  if (samples.length > 1) {
    const first = sides(samples[0]), last = sides(samples[samples.length - 1]);
    addQuad(first.right, first.left, first.leftBottom, first.rightBottom);
    addQuad(last.left, last.right, last.rightBottom, last.leftBottom);
  }
  const geo = new BufferGeometry();
  geo.setAttribute('position', new BufferAttribute(new Float32Array(vertices), 3));
  geo.setIndex(indices);
  geo.computeVertexNormals();
  return geo;
}
function railGeometry(samples: RoadSample[], halfWidth: number, height: number) {
  const vertices: number[] = [], indices: number[] = [];
  for (let i = 1; i < samples.length; i++) {
    const a = samples[i - 1], b = samples[i];
    // Rail only on elevated spans and clear of the driving surface.
    if (a.y < 2.3 || b.y < 2.3) continue;
    for (const sign of [-1, 1]) {
      const push = (s: RoadSample, h: number) =>
        [s.x + sign * s.nx * halfWidth, s.y + h, s.z + sign * s.nz * halfWidth];
      const index = vertices.length / 3;
      vertices.push(...push(a, 0.05), ...push(b, 0.05),
        ...push(b, height), ...push(a, height));
      indices.push(index, index + 1, index + 2, index, index + 2, index + 3);
    }
  }
  const geo = new BufferGeometry();
  geo.setAttribute('position', new BufferAttribute(new Float32Array(vertices), 3));
  geo.setIndex(indices); geo.computeVertexNormals();
  return geo;
}
export default function SolidRoad({ track, samples }: { track: TrackDocument; samples: RoadSample[] }) {
  const fascia = useMemo(() => sideGeometry(samples, track.width + 1.4, 1.5), [samples, track.width]);
  const rails = useMemo(() => railGeometry(samples, track.width / 2 + 0.55, 0.9), [samples, track.width]);
  useEffect(() => () => { fascia.dispose(); rails.dispose(); }, [fascia, rails]);
  return <>
    <mesh geometry={fascia} receiveShadow castShadow>
      <meshStandardMaterial color="#657477" roughness={0.91} side={DoubleSide}/>
    </mesh>
    <mesh geometry={rails} castShadow>
      <meshStandardMaterial color="#e3dfb6" metalness={0.27} roughness={0.48} side={DoubleSide}/>
    </mesh>
    {samples.filter((s, i) => i % 26 === 0 && s.y > 2.5).map((s, i) => (
      <group key={i}>
        {[-1, 1].map((sign) =>
          <mesh key={sign} position={[s.x + s.nx * (track.width / 2 - 0.7) * sign,
            (s.y - 1.5) / 2 - 0.8,
            s.z + s.nz * (track.width / 2 - 0.7) * sign]} castShadow>
            <boxGeometry args={[0.95, Math.max(0.2, s.y - 1.5), 0.95]}/>
            <meshStandardMaterial color="#536266" roughness={0.94}/>
          </mesh>
        )}
      </group>
    ))}
  </>;
}
