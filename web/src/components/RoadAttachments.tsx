import { useEffect, useMemo } from 'react';
import { BufferAttribute, BufferGeometry, DoubleSide } from 'three';
import { pointAt, roadLength, sampleRoad, type TrackDocument, type TrackProp } from '../lib/track';

// The gate is derived from the road, not a freely rotated world-space prop.
// Its supports always start at the two road shoulders, even after track edits.
export function CheckpointGate({ track, prop, selected }: {
  track: TrackDocument; prop: TrackProp; selected: boolean;
}) {
  const center = pointAt(track, prop.t);
  const halfSpan = track.width / 2 + 0.65;
  const span = 2 * halfSpan;
  const gateHeight = 5.4;
  return <group position={[center.x, center.y + 0.09, center.z]} rotation={[0, center.heading, 0]}>
    {[-1, 1].map((side) => <group key={side} position={[side * halfSpan, 0, 0]}>
      <mesh castShadow position={[0, gateHeight / 2, 0]}>
        <boxGeometry args={[0.30, gateHeight, 0.36]}/>
        <meshStandardMaterial color={selected ? '#ccf89b' : '#8acbe2'} metalness={0.32} roughness={0.42}/>
      </mesh>
      <mesh castShadow position={[0, 0.12, 0]}>
        <boxGeometry args={[0.8, 0.24, 0.85]}/>
        <meshStandardMaterial color="#36484c" roughness={0.8}/>
      </mesh>
    </group>)}
    <mesh castShadow position={[0, gateHeight, 0]}>
      <boxGeometry args={[span + 0.33, 0.42, 0.52]}/>
      <meshStandardMaterial color={selected ? '#baf58d' : '#7cc6e7'} metalness={0.3} roughness={0.52}/>
    </mesh>
    <mesh position={[0, gateHeight + 0.04, 0.30]}>
      <boxGeometry args={[span * 0.42, 0.22, 0.04]}/>
      <meshBasicMaterial color="#ecfaff"/>
    </mesh>
    {/* Transparent but raycastable: entire open arch is a large draggable hitbox. */}
    <mesh position={[0, gateHeight / 2 + 0.2, 0]}>
      <boxGeometry args={[span + 2.2, gateHeight + 0.9, 2.3]}/>
      <meshBasicMaterial transparent opacity={0} depthWrite={false} side={DoubleSide}/>
    </mesh>
    {selected && <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.14, 0]}>
      <ringGeometry args={[1.5, 1.65, 32]}/>
      <meshBasicMaterial color="#c9f696" side={DoubleSide}/>
    </mesh>}
  </group>;
}

export function makeStrip(track: TrackDocument, prop: TrackProp, left: number, right: number,
  startOffset: number, endOffset: number, height: number): BufferGeometry {
  const totalLength = roadLength(sampleRoad(track, 400));
  const steps = Math.max(2, Math.ceil((endOffset - startOffset) / 0.15));
  const vertices: number[] = [];
  const indices: number[] = [];
  for (let i = 0; i <= steps; i++) {
    const distance = startOffset + (endOffset - startOffset) * i / steps;
    const t = Math.max(0, Math.min(1, prop.t + distance / Math.max(totalLength, 1)));
    const s = pointAt(track, t);
    const centerOffset = Math.max(-track.width / 2 + 2.6, Math.min(track.width / 2 - 2.6, prop.offset));
    for (const across of [left, right]) {
      vertices.push(s.x + s.nx * (centerOffset + across), s.y + height,
        s.z + s.nz * (centerOffset + across));
    }
    if (i > 0) {
      const n = 2 * i;
      indices.push(n - 2, n, n - 1, n - 1, n, n + 1);
    }
  }
  const geometry = new BufferGeometry();
  geometry.setAttribute('position', new BufferAttribute(new Float32Array(vertices), 3));
  geometry.setIndex(indices);
  geometry.computeVertexNormals();
  return geometry;
}

// Surface decal is tessellated along the *same 3D spline* as the asphalt,
// including automatic overpasses and manually created bridge/ramp slopes.
export function BoosterSurface({ track, prop, selected }: {
  track: TrackDocument; prop: TrackProp; selected: boolean;
}) {
  const base = useMemo(() => makeStrip(track, prop, -2.4, 2.4, -3.1, 3.1, 0.22), [track, prop]);
  const outer = useMemo(() => makeStrip(track, prop, -2.15, 2.15, -2.85, 2.85, 0.236), [track, prop]);
  const stripes = useMemo(() =>
    [-1.65, -0.45, 0.75].map((at) => makeStrip(track, prop, -1.82, 1.82, at, at + 0.55, 0.257)),
  [track, prop]);
  useEffect(() => () => { base.dispose(); outer.dispose(); stripes.forEach((g) => g.dispose()); },
    [base, outer, stripes]);
  return <group>
    <mesh geometry={base}>
      <meshBasicMaterial color={selected ? '#c8fc8d' : '#86c64e'} side={DoubleSide}/>
    </mesh>
    <mesh geometry={outer}>
      <meshBasicMaterial color="#b5e976" side={DoubleSide}/>
    </mesh>
    {stripes.map((g, i) => <mesh key={i} geometry={g}>
      <meshBasicMaterial color="#e5ffa7" side={DoubleSide}/>
    </mesh>)}
  </group>;
}
