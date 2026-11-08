import { Suspense, useEffect, useMemo, useRef, useState } from 'react';
import { Canvas, useFrame, useThree, type ThreeEvent } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import { BufferAttribute, BufferGeometry, Camera, DoubleSide, Plane, Quaternion, Raycaster, Vector2, Vector3, type Group } from 'three';
import type { PropKind, RoadSample, TrackDocument, TrackProp } from '../lib/track';
import { pointAt, sampleRoad, visibleRoadSections } from '../lib/track';
import { isRuntimePoseTerminal, LivePoseInterpolator, type LivePose, type LivePoseFrame } from '../lib/livePose';
import { shouldReanchorAfterRenderStall, smoothFollowVector } from '../lib/cameraFollow';
import type { RuntimeSessionStatus } from '../lib/runtime';
import SolidRoad from './SolidRoad';
import { BoosterSurface, CheckpointGate } from './RoadAttachments';

export type CarPose = LivePose;
type Mode = 'build' | 'drive';

function ribbon(sections: RoadSample[][], left: number, right: number, offsetY: number) {
  const positions: number[] = [], indices: number[] = [];
  for (const section of sections) {
    const start = positions.length / 3;
    for (const s of section) {
      positions.push(s.x + s.nx * left, s.y + offsetY, s.z + s.nz * left);
      positions.push(s.x + s.nx * right, s.y + offsetY, s.z + s.nz * right);
    }
    for (let i = 0; i < section.length - 1; i++) {
      const k = start + i * 2;
      indices.push(k, k + 2, k + 1, k + 1, k + 2, k + 3);
    }
  }
  const geometry = new BufferGeometry();
  geometry.setAttribute('position', new BufferAttribute(new Float32Array(positions), 3));
  geometry.setIndex(indices);
  geometry.computeVertexNormals();
  return geometry;
}

function Road({ track, samples }: { track: TrackDocument; samples: RoadSample[] }) {
  const sections = useMemo(() => visibleRoadSections(track, samples), [track, samples]);
  const asphalt = useMemo(() => ribbon(sections, -track.width / 2, track.width / 2, 0.08), [sections, track.width]);
  const shoulder = useMemo(() => ribbon(sections, -track.width / 2 - 0.7, track.width / 2 + 0.7, -0.1), [sections, track.width]);
  const left = useMemo(() => ribbon(sections, -track.width / 2 + 0.55, -track.width / 2 + 0.75, 0.10), [sections, track.width]);
  const right = useMemo(() => ribbon(sections, track.width / 2 - 0.75, track.width / 2 - 0.55, 0.10), [sections, track.width]);
  const center = useMemo(() => ribbon(sections, -0.075, 0.075, 0.11), [sections]);
  useEffect(() => () => { asphalt.dispose(); shoulder.dispose(); left.dispose(); right.dispose(); center.dispose(); },
    [asphalt, shoulder, left, right, center]);
  return <>
    <mesh geometry={shoulder} receiveShadow><meshStandardMaterial color="#151e21" side={DoubleSide} roughness={0.98} /></mesh>
    <mesh geometry={asphalt} receiveShadow><meshStandardMaterial color="#354044" side={DoubleSide} roughness={0.91} metalness={0.05} /></mesh>
    <mesh geometry={left}><meshBasicMaterial color="#e3e9df" side={DoubleSide} /></mesh>
    <mesh geometry={right}><meshBasicMaterial color="#e3e9df" side={DoubleSide} /></mesh>
    <mesh geometry={center}><meshBasicMaterial color="#b4a781" transparent opacity={0.35} side={DoubleSide} /></mesh>
    <SolidRoad track={track} sections={sections} />
  </>;
}

function CarModel({ pose, livePose, follow, simulationSpeed, cameraEpoch }: {
  pose: CarPose; livePose: LivePoseInterpolator | null; follow: boolean;
  simulationSpeed: number; cameraEpoch: 'build' | 'live' | 'replay';
}) {
  const car = useRef<Group>(null);
  const { camera } = useThree();
  const front = useMemo(() => new Vector3(), []);
  const desiredCamera = useMemo(() => new Vector3(), []);
  const desiredTarget = useMemo(() => new Vector3(), []);
  const cameraTarget = useMemo(() => new Vector3(), []);
  const rotation = useMemo(() => new Quaternion(), []);
  const worldUp = useMemo(() => new Vector3(0, 1, 0), []);
  const cameraInitialized = useRef(false);
  const lastEpoch = useRef(cameraEpoch);
  const wasHidden = useRef(false);
  const lastRenderMs = useRef<number | null>(null);
  useEffect(() => {
    const visibilityChanged = () => {
      // A hidden browser is not rendering: re-anchor at the authoritative
      // pose on the FIRST visible frame rather than leaving the camera far
      // behind. This is never a Physics/vehicle position override.
      if (document.hidden) wasHidden.current = true;
    };
    document.addEventListener('visibilitychange', visibilityChanged);
    return () => document.removeEventListener('visibilitychange', visibilityChanged);
  }, []);
  useFrame((_, delta) => {
    const now = performance.now();
    if (lastRenderMs.current !== null && shouldReanchorAfterRenderStall(now - lastRenderMs.current)) {
      cameraInitialized.current = false;
    }
    lastRenderMs.current = now;
    if (lastEpoch.current !== cameraEpoch) {
      // An explicit Finish -> Replay starts a different camera epoch.
      // Reset at the user-requested mode change, not each telemetry packet.
      lastEpoch.current = cameraEpoch;
      cameraInitialized.current = false;
    }
    if (wasHidden.current && !document.hidden) {
      wasHidden.current = false;
      cameraInitialized.current = false;
    }
    const current = livePose?.sample(now) ?? pose;
    if (!car.current) return;
    // Python VehicleState.position is the body reference centre, ride_height=0.5m.
    // Never translate an authoritative physics frame vertically.
    const physicalPose = Boolean(current.orientation_wxyz);
    car.current.position.set(current.x, current.y + (physicalPose ? 0 : 0.5), current.z);
    if (current.orientation_wxyz) {
      const [w, x, y, z] = current.orientation_wxyz;
      rotation.set(x, y, z, w).normalize();
    } else {
      rotation.setFromAxisAngle(worldUp, current.heading);
    }
    car.current.quaternion.copy(rotation);
    if (follow) {
      front.set(0, 0, 1).applyQuaternion(rotation);
      desiredCamera.set(current.x - front.x * 19, current.y + 12.2, current.z - front.z * 19);
      desiredTarget.set(current.x + front.x * 6, current.y + 1.3, current.z + front.z * 6);
      if (!cameraInitialized.current) {
        // Entering Drive may change camera mode instantly; only the first
        // frame snaps the camera, never an incoming live Physics packet.
        camera.position.copy(desiredCamera);
        cameraTarget.copy(desiredTarget);
        cameraInitialized.current = true;
      } else {
        const multiplier = Math.max(0.25, Math.min(16, simulationSpeed));
        smoothFollowVector(camera.position, desiredCamera, delta, 4 * multiplier, 20 * multiplier);
        smoothFollowVector(cameraTarget, desiredTarget, delta, 4 * multiplier, 20 * multiplier);
      }
      camera.lookAt(cameraTarget);
    } else {
      cameraInitialized.current = false;
    }
    // Development-only bounded probe, enabled only with ?renderAudit=1. Never alters Physics or render pose.
    if (import.meta.env.DEV && new URLSearchParams(window.location.search).get('renderAudit') === '1') {
      type Sample = { t: number; camera: [number, number, number]; car: [number, number, number];
        target: [number, number, number]; quat: [number, number, number, number]; speed: number };
      const auditWindow = window as Window & { __F1RL_RENDER_AUDIT__?: Sample[] };
      const samples = auditWindow.__F1RL_RENDER_AUDIT__ ??= [];
      if (samples.length < 12000) samples.push({
        t: performance.now(), camera: [camera.position.x, camera.position.y, camera.position.z],
        car: [car.current.position.x, car.current.position.y, car.current.position.z],
        target: [cameraTarget.x, cameraTarget.y, cameraTarget.z],
        quat: [rotation.x, rotation.y, rotation.z, rotation.w], speed: current.speed,
      });
    }
  });
  return <group ref={car}>
    <mesh castShadow position={[0, 0.11, 0]}><boxGeometry args={[1.5, 0.52, 3.5]} /><meshStandardMaterial color="#f27555" roughness={0.35} metalness={0.28}/></mesh>
    <mesh castShadow position={[0, 0.44, -0.25]}><boxGeometry args={[0.9, 0.47, 1.65]} /><meshStandardMaterial color="#253039" roughness={0.36}/></mesh>
    <mesh castShadow position={[0, 0.28, 1.52]}><boxGeometry args={[0.42, 0.35, 1.15]} /><meshStandardMaterial color="#ec6046" metalness={0.3}/></mesh>
    <mesh castShadow position={[0, 0.10, 1.78]}><boxGeometry args={[2.6, 0.15, 0.39]} /><meshStandardMaterial color="#f5c85e" /></mesh>
    <mesh castShadow position={[0, 0.57, -1.60]}><boxGeometry args={[2.45, 0.13, 0.46]} /><meshStandardMaterial color="#202c32" /></mesh>
    {[-1.05, 1.05].flatMap((x) => [-1.5, 1.5].map((z) => (
      <group key={x + ':' + z} position={[x, -0.07, z]}>
        <mesh rotation={[0, 0, Math.PI / 2]} castShadow><cylinderGeometry args={[0.43, 0.43, 0.35, 14]}/><meshStandardMaterial color="#141a1d" roughness={0.9}/></mesh>
        <mesh rotation={[0, 0, Math.PI / 2]}><cylinderGeometry args={[0.20, 0.20, 0.366, 14]}/><meshStandardMaterial color="#858e93" metalness={0.6}/></mesh>
      </group>
    )))}
    <mesh position={[0, 0.77, -0.8]}><sphereGeometry args={[0.36, 14, 10]}/><meshStandardMaterial color="#121c24"/></mesh>
  </group>;
}

const labelColors: Record<PropKind, string> = {
  bridge: '#8396a5', ramp: '#eba65a', gap: '#fcab56', jump: '#c6ec75', barrier: '#ef714f', cone: '#e08d45',
  booster: '#b7ed6b', checkpoint: '#74c7ec',
};

function TrackObject({ prop, track, selected, onPick, onRelease }: {
  prop: TrackProp;
  track: TrackDocument;
  selected: boolean;
  onPick: (event: ThreeEvent<PointerEvent>) => void;
  onRelease: () => void;
}) {
  const p = pointAt(track, prop.t, (prop.kind === 'gap' || prop.kind === 'jump') ? 0 : prop.offset);
  const angle = p.heading + prop.rotation;
  const color = labelColors[prop.kind];
  if (prop.kind === 'checkpoint' || prop.kind === 'booster') {
    return <group onPointerDown={onPick} onPointerUp={onRelease} onClick={(e) => e.stopPropagation()}
      onPointerOver={(e) => { e.stopPropagation(); document.body.style.cursor = 'grab'; }}
      onPointerOut={() => { document.body.style.cursor = ''; }}>
      {prop.kind === 'checkpoint'
        ? <CheckpointGate track={track} prop={prop} selected={selected} />
        : <BoosterSurface track={track} prop={prop} selected={selected} />}
    </group>;
  }
  return <group position={[p.x, p.y + 0.13, p.z]} rotation={[0, angle, 0]} onPointerDown={onPick} onPointerUp={onRelease} onClick={(e) => e.stopPropagation()}>
    {selected && <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.02, 0]}>
      <ringGeometry args={[3.0, 3.18, 48]}/><meshBasicMaterial color="#d5f694" transparent opacity={0.94} side={DoubleSide} />
    </mesh>}
    {(prop.kind === 'bridge' || prop.kind === 'ramp' || prop.kind === 'gap' || prop.kind === 'jump') && <group position={[track.width / 2 + 1.9, 0, 0]}>
      {/* Selection handle lives beside the road: no overhead beam or ramp slab blocks the lane. */}
      <mesh castShadow position={[0, 0.9, 0]}>
        <cylinderGeometry args={[0.12, 0.12, 1.8, 8]}/><meshStandardMaterial color="#86999b"/>
      </mesh>
      <mesh castShadow position={[0, 2, 0]}>
        <boxGeometry args={[1.4, 0.9, 0.3]}/><meshStandardMaterial color={color} emissive={color} emissiveIntensity={0.12}/>
      </mesh>
    </group>}
    {prop.kind === 'barrier' && <>
      <mesh castShadow position={[0, 0.8, 0]}><boxGeometry args={[4.0, 1.5, 0.7]}/><meshStandardMaterial color="#ec7353"/></mesh>
      <mesh position={[0, 0.8, 0.37]}><boxGeometry args={[3.7, 0.24, 0.06]}/><meshBasicMaterial color="#fff0d6"/></mesh>
    </>}
    {prop.kind === 'cone' && <>
      <mesh castShadow position={[0, 0.55, 0]}><coneGeometry args={[0.7, 1.2, 6]}/><meshStandardMaterial color={color}/></mesh>
      <mesh position={[0, 0.06, 0]}><boxGeometry args={[1.3, 0.12, 1.3]}/><meshStandardMaterial color="#2b3235"/></mesh>
    </>}
  </group>;
}


function World({ track, mode, selectedId, onSelect, onPlace, armed, pose, livePose, simulationSpeed, cameraEpoch, roadSamples, onCamera, onDragChange }: {
  track: TrackDocument; mode: Mode; selectedId: string | null; armed: PropKind | null;
  pose: CarPose; livePose: LivePoseInterpolator | null; simulationSpeed: number;
  cameraEpoch: 'build' | 'live' | 'replay'; roadSamples?: RoadSample[]; onCamera: (camera: Camera) => void;
  onSelect: (id: string | null) => void;
  onPlace: (kind: PropKind | null, x: number, z: number, id?: string) => void;
  onDragChange: (id: string | null) => void;
}) {
  const { camera } = useThree();
  const samples = useMemo(() => roadSamples ?? sampleRoad(track), [track, roadSamples]);
  useEffect(() => { onCamera(camera); }, [camera, onCamera]);
  const choose = (event: ThreeEvent<PointerEvent>, id: string) => {
    if (mode !== 'build') return;
    event.stopPropagation();
    onSelect(id);
    document.body.style.cursor = 'grabbing';
    onDragChange(id);
  };
  const stopDrag = () => { document.body.style.cursor = ''; onDragChange(null); };
  return <>
    <color attach="background" args={['#111b20']} />
    <fog attach="fog" args={['#111b20', 140, 330]} />
    <ambientLight intensity={1.2} />
    <hemisphereLight args={['#d9f2ee', '#223037', 1.15]} />
    <directionalLight position={[40, 85, 50]} intensity={2.4} castShadow shadow-mapSize={[2048, 2048]} shadow-camera-left={-130} shadow-camera-right={130} shadow-camera-top={130} shadow-camera-bottom={-130}/>
    <gridHelper args={[400, 40, '#344147', '#263137']} position={[0, -0.6, 0]} />
    <Road track={track} samples={samples} />
    <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -0.8, 0]}
      onClick={(e) => { if (mode !== 'build') return; if (armed) onPlace(armed, e.point.x, e.point.z); else onSelect(null); }}
      onPointerUp={stopDrag}>
      <planeGeometry args={[2000, 2000]}/><meshBasicMaterial visible={false}/>
    </mesh>
    {track.props.map((p) => <TrackObject key={p.id} prop={p} track={track} selected={selectedId === p.id}
      onPick={(e) => choose(e, p.id)} onRelease={stopDrag}/>)}
    <CarModel pose={pose} livePose={livePose} follow={mode === 'drive'}
      simulationSpeed={simulationSpeed} cameraEpoch={cameraEpoch}/>
  </>;
}

export default function RacingScene({ track, mode, selectedId, armed, pose, roadSamples, liveFrame, liveStatus, simulationSpeed, cameraEpoch, onSelect, onPlace }: {
  track: TrackDocument; mode: Mode; selectedId: string | null; armed: PropKind | null; pose: CarPose; roadSamples?: RoadSample[];
  liveFrame: LivePoseFrame | null; liveStatus: RuntimeSessionStatus['status'] | null; simulationSpeed: number;
  cameraEpoch: 'build' | 'live' | 'replay';
  onSelect: (id: string | null) => void;
  onPlace: (kind: PropKind | null, x: number, z: number, id?: string) => void;
}) {
  const cameraRef = useRef<Camera | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const livePose = useRef(new LivePoseInterpolator(15)).current;
  useEffect(() => {
    const multiplier = Math.max(0.25, Math.min(16, simulationSpeed));
    livePose.setMaxVisualSpeedMps(15 * multiplier, performance.now());
    livePose.setPlaybackMultiplier(multiplier);
  }, [livePose, simulationSpeed]);
  const [dragging, setDragging] = useState<string | null>(null);
  const onCamera = (camera: Camera) => { cameraRef.current = camera; };
  useEffect(() => {
    if (liveFrame) livePose.push(liveFrame, performance.now());
    else if (liveStatus === 'running' && livePose.sample(performance.now())) {
      // The backend has explicitly reset its epoch; do not render a
      // one-frame jump from the old location to the new spawn.
      livePose.prepareRestart(performance.now());
    } else livePose.reset();
  }, [liveFrame, livePose]);
  useEffect(() => {
    if (liveStatus === 'paused') livePose.pause(performance.now());
    else if (liveStatus === 'running') livePose.resume();
    else if (isRuntimePoseTerminal(liveStatus)) livePose.finish(performance.now());
  }, [livePose, liveStatus]);
  // Listen at the DOM/window level after selection. Unlike a ground-mesh
  // pointer handler this keeps dragging when the pointer crosses the gate,
  // elevated road, or exits the canvas. Orbit is disabled while selected.
  useEffect(() => {
    if (mode !== 'build' || dragging === null) return;
    let lastMove = -Infinity;
    const move = (event: PointerEvent) => {
      if ((event.buttons & 1) === 0) { setDragging(null); return; }
      if (event.timeStamp - lastMove < 34) return;
      lastMove = event.timeStamp;
      const rect = containerRef.current?.getBoundingClientRect();
      const camera = cameraRef.current;
      if (!rect || !camera || rect.width <= 0 || rect.height <= 0) return;
      if (event.clientX < rect.left || event.clientX > rect.right ||
          event.clientY < rect.top || event.clientY > rect.bottom) return;
      const ndc = new Vector2(
        (event.clientX - rect.left) / rect.width * 2 - 1,
        -(event.clientY - rect.top) / rect.height * 2 + 1,
      );
      const ray = new Raycaster();
      ray.setFromCamera(ndc, camera);
      const target = new Vector3();
      if (ray.ray.intersectPlane(new Plane(new Vector3(0, 1, 0), 0), target)) {
        onPlace(null, target.x, target.z, dragging);
      }
    };
    const finish = () => { setDragging(null); document.body.style.cursor = ''; };
    window.addEventListener('pointermove', move);
    window.addEventListener('pointerup', finish);
    window.addEventListener('pointercancel', finish);
    window.addEventListener('blur', finish);
    return () => {
      window.removeEventListener('pointermove', move);
      window.removeEventListener('pointerup', finish);
      window.removeEventListener('pointercancel', finish);
      window.removeEventListener('blur', finish);
    };
  }, [mode, dragging, onPlace]);
  const onDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    const kind = e.dataTransfer.getData('application/x-f1rl-prop') as PropKind;
    if (mode !== 'build' || !['bridge', 'ramp', 'barrier', 'cone', 'booster', 'checkpoint', 'gap', 'jump'].includes(kind)) return;
    const rect = containerRef.current!.getBoundingClientRect();
    const ndc = new Vector2(((e.clientX - rect.left) / rect.width) * 2 - 1,
                           -((e.clientY - rect.top) / rect.height) * 2 + 1);
    const ray = new Raycaster();
    const camera = cameraRef.current;
    if (!camera) return;
    ray.setFromCamera(ndc, camera);
    const target = new Vector3();
    if (ray.ray.intersectPlane(new Plane(new Vector3(0, 1, 0), 0), target)) onPlace(kind, target.x, target.z);
  };
  const center = useMemo(() => {
    const positions = sampleRoad(track, 20);
    if (!positions.length) return new Vector3();
    return new Vector3(positions.reduce((a,p)=>a+p.x,0)/positions.length, 0, positions.reduce((a,p)=>a+p.z,0)/positions.length);
  }, [track.points]);
  return <div className="scene-container" ref={containerRef} onDragOver={(e) => e.preventDefault()} onDrop={onDrop}>
    <Canvas shadows dpr={[1, 1.8]} camera={{ position: [65, 122, 108], fov: 52, near: 0.1, far: 650 }}>
      <Suspense fallback={null}>
        <World track={track} mode={mode} selectedId={selectedId} armed={armed} pose={pose}
          livePose={mode === 'drive' ? livePose : null} simulationSpeed={simulationSpeed}
          cameraEpoch={cameraEpoch} roadSamples={roadSamples}
          onSelect={onSelect} onPlace={onPlace} onCamera={onCamera} onDragChange={setDragging}/>
        {mode === 'build' && <OrbitControls makeDefault target={[center.x, center.y, center.z]}
          enableDamping dampingFactor={0.09} enabled={!dragging}
          minDistance={12} maxDistance={260} minPolarAngle={0.20} maxPolarAngle={1.34}/>}
      </Suspense>
    </Canvas>
    {mode === 'build' && <div className="view-corner"><span className="status-dot"/> ORBIT VIEW <span className="tiny-separator">·</span> DRAG TO ROTATE <span className="tiny-separator">·</span> SCROLL TO ZOOM</div>}
  </div>;
}
