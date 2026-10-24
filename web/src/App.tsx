import { lazy, Suspense, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  ArrowLeft, ArrowRight, Box, Check, CircleHelp, Compass, Download,
  Flag, Gauge, Grip, Keyboard, Layers3, MousePointer2,
  Mountain, Pause, Pencil, Play, Plus, RotateCcw, RotateCw, Save, Sparkles,
  Trash2, TriangleAlert, Upload, Waves, X
} from 'lucide-react';
import SketchEditor from './components/SketchEditor';
import type { CarPose } from './components/RacingScene';
const RacingScene = lazy(() => import('./components/RacingScene'));
import { createTrack, nearestRoad, parseTrack, pointAt, PROP_KINDS, roadLength, sampleRoad,
         type Point2, type PropKind, type TrackDocument, type TrackProp } from './lib/track';
import { parseReplay, replayAt, type Replay } from './lib/replay';

type Mode = 'sketch' | 'build' | 'drive';
type DriveMode = 'manual' | 'replay';
const STORAGE_KEY = 'f1rl.racing-sandbox.draft.v1';
const assets: { kind: PropKind; title: string; description: string; icon: typeof Box; color: string }[] = [
  { kind: 'bridge', title: 'Bridge', description: 'Raised road section', icon: Layers3, color: 'steel' },
  { kind: 'ramp', title: 'Ramp', description: 'Launch / incline', icon: Mountain, color: 'orange' },
  { kind: 'barrier', title: 'Barrier', description: 'Road block', icon: Box, color: 'red' },
  { kind: 'cone', title: 'Cone', description: 'Small obstacle', icon: TriangleAlert, color: 'orange' },
  { kind: 'booster', title: 'Booster', description: 'Speed pad', icon: Waves, color: 'lime' },
  { kind: 'checkpoint', title: 'Checkpoint', description: 'Route marker', icon: Flag, color: 'blue' },
];

function initialTrack(): TrackDocument {
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    return stored ? parseTrack(JSON.parse(stored)) : createTrack();
  } catch { return createTrack(); }
}
function downloadJSON(name: string, data: unknown) {
  const url = URL.createObjectURL(new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' }));
  const element = document.createElement('a');
  element.href = url;
  element.download = name;
  document.body.appendChild(element);
  element.click();
  element.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1500);
}
function spawnPose(track: TrackDocument): CarPose {
  const p = pointAt(track, 0.015);
  return { x: p.x, y: p.y, z: p.z, heading: p.heading, speed: 0 };
}
const newId = () => typeof crypto !== 'undefined' && 'randomUUID' in crypto
  ? crypto.randomUUID() : String(Date.now()) + '-' + Math.random().toString(36).slice(2);

export default function App() {
  const [track, setTrack] = useState<TrackDocument>(initialTrack);
  const [mode, setMode] = useState<Mode>('sketch');
  const [selected, setSelected] = useState<string | null>(null);
  const [armed, setArmed] = useState<PropKind | null>(null);
  const [driveMode, setDriveMode] = useState<DriveMode>('manual');
  const [pose, setPose] = useState<CarPose>(() => spawnPose(initialTrack()));
  const [replay, setReplay] = useState<Replay | null>(null);
  const [replayTime, setReplayTime] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [toast, setToast] = useState('');
  const [help, setHelp] = useState(false);
  const [saved, setSaved] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null);
  const replayInput = useRef<HTMLInputElement>(null);
  const trackRef = useRef(track);
  const history = useRef<TrackDocument[]>([]);
  const redoStack = useRef<TrackDocument[]>([]);
  const selectedProp = track.props.find((p) => p.id === selected);
  const trackMeters = useMemo(() => roadLength(sampleRoad(track, 300)), [track]);
  const canDrive = track.points.length >= 2;
  const activePose = driveMode === 'replay' && replay ? (() => {
    const p = replayAt(replay, replayTime);
    return { x: p.position[0], y: p.position[1], z: p.position[2], heading: p.heading_rad, speed: p.speed_mps };
  })() : pose;

  useEffect(() => {
    try { localStorage.setItem(STORAGE_KEY, JSON.stringify(track)); setSaved(false); }
    catch { setSaved(false); }
  }, [track]);
  useEffect(() => {
    if (!toast) return;
    const timer = setTimeout(() => setToast(''), 3500);
    return () => clearTimeout(timer);
  }, [toast]);

  const changeTrack = useCallback((update: (prev: TrackDocument) => TrackDocument) => {
    const prev = trackRef.current;
    const next = update(prev);
    if (JSON.stringify(prev) === JSON.stringify(next)) return;
    history.current = [...history.current.slice(-59), prev];
    redoStack.current = [];
    trackRef.current = next;
    setTrack(next);
  }, []);
  const undo = () => {
    const last = history.current.pop();
    if (!last) return;
    redoStack.current.push(trackRef.current);
    trackRef.current = last;
    setTrack(last);
  };
  const redo = () => {
    const next = redoStack.current.pop();
    if (!next) return;
    history.current.push(trackRef.current);
    trackRef.current = next;
    setTrack(next);
  };
  const onSketch = (points: Point2[]) => changeTrack((prev) => ({ ...prev, points, props: points.length ? prev.props : [] }));
  const updateProp = (id: string, patch: Partial<TrackProp>) => changeTrack((prev) => ({
    ...prev, props: prev.props.map((p) => p.id === id ? { ...p, ...patch } : p),
  }));
  const onPlace = useCallback((kind: PropKind | null, x: number, z: number, moveId?: string) => {
    const target = nearestRoad(track, x, z);
    // Every playable prefab belongs to the route: auto-snap even if the drop is imprecise.
    if (!Number.isFinite(target.distance)) { setToast('Sketch a road first'); return; }
    const offset = Math.max(-track.width * 0.42, Math.min(track.width * 0.42, target.offset));
    if (moveId) {
      updateProp(moveId, { t: target.t, offset });
    } else if (kind && PROP_KINDS.includes(kind)) {
      const id = newId();
      changeTrack((prev) => ({ ...prev,
        props: [...prev.props, { id, kind, t: target.t, offset, rotation: 0 }],
      }));
      setSelected(id);
      setArmed(null);
      setToast(kind[0].toUpperCase() + kind.slice(1) + ' placed');
    }
  }, [track, changeTrack]);
  const startDrive = () => {
    if (!canDrive) return;
    setPose(spawnPose(track));
    setPlaying(false); setReplayTime(replay?.frames[0]?.time_s ?? 0);
    setMode('drive');
    setArmed(null); setSelected(null);
  };
  const changeMode = (target: Mode) => {
    if (target === 'drive') { startDrive(); return; }
    setMode(target); setArmed(null); setSelected(null); setPlaying(false);
  };
  const save = () => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(track));
    setSaved(true);
    setToast('Saved locally · export JSON for a portable copy');
  };
  const importMap = async (file?: File) => {
    if (!file) return;
    try {
      const next = parseTrack(JSON.parse(await file.text()));
      history.current.push(track);
      redoStack.current = [];
      trackRef.current = next;
      setTrack(next);
      setPose(spawnPose(next));
      setMode('build'); setSelected(null); setArmed(null);
      setToast('Map imported');
    } catch (error) { setToast(error instanceof Error ? error.message : 'Unable to open map'); }
  };
  const importReplay = async (file?: File) => {
    if (!file) return;
    try {
      const next = parseReplay(JSON.parse(await file.text()));
      setReplay(next);
      setReplayTime(next.frames[0].time_s);
      setPlaying(false); setDriveMode('replay');
      setToast('Replay loaded · development evidence only');
    } catch (error) { setToast(error instanceof Error ? error.message : 'Unable to open replay'); }
  };

  useEffect(() => {
    if (mode !== 'drive' || driveMode !== 'manual') return;
    const down = new Set<string>();
    const keyDown = (e: KeyboardEvent) => {
      if (['KeyW','KeyA','KeyS','KeyD','ArrowUp','ArrowLeft','ArrowDown','ArrowRight','Space'].includes(e.code)) {
        down.add(e.code); e.preventDefault();
      }
      if (e.code === 'Escape') setMode('build');
    };
    const keyUp = (e: KeyboardEvent) => { down.delete(e.code); };
    const blur = () => down.clear();
    window.addEventListener('keydown', keyDown);
    window.addEventListener('keyup', keyUp);
    window.addEventListener('blur', blur);
    let frame = 0, previous = performance.now();
    const tick = (now: number) => {
      const dt = Math.min((now - previous) / 1000, 0.04);
      previous = now;
      const accelerate = down.has('KeyW') || down.has('ArrowUp');
      const brake = down.has('KeyS') || down.has('ArrowDown') || down.has('Space');
      const steer = Number(down.has('KeyD') || down.has('ArrowRight')) - Number(down.has('KeyA') || down.has('ArrowLeft'));
      setPose((p) => {
        const speed = Math.max(0, Math.min(40, p.speed + (accelerate ? 10.5 : 0) * dt - (brake ? 21 : 0) * dt - p.speed * 0.48 * dt));
        const heading = p.heading + steer * speed * 0.012 * dt;
        const x = p.x + Math.sin(heading) * speed * dt;
        const z = p.z + Math.cos(heading) * speed * dt;
        const near = nearestRoad(track, x, z);
        const y = near.distance <= track.width / 2 ? pointAt(track, near.t).y : p.y;
        return { x, z, y, heading, speed };
      });
      frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return () => {
      cancelAnimationFrame(frame);
      window.removeEventListener('keydown', keyDown); window.removeEventListener('keyup', keyUp);
      window.removeEventListener('blur', blur);
    };
  }, [mode, driveMode]);

  useEffect(() => {
    if (!playing || !replay || driveMode !== 'replay' || mode !== 'drive') return;
    let frame = 0, last = performance.now();
    const advance = (now: number) => {
      const dt = Math.min(0.1, (now - last) / 1000);
      last = now;
      setReplayTime((t) => {
        const end = replay.frames[replay.frames.length - 1].time_s;
        if (t + dt >= end) { setPlaying(false); return end; }
        return t + dt;
      });
      frame = requestAnimationFrame(advance);
    };
    frame = requestAnimationFrame(advance);
    return () => cancelAnimationFrame(frame);
  }, [playing, replay, driveMode, mode]);

  return <div className="app">
    <header className="topbar">
      <div className="brand">
        <div className="brand-icon"><Gauge size={23} strokeWidth={2.5}/></div>
        <div className="brand-text"><strong>RACING<span>SANDBOX</span></strong><small>F1-RL / PLAYGROUND</small></div>
      </div>
      <nav className="mode-tabs" aria-label="Mode">
        <button className={mode === 'sketch' ? 'mode-tab current' : 'mode-tab'} onClick={() => changeMode('sketch')}><Pencil size={17}/> <span>Sketch</span></button>
        <button className={mode === 'build' ? 'mode-tab current' : 'mode-tab'} onClick={() => changeMode('build')} disabled={!canDrive}><Layers3 size={17}/> <span>Build</span></button>
        <button className={mode === 'drive' ? 'mode-tab current' : 'mode-tab'} onClick={() => changeMode('drive')} disabled={!canDrive}><Play size={17}/> <span>Drive</span></button>
      </nav>
      <div className="top-actions">
        <button className="icon-button" aria-label="Undo" title="Undo" onClick={undo} disabled={!history.current.length}><RotateCcw size={18}/></button>
        <button className="icon-button" aria-label="Redo" title="Redo" onClick={redo} disabled={!redoStack.current.length}><RotateCw size={18}/></button>
        <span className="header-divider"/>
        <button className="icon-button" aria-label="Import map" title="Import map" onClick={() => fileInput.current?.click()}><Upload size={18}/></button>
        <button className="icon-button" aria-label="Export map" title="Export map" onClick={() => downloadJSON(track.name.toLowerCase().replace(/[^a-z0-9-]+/g, '-') + '.json', track)}><Download size={18}/></button>
        <button className="save-button" onClick={save}><Save size={17}/><span>{saved ? 'Saved' : 'Save map'}</span></button>
      </div>
    </header>

    <main className={'workspace mode-' + mode}>
      <div className="workspace-heading">
        <div className="crumb"><span className="crumb-icon">◈</span> MY WORKSPACE <span className="crumb-slash">/</span> {mode === 'sketch' ? 'TRACK SKETCH' : mode === 'build' ? '3D BUILDER' : 'DRIVE SESSION'}</div>
        {mode !== 'drive' && <div className="track-title"><input aria-label="Track name" value={track.name} maxLength={100} onChange={(e) => changeTrack((t) => ({ ...t, name: e.target.value || 'Untitled track' }))}/><span className="editor-badge">DRAFT</span></div>}
        {mode === 'drive' && <div className="track-title"><h1>{track.name}</h1><span className="editor-badge">VISUAL PREVIEW</span></div>}
        {mode === 'sketch' && <p className="workspace-subtitle">Every great circuit starts with a line.</p>}
        {mode === 'build' && <p className="workspace-subtitle">Drag, drop, and make the track your own.</p>}
        {mode === 'drive' && <p className="workspace-subtitle">Take it for a spin — without leaving the editor.</p>}
      </div>

      {mode === 'sketch' ? <SketchEditor points={track.points} onChange={onSketch} onConvert={() => changeMode('build')}/> :
        <div className="world-stage">
          <Suspense fallback={<div className="scene-loading">Loading 3D world…</div>}>
            <RacingScene track={track} mode={mode} selectedId={selected} armed={armed} pose={activePose} onSelect={setSelected} onPlace={onPlace}/>
          </Suspense>
          {mode === 'build' && <>
            <div className="world-label"><span className="live-glow"/> 3D EDITOR <span>·</span> {Math.round(trackMeters)}M ROAD</div>
            <div className="build-instructions"><MousePointer2 size={16}/> {armed ? 'Click the road to place ' + armed : 'Drag an item from the dock onto the track'}</div>
            {selectedProp && <div className="prop-inspector">
              <div className="inspector-top"><strong>{selectedProp.kind[0].toUpperCase() + selectedProp.kind.slice(1)}</strong><button aria-label="Close selection" onClick={() => setSelected(null)}><X size={16}/></button></div>
              <span className="inspector-label">ROTATION</span>
              <div className="inspector-actions">
                <button title="Rotate counterclockwise" onClick={() => updateProp(selectedProp.id, { rotation: selectedProp.rotation - Math.PI / 12 })}><RotateCcw size={17}/></button>
                <span>{Math.round(selectedProp.rotation * 180 / Math.PI)}°</span>
                <button title="Rotate clockwise" onClick={() => updateProp(selectedProp.id, { rotation: selectedProp.rotation + Math.PI / 12 })}><RotateCw size={17}/></button>
              </div>
              <button className="remove-prop" onClick={() => { changeTrack((p) => ({...p, props: p.props.filter((x) => x.id !== selectedProp.id)})); setSelected(null); }}><Trash2 size={15}/> Remove object</button>
            </div>}
          </>}
          {mode === 'drive' && <div className="drive-overlay">
            <div className="drive-mode-switch">
              <button className={driveMode === 'manual' ? 'selected' : ''} onClick={() => {setDriveMode('manual'); setPlaying(false);}}>Manual</button>
              <button className={driveMode === 'replay' ? 'selected' : ''} onClick={() => {setDriveMode('replay'); setPlaying(false);}}>PPO Replay</button>
            </div>
            <div className="speed-readout"><span>SPEED</span><strong>{Math.round(activePose.speed * 3.6)}</strong><small>KM/H</small></div>
            <div className="drive-warning"><TriangleAlert size={14}/>{driveMode === 'manual' ? 'Arcade visual preview · not Python physics' : 'Imported replay · not independently verified'}</div>
            {driveMode === 'manual' ? <div className="driving-keys"><span className="keycap">W</span><span className="keycap">A</span><span className="keycap">S</span><span className="keycap">D</span> <span>Steer / accelerate · Space to brake</span></div> :
              replay ? <div className="replay-controls">
                <button aria-label={playing ? 'Pause replay' : 'Play replay'} onClick={() => {if (replayTime >= replay.frames[replay.frames.length - 1].time_s) setReplayTime(replay.frames[0].time_s); setPlaying(!playing);}}>{playing ? <Pause size={19}/> : <Play size={19}/>}</button>
                <input aria-label="Replay timeline" type="range" min={replay.frames[0].time_s} max={replay.frames[replay.frames.length - 1].time_s} step="0.01" value={replayTime} onChange={(e) => {setPlaying(false);setReplayTime(Number(e.target.value));}}/>
                <span>{replayTime.toFixed(1)}s</span><span className="replay-name">{replay.source.model_id}</span>
              </div> : <div className="replay-empty"><span>Have a PPO run?</span><button onClick={() => replayInput.current?.click()}><Upload size={16}/> Import TRAIN replay</button></div>}
            <div className="drive-bottom"><button onClick={() => {setPose(spawnPose(track));setPlaying(false);setReplayTime(replay?.frames[0]?.time_s ?? 0);}}><RotateCcw size={17}/> Restart</button><button className="leave-drive" onClick={() => changeMode('build')}><ArrowLeft size={17}/> Back to build</button></div>
          </div>}
        </div>}
      {mode === 'build' && <div className="build-dock">
        <div className="dock-heading"><div><strong>BUILD KIT</strong><span>Drag assets onto your track</span></div><span className="dock-count">06 ITEMS</span></div>
        <div className="asset-list">{assets.map(({kind, title, description, icon: Icon, color}) => (
          <button key={kind} draggable className={armed === kind ? 'asset selected' : 'asset'} title={description}
            onDragStart={(e) => {e.dataTransfer.setData('application/x-f1rl-prop', kind);e.dataTransfer.effectAllowed = 'copy';setArmed(kind);}}
            onDragEnd={() => setArmed(null)} onClick={() => setArmed(armed === kind ? null : kind)}>
            <span className={'asset-icon ' + color}><Icon size={25} strokeWidth={1.65}/></span>
            <strong>{title}</strong><span>{description}</span>
          </button>
        ))}</div>
        <div className="dock-foot"><span><Grip size={16}/> DRAG & DROP</span><span className="dock-foot-right">Objects are editable · Select to rotate or delete</span></div>
      </div>}
      {mode === 'sketch' && <div className="sketch-footer"><span><Sparkles size={15}/> Sketch is just the beginning</span><span>Step 01 / 03 <ArrowRight size={15}/></span></div>}
      {mode === 'drive' && <div className="drive-footer"><span><Check size={16}/> Ready for visual preview</span><span>Physics validation remains on the Python side</span></div>}
    </main>
    <div className="floating-help">
      <button aria-label="Help" className="help-button" onClick={() => setHelp(!help)}><CircleHelp size={20}/></button>
      {help && <div className="help-popover">
        <strong>HOW TO PLAY</strong>
        <p><Pencil size={15}/> Sketch: draw or adjust route points.</p>
        <p><Compass size={15}/> Build: orbit with mouse, zoom with wheel.</p>
        <p><Plus size={15}/> Drag an item or select then click road.</p>
        <p><Keyboard size={15}/> Drive: WASD / arrows + Space.</p>
        <small>Models, scores and physics aren't changed by the editor.</small>
      </div>}
    </div>
    {toast && <div className="toast" role="status"><Check size={17}/>{toast}<button aria-label="Dismiss" onClick={() => setToast('')}><X size={15}/></button></div>}
    <input type="file" accept=".json,application/json" ref={fileInput} hidden onChange={(e) => {void importMap(e.target.files?.[0]);e.target.value = '';}}/>
    <input type="file" accept=".json,application/json" ref={replayInput} hidden onChange={(e) => {void importReplay(e.target.files?.[0]);e.target.value = '';}}/>
  </div>;
}
