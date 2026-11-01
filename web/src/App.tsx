import { lazy, Suspense, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  ArrowLeft, Box, Check, CircleHelp, Compass, Download,
  Flag, Gauge, Keyboard, Layers3, MousePointer2,
  LoaderCircle, Mountain, Pause, Pencil, Play, Plus, RotateCcw, RotateCw, Save,
  Trash2, TriangleAlert, Upload, Waves, X
} from 'lucide-react';
import SketchEditor from './components/SketchEditor';
import type { CarPose } from './components/RacingScene';
const RacingScene = lazy(() => import('./components/RacingScene'));
import { createTrack, getAutoBridges, nearestRoad, parseTrack, pointAt, PROP_KINDS, roadLength, sampleRoad,
         type Point2, type PropKind, type TrackDocument, type TrackProp } from './lib/track';
import { parseReplay, replayAt, replayModelId, type Replay } from './lib/replay';
import { manualActionFromKeys } from './lib/manual';
import { isLivePoseFrameInOrder } from './lib/livePose';
import RuntimeOutcomeMessage from './RuntimeOutcomeMessage';
import {
  invalidateRuntimeTrackSelection, isCurrentCustomTrackValidation, resolveRuntimeTrackRequest,
  RuntimeApiError, RuntimeClient, type RuntimeCatalog,
  type RuntimeSessionStatus, type RuntimeSocketClient, type RuntimeTrackSelection,
} from './lib/runtime';

type Mode = 'sketch' | 'build' | 'drive';
type DriveMode = 'manual' | 'ppo' | 'replay';
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
  // Position the full chassis on the start segment, rather than leaving its rear wheels beyond the road cap.
  const total = roadLength(sampleRoad(track, 240));
  const p = pointAt(track, total > 0 ? Math.min(2.6 / total, 0.1) : 0);
  return { x: p.x, y: p.y, z: p.z, heading: p.heading, speed: 0 };
}
const newId = () => typeof crypto !== 'undefined' && 'randomUUID' in crypto
  ? crypto.randomUUID() : String(Date.now()) + '-' + Math.random().toString(36).slice(2);

export default function App() {
  const [track, setTrack] = useState<TrackDocument>(initialTrack);
  const [mode, setMode] = useState<Mode>('sketch');
  const [selected, setSelected] = useState<string | null>(null);
  const [armed, setArmed] = useState<PropKind | null>(null);
  const [driveMode, setDriveMode] = useState<DriveMode>('ppo');
  const [pose, setPose] = useState<CarPose>(() => spawnPose(initialTrack()));
  const [replay, setReplay] = useState<Replay | null>(null);
  const [replayTime, setReplayTime] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [playbackSpeed, setPlaybackSpeed] = useState(1);
  const [runtimeCatalog, setRuntimeCatalog] = useState<RuntimeCatalog | null>(null);
  const [catalogError, setCatalogError] = useState('');
  const [modelId, setModelId] = useState('A-seed0');
  const [runtimeTrackId, setRuntimeTrackId] = useState('straight-sprint-v1');
  const [runtimeSeed, setRuntimeSeed] = useState(412);
  const [runtimeStatus, setRuntimeStatus] = useState<RuntimeSessionStatus | null>(null);
  const [runtimeError, setRuntimeError] = useState('');
  const [runtimeReplay, setRuntimeReplay] = useState<Replay | null>(null);
  const [runtimeBusy, setRuntimeBusy] = useState(false);
  const [runtimeTrackSelection, setRuntimeTrackSelection] = useState<RuntimeTrackSelection>({ kind: 'train' });
  const trackValidationRevision = useRef(0);
  const [trackValidationBusy, setTrackValidationBusy] = useState(false);
  const [toast, setToast] = useState('');
  const [help, setHelp] = useState(false);
  const [saved, setSaved] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null);
  const replayInput = useRef<HTMLInputElement>(null);
  const sessionRef = useRef<string | null>(null);
  const runtimeSocketRef = useRef<RuntimeSocketClient | null>(null);
  const latestRuntimeFrame = useRef<RuntimeSessionStatus['latest_frame']>(null);
  const runtimeResetPending = useRef(false);
  const runtimeClient = useMemo(() => new RuntimeClient(), []);
  const trackRef = useRef(track);
  const history = useRef<TrackDocument[]>([]);
  const redoStack = useRef<TrackDocument[]>([]);
  const selectedProp = track.props.find((p) => p.id === selected);
  const trackMeters = useMemo(() => roadLength(sampleRoad(track, 300)), [track]);
  const autoBridges = useMemo(() => getAutoBridges(track).length, [track]);
  const canDrive = track.points.length >= 2;
  const activeReplay = replay ?? runtimeReplay;
  const replayTrackId = activeReplay?.schema === 'f1rl_racing_sandbox_replay_v2' ? activeReplay.provenance.track_id : runtimeTrackId;
  const activeRuntimeTrack = runtimeTrackSelection.kind === 'custom'
    ? runtimeTrackSelection.validated ?? undefined
    : runtimeCatalog?.tracks.find((item) => item.track_id === (driveMode === 'replay' ? replayTrackId : runtimeTrackId));
  const runtimeTrackRequest = resolveRuntimeTrackRequest(runtimeTrackSelection, runtimeTrackId, driveMode);
  const sceneTrack = mode === 'drive' && activeRuntimeTrack ? activeRuntimeTrack.preview : track;
  const sceneRoadSamples = mode === 'drive' ? activeRuntimeTrack?.road_samples : undefined;
  const activePose = driveMode === 'replay' && activeReplay ? (() => {
    const p = replayAt(activeReplay, replayTime);
    return { x: p.position[0], y: p.position[1], z: p.position[2], heading: p.heading_rad, speed: p.speed_mps };
  })() : runtimeStatus?.latest_frame ? {
    x: runtimeStatus.latest_frame.position[0], y: runtimeStatus.latest_frame.position[1], z: runtimeStatus.latest_frame.position[2],
    heading: runtimeStatus.latest_frame.heading_rad, speed: runtimeStatus.latest_frame.speed_mps,
  } : activeRuntimeTrack ? {
    x: activeRuntimeTrack.road_samples[0]?.x ?? 0, y: activeRuntimeTrack.road_samples[0]?.y ?? 0,
    z: activeRuntimeTrack.road_samples[0]?.z ?? 0, heading: activeRuntimeTrack.road_samples[0]?.heading ?? 0, speed: 0,
  } : pose;

  useEffect(() => {
    try { localStorage.setItem(STORAGE_KEY, JSON.stringify(track)); setSaved(false); }
    catch { setSaved(false); }
  }, [track]);
  useEffect(() => {
    if (!toast) return;
    const timer = setTimeout(() => setToast(''), 3500);
    return () => clearTimeout(timer);
  }, [toast]);

  useEffect(() => {
    if (mode !== 'drive' || runtimeCatalog || catalogError) return;
    let active = true;
    void runtimeClient.catalog().then((catalog) => {
      if (!active) return;
      setRuntimeCatalog(catalog);
      if (!catalog.tracks.some((item) => item.track_id === runtimeTrackId)) setRuntimeTrackId(catalog.tracks[0]?.track_id ?? '');
    }).catch((error: unknown) => {
      if (active) setCatalogError(error instanceof Error ? error.message : 'Python runtime is unavailable');
    });
    return () => { active = false; };
  }, [mode, runtimeCatalog, catalogError, runtimeClient, runtimeTrackId]);

  const invalidateCustomTrackValidation = useCallback(() => {
    trackValidationRevision.current += 1;
    setRuntimeTrackSelection(invalidateRuntimeTrackSelection);
    setRuntimeError('');
  }, []);
  const changeTrack = useCallback((update: (prev: TrackDocument) => TrackDocument) => {
    const prev = trackRef.current;
    const next = update(prev);
    if (JSON.stringify(prev) === JSON.stringify(next)) return;
    history.current = [...history.current.slice(-59), prev];
    redoStack.current = [];
    trackRef.current = next;
    setTrack(next);
    invalidateCustomTrackValidation();
  }, [invalidateCustomTrackValidation]);
  const undo = () => {
    const last = history.current.pop();
    if (!last) return;
    redoStack.current.push(trackRef.current);
    trackRef.current = last;
    setTrack(last);
    invalidateCustomTrackValidation();
  };
  const redo = () => {
    const next = redoStack.current.pop();
    if (!next) return;
    history.current.push(trackRef.current);
    trackRef.current = next;
    setTrack(next);
    invalidateCustomTrackValidation();
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
      // Checkpoint gates always span the road from edge to edge; the editor
      // only needs the route location, never manual lateral alignment.
      const movingKind = track.props.find((p) => p.id === moveId)?.kind;
      updateProp(moveId, { t: target.t, offset: movingKind === 'checkpoint' ? 0 : offset });
    } else if (kind && PROP_KINDS.includes(kind)) {
      const id = newId();
      changeTrack((prev) => ({ ...prev,
        props: [...prev.props, { id, kind, t: target.t, offset: kind === 'checkpoint' ? 0 : offset, rotation: 0 }],
      }));
      setSelected(id);
      setArmed(null);
      setToast(kind[0].toUpperCase() + kind.slice(1) + ' placed');
    }
  }, [track, changeTrack]);
  useEffect(() => {
    if (mode !== 'build') return;
    const shortcuts = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.ctrlKey || e.altKey || e.metaKey) return;
      if (e.code === 'Escape') { setArmed(null); setSelected(null); return; }
      if (/^Digit[1-6]$/.test(e.code)) {
        const kind = PROP_KINDS[Number(e.code.slice(-1)) - 1];
        setArmed((current) => current === kind ? null : kind);
      }
    };
    window.addEventListener('keydown', shortcuts);
    return () => window.removeEventListener('keydown', shortcuts);
  }, [mode]);
  const startDrive = () => {
    if (!canDrive) return;
    setPose(spawnPose(track));
    setPlaying(false); setReplayTime(activeReplay?.frames[0]?.time_s ?? 0);
    setDriveMode(replay ? 'replay' : 'ppo');
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
      invalidateCustomTrackValidation();
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
      setRuntimeReplay(null);
      setReplayTime(next.frames[0].time_s);
      setPlaying(false); setDriveMode('replay');
      setToast('Replay loaded · development evidence only');
    } catch (error) { setToast(error instanceof Error ? error.message : 'Unable to open replay'); }
  };

  const runRuntimeSession = async () => {
    const trackRequest = runtimeTrackRequest;
    if (!runtimeCatalog || !Number.isInteger(runtimeSeed) || runtimeSeed < 0 || runtimeSeed > 0xffffffff || !trackRequest) return;
    setRuntimeBusy(true);
    setRuntimeError(''); setReplay(null); setRuntimeReplay(null); setPlaying(false); setRuntimeStatus(null);
    latestRuntimeFrame.current = null;
    runtimeResetPending.current = false;
    let createdSessionId: string | null = null;
    try {
      const created = trackRequest.kind === 'custom'
        ? await runtimeClient.createCustomSession(track, driveMode === 'manual' ? 'manual' : 'ppo', runtimeSeed, driveMode === 'ppo' ? modelId : undefined)
        : driveMode === 'manual'
          ? await runtimeClient.createManualSession(trackRequest.trackId, runtimeSeed)
          : await runtimeClient.createPpoSession(modelId, trackRequest.trackId, runtimeSeed);
      createdSessionId = created.session_id;
      sessionRef.current = created.session_id;
      latestRuntimeFrame.current = created.latest_frame;
      setRuntimeStatus(created);
      const socket = await runtimeClient.connect(created.session_id, (event) => {
        if (event.type === 'ready') {
          const status = event.data as unknown as RuntimeSessionStatus;
          latestRuntimeFrame.current = status.latest_frame;
          setRuntimeStatus(status);
        }
        if (event.type === 'frame') {
          const frame = event.data as unknown as NonNullable<RuntimeSessionStatus['latest_frame']>;
          const previous = latestRuntimeFrame.current;
          if (!isLivePoseFrameInOrder(previous, frame, runtimeResetPending.current)) return;
          if (runtimeResetPending.current && frame.tick === 0) runtimeResetPending.current = false;
          latestRuntimeFrame.current = frame;
          setRuntimeStatus((current) => current ? { ...current, status: current.status === 'ready' ? 'running' : current.status, latest_frame: frame, steps: frame.tick } : current);
        }
        if (event.type === 'progress') {
          const progress = event.data as { steps?: number; max_steps?: number; step_progress?: number };
          setRuntimeStatus((current) => current ? { ...current,
            steps: progress.steps ?? current.steps, max_steps: progress.max_steps ?? current.max_steps,
            step_progress: progress.step_progress ?? current.step_progress,
            status: current.status === 'ready' ? 'running' : current.status,
          } : current);
        }
        if (event.type === 'status') {
          const status = event.data.status as RuntimeSessionStatus['status'];
          const reset = status === 'running' && event.data.reset === true;
          if (reset) {
            runtimeResetPending.current = true;
            latestRuntimeFrame.current = null;
          }
          setRuntimeStatus((current) => current ? {
            ...current, status,
            ...(reset ? { steps: 0, step_progress: 0, latest_frame: null, episode_result: null } : {}),
          } : current);
        }
        if (event.type === 'terminal') {
          runtimeResetPending.current = false;
          const result = event.data as unknown as NonNullable<RuntimeSessionStatus['episode_result']>;
          const state = String(result.status) as RuntimeSessionStatus['status'];
          setRuntimeStatus((current) => current ? { ...current, status: state, episode_result: result } : current);
          void runtimeClient.replay(created.session_id).then((completedReplay) => {
            if (sessionRef.current !== created.session_id) return;
            setRuntimeReplay(completedReplay);
            setReplayTime(completedReplay.frames[0].time_s);
            // Keep the authoritative terminal result visible; Replay is available on demand.
            setPlaying(false);
            sessionRef.current = null;
            runtimeSocketRef.current?.disconnect();
            runtimeSocketRef.current = null;
            setRuntimeBusy(false);
          }).catch((error: unknown) => {
            if (error instanceof Error) setRuntimeError(error.message);
            sessionRef.current = null;
            runtimeSocketRef.current?.disconnect();
            runtimeSocketRef.current = null;
            setRuntimeBusy(false);
          });
        }
        if (event.type === 'error') setRuntimeError(String(event.data.message ?? event.data.code ?? 'Runtime protocol error'));
      }, (error) => {
        if (sessionRef.current !== created.session_id) return;
        runtimeResetPending.current = false;
        setRuntimeError(error.message);
        setRuntimeBusy(false);
        sessionRef.current = null;
        runtimeSocketRef.current?.disconnect();
        runtimeSocketRef.current = null;
        void runtimeClient.close(created.session_id).catch(() => undefined);
      });
      runtimeSocketRef.current = socket;
      socket.start();
    } catch (error) {
      if (error instanceof RuntimeApiError) setRuntimeError(`${error.code}: ${error.message}`);
      else setRuntimeError(error instanceof Error ? error.message : 'Episode failed to run');
      sessionRef.current = null;
      runtimeSocketRef.current?.disconnect();
      runtimeSocketRef.current = null;
      setRuntimeBusy(false);
      if (createdSessionId) void runtimeClient.close(createdSessionId).catch(() => undefined);
    }
  };

  const validateCurrentTrack = async () => {
    const requestRevision = ++trackValidationRevision.current;
    setRuntimeTrackSelection({ kind: 'custom', validated: null });
    setTrackValidationBusy(true); setRuntimeError('');
    try {
      const converted = await runtimeClient.validateCustomTrack(track);
      if (!isCurrentCustomTrackValidation(requestRevision, trackValidationRevision.current)) return;
      setRuntimeTrackSelection({ kind: 'custom', validated: converted });
      setToast(`Python validated ${converted.surface_count} road surfaces · development track`);
    } catch (error) {
      if (isCurrentCustomTrackValidation(requestRevision, trackValidationRevision.current)) {
        setRuntimeTrackSelection({ kind: 'custom', validated: null });
        setRuntimeError(error instanceof RuntimeApiError ? `${error.code}: ${error.message}` : error instanceof Error ? error.message : 'Track validation failed');
      }
    } finally { setTrackValidationBusy(false); }
  };

  const useRegisteredTrack = () => {
    trackValidationRevision.current += 1;
    setRuntimeTrackSelection({ kind: 'train' });
    setRuntimeError('');
  };

  const stopRuntimeSession = async () => {
    setPlaying(false);
    try {
      if (driveMode === 'manual') runtimeSocketRef.current?.action(0, 0, 0);
      runtimeSocketRef.current?.stop();
    }
    catch (error) { setRuntimeError(error instanceof Error ? error.message : 'Unable to stop runtime session'); }
  };

  const controlRuntime = (command: 'pause' | 'resume' | 'reset') => {
    try {
      runtimeSocketRef.current?.[command]();
      if (command === 'reset') {
        runtimeResetPending.current = true;
        latestRuntimeFrame.current = null;
        pressedKeys.current.clear();
        if (driveMode === 'manual') runtimeSocketRef.current?.action(0, 0, 0);
        setRuntimeReplay(null); setPlaying(false); setRuntimeStatus((current) => current ? {
          ...current, status: 'running', steps: 0, step_progress: 0, latest_frame: null, episode_result: null,
        } : current);
      }
    } catch (error) { setRuntimeError(error instanceof Error ? error.message : 'Runtime control failed'); }
  };

  const pressedKeys = useRef(new Set<string>());

  useEffect(() => {
    if (mode !== 'drive' || driveMode !== 'manual' || !runtimeBusy) return;
    const keys = pressedKeys.current;
    const action = () => {
      const command = manualActionFromKeys(keys);
      try { runtimeSocketRef.current?.action(command.steering, command.throttle, command.brake); }
      catch (error) { setRuntimeError(error instanceof Error ? error.message : 'Unable to send manual action'); }
    };
    const controlled = new Set(['KeyW','KeyA','KeyS','KeyD','ArrowUp','ArrowLeft','ArrowDown','ArrowRight','Space']);
    const keyDown = (event: KeyboardEvent) => {
      if (!controlled.has(event.code) || event.target instanceof HTMLInputElement || event.target instanceof HTMLSelectElement || event.target instanceof HTMLTextAreaElement) return;
      event.preventDefault();
      const before = keys.size;
      keys.add(event.code);
      if (keys.size !== before) action();
    };
    const keyUp = (event: KeyboardEvent) => {
      if (keys.delete(event.code)) action();
    };
    const release = () => { if (keys.size) { keys.clear(); action(); } };
    const visibility = () => { if (document.visibilityState !== 'visible') release(); };
    window.addEventListener('keydown', keyDown);
    window.addEventListener('keyup', keyUp);
    window.addEventListener('blur', release);
    document.addEventListener('visibilitychange', visibility);
    return () => {
      release();
      window.removeEventListener('keydown', keyDown);
      window.removeEventListener('keyup', keyUp);
      window.removeEventListener('blur', release);
      document.removeEventListener('visibilitychange', visibility);
    };
  }, [mode, driveMode, runtimeBusy]);

  useEffect(() => {
    if (!playing || !activeReplay || driveMode !== 'replay' || mode !== 'drive') return;
    let frame = 0, last = performance.now();
    const advance = (now: number) => {
      const dt = Math.min(0.1, (now - last) / 1000);
      last = now;
      setReplayTime((t) => {
        const end = activeReplay.frames[activeReplay.frames.length - 1].time_s;
        if (t + dt >= end) { setPlaying(false); return end; }
        return t + dt * playbackSpeed;
      });
      frame = requestAnimationFrame(advance);
    };
    frame = requestAnimationFrame(advance);
    return () => cancelAnimationFrame(frame);
  }, [playing, activeReplay, playbackSpeed, driveMode, mode]);

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
      {mode === 'sketch' ? <SketchEditor points={track.points} onChange={onSketch} onConvert={() => changeMode('build')}/> :
        <div className="world-stage">
          <Suspense fallback={<div className="scene-loading">Loading 3D world…</div>}>
            <RacingScene track={mode === 'build' ? track : sceneTrack} mode={mode} selectedId={selected} armed={armed}
              pose={mode === 'build' ? spawnPose(track) : activePose} roadSamples={mode === 'drive' ? sceneRoadSamples : undefined}
              liveFrame={mode === 'drive' && driveMode !== 'replay' ? runtimeStatus?.latest_frame ?? null : null}
              liveStatus={mode === 'drive' && driveMode !== 'replay' ? runtimeStatus?.status ?? null : null}
              onSelect={setSelected} onPlace={onPlace}/>
          </Suspense>
          {mode === 'build' && <>
            <div className="world-label"><span className="live-glow"/> 3D EDITOR <span>·</span> {Math.round(trackMeters)}M ROAD {autoBridges > 0 && <span>· {autoBridges} AUTO BRIDGE{autoBridges > 1 ? 'S' : ''}</span>}</div>
            <div className="build-instructions"><MousePointer2 size={16}/> {armed ? 'Click the road to place ' + armed : 'Drag an item from the hotbar onto the track · keys 1–6'}</div>
            {selectedProp && <div className="prop-inspector">
              <div className="inspector-top"><strong>{selectedProp.kind[0].toUpperCase() + selectedProp.kind.slice(1)}</strong><button aria-label="Close selection" onClick={() => setSelected(null)}><X size={16}/></button></div>
              {selectedProp.kind === 'checkpoint' || selectedProp.kind === 'booster'
                ? <p className="inspector-auto">Auto-fitted to road · drag to reposition</p>
                : <>
                    <span className="inspector-label">ROTATION</span>
                    <div className="inspector-actions">
                      <button title="Rotate counterclockwise" onClick={() => updateProp(selectedProp.id, { rotation: selectedProp.rotation - Math.PI / 12 })}><RotateCcw size={17}/></button>
                      <span>{Math.round(selectedProp.rotation * 180 / Math.PI)}°</span>
                      <button title="Rotate clockwise" onClick={() => updateProp(selectedProp.id, { rotation: selectedProp.rotation + Math.PI / 12 })}><RotateCw size={17}/></button>
                    </div>
                  </>}
              <button className="remove-prop" onClick={() => { changeTrack((p) => ({...p, props: p.props.filter((x) => x.id !== selectedProp.id)})); setSelected(null); }}><Trash2 size={15}/> Remove object</button>
            </div>}
          </>}
          {mode === 'drive' && <div className="drive-overlay">
            <div className="drive-mode-switch">
              <button className={driveMode === 'manual' ? 'selected' : ''} disabled={runtimeBusy && driveMode !== 'manual'} onClick={() => {setDriveMode('manual'); setPlaying(false);}}>Manual</button>
              <button className={driveMode === 'ppo' ? 'selected' : ''} disabled={runtimeBusy && driveMode !== 'ppo'} onClick={() => {setDriveMode('ppo'); setPlaying(false);}}>PPO</button>
              <button className={driveMode === 'replay' ? 'selected' : ''} disabled={!activeReplay || runtimeBusy} onClick={() => {setDriveMode('replay'); setPlaying(false);}}>Replay</button>
            </div>
            <div className="speed-readout"><span>SPEED</span><strong>{Math.round(activePose.speed * 3.6)}</strong><small>KM/H</small></div>
            <div className="drive-warning"><TriangleAlert size={14}/>{driveMode === 'manual'
              ? `Manual Python physics · ${runtimeTrackSelection.kind === 'custom' ? 'custom' : 'TRAIN'} development only`
              : driveMode === 'ppo'
                ? `Frozen PPO · ${runtimeTrackSelection.kind === 'custom' ? 'custom' : 'TRAIN'} development · scientific admission disabled`
                : 'Recorded frames · imported data is not independently verified'}</div>
            {driveMode !== 'replay' ? <>
              <div className="runtime-panel">
                {driveMode === 'ppo' && <label>FROZEN MODEL<select aria-label="Frozen PPO model" value={modelId} onChange={(event) => setModelId(event.target.value)} disabled={!runtimeCatalog || runtimeBusy}>
                    {(runtimeCatalog?.models ?? []).map((model) => <option key={model.model_id} value={model.model_id}>{model.model_id}</option>)}
                  </select></label>}
                {runtimeTrackSelection.kind === 'custom' ? <div className="custom-track-selection">
                  {runtimeTrackSelection.validated ? <>
                    <span>PYTHON VALIDATED CUSTOM TRACK</span><strong>{runtimeTrackSelection.validated.preview.name}</strong>
                    <small>{runtimeTrackSelection.validated.surface_count} road surfaces · {runtimeTrackSelection.validated.checkpoint_count} checkpoints · {runtimeTrackSelection.validated.auto_bridge_count} auto overpasses</small>
                    {driveMode === 'ppo' && !runtimeTrackSelection.validated.ppo_compatible && <small className="runtime-error">Frozen PPO observation/action contract is incompatible.</small>}
                  </> : <>
                    <span>CUSTOM TRACK SELECTED</span>
                    <small className="runtime-error">{trackValidationBusy ? 'Validating current map…' : 'Validate the current map before running.'}</small>
                    <button className="runtime-validate-track" disabled={!runtimeCatalog || trackValidationBusy || runtimeBusy} onClick={() => void validateCurrentTrack()}>
                      {trackValidationBusy ? <><LoaderCircle size={14}/> VALIDATING IN PYTHON…</> : <><Check size={14}/> VALIDATE & USE CURRENT MAP</>}
                    </button>
                  </>}
                  <button disabled={runtimeBusy} onClick={useRegisteredTrack}>Use registered TRAIN track</button>
                </div> : <>
                  <label>TRAIN TRACK<select aria-label="TRAIN track" value={runtimeTrackId} onChange={(event) => setRuntimeTrackId(event.target.value)} disabled={!runtimeCatalog || runtimeBusy}>
                    {(runtimeCatalog?.tracks ?? []).map((item) => <option key={item.track_id} value={item.track_id}>{item.track_id}</option>)}
                  </select></label>
                  <button className="runtime-validate-track" disabled={!runtimeCatalog || trackValidationBusy || runtimeBusy} onClick={() => void validateCurrentTrack()}>
                    {trackValidationBusy ? <><LoaderCircle size={14}/> VALIDATING IN PYTHON…</> : <><Check size={14}/> VALIDATE & USE CURRENT MAP</>}
                  </button>
                </>}
                <label>SEED<input aria-label="Episode seed" type="number" min="0" max="4294967295" step="1" value={runtimeSeed} onChange={(event) => setRuntimeSeed(Number(event.target.value))} disabled={runtimeBusy}/></label>
                {catalogError ? <p className="runtime-error"><TriangleAlert size={15}/><span>{catalogError}. Start the Python runtime to run episodes.</span><button onClick={() => setCatalogError('')}>Retry connection</button></p> : !runtimeCatalog ?
                  <p className="runtime-loading"><LoaderCircle size={15}/> Connecting to local Python runtime…</p> : null}
                {runtimeError && <p className="runtime-error"><TriangleAlert size={15}/>{runtimeError}</p>}
                {runtimeStatus && <>
                  <div className="runtime-progress"><span>{runtimeStatus.status.toUpperCase()} · {runtimeStatus.steps}/{runtimeStatus.max_steps} STEPS</span><progress max="1" value={runtimeStatus.step_progress}/></div>
                  {runtimeStatus.latest_frame && <div className="runtime-telemetry">
                    <span>POSITION M <b>{runtimeStatus.latest_frame.position.map((value) => value.toFixed(1)).join(', ')}</b></span>
                    <span>HEADING <b>{(runtimeStatus.latest_frame.heading_rad * 180 / Math.PI).toFixed(1)}°</b></span>
                    <span>VELOCITY M/S <b>{runtimeStatus.latest_frame.velocity.map((value) => value.toFixed(1)).join(', ')}</b></span>
                    <span>ACCEL M/S² <b>{runtimeStatus.latest_frame.acceleration_mps2.map((value) => value.toFixed(1)).join(', ')}</b></span>
                    <span>STEER <b>{runtimeStatus.latest_frame.action.steering.toFixed(2)}</b></span>
                    <span>THROTTLE <b>{runtimeStatus.latest_frame.action.throttle.toFixed(2)}</b></span>
                    <span>BRAKE <b>{runtimeStatus.latest_frame.action.brake.toFixed(2)}</b></span>
                    <span>CHECKPOINTS <b>{runtimeStatus.latest_frame.completed_checkpoint_ids.length}/{runtimeStatus.latest_frame.checkpoint_count}</b></span>
                    <span>COLLISIONS <b>{runtimeStatus.latest_frame.collision_count}</b></span>
                    <span>{runtimeStatus.latest_frame.grounded ? 'GROUNDED' : 'AIRBORNE'}</span>
                  </div>}
                  <RuntimeOutcomeMessage status={runtimeStatus} />
                </>}
                <div className="runtime-actions">
                  <button className="runtime-run" onClick={() => void runRuntimeSession()} disabled={!runtimeCatalog || runtimeBusy || !Number.isInteger(runtimeSeed) || runtimeSeed < 0 || runtimeSeed > 0xffffffff || !runtimeTrackRequest}>
                    {runtimeBusy ? <><LoaderCircle size={16}/> {runtimeStatus?.status === 'paused' ? 'PAUSED' : driveMode === 'manual' ? 'MANUAL PYTHON SIMULATION' : 'RUNNING PPO INFERENCE'}</> : <><Play size={16}/> {driveMode === 'manual' ? 'START MANUAL DRIVE' : 'RUN EPISODE'}</>}
                  </button>
                  {runtimeBusy && <button className="runtime-stop" onClick={() => void stopRuntimeSession()}>Stop</button>}
                  <button className="runtime-import" onClick={() => replayInput.current?.click()}><Upload size={15}/> Import replay</button>
                </div>
                {runtimeBusy && <div className="runtime-live-controls">
                  <button onClick={() => controlRuntime(runtimeStatus?.status === 'paused' ? 'resume' : 'pause')}>{runtimeStatus?.status === 'paused' ? <Play size={14}/> : <Pause size={14}/>} {runtimeStatus?.status === 'paused' ? 'Resume' : 'Pause'}</button>
                  <button onClick={() => controlRuntime('reset')}><RotateCcw size={14}/> Reset</button>
                  <label>SIM SPEED<select aria-label="Simulation speed" value={runtimeStatus?.speed_multiplier ?? 1} onChange={(event) => {
                    const multiplier = Number(event.target.value);
                    try { runtimeSocketRef.current?.speed(multiplier); setRuntimeStatus((current) => current ? { ...current, speed_multiplier: multiplier } : current); }
                    catch (error) { setRuntimeError(error instanceof Error ? error.message : 'Unable to set simulation speed'); }
                  }}><option value="0.25">0.25×</option><option value="0.5">0.5×</option><option value="1">1×</option><option value="2">2×</option><option value="4">4×</option><option value="8">8×</option><option value="16">16×</option></select></label>
                </div>}
              </div>
            </> : activeReplay ? <div className="replay-controls">
              <button aria-label={playing ? 'Pause replay' : 'Play replay'} onClick={() => {if (replayTime >= activeReplay.frames[activeReplay.frames.length - 1].time_s) setReplayTime(activeReplay.frames[0].time_s); setPlaying(!playing);}}>{playing ? <Pause size={19}/> : <Play size={19}/>}</button>
              <input aria-label="Replay timeline" type="range" min={activeReplay.frames[0].time_s} max={activeReplay.frames[activeReplay.frames.length - 1].time_s} step="0.01" value={replayTime} onChange={(event) => {setPlaying(false);setReplayTime(Number(event.target.value));}}/>
              <span>{replayTime.toFixed(1)}s</span><span className="replay-name">{replayModelId(activeReplay)}</span>
              <select aria-label="Replay speed" value={playbackSpeed} onChange={(event) => setPlaybackSpeed(Number(event.target.value))}><option value="0.5">0.5×</option><option value="1">1×</option><option value="2">2×</option><option value="4">4×</option></select>
              <button aria-label="Restart replay" onClick={() => {setPlaying(false);setReplayTime(activeReplay.frames[0].time_s);}}><RotateCcw size={16}/></button>
              <button aria-label="Export replay JSON" onClick={() => downloadJSON('f1rl-runtime-replay.json', activeReplay)}><Download size={16}/></button>
            </div> : <div className="replay-empty"><span>No replay loaded</span><button onClick={() => replayInput.current?.click()}><Upload size={16}/> Import replay</button></div>}
            <div className="drive-bottom"><button onClick={() => {
              if (runtimeBusy) controlRuntime('reset');
              else if (driveMode === 'replay') {setPlaying(false);setReplayTime(activeReplay?.frames[0]?.time_s ?? 0);}
              else void runRuntimeSession();
            }}><RotateCcw size={17}/> {runtimeBusy ? 'Reset session' : 'Restart'}</button><button className="leave-drive" onClick={() => {if (runtimeBusy) void stopRuntimeSession(); changeMode('build');}}><ArrowLeft size={17}/> Back to build</button></div>
          </div>}
        </div>}
      {mode === 'build' && <div className="build-hotbar" role="toolbar" aria-label="Build hotbar">
        {assets.map(({kind, title, description, icon: Icon, color}, index) => (
          <button key={kind} draggable
            className={armed === kind ? 'hotbar-slot selected' : 'hotbar-slot'}
            aria-label={`Select ${title} (slot ${index + 1})`}
            aria-pressed={armed === kind}
            onDragStart={(e) => {e.dataTransfer.setData('application/x-f1rl-prop', kind);e.dataTransfer.effectAllowed='copy';setArmed(kind);}}
            onDragEnd={() => setArmed(null)} onClick={() => setArmed(armed === kind ? null : kind)}>
            <span className="hotbar-number">{index + 1}</span>
            <span className={'hotbar-model ' + color}><Icon size={29} strokeWidth={1.75}/></span>
            <span className="hotbar-tooltip" role="tooltip"><strong>{title}</strong><small>{description}</small></span>
          </button>
        ))}
      </div>}
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
