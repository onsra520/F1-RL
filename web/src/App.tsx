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
import { createT3V4JumpTrack, createTrack, getAutoBridges, nearestRoad, parseTrack, pointAt, PROP_KINDS, roadLength, sampleRoad,
         type Point2, type PropKind, type TrackDocument, type TrackProp } from './lib/track';
import { parseReplay, replayAt, replayModelId, type Replay } from './lib/replay';
import { manualActionFromKeys } from './lib/manual';
import { isLivePoseFrameInOrder } from './lib/livePose';
import { AUTO_CONFIG_STORAGE, loadAutoConfig, type AutoDriveConfig } from './lib/autoDriveConfig';
import AutoDriveSettings from './components/AutoDriveSettings';
import RuntimeOutcomeMessage from './RuntimeOutcomeMessage';
import {
  customTrackSelectionAfterMapEdit, isCurrentCustomTrackValidation, resolveRuntimeTrackRequest,
  matchesValidatedCustomTrack, shouldKeepReplayOnDrive, shouldValidateCustomTrackOnDrive,
  RuntimeApiError, RuntimeClient, type RuntimeCatalog,
  type RuntimeSessionStatus, type RuntimeSocketClient, type RuntimeTrackSelection,
} from './lib/runtime';

type Mode = 'sketch' | 'build' | 'drive';
type DriveMode = 'auto' | 'manual' | 'ppo' | 'hybrid' | 'replay';
const STORAGE_KEY = 'f1rl.racing-sandbox.draft.v1';
const REPLAY_TRACK_UNAVAILABLE: TrackDocument = {
  schema: 'f1rl_racing_sandbox_draft_v1', id: 'replay-track-unavailable', name: 'Track unavailable',
  width: 4, points: [], props: [],
};
const assets: { kind: PropKind; title: string; description: string; icon: typeof Box; color: string }[] = [
  { kind: 'bridge', title: 'Bridge', description: 'Raised road section', icon: Layers3, color: 'steel' },
  { kind: 'ramp', title: 'Ramp', description: 'Launch / incline', icon: Mountain, color: 'orange' },
  { kind: 'barrier', title: 'Barrier', description: 'Road block', icon: Box, color: 'red' },
  { kind: 'cone', title: 'Cone', description: 'Small obstacle', icon: TriangleAlert, color: 'orange' },
  { kind: 'booster', title: 'Booster', description: 'Speed pad', icon: Waves, color: 'lime' },
  { kind: 'checkpoint', title: 'Checkpoint', description: 'Route marker', icon: Flag, color: 'blue' },
  { kind: 'jump', title: 'V4 Certified Jump', description: '3.7 m real gap + launch and receiving ramp (development)', icon: Mountain, color: 'lime' },
  { kind: 'gap', title: 'Jump Gap', description: 'Experimental Physics V4: road void (2–6 m)', icon: TriangleAlert, color: 'orange' },
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
  const [driveMode, setDriveMode] = useState<DriveMode>('auto');
  const [autoConfig, setAutoConfig] = useState<AutoDriveConfig>(loadAutoConfig);
  const [autoConfigValid, setAutoConfigValid] = useState(true);
  const [experimentalPhysicsV4, setExperimentalPhysicsV4] = useState(false);
  const [pose, setPose] = useState<CarPose>(() => spawnPose(initialTrack()));
  const [replay, setReplay] = useState<Replay | null>(null);
  const [replayTime, setReplayTime] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [playbackSpeed, setPlaybackSpeed] = useState(1);
  const [runtimeCatalog, setRuntimeCatalog] = useState<RuntimeCatalog | null>(null);
  const [catalogError, setCatalogError] = useState('');
  const [modelId, setModelId] = useState('A-seed0');
  const [hybridModelId, setHybridModelId] = useState('wp7-calibration');
  const [runtimeTrackId, setRuntimeTrackId] = useState('straight-sprint-v1');
  const [runtimeSeed, setRuntimeSeed] = useState(412);
  const [runtimeStatus, setRuntimeStatus] = useState<RuntimeSessionStatus | null>(null);
  const [runtimeError, setRuntimeError] = useState('');
  const [runtimeReplay, setRuntimeReplay] = useState<Replay | null>(null);
  const [runtimeBusy, setRuntimeBusy] = useState(false);
  useEffect(() => {
    if (!runtimeCatalog || runtimeBusy) return;
    setHybridModelId((current) => runtimeCatalog.exploratoryModels.some((model) => model.model_id === current)
      ? current : runtimeCatalog.exploratoryModels[0]?.model_id ?? '');
  }, [runtimeCatalog, runtimeBusy]);
  const [runtimeConnected, setRuntimeConnected] = useState(false);
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
  const runtimeTerminalPending = useRef(false);
  const autoStartPending = useRef(false);
  const runtimeStartEpoch = useRef(0);
  const runtimeClient = useMemo(() => new RuntimeClient(), []);
  const trackRef = useRef(track);
  const history = useRef<TrackDocument[]>([]);
  const redoStack = useRef<TrackDocument[]>([]);
  const selectedProp = track.props.find((p) => p.id === selected);
  const trackMeters = useMemo(() => roadLength(sampleRoad(track, 300)), [track]);
  const autoBridges = useMemo(() => getAutoBridges(track).length, [track]);
  const canDrive = track.points.length >= 2;
  const activeReplay = replay ?? runtimeReplay;
  const currentReplayFrame = driveMode === 'replay' && activeReplay
    ? replayAt(activeReplay, replayTime) : null;
  const activeRuntimeTrack = driveMode === 'replay'
    ? activeReplay?.schema === 'f1rl_racing_sandbox_replay_v2'
      ? activeReplay.track_preview ?? runtimeCatalog?.tracks.find((item) => item.track_id === activeReplay.provenance.track_id)
      : undefined
    : runtimeTrackSelection.kind === 'custom'
      ? runtimeTrackSelection.validated ?? undefined
      : runtimeCatalog?.tracks.find((item) => item.track_id === runtimeTrackId);
  const runtimeTrackRequest = resolveRuntimeTrackRequest(runtimeTrackSelection, runtimeTrackId, driveMode);
  const replayTrackUnavailable = driveMode === 'replay' && activeReplay !== null && !activeRuntimeTrack;
  const sceneTrack = mode === 'drive'
    ? activeRuntimeTrack?.preview ?? (replayTrackUnavailable ? REPLAY_TRACK_UNAVAILABLE : track)
    : track;
  const sceneRoadSamples = mode === 'drive' ? activeRuntimeTrack?.road_samples : undefined;
  const activePose = currentReplayFrame ? (() => {
    const p = currentReplayFrame;
    return { x: p.position[0], y: p.position[1], z: p.position[2], heading: p.heading_rad, speed: p.speed_mps,
      ...(p.orientation_wxyz ? { orientation_wxyz: p.orientation_wxyz } : {}) };
  })() : runtimeStatus?.latest_frame ? {
    x: runtimeStatus.latest_frame.position[0], y: runtimeStatus.latest_frame.position[1], z: runtimeStatus.latest_frame.position[2],
    heading: runtimeStatus.latest_frame.heading_rad, speed: runtimeStatus.latest_frame.speed_mps,
    orientation_wxyz: runtimeStatus.latest_frame.orientation_wxyz,
  } : activeRuntimeTrack ? {
    x: activeRuntimeTrack.road_samples[0]?.x ?? 0, y: activeRuntimeTrack.road_samples[0]?.y ?? 0,
    z: activeRuntimeTrack.road_samples[0]?.z ?? 0, heading: activeRuntimeTrack.road_samples[0]?.heading ?? 0, speed: 0,
  } : pose;

  useEffect(() => {
    try { localStorage.setItem(AUTO_CONFIG_STORAGE, JSON.stringify(autoConfig)); } catch { /* storage optional */ }
  }, [autoConfig]);
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

  const cancelRuntime = useCallback(() => {
    runtimeStartEpoch.current += 1;
    setRuntimeConnected(false);
    const sessionId = sessionRef.current;
    sessionRef.current = null;
    runtimeSocketRef.current?.disconnect();
    runtimeSocketRef.current = null;
    if (sessionId) void runtimeClient.close(sessionId).catch(() => undefined);
    setRuntimeBusy(false);
  }, [runtimeClient]);

  const invalidateCustomTrackValidation = useCallback(() => {
    trackValidationRevision.current += 1;
    setTrackValidationBusy(false);
    setRuntimeTrackSelection(customTrackSelectionAfterMapEdit());
    cancelRuntime();
    latestRuntimeFrame.current = null;
    setRuntimeStatus(null);
    setPose(spawnPose(trackRef.current));
    setRuntimeError('');
  }, [cancelRuntime]);
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
      updateProp(moveId, (movingKind === 'gap' || movingKind === 'jump') ? { t: target.t } : { t: target.t, offset: movingKind === 'checkpoint' ? 0 : offset });
    } else if (kind && PROP_KINDS.includes(kind)) {
      const id = newId();
      changeTrack((prev) => ({ ...prev,
        props: [...prev.props, { id, kind, t: target.t, offset: kind === 'checkpoint' ? 0 : kind === 'gap' ? 3 : kind === 'jump' ? 3.7 : offset, rotation: 0 }],
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
      if (/^Digit[1-8]$/.test(e.code)) {
        const kind = PROP_KINDS[Number(e.code.slice(-1)) - 1];
        setArmed((current) => current === kind ? null : kind);
      }
    };
    window.addEventListener('keydown', shortcuts);
    return () => window.removeEventListener('keydown', shortcuts);
  }, [mode]);
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
      autoStartPending.current = false;
      cancelRuntime();
      setReplay(next);
      setRuntimeReplay(null);
      setReplayTime(next.frames[0].time_s);
      setPlaying(false); setDriveMode('replay');
      setToast('Replay loaded · development evidence only');
    } catch (error) { setToast(error instanceof Error ? error.message : 'Unable to open replay'); }
  };

  const runRuntimeSession = async () => {
    const trackRequest = runtimeTrackRequest;
    const validatedCustomTrack = runtimeTrackSelection.kind === 'custom' ? runtimeTrackSelection.validated : null;
    if (!runtimeCatalog || ((driveMode === 'auto' || driveMode === 'hybrid') && !autoConfigValid) || (driveMode === 'hybrid' && !runtimeCatalog.exploratoryModels.some((model) => model.model_id === hybridModelId)) || !Number.isInteger(runtimeSeed) || runtimeSeed < 0 || runtimeSeed > 0xffffffff || !trackRequest) return;
    const startEpoch = ++runtimeStartEpoch.current;
    const traceId = crypto.randomUUID();
    setRuntimeBusy(true);
    setRuntimeConnected(false);
    setRuntimeError(''); setReplay(null); setRuntimeReplay(null); setPlaying(false); setRuntimeStatus(null);
    latestRuntimeFrame.current = null;
    runtimeResetPending.current = false;
    runtimeTerminalPending.current = false;
    let createdSessionId: string | null = null;
    try {
      const created = trackRequest.kind === 'custom'
        ? await runtimeClient.createCustomSession(track, driveMode === 'manual' ? 'manual' : driveMode === 'auto' ? 'auto' : driveMode === 'hybrid' ? 'hybrid' : 'ppo', runtimeSeed, driveMode === 'ppo' ? modelId : driveMode === 'hybrid' ? hybridModelId : undefined, traceId, driveMode === 'auto' || driveMode === 'hybrid' ? autoConfig : undefined, driveMode === 'auto' && experimentalPhysicsV4)
        : driveMode === 'manual'
          ? await runtimeClient.createManualSession(trackRequest.trackId, runtimeSeed)
          : driveMode === 'auto'
            ? await runtimeClient.createAutoSession(trackRequest.trackId, runtimeSeed, autoConfig)
            : driveMode === 'hybrid'
              ? await runtimeClient.createHybridSession(hybridModelId, trackRequest.trackId, runtimeSeed, autoConfig)
            : await runtimeClient.createPpoSession(modelId, trackRequest.trackId, runtimeSeed);
      createdSessionId = created.session_id;
      if (runtimeStartEpoch.current !== startEpoch) {
        void runtimeClient.close(created.session_id).catch(() => undefined);
        return;
      }
      if (trackRequest.kind === 'custom' &&
          (!validatedCustomTrack || !matchesValidatedCustomTrack(validatedCustomTrack, created.track_preview))) {
        throw new RuntimeApiError('Python session track does not match the validated map', 'track_mismatch', 409, traceId);
      }
      sessionRef.current = created.session_id;
      latestRuntimeFrame.current = created.latest_frame;
      setRuntimeStatus(created);
      const socket = await runtimeClient.connect(created.session_id, (event) => {
        if (runtimeStartEpoch.current !== startEpoch) return;
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
          const progress = event.data as { steps?: number; max_steps?: number; step_progress?: number; hybrid?: RuntimeSessionStatus['hybrid'] };
          setRuntimeStatus((current) => current ? { ...current,
            steps: progress.steps ?? current.steps, max_steps: progress.max_steps ?? current.max_steps,
            step_progress: progress.step_progress ?? current.step_progress,
            hybrid: progress.hybrid ?? current.hybrid,
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
            ...(reset ? { steps: 0, step_progress: 0, latest_frame: null, episode_result: null, collision_terminal: undefined } : {}),
          } : current);
        }
        if (event.type === 'terminal') {
          runtimeTerminalPending.current = true;
          runtimeResetPending.current = false;
          const result = event.data as unknown as NonNullable<RuntimeSessionStatus['episode_result']>;
          const state = String(result.status) as RuntimeSessionStatus['status'];
          setRuntimeStatus((current) => current ? { ...current, status: state, episode_result: result } : current);
          void runtimeClient.replay(created.session_id).then((completedReplay) => {
            if (runtimeStartEpoch.current !== startEpoch || sessionRef.current !== created.session_id) return;
            const replayWithTrack = completedReplay.schema === 'f1rl_racing_sandbox_replay_v2' && created.track_preview
              ? { ...completedReplay, track_preview: {
                  track_id: created.track_preview.track_id,
                  track_sha256: created.track_preview.track_sha256,
                  preview: created.track_preview.preview,
                  road_samples: created.track_preview.road_samples,
                } }
              : completedReplay;
            setRuntimeReplay(replayWithTrack);
            if (completedReplay.schema === 'f1rl_racing_sandbox_replay_v2' && completedReplay.collision_terminal) {
              setRuntimeStatus((current) => current ? { ...current, collision_terminal: completedReplay.collision_terminal } : current);
            }
            setReplayTime(completedReplay.frames[0].time_s);
            // Keep the authoritative terminal result visible; Replay is available on demand.
            setPlaying(false);
            sessionRef.current = null;
            runtimeSocketRef.current?.disconnect();
            runtimeSocketRef.current = null;
            setRuntimeBusy(false);
          }).catch((error: unknown) => {
            if (runtimeStartEpoch.current !== startEpoch || sessionRef.current !== created.session_id) return;
            const expectedPreflight = error instanceof RuntimeApiError && error.code === 'replay_unavailable' &&
              result.steps === 0 && typeof result.reason === 'string' &&
              (result.reason.includes('Auto Drive cannot certify a road-width-safe corridor') ||
               result.reason.includes('Hybrid centerline does not avoid obstacles'));
            if (error instanceof Error && !expectedPreflight) setRuntimeError(error.message);
            sessionRef.current = null;
            runtimeSocketRef.current?.disconnect();
            runtimeSocketRef.current = null;
            setRuntimeBusy(false);
          });
        }
        if (event.type === 'error') setRuntimeError(String(event.data.message ?? event.data.code ?? 'Runtime protocol error'));
      }, (error) => {
        if (runtimeStartEpoch.current !== startEpoch || sessionRef.current !== created.session_id) return;
        runtimeResetPending.current = false;
        setRuntimeError(error.message);
        setRuntimeBusy(false);
        sessionRef.current = null;
        runtimeSocketRef.current?.disconnect();
        runtimeSocketRef.current = null;
        void runtimeClient.close(created.session_id).catch(() => undefined);
      });
      if (runtimeStartEpoch.current !== startEpoch) {
        socket.disconnect();
        return;
      }
      runtimeSocketRef.current = socket;
      socket.start();
      setRuntimeConnected(true);
    } catch (error) {
      if (runtimeStartEpoch.current !== startEpoch) {
        return;
      }
      if (error instanceof RuntimeApiError && error.code === 'track_mismatch' && error.traceId) {
        console.error('runtime session invariant failed', {
          trace_id: error.traceId, module: 'runtime', operation: 'run_session',
          class: 'App.runRuntimeSession', outcome: 'error', error_code: error.code, cause: error.message,
        });
        setRuntimeError(`${error.code}: Python track did not match validation · ${error.traceId}`);
      } else if (error instanceof RuntimeApiError) setRuntimeError(`${error.code}: ${error.message}`);
      else setRuntimeError(error instanceof Error ? error.message : 'Episode failed to run');
      sessionRef.current = null;
      runtimeSocketRef.current?.disconnect();
      runtimeSocketRef.current = null;
      setRuntimeBusy(false);
      if (createdSessionId) void runtimeClient.close(createdSessionId).catch(() => undefined);
    }
  };

  useEffect(() => {
    if (!autoStartPending.current || mode !== 'drive' || driveMode !== 'auto' ||
        !runtimeCatalog || trackValidationBusy || runtimeBusy || !runtimeTrackRequest ||
        !autoConfigValid || !Number.isInteger(runtimeSeed) || runtimeSeed < 0 || runtimeSeed > 0xffffffff) return;
    autoStartPending.current = false;
    void runRuntimeSession();
  }, [mode, driveMode, runtimeCatalog, trackValidationBusy, runtimeBusy,
      runtimeTrackSelection, runtimeTrackId, autoConfigValid, runtimeSeed]);

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
        autoStartPending.current = false;
        setRuntimeTrackSelection({ kind: 'custom', validated: null });
        setRuntimeError(error instanceof RuntimeApiError ? `${error.code}: ${error.message}` : error instanceof Error ? error.message : 'Track validation failed');
      }
    } finally {
      if (isCurrentCustomTrackValidation(requestRevision, trackValidationRevision.current)) setTrackValidationBusy(false);
    }
  };

  const useRegisteredTrack = () => {
    trackValidationRevision.current += 1;
    setTrackValidationBusy(false);
    setRuntimeTrackSelection({ kind: 'train' });
    setRuntimeError('');
  };

  const startDrive = (keepReplay: boolean) => {
    if (!canDrive) return;
    setPose(spawnPose(track));
    setPlaying(false); setReplayTime(activeReplay?.frames[0]?.time_s ?? 0);
    setDriveMode(keepReplay && activeReplay ? 'replay' : 'auto');
    setMode('drive');
    setArmed(null); setSelected(null);
  };
  const changeMode = (target: Mode) => {
    if (target === 'drive') {
      const keepReplay = shouldKeepReplayOnDrive(mode, driveMode, Boolean(activeReplay));
      autoStartPending.current = mode !== 'drive' && !keepReplay;
      startDrive(keepReplay);
      if (shouldValidateCustomTrackOnDrive(mode, keepReplay)) void validateCurrentTrack();
      return;
    }
    autoStartPending.current = false;
    if (mode === 'drive') cancelRuntime();
    setMode(target); setArmed(null); setSelected(null); setPlaying(false);
  };

  const stopRuntimeSession = async () => {
    setPlaying(false);
    if (!runtimeSocketRef.current) {
      autoStartPending.current = false;
      cancelRuntime();
      setRuntimeStatus(null);
      return;
    }
    try {
      if (driveMode === 'manual') runtimeSocketRef.current?.action(0, 0, 0);
      runtimeSocketRef.current?.stop();
    }
    catch (error) { setRuntimeError(error instanceof Error ? error.message : 'Unable to stop runtime session'); }
  };

  const controlRuntime = (command: 'pause' | 'resume' | 'reset') => {
    if (runtimeTerminalPending.current) return;
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
              simulationSpeed={mode === 'drive' && driveMode !== 'replay' ? runtimeStatus?.speed_multiplier ?? 1 : 1}
              cameraEpoch={mode === 'build' ? 'build' : driveMode === 'replay' ? 'replay' : 'live'}
              onSelect={setSelected} onPlace={onPlace}/>
          </Suspense>
          {mode === 'build' && runtimeTrackSelection.kind === 'custom'
            && !runtimeTrackSelection.validated && runtimeError &&
            <div className="build-validation-error" role="alert"><TriangleAlert size={15}/><span>Python Physics rejected this map · {runtimeError}</span></div>}
          {mode === 'build' && <>
            <div className="world-label"><span className="live-glow"/> 3D EDITOR <span>·</span> {Math.round(trackMeters)}M ROAD {autoBridges > 0 && <span>· {autoBridges} AUTO BRIDGE{autoBridges > 1 ? 'S' : ''}</span>}</div>
            <button className="runtime-validate-track" disabled={runtimeBusy}
              aria-label="Load T3 V4 jump test map"
              onClick={() => { changeTrack(() => createT3V4JumpTrack()); setExperimentalPhysicsV4(true);
                setSelected(null); setArmed(null); setToast('Development V4 jump map loaded. Validate in Python before Drive.'); }}>
              LOAD V4 JUMP TEST MAP (3.7 M)
            </button>
            <div className="build-instructions"><MousePointer2 size={16}/> {armed ? 'Click the road to place ' + armed : 'Drag an item from the hotbar onto the track · keys 1–6'}</div>
            {selectedProp && <div className="prop-inspector">
              <div className="inspector-top"><strong>{selectedProp.kind[0].toUpperCase() + selectedProp.kind.slice(1)}</strong><button aria-label="Close selection" onClick={() => setSelected(null)}><X size={16}/></button></div>
              {selectedProp.kind === 'checkpoint' || selectedProp.kind === 'booster' || selectedProp.kind === 'gap' || selectedProp.kind === 'jump'
                ? selectedProp.kind === 'gap'
                  ? <label>Gap length (2-6 metres)
                    <input aria-label="Gap length metres" type="number" min="2" max="6" step="0.25"
                      value={selectedProp.offset} onChange={e => updateProp(selectedProp.id, { offset: Number(e.target.value) })}/>
                    <small>Physics V4 development only; require safe takeoff and landing</small>
                  </label>
                  : selectedProp.kind === 'jump'
                    ? <p className="inspector-auto">Physics V4 development only: fixed 3.7 m void,
                      16 m launch, C1 receiving apron and absorption trough.
                      Use a straight road of at least 10 m width, adequate approach/runout and explicit V4.
                      No arbitrary jump size or Physics override.</p>
                    : <p className="inspector-auto">Auto-fitted to road - drag to reposition</p>
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
              <button className={driveMode === 'auto' ? 'selected' : ''} disabled={runtimeBusy && driveMode !== 'auto'} onClick={() => {setDriveMode('auto'); setPlaying(false);}}>Auto Drive</button>
              <button className={driveMode === 'manual' ? 'selected' : ''} disabled={runtimeBusy && driveMode !== 'manual'} onClick={() => {autoStartPending.current = false; setDriveMode('manual'); setPlaying(false);}}>Manual</button>
              <button className={driveMode === 'ppo' ? 'selected' : ''} disabled={runtimeBusy && driveMode !== 'ppo'} onClick={() => {autoStartPending.current = false; setDriveMode('ppo'); setPlaying(false);}}>PPO · development</button>
              <button className={driveMode === 'hybrid' ? 'selected' : ''} disabled={(runtimeBusy && driveMode !== 'hybrid') || !runtimeCatalog?.exploratoryModels.length} title={!runtimeCatalog?.exploratoryModels.length ? 'Archived exploratory checkpoints are unavailable on this runtime' : 'Hybrid output is development only'} onClick={() => {autoStartPending.current = false; setDriveMode('hybrid'); setPlaying(false);}}>Hybrid · development</button>
              <button className={driveMode === 'replay' ? 'selected' : ''} disabled={!activeReplay || runtimeBusy} onClick={() => {autoStartPending.current = false; setDriveMode('replay'); setPlaying(false);}}>Replay</button>
            </div>
            <div className="speed-readout"><span>SPEED</span><strong>{Math.round(activePose.speed * 3.6)}</strong><small>KM/H</small></div>
            {currentReplayFrame?.grounded !== undefined && <div className="replay-ground-state" aria-label="Replay Physics ground state">
              {currentReplayFrame.grounded ? 'RECORDED GROUNDED' : 'RECORDED AIRBORNE'}
            </div>}
            <div className="drive-warning"><TriangleAlert size={14}/>{driveMode === 'auto'
              ? `Auto Drive · Python Physics · ${runtimeTrackSelection.kind === 'custom' ? 'custom' : 'TRAIN'} development`
              : driveMode === 'manual'
              ? `Manual Python physics · ${runtimeTrackSelection.kind === 'custom' ? 'custom' : 'TRAIN'} development only`
              : driveMode === 'hybrid'
              ? runtimeStatus?.hybrid?.obstacle_takeover_steps !== undefined
                ? 'Hybrid PPO + Auto steering and braking · Auto sets the rate-limited steering target near obstacles · development only'
                : 'Hybrid PPO + Auto · obstacle steering target pending or not reported · development only'
              : driveMode === 'ppo'
                ? modelId.startsWith('wp7-')
                  ? 'Exploratory PPO · brake zero · reliable completion not validated · development only'
                  : `Frozen PPO · ${runtimeTrackSelection.kind === 'custom' ? 'custom' : 'TRAIN'} development · scientific admission disabled`
                : replay
                  ? 'Imported replay: recorded frames are not independently verified'
                  : 'Local Python Runtime replay: recorded development frames, scientific admission disabled'}</div>
            <div className="drive-warning">Frozen checkpoints are catalogued read-only. PPO and Hybrid runs produce development output only; scientific admission and training remain disabled.</div>
            {driveMode === 'replay' && activeReplay && <div className="replay-guidance">{activeReplay.label}
              {activeReplay.schema === 'f1rl_racing_sandbox_replay_v2' && activeReplay.hybrid?.obstacle_takeover_steps !== undefined &&
                ` · Auto obstacle steering target ${activeReplay.hybrid.obstacle_takeover_steps}/${activeReplay.hybrid.steps} steps (rate-limited)`}
            </div>}
            {driveMode !== 'replay' ? <>
              <div className="runtime-panel">
                {driveMode === 'hybrid' && <><label>EXPLORATORY PPO MODEL<select aria-label="Hybrid PPO model" value={hybridModelId} onChange={(event) => setHybridModelId(event.target.value)} disabled={!runtimeCatalog || runtimeBusy}>
                  {(runtimeCatalog?.exploratoryModels ?? []).map((model) => <option key={model.model_id} value={model.model_id}>{model.model_id}</option>)}
                </select></label>
                  <small>Auto assists steering and controls braking. This is not pure PPO or scientific acceptance.</small>
                  {runtimeCatalog?.exploratoryWarnings.map((warning) => <p className="runtime-error" key={warning}>{warning}</p>)}</>}
                {driveMode === 'ppo' && <><label>PPO MODEL<select aria-label="PPO model" value={modelId} onChange={(event) => setModelId(event.target.value)} disabled={!runtimeCatalog || runtimeBusy}>
                    <optgroup label="Frozen WP3">{(runtimeCatalog?.models ?? []).map((model) => <option key={model.model_id} value={model.model_id}>{model.model_id}</option>)}</optgroup>
                  </select></label>
                  {runtimeCatalog?.exploratoryWarnings.map((warning) => <p className="runtime-error" key={warning}>{warning}</p>)}</>}
                {runtimeTrackSelection.kind === 'custom' ? <div className="custom-track-selection">
                  {runtimeTrackSelection.validated ? <>
                    <span>PYTHON VALIDATED CUSTOM TRACK</span><strong>{runtimeTrackSelection.validated.preview.name}</strong>
                    <small>{runtimeTrackSelection.validated.surface_count} road surfaces · {runtimeTrackSelection.validated.checkpoint_count} checkpoints · {runtimeTrackSelection.validated.auto_bridge_count} auto overpasses</small>
                    {runtimeTrackSelection.validated.warnings.map((warning) => <small key={warning}>{warning}</small>)}
                    {(driveMode === 'ppo' || driveMode === 'hybrid') && !runtimeTrackSelection.validated.ppo_compatible && <small className="runtime-error">PPO observation/action contract is incompatible.</small>}
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
                {(driveMode === 'auto' || driveMode === 'hybrid') && <AutoDriveSettings config={autoConfig} onChange={setAutoConfig} onValidityChange={setAutoConfigValid}
                  disabled={runtimeBusy} physicalCapKmh={runtimeStatus?.mode === 'auto'
                    ? runtimeStatus.effective_max_speed_kmh : undefined}/>}
                {driveMode === 'auto' && runtimeTrackSelection.kind === 'custom' && <label>
                  <input aria-label="Experimental Physics V4" type="checkbox"
                    checked={experimentalPhysicsV4} disabled={runtimeBusy}
                    onChange={(event) => setExperimentalPhysicsV4(event.target.checked)}/>
                  Experimental Physics V4 / Race V4 (development only; may fail closed)
                </label>}
                <label>SEED<input aria-label="Episode seed" type="number" min="0" max="4294967295" step="1" value={runtimeSeed} onChange={(event) => setRuntimeSeed(Number(event.target.value))} disabled={runtimeBusy}/></label>
                {catalogError ? <p className="runtime-error"><TriangleAlert size={15}/><span>{catalogError}. Start the Python runtime to run episodes.</span><button onClick={() => setCatalogError('')}>Retry connection</button></p> : !runtimeCatalog ?
                  <p className="runtime-loading"><LoaderCircle size={15}/> Connecting to local Python runtime…</p> : null}
                {runtimeError && <p className="runtime-error"><TriangleAlert size={15}/>{runtimeError}</p>}
                {runtimeStatus && <>
                  {runtimeStatus.hybrid && <div className="runtime-physics-profile">HYBRID · BASE PPO STEERING {runtimeStatus.hybrid.ppo_steering_weight * 100}% · {runtimeStatus.hybrid.obstacle_takeover_steps === undefined ? 'OBSTACLE TARGET NOT REPORTED' : `AUTO OBSTACLE TARGET ${runtimeStatus.hybrid.obstacle_takeover_steps}/${runtimeStatus.hybrid.steps}${runtimeStatus.hybrid.obstacle_takeover_active ? ' · ACTIVE' : ''} · RATE-LIMITED`} · AUTO INTERVENTIONS (Δ PPO &gt; 0.05) {runtimeStatus.hybrid.steering_interventions}/{runtimeStatus.hybrid.steps}</div>}
                  {runtimeStatus.auto_drive_guidance === 'directed_gate_crossing' && <div className="runtime-physics-profile">
                    RIGHT TURN GATES · LOCAL CENTERLINE CROSSING · AUTO TARGET MAX {runtimeStatus.guidance_max_speed_kmh} KM/H
                  </div>}
                  <div className="runtime-physics-profile" aria-label="Authoritative Physics profile">
                    PHYSICS {runtimeStatus.physics?.ground_motion_version ?? 'not reported'}
                    {runtimeStatus.physics && ` | SHA ${runtimeStatus.physics.sha256.slice(0, 12)}`}
                  </div>
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
                    <span>ACCEPTED COLLISIONS <b>{runtimeStatus.latest_frame.collision_count}</b></span>
                    {runtimeStatus.collision_terminal && <span>UNACCEPTED CONTACTS <b>{runtimeStatus.collision_terminal.obstacle_ids.length}</b></span>}
                    <span>{runtimeStatus.latest_frame.grounded ? 'GROUNDED' : 'AIRBORNE'}</span>
                  </div>}
                  <RuntimeOutcomeMessage status={runtimeStatus} />
                </>}
                <div className="runtime-actions">
                  <button className="runtime-run" onClick={() => void runRuntimeSession()} disabled={!runtimeCatalog || runtimeBusy || ((driveMode === 'auto' || driveMode === 'hybrid') && !autoConfigValid) || (driveMode === 'hybrid' && !runtimeCatalog.exploratoryModels.some((model) => model.model_id === hybridModelId)) || !Number.isInteger(runtimeSeed) || runtimeSeed < 0 || runtimeSeed > 0xffffffff || !runtimeTrackRequest}>
                    {runtimeBusy ? <><LoaderCircle size={16}/> {runtimeStatus?.status === 'paused' ? 'PAUSED' : driveMode === 'manual' ? 'MANUAL PYTHON SIMULATION' : driveMode === 'auto' ? 'AUTO DRIVE · PYTHON PHYSICS' : driveMode === 'hybrid' ? 'HYBRID PPO + AUTO' : 'RUNNING PPO INFERENCE'}</> : <><Play size={16}/> {driveMode === 'manual' ? 'START MANUAL DRIVE' : driveMode === 'auto' ? 'RUN AUTO DRIVE' : driveMode === 'hybrid' ? 'RUN HYBRID' : 'RUN EPISODE'}</>}
                  </button>
                  {runtimeBusy && <button className="runtime-stop" onClick={() => void stopRuntimeSession()}>Stop</button>}
                  <button className="runtime-import" onClick={() => replayInput.current?.click()}><Upload size={15}/> Import replay</button>
                </div>
                {runtimeBusy && <div className="runtime-live-controls">
                  <button disabled={!runtimeConnected || runtimeTerminalPending.current} onClick={() => controlRuntime(runtimeStatus?.status === 'paused' ? 'resume' : 'pause')}>{runtimeStatus?.status === 'paused' ? <Play size={14}/> : <Pause size={14}/>} {runtimeStatus?.status === 'paused' ? 'Resume' : 'Pause'}</button>
                  <button disabled={!runtimeConnected || runtimeTerminalPending.current} onClick={() => controlRuntime('reset')}><RotateCcw size={14}/> Reset</button>
                  <label>SIM SPEED<select disabled={!runtimeConnected || runtimeTerminalPending.current} aria-label="Simulation speed" value={runtimeStatus?.speed_multiplier ?? 1} onChange={(event) => {
                    const multiplier = Number(event.target.value);
                    try { runtimeSocketRef.current?.speed(multiplier); setRuntimeStatus((current) => current ? { ...current, speed_multiplier: multiplier } : current); }
                    catch (error) { setRuntimeError(error instanceof Error ? error.message : 'Unable to set simulation speed'); }
                  }}><option value="0.25">0.25×</option><option value="0.5">0.5×</option><option value="1">1×</option><option value="2">2×</option><option value="4">4×</option><option value="8">8×</option><option value="16">16×</option></select></label>
                </div>}
              </div>
            </> : activeReplay ? <>
              {replayTrackUnavailable && <p className="runtime-error" role="alert">This replay has no matching track geometry; the current map will not be shown as a substitute.</p>}
              <div className="replay-controls">
              <button aria-label={playing ? 'Pause replay' : 'Play replay'} onClick={() => {if (replayTime >= activeReplay.frames[activeReplay.frames.length - 1].time_s) setReplayTime(activeReplay.frames[0].time_s); setPlaying(!playing);}}>{playing ? <Pause size={19}/> : <Play size={19}/>}</button>
              <input aria-label="Replay timeline" type="range" min={activeReplay.frames[0].time_s} max={activeReplay.frames[activeReplay.frames.length - 1].time_s} step="0.01" value={replayTime} onChange={(event) => {setPlaying(false);setReplayTime(Number(event.target.value));}}/>
              <span>{replayTime.toFixed(1)}s</span><span className="replay-name">{replayModelId(activeReplay)}</span>
              <select aria-label="Replay speed" value={playbackSpeed} onChange={(event) => setPlaybackSpeed(Number(event.target.value))}><option value="0.5">0.5×</option><option value="1">1×</option><option value="2">2×</option><option value="4">4×</option></select>
              <button aria-label="Restart replay" onClick={() => {setPlaying(false);setReplayTime(activeReplay.frames[0].time_s);}}><RotateCcw size={16}/></button>
              <button aria-label="Export replay JSON" onClick={() => downloadJSON('f1rl-runtime-replay.json', activeReplay)}><Download size={16}/></button>
              </div>
            </> : <div className="replay-empty"><span>No replay loaded</span><button onClick={() => replayInput.current?.click()}><Upload size={16}/> Import replay</button></div>}
            <div className="drive-bottom"><button disabled={runtimeBusy ? !runtimeConnected || runtimeTerminalPending.current : driveMode === 'auto' && !autoConfigValid} onClick={() => {
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
