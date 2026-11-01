import { parseReplay, type Replay } from './replay';
import type { TrackDocument } from './track';

export type RuntimeModel = { model_id: string; sha256: string; bytes: number };
export type RuntimeTrack = {
  track_id: string;
  split: 'TRAIN';
  purpose: 'development';
  world_sha256: string;
  preview: TrackDocument;
  road_samples: { x: number; y: number; z: number; t: number; heading: number; nx: number; nz: number }[];
};
export type RuntimeCatalog = { models: RuntimeModel[]; tracks: RuntimeTrack[] };
export type RuntimeCustomTrack = {
  schema_version: 'f1rl_custom_track_conversion_v1';
  development_only: true;
  track_id: string;
  track_sha256: string;
  preview: RuntimeTrack['preview'];
  road_samples: RuntimeTrack['road_samples'];
  auto_bridge_count: number;
  ppo_compatible: boolean;
  surface_count: number;
  checkpoint_count: number;
  obstacle_count: number;
  boost_pad_count: number;
  warnings: string[];
};
export type RuntimeTrackSelection = { kind: 'train' } | { kind: 'custom'; validated: RuntimeCustomTrack | null };
export type RuntimeTrackRequest = { kind: 'train'; trackId: string } | { kind: 'custom' };

export function invalidateRuntimeTrackSelection(selection: RuntimeTrackSelection): RuntimeTrackSelection {
  return selection.kind === 'custom' ? { kind: 'custom', validated: null } : selection;
}

export function isCurrentCustomTrackValidation(requestRevision: number, currentRevision: number): boolean {
  return requestRevision === currentRevision;
}

export function resolveRuntimeTrackRequest(
  selection: RuntimeTrackSelection,
  trackId: string,
  mode: 'ppo' | 'manual' | 'replay',
): RuntimeTrackRequest | null {
  if (mode === 'replay') return null;
  if (selection.kind === 'train') return { kind: 'train', trackId };
  if (!selection.validated || (mode === 'ppo' && !selection.validated.ppo_compatible)) return null;
  return { kind: 'custom' };
}

export function matchesValidatedCustomTrack(
  validated: RuntimeCustomTrack,
  sessionTrack: Pick<RuntimeCustomTrack, 'track_id' | 'track_sha256'> | null | undefined,
): boolean {
  return sessionTrack?.track_id === validated.track_id && sessionTrack.track_sha256 === validated.track_sha256;
}

export function shouldValidateCustomTrackOnDrive(
  previousMode: 'sketch' | 'build' | 'drive',
  keepReplay: boolean,
): boolean {
  return previousMode !== 'drive' && !keepReplay;
}

export type RuntimeSessionStatus = {
  protocol_version: 'f1rl_runtime_v1';
  session_id: string;
  mode: 'ppo' | 'manual';
  model_id: string | null;
  track_id: string;
  seed: number;
  status: 'ready' | 'running' | 'finished' | 'failed' | 'cancelled' | 'stopped' | 'paused';
  steps: number;
  max_steps: number;
  step_progress: number;
  latest_frame: RuntimeFrame | null;
  episode_result: Record<string, unknown> | null;
  error: string | null;
  provenance: 'TRAIN-development' | 'manual-development' | 'custom-development';
  speed_multiplier?: number;
  track_preview?: RuntimeCustomTrack;
};
export type RuntimeFrame = {
  schema_version: 'f1rl_runtime_frame_v1';
  tick: number;
  time_s: number;
  dt_s: number;
  position: [number, number, number];
  velocity: [number, number, number];
  acceleration_mps2: [number, number, number];
  orientation_wxyz: [number, number, number, number];
  heading_rad: number;
  speed_mps: number;
  grounded: boolean;
  action: { steering: number; throttle: number; brake: number };
  route_progress_m: number | null;
  route_progress_fraction: number | null;
  route_highwater_m: number;
  next_checkpoint_index: number | null;
  checkpoint_count: number;
  completed_checkpoint_ids: string[];
  checkpoint_crossings: string[];
  collision_count: number;
  collision_ids: string[];
  physical_event_types: string[];
  outcome: { kind: string; valid_sample: boolean; reason: string; scientific_evidence: false } | null;
};
export type RuntimeSocketEvent = {
  protocol_version: 'f1rl_runtime_ws_v1';
  type: 'ready' | 'status' | 'frame' | 'progress' | 'checkpoint' | 'collision' | 'terminal' | 'error' | 'ack';
  session_id: string;
  data: Record<string, unknown>;
};

// A local E2E backend may use another port; never accept a non-loopback endpoint.
const configuredRuntimeUrl: string = import.meta.env.VITE_RUNTIME_URL || '';
export const DEFAULT_RUNTIME_URL = /^http:\/\/(?:127\.0\.0\.1|localhost):\d+$/.test(configuredRuntimeUrl)
  ? configuredRuntimeUrl : 'http://127.0.0.1:8000';
const EXPECTED_MODEL_IDS = ['A', 'B', 'C', 'D'].flatMap((candidate) => [0, 1, 2].map((seed) => `${candidate}-seed${seed}`));

export class RuntimeApiError extends Error {
  constructor(message: string, readonly code: string, readonly status: number, readonly traceId?: string) {
    super(message);
    this.name = 'RuntimeApiError';
  }
}

async function request<T>(baseUrl: string, path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(new URL(path, baseUrl), init);
  } catch (error) {
    throw new RuntimeApiError(error instanceof Error ? error.message : 'Runtime is unavailable', 'network_error', 0);
  }
  if (!response.ok) {
    let payload: unknown;
    try { payload = await response.json(); } catch { payload = null; }
    const detail = (payload as { error?: { code?: string; message?: string } } | null)?.error;
    throw new RuntimeApiError(detail?.message ?? `Runtime request failed (${response.status})`, detail?.code ?? 'http_error', response.status);
  }
  if (response.status === 204) return undefined as T;
  return await response.json() as T;
}

export class RuntimeClient {
  constructor(readonly baseUrl = DEFAULT_RUNTIME_URL) {}

  async catalog(): Promise<RuntimeCatalog> {
    const [models, tracks] = await Promise.all([
      request<{ models: RuntimeModel[] }>(this.baseUrl, '/api/v1/models'),
      request<{ tracks: RuntimeTrack[] }>(this.baseUrl, '/api/v1/tracks'),
    ]);
    if (models.models.length !== 12 ||
        models.models.some((model) => !/^[A-D]-seed[0-2]$/.test(model.model_id) || !/^[a-f0-9]{64}$/.test(model.sha256) || !Number.isInteger(model.bytes) || model.bytes <= 0) ||
        EXPECTED_MODEL_IDS.some((modelId) => !models.models.some((model) => model.model_id === modelId)) ||
        tracks.tracks.length === 0 || tracks.tracks.some((track) => track.split !== 'TRAIN' || track.purpose !== 'development' ||
          !/^[a-f0-9]{64}$/.test(track.world_sha256) || track.preview?.schema !== 'f1rl_racing_sandbox_draft_v1' ||
          !Array.isArray(track.road_samples) || track.road_samples.length < 2 ||
          track.road_samples.some((point) => ![point.x, point.y, point.z, point.t, point.heading, point.nx, point.nz].every(Number.isFinite)))) {
      throw new RuntimeApiError('Runtime catalog is outside the development-only contract', 'invalid_catalog', 502);
    }
    return { models: models.models, tracks: tracks.tracks };
  }

  createPpoSession(modelId: string, trackId: string, seed: number): Promise<RuntimeSessionStatus> {
    return request(this.baseUrl, '/api/v1/sessions', {
      method: 'POST', headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ model_id: modelId, track_id: trackId, seed }),
    });
  }
  createManualSession(trackId: string, seed: number): Promise<RuntimeSessionStatus> {
    return request(this.baseUrl, '/api/v1/sessions', {
      method: 'POST', headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ mode: 'manual', track_id: trackId, seed }),
    });
  }
  createCustomSession(track: unknown, mode: 'manual' | 'ppo', seed: number, modelId?: string, traceId?: string): Promise<RuntimeSessionStatus> {
    const body: Record<string, unknown> = { mode, seed, custom_track: track };
    if (modelId) body.model_id = modelId;
    return request(this.baseUrl, '/api/v1/sessions', {
      method: 'POST', headers: { 'content-type': 'application/json', ...(traceId ? { 'x-trace-id': traceId } : {}) }, body: JSON.stringify(body),
    });
  }
  validateCustomTrack(track: unknown): Promise<RuntimeCustomTrack> {
    return request(this.baseUrl, '/api/v1/tracks/validate', {
      method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ track }),
    });
  }
  run(sessionId: string): Promise<RuntimeSessionStatus> {
    return request(this.baseUrl, `/api/v1/sessions/${encodeURIComponent(sessionId)}/run`, { method: 'POST' });
  }
  status(sessionId: string): Promise<RuntimeSessionStatus> {
    return request(this.baseUrl, `/api/v1/sessions/${encodeURIComponent(sessionId)}`);
  }
  async replay(sessionId: string): Promise<Replay> {
    return parseReplay(await request(this.baseUrl, `/api/v1/sessions/${encodeURIComponent(sessionId)}/replay`));
  }
  close(sessionId: string): Promise<void> {
    return request(this.baseUrl, `/api/v1/sessions/${encodeURIComponent(sessionId)}`, { method: 'DELETE' });
  }

  connect(sessionId: string, onEvent: (event: RuntimeSocketEvent) => void, onError: (error: Error) => void): Promise<RuntimeSocketClient> {
    return RuntimeSocketClient.open(this.baseUrl, sessionId, onEvent, onError);
  }
}

export class RuntimeSocketClient {
  private constructor(private readonly socket: WebSocket, readonly sessionId: string) {}

  static open(baseUrl: string, sessionId: string, onEvent: (event: RuntimeSocketEvent) => void, onError: (error: Error) => void): Promise<RuntimeSocketClient> {
    const endpoint = new URL(`/api/v1/sessions/${encodeURIComponent(sessionId)}/ws`, baseUrl);
    endpoint.protocol = endpoint.protocol === 'https:' ? 'wss:' : 'ws:';
    return new Promise((resolve, reject) => {
      const socket = new WebSocket(endpoint);
      let ready = false;
      const timer = window.setTimeout(() => {
        if (!ready) { socket.close(); reject(new RuntimeApiError('Runtime WebSocket handshake timed out', 'connection_timeout', 0)); }
      }, 8000);
      socket.onmessage = (message) => {
        try {
          const event = parseRuntimeSocketEvent(JSON.parse(String(message.data)), sessionId);
          onEvent(event);
          if (!ready && event.type === 'ready') {
            ready = true;
            window.clearTimeout(timer);
            resolve(new RuntimeSocketClient(socket, sessionId));
          }
        } catch (error) {
          const failure = error instanceof Error ? error : new Error('Invalid runtime WebSocket event');
          onError(failure);
          if (!ready) { window.clearTimeout(timer); socket.close(); reject(failure); }
        }
      };
      socket.onerror = () => {
        const failure = new RuntimeApiError('Runtime WebSocket connection failed', 'websocket_error', 0);
        onError(failure);
        if (!ready) { window.clearTimeout(timer); reject(failure); }
      };
      socket.onclose = (event) => {
        if (!ready) {
          window.clearTimeout(timer);
          reject(new RuntimeApiError(event.reason || 'Runtime WebSocket closed before ready', 'websocket_closed', 0));
        } else if (event.code !== 1000) {
          onError(new RuntimeApiError(event.reason || 'Runtime WebSocket disconnected; start a new session to resume', 'websocket_closed', 0));
        }
      };
    });
  }

  start(): void { this.send({ type: 'start' }); }
  pause(): void { this.send({ type: 'pause' }); }
  resume(): void { this.send({ type: 'resume' }); }
  reset(): void { this.send({ type: 'reset' }); }
  stop(): void { this.send({ type: 'stop' }); }
  speed(multiplier: number): void { this.send({ type: 'speed', multiplier }); }
  action(steering: number, throttle: number, brake: number): void { this.send({ type: 'action', steering, throttle, brake }); }
  disconnect(): void { this.socket.close(1000, 'client finished'); }

  private send(command: Record<string, unknown>): void {
    if (this.socket.readyState !== WebSocket.OPEN) throw new RuntimeApiError('Runtime WebSocket is not connected', 'websocket_closed', 0);
    this.socket.send(JSON.stringify({ protocol_version: 'f1rl_runtime_ws_v1', ...command }));
  }
}

export function parseRuntimeFrame(value: unknown): RuntimeFrame {
  if (!value || typeof value !== 'object') throw new Error('Runtime frame must be an object');
  const frame = value as Partial<RuntimeFrame>;
  const vector = (input: unknown, length: number) => Array.isArray(input) && input.length === length && input.every(Number.isFinite);
  if (frame.schema_version !== 'f1rl_runtime_frame_v1' || !Number.isInteger(frame.tick) || (frame.tick ?? -1) < 0 ||
      !Number.isFinite(frame.time_s) || (frame.time_s ?? -1) < 0 || !Number.isFinite(frame.dt_s) || (frame.dt_s ?? 0) <= 0 ||
      !vector(frame.position, 3) || !vector(frame.velocity, 3) || !vector(frame.acceleration_mps2, 3) ||
      !vector(frame.orientation_wxyz, 4) || !Number.isFinite(frame.heading_rad) || !Number.isFinite(frame.speed_mps) ||
      (frame.speed_mps ?? -1) < 0 || typeof frame.grounded !== 'boolean' || !frame.action ||
      !Number.isFinite(frame.action.steering) || frame.action.steering < -1 || frame.action.steering > 1 ||
      !Number.isFinite(frame.action.throttle) || frame.action.throttle < 0 || frame.action.throttle > 1 ||
      !Number.isFinite(frame.action.brake) || frame.action.brake < 0 || frame.action.brake > 1 ||
      !Number.isInteger(frame.collision_count) || (frame.collision_count ?? -1) < 0 ||
      !Array.isArray(frame.completed_checkpoint_ids) || !Array.isArray(frame.collision_ids) ||
      !Array.isArray(frame.physical_event_types)) throw new Error('Invalid runtime frame contract');
  return frame as RuntimeFrame;
}

export function parseRuntimeSocketEvent(value: unknown, expectedSessionId?: string): RuntimeSocketEvent {
  if (!value || typeof value !== 'object') throw new Error('Runtime event must be an object');
  const event = value as Partial<RuntimeSocketEvent>;
  const allowed = ['ready', 'status', 'frame', 'progress', 'checkpoint', 'collision', 'terminal', 'error', 'ack'];
  if (event.protocol_version !== 'f1rl_runtime_ws_v1' || !allowed.includes(String(event.type)) ||
      typeof event.session_id !== 'string' || (expectedSessionId && event.session_id !== expectedSessionId) ||
      !event.data || typeof event.data !== 'object' || Array.isArray(event.data)) throw new Error('Invalid runtime WebSocket event');
  if (event.type === 'frame') parseRuntimeFrame(event.data);
  return event as RuntimeSocketEvent;
}
