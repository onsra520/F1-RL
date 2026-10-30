export type ReplayFrame = {
  time_s: number;
  position: [number, number, number];
  heading_rad: number;
  speed_mps: number;
  tick?: number;
  dt_s?: number;
  velocity?: [number, number, number];
  acceleration_mps2?: [number, number, number];
  orientation_wxyz?: [number, number, number, number];
  grounded?: boolean;
  action?: { steering: number; throttle: number; brake: number };
  route_progress_m?: number | null;
  route_progress_fraction?: number | null;
  route_highwater_m?: number;
  next_checkpoint_index?: number | null;
  checkpoint_count?: number;
  completed_checkpoint_ids?: string[];
  checkpoint_crossings?: string[];
  collision_count?: number;
  collision_ids?: string[];
  physical_event_types?: string[];
  outcome?: { kind: string; valid_sample: boolean; reason: string; scientific_evidence: false } | null;
};
export type ReplayV1 = {
  schema: 'f1rl_racing_sandbox_replay_v1';
  label: string;
  source: { split: 'TRAIN' | 'DEVELOPMENT'; model_id: string; source_sha256?: string };
  frames: ReplayFrame[];
};
export type ReplayV2 = {
  schema: 'f1rl_racing_sandbox_replay_v2';
  label: string;
  provenance: {
    dataset_split: 'TRAIN' | 'DEVELOPMENT';
    runtime_mode: 'development';
    source_sha256: string;
    model_id: string;
    model_sha256: string | null;
    track_id: string;
    track_sha256: string;
    seed: number;
  };
  result: {
    status: 'finished' | 'failed' | 'cancelled' | 'stopped';
    valid_sample: boolean;
    successful_finish: boolean;
    reason: string | null;
    steps: number;
    duration_s: number;
    trace_sha256: string;
  };
  frames: ReplayFrame[];
};
export type Replay = ReplayV1 | ReplayV2;

const finite = (n: unknown): n is number => typeof n === 'number' && Number.isFinite(n);
const digest = (value: unknown): value is string => typeof value === 'string' && /^[a-f0-9]{64}$/.test(value);
const vector = (value: unknown, size: number): value is number[] => Array.isArray(value) && value.length === size && value.every(finite);

function validateFrame(frame: unknown): ReplayFrame {
  if (!frame || typeof frame !== 'object') throw new Error('Invalid replay frame');
  const f = frame as Partial<ReplayFrame>;
  if (!finite(f.time_s) || f.time_s < 0 || !vector(f.position, 3) || f.position.some((v) => Math.abs(v) >= 100000) ||
      !finite(f.heading_rad) || !finite(f.speed_mps) || f.speed_mps < 0) throw new Error('Invalid replay frame pose');
  if (f.tick !== undefined && (!Number.isInteger(f.tick) || f.tick < 0)) throw new Error('Invalid replay frame tick');
  if (f.dt_s !== undefined && (!finite(f.dt_s) || f.dt_s <= 0)) throw new Error('Invalid replay timestep');
  for (const key of ['velocity', 'acceleration_mps2'] as const) {
    if (f[key] !== undefined && !vector(f[key], 3)) throw new Error(`Invalid replay ${key}`);
  }
  if (f.orientation_wxyz !== undefined && !vector(f.orientation_wxyz, 4)) throw new Error('Invalid replay orientation');
  if (f.action !== undefined && (!f.action || !finite(f.action.steering) || f.action.steering < -1 || f.action.steering > 1 ||
      !finite(f.action.throttle) || f.action.throttle < 0 || f.action.throttle > 1 ||
      !finite(f.action.brake) || f.action.brake < 0 || f.action.brake > 1)) throw new Error('Invalid replay action');
  if (f.route_progress_m !== undefined && f.route_progress_m !== null && (!finite(f.route_progress_m) || f.route_progress_m < 0)) throw new Error('Invalid route progress');
  if (f.route_progress_fraction !== undefined && f.route_progress_fraction !== null && (!finite(f.route_progress_fraction) || f.route_progress_fraction < 0 || f.route_progress_fraction > 1)) throw new Error('Invalid route progress fraction');
  if (f.grounded !== undefined && typeof f.grounded !== 'boolean') throw new Error('Invalid grounded state');
  if (f.completed_checkpoint_ids !== undefined && !stringArray(f.completed_checkpoint_ids)) throw new Error('Invalid checkpoint ids');
  if (f.checkpoint_crossings !== undefined && !stringArray(f.checkpoint_crossings)) throw new Error('Invalid checkpoint crossings');
  if (f.collision_ids !== undefined && !stringArray(f.collision_ids)) throw new Error('Invalid collision ids');
  if (f.physical_event_types !== undefined && !stringArray(f.physical_event_types)) throw new Error('Invalid physical events');
  return { ...f, position: [...f.position] as [number, number, number] } as ReplayFrame;
}

function stringArray(value: unknown): value is string[] {
  return Array.isArray(value) && value.every((item) => typeof item === 'string' && item.length <= 200);
}

export function parseReplay(raw: unknown): Replay {
  if (!raw || typeof raw !== 'object') throw new Error('Invalid replay');
  const d = raw as Record<string, unknown>;
  if (typeof d.label !== 'string' || !d.label || d.label.length > 120 ||
      !Array.isArray(d.frames) || d.frames.length < 2 || d.frames.length > 5001) {
    throw new Error('Unsupported replay schema or frame count');
  }
  const frames = d.frames.map(validateFrame);
  if (frames.some((frame, index) => index > 0 && frame.time_s <= frames[index - 1].time_s)) {
    throw new Error('Invalid or unordered replay frames');
  }
  if (d.schema === 'f1rl_racing_sandbox_replay_v1') {
    const source = d.source as ReplayV1['source'] | undefined;
    if (!source || !['TRAIN', 'DEVELOPMENT'].includes(source.split) ||
        typeof source.model_id !== 'string' || !source.model_id.trim() || source.model_id.length > 100 ||
        (source.source_sha256 !== undefined && !digest(source.source_sha256))) {
      throw new Error('Unsupported replay schema or provenance (HELD-OUT is not allowed)');
    }
    return { schema: d.schema, label: d.label, source: { ...source }, frames };
  }
  if (d.schema === 'f1rl_racing_sandbox_replay_v2') {
    const provenance = d.provenance as ReplayV2['provenance'] | undefined;
    const result = d.result as ReplayV2['result'] | undefined;
    if (!provenance || !['TRAIN', 'DEVELOPMENT'].includes(provenance.dataset_split) ||
        provenance.runtime_mode !== 'development' || !digest(provenance.source_sha256) ||
        typeof provenance.model_id !== 'string' || !(provenance.model_id === 'manual' || /^[A-D]-seed[0-2]$/.test(provenance.model_id)) ||
        (provenance.model_sha256 !== null && !digest(provenance.model_sha256)) ||
        typeof provenance.track_id !== 'string' || !provenance.track_id || !digest(provenance.track_sha256) ||
        !Number.isInteger(provenance.seed) || provenance.seed < 0 || provenance.seed > 0xffffffff ||
        !result || !['finished', 'failed', 'cancelled', 'stopped'].includes(result.status) ||
        typeof result.valid_sample !== 'boolean' || typeof result.successful_finish !== 'boolean' ||
        !(result.reason === null || typeof result.reason === 'string') || !Number.isInteger(result.steps) ||
        !finite(result.duration_s) || result.duration_s < 0 || !digest(result.trace_sha256)) {
      throw new Error('Unsupported replay schema or provenance (HELD-OUT is not allowed)');
    }
    return { schema: d.schema, label: d.label, provenance: { ...provenance }, result: { ...result }, frames };
  }
  throw new Error('Unsupported replay schema or provenance (HELD-OUT is not allowed)');
}

export function replayModelId(replay: Replay): string {
  return replay.schema === 'f1rl_racing_sandbox_replay_v1' ? replay.source.model_id : replay.provenance.model_id;
}

export function replayAt(replay: Replay, time: number): ReplayFrame {
  const frames = replay.frames;
  const bounded = Math.min(Math.max(time, frames[0].time_s), frames[frames.length - 1].time_s);
  let lo = 0, hi = frames.length - 1;
  while (lo < hi) {
    const mid = (lo + hi) >>> 1;
    if (frames[mid].time_s < bounded) lo = mid + 1;
    else hi = mid;
  }
  const b = frames[lo], a = frames[Math.max(0, lo - 1)];
  if (b === a) return b;
  const u = (bounded - a.time_s) / (b.time_s - a.time_s);
  const turn = Math.atan2(Math.sin(b.heading_rad - a.heading_rad), Math.cos(b.heading_rad - a.heading_rad));
  return {
    ...b,
    time_s: bounded,
    position: [0, 1, 2].map((i) => a.position[i] + (b.position[i] - a.position[i]) * u) as [number, number, number],
    heading_rad: a.heading_rad + turn * u,
    speed_mps: a.speed_mps + (b.speed_mps - a.speed_mps) * u,
  };
}
