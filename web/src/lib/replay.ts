import { parseTrack, type RoadSample, type TrackDocument } from './track';
import { slerpWxyz } from './orientation';
import { EXPLORATORY_DIGESTS } from './exploratoryModels';
import { validateAutoConfig, type AutoDriveConfig } from './autoDriveConfig';

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
export type DevelopmentCollisionTerminal = {
  schema_version: 'f1rl_runtime_collision_failure_dev_v1';
  obstacle_ids: string[];
  last_accepted_tick: number;
  witness_interval_s: [number, number];
  diagnostic_penalty: number;
  penalty_kind: 'runtime_diagnostic_not_gym_reward';
  physics_transition_accepted: false;
  contact_time_certified: false;
  scientific_evidence: false;
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
  track_preview?: ReplayTrackPreview;
  auto_drive_config?: AutoDriveConfig;
  hybrid?: { ppo_steering_weight: number; steps: number; steering_interventions: number; obstacle_takeover_steps?: number; obstacle_takeover_active?: boolean };
  collision_terminal?: DevelopmentCollisionTerminal;
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
export type ReplayTrackPreview = {
  track_id: string;
  track_sha256: string;
  preview: TrackDocument;
  road_samples: RoadSample[];
};
export type Replay = ReplayV1 | ReplayV2;

const finite = (n: unknown): n is number => typeof n === 'number' && Number.isFinite(n);
const digest = (value: unknown): value is string => typeof value === 'string' && /^[a-f0-9]{64}$/.test(value);
const vector = (value: unknown, size: number): value is number[] => Array.isArray(value) && value.length === size && Array.from(value).every(finite);

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
  let orientation: ReplayFrame['orientation_wxyz'];
  if (f.orientation_wxyz !== undefined) {
    if (!vector(f.orientation_wxyz, 4)) throw new Error('Invalid replay orientation');
    const scale = Math.max(...f.orientation_wxyz.map(Math.abs));
    if (scale === 0) throw new Error('Invalid replay orientation');
    const scaled = f.orientation_wxyz.map(value => value / scale);
    const length = Math.hypot(...scaled);
    orientation = scaled.map(value => value / length) as ReplayFrame['orientation_wxyz'];
  }
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
  return { ...f, position: [...f.position] as [number, number, number],
    ...(orientation ? { orientation_wxyz: orientation } : {}) } as ReplayFrame;
}

function stringArray(value: unknown): value is string[] {
  return Array.isArray(value) && value.every((item) => typeof item === 'string' && item.length <= 200);
}

function preserveUnknownTrackFields(raw: unknown, validated: unknown): unknown {
  if (Array.isArray(validated)) {
    return validated.map((value, index) => preserveUnknownTrackFields(Array.isArray(raw) ? raw[index] : undefined, value));
  }
  if (validated && typeof validated === 'object') {
    const original = raw && typeof raw === 'object' && !Array.isArray(raw) ? raw as Record<string, unknown> : {};
    const parsed = validated as Record<string, unknown>;
    return Object.fromEntries(Object.entries({ ...original, ...parsed }).map(([key, value]) => [
      key, key in original && key in parsed ? preserveUnknownTrackFields(original[key], parsed[key]) : value,
    ]));
  }
  return validated;
}

function validateTrackPreview(raw: unknown, trackId: string, trackSha256: string): ReplayTrackPreview | undefined {
  if (raw === undefined) return undefined;
  if (!raw || typeof raw !== 'object') throw new Error('Invalid replay track preview');
  const value = raw as Record<string, unknown>;
  if (value.track_id !== trackId || value.track_sha256 !== trackSha256 || !Array.isArray(value.road_samples)) {
    throw new Error('Replay track preview does not match replay provenance');
  }
  const preview = preserveUnknownTrackFields(value.preview, parseTrack(value.preview)) as TrackDocument;
  const roadSamples = value.road_samples as RoadSample[];
  if (preview.id !== trackId || roadSamples.length < 2 || roadSamples.some((sample, index) =>
    !sample || ![sample.x, sample.y, sample.z, sample.t, sample.heading, sample.nx, sample.nz].every(finite) ||
    sample.t < 0 || sample.t > 1 || (index > 0 && sample.t <= roadSamples[index - 1].t))) {
    throw new Error('Invalid replay track geometry');
  }
  return {
    ...value,
    track_id: trackId,
    track_sha256: trackSha256,
    preview,
    road_samples: roadSamples.map((sample) => ({ ...sample })),
  };
}

function parseCollisionTerminal(raw: unknown, frames: ReplayFrame[]): DevelopmentCollisionTerminal | undefined {
  if (raw === undefined) return undefined;
  if (!raw || typeof raw !== 'object' || Array.isArray(raw)) throw new Error('Invalid runtime collision terminal');
  const c = raw as Partial<DevelopmentCollisionTerminal>;
  const last = frames[frames.length - 1];
  if (c.schema_version !== 'f1rl_runtime_collision_failure_dev_v1' ||
      !stringArray(c.obstacle_ids) || c.obstacle_ids.length < 1 ||
      !Number.isInteger(c.last_accepted_tick) || c.last_accepted_tick !== last.tick ||
      !vector(c.witness_interval_s, 2) ||
      Math.abs(c.witness_interval_s[0] - last.time_s) > 1e-7 ||
      c.witness_interval_s[1] <= c.witness_interval_s[0] ||
      !finite(c.diagnostic_penalty) || c.diagnostic_penalty > 0 ||
      c.penalty_kind !== 'runtime_diagnostic_not_gym_reward' ||
      c.physics_transition_accepted !== false || c.contact_time_certified !== false ||
      c.scientific_evidence !== false ||
      frames.some(f => (f.collision_ids?.length ?? 0) > 0)) {
    throw new Error('Invalid runtime collision terminal');
  }
  return { ...c } as DevelopmentCollisionTerminal;
}

export function parseReplay(raw: unknown): Replay {
  if (!raw || typeof raw !== 'object') throw new Error('Invalid replay');
  const d = raw as Record<string, unknown>;
  if (typeof d.label !== 'string' || !d.label || d.label.length > 120 ||
      !Array.isArray(d.frames) || d.frames.length < (d.schema === 'f1rl_racing_sandbox_replay_v2' && d.collision_terminal !== undefined ? 1 : 2) || d.frames.length > 5001) {
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
        typeof provenance.model_id !== 'string' || !(provenance.model_id === 'manual' || provenance.model_id === 'auto-drive' || /^[A-D]-seed[0-2]$/.test(provenance.model_id) ||
          (['wp7-calibration', 'wp7-confirmation'].includes(provenance.model_id) && provenance.dataset_split === 'DEVELOPMENT' &&
            provenance.model_sha256 === EXPLORATORY_DIGESTS[provenance.model_id] && result?.valid_sample === false)) ||
        (provenance.model_sha256 !== null && !digest(provenance.model_sha256)) ||
        typeof provenance.track_id !== 'string' || !provenance.track_id || !digest(provenance.track_sha256) ||
        !Number.isInteger(provenance.seed) || provenance.seed < 0 || provenance.seed > 0xffffffff ||
        !result || !['finished', 'failed', 'cancelled', 'stopped'].includes(result.status) ||
        typeof result.valid_sample !== 'boolean' || typeof result.successful_finish !== 'boolean' ||
        !(result.reason === null || typeof result.reason === 'string') || !Number.isInteger(result.steps) ||
        !finite(result.duration_s) || result.duration_s < 0 || !digest(result.trace_sha256)) {
      throw new Error('Unsupported replay schema or provenance (HELD-OUT is not allowed)');
    }
    const collisionTerminal = parseCollisionTerminal(d.collision_terminal, frames);
    if (collisionTerminal && (result.reason !== 'collision' || result.status !== 'failed' || result.valid_sample || result.successful_finish)) {
      throw new Error('Runtime collision must be a failed, invalid sample');
    }
    const trackPreview = validateTrackPreview(d.track_preview, provenance.track_id, provenance.track_sha256);
    const hybrid = d.hybrid as ReplayV2['hybrid'];
    if (hybrid && (!['wp7-calibration', 'wp7-confirmation'].includes(provenance.model_id) ||
        provenance.dataset_split !== 'DEVELOPMENT' || result.valid_sample ||
        ![0.2, 0.5].includes(hybrid.ppo_steering_weight) || !Number.isInteger(hybrid.steps) || hybrid.steps < 0 ||
        !Number.isInteger(hybrid.steering_interventions) || hybrid.steering_interventions < 0 || hybrid.steering_interventions > hybrid.steps ||
        ((hybrid.obstacle_takeover_steps === undefined) !== (hybrid.obstacle_takeover_active === undefined)) ||
        (hybrid.obstacle_takeover_steps !== undefined && (!Number.isInteger(hybrid.obstacle_takeover_steps) || hybrid.obstacle_takeover_steps < 0 || hybrid.obstacle_takeover_steps > hybrid.steps)) ||
        (hybrid.obstacle_takeover_active !== undefined && typeof hybrid.obstacle_takeover_active !== 'boolean'))) {
      throw new Error('Invalid hybrid development provenance');
    }
    if (d.hybrid !== undefined && (!hybrid || typeof hybrid !== 'object')) throw new Error('Invalid hybrid metadata');
    if (d.auto_drive_config !== undefined && provenance.model_id !== 'auto-drive' && !hybrid) throw new Error('Config belongs to Auto Drive or Hybrid only');
    const autoConfig = d.auto_drive_config === undefined ? undefined : validateAutoConfig(d.auto_drive_config);
    return {
      schema: d.schema, label: d.label, provenance: { ...provenance }, result: { ...result }, frames,
      ...(trackPreview ? { track_preview: trackPreview } : {}),
      ...(autoConfig ? { auto_drive_config: autoConfig } : {}),
      ...(hybrid ? { hybrid: { ...hybrid } } : {}),
      ...(collisionTerminal ? { collision_terminal: collisionTerminal } : {}),
    };
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
  // Range inputs quantize time to decimals such as 0.35, while a Physics
  // timestamp can be 0.35000000000000003. Interpret sub-nanosecond binary
  // rounding as the *same certified frame*, never as an earlier tick.
  // Do not enlarge this to a visual frame tolerance: that would announce
  // airborne/landing/collision events before their actual Physics samples.
  if (Math.abs(b.time_s - bounded) <= 1e-10) return b;
  const u = (bounded - a.time_s) / (b.time_s - a.time_s);
  const turn = Math.atan2(Math.sin(b.heading_rad - a.heading_rad), Math.cos(b.heading_rad - a.heading_rad));
  return {
    // Discrete Physics fields belong to the last certified sample at or
    // before the requested replay time. Never preview future "airborne",
    // "landing", collision or checkpoint events between frames.
    ...(u >= 1 ? b : a),
    time_s: bounded,
    position: [0, 1, 2].map((i) => a.position[i] + (b.position[i] - a.position[i]) * u) as [number, number, number],
    heading_rad: a.heading_rad + turn * u,
    speed_mps: a.speed_mps + (b.speed_mps - a.speed_mps) * u,
      ...(a.orientation_wxyz && b.orientation_wxyz
        ? { orientation_wxyz: slerpWxyz(a.orientation_wxyz, b.orientation_wxyz, u) } : {}),
  };
}
