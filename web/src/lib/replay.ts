export type ReplayFrame = {
  time_s: number;
  position: [number, number, number];
  heading_rad: number;
  speed_mps: number;
};
export type Replay = {
  schema: 'f1rl_racing_sandbox_replay_v1';
  label: string;
  source: { split: 'TRAIN' | 'DEVELOPMENT'; model_id: string; source_sha256?: string };
  frames: ReplayFrame[];
};

const finite = (n: unknown): n is number => typeof n === 'number' && Number.isFinite(n);

export function parseReplay(raw: unknown): Replay {
  if (!raw || typeof raw !== 'object') throw new Error('Invalid replay');
  const d = raw as Partial<Replay>;
  if (d.schema !== 'f1rl_racing_sandbox_replay_v1' || typeof d.label !== 'string' ||
      !d.label || d.label.length > 120 || !d.source || !['TRAIN', 'DEVELOPMENT'].includes(d.source.split) ||
      typeof d.source.model_id !== 'string' || !d.source.model_id.trim() || d.source.model_id.length > 100 ||
      (d.source.source_sha256 !== undefined && !/^[a-f0-9]{64}$/.test(d.source.source_sha256)) ||
      !Array.isArray(d.frames) || d.frames.length < 2 || d.frames.length > 50000) {
    throw new Error('Unsupported replay schema or provenance (HELD-OUT is not allowed)');
  }
  let prevTime = -Infinity;
  for (const frame of d.frames) {
    if (!frame || !finite(frame.time_s) || frame.time_s < 0 || frame.time_s <= prevTime ||
        !Array.isArray(frame.position) || frame.position.length !== 3 ||
        !frame.position.every((v) => finite(v) && Math.abs(v) < 100000) ||
        !finite(frame.heading_rad) || !finite(frame.speed_mps) || frame.speed_mps < 0) {
      throw new Error('Invalid or unordered replay frames');
    }
    prevTime = frame.time_s;
  }
  return {
    schema: d.schema, label: d.label,
    source: { split: d.source.split, model_id: d.source.model_id, source_sha256: d.source.source_sha256 },
    frames: d.frames.map((f) => ({
      time_s: f.time_s, position: [...f.position] as [number, number, number],
      heading_rad: f.heading_rad, speed_mps: f.speed_mps,
    })),
  };
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
    time_s: bounded,
    position: [0, 1, 2].map((i) => a.position[i] + (b.position[i] - a.position[i]) * u) as [number, number, number],
    heading_rad: a.heading_rad + turn * u,
    speed_mps: a.speed_mps + (b.speed_mps - a.speed_mps) * u,
  };
}
