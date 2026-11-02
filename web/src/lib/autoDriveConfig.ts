export type DrivingStyle = 'right' | 'centerline' | 'racing';
export type SpeedProfile = 'normal' | 'racing' | 'hyper';
export type AutoDriveConfig = {
  driving_style: DrivingStyle;
  speed_profile: SpeedProfile;
  target_max_speed_kmh: number;
  vehicle_width_m: number;
  lane_offset_m: number;
  edge_margin_m: number;
  min_lookahead_m: number;
  max_lookahead_m: number;
  speed_lookahead_s: number;
  corner_acceleration_mps2: number;
  braking_factor: number;
  acceleration_factor: number;
  steering_gain: number;
  recovery_speed_mps: number;
  grade_lookahead_m: number;
  moderate_grade_threshold: number;
  steep_grade_threshold: number;
  moderate_grade_speed_mps: number;
  steep_grade_speed_mps: number;
  corner_anticipation_m: number;
};
export const PRESET_SPEEDS_KMH: Record<SpeedProfile, number> = {
  normal: 45, racing: 90, hyper: 240,
};
export const DEFAULT_AUTO_CONFIG: AutoDriveConfig = {
  driving_style: 'right', speed_profile: 'normal', target_max_speed_kmh: 45,
  vehicle_width_m: 1.5, lane_offset_m: 0, edge_margin_m: 0.6,
  min_lookahead_m: 4, max_lookahead_m: 14, speed_lookahead_s: 0.65,
  corner_acceleration_mps2: 2.2, braking_factor: 0.65, acceleration_factor: 0.65,
  steering_gain: 1, recovery_speed_mps: 3, grade_lookahead_m: 20,
  moderate_grade_threshold: 0.1, steep_grade_threshold: 0.2,
  moderate_grade_speed_mps: 6, steep_grade_speed_mps: 4, corner_anticipation_m: 1.5,
};
export const AUTO_CONFIG_STORAGE = 'f1rl.auto-drive.settings.v1';
const ranges: Partial<Record<keyof AutoDriveConfig, [number, number]>> = {
  target_max_speed_kmh: [1, 300], vehicle_width_m: [0.401, 4], lane_offset_m: [0, 5],
  edge_margin_m: [0.2, 3], min_lookahead_m: [1, 15], max_lookahead_m: [2, 35],
  speed_lookahead_s: [0, 3], corner_acceleration_mps2: [0.201, 12],
  braking_factor: [0.101, 1], acceleration_factor: [0.101, 1], steering_gain: [0.251, 2],
  recovery_speed_mps: [0.501, 10], grade_lookahead_m: [2, 60],
  moderate_grade_threshold: [0.02, 0.5], steep_grade_threshold: [0.03, 0.8],
  moderate_grade_speed_mps: [0.501, 20], steep_grade_speed_mps: [0.501, 15],
  corner_anticipation_m: [0, 10],
};
export function validateAutoConfig(raw: unknown): AutoDriveConfig {
  if (!raw || typeof raw !== 'object' || Array.isArray(raw)) throw new Error('Invalid Auto Drive settings');
  const o = raw as Record<string, unknown>;
  if (Object.keys(o).some(k => !(k in DEFAULT_AUTO_CONFIG))) throw new Error('Unknown Auto Drive settings');
  const full = { ...DEFAULT_AUTO_CONFIG, ...o };
  if (!['right','centerline','racing'].includes(full.driving_style) ||
      !['normal','racing','hyper'].includes(full.speed_profile)) throw new Error('Invalid Auto Drive style/preset');
  for (const [key, [lo, hi]] of Object.entries(ranges)) {
    const value = full[key as keyof AutoDriveConfig];
    if (typeof value !== 'number' || !Number.isFinite(value) || value < lo || value > hi) {
      throw new Error('Invalid Auto Drive value: ' + key);
    }
  }
  if (full.min_lookahead_m > full.max_lookahead_m ||
      full.moderate_grade_threshold >= full.steep_grade_threshold ||
      full.steep_grade_speed_mps > full.moderate_grade_speed_mps) {
    throw new Error('Inconsistent Auto Drive settings');
  }
  return full;
}
export function loadAutoConfig(): AutoDriveConfig {
  try { return validateAutoConfig(JSON.parse(localStorage.getItem(AUTO_CONFIG_STORAGE) ?? 'null')); }
  catch { return { ...DEFAULT_AUTO_CONFIG }; }
}
export function selectSpeedProfile(config: AutoDriveConfig, profile: SpeedProfile): AutoDriveConfig {
  return { ...config, speed_profile: profile, target_max_speed_kmh: PRESET_SPEEDS_KMH[profile] };
}
