import { describe, expect, it } from 'vitest';
import { DEFAULT_AUTO_CONFIG, PRESET_SPEEDS_KMH, selectSpeedProfile, validateAutoConfig } from './autoDriveConfig';

describe('Auto Drive immutable snapshot validation', () => {
  it('defaults to right-hand, Normal speed and has three speed presets', () => {
    expect(DEFAULT_AUTO_CONFIG.driving_style).toBe('right');
    expect(Object.keys(PRESET_SPEEDS_KMH)).toEqual(['normal', 'racing', 'hyper']);
    expect(selectSpeedProfile(DEFAULT_AUTO_CONFIG, 'hyper').target_max_speed_kmh).toBe(240);
    expect(DEFAULT_AUTO_CONFIG.target_max_speed_kmh).toBe(45);
  });
  it('rejects invalid and inconsistent settings before backend transmission', () => {
    expect(() => validateAutoConfig({ ...DEFAULT_AUTO_CONFIG, target_max_speed_kmh: Infinity })).toThrow();
    expect(() => validateAutoConfig({ ...DEFAULT_AUTO_CONFIG, speed_profile: 'unsafe' })).toThrow();
    expect(() => validateAutoConfig({ ...DEFAULT_AUTO_CONFIG, min_lookahead_m: 20 })).toThrow();
    expect(() => validateAutoConfig({ ...DEFAULT_AUTO_CONFIG, unknown: true })).toThrow();
  });
  it('accepts the backend speed range and rejects values outside it', () => {
    for (const speed of [0.5, 1, 300]) expect(validateAutoConfig({ target_max_speed_kmh: speed }).target_max_speed_kmh).toBe(speed);
    for (const speed of [0, -1, 300.1, NaN, Infinity]) expect(() => validateAutoConfig({ target_max_speed_kmh: speed })).toThrow();
  });
});
