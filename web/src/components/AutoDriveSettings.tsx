import { useEffect, useState } from 'react';
import { DEFAULT_AUTO_CONFIG, PRESET_SPEEDS_KMH, selectSpeedProfile, validateAutoConfig,
  type AutoDriveConfig, type DrivingStyle, type SpeedProfile } from '../lib/autoDriveConfig';

const advanced: { key: keyof AutoDriveConfig; label: string; unit: string; step: number; min: number; max: number }[] = [
  { key: 'vehicle_width_m', label: 'Vehicle width', unit: 'm', step: 0.1, min: 0.5, max: 4 },
  { key: 'lane_offset_m', label: 'Lane offset (0=auto)', unit: 'm', step: 0.1, min: 0, max: 5 },
  { key: 'edge_margin_m', label: 'Edge margin', unit: 'm', step: 0.1, min: 0.2, max: 3 },
  { key: 'min_lookahead_m', label: 'Min lookahead', unit: 'm', step: 0.5, min: 1, max: 15 },
  { key: 'max_lookahead_m', label: 'Max lookahead', unit: 'm', step: 0.5, min: 2, max: 35 },
  { key: 'speed_lookahead_s', label: 'Speed lookahead', unit: 's', step: 0.05, min: 0, max: 3 },
  { key: 'corner_acceleration_mps2', label: 'Cornering accel', unit: 'm/s²', step: 0.1, min: 0.3, max: 12 },
  { key: 'braking_factor', label: 'Brake factor', unit: '×', step: 0.05, min: 0.15, max: 1 },
  { key: 'acceleration_factor', label: 'Accel factor', unit: '×', step: 0.05, min: 0.15, max: 1 },
  { key: 'steering_gain', label: 'Steering gain', unit: '×', step: 0.05, min: 0.3, max: 2 },
  { key: 'recovery_speed_mps', label: 'Recovery speed', unit: 'm/s', step: 0.5, min: 1, max: 10 },
  { key: 'grade_lookahead_m', label: 'Slope lookahead', unit: 'm', step: 1, min: 2, max: 60 },
  { key: 'moderate_grade_threshold', label: 'Moderate slope', unit: 'ratio', step: 0.01, min: 0.02, max: 0.5 },
  { key: 'steep_grade_threshold', label: 'Steep slope', unit: 'ratio', step: 0.01, min: 0.03, max: 0.8 },
  { key: 'moderate_grade_speed_mps', label: 'Moderate slope speed', unit: 'm/s', step: 0.5, min: 1, max: 20 },
  { key: 'steep_grade_speed_mps', label: 'Steep slope speed', unit: 'm/s', step: 0.5, min: 1, max: 15 },
  { key: 'corner_anticipation_m', label: 'Corner anticipation', unit: 'm', step: 0.5, min: 0, max: 10 },
];
export default function AutoDriveSettings({ config, onChange, disabled, physicalCapKmh }: {
  config: AutoDriveConfig;
  onChange: (config: AutoDriveConfig) => void;
  disabled: boolean;
  physicalCapKmh?: number;
}) {
  const [speedInput, setSpeedInput] = useState(String(config.target_max_speed_kmh));
  useEffect(() => setSpeedInput(String(config.target_max_speed_kmh)), [config.target_max_speed_kmh]);
  const update = (patch: Partial<AutoDriveConfig>) => {
    try { onChange(validateAutoConfig({ ...config, ...patch })); }
    catch { /* reject inconsistent values rather than sending invalid configuration */ }
  };
  return <section className="auto-drive-settings" aria-label="Auto Drive Settings">
    <div className="auto-settings-title"><strong>AUTO DRIVE SETTINGS</strong>
      <button type="button" disabled={disabled} onClick={() => onChange({ ...DEFAULT_AUTO_CONFIG })}>Reset defaults</button>
    </div>
    <label>DRIVING STYLE<select aria-label="Driving style" disabled={disabled} value={config.driving_style}
      onChange={event => update({ driving_style: event.target.value as DrivingStyle })}>
      <option value="right">Right-hand traffic</option><option value="centerline">Centerline</option>
      <option value="racing">Racing line (simulation)</option>
    </select></label>
    <label>SPEED PROFILE<select aria-label="Speed profile" disabled={disabled} value={config.speed_profile}
      onChange={event => onChange(selectSpeedProfile(config, event.target.value as SpeedProfile))}>
      {Object.keys(PRESET_SPEEDS_KMH).map(key => <option key={key} value={key}>{key.toUpperCase()}</option>)}
    </select></label>
    <label>MAX SPEED <span className="auto-speed-range">
      <input aria-label="Target max speed slider" type="range" min="1" max="300" step="1"
        disabled={disabled} value={config.target_max_speed_kmh}
        onChange={event => update({ target_max_speed_kmh: Number(event.target.value) })}/>
      <input aria-label="Target max speed km/h" type="text" inputMode="decimal"
        disabled={disabled} value={speedInput}
        onChange={event => { setSpeedInput(event.target.value); update({ target_max_speed_kmh: Number(event.target.value) }); }}
        onBlur={() => setSpeedInput(String(config.target_max_speed_kmh))}/>
      <small>km/h</small>
    </span></label>
    <small className="auto-drive-cap">Target {config.target_max_speed_kmh} km/h · Physical limit {physicalCapKmh == null
      ? 'determined by track vehicle' : physicalCapKmh.toFixed(1) + ' km/h'} · no synthetic velocity</small>
    <details className="auto-advanced"><summary>Advanced settings</summary>
      {advanced.map(field => <label key={field.key}>{field.label}
        <span className="auto-advanced-value">
          <input aria-label={field.label} type="number" min={field.min} max={field.max} step={field.step}
            disabled={disabled} value={config[field.key] as number}
            onChange={event => update({ [field.key]: Number(event.target.value) })}/>
          <small>{field.unit}</small>
        </span>
      </label>)}
    </details>
    {disabled && <small>Settings are locked for this session; restart to apply a new configuration.</small>}
  </section>;
}
