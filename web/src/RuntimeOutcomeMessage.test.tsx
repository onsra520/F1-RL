import { describe, expect, it } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import RuntimeOutcomeMessage, { runtimeOutcome } from './RuntimeOutcomeMessage';
import type { RuntimeSessionStatus } from './lib/runtime';

const base: RuntimeSessionStatus = {
  protocol_version: 'f1rl_runtime_v1', session_id: 's412', mode: 'ppo',
  model_id: 'A-seed0', track_id: 'straight-sprint-v1', seed: 412,
  status: 'finished', steps: 500, max_steps: 500, step_progress: 1,
  latest_frame: null, error: null, provenance: 'TRAIN-development',
  episode_result: { status: 'finished', valid_sample: true, successful_finish: false, reason: 'step_limit', steps: 500 },
};
const view = (status: Partial<RuntimeSessionStatus>) => renderToStaticMarkup(<RuntimeOutcomeMessage status={{ ...base, ...status }} />);

describe('Backend-authoritative runtime result rendering', () => {
  it('renders successful valid PPO finish', () => {
    const html = view({ episode_result: { valid_sample: true, successful_finish: true, reason: 'finish' } });
    expect(html).toContain('Successful Finish');
    expect(html).toContain('data-outcome="success"');
  });
  it('labels a Hybrid development finish as unvalidated', () => {
    const html = view({ mode: 'hybrid', provenance: 'hybrid-development', episode_result: {
      valid_sample: false, successful_finish: true, reason: 'finish',
    } });
    expect(html).toContain('Development Finish · Not Validated');
    expect(html).toContain('data-outcome="uncertified"');
    expect(html).not.toContain('Successful Finish');
  });
  it('renders valid PPO step limit without INVALID', () => {
    const html = view({});
    expect(html).toContain('Valid · Step Limit / Non-finish');
    expect(html).not.toContain('Invalid');
    expect(html).toContain('step_limit');
  });
  it('renders physical failure as invalid, preserving error', () => {
    const html = view({ status: 'failed', episode_result: { status: 'failed', valid_sample: false, successful_finish: false, reason: 'SuspensionResolutionError: invalid endpoint' } });
    expect(html).toContain('Invalid · Physics/Simulator Error');
    expect(html).toContain('SuspensionResolutionError');
  });
  it('renders unsupported road departure as a driving failure', () => {
    const status: RuntimeSessionStatus = {
      ...base,
      status: 'failed',
      episode_result: {
        status: 'failed', valid_sample: false, successful_finish: false,
        reason: 'SuspensionResolutionError: track_boundary_exit: front_right wheel center left road support',
      },
    };
    const html = view(status);
    expect(html).toContain('Driving Failure · Track Boundary');
    expect(html).toContain('A wheel left the road. Start a new run to try again.');
    expect(html).not.toContain('Physics/Simulator Error');
    expect(runtimeOutcome(status)?.kind).toBe('driving_failure');
  });
  it('identifies an unaccepted obstacle collision as a driving failure', () => {
    const status: RuntimeSessionStatus = {
      ...base, mode: 'auto', provenance: 'custom-development', status: 'failed',
      episode_result: { status: 'failed', valid_sample: false, successful_finish: false, reason: 'collision' },
    };
    const html = view(status);
    expect(html).toContain('Driving Failure · Obstacle Contact');
    expect(html).toContain('transition was not accepted');
    expect(html).not.toContain('Physics/Simulator Error');
    expect(runtimeOutcome(status)?.kind).toBe('driving_failure');
  });
  it.each([
    'ValueError: Auto Drive cannot certify a road-width-safe corridor around barrier; move the obstacle in Build Editor',
    'RuntimeFault: Hybrid centerline does not avoid obstacles; choose another driving style or remove obstacles',
  ])('explains an unsupported obstacle configuration: %s', (reason) => {
    const html = view({ mode: 'hybrid', provenance: 'hybrid-development', status: 'failed',
      episode_result: { status: 'failed', valid_sample: false, successful_finish: false, reason } });
    expect(html).toContain('Unsupported · Obstacle Configuration');
    expect(html).not.toContain('Physics/Simulator Error');
  });
  it.each(['cancelled', 'stopped'] as const)('renders %s without implying invalidity', (status) => {
    expect(view({ status, episode_result: null })).toContain('Cancelled / Stopped');
    expect(runtimeOutcome({ ...base, status, episode_result: null })?.kind).toBe('stopped');
  });
  it('does not certify manual and custom non-finish as PPO-valid', () => {
    const html = view({ mode: 'manual', provenance: 'manual-development', episode_result: { valid_sample: false, successful_finish: false } });
    expect(html).toContain('uncertified');
    expect(html).not.toContain('Invalid');
  });
  it('does not show a terminal outcome for running/transport errors', () => {
    expect(view({ status: 'running', error: 'network unavailable', episode_result: null })).toBe('');
  });
  it('labels a finished but uncertified PPO episode without inventing a physics error', () => {
    const result = runtimeOutcome({ ...base, episode_result: { valid_sample: false, successful_finish: true, reason: 'finish' } });
    expect(result?.kind).toBe('uncertified');
    expect(result?.label).toContain('Not Validated');
  });
  it('labels an uncertified PPO terminal non-finish without inventing a physics error', () => {
    const result = runtimeOutcome({ ...base, episode_result: { valid_sample: false, successful_finish: false, reason: 'episode_terminal' } });
    expect(result?.kind).toBe('uncertified');
  });
});
