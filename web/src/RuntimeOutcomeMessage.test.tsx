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
  it('fails closed on a contradictory registered PPO invalid sample', () => {
    expect(runtimeOutcome({ ...base, episode_result: { valid_sample: false, successful_finish: true } })?.kind).toBe('invalid');
  });
});
