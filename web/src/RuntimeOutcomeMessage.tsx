import type { RuntimeSessionStatus } from './lib/runtime';

type OutcomeView = { kind: 'success' | 'valid_nonfinish' | 'invalid' | 'driving_failure' | 'stopped' | 'uncertified'; label: string; detail: string };

/** Interpret only backend-declared terminal states; transport errors are displayed separately. */
export function runtimeOutcome(status: RuntimeSessionStatus): OutcomeView | null {
  const terminal = status.episode_result;
  const reason = typeof terminal?.reason === 'string' && terminal.reason.trim()
    ? terminal.reason : null;
  if (status.status === 'cancelled' || status.status === 'stopped') {
    return { kind: 'stopped', label: 'Cancelled / Stopped', detail: reason ?? status.status };
  }
  if (status.status === 'failed') {
    if (reason?.includes('Auto Drive cannot certify a road-width-safe corridor') ||
        reason?.includes('Hybrid centerline does not avoid obstacles')) {
      return { kind: 'invalid', label: 'Unsupported · Obstacle Configuration', detail: reason };
    }
    if (reason === 'collision') {
      return {
        kind: 'driving_failure',
        label: 'Driving Failure · Obstacle Contact',
        detail: 'The collision transition was not accepted. Move the obstacle or try a different route.',
      };
    }
    if (reason?.includes('track_boundary_exit:')) {
      return {
        kind: 'driving_failure',
        label: 'Driving Failure · Track Boundary',
        detail: 'A wheel left the road. Start a new run to try again.',
      };
    }
    return { kind: 'invalid', label: 'Invalid · Physics/Simulator Error', detail: reason ?? status.error ?? 'simulation failed' };
  }
  if (status.status !== 'finished' || !terminal) return null;
  // Custom/manual sessions do not carry the registered-PPO valid_sample certification.
  if (status.mode === 'ppo' && terminal.valid_sample !== true && status.provenance === 'TRAIN-development') {
    return { kind: 'invalid', label: 'Invalid · Physics/Simulator Error', detail: reason ?? 'uncertified PPO episode' };
  }
  if (terminal.successful_finish === true) {
    if (terminal.valid_sample !== true || status.provenance !== 'TRAIN-development') {
      return { kind: 'uncertified', label: 'Development Finish · Not Validated', detail: reason ?? 'finish reached' };
    }
    return { kind: 'success', label: 'Successful Finish', detail: reason ?? 'finish reached' };
  }
  if (status.mode === 'ppo' && terminal.valid_sample === true) {
    return { kind: 'valid_nonfinish', label: 'Valid · Step Limit / Non-finish', detail: reason ?? 'episode ended without finish' };
  }
  return { kind: 'uncertified', label: 'Completed · Non-finish (uncertified)', detail: reason ?? 'manual/custom development episode ended' };
}

export default function RuntimeOutcomeMessage({ status }: { status: RuntimeSessionStatus }) {
  const outcome = runtimeOutcome(status);
  if (!outcome) return null;
  return <p className="runtime-result" data-outcome={outcome.kind} aria-live="polite">
    <strong>{outcome.label}</strong> · {outcome.detail}
  </p>;
}
