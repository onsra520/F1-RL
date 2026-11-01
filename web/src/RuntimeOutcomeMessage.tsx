import type { RuntimeSessionStatus } from './lib/runtime';

type OutcomeView = { kind: 'success' | 'valid_nonfinish' | 'invalid' | 'stopped' | 'uncertified'; label: string; detail: string };

/** Interpret only backend-declared terminal states; transport errors are displayed separately. */
export function runtimeOutcome(status: RuntimeSessionStatus): OutcomeView | null {
  const terminal = status.episode_result;
  const reason = typeof terminal?.reason === 'string' && terminal.reason.trim()
    ? terminal.reason : null;
  if (status.status === 'cancelled' || status.status === 'stopped') {
    return { kind: 'stopped', label: 'Cancelled / Stopped', detail: reason ?? status.status };
  }
  if (status.status === 'failed') {
    return { kind: 'invalid', label: 'Invalid · Physics/Simulator Error', detail: reason ?? status.error ?? 'simulation failed' };
  }
  if (status.status !== 'finished' || !terminal) return null;
  // Custom/manual sessions do not carry the registered-PPO valid_sample certification.
  if (status.mode === 'ppo' && terminal.valid_sample !== true && status.provenance === 'TRAIN-development') {
    return { kind: 'invalid', label: 'Invalid · Physics/Simulator Error', detail: reason ?? 'uncertified PPO episode' };
  }
  if (terminal.successful_finish === true) {
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
