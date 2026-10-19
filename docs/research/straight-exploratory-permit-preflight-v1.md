# Straight exploratory permit: preparation contract v1

Owner decision on 2026-10-19 authorizes implementation and preflight only.
It does **not** authorize baseline execution, learning, GPU compute, or scientific
acceptance. The parent is Draft PR #48 at
`192ffc6ae62d5742dd454d2d2ed4165b99d04c31`.

## Scope and executable interfaces

Use a separate worktree and the existing `F1RLEnv`, `PPOModel`, project PPO backend,
and `spatial_rotating_owner_v3` factory. Only `straight-sprint-v1` is executable.
The underlying ground integrator is `spatial_rotating_compliant_lifecycle_v3`.
No physics, observation, action, reward, canonical asset, scientific owner
contract, threshold, or scientific permission is changed. Scientific registration
is inspected as the existing simulator configuration; this exploratory permit
does not turn that registration into training authority.

From the final clean source checkout:

```powershell
python -B scripts/exploratory_ppo.py preflight --output-directory D:/Projects/.evidence/F1-RL/exploratory-straight-pilot-v1/run-001 --permit D:/Projects/.evidence/F1-RL/exploratory-straight-preflight-v1/permit.json
```

This creates only an external preparation document. It does not construct a PPO
model, reset Gym, execute a policy, or collect optimizer updates. It refuses to
overwrite the permit or use an existing execution directory. The final delivery
report records the actual interpreter, source HEAD, permit digest and checks.

The future executable training command is:

```powershell
python -B scripts/exploratory_ppo.py execute --permit D:/Projects/.evidence/F1-RL/exploratory-straight-preflight-v1/permit.json --authorization D:/Projects/.evidence/F1-RL/exploratory-straight-preflight-v1/owner-execution.json
```

**Do not run it to train during preparation.** There is no production execution
grant. The committed exploratory owner-key registry is empty. This command denies
missing/invalid authority before model or execution directory allocation.

## Authority and isolation

The prepare document has `prepared_not_authorized` state. It binds exact clean
commit, actual source-file hashes, protocol/physics hashes, runtime/package/CPU
identity, seeds, PPO settings, fixed budget and an external output directory.
It records scientific `training_authorized=false` and exploratory
`exploratory_execution_authorized=false`. A document hash is integrity, not consent.

Only a separately authenticated owner key in
`configs/exploratory/straight-owner-keys-v1.json` can authorize execution. No key
embedded in an envelope or caller-provided `authorized=true` flag is trusted.
The owner supplies the public key through an authenticated channel; no agent
generates an owner identity or private key. Commit a real anchor **before** a
fresh source pin and fresh permit preparation. No WP7 trust registry is edited.

The execution envelope has exactly `payload`, `key_id`, `signature_hex`.
Its payload has exactly:

```json
{
  "schema_version": "straight_exploratory_execution_v1",
  "purpose": "execute_exploratory_only",
  "permit_sha256": "<exact prepared permit SHA-256>",
  "issued_at_utc": "<owner-issued UTC timestamp>",
  "expires_at_utc": "<owner-issued UTC timestamp>"
}
```

Ed25519 signs the ASCII domain prefix
`F1-RL/exploratory/straight-execution/v1\0` followed by canonical payload JSON
(`sort_keys=True`, separators `(',', ':')`, finite values). Keys include
`key_id`, `public_key_hex`, `revoked`, `not_before_utc`, `expires_at_utc`.
Preparation decisions and scientific review signatures have different purposes
and cannot authorize this path. Expiry and revocation fail closed.

The parent verifies before reserving a one-use execution directory. A separate
worker independently verifies before claiming it. A process-bound, one-use
session issues capabilities only to its reconstructed Straight environments.
PPO creation and `learn()` reauthenticate those capabilities, including wrapped
single-env vectors; inference environments cannot grant learning permission.
Ordinary research environments still require scientific admission. Their
`requires_scientific_admission` property stays true. Scientific `TrainingRunner`
and registered cell evaluation remain independently closed. CUDA learning is
rejected in both construction and learning, including loaded models.

These are application execution controls, not an OS sandbox against someone
running arbitrary replacement Python code or controlling the repository/filesystem.

## Budget and measurements

Fixed maximum: 32,768 learning interactions, one environment, PPO seed 0,
TRAIN perturbation seed 1000, CPU only, Torch intra-op at most 4 threads and
interop 1, BLAS/OpenMP pools at most 4 threads. These are compute-pool limits,
not a promise that the Python process has only four total OS service threads.
No GPU device is selected. The executable bootstrap hides CUDA devices before
importing the ML stack. PPO hyperparameters reuse the existing pilot profile
(`n_steps=256`, `batch_size=64`, `n_epochs=10`, learning rate `0.0003`).
The 32,768-step cap is an exact multiple of the 256-step rollout.

One parent deadline covers authorization/startup, random baseline, untrained
baseline, learning, final evaluation and replay: at most 3,600 seconds before
process termination is requested. The worker also enforces the remaining time,
grant expiry, and the learning-step cap before interaction. Baseline/evaluation
interactions do not count as learning timesteps but consume the same wall budget.
Runtime duration is unknown until authorized execution. Baselines can exhaust
the budget before learning; there is no automatic budget extension.

Random, untrained PPO, and trained-final PPO each use 32 fixed TRAIN perturbation
seeds 10000–10031. Corresponding episodes share the same reset/draw identity.
Random actions are uniform over the existing action bounds with an independent
per-episode NumPy RNG. PPO evaluation uses deterministic inference. The same
initial PPO model supplies the untrained baseline and subsequently learns.
No scripted driving actions enter PPO evaluation or training.

Per episode record simulator outcome validity/success, normalized route high-water
progress, collision event count, hard-route terminal events, simulator elapsed
duration, return, steps and full step-record trace digest. No invalid episode is
replaced. Failures remain failures. Simulator duration is the reported physical
tick clock; owner terminal reward-prefix semantics remain unchanged.

Final saved-policy reload and inference supply the trained comparison. Replay
the first trained episode with the same seed and compare metrics plus trace
digest exactly. Compare all 32 episodes, not selected successes. Intermediate
checkpoints are diagnostic only. No statistical scientific acceptance claim is
made from this exploratory cohort. Learning evidence requires positive actual
timesteps, positive observed optimizer calls and changed weight hashes;
autonomous course completion is a separate observed outcome, never inferred
from software tests or parameter changes.

## Checkpoints, timeout and recovery

Save an untrained archive, checkpoints every 8,192 actual steps, and trained-final
after learning returns. ZIPs are written to pending files then atomically renamed;
an artifact ledger records raw SHA-256, actual timesteps, PPO epoch-update counter
and observed optimizer-step count. SB3 step callbacks run before the current
rollout's optimizer updates, so interval checkpoints explicitly record the
updates actually present; their step labels do not assert otherwise.

SB3 CSV training metrics, learning-step/update counters, episode returns/outcomes,
evaluation JSONL, state, supervisor status, worker stdout/stderr, replay and final
comparison are retained externally. JSON state replacement is atomic; JSONL
records are flushed and fsynced. Parent timeout preserves completed checkpoints
and logs. A killed pending ZIP or partial last JSONL line is not accepted as a
verified artifact. The supervisor status controls whether a stale worker
`running` state is terminal. A failure may leave a valid final archive without a
completed evaluation; that is not an autonomous success claim.

No auto-resume, redraw, checkpoint selection or run-directory reuse is provided.
Offline recovery inspects the ledger and verifies completed archive hashes. A
new training attempt or continuation requires a new owner-approved execution
scope and fresh one-use output identity; it cannot silently restart the budget.

## Verification scope and remaining inputs

Preflight verification uses signature/negative/boundary tests and fake worker,
environment and model orchestration. It must not perform a real PPO learning run,
random-policy baseline or scripted driving rollout. Existing scientific guard
tests may construct fixtures/models without collecting learning samples.

Final delivery must state exact checks/results and distinguish simulated test
orchestration from real learning. No trained checkpoint, empirical learning
curve or autonomous success exists from preparation. Remaining execution inputs
are a separately authenticated real owner public key, a fresh source-bound
permit after its commit, and explicit owner execution authorization signed over
that permit. Stop after delivering verified preflight.
