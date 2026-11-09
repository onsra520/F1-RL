# Proposed personal-repository TRAIN-5 amendment

**Draft for owner review. This document is not an approval, signature, permit, or instruction to start training.** It proposes a practical route for this personal repository while keeping the original scientific claims and historical evidence honest. The owner controls whether to adopt it.

## Proposed scope

- Six simulation-only TRAIN-5 runs: three from scratch and three initialized from the three registered frozen PPO checkpoints, using the seeds, budget, scenarios and evaluation plan in the current v4 protocol and review manifest.
- Research and development only. No claim of independently replicated science, historical C2 replay, real-car safety, physical calibration, or HELD-OUT generalization follows from these runs.
- The original checkpoints must remain unchanged. Only newly created output directories outside the source checkout may hold run outputs.

## Proposed review for a personal repository

The owner nominates GitHub account `onsra520` for an internal scientific/technical review. An actual review or acknowledgement on the exact proposed source and manifest may be recorded through GitHub if the person accepts; do not describe it as independent unless their role and independence are established. Owner review is recorded separately. This draft proposes replacing mandatory external reviewer PKI for this **internal engineering scope only**, not fabricating a review or changing existing scientific freeze or HELD-OUT conditions.

The existing local exploratory path already distinguishes explicit, scope-bound owner acknowledgement from a cryptographic identity claim. The TRAIN-5 internal path instead uses a separate, exact-run owner signature and a process-local capability. The owner has confirmed and enrolled their public key; no real run authorization has been signed. Existing straight-track exploratory permissions do not authorize TRAIN-5 and are not reused as if they did.

## Disclosures requiring an explicit owner decision

1. Fifteen old Actions job logs cannot be retrieved. Their metadata reports failure, without retained step details or artifacts. Searches of accessible local, reachable Git and indexed Drive history are bounded; global non-use of the proposed seeds is not proven. The old-seed exception does not automatically apply to the new seeds. The owner may decide whether to accept this disclosed limitation for internal simulations; the decision is currently **pending**.
2. The original historical C2 action/trajectory replay and its archived source files have not been recovered. The historical gate remains **open**. A new current-source engineering replay can provide separate evidence but cannot be called the original historical replay. The owner may decide whether an internal simulation can proceed with that gate still open and no historical C2 claim; the decision is currently **pending**.
3. Physics calibration, reset sampling ranges, and the development thresholds are engineering assumptions, not independent physical validation. Three policies per cohort and 60 episodes per track/condition/policy do not alone justify a causal or high-power statistical claim.

## Engineering work before any approval can authorize execution

1. Version and source-bind the amended scope, accepted disclosures, exact scenario factories, checkpoint identities, complete six-run manifests, source revision and output destinations. Retain prior research records unchanged.
2. The dedicated TRAIN-5 internal path now connects the registered runner, eight-selector environment and PPO guard, with owner signature and source/manifest scope. It validates actual factories, registered configuration and checkpoint hashes and isolates outputs. Review the implementation before enrolling a real owner key.
3. Fake-key and fake-model tests cover the six registered runs, missing or altered approval/checkpoint, replay, expiry and invalid scenario/roster. Confirm full regression and exact-commit Windows/Linux CI before accepting the implementation. Fixture material does not authorize real training.
4. Commit the exact candidate and pass Windows/Linux CI. Present the exact revision, manifest hashes, disclosure decisions, six-run budget and output location to the owner before requesting a real signature or local acknowledgement. A real approval must be made by the owner, not synthesized by an assistant.

Until all applicable steps and an explicit owner decision are complete, the existing TRAIN-5 and HELD-OUT guards remain closed.
