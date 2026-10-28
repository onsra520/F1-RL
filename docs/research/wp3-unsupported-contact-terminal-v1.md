# WP3 training failure policy v1

This policy is part of `wp3-abcd-12-models-v2` and applies identically to all
12 TRAIN tuning runs. It is a training-only response to the exact C0 failure
observed on `slalom-v1`; it does not change the registered physics, observation
or action contracts, or the evaluation procedure.

The policy triggers only when the spatial contact integrator raises
`SuspensionResolutionError` with `failure_code == "unsupported_contact"`,
which is assigned only for an `outside` support verdict whose reason is
`invalid endpoint body or wheel support`. The failed physics transition is not
accepted. Training terminates the episode from the last certified simulator
snapshot and emits a transition marked `physics_abort_terminal`, preserving the
original `invalid_physical_transition` research outcome and exact exception
detail for audit. The training reward for that terminal row is zero registered
reward minus the candidate's existing squared-lateral penalty and fixed
terminal penalty. It is counted separately from natural and curriculum
terminations.

Every other `SuspensionResolutionError`, every other simulator error, and every
error under the smoke profile still aborts the run. Evaluation does not install
this policy: an unsupported physical transition invalidates that evaluation
cell under the frozen no-redraw rule. The registered 4 m hard-route boundary,
finish/checkpoint rules, physics configuration, observations, and actions are
unchanged.

The v1 failed C0 remains immutable and is not reused as a v2 model. Because
this rule changes training termination/reward semantics, all 12 models must be
trained under the v2 frozen source and configuration before comparison.
