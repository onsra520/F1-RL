# Physics v4 handoff evidence

Read the [draft plan](../PHYSICS-V4-LEFT-TURN-PLAN.md) and [current spec](../PHYSICS-V4-LEFT-TURN-SPEC.md) before implementation. The handoff branch starts at root baseline 2417570f32f6879c2651bce37d88fd6ac83a2ea4; docs may be committed afterward. No implementation changes are included in this handoff preparation.

Historical evidence is copied byte-for-byte into evidence/. Verify every evidence file using evidence/SHA256SUMS. The evidence directory's .gitattributes preserves original bytes across Git checkouts. These files document original runs and may contain machine-specific paths; those paths are provenance, not instructions to use the same directory. Run new diagnostics from a configurable output directory outside the checkout. No HELD-OUT geometry is included.

- [Phase 2A report](evidence/phase2a-root-cause-audit-20261106/report.md), three root-cause JSON artifacts, audit manifest and replay drivers.
- [Phase 2B report](evidence/phase2b-smoke-profile-wiring-20261106/PHASE-2B-SMOKE-PROFILE-FIX-REPORT.md), eight-case replay and driver. The file named summary-bug-attempt is a historical failed intermediate, not the final replay result.
- [Phase 2C report](evidence/phase2c-ppo-smoke-20261106/synced-smoke-run/PHASE-2C-PPO-SMOKE-REPORT.md), original reports, restore/design note, preflight and initial-commit audit, 12-case results, replay driver and all 12 per-step episode traces.
- The original Phase 2C report marked 73-test acceptance BLOCKED because the restore script was absent then. It was subsequently restored; the historical report remains unchanged. See the restore/design note and baseline commit for the later 73-test pass.
- Phase 2C shows 4 finishes, 8 valid physical crashes and zero invalid transitions. Initial PPO diagnosis exists; the draft plan calls for a more detailed nine-case investigation.
- The restored baseline excludes documentation and historical evidence. The current handoff package supplies the relevant copied artifacts but does not imply a new run, retraining or scientific Physics signoff.

The historical design-options note describes an earlier proposal to modify the existing roster. It is preserved as evidence; the current spec and draft plan supersede that proposal by using a separate five-track configuration.
