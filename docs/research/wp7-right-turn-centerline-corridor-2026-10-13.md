# WP7 Recovery — Right Turn TRAIN road-union / corner-cutting audit

## Decision: evidence verified, **route-corridor policy NOT APPROVED**

Right Turn has **two perpendicular 12 m-wide horizontal road surfaces**, overlapping near the 90-degree junction. The current simulator checks per-step vehicle centre positions, but a centre could cross a non-road wedge between consecutive simulator samples.

To remove this uncertainty for the four-TRAIN exploratory pilots without changing physics, `f1rl/learning/corridor_audit.py` provides an **exact 2D segment clipping algorithm against the union of the two finite road rectangles**. The independent audit replays only the archived Right Turn final PPO episodes, on original source and assets. It refuses slanted ramp patches instead of flattening their geometry.

### Result, real final PPO on original Right Turn TRAIN

| Metric | Calibration (0/0) | Confirmation (1/1) |
|---|---:|---:|
| Final episode / archived match | 1/1 PASS | 1/1 PASS |
| Centre distance travelled outside road union | **0.000 m** | **0.000 m** |
| Endpoint samples outside road union | 0 | 0 |
| Max *sampled* distance to authored route | 5.511796 m | 4.415475 m |
| Conservative upper bound for distance to route **between samples** | 6.042907 m | 4.595270 m |
| Simulator off-track lateral threshold | 6 m | 6 m |

The centre path lies within the two road rectangles in both runs, including **every straight segment between consecutive 0.1 s samples**. This proves **continuous road-centreline inclusion under linear interpolation**, not physical body road inclusion.

The max sampled route deviation was already measured in the earlier trajectory audit, and exactly matches it here. For the unobserved interior of each segment, distance to the fixed authored polyline is **1-Lipschitz** in horizontal position. The audit uses `max(endpoint_distances) + segment_horizontal_length / 2` as a conservative upper bound per segment, and takes the maximum across segments.

**Important:** Calibration's bound `6.042907 m` exceeds the `6 m` simulator off-track threshold. This **does not mean the vehicle actually exceeded 6 m**, nor imply an off-track termination: an upper bound is not an observed excursion, and the runtime rule also requires 10 consecutive over-threshold *sampled* states. It means the conservative bound alone cannot prove that the complete interpolated path stays within a 6 m authored-route corridor. The final policy still finished the track by the unchanged original rules.

### Geometry approval remains a blocker

The official vehicle length/width/height and local body-centre offset **are still unspecified**. No vehicle footprint was invented or silently bound into the original model. Geometrically valid centreline motion does not certify wheel/contact patch or body corners remain on the road.

Separately, the fact that the vehicle stays within road-surface union does **not** approve how much shortcutting across the authored route should be allowed. A scientifically defensible corner-cutting rule requires course-design intent and a threshold chosen independently of the observed results. **Do not choose 5.6 m merely because the existing calibration attained 5.511796 m.**

### Artifact and provenance

Independent report:

`D:\Projects\.archives\F1-RL\wp7-exploratory\audits\wp7-right-turn-centerline-corridor-v1.json`

SHA256 (verified after creation):
`cdf22fc26b9fdee85d82c56dea0d26c900cc5e318010d989df3fca51f1049a62`.

Both reports link original run IDs, original evidence-manifest SHA256, PPO final-model SHA256, original training-source revision `1478cf6b9f671a08c5ace3878c8b19f34da951cc`, and zero HELD-OUT rollouts. No historical file is rewritten.

This new audit becomes the third linked evidence bundle in **Freeze Readiness Gate v2**, along with the original XZ trajectory audit and 3D swept proxy audit. The gate remains fail-closed (`scientific_freeze_permitted=false`, `heldout_evaluation_permitted=false`).

### Reproduce using a new output path

```powershell
cd D:\Projects\.worktrees\F1-RL\wp7-rebuild
.\.venv\Scripts\python.exe scripts\wp7_audit_right_turn_corridor.py `
  --run "D:\Projects\.archives\F1-RL\wp7-exploratory\2026-10-11_205037_ppo-multitrack-seed0-sample0" `
  --run "D:\Projects\.archives\F1-RL\wp7-exploratory\2026-10-12_203835_ppo-multitrack-seed1-sample1" `
  --output "D:\Projects\.archives\F1-RL\wp7-exploratory\audits\wp7-right-turn-centerline-corridor-rerun.json"
```

The CLI refuses overwriting existing audit evidence and paths inside the Git worktree. No HELD-OUT asset or rollout, PPO re-training, changes to original simulator, `main` merge or WP8 was performed.

**Next review:** register actual body geometry, define permissible route corridor from course-design intent, specify a prospective robustness protocol, and approve storage backup / capture prospective environment provenance before considering a new scientifically frozen candidate.
