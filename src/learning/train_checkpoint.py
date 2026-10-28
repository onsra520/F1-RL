"""TRAIN-only checkpoint evaluation; never trains or touches held-out policies.

Any report from this recovery module remains provisional until a reviewed pilot
executor verifies model/run provenance, checkpoint schedule and protocol gates.
"""
from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
from pathlib import Path

from .algorithms.ppo import PPOModel
from .evaluation import EpisodeMetrics, EvaluationPolicy, evaluate_episode, summarize_track
from .pilot_preflight import TRAIN_IDS, build_pilot_preflight
from .reconstruction import RecoveryCandidate, build_training_scenarios


def evaluate_training_checkpoint(
    *,
    repository_root: str | Path,
    candidate_path: str | Path,
    runtime_config_path: str | Path,
    checkpoint_path: str | Path,
    episodes_per_track: int = 1,
    test_policy: EvaluationPolicy | None = None,
) -> dict[str, object]:
    """Evaluate exactly four TRAIN environments; never evaluate heldout tracks.

    The `test_policy` argument exists only for controlled mock verification:
    when supplied, the result is explicitly labeled non-scientific.
    """
    if (
        isinstance(episodes_per_track, bool)
        or not isinstance(episodes_per_track, int)
        or episodes_per_track <= 0
    ):
        raise ValueError("episodes_per_track must be a positive integer")
    archive = Path(checkpoint_path).resolve()
    if archive.suffix.lower() != ".zip" or not archive.is_file():
        raise ValueError("checkpoint_path must be an existing .zip model archive")
    if not isinstance(test_policy, (type(None), EvaluationPolicy)):
        raise ValueError("test_policy must satisfy EvaluationPolicy")

    preflight = build_pilot_preflight(
        repository_root=repository_root,
        candidate_path=candidate_path,
        runtime_config_path=runtime_config_path,
    )
    candidate = RecoveryCandidate.from_yaml(candidate_path)
    scenarios = build_training_scenarios(candidate)
    selected = tuple(item.track_id for item in scenarios.scenarios)
    if selected != TRAIN_IDS or preflight["training_track_ids"] != list(TRAIN_IDS):
        raise ValueError("TRAIN track boundary changed during checkpoint preparation")

    # Never evaluate a learned policy on HELD-OUT data. The caller supplies only
    # one archive; load happens after profile + split + static fingerprint checks.
    policy: EvaluationPolicy = (
        test_policy if test_policy is not None else PPOModel.load(archive,device="cpu")
    )
    summaries: list[dict[str, object]] = []
    for scenario in scenarios.scenarios:
        episodes: list[EpisodeMetrics] = []
        for _ in range(episodes_per_track):
            metrics = evaluate_episode(
                track_id=scenario.track_id,
                factory=scenario.factory,
                policy=policy,
            )
            if not isinstance(metrics, EpisodeMetrics) or metrics.track_id != scenario.track_id:
                raise ValueError("checkpoint evaluator returned mismatched TRAIN track evidence")
            episodes.append(metrics)
        summaries.append(asdict(summarize_track(tuple(episodes))))

    return {
        "schema_version": "wp7-recovery-train-checkpoint-v1",
        "checkpoint_name": archive.name,
        "checkpoint_sha256": sha256(archive.read_bytes()).hexdigest(),
        "policy_source": (
            "injected_test_policy_not_scientific_evidence"
            if test_policy is not None
            else "ppo_archive_requires_run_provenance_review"
        ),
        "scientific_evidence": False,
        "track_ids": list(selected),
        "episode_count_per_track": episodes_per_track,
        "track_summaries": summaries,
        "all_train_tracks_successful": all(
            summary["success_count"] == episodes_per_track
            for summary in summaries
        ),
        "heldout_rollouts": 0,
        "pilot_review_blockers": list(preflight["review_blockers"]),
        "candidate_source_fingerprints": list(preflight["fingerprints"]),
    }
