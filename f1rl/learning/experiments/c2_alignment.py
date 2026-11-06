"""Check a proposed canonical C2 draft against the archived TRAIN recovery contract.

This is an identity/integrity check, not a release gate or a migration of the
two original 200k models to protocol v2. It constructs no HELD-OUT episodes.
"""
from __future__ import annotations

from .definition import ExperimentDefinition


def verify_draft_c2_alignment(
    canonical: ExperimentDefinition,
    recovery: object,
) -> None:
    """Fail closed if a canonical v2 draft invents dynamics or changes C2.

    Importing RecoveryCandidate here avoids a circular dependency when the
    experiments package is imported by the original WP7 reconstruction.
    """
    from ..reconstruction import RecoveryCandidate

    if not isinstance(canonical, ExperimentDefinition) or not isinstance(recovery, RecoveryCandidate):
        raise ValueError("alignment expects an ExperimentDefinition and RecoveryCandidate")
    if (
        canonical.metadata.protocol_version != 2
        or canonical.metadata.status != "draft"
        or canonical.training.protocol_status != "pilot_owned"
        or canonical.seeds.ppo
        or canonical.seeds.sampling
    ):
        raise ValueError("C2 alignment is only valid for unresolved draft protocols")

    base = recovery.experiment
    if base.metadata.protocol_version != 1 or base.metadata.status != "draft":
        raise ValueError("original v1 recovery baseline changed")
    expected_behavior = {
        "dynamics": canonical.semantics.dynamics,
        "navigation": canonical.semantics.navigation,
        "observation": canonical.semantics.observation,
        "reward": canonical.semantics.reward,
        "action": canonical.semantics.action,
        "race": canonical.semantics.race,
    }
    if recovery.behavior != expected_behavior:
        raise ValueError("canonical v2 semantics diverge from TRAIN recovery candidate")
    if (
        canonical.vehicle != base.vehicle
        or canonical.simulation != base.simulation
        or canonical.observation != base.observation
        or canonical.episode != base.episode
        or canonical.tracks != base.tracks
        or canonical.evaluation != base.evaluation
    ):
        raise ValueError("canonical v2 non-training environment/split differs from WP1B/WP3")
    if (
        canonical.reward.route_progress_weight_per_meter != recovery.route_progress_weight_per_meter
        or canonical.reward.collision_penalty != recovery.collision_penalty
        or canonical.reward.time_penalty_per_second != recovery.time_penalty_per_second
        or canonical.reward.offtrack_penalty != recovery.offtrack_penalty
        or canonical.offtrack is None
        or canonical.offtrack.lateral_limit_meters != recovery.offtrack.lateral_limit_meters
        or canonical.offtrack.max_consecutive_steps != recovery.offtrack.max_consecutive_steps
    ):
        raise ValueError("canonical v2 reward/offtrack numerical contract drifted")
    if canonical.training.runtime_config.name != "wp7-pilot-200k.yml":
        raise ValueError("draft C2 must explicitly reference its archived pilot profile")
