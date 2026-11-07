"""Deterministic, TRAIN-only WP7 candidate preflight; never calls PPO training.

This report is intentionally non-executable and marked review_required.
It fingerprints the *actual* recovery candidate and 200k profile, instead of
misusing the WP6 provenance role that points to the unrelated 50k config.
"""
from __future__ import annotations

from hashlib import sha256
from math import acos, degrees, hypot, tan
from pathlib import Path

from ..sandbox.geometry.rotation import Quaternion
from ..track_assets import load_track_asset, load_track_manifest
from .reconstruction import RecoveryCandidate, build_training_scenarios
from .training.config import TrainingRunConfig


TRAIN_IDS = (
    "straight-sprint-v1",
    "slalom-v1",
    "right-turn-v1",
    "ramp-v1",
)
PILOT_MILESTONES = tuple(range(25_000, 200_001, 25_000))
SEED_PAIRS = ((0, 0, "calibration"), (1, 1, "confirmation"))


def _fingerprint(root: Path, file_path: Path) -> dict[str, str]:
    path = file_path.resolve(strict=True)
    try:
        rel = path.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError("pilot fingerprint path must remain in repository") from exc
    if not path.is_file():
        raise ValueError("pilot fingerprint path must be a file")
    return {"path": rel, "sha256": sha256(path.read_bytes()).hexdigest()}


def _heading_change_deg(points: tuple) -> float:
    maximum = 0.0
    for first, middle, last in zip(points, points[1:], points[2:]):
        a = (middle.x-first.x, middle.z-first.z)
        b = (last.x-middle.x, last.z-middle.z)
        na, nb = hypot(*a), hypot(*b)
        if na <= 1e-9 or nb <= 1e-9:
            raise ValueError("authored route has zero horizontal heading")
        cosine = max(-1.0, min(1.0, (a[0]*b[0]+a[1]*b[1])/(na*nb)))
        maximum = max(maximum, degrees(acos(cosine)))
    return maximum


def _rectangle_clearance(points: tuple, obstacles: tuple) -> float | None:
    if not obstacles:
        return None
    nearest = float("inf")
    for obstacle in obstacles:
        if obstacle.volume.orientation != Quaternion.identity():
            # A rotated obstacle needs local-frame rectangle distance.
            # Reject rather than falsely claim an axis-aligned clearance.
            raise ValueError("route preflight cannot certify rotated obstacle rectangles")
    for a, b in zip(points, points[1:]):
        length = hypot(b.x-a.x, b.z-a.z)
        subdivisions = max(1, int(length / 0.1) + 1)
        for i in range(subdivisions+1):
            fraction = i / subdivisions
            x = a.x+(b.x-a.x)*fraction
            z = a.z+(b.z-a.z)*fraction
            for obstacle in obstacles:
                volume = obstacle.volume
                dx = max(0.0, abs(x-volume.center.x)-volume.half_extents.x)
                dz = max(0.0, abs(z-volume.center.z)-volume.half_extents.z)
                nearest = min(nearest, hypot(dx,dz))
    return nearest


def build_pilot_preflight(
    *,
    repository_root: str | Path,
    candidate_path: str | Path,
    runtime_config_path: str | Path,
) -> dict[str, object]:
    """Report readiness risks. Does not start runs, save models, or write output."""
    root = Path(repository_root).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("repository_root must be a directory")
    candidate_file = Path(candidate_path).resolve(strict=True)
    profile_file = Path(runtime_config_path).resolve(strict=True)
    candidate = RecoveryCandidate.from_yaml(candidate_file)
    config = TrainingRunConfig.from_yaml(profile_file)
    reference = TrainingRunConfig.from_yaml(root / "config.yml")
    if (
        config.total_timesteps != 200_000
        or config.checkpoint_every_steps != 25_000
        or config.ppo.seed != 0
        or config.sampling_seed != 0
        or config.ppo.device != "cpu"
        or config.vector_env.n_envs != 8
        or config.vector_env.backend != "dummy"
        or config.vector_env.start_method != "spawn"
        or config.ppo != reference.ppo
        or config.vector_env != reference.vector_env
    ):
        raise ValueError("pilot recovery profile differs from registered 200k CPU 8-env baseline")
    if candidate.status != "candidate":
        raise ValueError("pilot preflight requires recovery candidate status")
    if candidate.experiment.tracks.training_ids != TRAIN_IDS:
        raise ValueError("pilot TRAIN split identity changed")
    # Scenario instantiation validates spaces and reward binding, without
    # training or evaluating any learned policy.
    scenarios = build_training_scenarios(candidate)
    if tuple(s.track_id for s in scenarios.scenarios) != TRAIN_IDS:
        raise ValueError("pilot scenarios must be exactly the four TRAIN tracks")

    experiment = candidate.experiment
    fingerprints = [
        _fingerprint(root, candidate_file),
        _fingerprint(root, profile_file),
        _fingerprint(root, candidate_file.parent / "../experiments/ppo-generalization-v1.yml"),
        _fingerprint(root, experiment.tracks.manifest),
    ]
    audits: list[dict[str, object]] = []
    blockers: list[str] = []
    # Pilot audits TRAIN-4 assets; preserve HELD-OUT metadata without opening geometry.
    train_manifest = {entry.track_id: entry for entry in load_track_manifest(
        experiment.tracks.manifest, validate_asset_ids=TRAIN_IDS
    ).tracks}
    minimum_turn_radius = float(scenarios.scenarios[0].factory.vehicle_parameters.wheelbase) / tan(
        float(scenarios.scenarios[0].factory.vehicle_parameters.max_steering_angle)
    )
    for entry in scenarios.scenarios:
        manifest_entry = train_manifest[entry.track_id]
        path = experiment.tracks.manifest.parent / manifest_entry.path
        fingerprints.append(_fingerprint(root, path))
        asset = load_track_asset(path)
        if asset.navigation is None:
            raise ValueError("pilot track is missing authored route")
        points = asset.navigation.points
        max_turn = _heading_change_deg(points)
        min_clearance = _rectangle_clearance(points, asset.world.obstacles)
        audits.append({
            "track_id": entry.track_id,
            "polyline_length_meters": round(
                sum((b-a).magnitude for a,b in zip(points,points[1:])), 4
            ),
            "maximum_heading_change_degrees": round(max_turn, 4),
            "minimum_obstacle_clearance_meters": (
                round(min_clearance, 4) if min_clearance is not None else None
            ),
        })
        if max_turn > 45.0:
            blockers.append(
                f"{entry.track_id}: sharp authored junction {max_turn:.1f}deg must be reviewed against "
                f"the approximate {minimum_turn_radius:.2f}m minimum bicycle turning radius"
            )
        if min_clearance is not None and min_clearance < 2.0:
            blockers.append(
                f"{entry.track_id}: {min_clearance:.2f}m point-to-obstacle clearance is too tight "
                "to certify without a swept-volume vehicle feasibility check"
            )
    blockers.append(
        "pilot executor/25k checkpoint evaluator and content-v2 provenance are not yet "
        "verified end-to-end; no models or pilot scores have been produced"
    )
    return {
        "schema_version": "wp7-recovery-pilot-preflight-v1",
        "status": "review_required",
        "training_track_ids": list(TRAIN_IDS),
        "training_seed_pairs": [
            {"ppo": ppo, "sampling": sampling, "role": role}
            for ppo, sampling, role in SEED_PAIRS
        ],
        "requested_total_timesteps": config.total_timesteps,
        "checkpoint_every_steps": config.checkpoint_every_steps,
        "evaluation_milestones": list(PILOT_MILESTONES),
        "ppo_device": config.ppo.device,
        "vector_env": {
            "n_envs": config.vector_env.n_envs,
            "backend": config.vector_env.backend,
            "start_method": config.vector_env.start_method,
        },
        "candidate_semantics": dict(candidate.behavior),
        "fingerprints": fingerprints,
        "route_audit": audits,
        "review_blockers": blockers,
        "trained_models": 0,
        "heldout_rollouts": 0,
        "final_seed_replicates_executed": False,
    }
