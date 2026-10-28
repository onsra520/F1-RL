"""Read-only TRAIN replay of archived WP7 PPO models; no HELD-OUT access.

Audit physical *proxy* gaps and continuous XZ path segments, independent from
point-based engine collision events. XZ gaps ignore height and therefore can
produce false-positive proximity warnings for vehicles passing over obstacles.
Vehicle body dimensions are unspecified; this cannot certify body clearance.
"""
from __future__ import annotations

from hashlib import sha256
from math import hypot, isfinite
from pathlib import Path

from ..sandbox.geometry.rotation import Quaternion
from ..sandbox.geometry.vector import Vec3
from .algorithms.ppo import PPOModel
from .pilot_milestones import verify_pilot_evidence_manifest
from .pilot_preflight import TRAIN_IDS
from .reconstruction import RecoveryCandidate, build_training_scenarios
from .route_navigation import AuthoredRoute

PROXY_RADII_METERS = (0.5, 1.0, 1.5)


def _point_segment_distance(
    px: float, pz: float, ax: float, az: float, bx: float, bz: float
) -> float:
    dx, dz = bx - ax, bz - az
    length2 = dx * dx + dz * dz
    fraction = max(0.0, min(1.0, ((px - ax) * dx + (pz - az) * dz) / length2)) if length2 > 1e-18 else 0.0
    return hypot(px - ax - fraction * dx, pz - az - fraction * dz)


def segment_rect_clearance_xz(
    ax: float, az: float, bx: float, bz: float,
    cx: float, cz: float, half_x: float, half_z: float
) -> float:
    """Exact minimum XZ distance from finite segment to axis-aligned rectangle."""
    if half_x <= 0.0 or half_z <= 0.0 or not all(
        isfinite(x) for x in (ax, az, bx, bz, cx, cz, half_x, half_z)
    ):
        raise ValueError("clearance geometry must be finite, with positive rectangle extents")
    xmin, xmax = cx - half_x, cx + half_x
    zmin, zmax = cz - half_z, cz + half_z
    # Liang-Barsky segment/rectangle intersection, including tangency.
    dx, dz = bx - ax, bz - az
    low, high = 0.0, 1.0
    intersects = True
    for start, direction, minimum, maximum in (
        (ax, dx, xmin, xmax), (az, dz, zmin, zmax)
    ):
        if abs(direction) < 1e-14:
            if start < minimum or start > maximum:
                intersects = False
                break
        else:
            first, second = (minimum - start) / direction, (maximum - start) / direction
            low = max(low, min(first, second))
            high = min(high, max(first, second))
            if low > high:
                intersects = False
                break
    if intersects:
        return 0.0

    def to_rect(x: float, z: float) -> float:
        return hypot(max(xmin - x, 0.0, x - xmax), max(zmin - z, 0.0, z - zmax))

    corners = ((xmin,zmin),(xmin,zmax),(xmax,zmin),(xmax,zmax))
    return min(
        to_rect(ax, az), to_rect(bx, bz),
        *(_point_segment_distance(x,z,ax,az,bx,bz) for x,z in corners),
    )


def _hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def audit_trajectory_run(*, repository_root: str | Path, run_directory: str | Path) -> dict[str, object]:
    """Evaluate only TRAIN on the archived final PPO, preserving all run artifacts."""
    root = Path(repository_root).resolve(strict=True)
    run = Path(run_directory).resolve(strict=True)
    integrity = verify_pilot_evidence_manifest(run)
    if integrity["status"] != "verified_exploratory_evidence_not_scientific":
        raise ValueError("trajectory audit requires archived genuine exploratory evidence")
    import json
    final_report = json.loads((run / "evaluations" / "milestone-200000.json").read_text("utf-8"))
    manifest = json.loads((run / "evaluations" / "manifest.json").read_text("utf-8"))
    if final_report["evaluation"]["policy_source"] != "ppo_archive_requires_run_provenance_review":
        raise ValueError("trajectory audit refuses test-policy evidence")
    if final_report["evaluation"]["track_ids"] != list(TRAIN_IDS):
        raise ValueError("trajectory audit refuses non-TRAIN tracks")

    # The original committed source may differ from the audit implementation
    # HEAD, but every source file that generated the archived model must match.
    for source in manifest["source_fingerprints"]:
        path = (root / source["path"]).resolve(strict=True)
        if not path.is_relative_to(root) or _hash(path) != source["sha256"]:
            raise ValueError("TRAIN candidate/source fingerprint drift: " + source["path"])

    final_archive = (run / final_report["model_path"]).resolve(strict=True)
    if not final_archive.is_relative_to(run) or _hash(final_archive) != final_report["checkpoint_sha256"]:
        raise ValueError("final PPO archive SHA256 mismatch")
    policy = PPOModel.load(final_archive, device="cpu")
    candidate = RecoveryCandidate.from_yaml(root / "configs" / "recovery" / "wp7-c2-candidate.yml")
    scenarios = build_training_scenarios(candidate)
    if tuple(entry.track_id for entry in scenarios.scenarios) != TRAIN_IDS:
        raise ValueError("TRAIN scenario boundary changed")

    rows: list[dict[str, object]] = []
    for entry, expected in zip(scenarios.scenarios, final_report["evaluation"]["track_summaries"], strict=True):
        if expected["track_id"] != entry.track_id or expected["episode_count"] != 1:
            raise ValueError("archived evaluation must contain exactly one TRAIN episode per track")
        factory = entry.factory
        navigation = AuthoredRoute(factory.navigation)
        obstacles = factory.track.obstacles
        for obstacle in obstacles:
            if obstacle.volume.orientation != Quaternion.identity():
                raise ValueError("rotated obstacles need a separate geometric audit")
        env = factory.create()
        try:
            observation, info = env.reset()
            point = Vec3(*info["telemetry"]["vehicle_position"])
            clearance = float("inf")
            max_offroute = navigation.horizontal_distance(point)
            max_displacement = 0.0
            obstacle_hit_count = 0
            steps = 0
            terminated = truncated = False
            while not (terminated or truncated):
                action = policy.predict(observation, deterministic=True)
                observation, _, terminated, truncated, info = env.step(action)
                next_point = Vec3(*info["telemetry"]["vehicle_position"])
                steps += 1
                max_displacement = max(
                    max_displacement, (next_point - point).magnitude
                )
                max_offroute = max(max_offroute, navigation.horizontal_distance(next_point))
                obstacle_hit_count += len(info["telemetry"]["obstacle_hit_ids"])
                for obstacle in obstacles:
                    volume = obstacle.volume
                    clearance = min(clearance, segment_rect_clearance_xz(
                        point.x, point.z, next_point.x, next_point.z,
                        volume.center.x, volume.center.z,
                        volume.half_extents.x, volume.half_extents.z,
                    ))
                point = next_point
            success = bool(info["finished"]) and bool(terminated) and not bool(truncated)
            expected_success = int(expected["success_count"]) == 1
            if success != expected_success or obstacle_hit_count != int(expected["mean_collision_count"]):
                raise ValueError("deterministic TRAIN trajectory disagrees with archived final evaluation")
            if success and abs(steps * factory.simulation_config.fixed_dt - float(expected["mean_completion_time_for_successes"])) > 1e-6:
                raise ValueError("deterministic TRAIN completion time differs from archive")
            gap = None if not obstacles else round(clearance, 6)
            rows.append({
                "track_id": entry.track_id,
                "success": success,
                "steps": steps,
                "simulator_obstacle_hit_count": obstacle_hit_count,
                "obstacle_count": len(obstacles),
                "min_continuous_xz_point_to_obstacle_m": gap,
                "max_horizontal_route_deviation_m": round(max_offroute, 6),
                "max_step_displacement_m": round(max_displacement, 6),
                "proxy_radii_clearances": {
                    f"{radius:g}m": None if gap is None else gap > radius
                    for radius in PROXY_RADII_METERS
                },
            })
        finally:
            env.close()
    return {
        "schema_version": "wp7-recovery-train-trajectory-audit-v1",
        "status": "requires_physical_footprint_review",
        "scientific_evidence": False,
        "heldout_rollouts": 0,
        "source_git_revision": final_report["git_commit"],
        "run_id": run.name,
        "final_model_sha256": final_report["checkpoint_sha256"],
        "evidence_manifest_sha256": _hash(run / "evaluations" / "manifest.json"),
        "method": "deterministic_single_episode_per_train_track; swept_XZ_segment_to_AABB",
        "limitations": [
            "Physics detects obstacle occupancy at sampled vehicle centre only.",
            "Continuous planar XZ proxy ignores height, so an airborne clearance warning may be conservative.",
            "Vehicle body dimensions are not defined; proposed radii are sensitivity probes, not actual width.",
            "Route deviation is diagnostic; no post-hoc threshold is used to redefine TRAIN success.",
            "One deterministic episode per TRAIN track does not establish probabilistic robustness.",
        ],
        "results": rows,
    }
