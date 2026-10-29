"""Registered WP7 TRAIN scenario construction, without admission side effects."""

from dataclasses import replace

from ..rl.pilot_preflight import TRAIN_IDS
from ..rl.reconstruction import RecoveryCandidate, build_training_scenarios
from ..rl.rewards import (
    RewardComposer,
    RouteProgressReward,
    CollisionPenalty,
    TimePenalty,
)
from ..rl.train_reset import TrainResetConfig
from ..rl.action import ActionContract
from ..rl.wp7_race import WP7RaceConfig
from ..simulation.ballistic_flight import FlightState
from ..core.physics import PhysicsConfig
from ..simulation.geometry.vector import Vec3
from ..simulation.independent_suspension_dynamics import (
    IndependentSuspensionDynamicsPolicy,
    SuspensionDynamicsState,
)
from ..simulation.multi_surface_support import MultiSurfaceSuspensionPolicy
from ..simulation.ramp_support import WheelContactPoints
from ..simulation.coupled_longitudinal_dynamics import LongitudinalContactPolicy
from ..simulation.spatial_contact_dynamics import SpatialContactPolicy
from ..tracks import load_track_manifest
from .wp7_track_feasibility import BODY
from src.training.env_factory import TrainingScenario, TrainingScenarioSet


def prepare_scenarios(root, registration, *, seed_index: int):
    """Build registered TRAIN scenarios for exactly one paired seed replicate.

    This is preparation only. A correct seed binding does not grant PPO admission.
    """
    seeds = registration["seed_roster"]
    if type(seed_index) is not int or not 0 <= seed_index < len(seeds):
        raise ValueError("explicit registered seed_index required for TRAIN scenarios")
    candidate = RecoveryCandidate.from_yaml(
        root / "configs/recovery/wp7-c2-candidate.yml"
    )
    baseline = build_training_scenarios(candidate)
    entries = {
        e.track_id: e
        for e in load_track_manifest(
            root / "assets/tracks/manifest.json", validate_asset_ids=TRAIN_IDS
        ).tracks
    }
    registry = tuple(entries[identity] for identity in TRAIN_IDS)
    owner = registration["schema_version"] in (
        "wp7_experiment_registration_v3",
        "wp7_experiment_registration_v4",
    )
    reset = TrainResetConfig(
        seeds[seed_index]["perturbation"],
        registry,
        **({"version": "wp7_train_reset_lifecycle_research_v1"} if owner else {}),
    )
    wheels = WheelContactPoints(
        Vec3(-0.75, -0.5, 1.5),
        Vec3(0.75, -0.5, 1.5),
        Vec3(-0.75, -0.5, -1.5),
        Vec3(0.75, -0.5, -1.5),
    )
    contact = MultiSurfaceSuspensionPolicy(wheels, 0.12, 0.12)
    suspension = IndependentSuspensionDynamicsPolicy(
        contact,
        **registration["engineering_suspension"],
        angular_rate_mapping="yaw_rx_rz_v1",
    )
    spatial = registration["schema_version"] in (
        "wp7_experiment_registration_v2",
        "wp7_experiment_registration_v3",
        "wp7_experiment_registration_v4",
    )
    spatial_policy = registration.get("engineering_spatial_contact")
    physics = PhysicsConfig(
        "wp7_lifecycle_v1" if owner else "wp7_suspension_v1",
        BODY,
        suspension,
        ground_motion_version=registration["ground_motion_version"]
        if registration["schema_version"] == "wp7_experiment_registration_v4"
        else "spatial_compliant_lifecycle_v1"
        if owner
        else "spatial_contact_v1"
        if spatial
        else "arc_distance_v1",
        spawn_reference="authored_horizontal_road_v1",
        spatial_contact_policy=(
            SpatialContactPolicy(
                **{
                    **spatial_policy,
                    "traction": LongitudinalContactPolicy(**spatial_policy["traction"]),
                }
            )
            if spatial
            else None
        ),
    )
    rewards = RewardComposer(
        (RouteProgressReward(1.0), CollisionPenalty(5.0), TimePenalty(0.1))
    )
    scenarios = []
    for row in baseline.scenarios:
        initial = row.factory.initial_state
        contact = row.factory.track.surface_at(initial.vehicle.position, 0.0)
        if contact is None or contact.normal != Vec3(0.0, 1.0, 0.0):
            raise ValueError("registered TRAIN spawn requires horizontal authored road")
        position = initial.vehicle.position + Vec3(
            0.0, row.factory.vehicle_parameters.ride_height, 0.0
        )
        initial = replace(
            initial,
            vehicle=replace(initial.vehicle, position=position),
            suspension=SuspensionDynamicsState(position.y),
            **({"flight": FlightState("grounded")} if owner else {}),
        )
        factory = replace(
            row.factory,
            offtrack_config=None,
            physics_config=physics,
            race_config=WP7RaceConfig(
                version="wp7_directed_lifecycle_owner_v2",
                hard_violation="terminate_valid_failure",
            )
            if owner
            else WP7RaceConfig(),
            reset_config=reset,
            reset_track_id=row.track_id,
            reward_composer=rewards,
            initial_state=initial,
            action_contract=ActionContract("steering-throttle-brake-v2")
            if spatial
            else None,
        )
        scenarios.append(TrainingScenario(row.track_id, factory))
    return TrainingScenarioSet(tuple(scenarios))
