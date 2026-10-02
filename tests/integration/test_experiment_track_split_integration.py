from pathlib import Path

from f1rl.learning.episode import EpisodeConfig
from f1rl.learning.experiments import ExperimentDefinition, resolve_track_split
from f1rl.learning.observation import ObservationConfig, ObservationEncoder
from f1rl.learning.rewards import (
    CollisionPenalty,
    RewardComposer,
    TargetProgressReward,
    TimePenalty,
)
from f1rl.learning.training import (
    F1RLEnvFactory,
    TrainingScenario,
    TrainingScenarioSet,
)
from f1rl.runtime import SimulationConfig
from f1rl.sandbox.geometry.vector import Vec3
from f1rl.sandbox.perception import RangefinderSpec, SensorSuite
from f1rl.sandbox.vehicle import VehicleParameters
from f1rl.track_assets import build_initial_world_state, load_track_asset


PROFILE = Path("configs/experiments/ppo-generalization-v1.yml")
TRACK_ROOT = Path("assets/tracks")


def _shared_contract(definition: ExperimentDefinition):
    vehicle = VehicleParameters(
        definition.vehicle.wheelbase,
        definition.vehicle.ride_height,
        definition.vehicle.max_steering_angle,
        definition.vehicle.max_acceleration,
        definition.vehicle.max_braking,
        definition.vehicle.rolling_drag,
        definition.vehicle.air_drag,
        definition.vehicle.max_ground_speed,
        definition.vehicle.gravity,
    )
    simulation = SimulationConfig(
        definition.simulation.fixed_dt,
        definition.simulation.max_surface_distance,
    )
    sensor_suite = SensorSuite(
        tuple(
            RangefinderSpec(spec.id, Vec3(*spec.local_direction), spec.max_distance)
            for spec in definition.observation.rangefinders
        )
    )
    observation = ObservationEncoder(
        ObservationConfig(
            sensor_suite,
            definition.observation.velocity_scale,
            definition.observation.target_distance_scale,
        )
    )
    rewards = RewardComposer(
        (
            TargetProgressReward(definition.reward.target_progress_weight_per_meter),
            CollisionPenalty(definition.reward.collision_penalty),
            TimePenalty(definition.reward.time_penalty_per_second),
        )
    )
    episode = EpisodeConfig(definition.episode.max_steps)
    return vehicle, simulation, observation, rewards, episode


def _factory(definition: ExperimentDefinition, entry, shared) -> F1RLEnvFactory:
    vehicle, simulation, observation, rewards, episode = shared
    asset = load_track_asset(TRACK_ROOT / entry.path)
    initial_state = build_initial_world_state(
        asset,
        max_surface_distance=definition.simulation.max_surface_distance,
    )
    return F1RLEnvFactory(
        track=asset.world,
        initial_state=initial_state,
        vehicle_parameters=vehicle,
        simulation_config=simulation,
        observation_encoder=observation,
        reward_composer=rewards,
        episode_config=episode,
    )


def _mechanic_classes(entry) -> set[str]:
    world = load_track_asset(TRACK_ROOT / entry.path).world
    mechanics: set[str] = set()
    if world.obstacles:
        mechanics.add("obstacle")
    if world.boost_pads:
        mechanics.add("boost")
    if world.slow_zones:
        mechanics.add("slow")
    if any(abs(surface.forward.y) > 1e-12 for surface in world.surfaces):
        mechanics.add("elevation")
    return mechanics


def test_heldout_mechanics_are_known_from_training_while_geometry_remains_split() -> None:
    definition = ExperimentDefinition.from_yaml(PROFILE)
    split = resolve_track_split(definition.tracks)

    training_mechanics = set().union(*(_mechanic_classes(entry) for entry in split.training))
    heldout_mechanics = set().union(*(_mechanic_classes(entry) for entry in split.heldout))

    assert training_mechanics == {"obstacle", "boost", "slow", "elevation"}
    assert heldout_mechanics == {"obstacle", "boost", "slow"}
    assert heldout_mechanics <= training_mechanics


def test_canonical_split_initial_states_are_fair_and_grounded() -> None:
    definition = ExperimentDefinition.from_yaml(PROFILE)
    split = resolve_track_split(definition.tracks)
    zero = Vec3(0.0, 0.0, 0.0)

    for entry in (*split.training, *split.heldout):
        asset = load_track_asset(TRACK_ROOT / entry.path)
        state = build_initial_world_state(
            asset,
            max_surface_distance=definition.simulation.max_surface_distance,
        )

        assert state.vehicle.position == asset.spawn.position
        assert state.vehicle.orientation == asset.spawn.orientation
        assert state.vehicle.velocity == zero
        assert state.vehicle.angular_velocity == zero
        assert state.vehicle.grounded is True
        assert state.track.collected_pickup_ids == frozenset()
        assert state.interactions.active_checkpoint_ids == frozenset()
        assert state.interactions.active_finish_line_ids == frozenset()
        assert state.score.points == 0


def test_canonical_split_factories_share_wp1b_non_track_contract_and_spaces() -> None:
    definition = ExperimentDefinition.from_yaml(PROFILE)
    split = resolve_track_split(definition.tracks)
    entries = (*split.training, *split.heldout)
    shared = _shared_contract(definition)
    factories = tuple(_factory(definition, entry, shared) for entry in entries)
    scenarios = TrainingScenarioSet(
        tuple(
            TrainingScenario(entry.track_id, factory)
            for entry, factory in zip(entries, factories, strict=True)
        )
    )

    reference = factories[0]
    for factory in factories[1:]:
        assert factory.vehicle_parameters == reference.vehicle_parameters
        assert factory.simulation_config == reference.simulation_config
        assert factory.observation_encoder == reference.observation_encoder
        assert factory.reward_composer == reference.reward_composer
        assert factory.episode_config == reference.episode_config

    assert scenarios.action_space.shape == (2,)
    assert scenarios.observation_space.shape == (16,)
