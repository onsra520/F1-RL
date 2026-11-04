"""Development-only, absorbing terminal for witnessed v3 obstacle contacts.

A SAT body witness identifies the obstacle but does NOT certify the first contact
time, point, impulse, or a post-impact physical state. Keep the last accepted
vehicle pose and emit one terminal *bookkeeping* frame; never use this path for
PPO, registered tracks, v4 crash scoring, or scientific evidence.
"""
from __future__ import annotations

from src.rl.rewards import CollisionPenalty
from src.simulation.ballistic_flight import FlightPhysicsError


CONTRACT = "f1rl_custom_auto_collision_terminal_dev_v1"


class DevelopmentCollisionTerminalEnv:
    """Opt-in adapter exclusively for custom Auto Drive with v3 Physics."""

    def __init__(self, env) -> None:
        physics = env._physics_config
        if (physics is None or physics.ground_motion_version !=
                "spatial_rotating_compliant_lifecycle_v3" or
                env._reset_config is not None):
            raise ValueError("development collision terminal requires non-TRAIN v3 Physics")
        self._env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self._terminal_snapshot = None
        self._terminal_outcome = None

    def __getattr__(self, name):
        return getattr(self._env, name)

    @property
    def last_research_outcome(self):
        return self._terminal_outcome or self._env.last_research_outcome

    def reset(self, **kwargs):
        self._terminal_snapshot = None
        self._terminal_outcome = None
        return self._env.reset(**kwargs)

    def step(self, action):
        if self._terminal_snapshot is not None:
            raise RuntimeError("absorbing development collision terminal: reset required")
        try:
            return self._env.step(action)
        except FlightPhysicsError as error:
            ids = getattr(error, "witnessed_obstacle_ids", ())
            if error.failure_code != "verified_obstacle_contact" or not ids:
                raise
            known = {item.id for item in self._env._track.obstacles}
            if not all(isinstance(item, str) and item in known for item in ids):
                raise
            return self._terminal_on_witness(tuple(dict.fromkeys(ids)))

    def _terminal_on_witness(self, obstacle_ids):
        env = self._env
        snapshot = env._simulation.snapshot
        if env._progress is None or env._progress.finished:
            raise RuntimeError("development collision cannot terminate without an active episode")
        dt = env._simulation_config.fixed_dt
        # The last accepted physical state is the only pose we may certify.
        # The contact occurred somewhere within this attempted tick. Do not
        # invent a post-impact state, a substep collision time, or route credit.
        pose = snapshot.world.vehicle
        frame = env.runtime_snapshot()
        frame.update(
            tick=snapshot.tick + 1,
            time_s=snapshot.elapsed_time + dt,
            physical_event_types=("ObstacleHit",),
        )
        self._terminal_snapshot = frame
        penalties = {
            term.name: (-float(term.penalty) if isinstance(term, CollisionPenalty) else 0.0)
            for term in env._reward_composer.terms
        }
        outcome = {
            "version": CONTRACT,
            "kind": "collision_terminal_development",
            "valid_sample": False,
            "reason": "collision",
            "scientific_evidence": False,
        }
        self._terminal_outcome = outcome
        info = env._reset_info()
        # The bookkeeping tick has no accepted Physics transition. The
        # preceding accepted frame is available separately in the replay.
        physics_info = info.get("physics")
        if isinstance(physics_info, dict):
            info["physics"] = {
                "configuration": physics_info.get("configuration"),
                "sha256": physics_info.get("sha256"),
                "transition": None,
            }
        info.update(
            tick=snapshot.tick + 1,
            elapsed_time=snapshot.elapsed_time + dt,
            reward_components=penalties,
            telemetry={
                "vehicle_position": (pose.position.x, pose.position.y, pose.position.z),
                "vehicle_velocity": (pose.velocity.x, pose.velocity.y, pose.velocity.z),
                "vehicle_speed": float(pose.speed),
                "grounded": pose.grounded,
                "obstacle_hit_ids": obstacle_ids,
                "applied_boost_pad_ids": (),
                "applied_slow_zone_ids": (),
            },
            research_outcome=outcome,
            collision_terminal={
                "version": CONTRACT,
                "obstacle_ids": list(obstacle_ids),
                "witness_interval_s": [snapshot.elapsed_time, snapshot.elapsed_time + dt],
                "pose_kind": "last_accepted_precontact",
                "reward_kind": "configured_collision_penalty_only",
                "certified_contact_time": False,
                "physical_impulse_applied": False,
                "scientific_evidence": False,
            },
            route_highwater_meters=env._route_highwater,
        )
        # Do not set WP7 race.terminal_failure: this collision has no
        # certified WP7 crossing/impact score and is not an off-route event.
        observation = env._encode_observation(
            pose, env._route_target(pose, env._progress)
        )
        return observation, float(sum(penalties.values())), True, False, info

    def runtime_snapshot(self):
        if self._terminal_snapshot is not None:
            return dict(self._terminal_snapshot)
        return self._env.runtime_snapshot()

    def close(self):
        return self._env.close()
