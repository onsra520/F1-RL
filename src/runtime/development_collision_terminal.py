"""Development-only collision evidence at the runtime boundary.

No Gym transition, Physics tick, reward, or replay frame is synthesized here.
Only an explicitly witnessed v3 custom Auto Drive obstacle contact is mapped
to an unscored runtime failure. This is not valid PPO/scientific evidence.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.rl.rewards import CollisionPenalty
from src.simulation.ballistic_flight import FlightPhysicsError

CONTRACT = "f1rl_runtime_collision_failure_dev_v1"
V3 = "spatial_rotating_compliant_lifecycle_v3"


@dataclass(frozen=True, slots=True)
class RuntimeCollisionEvidence:
    obstacle_ids: tuple[str, ...]
    last_accepted_tick: int
    last_accepted_time_s: float
    attempted_tick_end_s: float
    diagnostic_penalty: float

    def document(self) -> dict[str, object]:
        return {
            "schema_version": CONTRACT,
            "obstacle_ids": list(self.obstacle_ids),
            "last_accepted_tick": self.last_accepted_tick,
            "witness_interval_s": [
                self.last_accepted_time_s, self.attempted_tick_end_s,
            ],
            "diagnostic_penalty": self.diagnostic_penalty,
            "penalty_kind": "runtime_diagnostic_not_gym_reward",
            "physics_transition_accepted": False,
            "contact_time_certified": False,
            "scientific_evidence": False,
        }


def inspect_collision_error(error: Exception, env) -> RuntimeCollisionEvidence | None:
    """Recognize only a witnessed custom v3 failure; never turn unknown into hit."""
    if not isinstance(error, FlightPhysicsError) or error.failure_code != "verified_obstacle_contact":
        return None
    physics = env._physics_config
    if (physics is None or physics.ground_motion_version != V3 or
            env._reset_config is not None or env._progress is None or
            env._progress.finished):
        return None
    ids = getattr(error, "witnessed_obstacle_ids", ())
    known = {item.id for item in env._track.obstacles}
    if (not isinstance(ids, tuple) or not ids or
            any(not isinstance(item, str) or item not in known for item in ids)):
        return None
    snapshot = env._simulation.snapshot
    dt = env._simulation_config.fixed_dt
    penalty = -sum(float(term.penalty) for term in env._reward_composer.terms
                   if isinstance(term, CollisionPenalty))
    return RuntimeCollisionEvidence(
        obstacle_ids=tuple(dict.fromkeys(ids)),
        last_accepted_tick=snapshot.tick,
        last_accepted_time_s=snapshot.elapsed_time,
        attempted_tick_end_s=snapshot.elapsed_time + dt,
        diagnostic_penalty=penalty,
    )
