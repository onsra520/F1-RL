"""Fail-closed policy-to-execution configuration binding for future WP7 runs.

Preparing a bound execution is not permission to train. Runner admission remains
an independent gate and historical C2 execution remains unchanged.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path

from src.training.config import TrainingRunConfig
from src.training.env_factory import TrainingScenarioSet
from ..rl.action import ActionContract
from ..rl.route_navigation import AuthoredRoute
from .wp7_protocol import (
    build_protocol, prepare_training_scenarios, validate_protocol, digest,
)


@dataclass(frozen=True, slots=True)
class PolicyBoundTrainingConfig:
    root: Path
    policy_id: str
    protocol_sha256: str
    configuration: TrainingRunConfig
    scenarios: TrainingScenarioSet
    receipt: dict[str, object]
    profile: str = 'pv5_v1'

    def validate(self, config: TrainingRunConfig, scenarios: TrainingScenarioSet) -> None:
        """Rebuild from disk and compare exact policy, PPO, sampling and reset IDs."""
        if not isinstance(config, TrainingRunConfig) or not isinstance(scenarios, TrainingScenarioSet):
            raise ValueError("registered policy requires TrainingRunConfig and TrainingScenarioSet")
        current = build_policy_execution(self.root, self.policy_id,profile=self.profile)
        if self.protocol_sha256 != current.protocol_sha256:
            raise ValueError("registered protocol/source drift before execution")
        if config != current.configuration:
            raise ValueError("PPO/sampling/training configuration does not match registered policy")
        if scenarios != current.scenarios:
            raise ValueError("TRAIN roster, physics or perturbation seed differs from registered policy")
        if self.receipt != current.receipt:
            raise ValueError("registered execution receipt is stale or malformed")
        if self.receipt["training_authorized"] is not False:
            raise ValueError("execution preparation cannot authorize scientific training")


def build_policy_execution(repository_root: Path | str, policy_id: str,*,profile='pv5_v1') -> PolicyBoundTrainingConfig:
    """Bind exactly one registered policy to a runner config and TRAIN scenario set.

    Does not create models, reset a Gym environment, train, or open artifacts.
    """
    if not isinstance(policy_id, str) or not policy_id:
        raise ValueError("registered policy_id required")
    root = Path(repository_root).resolve(strict=True)
    protocol = build_protocol(root,profile=profile)
    validate_protocol(protocol, root)
    scenarios, identity = prepare_training_scenarios(root, protocol, policy_id)
    bundle = identity["seed_bundle"]
    policy = next(p for p in protocol["policies"] if p["policy_id"] == policy_id)
    baseline = TrainingRunConfig.from_yaml(root / protocol["registration"]["training_profile"])
    config = replace(
        baseline,
        ppo=replace(baseline.ppo, seed=bundle["ppo"]),
        sampling_seed=bundle["sampling"],
    )
    if asdict(config.ppo) != policy["ppo"]:
        raise ValueError("registered PPO profile diverges from execution config")
    if tuple(identity["track_ids"]) != tuple(policy["training_ids"]):
        raise ValueError("registered TRAIN roster mismatch")
    if any(s.factory.reset_config.seed != bundle["perturbation"] for s in scenarios.scenarios):
        raise ValueError("registered TRAIN perturbation seed mismatch")
    receipt = {
        "schema_version": "wp7_policy_execution_binding_v2",
        "policy_id": policy_id,
        "protocol_sha256": protocol["sha256"],
        "seed_bundle": dict(bundle),
        "training_track_ids": list(identity["track_ids"]),
        "physics_sha256": identity["physics_sha256"],
        "reset_sha256": identity["reset_sha256"],
        "ppo": asdict(config.ppo),
        "sampling_seed": config.sampling_seed,
        "vector_environment": asdict(config.vector_env),
        "race_sha256_by_track": {s.track_id:s.factory.race_config.provenance(
            s.factory.track,AuthoredRoute(s.factory.navigation))["sha256"] for s in scenarios.scenarios},
        "action_contract": asdict(scenarios.scenarios[0].factory.action_contract or ActionContract()),
        "observation_sha256": digest(protocol["observation"]),
        "simulation_sha256": digest(protocol["simulation"]),
        "checkpoint_plan": {"training_steps":config.total_timesteps,
                            "checkpoint_every_steps":config.checkpoint_every_steps,
                            "selection":"final_model_primary"},
        "training_authorized": False,
    }
    receipt["sha256"] = digest(receipt)
    return PolicyBoundTrainingConfig(root, policy_id, protocol["sha256"], config, scenarios, receipt,profile)
