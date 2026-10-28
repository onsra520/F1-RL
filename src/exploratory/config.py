"""Single, strict runtime configuration resolver for bounded exploratory PPO.

The root YAML also contains the legacy training profile. Exploratory overrides
are explicit and cannot authorize scientific runs or broaden approved physics.
"""
from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import json
import os
from pathlib import Path

import yaml

from ..rl.algorithms.ppo import PPOConfig

TRACK = "straight-sprint-v1"
PROFILE = "spatial_rotating_owner_v3"
POLICIES = ("random", "untrained", "trained_final")
MAX_WALL_SECONDS = 3600
MAX_LEARNING_STEPS = 32768
MAX_EVALUATION_EPISODES = 32


class _UniqueLoader(yaml.SafeLoader):
    pass


def _unique_mapping(loader, node):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=True)
        if not isinstance(key, str) or key in result:
            raise ValueError("duplicate or non-string YAML config key")
        result[key] = loader.construct_object(value_node, deep=True)
    return result


_UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _unique_mapping)


def _mapping(value, keys, name):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ValueError(f"{name}: expected exactly {sorted(keys)}")
    return value


def _positive(value, name, *, minimum=1, maximum=None):
    if type(value) is not int or value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"{name}: invalid integer or configured safety limit exceeded")
    return value


def _boolean(value, name):
    if type(value) is not bool:
        raise ValueError(f"{name}: expected boolean")
    return value


def _threads(value, available, name):
    if value == "auto":
        return available
    return _positive(value, name, maximum=available)


def resolve(root):
    """Return only JSON-compatible, validated values; never run simulator code."""
    path = Path(root).resolve(strict=True) / "config.yml"
    if path.is_symlink() or not path.is_file():
        raise ValueError("trusted repository config.yml required")
    raw = path.read_bytes()
    try:
        document = yaml.load(raw.decode("utf-8"), Loader=_UniqueLoader)
    except (UnicodeError, yaml.YAMLError) as exc:
        raise ValueError("invalid YAML configuration") from exc
    _mapping(document, {"schema_version", "training", "ppo", "vector_env", "exploratory"}, "root")
    if type(document["schema_version"]) is not int or document["schema_version"] != 2:
        raise ValueError("config schema_version 2 required")
    exp = _mapping(document["exploratory"],
        {"experiment", "resources", "training", "ppo", "vector_env", "evaluation", "execution", "logging"}
            | ({'demo_training'} if isinstance(document['exploratory'], dict) and 'demo_training' in document['exploratory'] else set()),
        "exploratory")
    from .reward import REGISTERED, validate
    demo_training = validate(exp.get('demo_training', REGISTERED))
    identity = _mapping(exp["experiment"], {"mode", "track", "physics_profile"}, "experiment")
    if identity != {"mode": "exploratory", "track": TRACK, "physics_profile": PROFILE}:
        raise ValueError("exploratory scope is limited to registered TRAIN Straight physics")
    res = _mapping(exp["resources"], {"cpu"}, "resources")
    cpu = _mapping(res["cpu"], {"threads", "interop_threads", "blas_threads"}, "resources.cpu")
    available = _positive(os.cpu_count() or 1, "available processors")
    threads = _threads(cpu["threads"], available, "resources.cpu.threads")
    interop = _positive(cpu["interop_threads"], "resources.cpu.interop_threads", maximum=available)
    blas = _threads(cpu["blas_threads"], available, "resources.cpu.blas_threads")

    train = _mapping(exp["training"],
        {"total_timesteps", "checkpoint_every_steps", "sampling_seed", "artifacts_root"}, "training")
    steps = _positive(train["total_timesteps"], "training.total_timesteps", maximum=MAX_LEARNING_STEPS)
    checkpoint = _positive(train["checkpoint_every_steps"], "training.checkpoint_every_steps", maximum=steps)
    reset_seed = _positive(train["sampling_seed"], "training.sampling_seed", minimum=0, maximum=2**32-1)
    if not isinstance(train["artifacts_root"], str) or not train["artifacts_root"].strip():
        raise ValueError("training.artifacts_root must be a path")

    vector = _mapping(exp["vector_env"], {"n_envs", "backend", "start_method"}, "vector_env")
    if vector != {"n_envs": 1, "backend": "dummy", "start_method": "spawn"}:
        raise ValueError("exploratory admission currently permits only one dummy environment")

    ppo = _mapping(exp["ppo"], {"learning_rate", "n_steps", "batch_size", "n_epochs",
        "gamma", "gae_lambda", "clip_range", "ent_coef", "vf_coef", "max_grad_norm",
        "seed", "device"}, "ppo")
    validated_ppo = PPOConfig(**ppo)
    if validated_ppo.device != "cpu" or validated_ppo.seed != 0:
        raise ValueError("Straight exploratory scope requires CPU and registered PPO seed 0")
    if steps % validated_ppo.n_steps or checkpoint % validated_ppo.n_steps:
        raise ValueError("training budgets/checkpoints must align to complete PPO rollouts")
    if validated_ppo.n_steps % validated_ppo.batch_size:
        raise ValueError("PPO batch_size must divide n_steps for one environment")

    ev = _mapping(exp["evaluation"],
        {"episodes_per_policy", "seed_start", "policies", "deterministic", "replay_episode"},
        "evaluation")
    count = _positive(ev["episodes_per_policy"], "evaluation.episodes_per_policy",
                      maximum=MAX_EVALUATION_EPISODES)
    seed_start = _positive(ev["seed_start"], "evaluation.seed_start",
                           minimum=0, maximum=2**32-count)
    if type(ev["policies"]) is not list or tuple(ev["policies"]) != POLICIES:
        raise ValueError("exploratory comparison must retain random/untrained/trained_final")
    if _boolean(ev["deterministic"], "evaluation.deterministic") is not True:
        raise ValueError("deterministic PPO evaluation/replay required")
    replay = _positive(ev["replay_episode"], "evaluation.replay_episode", minimum=0, maximum=count-1)

    execution = _mapping(exp["execution"],
        {"wall_seconds", "schedule", "preserve_checkpoints_on_failure"}, "execution")
    wall = _positive(execution["wall_seconds"], "execution.wall_seconds", maximum=MAX_WALL_SECONDS)
    if execution["schedule"] not in ("baseline_first", "learning_first"):
        raise ValueError("execution.schedule must be baseline_first or learning_first")
    if _boolean(execution["preserve_checkpoints_on_failure"], "execution.preserve_checkpoints_on_failure") is not True:
        raise ValueError("checkpoint survival is mandatory")

    logging = _mapping(exp["logging"],
        {"progress_every_steps", "save_metrics", "save_episode_results"}, "logging")
    progress = _positive(logging["progress_every_steps"], "logging.progress_every_steps", maximum=steps)
    if validated_ppo.n_steps % progress:
        raise ValueError("logging.progress_every_steps must divide a rollout")
    if any(_boolean(logging[key], f"logging.{key}") is not True
           for key in ("save_metrics", "save_episode_results")):
        raise ValueError("evidence-producing logging may not be disabled")

    return dict(
        demo_training=demo_training,
        config_sha256=sha256(raw).hexdigest(), track=TRACK, profile=PROFILE,
        policy=f"singletrack-{TRACK}-seed0",
        limits=dict(learning_timesteps=steps, checkpoint_steps=checkpoint,
            threads=threads, interop_threads=interop, blas_threads=blas,
            wall_seconds=wall, n_envs=1, device="cpu"),
        seeds=dict(ppo=validated_ppo.seed, training_reset=reset_seed,
            evaluation=list(range(seed_start, seed_start + count))),
        ppo=asdict(validated_ppo),
        evaluation=dict(episodes_per_policy=count, policies=list(POLICIES),
            actions="deterministic_for_PPO", selection="final_only_intermediate_diagnostic",
            replay="first_trained_episode" if replay == 0 else f"trained_episode_{replay}",
            replay_episode=replay),
        execution=dict(schedule=execution["schedule"], preserve_checkpoints_on_failure=True),
        logging=dict(progress_every_steps=progress, save_metrics=True, save_episode_results=True),
        artifacts_root=train["artifacts_root"],
    )
