"""Draft TRAIN-5 v4 final-checkpoint wiring; scientific admission stays separate."""
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path

import yaml

from ..learning.algorithms.ppo import PPOConfig
from ..learning.training import TrainingRunConfig, TrainingRunResult, VectorEnvConfig
from .wp7_train5_v4_budget import PINNED_PROFILE_SHA, load_v4_budget, validate_v4_budget
from .wp7_train5_v4_dev_metrics import evaluate_dev_metrics_v4
from .wp7_train5_v4_runner import V4TrainingRunner, initialize_policy_from_archive

FROZEN_CHECKPOINTS = {
    "C-seed0": "15373c0237e883fa2f980dfecbda4809aabc283c0d77a8b4d783e2ab5bf11492",
    "B-seed1": "695f1c649235775eb1446ac0718b69b5951a1242b389221a5318db540a716021",
    "A-seed2": "e2a0eb0deef34fad7bf2bf2322c0c13854869372263a4652d4c6f1c422d43adc",
}
FROZEN_BY_REPLICATE = ("C-seed0", "B-seed1", "A-seed2")


def _prepare_v4_runner(root: Path, artifacts_root: Path, *, ppo_seed: int, sampling_seed: int,
                       frozen_checkpoint: str | None = None) -> V4TrainingRunner:
    root = Path(root).resolve()
    artifacts_root = Path(artifacts_root).resolve()
    if artifacts_root == root or root in artifacts_root.parents:
        raise ValueError("v4 artifacts must remain outside the source checkout")
    protocol = load_v4_budget(root)
    pilot_path = root / "configs/recovery/wp7-pilot-200k.yml"
    pilot_bytes = pilot_path.read_bytes().replace(b"\r\n", b"\n")
    if sha256(pilot_bytes).hexdigest() != PINNED_PROFILE_SHA:
        raise ValueError("legacy YAML profile differs from source-bound hash")
    pilot = yaml.safe_load(pilot_bytes)
    budget = validate_v4_budget(protocol, pilot)
    initializer = None
    initial_checkpoint = None
    if frozen_checkpoint is not None:
        if frozen_checkpoint not in FROZEN_CHECKPOINTS:
            raise ValueError("unregistered frozen checkpoint")
        archive = root / "models" / frozen_checkpoint / "model-final.zip"
        expected = FROZEN_CHECKPOINTS[frozen_checkpoint]
        initial_checkpoint = {"id": frozen_checkpoint, "sha256": expected, "policy_only": True}

        def read_verified_checkpoint() -> bytes:
            try:
                contents = archive.read_bytes()
            except OSError as exc:
                raise ValueError("frozen checkpoint cannot be read") from exc
            if sha256(contents).hexdigest() != expected:
                raise ValueError("frozen checkpoint SHA-256 mismatch")
            return contents

        read_verified_checkpoint()

        def initializer(model) -> None:
            initialize_policy_from_archive(model, read_verified_checkpoint())

    config = TrainingRunConfig(
        ppo=replace(PPOConfig(**pilot["ppo"]), seed=ppo_seed),
        vector_env=VectorEnvConfig(**pilot["vector_env"]),
        total_timesteps=pilot["training"]["total_timesteps"],
        sampling_seed=sampling_seed, artifacts_root=artifacts_root,
        checkpoint_every_steps=None,
    )

    def verify_final(path: Path, actual_steps: int) -> None:
        if type(actual_steps) is not int or actual_steps != budget["actual_final_checkpoint_step"]:
            raise ValueError("v4 final checkpoint has unexpected timestep")
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError("v4 final checkpoint is missing or empty")

    return V4TrainingRunner(config, final_model_observer=verify_final,
                            model_initializer=initializer, initial_checkpoint=initial_checkpoint)


def prepare_v4_runner(root: Path, artifacts_root: Path, *, ppo_seed: int, sampling_seed: int,
                      frozen_checkpoint: str | None = None) -> V4TrainingRunner:
    from .wp7_train5_v4_admission import development_seeds, load_v4_protocol, validate_v4_protocol

    protocol, protocol_sha256 = load_v4_protocol(root)
    validate_v4_protocol(root, protocol)
    reserved = {seed for run in protocol["runs"] for seed in run["seeds"].values()}
    reserved.update(development_seeds(root, protocol, protocol_sha256).values())
    n_envs = protocol["budget"]["vector_envs"]
    for run in protocol["runs"]:
        reserved.update(range(run["seeds"]["ppo"], run["seeds"]["ppo"] + n_envs))
    registration = json.loads((Path(root) / "configs/research/wp7-train5-paired-v2-registration.json").read_text(encoding="utf8"))
    for group in registration["proposed_training_seeds"].values():
        for run in group:
            reserved.update(run.values())
            reserved.update(range(run["ppo"], run["ppo"] + n_envs))
    if type(ppo_seed) is not int or not 0 <= ppo_seed <= 2**32 - n_envs:
        raise ValueError("PPO seed and vector ranks must fit uint32")
    if reserved.intersection(range(ppo_seed, ppo_seed + n_envs)) or sampling_seed in reserved:
        raise ValueError("reserved TRAIN-5 seeds require the registered fail-closed runner")
    return _prepare_v4_runner(root, artifacts_root, ppo_seed=ppo_seed, sampling_seed=sampling_seed,
                              frozen_checkpoint=frozen_checkpoint)


def prepare_v4_registered_runner(root: Path, artifacts_root: Path, *, cohort: str, replicate: int) -> V4TrainingRunner:
    from .wp7_train5_v4_admission import load_v4_protocol, validate_v4_protocol

    protocol, _ = load_v4_protocol(root)
    validate_v4_protocol(root, protocol)
    if type(replicate) is not int or replicate not in range(3):
        raise ValueError("registered replicate must be 0, 1, or 2")
    matches = [run for run in protocol["runs"] if run["cohort"] == cohort and run["replicate"] == replicate]
    if len(matches) != 1:
        raise ValueError("unknown registered cohort")
    run = matches[0]
    runner = _prepare_v4_runner(root, artifacts_root, ppo_seed=run["seeds"]["ppo"],
                                sampling_seed=run["seeds"]["sampling"], frozen_checkpoint=run["initial_checkpoint"])
    runner._registered_run = {"track_ids": tuple(protocol["track_ids"]),
                              "perturbation_seed": run["seeds"]["perturbation"], "root": Path(root).resolve(),
                              "cohort": cohort, "replicate": replicate}
    return runner


def evaluate_v4_runs(records: list[dict], runs: dict, contract: dict, protocol: dict, pilot: dict) -> dict:
    budget = validate_v4_budget(protocol, pilot)
    expected = {(cohort, replicate) for cohort in ("scratch", "frozen_initialization") for replicate in range(3)}
    if not isinstance(runs, dict) or set(runs) != expected:
        raise ValueError("six registered final runs are required")
    from .wp7_train5_v4_admission import development_seeds, load_v4_protocol, validate_v4_protocol
    source_root = Path(__file__).resolve().parents[2]
    pilot_bytes = (source_root / "configs/recovery/wp7-pilot-200k.yml").read_bytes().replace(b"\r\n", b"\n")
    if sha256(pilot_bytes).hexdigest() != PINNED_PROFILE_SHA or yaml.safe_load(pilot_bytes) != pilot:
        raise ValueError("evaluation pilot configuration differs from locked source")
    draft, draft_sha256 = load_v4_protocol(source_root)
    validate_v4_protocol(source_root, draft)
    expected_seeds = development_seeds(source_root, draft, draft_sha256)
    allocations = {("scratch" if run["cohort"] == "train5_from_scratch" else "frozen_initialization", run["replicate"]): run
                   for run in draft["runs"]}
    digests = {}
    for key, run in runs.items():
        if (not isinstance(run, TrainingRunResult) or run.checkpoint_paths
                or run.requested_total_timesteps != protocol["budget"]["requested_steps"]
                or run.actual_total_timesteps != budget["actual_final_checkpoint_step"]
                or not run.final_model_path.is_file()):
            raise ValueError("run does not satisfy the v4 final-checkpoint contract")
        try:
            state = json.loads((run.run_directory / "state.json").read_text(encoding="utf8"))
            recorded_path = (run.run_directory / state["final_model"]).resolve()
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise ValueError("completed run state is missing or invalid") from exc
        expected_ppo = {**pilot["ppo"], "seed": allocations[key]["seeds"]["ppo"]}
        if (state.get("status") != "completed" or state.get("run_id") != run.run_id
                or run.run_directory.name != run.run_id or state.get("mode") != "multitrack"
                or state.get("requested_total_timesteps") != run.requested_total_timesteps
                or state.get("actual_total_timesteps") != run.actual_total_timesteps
                or state.get("checkpoint_every_steps") is not None
                or state.get("latest_checkpoint") is not None
                or state.get("ppo") != expected_ppo or state.get("vector_env") != pilot["vector_env"]
                or state.get("sampling_seed") != allocations[key]["seeds"]["sampling"]
                or state.get("training_reset_seed") != allocations[key]["seeds"]["perturbation"]
                or state.get("training_tracks") != draft["track_ids"]
                or recorded_path != run.final_model_path.resolve()
                or not recorded_path.is_relative_to(run.run_directory.resolve())):
            raise ValueError("completed run state does not match final checkpoint")
        checkpoint = state.get("initial_checkpoint")
        if key[0] == "scratch":
            if checkpoint is not None:
                raise ValueError("scratch run has a frozen initializer")
        else:
            identity = FROZEN_BY_REPLICATE[key[1]]
            if checkpoint != {"id": identity, "sha256": FROZEN_CHECKPOINTS[identity], "policy_only": True}:
                raise ValueError("frozen initializer does not match registered replicate")
        with run.final_model_path.open("rb") as source:
            digest = sha256()
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
        if run.final_model_path.stat().st_size == 0:
            raise ValueError("final checkpoint must not be empty")
        digests[key] = digest.hexdigest()
    if not isinstance(records, list) or any(
        not isinstance(row, dict) or (row.get("cohort"), row.get("replicate")) not in digests
        or row.get("checkpoint_sha256") != digests[(row.get("cohort"), row.get("replicate"))]
        for row in records
    ):
        raise ValueError("development records do not match saved final checkpoints")
    clean_records = []
    for row in records:
        cohort = "train5_from_scratch" if row["cohort"] == "scratch" else "train5_from_frozen_checkpoint"
        seed_key = (cohort, row["replicate"], row["track_id"], row["condition"], row["episode_index"])
        if type(row.get("evaluation_seed")) is not int or expected_seeds.get(seed_key) != row["evaluation_seed"]:
            raise ValueError("development record seed differs from registered protocol")
        clean_records.append({key: value for key, value in row.items() if key != "evaluation_seed"})
    return evaluate_dev_metrics_v4(clean_records, contract, protocol, pilot)
