"""TRAIN-5 v4 draft validation and non-authorizing admission assessment."""
from hashlib import sha256
import json
from pathlib import Path

from .wp7_train5_v4_checkpoint import FROZEN_BY_REPLICATE
from .wp7_trusted_reviews import verify_repository_run_reviews


class V4AdmissionError(ValueError):
    pass


SEED_DERIVATION = "SHA256 canonical JSON [domain, protocol_sha256, cohort, replicate, track_id, condition, episode_index]; first 4 bytes big-endian uint32; domain=F1RL-TRAIN5-V4-DEV-SEED; reject collisions among development, reserved training and vector-rank seeds before use"
POLICY_SEED_DERIVATION = "SHA256 canonical JSON [domain, protocol_sha256, cohort, replicate, track_id, condition, episode_index]; first 4 bytes big-endian uint32; domain=F1RL-TRAIN5-V4-DEV-POLICY-SEED; reject collisions with environment, training, historical and policy seeds before use"
ANALYSIS = ["report counts and rates for every policy, cohort, track, condition and 60-episode cell",
            "report 95% Wilson intervals within each cell only as conditional descriptive intervals; do not pool episodes as independent policies",
            "report the mean and range of policy-level rates per cohort; no formal between-cohort significance test with three policies per cohort",
            "report zero-event upper bounds and 60-episode detection probability 1-(1-p)^60 as conditional on independent episodes; no unqualified power claim",
            "apply the fixed thresholds without post-hoc tuning; describe cohort comparisons without causal finetuning claims"]
SEED_PROPOSAL_DOMAIN = "F1RL-TRAIN5-V4-SEED-PROPOSAL-20261108"
SOURCE_FILES = (
    "f1rl/research/wp7_train5_v4_admission.py", "f1rl/research/wp7_train5_v4_checkpoint.py",
    "f1rl/research/wp7_train5_v4_runner.py", "f1rl/research/wp7_train5_v4_budget.py",
    "f1rl/research/wp7_train5_v4_dev_metrics.py", "f1rl/research/wp7_train5_reset_perturbation_v3.py",
    "f1rl/learning/algorithms/ppo.py", "f1rl/learning/training/runner.py",
    "configs/research/wp7-train5-paired-v2-registration.json", "configs/research/wp7-core-readiness-matrix-v1.json",
    "configs/research/wp7-train5-v4-local-checkpoint-draft.json", "configs/recovery/wp7-pilot-200k.yml",
    "configs/research/wp7-train5-global-seed-registry-owner-exception-20261107.json",
)


def source_file_roster(root: Path, registered: dict) -> tuple[str, ...]:
    root = Path(root)
    python_sources = {p.relative_to(root).as_posix() for p in (root / "f1rl").rglob("*.py")}
    return tuple(sorted(set(SOURCE_FILES) | python_sources | set(registered["source_sha256"])))


def proposed_training_seed(cohort: str, replicate: int, role: str) -> int:
    value = f"{SEED_PROPOSAL_DOMAIN}/{cohort}/{replicate}/{role}".encode("ascii")
    return int.from_bytes(sha256(value).digest()[:4], "big")


def load_v4_protocol(root: Path) -> tuple[dict, str]:
    path = Path(root) / "configs/research/wp7-train5-v4-simulation-protocol-draft.json"
    data = path.read_bytes().replace(b"\r\n", b"\n")
    return json.loads(data), sha256(data).hexdigest()


def validate_v4_protocol(root: Path, protocol: dict) -> None:
    source = (Path(root) / "configs/research/wp7-train5-paired-v2-registration.json").read_bytes().replace(b"\r\n", b"\n")
    if sha256(source).hexdigest() != "3dd9ca9776b229e9e20bdc018cc1b67e93229d93a9754d53211197e8db60a576":
        raise V4AdmissionError("registered source changed")
    registered = json.loads(source)
    if not isinstance(protocol, dict) or protocol.get("schema") != "f1rl_train5_v4_simulation_protocol_draft_v1":
        raise V4AdmissionError("invalid protocol schema")
    bindings = protocol.get("source_bindings")
    roster = source_file_roster(Path(root), registered)
    if not isinstance(bindings, dict) or set(bindings) != set(roster):
        raise V4AdmissionError("protocol source roster differs")
    resolved_root = Path(root).resolve()
    for relative in roster:
        path = resolved_root / relative
        if path.is_symlink() or not path.resolve().is_relative_to(resolved_root):
            raise V4AdmissionError("protocol source must stay within checkout")
        try:
            contents = path.read_bytes()
        except OSError as exc:
            raise V4AdmissionError("protocol source is missing") from exc
        if path.suffix.lower() != ".zip":
            contents = contents.replace(b"\r\n", b"\n")
        if sha256(contents).hexdigest() != bindings[relative]:
            raise V4AdmissionError(f"protocol source changed: {relative}")
    if (protocol.get("status") != "DRAFT_NOT_AUTHORIZED"
            or protocol.get("claim_scope") != "SIMULATION_ONLY_FIVE_TRAIN_TRACKS"
            or protocol.get("excluded_claims") != ["real_vehicle_safety", "sim_to_real_accuracy", "physical_calibration", "heldout_generalization"]
            or protocol.get("track_ids") != registered["train_ids"]):
        raise V4AdmissionError("protocol scope or track roster differs")
    expected = []
    for cohort, entries in registered["proposed_training_seeds"].items():
        for i, _ in enumerate(entries):
            seeds = {role: proposed_training_seed(cohort, i, role) for role in ("ppo", "sampling", "perturbation")}
            expected.append({"cohort": cohort, "replicate": i, "seeds": seeds,
                             "initial_checkpoint": FROZEN_BY_REPLICATE[i] if cohort == "train5_from_frozen_checkpoint" else None})
    if protocol.get("runs") != expected:
        raise V4AdmissionError("run, seed or checkpoint allocation differs")
    seed_values = [seed for run in expected for seed in run["seeds"].values()]
    vector_ranges = [set(range(run["seeds"]["ppo"], run["seeds"]["ppo"] + 8)) for run in expected]
    historical = {seed for group in registered["proposed_training_seeds"].values() for run in group for seed in run.values()}
    historical_vectors = {run["ppo"] + rank for group in registered["proposed_training_seeds"].values() for run in group for rank in range(8)}
    if (len(set(seed_values)) != 18 or any(max(ranks) >= 2**32 for ranks in vector_ranges)
            or any(a & b for i, a in enumerate(vector_ranges) for b in vector_ranges[i + 1:])
            or any(ranks.intersection(set(seed_values) - {min(ranks)}) for ranks in vector_ranges)
            or (set(seed_values).union(*vector_ranges) & (historical | historical_vectors))):
        raise V4AdmissionError("proposed or vector seeds overlap")
    if protocol.get("budget") != {"requested_steps": 200000, "actual_final_steps": 200704,
                                   "vector_envs": 8, "rollout_steps": 256,
                                   "checkpoint_strategy": "FINAL_ONLY"}:
        raise V4AdmissionError("checkpoint budget differs")
    if protocol.get("reset") != {"lateral_m": [-.5, .5], "heading_deg": [-5, 5],
                                  "max_rejection_draws": 32, "calibration": "ENGINEERING_ASSUMPTION_NOT_EMPIRICAL"}:
        raise V4AdmissionError("reset assumptions differ")
    development = protocol.get("development", {})
    if (development.get("conditions") != ["nominal", "perturbation"]
            or development.get("policy_inference") != "stochastic_cpu"
            or development.get("policy_seed_derivation") != POLICY_SEED_DERIVATION
            or development.get("tilt_envelope_outcome") != "report_separately_as_valid_nonfinish"
            or (development.get("episodes_per_cell"), development.get("cells"), development.get("total_episodes")) != (60, 60, 3600)
            or development.get("seed_derivation") != SEED_DERIVATION or development.get("analysis") != ANALYSIS
            or development.get("thresholds") != {"nominal_finish_rate_min": .9, "per_track_finish_rate_min": .8,
                "hard_route_violation_rate_max": .02, "certified_physical_crash_rate_max": .01,
                "invalid_or_uncertified_physics_count_max": 0}):
        raise V4AdmissionError("development design or thresholds differ")
    if protocol.get("seed_exception") != {"historical_18_status": "CLOSED_WITH_OWNER_ACCEPTED_EVIDENCE_EXCEPTION",
            "historical_exception_applies_to_new_seeds": False, "new_seed_registry": "OPEN_LOCAL_AUDIT_ONLY",
            "missing_historical_job_logs": 15, "logs_recovered": False, "global_nonuse_proven": False}:
        raise V4AdmissionError("historical seed exception differs")
    history = protocol.get("seed_history", {})
    if (not isinstance(history, dict) or history.get("proposal_domain") != SEED_PROPOSAL_DOMAIN
            or history.get("searched_seed_values") != 60 or history.get("matching_files") != 0
            or history.get("git_reachable_matching_commits") != 0 or history.get("read_errors") != 0
            or history.get("global_nonuse_proven") is not False
            or history.get("new_registry_status") != "OPEN_LOCAL_AUDIT_ONLY"):
        raise V4AdmissionError("fresh seed history scope differs")
    permissions = protocol.get("permissions")
    if (not isinstance(permissions, dict)
            or set(permissions) != {"training", "finetuning", "heldout", "full06", "scientific_release"}
            or any(value is not False for value in permissions.values())):
        raise V4AdmissionError("draft cannot grant permissions")
    if protocol.get("review") != {"owner_signature": False, "independent_reviewer_signature": False,
                                   "statistical_power_review": "PENDING"}:
        raise V4AdmissionError("draft review status differs")


def development_seeds(root: Path, protocol: dict, protocol_sha256: str) -> dict:
    validate_v4_protocol(root, protocol)
    if not isinstance(protocol_sha256, str) or len(protocol_sha256) != 64 or any(c not in "0123456789abcdef" for c in protocol_sha256):
        raise V4AdmissionError("canonical protocol SHA-256 required")
    saved_protocol, saved_sha256 = load_v4_protocol(root)
    if protocol != saved_protocol or protocol_sha256 != saved_sha256:
        raise V4AdmissionError("development seeds require the saved protocol and its SHA-256")
    reserved = {seed for run in protocol["runs"] for seed in run["seeds"].values()}
    if len(reserved) != 18:
        raise V4AdmissionError("reserved seed collision")
    seeds = {}
    used = set(reserved)
    for run in protocol["runs"]:
        used.update(range(run["seeds"]["ppo"], run["seeds"]["ppo"] + protocol["budget"]["vector_envs"]))
    registration = json.loads((Path(root) / "configs/research/wp7-train5-paired-v2-registration.json").read_text(encoding="utf8"))
    for group in registration["proposed_training_seeds"].values():
        for run in group:
            used.update(run.values())
            used.update(range(run["ppo"], run["ppo"] + 8))
    for run in protocol["runs"]:
        for track in protocol["track_ids"]:
            for condition in protocol["development"]["conditions"]:
                for episode in range(60):
                    key = (run["cohort"], run["replicate"], track, condition, episode)
                    payload = ["F1RL-TRAIN5-V4-DEV-SEED", protocol_sha256, *key]
                    encoded = json.dumps(payload, separators=(",", ":"), ensure_ascii=True).encode("ascii")
                    seed = int.from_bytes(sha256(encoded).digest()[:4], "big")
                    if seed in used:
                        raise V4AdmissionError("development seed collision; no retry permitted")
                    seeds[key] = seed
                    used.add(seed)
    return seeds


def development_policy_seeds(root: Path, protocol: dict, protocol_sha256: str) -> dict:
    env = development_seeds(root, protocol, protocol_sha256)
    reserved = set(env.values())
    for run in protocol["runs"]:
        reserved.update(run["seeds"].values())
        reserved.update(range(run["seeds"]["ppo"], run["seeds"]["ppo"] + protocol["budget"]["vector_envs"]))
    registration = json.loads((Path(root) / "configs/research/wp7-train5-paired-v2-registration.json").read_text(encoding="utf8"))
    for group in registration["proposed_training_seeds"].values():
        for run in group:
            reserved.update(run.values())
            reserved.update(range(run["ppo"], run["ppo"] + 8))
    result = {}
    for key in env:
        payload = ["F1RL-TRAIN5-V4-DEV-POLICY-SEED", protocol_sha256, *key]
        encoded = json.dumps(payload, separators=(",", ":"), ensure_ascii=True).encode("ascii")
        seed = int.from_bytes(sha256(encoded).digest()[:4], "big")
        if seed in reserved:
            raise V4AdmissionError("policy seed collision; no retry permitted")
        result[key] = seed
        reserved.add(seed)
    return result


def training_vector_seed_overlaps(protocol: dict) -> dict:
    owners = {}
    for run in protocol["runs"]:
        for rank in range(protocol["budget"]["vector_envs"]):
            seed = run["seeds"]["ppo"] + rank
            owners.setdefault(seed, []).append((run["cohort"], run["replicate"], rank))
    return {seed: values for seed, values in owners.items() if len(values) > 1}


def assess_v4_admission(root: Path, *, review_envelope=None, run_manifest_sha256=None) -> dict:
    protocol, protocol_sha256 = load_v4_protocol(root)
    validate_v4_protocol(root, protocol)
    development_seeds(root, protocol, protocol_sha256)
    development_policy_seeds(root, protocol, protocol_sha256)
    matrix_bytes = (Path(root) / "configs/research/wp7-core-readiness-matrix-v1.json").read_bytes().replace(b"\r\n", b"\n")
    if sha256(matrix_bytes).hexdigest() != "2d7c126715f95e13ce93682c9fb70ea958d547fae605d736cbbbbac1e2ae9549":
        raise V4AdmissionError("compatibility matrix differs from locked source")
    matrix = json.loads(matrix_bytes)
    expected_subsystems = {"simulation_runtime", "vehicle_dynamics", "track_feasibility", "race_episode",
                           "gym_reset", "perception_control", "train_ppo", "evaluation", "provenance_admission"}
    if len(matrix["subsystems"]) != 9 or {s["id"] for s in matrix["subsystems"]} != expected_subsystems:
        raise V4AdmissionError("compatibility matrix does not contain the nine required subsystems")
    open_gates = [s["id"] for s in matrix["subsystems"] if s["compatibility"]["status"] != "passed"]
    signatures_match_registry = False
    if review_envelope is not None:
        try:
            verify_repository_run_reviews(root, review_envelope, protocol_sha256=protocol_sha256,
                                          run_manifest_sha256=run_manifest_sha256)
            signatures_match_registry = True
        except (OSError, ValueError, TypeError):
            pass
    return {"protocol_sha256": protocol_sha256, "training_permitted": False,
            "heldout_permitted": False, "open_compatibility_gates": open_gates,
            "training_vector_seed_overlap_count": len(training_vector_seed_overlaps(protocol)),
            "signatures_match_repository_registry": signatures_match_registry,
            "registry_independently_authenticated": False,
            "run_manifest_bytes_verified": False,
            "blockers": ["SCIENTIFIC_PROTOCOL_NOT_ADMITTED", "TRAINING_GUARDS_NOT_INTEGRATED",
                         "NEW_SEED_GLOBAL_REGISTRY_OPEN", "HISTORICAL_C2_AND_EXACT_COMMIT_CROSS_OS_EVIDENCE_REQUIRED"]
                        + (["PPO_VECTOR_SEED_RANK_OVERLAP"] if training_vector_seed_overlaps(protocol) else [])}


def require_v4_training_admission(root: Path, **kwargs) -> None:
    assess_v4_admission(root, **kwargs)
    raise V4AdmissionError("TRAIN-5 v4 training admission is not granted")
