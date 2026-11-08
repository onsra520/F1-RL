"""TRAIN-only factory comparison and reset preparation; no rollout or admission."""
from dataclasses import replace
from pathlib import Path

from f1rl.learning.reconstruction import RecoveryCandidate, build_training_scenarios
from f1rl.learning.train_reset import TrainResetConfig
from f1rl.learning.training.env_factory import TrainingScenario, TrainingScenarioSet
from f1rl.research.wp3_eval_only_coordinator import FULL06_RUNTIME_PROFILE, apply_runtime_profile
from f1rl.research.wp7_protocol import load_registration, prepare_scenarios
from f1rl.research.wp7_train5_v2_protocol import CANDIDATE, MANIFEST, RUNTIME_SPEC, TRAIN5_IDS, _manifest_contract
from f1rl.research.wp7_train5_v4_admission import load_v4_protocol, require_v4_training_admission, validate_v4_protocol
from f1rl.sandbox.ballistic_flight import FlightState
from f1rl.sandbox.geometry.vector import Vec3
from f1rl.sandbox.independent_suspension_dynamics import SuspensionDynamicsState
from f1rl.track_assets import load_track_manifest
from scripts.wp7_train5_handoff_preflight import audit


def preview_registered_factories(root: Path, evidence_root: Path, *, cohort: str, replicate: int) -> TrainingScenarioSet:
    root = Path(root).resolve(strict=True)
    protocol, _ = load_v4_protocol(root)
    validate_v4_protocol(root, protocol)
    if type(replicate) is not int or replicate not in range(3):
        raise ValueError("replicate must be 0, 1, or 2")
    rows = [row for row in protocol["runs"] if row["cohort"] == cohort and row["replicate"] == replicate]
    if len(rows) != 1:
        raise ValueError("unknown registered cohort")
    receipt = audit(root, Path(evidence_root))
    if receipt["external_witness_status"] != "FOUR_EXTERNAL_WITNESSES_SHA256_VERIFIED":
        raise ValueError("external source witnesses are required")
    _manifest_contract(root)
    candidate = RecoveryCandidate.from_yaml(root / CANDIDATE)
    experiment = candidate.experiment
    if (experiment.metadata.id != "ppo-generalization-five-track-paired-v2"
            or experiment.metadata.status != "draft" or experiment.training.protocol_status != "pilot_owned"
            or experiment.tracks.training_ids != TRAIN5_IDS
            or experiment.tracks.manifest.resolve() != (root / MANIFEST).resolve()):
        raise ValueError("registered candidate or training roster differs")
    raw = build_training_scenarios(candidate)
    registration = load_registration(root, profile="spatial_rotating_owner_v3")
    template = next(row.factory for row in prepare_scenarios(root, registration, seed_index=0).scenarios
                    if row.track_id == "right-turn-v1")
    manifest = load_track_manifest(root / MANIFEST, validate_asset_ids=TRAIN5_IDS)
    by_id = {entry.track_id: entry for entry in manifest.tracks}
    reset = TrainResetConfig(seed=rows[0]["seeds"]["perturbation"],
                             training_registry=tuple(by_id[name] for name in TRAIN5_IDS),
                             version="wp7_train_reset_lifecycle_research_v1", max_attempts=32)
    result = []
    for row in raw.scenarios:
        height = template.vehicle_parameters.ride_height
        initial = replace(row.factory.initial_state,
                          vehicle=replace(row.factory.initial_state.vehicle,
                              position=row.factory.initial_state.vehicle.position + Vec3(0, height, 0)),
                          suspension=SuspensionDynamicsState(row.factory.initial_state.vehicle.position.y + height),
                          flight=FlightState("grounded"))
        factory = replace(template, track=row.factory.track, navigation=row.factory.navigation,
                          initial_state=initial, reset_config=None, reset_track_id=None)
        factory, provenance = apply_runtime_profile(factory, FULL06_RUNTIME_PROFILE)
        if provenance["effective"] != RUNTIME_SPEC or not factory.requires_scientific_admission:
            raise ValueError("registered physics profile or admission guard differs")
        factory = replace(factory, reset_config=reset, reset_track_id=row.track_id)
        result.append(TrainingScenario(row.track_id, factory))
    if tuple(row.track_id for row in result) != TRAIN5_IDS:
        raise ValueError("registered TRAIN-5 scenario roster differs")
    return TrainingScenarioSet(tuple(result))


def validate_registered_factories(root: Path, evidence_root: Path, scenarios: TrainingScenarioSet,
                                  *, cohort: str, replicate: int) -> None:
    if not isinstance(scenarios, TrainingScenarioSet):
        raise ValueError("a TRAIN-5 scenario set is required")
    expected = preview_registered_factories(root, evidence_root, cohort=cohort, replicate=replicate)
    if scenarios != expected:
        raise ValueError("scenario geometry, configuration or reset differs from registered factories")


def require_registered_factory_admission(root: Path, evidence_root: Path, scenarios: TrainingScenarioSet,
                                         *, cohort: str, replicate: int) -> None:
    validate_registered_factories(root, evidence_root, scenarios, cohort=cohort, replicate=replicate)
    require_v4_training_admission(root)
