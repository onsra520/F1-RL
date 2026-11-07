"""Versioned TRAIN-5 v3 reset adapter: preparation/engineering fixtures only.

Wraps source-bound TRAIN-5 v2 five-track preview and existing safe lifecycle
TrainResetConfig without changing frozen source. No PPO or scientific admission.
"""
from __future__ import annotations
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
import json

from ..learning.train_reset import TrainResetConfig
from ..learning.training.env_factory import TrainingScenario, TrainingScenarioSet
from ..track_assets import load_track_manifest
from .wp7_train5_v2_protocol import (
    MANIFEST, TRAIN5_IDS, REGISTRATION, preview_train5_v2,
)

VERSION = "f1rl_train5_v3_engineering_reset_adapter_v1"
FIXTURE_SEEDS = (20261107, 20261108, 20261109)
RESET_VERSION = "wp7_train_reset_lifecycle_research_v1"
MAX_ATTEMPTS = 32


def prepare_train5_perturbed_scenarios(
    root: Path, *, evidence_root: Path, perturbation_seed: int,
    registration_file: Path | None = None,
) -> TrainingScenarioSet:
    """Return 5 registered TRAIN-only stochastic-reset factories; NO admission."""
    root = Path(root).resolve(strict=True)
    if type(perturbation_seed) is not int or not 0 <= perturbation_seed <= 2**32 - 1:
        raise ValueError("perturbation seed must be uint32")
    registration_file = registration_file or root / REGISTRATION
    draft = json.loads(registration_file.read_text(encoding="utf8"))
    if draft["status"] != "draft_not_authorized" or draft["training_permission"] is not False:
        raise ValueError("frozen draft source cannot authorize training")
    reserved = {
        value for group in draft["proposed_training_seeds"].values()
        for row in group for value in row.values()
    }
    if len(reserved) != 18 or perturbation_seed in reserved:
        raise ValueError("fixture seed cannot consume a reserved training seed")
    original = preview_train5_v2(
        root, registration_file=registration_file, evidence_root=evidence_root,
    )
    manifest = load_track_manifest(
        root / MANIFEST, validate_asset_ids=TRAIN5_IDS,
    )
    by_id = {entry.track_id: entry for entry in manifest.tracks}
    entries = tuple(by_id[track] for track in TRAIN5_IDS)
    reset = TrainResetConfig(
        seed=perturbation_seed, training_registry=entries,
        version=RESET_VERSION, max_attempts=MAX_ATTEMPTS,
    )
    revised = []
    for scenario in original.scenarios:
        if scenario.track_id not in TRAIN5_IDS:
            raise ValueError("unexpected non-TRAIN track")
        if scenario.factory.reset_config is not None:
            raise ValueError("nominal preview already contains reset config")
        factory = replace(
            scenario.factory, reset_config=reset,
            reset_track_id=scenario.track_id,
        )
        if not factory.requires_scientific_admission:
            raise ValueError("cannot disable scientific admission")
        revised.append(TrainingScenario(scenario.track_id, factory))
    result = TrainingScenarioSet(tuple(revised))
    if tuple(x.track_id for x in result.scenarios) != TRAIN5_IDS:
        raise ValueError("wrong five-track order")
    return result


def fixture_identity(root: Path, evidence_root: Path, seed: int) -> dict:
    """Read-only reference identity; never allocates registered training seeds."""
    scenario_set = prepare_train5_perturbed_scenarios(
        root, evidence_root=evidence_root, perturbation_seed=seed
    )
    identities = {
        row.track_id: row.factory.reset_config.identity_for_track(row.track_id)
        for row in scenario_set.scenarios
    }
    return {
        "schema": VERSION, "seed": seed,
        "factory_count": len(scenario_set.scenarios),
        "reset_contract": RESET_VERSION,
        "max_attempts": MAX_ATTEMPTS, "tracks": identities,
        "training_permission": False,
        "scientific_validation": False,
        "used_reserved_training_seeds": False,
    }
