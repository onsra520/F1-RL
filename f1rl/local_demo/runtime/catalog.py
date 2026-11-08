from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, replace
from hashlib import sha256
from itertools import pairwise
from math import atan2, hypot
from pathlib import Path, PurePosixPath, PureWindowsPath
from types import MappingProxyType

import gymnasium as gym
import numpy as np

from f1rl.learning.reconstruction import RecoveryCandidate, build_training_scenarios
from f1rl.learning.training.env_factory import F1RLEnvFactory, TrainingScenario
from f1rl.local_demo.runtime.exploratory_models import (
    EXPLORATORY_IDS,
    ExploratoryModel,
    discover_exploratory_models,
    load_exploratory_policy,
)
from f1rl.research import wp3_eval_only_coordinator, wp7_train5_v2_protocol
from f1rl.research.wp7_protocol import load_registration, prepare_scenarios
from f1rl.sandbox.geometry.vector import Vec3
from f1rl.track_assets import load_track_manifest
from f1rl.track_assets.canonical import fingerprint_track_world

MAX_SEED = 2**32 - 1
MAX_RUNTIME_STEPS = 5000
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_MODEL_ID = re.compile(r"[A-D]-seed[0-2]\Z")


@dataclass(frozen=True, slots=True)
class FrozenModel:
    model_id: str
    sha256: str
    bytes: int


@dataclass(frozen=True, slots=True)
class RuntimeTrack:
    track_id: str
    split: str = "TRAIN"
    purpose: str = "development"
    preview: dict[str, object] | None = None
    road_samples: tuple[dict[str, float], ...] = ()


def _unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("runtime registry contains duplicate fields")
        value[key] = item
    return value


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _registration(root: Path) -> dict[str, object]:
    path = _inside(root, wp7_train5_v2_protocol.REGISTRATION)
    document = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
    if not isinstance(document, dict):
        raise ValueError("TRAIN-5 source registration must be an object")
    digest_document = {key: value for key, value in document.items() if key != "preflight_digest"}
    digest = sha256(json.dumps(digest_document, sort_keys=True, allow_nan=False, separators=(",", ":")).encode("utf-8")).hexdigest()
    if (
        document.get("preflight_digest") != digest
        or document.get("status") != "draft_not_authorized"
        or document.get("baseline_head") != "1de279c37e62a27531d661480d20fa6e095c4e84"
        or document.get("train_ids") != list(wp7_train5_v2_protocol.TRAIN5_IDS)
        or document.get("heldout_ids") != list(wp7_train5_v2_protocol.HELDOUT_IDS)
        or document.get("seed_audit_status") != "PROPOSED_UNVERIFIED_GLOBAL"
        or document.get("training_permission") is not False
        or document.get("smoke_permission") is not False
        or document.get("heldout_permission") is not False
        or document.get("owner_approval") is not None
        or document.get("global_seed_registry_receipt") is not None
    ):
        raise ValueError("TRAIN-5 registration must remain source-valid and fail-closed")
    return document


def _source_file_matches(root: Path, relative: str, expected: str, bridge: dict[str, object]) -> bool:
    actual = _file_sha256(_inside(root, relative))
    if actual == expected:
        return True
    record = bridge.get("source_blobs", {}).get(relative)
    return (
        isinstance(record, dict)
        and record.get("relation") == "CRLF_to_LF_only"
        and actual == record.get("git_blob_sha256")
    )


def _inside(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise ValueError("runtime input path must be a safe relative path")
    path = (root / relative).resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("runtime input must be a file inside the repository")
    if path.is_relative_to(root / "assets/tracks/evaluation"):
        raise ValueError("runtime catalog cannot access HELD-OUT geometry")
    return path


def _validate_train_manifest_geometry_paths(root: Path, manifest_path: Path) -> None:
    document = json.loads(manifest_path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
    if not isinstance(document, dict) or not isinstance(document.get("tracks"), list):
        raise ValueError("TRAIN track manifest must contain a tracks array")
    heldout_root = (root / "assets/tracks/evaluation").resolve(strict=False)
    for entry in document["tracks"]:
        if not isinstance(entry, dict):
            raise ValueError("TRAIN track manifest entries must be objects")
        if entry.get("track_id") not in wp7_train5_v2_protocol.TRAIN5_IDS:
            continue
        relative = entry.get("path")
        if (not isinstance(relative, str) or not relative or "\\" in relative
                or PurePosixPath(relative).is_absolute() or PureWindowsPath(relative).drive
                or ".." in PurePosixPath(relative).parts):
            raise ValueError("TRAIN geometry manifest path must be a safe relative POSIX path")
        target = (manifest_path.parent / Path(*PurePosixPath(relative).parts)).resolve(strict=False)
        if not target.is_relative_to(root) or target == heldout_root or target.is_relative_to(heldout_root):
            raise ValueError("TRAIN geometry manifest path cannot access HELD-OUT geometry")


def _load_registered_train_manifest(root: Path, registration: dict, bridge: dict):
    relative = wp7_train5_v2_protocol.MANIFEST
    manifest_path = _inside(root, relative)
    sources = registration.get("source_sha256")
    expected = sources.get(relative) if isinstance(sources, Mapping) else None
    if not isinstance(expected, str) or not _source_file_matches(root, relative, expected, bridge):
        raise ValueError("TRAIN track manifest source binding mismatch")
    _validate_train_manifest_geometry_paths(root, manifest_path)
    return load_track_manifest(manifest_path, validate_asset_ids=wp7_train5_v2_protocol.TRAIN5_IDS)


def runtime_source_sha256(root: Path) -> str:
    root = root.resolve(strict=True)
    paths = [*root.joinpath("f1rl").rglob("*.py"), root / "pyproject.toml", root / "run.py"]
    digest = sha256()
    for path in sorted(set(paths)):
        resolved = path.resolve(strict=True)
        if not resolved.is_file() or not resolved.is_relative_to(root):
            raise ValueError("runtime source identity file must stay inside the repository")
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def runtime_inputs_sha256(root: Path) -> str:
    root = root.resolve(strict=True)
    registration = _registration(root)
    sources = registration.get("source_sha256")
    bridge = json.loads(
        _inside(root, "configs/research/wp7-train5-paired-v2-git-blob-lock.json").read_text(encoding="utf-8"),
        object_pairs_hook=_unique_object,
    )
    required = (
        wp7_train5_v2_protocol.MANIFEST,
        wp7_train5_v2_protocol.CANDIDATE,
        wp7_train5_v2_protocol.EXPERIMENT,
        "configs/research/wp7-rotating-owner-protocol-v3.json",
        "configs/research/wp7-spatial-ground-protocol-v1.json",
        "configs/research/wp7-hard-route-outcome-v2.json",
        "configs/recovery/wp7-owner-decisions-preregistered-v1.yml",
    )
    entries = _load_registered_train_manifest(root, registration, bridge)
    paths = [_inside(root, wp7_train5_v2_protocol.REGISTRATION), _inside(root, "configs/research/wp7-train5-paired-v2-git-blob-lock.json")]
    paths.extend(_inside(root, name) for name in required)
    for entry in entries.tracks:
        if entry.track_id in wp7_train5_v2_protocol.TRAIN5_IDS:
            relative = f"assets/tracks/{entry.path}"
            expected = sources.get(relative) if isinstance(sources, Mapping) else None
            if not isinstance(expected, str) or not _source_file_matches(root, relative, expected, bridge):
                raise ValueError(f"TRAIN geometry source binding mismatch: {entry.track_id}")
            paths.append(_inside(root, relative))
    for model_id in wp7_train5_v2_protocol.MODEL_IDS:
        relative = f"models/{model_id}/model-final.zip"
        expected = sources.get(relative) if isinstance(sources, Mapping) else None
        path = _inside(root, relative)
        if not isinstance(expected, str) or _SHA256.fullmatch(expected) is None or _file_sha256(path) != expected:
            raise ValueError(f"frozen model source binding mismatch: {model_id}")
        paths.append(path)
    digest = sha256()
    for path in sorted(set(paths)):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _spaces_compatible(first: gym.Space, second: gym.Space) -> bool:
    if type(first) is not type(second) or first.shape != second.shape or first.dtype != second.dtype:
        return False
    if isinstance(first, gym.spaces.Box):
        return isinstance(second, gym.spaces.Box) and np.array_equal(first.low, second.low) and np.array_equal(first.high, second.high)
    return first == second


class RuntimeCatalog:
    """Read-only development catalog for frozen model identity and TRAIN-5 factories."""

    def __init__(self, repository_root: str | Path) -> None:
        self.root = Path(repository_root).resolve(strict=True)
        self._models, self._model_paths = self._read_models()
        self._exploratory_models, self._exploratory_paths, self._exploratory_warnings = discover_exploratory_models(self.root)
        self._track_factories: Mapping[str, F1RLEnvFactory] | None = None
        self._track_hashes: Mapping[str, str] | None = None
        self._source_sha256: str | None = None
        self._inputs_sha256: str | None = None

    @property
    def models(self) -> tuple[FrozenModel, ...]:
        return self._models

    @property
    def exploratory_models(self) -> tuple[ExploratoryModel, ...]:
        return self._exploratory_models

    @property
    def exploratory_warnings(self) -> tuple[str, ...]:
        return self._exploratory_warnings

    @property
    def tracks(self) -> tuple[RuntimeTrack, ...]:
        self._load_tracks()
        return tuple(self._track_preview(track_id) for track_id in wp7_train5_v2_protocol.TRAIN5_IDS)

    def _track_preview(self, track_id: str) -> RuntimeTrack:
        factory = self.track_factory(track_id)
        navigation = factory.navigation
        if navigation is None:
            raise ValueError("registered TRAIN track requires authored navigation")
        surfaces = {surface.id: surface for surface in factory.track.surfaces}
        width = min(surfaces[surface_id].width for surface_id in navigation.surface_ids)
        cumulative = [0.0]
        for first, second in pairwise(navigation.points):
            cumulative.append(cumulative[-1] + (second - first).magnitude)
        total = cumulative[-1]
        samples: list[dict[str, float]] = []
        for index, (first, second) in enumerate(pairwise(navigation.points)):
            length = (second - first).magnitude
            subdivisions = max(1, int(length / 2.0))
            for step in range(subdivisions + 1):
                if index and step == 0:
                    continue
                fraction = step / subdivisions
                x = first.x + (second.x - first.x) * fraction
                y = first.y + (second.y - first.y) * fraction
                z = first.z + (second.z - first.z) * fraction
                dx, dz = second.x - first.x, second.z - first.z
                horizontal = hypot(dx, dz)
                progress = cumulative[index] + length * fraction
                samples.append({
                    "x": x, "y": y, "z": z,
                    "t": progress / total if total else 0.0,
                    "heading": atan2(dx, dz) if horizontal else 0.0,
                    "nx": -dz / horizontal if horizontal else 1.0,
                    "nz": dx / horizontal if horizontal else 0.0,
                })
        props = [{
            "id": f"{track_id}-checkpoint-{order + 1}",
            "kind": "checkpoint",
            "t": cumulative[point_index] / total if total else 0.0,
            "offset": 0.0,
            "rotation": 0.0,
        } for order, point_index in enumerate(navigation.checkpoint_point_indices)]
        preview: dict[str, object] = {
            "schema": "f1rl_racing_sandbox_draft_v1",
            "id": track_id,
            "name": track_id.replace("-", " ").title(),
            "width": width,
            "points": [{"x": point.x, "z": point.z} for point in navigation.points],
            "props": props,
        }
        return RuntimeTrack(track_id, preview=preview, road_samples=tuple(samples))

    def model(self, model_id: str) -> FrozenModel | ExploratoryModel:
        for entry in (*self._models, *self._exploratory_models):
            if entry.model_id == model_id:
                return entry
        raise ValueError("unregistered runtime model id")

    def track_factory(self, track_id: str) -> F1RLEnvFactory:
        self._load_tracks()
        try:
            return self._track_factories[track_id]
        except (KeyError, TypeError) as error:
            raise ValueError("unknown TRAIN track id") from error

    def create_environment(self, track_id: str, *, seed: int, mode: str):
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= MAX_SEED:
            raise ValueError(f"seed must be an integer in [0, {MAX_SEED}]")
        if mode not in ("manual", "auto", "ppo", "hybrid"):
            raise ValueError("unknown development runtime mode")
        factory = self.track_factory(track_id)
        return replace(
            factory,
            episode_config=replace(factory.episode_config, max_steps=min(factory.episode_config.max_steps, MAX_RUNTIME_STEPS)),
        ).create()

    def track_sha256(self, track_id: str) -> str:
        self._load_tracks()
        try:
            return self._track_hashes[track_id]
        except KeyError as error:
            raise ValueError("unknown TRAIN track id") from error

    @property
    def source_sha256(self) -> str:
        if self._source_sha256 is None:
            self._source_sha256 = runtime_source_sha256(self.root)
        return self._source_sha256

    @property
    def inputs_sha256(self) -> str:
        if self._inputs_sha256 is None:
            self._inputs_sha256 = runtime_inputs_sha256(self.root)
        return self._inputs_sha256

    def load_model(self, model_id: str, *, env=None, observation_encoder=None):
        entry = self.model(model_id)
        if model_id in EXPLORATORY_IDS:
            path = self._exploratory_paths.get(model_id)
            if path is None or path.resolve(strict=True) != path:
                raise ValueError("exploratory model checkpoint path changed before loading")
            if path.stat().st_size != entry.bytes or _file_sha256(path) != entry.sha256:
                raise ValueError("exploratory model checkpoint integrity mismatch")
            return load_exploratory_policy(path, entry.sha256, env, observation_encoder)
        path = self._model_paths[model_id]
        if path.stat().st_size != entry.bytes or _file_sha256(path) != entry.sha256:
            raise ValueError("frozen model checkpoint integrity mismatch")
        from f1rl.learning.algorithms.ppo import PPOModel

        model = PPOModel.load(path, device="cpu")
        if env is not None and (
            not _spaces_compatible(model._model.observation_space, env.observation_space)
            or not _spaces_compatible(model._model.action_space, env.action_space)
        ):
            raise ValueError("frozen model observation/action spaces are incompatible with the runtime environment")
        return model

    def _read_models(self) -> tuple[tuple[FrozenModel, ...], Mapping[str, Path]]:
        document = _registration(self.root)
        if (
            not isinstance(document, dict)
            or document.get("status") != "draft_not_authorized"
            or document.get("training_permission") is not False
            or document.get("smoke_permission") is not False
            or document.get("heldout_permission") is not False
        ):
            raise ValueError("TRAIN-5 runtime registry must remain in its fail-closed draft state")
        sources = document.get("source_sha256")
        if not isinstance(sources, dict):
            raise ValueError("runtime registry source bindings are missing")
        records: list[FrozenModel] = []
        paths: dict[str, Path] = {}
        for model_id in wp7_train5_v2_protocol.MODEL_IDS:
            if _MODEL_ID.fullmatch(model_id) is None:
                raise ValueError("unregistered frozen model identity")
            relative = f"models/{model_id}/model-final.zip"
            path = _inside(self.root, relative)
            digest = sources.get(relative)
            if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None or _file_sha256(path) != digest:
                raise ValueError(f"frozen model source binding mismatch: {model_id}")
            records.append(FrozenModel(model_id, digest, path.stat().st_size))
            paths[model_id] = path
        return tuple(records), MappingProxyType(paths)

    def _load_tracks(self) -> None:
        if self._track_factories is not None:
            return
        root = self.root
        registration = _registration(root)
        bridge = json.loads(
            _inside(root, "configs/research/wp7-train5-paired-v2-git-blob-lock.json").read_text(encoding="utf-8"),
            object_pairs_hook=_unique_object,
        )
        if (
            registration.get("train_ids") != list(wp7_train5_v2_protocol.TRAIN5_IDS)
            or registration.get("heldout_ids") != list(wp7_train5_v2_protocol.HELDOUT_IDS)
            or registration.get("training_permission") is not False
            or registration.get("heldout_permission") is not False
        ):
            raise ValueError("runtime registration roster or closed admission state changed")
        manifest_path = _inside(root, wp7_train5_v2_protocol.MANIFEST)
        manifest = _load_registered_train_manifest(root, registration, bridge)
        candidate = RecoveryCandidate.from_yaml(_inside(root, wp7_train5_v2_protocol.CANDIDATE))
        experiment = candidate.experiment
        if (
            experiment.metadata.status != "draft"
            or experiment.training.protocol_status != "pilot_owned"
            or experiment.tracks.training_ids != wp7_train5_v2_protocol.TRAIN5_IDS
            or experiment.tracks.heldout_ids != wp7_train5_v2_protocol.HELDOUT_IDS
        ):
            raise ValueError("runtime candidate must keep the registered five TRAIN / two HELD-OUT split")
        raw = build_training_scenarios(candidate)
        owner = load_registration(root, profile="spatial_rotating_owner_v3")
        template = next(
            row.factory for row in prepare_scenarios(root, owner, seed_index=0).scenarios
            if row.track_id == "right-turn-v1"
        )
        from f1rl.sandbox.ballistic_flight import FlightState
        from f1rl.sandbox.independent_suspension_dynamics import SuspensionDynamicsState

        height = template.vehicle_parameters.ride_height
        factories: dict[str, F1RLEnvFactory] = {}
        hashes = {}
        manifest_entries = {entry.track_id: entry for entry in manifest.tracks}
        for row in raw.scenarios:
            entry = manifest_entries[row.track_id]
            registered = registration["source_sha256"]
            relative = f"assets/tracks/{entry.path}"
            if not _source_file_matches(root, relative, registered.get(relative), bridge):
                raise ValueError(f"TRAIN geometry source binding mismatch: {row.track_id}")
            initial = replace(
                row.factory.initial_state,
                vehicle=replace(
                    row.factory.initial_state.vehicle,
                    position=row.factory.initial_state.vehicle.position + Vec3(0.0, height, 0.0),
                ),
                suspension=SuspensionDynamicsState(row.factory.initial_state.vehicle.position.y + height),
                flight=FlightState("grounded"),
            )
            factory = replace(
                template,
                track=row.factory.track,
                navigation=row.factory.navigation,
                initial_state=initial,
                reset_config=None,
                reset_track_id=None,
            )
            factory, provenance = wp3_eval_only_coordinator.apply_runtime_profile(
                factory, wp3_eval_only_coordinator.FULL06_RUNTIME_PROFILE
            )
            if (
                provenance["effective"] != wp7_train5_v2_protocol.RUNTIME_SPEC
                or factory.reset_config is not None
                or factory.action_contract.version != "steering-throttle-brake-v2"
            ):
                raise ValueError("TRAIN-5 runtime factory differs from its source-bound v4 profile")
            factories[row.track_id] = factory
            hashes[row.track_id] = entry.world_sha256
        if tuple(factories) != wp7_train5_v2_protocol.TRAIN5_IDS:
            raise ValueError("runtime catalog must contain exactly the registered TRAIN-5 roster")
        self._track_factories = MappingProxyType(factories)
        self._track_hashes = MappingProxyType(hashes)
