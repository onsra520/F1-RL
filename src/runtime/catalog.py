from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, replace
from hashlib import sha256
from itertools import pairwise
from math import atan2, hypot
from pathlib import Path
from threading import Lock
from types import MappingProxyType

from src.rl.pilot_preflight import TRAIN_IDS

MODEL_MANIFEST = Path(
    "docs/research/evidence/wp3-issue51-frozen-final-models-20261028"
    "/WP3-FROZEN-MODELS-SHA256-MANIFEST.json"
)
MODEL_DIRECTORY = Path("docs/research/evidence/wp3-issue51-frozen-final-models-20261028")
RUNTIME_PROFILE = "spatial_rotating_owner_v3"
MAX_SEED = 2**32 - 1
MAX_RUNTIME_STEPS = 5000
_MODEL_ID = re.compile(r"[A-D]-seed[0-2]\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_EXPECTED_MODEL_IDS = tuple(
    f"{candidate}-seed{seed}" for candidate in "ABCD" for seed in range(3)
)


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


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("frozen model manifest contains duplicate fields")
        value[key] = item
    return value


class RuntimeCatalog:
    """Read-only registry of the frozen PPO checkpoints and TRAIN environments."""

    _cpu_lock = Lock()
    _cpu_configured = False

    def __init__(self, repository_root: str | Path) -> None:
        self.root = Path(repository_root).resolve(strict=True)
        self._models, self._model_paths = self._read_models()
        self._track_factories: Mapping[str, object] | None = None
        self._source_sha256: str | None = None

    @property
    def models(self) -> tuple[FrozenModel, ...]:
        return self._models

    @property
    def tracks(self) -> tuple[RuntimeTrack, ...]:
        self._load_tracks()
        return tuple(self._track_preview(track_id) for track_id in TRAIN_IDS)

    def _track_preview(self, track_id: str) -> RuntimeTrack:
        factory = self.track_factory(track_id)
        navigation = factory.navigation
        if navigation is None:
            raise ValueError("registered TRAIN track requires authored navigation")
        points = navigation.points
        surfaces = {surface.id: surface for surface in factory.track.surfaces}
        widths = [surfaces[surface_id].width for surface_id in navigation.surface_ids]
        # Registered tracks have a fixed width per route segment; represent the
        # narrowest authored segment in the editor's single-width preview.
        width = min(widths)
        cumulative = [0.0]
        for first, second in pairwise(points):
            cumulative.append(cumulative[-1] + (second - first).magnitude)
        total = cumulative[-1]
        samples: list[dict[str, float]] = []
        for index, (first, second) in enumerate(pairwise(points)):
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
                heading = atan2(dx, dz) if horizontal else 0.0
                progress = cumulative[index] + length * fraction
                samples.append({
                    "x": x, "y": y, "z": z,
                    "t": progress / total if total else 0.0,
                    "heading": heading,
                    "nx": -dz / horizontal if horizontal else 1.0,
                    "nz": dx / horizontal if horizontal else 0.0,
                })
        props = []
        for order, point_index in enumerate(navigation.checkpoint_point_indices):
            props.append({
                "id": f"{track_id}-checkpoint-{order + 1}",
                "kind": "checkpoint",
                "t": cumulative[point_index] / total if total else 0.0,
                "offset": 0.0,
                "rotation": 0.0,
            })
        preview: dict[str, object] = {
            "schema": "f1rl_racing_sandbox_draft_v1",
            "id": track_id,
            "name": track_id.replace("-", " ").title(),
            "width": width,
            "points": [{"x": point.x, "z": point.z} for point in points],
            "props": props,
        }
        return RuntimeTrack(track_id, preview=preview, road_samples=tuple(samples))

    def model(self, model_id: str) -> FrozenModel:
        if not isinstance(model_id, str):
            raise TypeError("model id must be a string")
        for entry in self._models:
            if entry.model_id == model_id:
                return entry
        raise ValueError("unregistered frozen model id")

    def track_factory(self, track_id: str):
        self._load_tracks()
        try:
            return self._track_factories[track_id]
        except (KeyError, TypeError) as error:
            raise ValueError("unknown TRAIN track id") from error

    def create_environment(self, track_id: str, *, seed: int):
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= MAX_SEED:
            raise ValueError(f"seed must be an integer in [0, {MAX_SEED}]")
        factory = self.track_factory(track_id)
        if factory.reset_config is None:
            raise ValueError("registered TRAIN environment requires its reset contract")
        return replace(
            factory,
            reset_config=replace(factory.reset_config, seed=seed),
            episode_config=replace(
                factory.episode_config,
                max_steps=min(factory.episode_config.max_steps, MAX_RUNTIME_STEPS),
            ),
        ).create()

    @property
    def source_sha256(self) -> str:
        if self._source_sha256 is None:
            digest = sha256()
            paths = sorted((self.root / "src").rglob("*.py"))
            paths.extend((
                self.root / "pyproject.toml",
                self.root / "configs/research/wp7-rotating-owner-protocol-v3.json",
                self.root / "configs/recovery/wp7-c2-candidate.yml",
            ))
            for path in sorted(set(paths)):
                if not path.is_file() or not path.resolve(strict=True).is_relative_to(self.root):
                    raise ValueError("runtime source identity file must stay inside the repository")
                digest.update(path.relative_to(self.root).as_posix().encode("utf-8"))
                digest.update(b"\0")
                with path.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(chunk)
            self._source_sha256 = digest.hexdigest()
        return self._source_sha256

    def track_sha256(self, track_id: str) -> str:
        factory = self.track_factory(track_id)
        if factory.reset_config is None:
            raise ValueError("registered TRAIN environment requires its reset contract")
        return factory.reset_config.identity_for_track(track_id)["world_sha256"]

    def load_model(self, model_id: str, *, env=None):
        entry = self.model(model_id)
        path = self._model_paths[model_id]
        if _file_sha256(path) != entry.sha256:
            raise ValueError("frozen model checkpoint SHA-256 changed before loading")

        from src.rl.algorithms.ppo import (
            PPOModel,
            configure_cpu_runtime,
            validate_cpu_runtime,
        )

        with self._cpu_lock:
            if not self._cpu_configured:
                try:
                    configure_cpu_runtime(1, 1)
                except RuntimeError:
                    validate_cpu_runtime(1)
                self.__class__._cpu_configured = True
        return PPOModel.load(path, env=env, device="cpu")

    def _read_models(self) -> tuple[tuple[FrozenModel, ...], Mapping[str, Path]]:
        manifest_path = (self.root / MODEL_MANIFEST).resolve(strict=True)
        if not manifest_path.is_relative_to(self.root):
            raise ValueError("frozen model manifest must stay inside the repository")
        try:
            document = json.loads(
                manifest_path.read_text(encoding="utf-8"),
                object_pairs_hook=_unique_object,
            )
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ValueError("frozen model manifest could not be read") from error
        if (
            not isinstance(document, dict)
            or document.get("schema_version") != "wp3_issue51_frozen_final_models_evidence_v1"
            or document.get("count") != len(_EXPECTED_MODEL_IDS)
            or document.get("heldout_access") is not False
            or document.get("retraining") is not False
            or not isinstance(document.get("models"), list)
            or len(document["models"]) != len(_EXPECTED_MODEL_IDS)
        ):
            raise ValueError("frozen model manifest does not match the runtime registry contract")

        records = []
        paths = {}
        for raw in document["models"]:
            if not isinstance(raw, dict):
                raise TypeError("frozen model manifest entries must be objects")
            model_id = raw.get("model_id")
            digest = raw.get("sha256")
            file_name = raw.get("file")
            size = raw.get("bytes")
            if (
                set(raw) != {"archive_members", "bytes", "file", "model_id", "sha256"}
                or not isinstance(model_id, str)
                or _MODEL_ID.fullmatch(model_id) is None
                or model_id in paths
                or not isinstance(digest, str)
                or _SHA256.fullmatch(digest) is None
            ):
                raise ValueError("invalid frozen model manifest entry")
            if not isinstance(file_name, str):
                raise TypeError("frozen model manifest path must be a string")
            if file_name != f"models/{model_id}/model-final.zip":
                raise ValueError("frozen model manifest contains an unregistered model path")
            if isinstance(size, bool) or not isinstance(size, int):
                raise TypeError("frozen model manifest size must be an integer")
            if size <= 0:
                raise ValueError("frozen model manifest size must be positive")
            path = (self.root / MODEL_DIRECTORY / file_name).resolve(strict=True)
            if not path.is_relative_to(self.root) or not path.is_file():
                raise ValueError("frozen model checkpoint must stay inside the repository")
            if path.stat().st_size != size or _file_sha256(path) != digest:
                raise ValueError(f"frozen model checkpoint integrity mismatch: {model_id}")
            records.append(FrozenModel(model_id, digest, size))
            paths[model_id] = path

        if tuple(record.model_id for record in records) != _EXPECTED_MODEL_IDS:
            raise ValueError("frozen model registry must contain A/B/C/D seed 0/1/2 in order")
        return tuple(records), MappingProxyType(paths)

    def _load_tracks(self) -> None:
        if self._track_factories is not None:
            return
        from src.validation.wp7_registration import load_registration
        from src.validation.wp7_scenarios import prepare_scenarios

        registration = load_registration(self.root, profile=RUNTIME_PROFILE)
        scenarios = prepare_scenarios(self.root, registration, seed_index=0)
        factories = {row.track_id: row.factory for row in scenarios.scenarios}
        if tuple(factories) != TRAIN_IDS:
            raise ValueError("registered TRAIN environment registry changed")
        self._track_factories = MappingProxyType(factories)
