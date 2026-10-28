from __future__ import annotations

from dataclasses import dataclass

from ...tracks import TrackManifestEntry, load_track_manifest
from .definition import TrackExperimentConfig


@dataclass(frozen=True, slots=True)
class ResolvedTrackSplit:
    training: tuple[TrackManifestEntry, ...]
    heldout: tuple[TrackManifestEntry, ...]

    def __post_init__(self) -> None:
        for name, entries in (("training", self.training), ("heldout", self.heldout)):
            if not isinstance(entries, tuple) or not entries:
                raise ValueError(f"{name} must be a non-empty tuple")
            if not all(isinstance(entry, TrackManifestEntry) for entry in entries):
                raise ValueError(f"{name} must contain TrackManifestEntry values")


def resolve_track_split(config: TrackExperimentConfig) -> ResolvedTrackSplit:
    if not isinstance(config, TrackExperimentConfig):
        raise ValueError("config must be TrackExperimentConfig")

    manifest = load_track_manifest(config.manifest, validate_asset_ids=config.training_ids)
    by_id = {entry.track_id: entry for entry in manifest.tracks}
    def resolve(ids: tuple[str, ...], partition: str) -> tuple[TrackManifestEntry, ...]:
        entries: list[TrackManifestEntry] = []
        for track_id in ids:
            try:
                entries.append(by_id[track_id])
            except KeyError as exc:
                raise ValueError(
                    f"{partition} track_id is missing from manifest: {track_id}"
                ) from exc
        return tuple(entries)

    training = resolve(config.training_ids, "training")
    heldout = resolve(config.heldout_ids, "heldout")
    selected = (*training, *heldout)
    world_hashes = tuple(entry.world_sha256 for entry in selected)
    if len(world_hashes) != len(set(world_hashes)):
        raise ValueError(
            "selected research tracks must have unique world_sha256 geometry identities"
        )

    return ResolvedTrackSplit(training, heldout)
