"""Fail-closed WP7 candidate milestone evaluation and evidence registry.

This is deliberately NOT a training command. It is attachable to the existing
TrainingRunner and remains blocked for real execution until review gates close.
"""
from __future__ import annotations

from collections.abc import Callable
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Any

from .pilot_preflight import PILOT_MILESTONES, TRAIN_IDS, build_pilot_preflight
from .train_checkpoint import evaluate_training_checkpoint


_SHA_RE = re.compile(r"[0-9a-f]{64}\Z")


def _hash_file(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _json_object(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _write_new_json(path: Path, document: dict[str, object]) -> None:
    """Single-owner atomic same-directory publish; never replace known evidence."""
    if path.exists():
        raise FileExistsError(f"pilot report already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"pilot report temporary file already exists: {temporary}")
    encoded = (json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    try:
        with temporary.open("xb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        if path.exists():
            raise FileExistsError(f"pilot report already exists: {path}")
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _git_revision(root: Path) -> tuple[str, bool]:
    try:
        revision = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True, encoding="utf-8",
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"],
            check=True, capture_output=True, text=True, encoding="utf-8",
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError("pilot recorder requires a Git worktree") from error
    if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ValueError("pilot recorder cannot verify Git HEAD")
    return revision, bool(dirty)


class PilotMilestoneRecorder:
    """Attach on_checkpoint/on_final to TrainingRunner; real pilot gated.

    A mock-mode observer may use an injected evaluator to validate artifact
    lifecycle without running PPO. Such records are NEVER scientific evidence.
    """

    def __init__(
        self,
        *,
        repository_root: str | Path,
        candidate_path: str | Path,
        runtime_config_path: str | Path,
        evaluator: Callable[..., dict[str, object]] | None = None,
        episodes_per_track: int = 1,
        mock_mode: bool = False,
    ) -> None:
        if type(mock_mode) is not bool:
            raise ValueError("mock_mode must be bool")
        if isinstance(episodes_per_track, bool) or not isinstance(episodes_per_track, int) or episodes_per_track <= 0:
            raise ValueError("episodes_per_track must be positive integer")
        if evaluator is not None and not callable(evaluator):
            raise ValueError("evaluator must be callable or None")
        if mock_mode and evaluator is None:
            raise ValueError("mock_mode requires an explicit mock evaluator")
        self._root = Path(repository_root).resolve(strict=True)
        self._candidate = Path(candidate_path).resolve(strict=True)
        self._profile = Path(runtime_config_path).resolve(strict=True)
        self._preflight = build_pilot_preflight(
            repository_root=self._root,
            candidate_path=self._candidate,
            runtime_config_path=self._profile,
        )
        self._git_sha, self._initial_dirty = _git_revision(self._root)
        if not mock_mode and (self._preflight["review_blockers"] or self._initial_dirty):
            raise RuntimeError("WP7 pilot review is required; refuse live evidence execution")
        self._mock = mock_mode
        self._evaluator = evaluator or evaluate_training_checkpoint
        self._episodes_per_track = episodes_per_track
        self._fingerprints = tuple(dict(item) for item in self._preflight["fingerprints"])
        self._milestones: list[int] = []
        self._reports: list[dict[str, object]] = []
        self._run_directory: Path | None = None

    @property
    def source_fingerprints(self) -> tuple[dict[str, str], ...]:
        return tuple(dict(item) for item in self._fingerprints)

    @property
    def milestones_recorded(self) -> tuple[int, ...]:
        return tuple(self._milestones)

    def _verify_source_identity(self) -> None:
        revision, dirty = _git_revision(self._root)
        if revision != self._git_sha:
            raise ValueError("pilot source Git HEAD changed during evaluation")
        if not self._mock and dirty:
            raise ValueError("pilot Git worktree became dirty")
        for entry in self._fingerprints:
            path = (self._root / entry["path"]).resolve(strict=True)
            if not path.is_relative_to(self._root) or _hash_file(path) != entry["sha256"]:
                raise ValueError(f"pilot source sha256 changed: {entry['path']}")

    def _verify_previous_evidence(self) -> None:
        """Detect changes to both previous reports and their saved model archives."""
        if self._run_directory is None:
            return
        for record in self._reports:
            report_path = self._run_directory / str(record["path"])
            if not report_path.is_file() or _hash_file(report_path) != record["sha256"]:
                raise ValueError("pilot report sha256 changed after publication")
            document = _json_object(json.loads(report_path.read_text("utf-8")), "past pilot report")
            archive = (self._run_directory / str(document["model_path"])).resolve(strict=True)
            if not archive.is_relative_to(self._run_directory) or _hash_file(archive) != document["checkpoint_sha256"]:
                raise ValueError("pilot checkpoint sha256 changed after report publication")

    def _record(self, archive_path: Path, actual_steps: int, *, final: bool) -> None:
        if isinstance(actual_steps, bool) or not isinstance(actual_steps, int) or actual_steps <= 0:
            raise ValueError("pilot actual timestep must be positive integer")
        if len(self._milestones) >= len(PILOT_MILESTONES):
            raise ValueError("pilot milestone already completed")
        expected = PILOT_MILESTONES[len(self._milestones)]
        if final != (expected == PILOT_MILESTONES[-1]):
            raise ValueError("pilot final milestone order is incorrect")
        limit = (
            PILOT_MILESTONES[len(self._milestones)+1]
            if len(self._milestones)+1 < len(PILOT_MILESTONES)
            else PILOT_MILESTONES[-1] + 25_000
        )
        if not expected <= actual_steps < limit:
            kind = "final milestone" if final else "pilot milestone"
            raise ValueError(f"{kind} {expected} skipped or overshot: {actual_steps}")
        archive = Path(archive_path).resolve(strict=True)
        if not archive.is_file() or archive.suffix.lower() != ".zip":
            raise ValueError("pilot checkpoint archive must be an existing .zip")
        run_dir = archive.parent if final else archive.parent.parent
        if final:
            if not archive.name.endswith("-final-model.zip"):
                raise ValueError("pilot final milestone needs final-model archive")
        else:
            if archive.parent.name != "checkpoints" or not archive.name.endswith(f"-step{actual_steps}.zip"):
                raise ValueError("pilot checkpoint archive name does not match actual steps")
        if self._run_directory is None:
            self._run_directory = run_dir
        if self._run_directory != run_dir:
            raise ValueError("pilot checkpoint moved between run directories")
        state_path = run_dir / "state.json"
        state = _json_object(json.loads(state_path.read_text(encoding="utf-8")), "run state")
        if state.get("status") != "running":
            raise ValueError("pilot cannot append evidence to a non-running training state")
        if state.get("training_tracks") != list(TRAIN_IDS):
            raise ValueError("pilot run must own exactly four TRAIN tracks")
        self._verify_source_identity()
        self._verify_previous_evidence()
        prior_hash = _hash_file(archive)
        if re.fullmatch(_SHA_RE, prior_hash) is None:
            raise ValueError("pilot checkpoint sha256 invalid")

        evaluated = _json_object(self._evaluator(
            repository_root=self._root,
            candidate_path=self._candidate,
            runtime_config_path=self._profile,
            checkpoint_path=archive,
            episodes_per_track=self._episodes_per_track,
        ), "checkpoint evaluation")
        after_hash = _hash_file(archive)
        if evaluated.get("checkpoint_sha256") != prior_hash or after_hash != prior_hash:
            raise ValueError("pilot checkpoint sha256 changed during evaluation")
        if (
            evaluated.get("checkpoint_name") != archive.name
            or evaluated.get("track_ids") != list(TRAIN_IDS)
            or evaluated.get("heldout_rollouts") != 0
            or evaluated.get("scientific_evidence") is not False
            or evaluated.get("episode_count_per_track") != self._episodes_per_track
            or evaluated.get("candidate_source_fingerprints") != list(self._fingerprints)
        ):
            raise ValueError("pilot evaluation violated TRAIN-only evidence contract")
        if self._mock and evaluated.get("policy_source") != "injected_test_policy_not_scientific_evidence":
            raise ValueError("mock pilot must label evaluation as non-scientific")
        summaries = evaluated.get("track_summaries")
        if (
            not isinstance(summaries, list)
            or [part.get("track_id") if isinstance(part, dict) else None for part in summaries] != list(TRAIN_IDS)
        ):
            raise ValueError("pilot evaluation summaries do not match TRAIN tracks")
        self._verify_source_identity()

        model_path = archive.relative_to(run_dir).as_posix()
        report: dict[str, object] = {
            "schema_version": "wp7-recovery-pilot-milestone-v1",
            "status": "mock_verified_not_scientific_evidence" if self._mock else "candidate_unreviewed",
            "scientific_evidence": False,
            "git_commit": self._git_sha,
            "git_dirty_at_creation": self._initial_dirty,
            "milestone_timesteps": expected,
            "actual_timesteps": actual_steps,
            "is_final": final,
            "model_path": model_path,
            "checkpoint_sha256": prior_hash,
            "source_fingerprints": list(self._fingerprints),
            "evaluation": evaluated,
            "heldout_rollouts": 0,
        }
        relative_report = f"evaluations/milestone-{expected:06d}.json"
        full_report = run_dir / relative_report
        _write_new_json(full_report, report)
        record = {"path": relative_report, "sha256": _hash_file(full_report)}
        self._reports.append(record)
        self._milestones.append(expected)
        if final:
            if tuple(self._milestones) != PILOT_MILESTONES:
                raise ValueError("pilot milestone set is incomplete")
            _write_new_json(run_dir / "evaluations" / "manifest.json", {
                "schema_version": "wp7-recovery-pilot-manifest-v1",
                "status": "mock_verified_not_scientific_evidence" if self._mock else "candidate_unreviewed",
                "scientific_evidence": False,
                "git_commit": self._git_sha,
                "milestones": list(self._milestones),
                "reports": self._reports,
                "source_fingerprints": list(self._fingerprints),
                "heldout_rollouts": 0,
            })

    def on_checkpoint(self, archive_path: Path, actual_steps: int) -> None:
        self._record(archive_path, actual_steps, final=False)

    def on_final(self, archive_path: Path, actual_steps: int) -> None:
        self._record(archive_path, actual_steps, final=True)



def verify_pilot_evidence_manifest(run_directory: str | Path) -> dict[str, object]:
    """Read-only re-verification for a copied or archived MOCK evidence directory.

    This proves file identity, not scientific correctness or PPO performance.
    """
    root = Path(run_directory).resolve(strict=True)
    manifest_path = root / "evaluations" / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError("pilot evidence manifest is missing")
    manifest = _json_object(json.loads(manifest_path.read_text("utf-8")), "pilot evidence manifest")
    if (
        manifest.get("schema_version") != "wp7-recovery-pilot-manifest-v1"
        or manifest.get("status") != "mock_verified_not_scientific_evidence"
        or manifest.get("scientific_evidence") is not False
        or manifest.get("heldout_rollouts") != 0
        or manifest.get("milestones") != list(PILOT_MILESTONES)
    ):
        raise ValueError("pilot evidence manifest is not a verified mock milestone set")
    reports = manifest.get("reports")
    if not isinstance(reports, list) or len(reports) != len(PILOT_MILESTONES):
        raise ValueError("pilot evidence must contain eight report hashes")
    for step, entry in zip(PILOT_MILESTONES, reports, strict=True):
        if not isinstance(entry, dict) or set(entry) != {"path", "sha256"}:
            raise ValueError("pilot report index has invalid format")
        expected_path = f"evaluations/milestone-{step:06d}.json"
        if entry["path"] != expected_path:
            raise ValueError("pilot report path mismatch")
        file = root / expected_path
        if not file.is_file() or _hash_file(file) != entry["sha256"]:
            raise ValueError("pilot report sha256 mismatch")
        report = _json_object(json.loads(file.read_text("utf-8")), "pilot report")
        if (
            report.get("milestone_timesteps") != step
            or report.get("is_final") is not (step == PILOT_MILESTONES[-1])
            or report.get("scientific_evidence") is not False
            or report.get("heldout_rollouts") != 0
            or report.get("source_fingerprints") != manifest.get("source_fingerprints")
            or report.get("git_commit") != manifest.get("git_commit")
        ):
            raise ValueError("pilot report contract mismatch")
        actual = report.get("actual_timesteps")
        if isinstance(actual, bool) or not isinstance(actual, int) or not step <= actual < step + 25_000:
            raise ValueError("pilot report actual timestep outside milestone window")
        model_path = report.get("model_path")
        if not isinstance(model_path, str) or not model_path:
            raise ValueError("pilot report lacks model path")
        model = (root / model_path).resolve(strict=True)
        if not model.is_relative_to(root) or model.suffix.lower() != ".zip":
            raise ValueError("pilot report model path leaves run directory")
        if _hash_file(model) != report.get("checkpoint_sha256"):
            raise ValueError("pilot checkpoint sha256 mismatch")
    return {
        "status": "verified_mock_evidence_not_scientific",
        "verified_milestones": len(reports),
        "heldout_rollouts": 0,
        "scientific_evidence": False,
    }
