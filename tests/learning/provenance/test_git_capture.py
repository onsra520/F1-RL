from pathlib import Path
import subprocess

import pytest

from f1rl.learning.provenance import _capture_git_provenance


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test User")
    tracked = repo / "tracked.txt"
    tracked.write_text("base\n", encoding="utf-8")
    _git(repo, "add", "tracked.txt")
    _git(repo, "commit", "-m", "initial")
    return repo


def test_clean_repo_captures_head_and_clean_status(tmp_path: Path) -> None:
    repo = _repo(tmp_path)

    value = _capture_git_provenance(repo)

    assert value.commit_sha == _git(repo, "rev-parse", "HEAD").stdout.strip()
    assert value.dirty is False


def test_unstaged_tracked_modification_is_dirty(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    (repo / "tracked.txt").write_text("changed\n", encoding="utf-8")
    assert _capture_git_provenance(repo).dirty is True


def test_staged_modification_is_dirty(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    (repo / "tracked.txt").write_text("changed\n", encoding="utf-8")
    _git(repo, "add", "tracked.txt")
    assert _capture_git_provenance(repo).dirty is True


def test_untracked_nonignored_file_is_dirty(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    (repo / "new.txt").write_text("new\n", encoding="utf-8")
    assert _capture_git_provenance(repo).dirty is True


def test_ignored_file_alone_does_not_mark_dirty(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    (repo / ".gitignore").write_text("ignored.txt\n", encoding="utf-8")
    _git(repo, "add", ".gitignore")
    _git(repo, "commit", "-m", "ignore rule")
    (repo / "ignored.txt").write_text("ignored\n", encoding="utf-8")

    assert _capture_git_provenance(repo).dirty is False


def test_ignored_file_does_not_hide_real_dirty_content(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    (repo / ".gitignore").write_text("ignored.txt\n", encoding="utf-8")
    _git(repo, "add", ".gitignore")
    _git(repo, "commit", "-m", "ignore rule")
    (repo / "ignored.txt").write_text("ignored\n", encoding="utf-8")
    (repo / "real.txt").write_text("real\n", encoding="utf-8")

    assert _capture_git_provenance(repo).dirty is True


def test_git_capture_rejects_missing_or_non_directory_root(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="repository_root"):
        _capture_git_provenance(tmp_path / "missing")

    file_path = tmp_path / "file"
    file_path.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="repository_root"):
        _capture_git_provenance(file_path)


def test_git_capture_rejects_non_repository(tmp_path: Path) -> None:
    directory = tmp_path / "plain"
    directory.mkdir()
    with pytest.raises(RuntimeError, match="Git"):
        _capture_git_provenance(directory)
