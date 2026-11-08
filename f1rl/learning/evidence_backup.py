"""Deterministic local SHA256 manifest for independently restored model/evidence files.

This does NOT upload artifacts to any cloud service, and it does not attest
an off-device backup. A future operator must copy data to private Drive,
download into an independent directory, then verify this manifest there.
"""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
from typing import Any


def _sha(path: Path) -> str:
    h = sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            h.update(block)
    return h.hexdigest()


def _checked_relpath(value: str) -> Path:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("manifest paths must use safe relative POSIX notation")
    path=PurePosixPath(value)
    if path.is_absolute() or any(item in ("", ".", "..") for item in value.split("/")):
        raise ValueError("manifest entry escapes evidence directory")
    # POSIX-relative syntax alone does not exclude Windows drives, streams or aliases.
    for part in path.parts:
        if (part.endswith((".", " ")) or any(ord(c)<32 or c in '<>:"|?*' for c in part)
                or part.split(".",1)[0].upper() in
                {"CON","PRN","AUX","NUL",*(f"COM{i}" for i in "123456789¹²³"),*(f"LPT{i}" for i in "123456789¹²³")}):
            raise ValueError("manifest paths must use portable file names")
    return Path(*path.parts)


def build_evidence_manifest(directory: Path) -> dict[str, Any]:
    """Hash an evidence directory without following file or directory symlinks."""
    directory = Path(directory)
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("evidence root must be a physical directory")
    files = sorted(
        (p for p in directory.rglob("*") if p.is_file() or p.is_symlink()),
        key=lambda p: p.relative_to(directory).as_posix(),
    )
    if not files:
        raise ValueError("evidence directory contains no files")
    entries=[]
    for file in files:
        if file.is_symlink() or any(parent.is_symlink() for parent in file.parents if parent != directory):
            raise ValueError("evidence must not contain symlinks")
        relative = file.relative_to(directory).as_posix()
        _checked_relpath(relative)
        if not file.is_file():
            raise ValueError("evidence entries must be regular files")
        entries.append({
            "path": relative,
            "size_bytes": file.stat().st_size,
            "sha256": _sha(file),
        })
    return {
        "schema_version": 1,
        "algorithm": "sha256",
        "entries": entries,
    }


def verify_evidence_manifest(directory: Path, document: dict[str, Any]) -> int:
    """Verify a copied/restored directory and require no extra/missing files."""
    directory=Path(directory)
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("restore root must be a physical directory")
    if (
        not isinstance(document, dict) or set(document) != {"schema_version","algorithm","entries"}
        or type(document["schema_version"]) is not int or document["schema_version"] != 1
        or document["algorithm"] != "sha256"
        or not isinstance(document["entries"], list) or not document["entries"]
    ):
        raise ValueError("invalid evidence manifest")
    registered=set()
    for entry in document["entries"]:
        if not isinstance(entry, dict) or set(entry) != {"path","size_bytes","sha256"}:
            raise ValueError("invalid evidence entry")
        relative=_checked_relpath(entry["path"])
        if relative.as_posix() in registered:
            raise ValueError("duplicate evidence entry")
        registered.add(relative.as_posix())
        if type(entry["size_bytes"]) is not int or entry["size_bytes"] < 0:
            raise ValueError("invalid evidence size")
        digest=entry["sha256"]
        if (
            not isinstance(digest,str) or len(digest)!=64
            or any(x not in "0123456789abcdef" for x in digest)
        ):
            raise ValueError("invalid evidence SHA256")
        restored = directory/relative
        if restored.is_symlink() or any(
            parent.is_symlink() for parent in restored.parents if parent != directory
        ):
            raise ValueError("evidence symlinks are forbidden")
        if not restored.is_file() or restored.stat().st_size != entry["size_bytes"] or _sha(restored) != digest:
            raise ValueError(f"evidence restore differs for {relative.as_posix()}")
    actual={
        p.relative_to(directory).as_posix()
        for p in directory.rglob("*") if p.is_file() or p.is_symlink()
    }
    if actual != registered:
        raise ValueError("evidence restore contains extra or missing files")
    return len(registered)


def serialize_manifest(document: dict[str, Any]) -> str:
    """Stable bytes suitable for a separately retained digest/signoff."""
    return json.dumps(document,ensure_ascii=False,indent=2,sort_keys=True)+"\n"
