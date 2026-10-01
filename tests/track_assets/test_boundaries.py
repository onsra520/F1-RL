import ast
from pathlib import Path

import f1rl.track_assets as track_assets


PRODUCTION = Path("f1rl")
EXPECTED_PUBLIC_API = {
    "SpawnPose",
    "TrackAsset",
    "TrackManifest",
    "TrackManifestEntry",
    "load_track_asset",
    "load_track_manifest",
    "build_initial_world_state",
}
EXPECTED_SB3_OWNERS = {
    Path("f1rl/learning/algorithms/ppo.py"),
    Path("f1rl/learning/training/vector_env.py"),
}


def _imports(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name, 0
        elif isinstance(node, ast.ImportFrom):
            yield node.module or "", node.level


def _imports_prefix(path: Path, prefixes: tuple[str, ...], relative_names: tuple[str, ...] = ()) -> bool:
    for module, level in _imports(path):
        if any(module == prefix or module.startswith(prefix + ".") for prefix in prefixes):
            return True
        if level and any(module == name or module.startswith(name + ".") for name in relative_names):
            return True
    return False


def test_track_assets_public_api_is_exact():
    assert set(track_assets.__all__) == EXPECTED_PUBLIC_API
    assert all(hasattr(track_assets, name) for name in EXPECTED_PUBLIC_API)


def test_sandbox_does_not_import_track_assets():
    offenders = [
        path
        for path in (PRODUCTION / "sandbox").rglob("*.py")
        if _imports_prefix(path, ("f1rl.track_assets",), ("track_assets",))
    ]
    assert offenders == []


def test_track_assets_depends_only_on_allowed_lower_level_code():
    banned = (
        "f1rl.learning",
        "f1rl.runtime",
        "gymnasium",
        "stable_baselines3",
        "torch",
        "fastapi",
    )
    offenders = [
        path
        for path in (PRODUCTION / "track_assets").rglob("*.py")
        if _imports_prefix(path, banned, ("learning", "runtime"))
    ]
    assert offenders == []


def test_stable_baselines3_import_owners_remain_exactly_two():
    owners = {
        path
        for path in PRODUCTION.rglob("*.py")
        if _imports_prefix(path, ("stable_baselines3",))
    }
    assert owners == EXPECTED_SB3_OWNERS


def test_production_has_no_direct_torch_imports():
    offenders = [
        path
        for path in PRODUCTION.rglob("*.py")
        if _imports_prefix(path, ("torch",))
    ]
    assert offenders == []


def test_production_has_no_mock_or_fake_identifiers():
    offenders = []
    for path in PRODUCTION.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        names = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                names.append(node.name)
            elif isinstance(node, ast.Name):
                names.append(node.id)
        if any(name.startswith(("Mock", "Fake")) for name in names):
            offenders.append(path)
    assert offenders == []
