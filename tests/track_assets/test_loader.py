import json

import pytest

from src.simulation.entities.boost_pad import BoostPad
from src.simulation.entities.checkpoint import Checkpoint
from src.simulation.entities.finish_line import FinishLine
from src.simulation.entities.obstacle import Obstacle
from src.simulation.entities.score_pickup import ScorePickup
from src.simulation.entities.slow_zone import SlowZone
from src.simulation.track.patch import PlanarSurfacePatch
from src.tracks import load_track_asset


def _volume(center):
    return {
        "center": center,
        "half_extents": [1, 1, 1],
        "orientation": [1, 0, 0, 0],
    }


def _asset_dict():
    return {
        "schema_version": 1,
        "track_id": "all-categories",
        "spawn": {"position": [0, 0, 0], "orientation": [1, 0, 0, 0]},
        "surfaces": [
            {
                "id": "surface-main",
                "origin": [0, 0, 20],
                "forward": [0, 0, 1],
                "normal": [0, 1, 0],
                "width": 12,
                "length": 40,
            }
        ],
        "obstacles": [{"id": "obstacle-1", "volume": _volume([1, 1, 10])}],
        "boost_pads": [
            {"id": "boost-1", "volume": _volume([0, 1, 15]), "acceleration_bonus": 2}
        ],
        "slow_zones": [
            {"id": "slow-1", "volume": _volume([0, 1, 20]), "drag_multiplier": 1.5}
        ],
        "checkpoints": [
            {"id": "checkpoint-0", "volume": _volume([0, 1, 25]), "order": 0}
        ],
        "finish_lines": [{"id": "finish", "volume": _volume([0, 1, 35])}],
        "score_pickups": [
            {"id": "pickup-1", "volume": _volume([0, 1, 30]), "points": 5}
        ],
    }


def _write(tmp_path, data, name="track.json"):
    path = tmp_path / name
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_loads_complete_asset_into_all_sandbox_categories(tmp_path):
    asset = load_track_asset(_write(tmp_path, _asset_dict()))
    assert asset.track_id == "all-categories"
    assert isinstance(asset.world.surfaces[0], PlanarSurfacePatch)
    assert isinstance(asset.world.obstacles[0], Obstacle)
    assert isinstance(asset.world.boost_pads[0], BoostPad)
    assert isinstance(asset.world.slow_zones[0], SlowZone)
    assert isinstance(asset.world.checkpoints[0], Checkpoint)
    assert isinstance(asset.world.finish_lines[0], FinishLine)
    assert isinstance(asset.world.score_pickups[0], ScorePickup)
    assert asset.world.checkpoints[0].order == 0
    assert asset.world.score_pickups[0].points == 5


@pytest.mark.parametrize("missing", ["track_id", "spawn", "surfaces", "finish_lines"])
def test_missing_root_key_is_rejected(tmp_path, missing):
    data = _asset_dict()
    del data[missing]
    with pytest.raises(ValueError, match=missing):
        load_track_asset(_write(tmp_path, data))


def test_unknown_root_and_nested_keys_are_rejected(tmp_path):
    root = _asset_dict()
    root["extra"] = 1
    with pytest.raises(ValueError, match="extra"):
        load_track_asset(_write(tmp_path, root, "root.json"))

    nested = _asset_dict()
    nested["spawn"]["extra"] = 1
    with pytest.raises(ValueError, match="extra"):
        load_track_asset(_write(tmp_path, nested, "nested.json"))


def test_duplicate_keys_are_rejected_at_root_and_nested_levels(tmp_path):
    root = tmp_path / "duplicate-root.json"
    root.write_text('{"schema_version":1,"schema_version":1}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate.*schema_version"):
        load_track_asset(root)

    nested = tmp_path / "duplicate-nested.json"
    text = json.dumps(_asset_dict())
    text = text.replace('"position": [0, 0, 0]', '"position": [0, 0, 0], "position": [1, 0, 0]')
    nested.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate.*position"):
        load_track_asset(nested)


@pytest.mark.parametrize("schema", [True, 0, 3, "1"])
def test_schema_version_is_restricted_to_v1_v2(tmp_path, schema):
    data = _asset_dict()
    data["schema_version"] = schema
    with pytest.raises(ValueError, match="schema_version"):
        load_track_asset(_write(tmp_path, data))


@pytest.mark.parametrize("track_id", ["", "   ", 7, None])
def test_track_id_must_be_non_empty_string(tmp_path, track_id):
    data = _asset_dict()
    data["track_id"] = track_id
    with pytest.raises(ValueError, match="track_id"):
        load_track_asset(_write(tmp_path, data))


def test_vector_and_quaternion_lengths_are_strict(tmp_path):
    vector = _asset_dict()
    vector["spawn"]["position"] = [0, 0]
    with pytest.raises(ValueError, match="position"):
        load_track_asset(_write(tmp_path, vector, "vector.json"))

    quaternion = _asset_dict()
    quaternion["spawn"]["orientation"] = [1, 0, 0]
    with pytest.raises(ValueError, match="orientation"):
        load_track_asset(_write(tmp_path, quaternion, "quat.json"))


def test_bool_is_rejected_for_continuous_and_discrete_numbers(tmp_path):
    continuous = _asset_dict()
    continuous["surfaces"][0]["width"] = True
    with pytest.raises(ValueError, match="width"):
        load_track_asset(_write(tmp_path, continuous, "continuous.json"))

    discrete = _asset_dict()
    discrete["checkpoints"][0]["order"] = True
    with pytest.raises(ValueError, match="order"):
        load_track_asset(_write(tmp_path, discrete, "discrete.json"))


def test_non_finite_json_constants_and_overflow_are_rejected(tmp_path):
    for token in ("NaN", "Infinity", "-Infinity"):
        path = tmp_path / f"constant-{token.replace('-', 'neg')}.json"
        path.write_text(json.dumps(_asset_dict()).replace('"width": 12', f'"width": {token}'), encoding="utf-8")
        with pytest.raises(ValueError, match="finite|constant|width"):
            load_track_asset(path)

    overflow = tmp_path / "overflow.json"
    overflow.write_text(json.dumps(_asset_dict()).replace('"width": 12', '"width": 1e309'), encoding="utf-8")
    with pytest.raises(ValueError, match="width"):
        load_track_asset(overflow)


def test_domain_validation_failures_keep_field_context(tmp_path):
    bad_surface = _asset_dict()
    bad_surface["surfaces"][0]["forward"] = [0, 1, 0]
    with pytest.raises(ValueError, match="surfaces"):
        load_track_asset(_write(tmp_path, bad_surface, "surface.json"))

    bad_volume = _asset_dict()
    bad_volume["obstacles"][0]["volume"]["half_extents"] = [0, 1, 1]
    with pytest.raises(ValueError, match="obstacles"):
        load_track_asset(_write(tmp_path, bad_volume, "volume.json"))


def test_world_wide_duplicate_ids_are_rejected(tmp_path):
    data = _asset_dict()
    data["obstacles"][0]["id"] = "surface-main"
    with pytest.raises(ValueError, match="ids.*unique|unique.*ids"):
        load_track_asset(_write(tmp_path, data))


def test_source_format_key_order_and_entity_array_order_do_not_change_hashes(tmp_path):
    first = _asset_dict()
    second = _asset_dict()
    second["surfaces"].append(
        {
            "id": "surface-a",
            "origin": [0, 0, 60],
            "forward": [0, 0, 1],
            "normal": [0, 1, 0],
            "width": 12.0,
            "length": 20.0,
        }
    )
    first["surfaces"] = list(reversed(second["surfaces"]))

    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    a.write_text(json.dumps(first, indent=2), encoding="utf-8")
    b.write_text(json.dumps(second, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    loaded_a = load_track_asset(a)
    loaded_b = load_track_asset(b)
    assert loaded_a.world_sha256 == loaded_b.world_sha256
    assert loaded_a.content_sha256 == loaded_b.content_sha256
