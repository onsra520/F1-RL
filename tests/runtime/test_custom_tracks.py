from pathlib import Path

import numpy as np
import pytest

from src.runtime.catalog import RuntimeCatalog
from src.runtime.custom_tracks import TrackValidationError, convert_custom_track

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def catalog():
    return RuntimeCatalog(ROOT)


def draft(points, props=(), *, width=12):
    return {
        "schema": "f1rl_racing_sandbox_draft_v1",
        "id": "sandbox-test",
        "name": "Runtime test circuit",
        "width": width,
        "points": [{"x": x, "z": z} for x, z in points],
        "props": list(props),
    }


def prop(kind, t, *, offset=0, rotation=0, identity=None):
    return {"id": identity or f"{kind}-{t}", "kind": kind, "t": t, "offset": offset, "rotation": rotation}


def test_custom_track_converts_to_authoritative_world_and_manual_gym_environment(catalog):
    document = draft([(0, 0), (0, 40), (20, 60)], [prop("checkpoint", 0.55)])
    converted = convert_custom_track(document, catalog)

    assert converted.schema_version == "f1rl_custom_track_conversion_v1"
    assert converted.development_only is True
    assert converted.factory.reset_config is None
    assert converted.factory.navigation is not None
    assert len(converted.factory.track.checkpoints) == 1
    assert len(converted.road_samples) >= len(converted.factory.navigation.points)

    env = converted.factory.create()
    try:
        observation, _ = env.reset(seed=73)
        assert np.isfinite(observation).all()
        next_observation, _, _, _, _ = env.step(np.zeros(3, dtype=np.float32))
        assert np.isfinite(next_observation).all()
        snapshot = env.runtime_snapshot()
        assert snapshot["tick"] == 1
        assert np.isfinite(snapshot["position"]).all()
    finally:
        env.close()


def test_custom_track_adapts_ramps_and_converts_separated_overpass_decks(catalog):
    ramped = convert_custom_track(draft([(0, 0), (0, 80)], [prop("ramp", 0.5)]), catalog)
    assert max(sample["y"] for sample in ramped.road_samples) >= 2.7
    assert any(abs(surface.normal.y - 1.0) > 1e-4 for surface in ramped.factory.track.surfaces)

    crossing = convert_custom_track(draft([(0, 0), (100, 100), (0, 100), (100, 0), (160, 0)]), catalog)
    assert crossing.auto_bridge_count == 1
    assert max(sample["y"] for sample in crossing.road_samples) >= 1.8
    assert all(surface.route_index is not None for surface in crossing.factory.track.surfaces)


def test_custom_track_converts_elevated_turn_and_reports_short_approach(catalog):
    elevated = convert_custom_track(draft([(0, 0), (0, 60), (40, 100), (80, 60), (80, 0)], [
        prop("bridge", 0.5),
    ]), catalog)
    assert max(sample["y"] for sample in elevated.road_samples) >= 4.9
    assert max(abs(surface.normal.x) + abs(surface.normal.z) for surface in elevated.factory.track.surfaces) > 0.1

    with pytest.raises(TrackValidationError, match="lengthen the map") as error:
        convert_custom_track(draft([(0, 0), (0, 40)], [prop("bridge", 0.03)]), catalog)
    assert error.value.code == "unsupported_slope"


def test_custom_track_maps_bridge_obstacle_boost_checkpoint_and_finish(catalog):
    converted = convert_custom_track(draft([(0, 0), (0, 110)], [
        prop("bridge", 0.5),
        prop("barrier", 0.4, offset=-4),
        prop("cone", 0.5, offset=4),
        prop("booster", 0.6),
        prop("ramp", 0.75),
        prop("checkpoint", 0.88),
    ]), catalog)

    assert max(sample["y"] for sample in converted.road_samples) >= 4.9
    ramp_height = min(converted.road_samples, key=lambda sample: abs(sample["t"] - 0.75))["y"]
    assert 2.5 <= ramp_height <= 2.9
    assert len(converted.factory.track.obstacles) == 2
    assert len(converted.factory.track.boost_pads) == 1
    assert len(converted.factory.track.checkpoints) == 1
    assert len(converted.factory.track.finish_lines) == 1


def test_custom_track_rejects_unhashable_prop_kind_without_crashing(catalog):
    with pytest.raises(TrackValidationError) as error:
        convert_custom_track(draft([(0, 0), (0, 40)], [{
            "id": "bad-kind", "kind": [], "t": 0.5, "offset": 0, "rotation": 0,
        }]), catalog)
    assert error.value.code == "invalid_props"


@pytest.mark.parametrize("document,code", [
    ({}, "invalid_schema"),
    (draft([(0, 0), (0, 40)] + [(i, 0) for i in range(511)]), "resource_limit"),
    (draft([(0, 0), (0, 40)], [prop("checkpoint", 0.01)]), "checkpoint_order"),
    (draft([(0, 0), (0, 40)], [prop("checkpoint", 0.5, offset=1)]), "unsupported_checkpoint_transform"),
])
def test_custom_track_reports_specific_validation_codes(catalog, document, code):
    with pytest.raises(TrackValidationError) as error:
        convert_custom_track(document, catalog)
    assert error.value.code == code
