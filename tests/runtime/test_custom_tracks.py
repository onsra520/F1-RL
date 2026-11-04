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


@pytest.mark.parametrize("points", [
    [(0, 0), (100, 100), (0, 100), (100, 0), (0, 0),
     (100, 100), (0, 100), (100, 0), (180, 0)],
    [(0, 0), (100, 100), (0, 100), (100, 0), (0, 20),
     (100, 120), (0, 120), (100, 20), (180, 20)],
])
def test_unsupported_three_pass_multideck_never_falls_back_to_another_road_surface(catalog, points):
    """Until three-tier support is certified, reject rather than silently select a deck."""
    with pytest.raises(TrackValidationError) as exc:
        convert_custom_track(draft(points, width=11), catalog)
    assert exc.value.code in {"unsupported_intersection", "unsupported_slope",
                              "unsupported_surface_intersection"}


def test_certified_three_tier_crossing_has_three_distinct_decks(catalog):
    # Three independent crossings of (0,0) with separated route visits and
    # long ramps. The fourth crossover near x=73 is also clearance checked.
    coordinates = [
        (-65, -65), (-35, -35), (0, 0), (35, 35), (65, 65),
        (65, -65), (35, -35), (0, 0), (-35, 35), (-65, 65),
        (-75, -15), (-35, 0), (0, 0), (35, 0), (75, 0), (145, -25),
    ]
    converted = convert_custom_track(draft(coordinates, width=10), catalog)
    decks = sorted(sample['y'] for sample in converted.road_samples
                   if abs(sample['x']) < 0.6 and abs(sample['z']) < 0.6)
    # Many samples near the crossing; group by height and require exactly
    # three separated road levels with independently certified geometry.
    heights = []
    for height in decks:
        if not heights or height - heights[-1] > 1:
            heights.append(height)
    assert len(heights) >= 3, heights
    assert heights[1] - heights[0] > 1.8
    assert heights[2] - heights[1] > 1.8
    assert converted.auto_bridge_count >= 2
    env = converted.factory.create()
    try:
        obs, _ = env.reset(seed=412)
        assert np.isfinite(obs).all()
        assert env.runtime_snapshot()['grounded'] is True
    finally:
        env.close()

def test_v4_gap_is_authoritative_empty_road_support_with_hash_bound_geometry(catalog):
    """A certified gap is missing asphalt in Physics, not a cosmetic prop."""
    from src.simulation.geometry.vector import Vec3

    doc = draft([(0, 0), (0, 120)], [prop("gap", 0.5, offset=4.0)])
    converted = convert_custom_track(doc, catalog)
    surfaces = converted.factory.track.surfaces
    assert all(geometry.id for geometry in surfaces)
    assert set(converted.factory.navigation.surface_ids).issubset({s.id for s in surfaces})
    # No supporting polygon beneath the center of the gap.
    assert converted.factory.track.surface_at(Vec3(0., 0., 60.), 0.) is None
    assert converted.factory.track.surface_at(Vec3(0., 0., 55.), 0.) is not None
    assert converted.factory.track.surface_at(Vec3(0., 0., 65.), 0.) is not None
    assert len(surfaces) < len(convert_custom_track(draft([(0, 0), (0, 120)]),catalog).factory.track.surfaces)
    changed = convert_custom_track(draft([(0, 0), (0, 120)], [prop("gap", 0.5, offset=3.0)]), catalog)
    assert changed.track_sha256 != converted.track_sha256


@pytest.mark.parametrize("t,length,rotation",[
    (0.1,4.,0.),(0.9,4.,0.),(0.5,1.,0.),(0.5,12.,0.),(0.5,4.,.1),
])
def test_gap_rejects_unsafe_geometry_and_unregistered_rotation(catalog,t,length,rotation):
    with pytest.raises(TrackValidationError):
        convert_custom_track(draft([(0,0),(0,120)],[prop("gap",t,offset=length,rotation=rotation)]),catalog)

@pytest.mark.parametrize("value", [None, "3", {}, [], True, float("nan")])
def test_gap_length_rejects_nonfinite_or_non_numeric_json_types(catalog, value):
    with pytest.raises(TrackValidationError):
        convert_custom_track(
            draft([(0, 0), (0, 120)], [prop("gap", 0.5, offset=value)]),
            catalog,
        )

def test_custom_booster_volume_includes_physical_vehicle_center(catalog):
    """Boost modifier is a source-bound force, not a decorative unreachable zone."""
    from src.simulation.geometry.vector import Vec3
    from src.simulation.interactions.resolver import resolve_motion_modifiers
    converted = convert_custom_track(
        draft([(0, 0), (0, 86)], [prop("booster", 0.3)]), catalog,
    )
    factory = converted.factory
    pad = factory.track.boost_pads[0]
    chassis = Vec3(pad.volume.center.x, factory.vehicle_parameters.ride_height,
                   pad.volume.center.z)
    assert pad.volume.contains(chassis)
    query = factory.track.query_point(chassis, 1.0, factory.initial_state.track)
    assert query.contact is not None
    assert resolve_motion_modifiers(query).longitudinal_acceleration_bonus == 3.0


def test_t3_versioned_jump_geometry_matches_c1_launch_and_receiving_trough(catalog):
    """The Python world must be physically open and follow versioned heights."""
    import json
    from math import isclose

    from src.runtime.custom_tracks import JUMP_LAUNCH_CONTRACT, _jump_height
    from src.simulation.geometry.vector import Vec3
    from src.simulation.multi_surface_support import _atlas

    doc = json.loads((ROOT / "examples/t3-v4-launch-landing-map.json").read_text(
        encoding="utf-8"))
    converted = convert_custom_track(doc, catalog)
    assert JUMP_LAUNCH_CONTRACT == "custom_jump_launch_receiver_v1"
    assert converted.development_only is True
    assert converted.ppo_compatible is False
    assert len(converted.factory.track.boost_pads) == 7
    surfaces = converted.factory.track.surfaces
    assert _atlas(surfaces, 1e-6) is not None
    # True missing support below the 3.7 m flight path.
    assert converted.factory.track.surface_at(Vec3(0., 0.6, 45.), 0.) is None
    assert converted.factory.track.surface_at(Vec3(0., 0.8, 43.15), 0.01) is not None
    assert converted.factory.track.surface_at(Vec3(0., 0.4, 46.85), 0.01) is not None
    jump = doc["props"][0]
    for z, y in [(0,0), (27.15,0.), (43.15,.8), (45.,.6),
                 (46.85,.4), (48.35,.4), (52.35,.18),
                 (57.35,.18), (73.35,0.), (90.,0.)]:
        assert isclose(_jump_height(z, jump, 90.), y, abs_tol=1e-9), z
    # Source navigation and world use the exact sampled elevations.
    for point in converted.factory.navigation.points:
        assert isclose(point.y, _jump_height(point.z, jump, 90.), abs_tol=1e-8)
    assert max(abs((b.y-a.y)/(b.z-a.z)) for a,b in zip(
        converted.factory.navigation.points,converted.factory.navigation.points[1:])) <= 0.18
    # Boost volumes are actual acceleration sources placed on the raised road.
    for pad in converted.factory.track.boost_pads:
        surface_z = pad.volume.center.z
        expected_y = _jump_height(surface_z, jump, 90.)
        assert pad.volume.center.y == pytest.approx(
            expected_y + converted.factory.vehicle_parameters.ride_height, abs=1e-7)


@pytest.mark.parametrize("invalid", [
    {"offset": 4.0}, {"rotation": 0.1}, {"t": 0.2},
])
def test_jump_v1_rejects_uncertified_dimensions_or_placement(catalog, invalid):
    import json

    doc = json.loads((ROOT / "examples/t3-v4-launch-landing-map.json").read_text(
        encoding="utf-8"))
    doc["props"][0] = {**doc["props"][0], **invalid}
    with pytest.raises(TrackValidationError):
        convert_custom_track(doc,catalog)


def test_jump_v1_rejects_curved_route_narrow_width_and_combined_ramp(catalog):
    import json

    original = json.loads((ROOT / "examples/t3-v4-launch-landing-map.json").read_text(
        encoding="utf-8"))
    for invalid in (
        {**original, "width": 8.0},
        {**original, "points": [{"x":0,"z":0},{"x":0,"z":45},{"x":0,"z":90}]},
        {**original, "props": original["props"] + [
            {"id":"extra-ramp","kind":"ramp","t":0.4,"offset":0.0,"rotation":0.0}]},
    ):
        with pytest.raises(TrackValidationError):
            convert_custom_track(invalid,catalog)
