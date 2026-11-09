from dataclasses import replace
import json
from pathlib import Path

import pytest

from f1rl.local_demo.runtime.catalog import RuntimeCatalog
from f1rl.local_demo.runtime.custom_tracks import convert_custom_track
from f1rl.runtime.simulation import Simulation
from f1rl.sandbox.body_collision import VehicleBodyGeometry
from f1rl.sandbox.geometry.vector import Vec3
from f1rl.sandbox.spatial_contact_dynamics import prepare_spatial_contact_geometry
from f1rl.sandbox.track.patch import PlanarSurfacePatch
from f1rl.sandbox.vehicle import VehicleControl

import f1rl.runtime.simulation as simulation_module


ROOT = Path(__file__).resolve().parents[2]
T05_FIXTURE = Path(__file__).parent / "fixtures" / "t05-original-obstacle-track.json"


def _t05_simulation() -> Simulation:
    catalog = RuntimeCatalog(ROOT)
    converted = convert_custom_track(
        json.loads(T05_FIXTURE.read_text(encoding="utf-8")), catalog,
    )
    factory = converted.factory
    return Simulation(
        factory.track, factory.initial_state, factory.vehicle_parameters,
        factory.simulation_config, physics_config=factory.physics_config,
    )


def test_geometry_context_matches_only_identical_geometry_inputs() -> None:
    surfaces = (
        PlanarSurfacePatch(
            "road", Vec3(0.0, 0.0, 0.0), Vec3(0.0, 0.0, 1.0),
            Vec3(0.0, 1.0, 0.0), 8.0, 20.0,
        ),
    )
    body = VehicleBodyGeometry(1.8, 1.2, 4.0, Vec3(0.0, 0.0, 0.0))
    context = prepare_spatial_contact_geometry(surfaces, 1e-6, body)

    assert context.matches(surfaces, 1e-6, body)
    assert context.atlas is not None
    assert context.patch_index is not None
    assert context.patch_index.source_patches is context.atlas[0]
    cell = next(iter(context.patch_index.cells))
    with pytest.raises(TypeError):
        context.patch_index.cells[cell] = ()
    assert not context.matches(surfaces, 2e-6, body)
    assert not context.matches(surfaces, 1e-6, replace(body, width_m=1.9))
    assert not context.matches(surfaces + (replace(surfaces[0], id="road-copy"),), 1e-6, body)


def test_simulation_reuses_v4_geometry_across_steps_and_rebuilds_after_track_change(
    monkeypatch,
) -> None:
    simulation = _t05_simulation()
    prepare = simulation_module.prepare_spatial_contact_geometry
    built = []

    def record_build(*args, **kwargs):
        context = prepare(*args, **kwargs)
        built.append(context)
        return context

    monkeypatch.setattr(simulation_module, "prepare_spatial_contact_geometry", record_build)

    simulation.step(VehicleControl(0.0, 0.0, 0.0))
    first = built[0]
    simulation.reset()
    simulation.step(VehicleControl(0.0, 0.0, 0.0))
    assert len(built) == 1

    extra_surface = PlanarSurfacePatch(
        "cache-invalidation-probe", Vec3(0.0, 0.0, 10000.0),
        Vec3(0.0, 0.0, 1.0), Vec3(0.0, 1.0, 0.0), 4.0, 4.0,
    )
    simulation._track = replace(
        simulation._track, surfaces=simulation._track.surfaces + (extra_surface,),
    )
    second = simulation._spatial_geometry_context_for_step()

    assert len(built) == 2
    assert second is not first
    assert second.matches(simulation._track.surfaces, first.seam_height_tolerance_m,
                          first.body_geometry)


def test_invalid_atlas_geometry_context_remains_unavailable() -> None:
    surfaces = (
        PlanarSurfacePatch(
            "lower", Vec3(0.0, 0.0, 0.0), Vec3(0.0, 0.0, 1.0),
            Vec3(0.0, 1.0, 0.0), 8.0, 20.0,
        ),
        PlanarSurfacePatch(
            "raised-overlap", Vec3(0.0, 1.0, 0.0), Vec3(0.0, 0.0, 1.0),
            Vec3(0.0, 1.0, 0.0), 8.0, 20.0,
        ),
    )
    body = VehicleBodyGeometry(1.8, 1.2, 4.0, Vec3(0.0, 0.0, 0.0))

    context = prepare_spatial_contact_geometry(surfaces, 1e-6, body)

    assert context.atlas is None
    assert context.patch_index is None
    assert not context.patches_by_id
