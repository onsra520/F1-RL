"""Finite road-patch clearance at the real TRAIN ramp crest."""
from dataclasses import replace
from math import cos, pi, tan
import pytest

from f1rl.sandbox.body_collision import VehicleBodyGeometry, obb_separation_lower_bound
from f1rl.sandbox.geometry.rotation import Quaternion
from f1rl.sandbox.geometry.vector import Vec3
from f1rl.sandbox.independent_suspension_dynamics import SuspensionResolutionError
from f1rl.sandbox.multi_surface_support import _atlas, certify_multisurface_support
from f1rl.sandbox.track.patch import PlanarSurfacePatch
from f1rl.sandbox.vehicle import VehicleState
from tests.sandbox.test_road_relative_suspension import ramp_translation


def test_real_crest_body_is_not_tested_against_upper_plane_behind_its_patch():
    factory, vehicle, dynamic, policy = ramp_translation()
    ramp = next(p for p in factory.track.surfaces if p.id == 'road-incline')
    z = 107.40757983645895
    y = ramp.origin.y + (z-ramp.origin.z)*tan(-dynamic.pitch_x_rad) + .5/cos(dynamic.pitch_x_rad)
    vehicle = replace(vehicle, position=Vec3(0., y, z))
    verdict = certify_multisurface_support(vehicle, vehicle, factory.physics_config.body_geometry,
        policy.contact, factory.track.surfaces, body_clearance_version='bounded_patch_planes_v2')
    assert verdict.valid, verdict


def test_interior_crest_penetration_is_rejected_with_valid_wheel_travel():
    from f1rl.sandbox.track.patch import PlanarSurfacePatch
    from f1rl.sandbox.geometry.rotation import Quaternion
    factory, vehicle, _, policy = ramp_translation()
    normal_left = Vec3(-.2, 1., 0.).normalized()
    normal_right = Vec3(.2, 1., 0.).normalized()
    roads = tuple(PlanarSurfacePatch(name, Vec3(x, -.35, 0.), Vec3(0., 0., 1.), normal,
                    6./normal.y, 20.) for name,x,normal in
                  (('left', -3., normal_left), ('right', 3., normal_right)))
    vehicle = replace(vehicle, position=Vec3(0., .5, 0.), orientation=Quaternion.identity())
    verdict = certify_multisurface_support(vehicle, vehicle, factory.physics_config.body_geometry,
        policy.contact, roads, body_clearance_version='bounded_patch_planes_v2')
    assert verdict.status == 'outside'


@pytest.mark.parametrize('version', [None, 'unregistered'])
def test_unregistered_clearance_versions_are_rejected(version):
    factory, vehicle, _, policy = ramp_translation()
    with pytest.raises(ValueError, match='clearance version'):
        certify_multisurface_support(vehicle, vehicle, factory.physics_config.body_geometry,
            policy.contact, factory.track.surfaces, body_clearance_version=version)


def test_numerically_unsupported_scale_is_not_a_penetration_witness():
    from f1rl.sandbox.track.patch import PlanarSurfacePatch
    from f1rl.sandbox.geometry.rotation import Quaternion
    factory, vehicle, _, policy = ramp_translation()
    vehicle = replace(vehicle, position=Vec3(1e7, .5, 0.), orientation=Quaternion.identity())
    road = PlanarSurfacePatch('large-offset', Vec3(1e7, 0., 0.), Vec3(0., 0., 1.), Vec3(0., 1., 0.), 12., 20.)
    verdict = certify_multisurface_support(vehicle, vehicle, factory.physics_config.body_geometry,
        policy.contact, (road,), body_clearance_version='bounded_patch_planes_v2')
    assert verdict.status == 'unsupported'


def test_bounded_patch_spatial_index_is_conservative_and_prunes_distant_patches():
    from f1rl.sandbox.swept_surface_clearance import _build_bounded_patch_index

    roads = tuple(
        PlanarSurfacePatch(
            f'road-{index:03d}', Vec3(0., 0., index * 2.), Vec3(0., 0., 1.),
            Vec3(0., 1., 0.), 12., 2.,
        )
        for index in range(142)
    )
    atlas = _atlas(roads, 1e-6)
    assert atlas is not None
    body = VehicleBodyGeometry(1.8, 1.2, 4., Vec3(0., 0., 0.))
    index = _build_bounded_patch_index(atlas, body)
    poses = (
        (Vec3(0., 0., 75.), Quaternion.identity()),
        (Vec3(5.8, 0.4, 75.), Quaternion.from_axis_angle(Vec3(0., 1., 0.), pi / 4.)),
        (Vec3(0., 3., 75.), Quaternion.identity()),
        (Vec3(0., 0., 400.), Quaternion.identity()),
        *((Vec3((index % 7 - 3) * 1.5, (index % 4) * 0.45,
                index * 8. - 28.),
           Quaternion.from_axis_angle(Vec3(1., 1., 0.).normalized(), index * pi / 19.))
          for index in range(20)),
    )

    for position, orientation in poses:
        box = body.box_at(position, orientation)
        for threshold in (1e-9, 0.05, 0.8, 5.):
            candidates = set(index.candidate_indices(box, threshold))
            exact = tuple(
                obb_separation_lower_bound(box, patch)
                for patch in index.obstacles
            )
            assert all(
                separation > threshold or patch_index in candidates
                for patch_index, separation in enumerate(exact)
            )
            assert index.has_separation_at_most(box, threshold) is any(
                separation <= threshold for separation in exact
            )

    nearby = body.box_at(Vec3(0., 0., 75.), Quaternion.identity())
    assert len(index.candidate_indices(nearby, 0.1)) < len(roads) // 4
    assert index.candidate_indices(nearby, float('nan')) == index.all_indices
    assert index.candidate_indices_near_xz(float('nan'), 75., 0.1) == index.all_indices
    assert index.candidate_indices_near_xz(0., 75., float('inf')) == index.all_indices
    assert index.has_separation_at_most(nearby, float('nan')) is True


def test_swept_patch_index_preserves_clear_and_contact_verdicts():
    from f1rl.sandbox.swept_surface_clearance import (
        _build_bounded_patch_index,
        certify_swept_chassis_patch_intersection_free,
    )

    roads = tuple(
        PlanarSurfacePatch(
            f'road-{index:03d}', Vec3(0., 0., index * 2.), Vec3(0., 0., 1.),
            Vec3(0., 1., 0.), 12., 2.,
        )
        for index in range(142)
    )
    atlas = _atlas(roads, 1e-6)
    assert atlas is not None
    body = VehicleBodyGeometry(1.8, 1.2, 4., Vec3(0., 0., 0.))
    index = _build_bounded_patch_index(atlas, body)

    def run(y, patch_index):
        state = VehicleState(Vec3(0., y, 75.), Vec3(0., 0., 0.),
                             Quaternion.identity(), Vec3(0., 0., 0.), False)
        return certify_swept_chassis_patch_intersection_free(
            lambda _time: state, 0., 1., body, atlas,
            max_subdivisions=0, patch_index=patch_index,
        )

    assert run(3., index) == run(3., None)
    with pytest.raises(SuspensionResolutionError, match='bounded chassis witnessed physical road contact'):
        run(0., index)
    with pytest.raises(SuspensionResolutionError, match='bounded chassis witnessed physical road contact'):
        run(0., None)


def test_indexed_finite_wheel_coverage_matches_exhaustive_and_prunes_patches():
    from f1rl.sandbox.finite_tire_footprint_v4 import sample_finite_wheel_coverage
    from f1rl.sandbox.swept_surface_clearance import _build_bounded_patch_index

    roads = tuple(
        PlanarSurfacePatch(
            f'road-{index:03d}', Vec3(0., 0., index * 2.), Vec3(0., 0., 1.),
            Vec3(0., 1., 0.), 12., 2.,
        )
        for index in range(142)
    )
    atlas = _atlas(roads, 1e-6)
    assert atlas is not None
    factory, vehicle, _, suspension = ramp_translation()
    body = factory.physics_config.body_geometry
    assert body is not None
    index = _build_bounded_patch_index(atlas, body)

    for z in (3., 75., 190., 300.):
        state = replace(
            vehicle, position=Vec3(0., 0.5, z), velocity=Vec3(0., 0., 0.),
            orientation=Quaternion.identity(),
        )
        exhaustive = sample_finite_wheel_coverage(state, suspension.contact, atlas)
        indexed = sample_finite_wheel_coverage(
            state, suspension.contact, atlas, patch_index=index,
        )
        assert indexed == exhaustive
        for local in suspension.contact.wheels.positions:
            point = state.position + state.orientation.rotate(local)
            assert len(index.candidate_indices_near_xz(point.x, point.z, 0.1)) < len(roads) // 4


def test_precomputed_coverage_edges_match_polygon_reference_exactly():
    from f1rl.sandbox.finite_tire_footprint_v4 import _single_polygon_coverage
    from f1rl.sandbox.swept_surface_clearance import _build_bounded_patch_index

    roads = tuple(
        PlanarSurfacePatch(
            f'road-{index:03d}', Vec3(0., 0., index * 2.), Vec3(0., 0., 1.),
            Vec3(0., 1., 0.), 12., 2.,
        )
        for index in range(142)
    )
    atlas = _atlas(roads, 1e-6)
    assert atlas is not None
    body = VehicleBodyGeometry(1.8, 1.2, 4., Vec3(0., 0., 0.))
    index = _build_bounded_patch_index(atlas, body)

    for patch_index in (0, 70, 141):
        polygon = atlas[1][patch_index]
        xs, zs = zip(*polygon, strict=True)
        samples = (
            (sum(xs) / len(xs), sum(zs) / len(zs)),
            (min(xs) - 0.05, sum(zs) / len(zs)),
            (max(xs) + 0.05, sum(zs) / len(zs)),
            (sum(xs) / len(xs), min(zs) - 0.05),
            (sum(xs) / len(xs), max(zs) + 0.05),
        )
        for x, z in samples:
            expected = _single_polygon_coverage(x, z, polygon, 0.1)
            actual = _single_polygon_coverage(
                x, z, polygon, 0.1,
                edge_coefficients=index.coverage_edges[patch_index],
            )
            assert actual == expected


def test_tire_cell_candidate_cache_preserves_exhaustive_sorted_order():
    from f1rl.sandbox.swept_surface_clearance import _build_bounded_patch_index

    roads = tuple(
        PlanarSurfacePatch(
            f'road-{index:03d}', Vec3(0., 0., index * 2.), Vec3(0., 0., 1.),
            Vec3(0., 1., 0.), 12., 2.,
        )
        for index in range(142)
    )
    atlas = _atlas(roads, 1e-6)
    assert atlas is not None
    body = VehicleBodyGeometry(1.8, 1.2, 4., Vec3(0., 0., 0.))
    index = _build_bounded_patch_index(atlas, body)

    for cell, patch_indices in index.footprint_cells.items():
        x = (cell[0] + 0.5) * index.footprint_cell_size_m
        z = (cell[1] + 0.5) * index.footprint_cell_size_m
        expected = tuple(sorted(set(index.always_footprint_indices).union(patch_indices)))
        assert index.footprint_query_candidates[cell] == expected
        assert index.candidate_indices_near_xz(x, z, 0.1) == expected

    assert index.footprint_query_candidates.get((2500, 2500)) is None
    assert index.candidate_indices_near_xz(10000., 10000., 0.1) == tuple(
        index.always_footprint_indices
    )
