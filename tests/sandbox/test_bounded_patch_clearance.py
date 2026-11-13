"""Finite road-patch clearance at the real TRAIN ramp crest."""
from dataclasses import replace
from decimal import Decimal, localcontext
from math import cos, nextafter, pi, tan, ulp
from random import Random
import pytest

from f1rl.sandbox.body_collision import (
    VehicleBodyGeometry,
    obb_separation_at_most,
    obb_separation_lower_bound,
)
from f1rl.sandbox.geometry.rotation import Quaternion
from f1rl.sandbox.geometry.vector import Vec3
from f1rl.sandbox.independent_suspension_dynamics import SuspensionResolutionError
from f1rl.sandbox.multi_surface_support import _atlas, certify_multisurface_support
from f1rl.sandbox.track.patch import PlanarSurfacePatch
from f1rl.sandbox.vehicle import VehicleState
from tests.sandbox.test_road_relative_suspension import ramp_translation


def _reference_obb_separation(first, second):
    with localcontext() as context:
        context.prec = 80
        decimal = Decimal.from_float
        basis = ((1., 0., 0.), (0., 1., 0.), (0., 0., 1.))

        def dot(left, right):
            return sum((a*b for a, b in zip(left, right)), Decimal(0))

        def cross(left, right):
            return (
                left[1]*right[2]-left[2]*right[1],
                left[2]*right[0]-left[0]*right[2],
                left[0]*right[1]-left[1]*right[0],
            )

        def rotated_axes(box):
            q = tuple(decimal(float(v)) for v in (
                box.orientation.w, box.orientation.x,
                box.orientation.y, box.orientation.z,
            ))
            norm = sum((component*component for component in q), Decimal(0)).sqrt()
            w, x, y, z = (component/norm for component in q)
            vector = (x, y, z)
            norm_squared = dot(vector, vector)
            axes = []
            for raw in basis:
                value = tuple(decimal(v) for v in raw)
                projection = dot(vector, value)
                perpendicular = cross(vector, value)
                axes.append(tuple(
                    2*vector[i]*projection
                    + value[i]*(w*w-norm_squared)
                    + 2*w*perpendicular[i]
                    for i in range(3)
                ))
            return tuple(axes)

        first_axes = rotated_axes(first)
        second_axes = rotated_axes(second)
        first_half = tuple(decimal(float(v)) for v in (
            first.half_extents.x, first.half_extents.y, first.half_extents.z,
        ))
        second_half = tuple(decimal(float(v)) for v in (
            second.half_extents.x, second.half_extents.y, second.half_extents.z,
        ))
        offset = tuple(
            decimal(float(b))-decimal(float(a))
            for a, b in zip(
                (first.center.x, first.center.y, first.center.z),
                (second.center.x, second.center.y, second.center.z),
            )
        )
        gaps = []
        for raw_axis in (
            *first_axes,
            *second_axes,
            *(cross(a, b) for a in first_axes for b in second_axes),
        ):
            magnitude_squared = dot(raw_axis, raw_axis)
            if magnitude_squared == 0:
                continue
            magnitude = magnitude_squared.sqrt()
            axis = tuple(v/magnitude for v in raw_axis)
            first_radius = sum((
                extent*abs(dot(axis, unit))
                for extent, unit in zip(first_half, first_axes)
            ), Decimal(0))
            second_radius = sum((
                extent*abs(dot(axis, unit))
                for extent, unit in zip(second_half, second_axes)
            ), Decimal(0))
            gaps.append(abs(dot(offset, axis))-first_radius-second_radius)
        return max(gaps)


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
            candidate_indices = index.candidate_indices(box, threshold)
            candidates = set(candidate_indices)
            assert index.candidate_indices(box, threshold) == candidate_indices
            exact = tuple(
                _reference_obb_separation(box, patch)
                for patch in index.obstacles
            )
            assert all(
                separation > Decimal.from_float(threshold) or patch_index in candidates
                for patch_index, separation in enumerate(exact)
            )
            assert index.has_separation_at_most(box, threshold) is any(
                separation <= Decimal.from_float(threshold) for separation in exact
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


def test_layered_tire_support_ignores_clearance_separated_overhead_deck():
    from f1rl.sandbox.finite_tire_footprint_v4 import sample_finite_wheel_coverage
    from f1rl.sandbox.multi_surface_support import (
        MultiSurfaceSuspensionPolicy, minimum_layer_clearance_m,
    )
    from f1rl.sandbox.ramp_support import WheelContactPoints

    lower = PlanarSurfacePatch(
        "lower", Vec3(0., 0., 0.), Vec3(0., 0., 1.), Vec3(0., 1., 0.), 40., 40.,
    )
    upper = PlanarSurfacePatch(
        "upper", Vec3(0., 5.27, 0.), Vec3(1., 0., 0.), Vec3(0., 1., 0.), 8., 12.,
    )
    body = VehicleBodyGeometry(1.8, 1.2, 4., Vec3(0., 0., 0.))
    clearance = minimum_layer_clearance_m(body)
    atlas = _atlas(
        (lower, upper), 1e-6, layered_clearance_m=clearance,
        continuous_seam_tolerance_m=.01,
    )
    assert atlas is not None
    contact = MultiSurfaceSuspensionPolicy(
        WheelContactPoints(
            Vec3(-.7, .2, 1.4), Vec3(.7, .2, 1.4),
            Vec3(-.7, .2, -1.4), Vec3(.7, .2, -1.4),
        ), .6, 1.5,
    )
    vehicle = VehicleState(
        Vec3(0., .5, 0.), Vec3(0., 0., 0.), Quaternion.identity(),
        Vec3(0., 0., 0.), True,
    )

    with pytest.raises(SuspensionResolutionError, match="finite-tire wheel exceeds unchanged bump stop"):
        sample_finite_wheel_coverage(vehicle, contact, atlas)
    coverage = sample_finite_wheel_coverage(
        vehicle, contact, atlas, layered_clearance_m=clearance,
        blend_surface_support=True,
    )
    assert tuple(item.contact.patch_id for item in coverage) == ("lower",) * 4
    upper_vehicle = replace(vehicle, position=Vec3(0., 5.77, 0.))
    upper_coverage = sample_finite_wheel_coverage(
        upper_vehicle, contact, atlas, layered_clearance_m=clearance,
        blend_surface_support=True,
    )
    assert tuple(item.contact.patch_id for item in upper_coverage) == ("upper",) * 4


def test_tire_support_height_is_continuous_across_tolerated_generated_seam():
    from f1rl.sandbox.finite_tire_footprint_v4 import sample_finite_wheel_coverage
    from f1rl.sandbox.multi_surface_support import MultiSurfaceSuspensionPolicy
    from f1rl.sandbox.ramp_support import WheelContactPoints

    roads = (
        PlanarSurfacePatch(
            "surface-0000", Vec3(0., 0., 0.), Vec3(0., 0., 1.),
            Vec3(0., 1., 0.), 40., 40.,
        ),
        PlanarSurfacePatch(
            "surface-0001", Vec3(0., .005, 0.), Vec3(0., 0., 1.),
            Vec3(0., 1., 0.), 10., 40.,
        ),
    )
    atlas = _atlas(roads, 1e-6, continuous_seam_tolerance_m=.01)
    assert atlas is not None
    contact = MultiSurfaceSuspensionPolicy(
        WheelContactPoints(
            Vec3(-.7, .2, 1.4), Vec3(.7, .2, 1.4),
            Vec3(-.7, .2, -1.4), Vec3(.7, .2, -1.4),
        ), .6, 1.5,
    )

    def front_left_height(x):
        vehicle = VehicleState(
            Vec3(x, .5, 0.), Vec3(0., 0., 0.), Quaternion.identity(),
            Vec3(0., 0., 0.), True,
        )
        return sample_finite_wheel_coverage(
            vehicle, contact, atlas, blend_surface_support=True,
        )[0].contact.ground_height_m

    assert abs(front_left_height(5.6) - front_left_height(5.7)) < .001


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


def test_indexed_sat_avoids_false_contact_at_1e9m_rounded_body_axis():
    from f1rl.sandbox.swept_surface_clearance import _build_bounded_patch_index

    road = PlanarSurfacePatch(
        "road", Vec3(60.0, -0.8884474173192283, 0.5604337803032049),
        Vec3(0.2810253020973806, -0.029913293028186447, 0.9592340561517226),
        Vec3(-0.28737537016871983, 0.9510277254621118, 0.11384929522291205),
        12.0, 8.0,
    )
    atlas = _atlas((road,), 1e-6)
    assert atlas is not None
    body = VehicleBodyGeometry(1.8, 1.2, 4.0, Vec3(0.0, 0.0, 0.0))
    index = _build_bounded_patch_index(atlas, body)
    body_box = body.box_at(
        Vec3(59.82757477732402, -0.31783078013990584, 0.6287433576646507),
        index.obstacles[0].orientation,
    )
    threshold = 1e-9

    exact_gap = _reference_obb_separation(body_box, index.obstacles[0])
    assert exact_gap > Decimal.from_float(threshold)
    assert obb_separation_lower_bound(body_box, index.obstacles[0]) <= threshold
    assert index.candidate_indices(body_box, threshold) == (0,)
    assert index.has_separation_at_most(body_box, threshold) is False


def test_sloped_seam_grazing_queries_match_exhaustive_sat():
    from math import nextafter
    from f1rl.sandbox.swept_surface_clearance import _build_bounded_patch_index

    slope = Quaternion.from_axis_angle(Vec3(1.0, 0.0, 0.0), 0.2)
    forward = slope.rotate(Vec3(0.0, 0.0, 1.0))
    normal = slope.rotate(Vec3(0.0, 1.0, 0.0))
    right = normal.cross(forward)
    roads = tuple(
        PlanarSurfacePatch(f"road-{index:02d}", forward * (2.0 * index),
                           forward, normal, 12.0, 2.0)
        for index in range(32)
    )
    atlas = _atlas(roads, 1e-6)
    assert atlas is not None
    body = VehicleBodyGeometry(1.8, 1.2, 4.0, Vec3(0.0, 0.0, 0.0))
    index = _build_bounded_patch_index(atlas, body)
    yaw = Quaternion.from_axis_angle(Vec3(0.0, 1.0, 0.0), pi / 4.0)
    roll = Quaternion.from_axis_angle(Vec3(0.0, 0.0, 1.0), 0.1)

    for road_index in (0, 15, 31):
        seam = forward * (2.0 * road_index + 1.0)
        poses = (
            (seam + normal * 0.6, slope),
            (seam + right * 6.9 + normal * 0.6, yaw * slope),
            (seam + normal * 3.0, roll * yaw * slope),
        )
        for position, orientation in poses:
            box = body.box_at(position, orientation)
            exact = tuple(_reference_obb_separation(box, patch)
                          for patch in index.obstacles)
            thresholds = {0.0, 1e-9, 0.05, 0.8, 5.0}
            for separation in exact:
                boundary = float(max(Decimal(0), separation))
                thresholds.update((
                    max(0.0, nextafter(boundary, float("-inf"))),
                    boundary,
                    nextafter(boundary, float("inf")),
                ))
            for threshold in thresholds:
                candidates = set(index.candidate_indices(box, threshold))
                assert all(
                    separation > Decimal.from_float(threshold) or patch_index in candidates
                    for patch_index, separation in enumerate(exact)
                )
                assert index.has_separation_at_most(box, threshold) is any(
                    separation <= Decimal.from_float(threshold) for separation in exact
                )


def test_indexed_sat_rejects_float_rounded_gap_above_exact_threshold():
    from f1rl.sandbox.swept_surface_clearance import _build_bounded_patch_index

    road = PlanarSurfacePatch(
        "road", Vec3(3000.0, 0.42033277003266445, -1.6078591264335933),
        Vec3(0.9935999001912907, -0.09714244951059339, 0.05764184975292909),
        Vec3(0.09562974098285151, 0.9950102657430676, 0.02845213006899522),
        7.1877890083808875, 12.569201802526118,
    )
    atlas = _atlas((road,), 1e-6)
    assert atlas is not None
    body = VehicleBodyGeometry(1.8, 1.2, 4.0, Vec3(0.0, 0.0, 0.0))
    index = _build_bounded_patch_index(atlas, body)
    box = body.box_at(
        Vec3(2986.365267907609, -6.399540499386711, 10.97115630004828),
        Quaternion(0.05460234904118441, 0.08992362469288061,
                   0.9835051136761405, 0.14713944601690224),
    )
    exact_gap = _reference_obb_separation(box, index.obstacles[0])

    threshold = float(exact_gap)
    assert Decimal.from_float(threshold) < exact_gap
    assert obb_separation_lower_bound(box, index.obstacles[0]) <= threshold
    assert index.candidate_indices(box, threshold) == (0,)
    assert index.has_separation_at_most(box, threshold) is False


def test_indexed_sat_matches_independent_oracle_for_all_obstacles_and_thresholds():
    from f1rl.sandbox.swept_surface_clearance import _build_bounded_patch_index

    random = Random(0x5A71C0DE)

    def make_patch(patch_id, origin, orientation, width, length):
        return PlanarSurfacePatch(
            patch_id, origin, orientation.rotate(Vec3(0., 0., 1.)),
            orientation.rotate(Vec3(0., 1., 0.)), width, length,
        )

    roads = []
    for row in range(3):
        for column in range(8):
            orientation = (
                Quaternion.from_axis_angle(Vec3(0., 1., 0.), random.uniform(-pi, pi))
                * Quaternion.from_axis_angle(Vec3(1., 0., 0.), random.uniform(-0.2, 0.2))
                * Quaternion.from_axis_angle(Vec3(0., 0., 1.), random.uniform(-0.2, 0.2))
            )
            roads.append(make_patch(
                f"random-{row:02d}-{column:02d}",
                Vec3(-400. + column * 60., random.uniform(-2., 2.), -250. + row * 60.),
                orientation, random.uniform(10., 16.), random.uniform(10., 18.),
            ))
    for boundary in range(8):
        roads.append(make_patch(
            f"cell-boundary-{boundary:02d}",
            Vec3((boundary % 4) * 4., 0., (boundary // 4) * 4.),
            Quaternion.identity(), 12., 12.,
        ))
    for seam in range(12):
        orientation = (
            Quaternion.from_axis_angle(Vec3(0., 1., 0.), random.uniform(-pi, pi))
            * Quaternion.from_axis_angle(Vec3(1., 0., 0.), random.uniform(-0.12, 0.12))
        )
        origin = Vec3(900. + (seam % 6) * 50., random.uniform(-1., 1.),
                      900. + (seam // 6) * 50.)
        first = make_patch(f"seam-{seam:02d}-a", origin, orientation, 12., 10.)
        roads.extend((first, make_patch(
            f"seam-{seam:02d}-b",
            origin + orientation.rotate(Vec3(0., 0., 10.)), orientation, 12., 10.,
        )))

    atlas = _atlas(tuple(roads), 1e-6)
    assert atlas is not None
    body = VehicleBodyGeometry(1.8, 1.2, 4., Vec3(0., 0., 0.))
    index = _build_bounded_patch_index(atlas, body)
    patch_indices = {patch.id: number for number, patch in enumerate(index.source_patches)}
    body_half = body.half_extents
    poses = []
    wanted_gaps = (-2e-9, -1e-9, 0., 0.5e-9, 1e-9, 2e-9, 0.05, 0.8, 5.)
    for pose_number in range(24):
        patch = index.source_patches[pose_number % len(index.source_patches)]
        patch_index = patch_indices[patch.id]
        patch_orientation = Quaternion.from_basis(patch.right, patch.normal, patch.forward)
        body_orientation = (
            patch_orientation
            * Quaternion.from_axis_angle(Vec3(0., 1., 0.), random.uniform(-0.3, 0.3))
            * Quaternion.from_axis_angle(Vec3(0., 0., 1.), random.uniform(-0.12, 0.12))
        )
        body_axes = tuple(body_orientation.rotate(axis) for axis in (
            Vec3(1., 0., 0.), Vec3(0., 1., 0.), Vec3(0., 0., 1.),
        ))
        support = sum(half * abs(axis.dot(patch.normal)) for half, axis in zip(
            (body_half.x, body_half.y, body_half.z), body_axes, strict=True,
        ))
        gap = wanted_gaps[pose_number % len(wanted_gaps)]
        center = patch.origin + patch.normal * (support + 0.5e-9 + gap)
        poses.append((body.box_at(center, body_orientation), patch_index))

    for boundary in range(8):
        patch_index = patch_indices[f"cell-boundary-{boundary:02d}"]
        gap = wanted_gaps[boundary % len(wanted_gaps)]
        poses.append((body.box_at(
            Vec3((boundary % 4) * 4., 0.600000001 + gap, (boundary // 4) * 4.),
            Quaternion.identity(),
        ), patch_index))
    for far in range(8):
        orientation = Quaternion.from_axis_angle(
            Vec3(1., 1., 0.).normalized(), random.uniform(-pi, pi),
        )
        poses.append((body.box_at(
            Vec3(5000. + far * 20., 30. + far, 5000. - far * 15.), orientation,
        ), far % len(index.obstacles)))

    false_negative_candidates = 0
    conservative_candidate_extras = 0
    false_negative_decisions = 0
    false_positive_decisions = 0
    float_sat_false_negatives = 0
    float_sat_false_positives = 0
    shared_sat_false_negatives = 0
    shared_sat_false_positives = 0
    shared_sat_changes_outside_band = 0
    positive_queries = 0
    negative_queries = 0
    comparisons = 0
    threshold_evaluations = 0
    oracle_gaps = [
        tuple(_reference_obb_separation(box, obstacle) for obstacle in index.obstacles)
        for box, _ in poses
    ]
    for pose_number, ((box, target_index), exact_gaps) in enumerate(zip(
        poses, oracle_gaps, strict=True,
    )):
        target_gap = max(Decimal(0), exact_gaps[target_index])
        rounded_target = float(target_gap)
        thresholds = {
            0., 1e-9, 0.05, 0.8, 5.,
            max(0., nextafter(rounded_target, float("-inf"))),
            rounded_target,
            nextafter(rounded_target, float("inf")),
        }
        for threshold in thresholds:
            threshold_evaluations += 1
            exact_threshold = Decimal.from_float(threshold)
            expected_indices = {
                obstacle_index for obstacle_index, gap in enumerate(exact_gaps)
                if gap <= exact_threshold
            }
            candidates = set(index.candidate_indices(box, threshold))
            false_negative_candidates += len(expected_indices - candidates)
            conservative_candidate_extras += len(candidates - expected_indices)
            expected = bool(expected_indices)
            actual = index.has_separation_at_most(box, threshold)
            for exact_gap, obstacle in zip(exact_gaps, index.obstacles, strict=True):
                float_gap = obb_separation_lower_bound(box, obstacle)
                exact_contact = exact_gap <= exact_threshold
                shared_contact = obb_separation_at_most(box, obstacle, threshold)
                shared_sat_false_negatives += int(exact_contact and not shared_contact)
                shared_sat_false_positives += int(not exact_contact and shared_contact)
                float_sat_false_negatives += int(
                    exact_contact and float_gap > threshold
                )
                float_sat_false_positives += int(
                    not exact_contact and float_gap <= threshold
                )
                scale = max(
                    1., threshold,
                    abs(box.center.x), abs(box.center.y), abs(box.center.z),
                    abs(obstacle.center.x), abs(obstacle.center.y), abs(obstacle.center.z),
                    box.half_extents.x, box.half_extents.y, box.half_extents.z,
                    obstacle.half_extents.x, obstacle.half_extents.y,
                    obstacle.half_extents.z,
                )
                changed = (float_gap <= threshold) != shared_contact
                shared_sat_changes_outside_band += int(
                    changed and abs(float_gap - threshold) > 256. * ulp(scale)
                )
            false_negative_decisions += int(expected and not actual)
            false_positive_decisions += int(actual and not expected)
            positive_queries += int(expected)
            negative_queries += int(not expected)
            comparisons += len(index.obstacles)

    assert comparisons == threshold_evaluations * len(index.obstacles)
    assert threshold_evaluations >= len(poses) * 7
    assert positive_queries > 0 and negative_queries > 0
    assert false_negative_candidates == 0
    assert false_negative_decisions == 0
    assert false_positive_decisions == 0
    assert shared_sat_false_negatives == 0
    assert shared_sat_false_positives == 0
    assert shared_sat_changes_outside_band == 0
    assert float_sat_false_negatives > 0
    assert float_sat_false_positives > 0
    assert conservative_candidate_extras > 0


def test_indexed_sat_has_exact_representable_threshold_and_fails_closed_on_invalid_inputs():
    from f1rl.sandbox.swept_surface_clearance import _build_bounded_patch_index

    road = PlanarSurfacePatch(
        "road", Vec3(0., 0., 0.), Vec3(0., 0., 1.), Vec3(0., 1., 0.), 12., 12.,
    )
    atlas = _atlas((road,), 1e-6)
    assert atlas is not None
    body = VehicleBodyGeometry(1., 1., 2., Vec3(0., 0., 0.))
    index = _build_bounded_patch_index(atlas, body)
    exact_box = body.box_at(Vec3(6.75, 0., 0.), Quaternion.identity())
    exact_gap = _reference_obb_separation(exact_box, index.obstacles[0])
    assert exact_gap == Decimal.from_float(0.25)
    assert index.has_separation_at_most(exact_box, 0.25) is True
    assert index.has_separation_at_most(exact_box, nextafter(0.25, 0.)) is False
    assert index.has_separation_at_most(exact_box, nextafter(0.25, 1.)) is True
    integer_threshold_box = body.box_at(Vec3(7.5, 0., 0.), Quaternion.identity())
    assert index.has_separation_at_most(integer_threshold_box, 1) is True

    invalid_box = type(exact_box)(None, exact_box.half_extents, exact_box.orientation)
    assert index.candidate_indices(invalid_box, 0.25) == index.all_indices
    assert index.has_separation_at_most(invalid_box, 0.25) is True
    for threshold in (float("nan"), float("inf"), -1.):
        assert index.candidate_indices(exact_box, threshold) == index.all_indices
        assert index.has_separation_at_most(exact_box, threshold) is True
