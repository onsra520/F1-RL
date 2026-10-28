"""Finite road-patch clearance at the real TRAIN ramp crest."""
from dataclasses import replace
from math import cos, tan
import pytest

from src.simulation.geometry.vector import Vec3
from src.simulation.multi_surface_support import certify_multisurface_support
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
    from src.simulation.track.patch import PlanarSurfacePatch
    from src.simulation.geometry.rotation import Quaternion
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
    from src.simulation.track.patch import PlanarSurfacePatch
    from src.simulation.geometry.rotation import Quaternion
    factory, vehicle, _, policy = ramp_translation()
    vehicle = replace(vehicle, position=Vec3(1e7, .5, 0.), orientation=Quaternion.identity())
    road = PlanarSurfacePatch('large-offset', Vec3(1e7, 0., 0.), Vec3(0., 0., 1.), Vec3(0., 1., 0.), 12., 20.)
    verdict = certify_multisurface_support(vehicle, vehicle, factory.physics_config.body_geometry,
        policy.contact, (road,), body_clearance_version='bounded_patch_planes_v2')
    assert verdict.status == 'unsupported'
