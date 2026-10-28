import pytest

from src.simulation.multi_surface_support import certify_multisurface_support
from tests.research.test_wp3_physics_domain_regression import _factory, _pose


def _certify(state, *, mode="full_footprint_v1"):
    factory = _factory("right-turn-v1")
    return certify_multisurface_support(
        state,
        state,
        factory.physics_config.body_geometry,
        factory.physics_config.independent_suspension_policy.contact,
        factory.track.surfaces,
        body_clearance_version="bounded_patch_planes_v2",
        contact_domain="unilateral_free_extension_v1",
        body_coverage_domain=mode,
    )


def test_overhang_requires_explicit_opt_in():
    state = _pose(0.49965, 0.50083, 67.93203, 0.10)
    assert _certify(state).status == "outside"
    assert _certify(state, mode="wheel_supported_overhang_v1").valid


def test_supported_overhang_is_certified_through_continuous_yaw_sweep():
    factory = _factory("right-turn-v1")
    previous = _pose(0.49965, 0.50083, 67.93203, 0.08)
    candidate = _pose(0.49965, 0.50083, 67.93203, 0.10)
    verdict = certify_multisurface_support(
        previous,
        candidate,
        factory.physics_config.body_geometry,
        factory.physics_config.independent_suspension_policy.contact,
        factory.track.surfaces,
        body_clearance_version="bounded_patch_planes_v2",
        contact_domain="unilateral_free_extension_v1",
        body_coverage_domain="wheel_supported_overhang_v1",
    )
    assert verdict.valid


def test_overhang_does_not_admit_missing_wheel_support():
    state = _pose(100.0, 0.5, 67.93203, 0.10)
    verdict = _certify(state, mode="wheel_supported_overhang_v1")
    assert not verdict.valid


@pytest.mark.parametrize("mode", ["", "anything", "unbounded"])
def test_overhang_rejects_unregistered_domain(mode):
    with pytest.raises(ValueError, match="body coverage domain"):
        _certify(_pose(0.49965, 0.50083, 67.93203, 0.10), mode=mode)


def test_overhang_rejects_incompatible_contact_domain():
    factory = _factory("right-turn-v1")
    with pytest.raises(ValueError, match="unilateral contact and bounded clearance"):
        certify_multisurface_support(
            _pose(0.49965, 0.50083, 67.93203, 0.10),
            _pose(0.49965, 0.50083, 67.93203, 0.10),
            factory.physics_config.body_geometry,
            factory.physics_config.independent_suspension_policy.contact,
            factory.track.surfaces,
            body_clearance_version="bounded_patch_planes_v2",
            contact_domain="four_supported_v1",
            body_coverage_domain="wheel_supported_overhang_v1",
        )
