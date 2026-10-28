"""Analytical coplanar references; no empirical validation claim."""
from dataclasses import replace
from math import sqrt

import pytest

from src.sandbox.ballistic_flight import advance_ballistic_flight, FlightPhysicsError
from src.sandbox.geometry.vector import Vec3
from tests.sandbox.test_manifold_flight import BODY, PHYSICS, FLIGHT, MANIFOLD, ROAD, source


def run(patches, restitution=0.):
    return advance_ballistic_flight(
        source(), replace(FLIGHT, manifold=replace(
            MANIFOLD, coplanar_flat_contacts=True, restitution=restitution,
        )), BODY, patches, (), dt=.35, gravity_mps2=9.81,
        landing_dynamics=PHYSICS,
    )


@pytest.mark.parametrize("restitution", [0., .5])
def test_adjacent_coplanar_contact_matches_quadratic_time_and_impulse(restitution):
    patches = (
        replace(ROAD, id="left", origin=Vec3(-5., 0., 0.), width=10.),
        replace(ROAD, id="right", origin=Vec3(5., 0., 0.), width=10.),
    )
    out = run(patches, restitution)
    audit = out.manifold_audit
    speed = sqrt(1. + 2*9.81*.55)
    assert out.transitions[0].time_s == pytest.approx((speed-1)/9.81, abs=1e-8)
    assert len(audit.contacts) == 4
    assert {c.patch_id for c in audit.contacts} == {"left", "right"}
    assert audit.total_linear_impulse_ns.y == pytest.approx(1200*(1+restitution)*speed, abs=.001)
    assert audit.post_kinetic_energy_j == pytest.approx(audit.pre_kinetic_energy_j*restitution**2, abs=1e-4)
    assert out.vehicle.grounded == (restitution == 0.)


def test_overlapping_coplanar_seam_does_not_duplicate_wheel_impulses():
    out = run((ROAD, replace(ROAD, id="duplicate")))
    assert len(out.manifold_audit.contacts) == 4
    assert {c.wheel_index for c in out.manifold_audit.contacts} == set(range(4))


def test_coplanar_opt_in_rejects_different_plane_instead_of_snapping():
    with pytest.raises(FlightPhysicsError, match="coplanar|stacked decks"):
        run((ROAD, replace(ROAD, id="raised", origin=Vec3(0., .01, 0.))))


@pytest.mark.parametrize("value", [1, "true", None])
def test_coplanar_registration_is_strict(value):
    with pytest.raises(ValueError, match="must be bool"):
        replace(MANIFOLD, coplanar_flat_contacts=value)
