"""First-event timing must meet the registered bound, not just a sign change."""
from math import sqrt

import pytest

from f1rl.sandbox.angular_landing_impulse import AngularLandingImpulsePolicy, ImpulseResolutionError
from f1rl.sandbox.ballistic_flight import _flight_pose
from f1rl.sandbox.multi_surface_support import _atlas
from f1rl.sandbox.rotating_touchdown import first_rotating_wheel_touchdown
from tests.sandbox.test_manifold_flight import CONTACT, ROAD, source


def event(policy):
    return first_rotating_wheel_touchdown(lambda t: _flight_pose(source(), t, 9.81),
        CONTACT.wheels, _atlas((ROAD,), CONTACT.seam_height_tolerance_m), .35, 9.81,
        policy, certify_first_root=True)


def test_exhausted_bisection_budget_cannot_claim_requested_time_accuracy():
    with pytest.raises(ImpulseResolutionError, match="event-time tolerance"):
        event(AngularLandingImpulsePolicy(event_bracket_segments=8, event_bisections=8))


def test_certified_first_event_matches_independent_quadratic_root():
    found = event(AngularLandingImpulsePolicy())
    assert found.time_s == pytest.approx((sqrt(1.+2*9.81*.55)-1)/9.81, abs=1e-8)
