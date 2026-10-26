"""Physics v4: crossing roll envelope is signed full-quaternion motion, not crash.

These adversarial tests make it impossible to label the tilt bound a fictitious
rigid ground collision or accept forged timing, pitch-singular or yaw crossings.
"""
from dataclasses import replace
import pytest

from f1rl.sandbox.geometry.rotation import Quaternion
from f1rl.sandbox.geometry.vector import Vec3
from f1rl.sandbox.vehicle import VehicleState
from f1rl.sandbox.full_quaternion_roll_transition_v4 import (
    FullQuaternionRollTransition,
    certify_full_quaternion_roll_transition,
    verify_full_quaternion_roll_transition,
)

Z=Vec3(0.,0.,1.)
UP=Vec3(0.,1.,0.)
ZERO=Vec3(0.,0.,0.)
DURATION=.00625

def pose(roll, yaw=0.):
    rotation=(Quaternion.from_axis_angle(UP,yaw)
              * Quaternion.from_axis_angle(Z,roll))
    return VehicleState(Vec3(0.,.8,0.),ZERO,rotation,Vec3(0.,0.,2.),True)

@pytest.mark.parametrize('sign',[1,-1])
def test_unique_monotone_roll_crossing_has_source_bounded_first_time(sign):
    first,last=pose(sign*.69),pose(sign*.71)
    c=certify_full_quaternion_roll_transition(
        first,last,bound_rad=.7,duration_s=DURATION)
    assert isinstance(c,FullQuaternionRollTransition)
    assert c.sign==sign
    assert 0.<c.first_clear_s<c.first_crossing_s<DURATION
    assert c.first_crossing_s-c.first_clear_s<=1e-10
    assert c.first_crossing_s==pytest.approx(DURATION/2.,abs=1e-8)
    assert verify_full_quaternion_roll_transition(
        first,last,c,bound_rad=.7,duration_s=DURATION)
    # No invented impact, dissipated kinetic energy or reward.
    assert not hasattr(c,'impulse_n_s')
    assert not hasattr(c,'energy_lost_j')
    assert not hasattr(c,'postimpact')

def test_no_crossing_produces_no_event():
    a,b=pose(.2),pose(.3)
    assert certify_full_quaternion_roll_transition(
        a,b,bound_rad=.7,duration_s=DURATION) is None
    assert verify_full_quaternion_roll_transition(
        a,b,None,bound_rad=.7,duration_s=DURATION)
    assert not verify_full_quaternion_roll_transition(
        a,b,FullQuaternionRollTransition(.001,.00100000005,.7,1,.2,.3),
        bound_rad=.7,duration_s=DURATION)

@pytest.mark.parametrize('field,value',[
    ('first_clear_s',.001),
    ('first_crossing_s',.005),
    ('boundary_rad',.71),
    ('sign',-1),
    ('start_roll_rad',.68),
    ('end_roll_rad',.72),
    ('schema_version','forged'),
])
def test_forged_roll_source_and_time_rejected(field,value):
    first,last=pose(.69),pose(.71)
    c=certify_full_quaternion_roll_transition(first,last,bound_rad=.7,duration_s=DURATION)
    bad=replace(c,**{field:value})
    assert not verify_full_quaternion_roll_transition(
        first,last,bad,bound_rad=.7,duration_s=DURATION)

def test_yaw_only_is_never_rollover():
    assert certify_full_quaternion_roll_transition(
        pose(.2,yaw=0.),pose(.2,yaw=.03),
        bound_rad=.7,duration_s=DURATION) is None

@pytest.mark.parametrize('case',[
    (pose(-.69),pose(.71)),
    (pose(.71),pose(.69)),
    (pose(.68),pose(.69)),
])
def test_not_first_monotone_envelope_crossing(case):
    first,last=case
    if first.orientation==pose(-.69).orientation:
        with pytest.raises(ValueError):
            certify_full_quaternion_roll_transition(
                first,last,bound_rad=.7,duration_s=DURATION)
    else:
        assert certify_full_quaternion_roll_transition(
            first,last,bound_rad=.7,duration_s=DURATION) is None

@pytest.mark.parametrize('bad',[-1.,0.,.75,float('nan')])
def test_policy_not_weakened_for_bad_tilt_bounds(bad):
    with pytest.raises(ValueError):
        certify_full_quaternion_roll_transition(
            pose(.69),pose(.71),bound_rad=bad,duration_s=DURATION)
