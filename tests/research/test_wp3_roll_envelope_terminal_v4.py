"""The roll envelope is a witnessed noncollision terminal, not a crash alias."""
from dataclasses import replace
import pytest
from src.simulation.geometry.vector import Vec3
from src.simulation.geometry.rotation import Quaternion
from src.simulation.vehicle import VehicleState
from src.simulation.center_trajectory import CenterSegment
from src.simulation.roll_envelope_terminal import RollEnvelopeStop,VERSION
from src.rl.roll_envelope_registered import (
    descriptor,validate_registered_roll_envelope_info)
from src.rl.crash_terminal_registered import _sha,RACE_V4,PHYSICS_V4
from src.evaluation.evaluator import _valid_terminal_failure

UPROLL=Vec3(0.,0.,1.)
POS=Vec3(0.,.5,0.)
VEL=Vec3(1.,0.,0.)
W=Vec3(0.,0.,1.)

def pose(angle):
    return VehicleState(POS,VEL,Quaternion.from_axis_angle(UPROLL,angle),W,True)

def stop():
    accepted=pose(.699999999999)
    rejected=pose(.700000000001)
    prefix=(CenterSegment(0.,.001,POS,POS),)
    return RollEnvelopeStop(.00099999995,.001,.7,1,.69,
         .699999999999,.700000000001,accepted,rejected,prefix,object())

def test_noncollision_source_witness_is_distinct_from_physical_crash():
    s=stop()
    assert s.version==VERSION
    assert s.accepted.position==s.rejected.position
    assert s.first_clear_s<s.first_outside_s
    assert s.roll_limit_rad==.7

@pytest.mark.parametrize("change",[
    {"version":"wp3_grounded_wheel_hard_stop_crash_v4_dev"},
    {"first_clear_s":.002},
    {"first_outside_s":.002},
    {"roll_limit_rad":.75},
    {"side":-1},
    {"rejected_roll_rad":.69},
    {"clear_roll_rad":.71},
    {"rejected":pose(.6)},
])
def test_forged_or_out_of_domain_roll_witness_is_rejected(change):
    with pytest.raises(ValueError):
        replace(stop(),**change)

def recorded_info():
    s=stop()
    q=s.accepted.orientation
    pos=(POS.x,POS.y,POS.z)
    cfg={"ground_motion_version":PHYSICS_V4}
    info={
        "physics":{"configuration":cfg,"sha256":_sha(cfg),
                   "transition":{"solvers":("certified_roll_envelope_absorbing_v1_development",)}},
        "race":{"configuration":{"version":RACE_V4},
                "terminal_reason":"tilt_envelope_exit","terminal_failure":True,
                "terminal_time_s":s.first_clear_s,
                "reward_valid_prefix_s":s.first_clear_s,
                "physical_tick_end_time_s":.01,
                "reward_prefix_position":pos,"tick_local_crossing_times_s":()},
        "telemetry":{"vehicle_position":pos,"vehicle_velocity":(1.,0.,0.),
                     "obstacle_hit_ids":()},
        "research_outcome":{"version":"wp7_episode_outcome_v1",
                            "kind":"valid_terminal_failure","valid_sample":True,
                            "reason":"tilt_envelope_exit"},
        "finished":False,
        "roll_envelope_terminal":{
            "version":VERSION,"descriptor_sha256":_sha(descriptor()),
            "first_clear_s":s.first_clear_s,"first_outside_s":s.first_outside_s,
            "roll_limit_rad":s.roll_limit_rad,"side":1,"initial_roll_rad":.69,
            "clear_roll_rad":s.clear_roll_rad,"rejected_roll_rad":s.rejected_roll_rad,
            "accepted_position":pos,"accepted_velocity_mps":(1.,0.,0.),
            "accepted_orientation_wxyz":(q.w,q.x,q.y,q.z),
            "collision":False,"scientific_evidence":False,
            "impulse_n_s":(0.,0.,0.),"angular_impulse_nms":(0.,0.,0.),
            "collision_energy_lost_j":0.,
        },
    }
    return info

def test_registered_info_only_allows_non_collision_envelope_terminal():
    i=recorded_info()
    assert validate_registered_roll_envelope_info(i)
    assert _valid_terminal_failure(i)
    assert descriptor()["scientific_evidence_allowed"] is False

@pytest.mark.parametrize(("section","key","value"),[
    ("roll_envelope_terminal","impulse_n_s",(2.,0.,0.)),
    ("roll_envelope_terminal","collision",True),
    ("roll_envelope_terminal","scientific_evidence",True),
    ("roll_envelope_terminal","clear_roll_rad",.705),
    ("roll_envelope_terminal","first_clear_s",.002),
    ("roll_envelope_terminal","descriptor_sha256","FAKE"),
    ("race","terminal_reason","physical_crash"),
    ("race","reward_valid_prefix_s",.009),
    ("telemetry","obstacle_hit_ids",("road",)),
    ("research_outcome","reason","physical_crash"),
])
def test_untrusted_envelope_info_cannot_mint_valid_terminal(section,key,value):
    i=recorded_info()
    i[section][key]=value
    assert not validate_registered_roll_envelope_info(i)
    assert not _valid_terminal_failure(i)
