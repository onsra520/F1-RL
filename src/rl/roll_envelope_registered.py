"""Registered, source-audited development tilt-domain exit; never a crash."""
from __future__ import annotations
from collections.abc import Mapping
from math import atan2,isclose,isfinite
from .crash_terminal_registered import RACE_V4,PHYSICS_V4,_sha
from ..simulation.roll_envelope_terminal import VERSION

def descriptor():
    return {
        "schema_version":"wp3_tilt_envelope_terminal_v1",
        "source_version":VERSION,
        "race_version":RACE_V4,
        "physics_version":PHYSICS_V4,
        "terminal_reason":"tilt_envelope_exit",
        "classification":"valid_noncollision_model_domain_termination_development",
        "impulse_n_s":(0.,0.,0.),
        "angular_impulse_nms":(0.,0.,0.),
        "collision_energy_lost_j":0.,
        "reward_scoring":"accepted_preboundary_prefix_only",
        "scientific_evidence_allowed":False,
    }

def validate_registered_roll_envelope_info(info:Mapping)->bool:
    try:
        t=info["roll_envelope_terminal"]
        race=info["race"]
        phy=info["physics"]
        outcome=info["research_outcome"]
        telemetry=info["telemetry"]
        if (not all(isinstance(v,Mapping) for v in (t,race,phy,outcome,telemetry))
            or info.get("physical_crash_terminal") is not None
            or t.get("version")!=VERSION
            or t.get("descriptor_sha256")!=_sha(descriptor())
            or race.get("configuration",{}).get("version")!=RACE_V4
            or phy.get("configuration",{}).get("ground_motion_version")!=PHYSICS_V4
            or phy.get("sha256")!=_sha(phy["configuration"])
            or "certified_roll_envelope_absorbing_v1_development"
               not in phy.get("transition",{}).get("solvers",())
            or race.get("terminal_reason")!="tilt_envelope_exit"
            or race.get("terminal_failure") is not True
            or outcome.get("kind")!="valid_terminal_failure"
            or outcome.get("reason")!="tilt_envelope_exit"
            or outcome.get("valid_sample") is not True
            or info.get("finished") is not False
            or t.get("scientific_evidence") is not False
            or t.get("collision") is not False
            or t.get("impulse_n_s")!=(0.,0.,0.)
            or t.get("angular_impulse_nms")!=(0.,0.,0.)
            or t.get("collision_energy_lost_j")!=0.):
            return False
        nums=("first_clear_s","first_outside_s","roll_limit_rad",
              "initial_roll_rad","clear_roll_rad","rejected_roll_rad")
        if not all(type(t.get(k)) in (float,int) and isfinite(t[k]) for k in nums):
            return False
        clear,outside,limit=t["first_clear_s"],t["first_outside_s"],t["roll_limit_rad"]
        side=t.get("side")
        q=t.get("accepted_orientation_wxyz")
        if (type(side) is not int or side not in (-1,1)
            or not isinstance(q,tuple) or len(q)!=4
            or any(type(v) not in (int,float) or not isfinite(v) for v in q)
            or abs(sum(v*v for v in q)-1.)>1e-7
            or not 0.<clear<outside<=race["physical_tick_end_time_s"]
            or outside-clear>1.0001e-10
            or not 0.<limit<.75
            or not abs(t["initial_roll_rad"])<limit
            or not 0.<side*t["clear_roll_rad"]<=limit
            or not side*t["rejected_roll_rad"]>limit
            or not isclose(race["terminal_time_s"],clear,abs_tol=1e-12)
            or not isclose(race["reward_valid_prefix_s"],clear,abs_tol=1e-12)
            or not isclose(t["clear_roll_rad"],
                atan2(2*(q[0]*q[3]+q[1]*q[2]),
                      1-2*(q[1]*q[1]+q[3]*q[3])),abs_tol=1e-8)
            or telemetry.get("vehicle_position")!=race.get("reward_prefix_position")
            or telemetry.get("vehicle_position")!=t.get("accepted_position")
            or tuple(t.get("accepted_velocity_mps",()))!=telemetry.get("vehicle_velocity")
            or telemetry.get("obstacle_hit_ids")!=()
            or not isinstance(race.get("tick_local_crossing_times_s"),tuple)
            or any(not 0.<=v<=clear+1e-10
                   for v in race["tick_local_crossing_times_s"])):
            return False
        return True
    except (TypeError,ValueError,KeyError,IndexError,AttributeError):
        return False
