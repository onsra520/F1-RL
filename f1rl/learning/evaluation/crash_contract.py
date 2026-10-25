"""Development-only WP3 physical-crash evaluator evidence validation.

This validates live simulator-sourced crash fields, transition source and
temporal reward/gate prefix. Does NOT authorize registered 5,760 Eval.
"""
from __future__ import annotations
from collections.abc import Mapping
from math import isclose,isfinite
from hashlib import sha256
import json

def _numeric(x):
    return type(x) in (int,float) and isfinite(x)

def _triple(value):
    return isinstance(value,tuple) and len(value)==3 and all(_numeric(v) for v in value)

def validate_crash_terminal_info(info:Mapping)->bool:
    try:
        crash=info["physical_crash_terminal"]
        race=info["race"]
        phy=info["physics"]
        trans=phy["transition"]
        conf=race["configuration"]
        if (not all(isinstance(x,Mapping) for x in (crash,race,phy,trans,conf))
            or conf["version"]!="wp7_directed_lifecycle_crash_dev_v3"
            or race.get("terminal_reason")!="physical_crash"
            or race.get("terminal_failure") is not True
            or not ((crash.get("version")=="wp3_grounded_applied_crash_v4_dev"
                     and not crash.get('obstacle_id','').startswith('road-patch:')
                     and "grounded_face_impulse_absorbing_v4_development" in trans["solvers"])
                    or (crash.get("version")=="wp3_grounded_applied_edge_crash_v4_dev"
                     and not crash.get('obstacle_id','').startswith('road-patch:')
                     and "grounded_edge_impulse_absorbing_v4_development" in trans["solvers"])
                    or (crash.get("version")=="wp3_grounded_applied_road_crash_v4_dev"
                     and crash.get('obstacle_id','').startswith('road-patch:')
                     and "grounded_road_obb_impulse_absorbing_v4_development" in trans["solvers"])
                    or (crash.get("version")=="wp3_grounded_wheel_hard_stop_crash_v4_dev"
                     and crash.get('obstacle_id','').startswith('wheel-hard-stop:')
                     and "grounded_wheel_hard_stop_absorbing_v4_development" in trans["solvers"]))
            or crash.get("scientific_evidence") is not False
            or phy["configuration"]["ground_motion_version"]!="spatial_road_edge_partial_lifecycle_v4"
            or phy.get('sha256') != sha256(json.dumps(phy['configuration'],
                sort_keys=True,separators=(',',':'),allow_nan=False).encode('utf-8')).hexdigest()
            or race.get('sha256') != sha256(json.dumps({key:race[key] for key in (
                'configuration','course','outcome_contract','development_crash_contract')},
                sort_keys=True,separators=(',',':'),allow_nan=False).encode('utf-8')).hexdigest()
            or not isinstance(crash["obstacle_id"],str) or not crash["obstacle_id"].strip()):
            return False
        lo=crash["last_proven_clear_s"]
        hi=crash["impact_time_s"]
        overlap=crash["first_overlap_s"]
        prefix=race["reward_valid_prefix_s"]
        failure=race["terminal_time_s"]
        dt=race["physical_tick_end_time_s"]
        before,after,lost=(crash[x] for x in (
            "kinetic_before_j","kinetic_after_j","energy_lost_j"))
        if (not all(_numeric(x) for x in (lo,hi,overlap,prefix,failure,dt,before,after,lost))
            or not 0.<=lo<hi<=dt
            or not isclose(overlap,hi,abs_tol=1e-10)
            or not isclose(prefix,hi,abs_tol=1e-9)
            or not isclose(failure,hi,abs_tol=1e-9)
            or not 0.<=after<=before
            or lost<=0. or not isclose(before-after,lost,rel_tol=1e-9,abs_tol=1e-7)
            or not all(_triple(crash[x]) for x in (
                "linear_impulse_n_s","angular_impulse_nms",
                "accepted_velocity_mps","accepted_angular_velocity_radps"))):
            return False
        info_telemetry=info["telemetry"]
        if (not isinstance(info_telemetry,Mapping)
            or info_telemetry.get("vehicle_velocity")!=crash["accepted_velocity_mps"]
            or not _triple(race["reward_prefix_position"])):
            return False
        # No directed checkpoint can occur after the owner-scored prefix.
        times=race["tick_local_crossing_times_s"]
        return (isinstance(times,tuple)
                and all(_numeric(t) and 0.<=t<=hi+1e-10 for t in times))
    except (TypeError,ValueError,KeyError,IndexError):
        return False
