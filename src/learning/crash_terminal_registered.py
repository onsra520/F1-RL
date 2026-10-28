"""Versioned source-bound WP3 crash TERMINAL evidence contract.

ENGINEERING registration of fields and evaluator behavior is NOT independent
physics-domain validation or scientific release permission. A real simulator
PhysicsConfig.validate_transition must have audited the applied rigid impulse
before F1RLEnv emits these fields. No direct metadata can itself mint an Eval
episode; the evaluator verifies a bound provenance, timing, momentum, energy
and approved source list and still requires future owner scientific admission.
"""
from __future__ import annotations
from collections.abc import Mapping
from hashlib import sha256
from math import isclose,isfinite
import json

from ..sandbox.geometry.rotation import Quaternion
from ..sandbox.geometry.vector import Vec3

RACE_V4="wp7_directed_lifecycle_crash_contract_v4"
CRASH_V1="wp3_physical_crash_terminal_v1"
PHYSICS_V4="spatial_road_edge_partial_lifecycle_v4"
SOURCES={
  "wp3_grounded_applied_crash_v4_dev":
      ("obstacle_face","grounded_face_impulse_absorbing_v4_development"),
  "wp3_grounded_applied_edge_crash_v4_dev":
      ("obstacle_edge","grounded_edge_impulse_absorbing_v4_development"),
  "wp3_grounded_applied_road_crash_v4_dev":
      ("bounded_road_body","grounded_road_obb_impulse_absorbing_v4_development"),
  "wp3_airborne_applied_road_crash_v4_dev":
      ("bounded_road_body","airborne_road_obb_impulse_absorbing_v4_development"),
  "wp3_grounded_wheel_hard_stop_crash_v4_dev":
      ("wheel_hard_stop","grounded_wheel_hard_stop_absorbing_v4_development"),
  "wp3_grounded_wheel_edge_stop_crash_v4_dev":
      ("wheel_edge_hard_stop","grounded_wheel_edge_hard_stop_absorbing_v4_development"),
}


def descriptor()->dict:
    """Stable digest-bound engineering evaluator contract, not a signature."""
    return {
      "schema_version":CRASH_V1,
      "race_version":RACE_V4,
      "required_physics_version":PHYSICS_V4,
      "terminal_kind":"valid_terminal_failure",
      "terminal_reason":"physical_crash",
      "terminal_is_finish":False,
      "terminal_is_truncation":False,
      "failure_scoring":"preimpact_reward_and_event_prefix_only",
      "time_origin":"tick_local_impact_s_absolute_transition_start_s",
      "impulse":"world_linear_and_angular_applied_and_audited",
      "energy":"nonincreasing_rigid_kinetic_energy",
      "source_versions":sorted(SOURCES),
      "owner_scientific_signoff":False,
      "independent_physics_reviewer_signature":None,
      "scientific_evidence_allowed":False,
      "release_scope":"ENGINEERING_VALID_FAILED_EPISODE_NOT_SCIENTIFIC_EVAL",
    }


def _sha(document:object)->str:
    return sha256(json.dumps(document,sort_keys=True,separators=(",",":"),
                       allow_nan=False).encode("utf8")).hexdigest()


def _num(v):
    return type(v) in (int,float) and isfinite(v)


def _vector(value):
    if not isinstance(value,tuple) or len(value)!=3 or not all(_num(x) for x in value):
        raise ValueError("untrusted finite Vec3")
    return Vec3(*value)


def _orientation(value):
    if not isinstance(value,tuple) or len(value)!=4 or not all(_num(x) for x in value):
        raise ValueError("untrusted finite quaternion")
    q=Quaternion(*value)
    if abs(sum(x*x for x in value)-1.)>1e-7:
        raise ValueError("unregistered non-unit quaternion")
    return q


def _angular_momentum(q,ang,inertias):
    back=Quaternion(q.w,-q.x,-q.y,-q.z).rotate(ang)
    return q.rotate(Vec3(back.x*inertias.x,back.y*inertias.y,back.z*inertias.z))


def validate_registered_crash_terminal_info(info:Mapping)->bool:
    try:
        if not isinstance(info,Mapping):
            return False
        crash=info["physical_crash_terminal"]
        race=info["race"]
        phy=info["physics"]
        telemetry=info["telemetry"]
        outcome=info["research_outcome"]
        if not all(isinstance(x,Mapping) for x in (crash,race,phy,telemetry,outcome)):
            return False
        conf=race["configuration"]
        config=phy["configuration"]
        trans=phy["transition"]
        if not all(isinstance(x,Mapping) for x in (conf,config,trans)):
            return False
        version=crash.get("source_version")
        if (conf.get("version")!=RACE_V4
            or race.get("physical_crash_contract")!=descriptor()
            or race.get("sha256")!=_sha({k:race[k] for k in
                 ("configuration","course","outcome_contract","physical_crash_contract")})
            or crash.get("version")!=CRASH_V1
            or crash.get("contract_sha256")!=_sha(descriptor())
            or version not in SOURCES
            or crash.get("contact_kind")!=SOURCES[version][0]
            or SOURCES[version][1] not in trans.get("solvers",())
            or not isinstance(crash.get("obstacle_id"),str)
            or not crash["obstacle_id"].strip()
            or crash["obstacle_id"].startswith("road-patch:")!=(crash["contact_kind"]=="bounded_road_body")
            or crash.get("scientific_evidence") is not False
            or crash.get("science_review_approved") is not False
            or config.get("ground_motion_version")!=PHYSICS_V4
            or phy.get("sha256")!=_sha(config)
            or race.get("terminal_reason")!="physical_crash"
            or race.get("terminal_failure") is not True
            or info.get("finished") is not False
            or outcome.get("version")!="wp7_episode_outcome_v1"
            or outcome.get("kind")!="valid_terminal_failure"
            or outcome.get("reason")!="physical_crash"
            or outcome.get("valid_sample") is not True
            or race.get("completed_checkpoint_ids") is None
            or telemetry.get("obstacle_hit_ids")!=(crash["obstacle_id"],)
        ):
            return False
        lo=crash["last_proven_clear_s"]
        hi=crash["impact_time_s"]
        first_overlap=crash["first_overlap_s"]
        prefix=race["reward_valid_prefix_s"]
        terminal=race["terminal_time_s"]
        fixed_dt=race["physical_tick_end_time_s"]
        origin=race["crossing_time_origin_s"]
        times=race["tick_local_crossing_times_s"]
        if (not all(_num(t) for t in (lo,hi,first_overlap,prefix,terminal,fixed_dt,origin))
            or not 0.<=lo<hi<=fixed_dt or fixed_dt<=0.
            or not isclose(first_overlap,hi,abs_tol=1e-10)
            or not isclose(prefix,hi,abs_tol=1e-9)
            or not isclose(terminal,hi,abs_tol=1e-9)
            or origin<0. or not isclose(origin+fixed_dt,info["elapsed_time"],abs_tol=1e-9)
            or not _num(trans.get("start_time_s"))
            or not _num(trans.get("end_time_s"))
            or not isclose(origin,trans["start_time_s"],abs_tol=1e-9)
            or not isclose(origin+fixed_dt,trans["end_time_s"],abs_tol=1e-9)
            or not isinstance(times,tuple)
            or any(not _num(t) or not 0.<=t<=hi+1e-10 for t in times)
            or len(times)!=len(race["directed_crossings"])
            # Progress is frozen at the event, not at the original tick end.
            or not _num(info["elapsed_time"])
        ):
            return False
        pre=_vector(crash["preimpact_velocity_mps"])
        post=_vector(crash["accepted_velocity_mps"])
        pre_w=_vector(crash["preimpact_angular_velocity_radps"])
        post_w=_vector(crash["accepted_angular_velocity_radps"])
        normal_impulse=_vector(crash["linear_impulse_n_s"])
        angular_impulse=_vector(crash["angular_impulse_nms"])
        q=_orientation(crash["contact_orientation_wxyz"])
        conf_mass=config["independent_suspension_policy"]
        if not isinstance(conf_mass,Mapping):
            return False
        mass=conf_mass["mass_kg"]
        inertia=_vector((
           conf_mass["pitch_inertia_kg_m2"],
           config["spatial_contact_policy"]["yaw_inertia_kg_m2"],
           conf_mass["roll_inertia_kg_m2"]
        ))
        if not _num(mass) or mass<=0. or min(inertia.x,inertia.y,inertia.z)<=0.:
            return False
        before=crash["kinetic_before_j"]
        after=crash["kinetic_after_j"]
        lost=crash["energy_lost_j"]
        if (not all(_num(x) for x in (before,after,lost))
            or before<=0. or after<0. or after>=before or lost<=0.
            or not isclose(before-after,lost,rel_tol=1e-9,abs_tol=1e-7)
            or (normal_impulse-(post-pre)*mass).magnitude>1e-7
            or (angular_impulse-(
                _angular_momentum(q,post_w,inertia)-
                _angular_momentum(q,pre_w,inertia))).magnitude>1e-7
            or telemetry.get("vehicle_velocity")!=crash["accepted_velocity_mps"]
            or tuple(crash["contact_position_world"])!=race["reward_prefix_position"]
            or not all(_num(x) for x in crash["contact_position_world"])
            or len(crash["contact_position_world"])!=3):
            return False
        # The original authored flight/ground physics source also audited
        # work/energy, conservation and strict event prefix before this check.
        # No direct payload replaces that runtime proof.
        return True
    except (TypeError,ValueError,KeyError,IndexError,AttributeError,OverflowError,ZeroDivisionError):
        return False
