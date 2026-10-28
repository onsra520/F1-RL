"""WP3 separately VERSIONED crash scoring contract, never scientific signoff."""
from copy import deepcopy
from dataclasses import replace
import numpy as np
import pytest

from f1rl.learning.crash_terminal_registered import (
    RACE_V4,CRASH_V1,descriptor,_sha,validate_registered_crash_terminal_info
)
from f1rl.learning.wp7_race import WP7RaceConfig
from f1rl.learning.evaluation import evaluate_episode
from f1rl.learning.evaluation.evaluator import _step_evidence,_valid_terminal_failure
from f1rl.learning.evaluation.crash_contract import validate_crash_terminal_info
from tests.integration.test_wp3_applied_crash_end_to_end import crash_factory
from tests.learning.evaluation.mocks.policies import ConstantPolicy


@pytest.fixture
def observed():
    original=crash_factory()
    new=replace(original,race_config=WP7RaceConfig(
        version=RACE_V4,hard_violation="terminate_valid_failure"))
    env=new.create()
    try:
        env.reset()
        obs,reward,terminated,truncated,info=env.step(np.zeros(3,dtype=np.float32))
        assert terminated and not truncated
        assert not info["finished"]
        assert not info["physical_crash_terminal"]["scientific_evidence"]
        assert validate_registered_crash_terminal_info(info)
        return new,info,reward
    finally:env.close()


def test_protocol_is_distinct_scoped_and_unsigned(observed):
    factory,info,reward=observed
    assert descriptor()["race_version"]==RACE_V4
    assert descriptor()["owner_scientific_signoff"] is False
    assert descriptor()["independent_physics_reviewer_signature"] is None
    assert descriptor()["scientific_evidence_allowed"] is False
    assert factory.race_config.provenance(factory.track,
        __import__('f1rl.learning.route_navigation',fromlist=['AuthoredRoute']).AuthoredRoute(factory.navigation))["sha256"]==info["race"]["sha256"]
    assert info["physical_crash_terminal"]["version"]==CRASH_V1
    assert info["physical_crash_terminal"]["contract_sha256"]==_sha(descriptor())
    assert info["research_outcome"]["reason"]=="physical_crash"
    assert not validate_crash_terminal_info(info)
    assert _valid_terminal_failure(info)
    _step_evidence(info,executed_steps=1,fixed_dt=factory.simulation_config.fixed_dt)


def test_source_bound_evaluator_stops_at_real_crash_prefix(observed):
    factory,info,first_reward=observed
    samples=[]
    result=evaluate_episode(track_id="wp3-registered-contract-fixture",
        factory=factory,policy=ConstantPolicy(np.zeros(3,dtype=np.float32)),
        step_observer=samples.append)
    assert result.success is False
    assert not result.truncated
    assert result.steps==1
    assert result.collision_count==1
    assert result.episode_return==pytest.approx(first_reward,abs=1e-8)
    assert len(samples)==1
    live=samples[0]["info"]
    assert live["research_outcome"]["reason"]=="physical_crash"
    assert live["race"]["reward_valid_prefix_s"]==pytest.approx(
        live["physical_crash_terminal"]["impact_time_s"])
    assert live["race"]["terminal_time_s"]==pytest.approx(
        live["physical_crash_terminal"]["impact_time_s"])
    assert all(t<=live["race"]["terminal_time_s"]
               for t in live["race"]["tick_local_crossing_times_s"])


@pytest.mark.parametrize(("section","field","replacement"),[
    ("physical_crash_terminal","contract_sha256","0"*64),
    ("physical_crash_terminal","version","wp3_grounded_applied_crash_v4_dev"),
    ("physical_crash_terminal","source_version","forged"),
    ("physical_crash_terminal","contact_kind","bounded_road_body"),
    ("physical_crash_terminal","obstacle_id","road-patch:fake"),
    ("physical_crash_terminal","last_proven_clear_s",.1),
    ("physical_crash_terminal","impact_time_s",.1),
    ("physical_crash_terminal","energy_lost_j",9999.),
    ("physical_crash_terminal","kinetic_after_j",9999.),
    ("physical_crash_terminal","linear_impulse_n_s",(100.,0.,0.)),
    ("physical_crash_terminal","angular_impulse_nms",(0.,0.,0.)),
    ("physical_crash_terminal","preimpact_velocity_mps",(9.,0.,0.)),
    ("physical_crash_terminal","accepted_angular_velocity_radps",(0.,0.,0.)),
    ("physical_crash_terminal","contact_orientation_wxyz",(2.,0.,0.,0.)),
    ("physical_crash_terminal","contact_position_world",(1.,2.,3.)),
    ("physical_crash_terminal","science_review_approved",True),
    ("physical_crash_terminal","scientific_evidence",True),
    ("race","terminal_reason","finish"),
    ("race","reward_valid_prefix_s",.1),
    ("race","terminal_failure",False),
    ("race","crossing_time_origin_s",999.),
    ("physics","transition",{"solvers":("grounded_face_impulse_absorbing_v4_development",),
                              "start_time_s":1.,"end_time_s":1.1}),
    ("race","physical_crash_contract",{}),
    ("race","sha256","0"*64),
    ("race","tick_local_crossing_times_s",(.099,)),
    ("physics","sha256","0"*64),
    ("research_outcome","reason","physical_crash_development"),
    ("research_outcome","valid_sample",False),
])
def test_tampered_impulse_times_or_reviewer_claim_fail_closed(observed,section,field,replacement):
    f,original,_=observed
    candidate=deepcopy(original)
    candidate[section][field]=replacement
    assert not validate_registered_crash_terminal_info(candidate)
    assert not _valid_terminal_failure(candidate)
    with pytest.raises(ValueError,match="unregistered physical crash terminal"):
        _step_evidence(candidate,executed_steps=1,fixed_dt=f.simulation_config.fixed_dt)


def test_wrong_race_version_cannot_be_relabelled_without_new_provenance(observed):
    factory,info,_=observed
    for label in ("wp7_directed_lifecycle_crash_dev_v3","wp7_directed_lifecycle_owner_v2"):
        tampered=deepcopy(info)
        tampered["race"]["configuration"]["version"]=label
        assert not validate_registered_crash_terminal_info(tampered)


def test_legacy_crash_dev_contract_is_still_supported_without_silent_promotion():
    f=crash_factory()
    env=f.create()
    try:
        env.reset()
        *_,info=env.step(np.zeros(3,dtype=np.float32))
        assert info["research_outcome"]["reason"]=="physical_crash_development"
        assert validate_crash_terminal_info(info)
        assert not validate_registered_crash_terminal_info(info)
    finally:env.close()
