"""Owner-approved valid failure and unchanged reward terms on valid tick prefix."""
from dataclasses import replace
import numpy as np
import pytest
from f1rl.learning.wp7_race import WP7RaceConfig
from f1rl.learning.rewards import RewardComposer,TimePenalty
from tests.integration.test_wp7_race_contract import sideways_factory


def owner_factory():
    return replace(sideways_factory(.5),race_config=WP7RaceConfig(
        version='wp7_directed_grounded_owner_v2',hard_violation='terminate_valid_failure'),
        reward_composer=RewardComposer((TimePenalty(2.),)))


def test_hard_exit_is_valid_terminal_failure_with_only_pre_exit_registered_reward():
    env=owner_factory().create()
    try:
        env.reset()
        first=env.step(np.zeros(2,dtype=np.float32))
        assert first[1]==pytest.approx(-1.) and not first[2]
        observation,reward,terminated,truncated,info=env.step(np.zeros(2,dtype=np.float32))
        assert env.observation_space.contains(observation)
        assert terminated and not truncated
        assert reward==pytest.approx(-.6)
        assert info['reward_components']=={'time':pytest.approx(-.6)}
        assert info['research_outcome']['kind']=='valid_terminal_failure'
        assert info['research_outcome']['valid_sample']
        assert info['race']['terminal_time_s']==pytest.approx(.3)
        assert info['race']['reward_valid_prefix_s']==pytest.approx(.3)
        with pytest.raises(RuntimeError,match='reset'):
            env.step(np.zeros(2,dtype=np.float32))
        assert env.requires_scientific_admission
    finally:
        env.close()


def diagonal_factory(finish):
    from math import pi
    from f1rl.sandbox.geometry.vector import Vec3
    from f1rl.sandbox.geometry.rotation import Quaternion
    from tests.integration.test_wp7_race_contract import race_factory
    base=race_factory(checkpoints=(),finish=finish,speed=0.,max_steps=1)
    vehicle=replace(base.initial_state.vehicle,velocity=Vec3(5.,0.,5.),
                    orientation=Quaternion.from_axis_angle(Vec3(0.,1.,0.),pi/4.))
    return replace(base,track=replace(base.track,surfaces=(replace(base.track.surfaces[0],width=30.),)),
        initial_state=replace(base.initial_state,vehicle=vehicle),
        simulation_config=replace(base.simulation_config,fixed_dt=1.),
        race_config=owner_factory().race_config,reward_composer=RewardComposer((TimePenalty(2.),)))


@pytest.mark.parametrize('finish,expected,kind',[(3.5,-1.4,'successful_finish'),(4.5,-1.6,'valid_terminal_failure')])
def test_finish_hard_exit_and_timeout_follow_actual_event_order(finish,expected,kind):
    env=diagonal_factory(finish).create()
    try:
        env.reset()
        _,reward,terminated,truncated,info=env.step(np.zeros(2,dtype=np.float32))
        assert terminated and not truncated
        assert reward==pytest.approx(expected)
        assert info['research_outcome']['kind']==kind
        assert info['finished']==(kind=='successful_finish')
        assert info['race']['directed_crossings']==(('finish',) if kind=='successful_finish' else ())
    finally:
        env.close()


def test_zero_duration_outward_boundary_failure_has_no_eligible_reward():
    base=owner_factory()
    env=replace(base,simulation_config=replace(base.simulation_config,fixed_dt=.4)).create()
    try:
        env.reset()
        assert not env.step(np.zeros(2,dtype=np.float32))[2]
        boundary=env.step(np.zeros(2,dtype=np.float32))
        assert not boundary[2] and not boundary[3]
        _,reward,terminated,truncated,info=env.step(np.zeros(2,dtype=np.float32))
        assert terminated and not truncated and reward==0.
        assert info['race']['terminal_time_s']==0.
        assert info['reward_components']=={'time':0.}
    finally:
        env.close()


def test_owner_contract_is_versioned_and_cannot_grant_training_or_change_reward(tmp_path):
    import json
    from pathlib import Path
    from hashlib import sha256
    from f1rl.learning.hard_route_outcome import owner_outcome_contract
    root=Path(__file__).resolve().parents[2]
    old=root/'configs/recovery/wp7-owner-decisions-preregistered-v1.yml'
    assert sha256(old.read_bytes()).hexdigest()=='d9dd49cef664b083a30c98cf6b1d9205ca9a45522ed1ef431b49289edf72c0d1'
    approved=owner_outcome_contract(root/'configs/research/wp7-hard-route-outcome-v2.json')
    for key,value in [('terminal_penalty',1.),('training_permitted',True),('outcome','successful_finish')]:
        document=dict(approved['configuration']);document[key]=value
        path=tmp_path/'changed.json';path.write_text(json.dumps(document),encoding='utf-8')
        with pytest.raises(ValueError,match='owner-v2'):
            owner_outcome_contract(path)


def test_actual_owner_config_boundary_rejects_changed_default_addendum(monkeypatch):
    import json
    from pathlib import Path
    from f1rl.learning.hard_route_outcome import owner_outcome_contract
    approved=owner_outcome_contract()['configuration']
    changed=dict(approved,terminal_penalty=1.)
    original=Path.read_bytes
    def read(path,*args,**kwargs):
        if path.name=='wp7-hard-route-outcome-v2.json':
            return json.dumps(changed).encode('utf-8')
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'read_bytes',read)
    with pytest.raises(ValueError,match='owner-v2'):
        WP7RaceConfig(version='wp7_directed_grounded_owner_v2',hard_violation='terminate_valid_failure')


def test_lifecycle_owner_v2_uses_the_same_failure_and_reward_prefix_rule():
    from f1rl.sandbox.ballistic_flight import FlightState
    from tests.integration.test_spatial_compliant_gym import lifecycle_factory
    base=owner_factory()
    config=lifecycle_factory().physics_config
    env=replace(base,physics_config=config,
        initial_state=replace(base.initial_state,flight=FlightState('grounded')),
        race_config=WP7RaceConfig(version='wp7_directed_lifecycle_owner_v2',hard_violation='terminate_valid_failure')).create()
    try:
        env.reset()
        assert not env.step(np.zeros(2,dtype=np.float32))[2]
        _,reward,terminated,truncated,info=env.step(np.zeros(2,dtype=np.float32))
        assert terminated and not truncated and reward==pytest.approx(-.6)
        assert info['research_outcome']['kind']=='valid_terminal_failure'
        assert info['physics']['transition']['physical_mode']=='wp7_lifecycle_v1'
    finally:
        env.close()


def test_registered_progress_time_collision_terms_keep_coefficients_and_prefix_position():
    from math import hypot
    from f1rl.learning.rewards import TargetProgressReward,RouteProgressReward,CollisionPenalty
    composer=RewardComposer((TargetProgressReward(1.3),RouteProgressReward(.7),TimePenalty(2.),CollisionPenalty(19.)))
    env=replace(owner_factory(),reward_composer=composer).create()
    try:
        env.reset()
        env.step(np.zeros(2,dtype=np.float32))
        _,reward,terminated,_,info=env.step(np.zeros(2,dtype=np.float32))
        expected=1.3*(hypot(50.,2.5)-hypot(50.,4.))
        assert terminated
        assert info['reward_components']=={'target_progress':pytest.approx(expected),'route_progress':pytest.approx(0.,abs=1e-12),
                                          'time':pytest.approx(-.6),'collision':0.}
        assert reward==pytest.approx(expected-.6)
        assert composer.terms[0].weight_per_meter==1.3 and composer.terms[-1].penalty==19.
    finally:
        env.close()


def test_approved_outcome_does_not_admit_training_or_allocate_artifacts(tmp_path):
    from f1rl.learning.training.runner import TrainingRunner
    from f1rl.learning.training.config import TrainingRunConfig,VectorEnvConfig
    from f1rl.learning.algorithms.ppo import PPOConfig
    from f1rl.research.wp7_admission import ScientificAdmissionError
    config=TrainingRunConfig(PPOConfig(),VectorEnvConfig(n_envs=1),artifacts_root=tmp_path/'forbidden')
    with pytest.raises(ScientificAdmissionError):
        TrainingRunner(config).run_single(owner_factory())
    assert not config.artifacts_root.exists()


def test_owner_contract_cannot_hash_different_bytes_from_the_validated_document(monkeypatch):
    import json
    from pathlib import Path
    from f1rl.learning.hard_route_outcome import owner_outcome_contract
    approved=owner_outcome_contract()['configuration']
    changed=json.dumps(dict(approved,training_permitted=True)).encode('utf-8')
    original=Path.read_bytes
    def read(path):
        return changed if path.name=='wp7-hard-route-outcome-v2.json' else original(path)
    monkeypatch.setattr(Path,'read_bytes',read)
    with pytest.raises(ValueError,match='owner-v2'):
        owner_outcome_contract()
