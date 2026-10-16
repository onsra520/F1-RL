"""Explicit braking contract; legacy policy dimensions remain unchanged."""
from dataclasses import replace

import numpy as np
import pytest

from f1rl.learning.action import ActionContract
from tests.integration.test_wp7_lifecycle_race import lifecycle_race_factory


def test_explicit_brake_contract_decelerates_actual_grounded_lifecycle():
    base = lifecycle_race_factory(launch=False,finish=20.)
    candidate = replace(base,action_contract=ActionContract('steering-throttle-brake-v2'))
    env = candidate.create()
    try:
        env.reset()
        assert env.action_space.shape == (3,)
        info = env.step(np.array([0.,0.,1.],dtype=np.float32))[-1]
        assert info['telemetry']['vehicle_velocity'][2] == pytest.approx(3.)
        with pytest.raises(ValueError,match='shape'):
            env.step(np.array([0.,0.],dtype=np.float32))
    finally:
        env.close()


def test_legacy_two_action_contract_still_coasts_with_brake_zero():
    env = lifecycle_race_factory(launch=False,finish=20.).create()
    try:
        env.reset()
        assert env.action_space.shape == (2,)
        assert env.step(np.array([0.,0.],dtype=np.float32))[-1]['telemetry']['vehicle_velocity'][2] == pytest.approx(5.)
    finally:
        env.close()


def test_braking_cannot_reverse_stationary_or_slow_vehicle():
    base = lifecycle_race_factory(launch=False,finish=20.)
    base = replace(base,initial_state=replace(base.initial_state,
        vehicle=replace(base.initial_state.vehicle,velocity=replace(base.initial_state.vehicle.velocity,z=.5))),
        action_contract=ActionContract('steering-throttle-brake-v2'))
    env = base.create()
    try:
        env.reset()
        for _ in range(2):
            assert env.step(np.array([0.,0.,1.],dtype=np.float32))[-1]['telemetry']['vehicle_velocity'][2] == 0.
    finally:
        env.close()


def test_airborne_controls_do_not_invent_forces_and_observations_stay_valid():
    base = replace(lifecycle_race_factory(finish=20.),action_contract=ActionContract('steering-throttle-brake-v2'))
    a,b = base.create(),base.create()
    try:
        a.reset(); b.reset()
        obs,_,_,_,first = a.step(np.array([1.,1.,1.],dtype=np.float32))
        other,_,_,_,second = b.step(np.array([-1.,0.,0.],dtype=np.float32))
        assert np.array_equal(obs,other)
        assert first['telemetry'] == second['telemetry']
        phases = {False}
        for _ in range(8):
            assert a.observation_space.contains(obs) and np.isfinite(obs).all()
            obs,_,_,_,info = a.step(np.zeros(3,dtype=np.float32))
            phases.add(info['telemetry']['grounded'])
        assert phases == {False,True}
    finally:
        a.close(); b.close()


@pytest.mark.parametrize('action',[[0.,0.,-1.],[0.,0.,float('nan')],[0.,0.,2.]])
def test_invalid_brake_actions_rejected(action):
    env = replace(lifecycle_race_factory(launch=False),
        action_contract=ActionContract('steering-throttle-brake-v2')).create()
    try:
        env.reset()
        with pytest.raises(ValueError):
            env.step(np.array(action,dtype=np.float32))
    finally:
        env.close()


def test_legacy_saved_model_cannot_silently_migrate_to_three_actions(tmp_path):
    from f1rl.learning.algorithms.ppo import PPOConfig, PPOModel
    from tests.learning.evaluation.test_evaluator import _factory
    base = _factory(max_steps=2)
    old = base.create()
    new = replace(base,action_contract=ActionContract('steering-throttle-brake-v2')).create()
    try:
        model = PPOModel.create(old,PPOConfig(n_steps=2,batch_size=2,n_epochs=1,device='cpu'))
        archive = model.save(tmp_path/'legacy.zip')
        assert model.num_timesteps == 0
        with pytest.raises(ValueError,match='spaces'):
            PPOModel.load(archive,env=new,device='cpu')
        loaded = PPOModel.load(archive,env=old,device='cpu')
        assert loaded.predict(old.reset()[0]).shape == (2,)
    finally:
        old.close()
        new.close()
