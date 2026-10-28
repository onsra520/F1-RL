"""Actual TRAIN preparation/reset stepping, not policy training or evaluation."""
from pathlib import Path

import numpy as np
import pytest

from src.research.wp7_protocol import load_registration, prepare_scenarios

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope='module')
def scenarios():
    return prepare_scenarios(ROOT,load_registration(ROOT),seed_index=0).scenarios


@pytest.mark.parametrize('index',range(4))
def test_actual_registered_train_reset_and_first_physics_step(scenarios,index):
    factory = scenarios[index].factory
    initial = factory.initial_state.vehicle.position
    authored = factory.navigation.points[0]
    assert initial.y-authored.y == pytest.approx(factory.vehicle_parameters.ride_height)
    env = factory.create()
    try:
        observation,info = env.reset(seed=23)
        assert env.observation_space.contains(observation)
        assert info['train_reset']['rank']==0
        observation,_,terminated,truncated,info = env.step(np.array([0.,.2],dtype=np.float32))
        assert env.observation_space.contains(observation)
        assert not terminated and not truncated
        assert info['telemetry']['grounded']
        assert info['telemetry']['vehicle_position'][1] == pytest.approx(initial.y,abs=1e-8)
    finally:
        env.close()


@pytest.mark.parametrize('change',['wrong_height','legacy_reference'])
def test_registered_spawn_reference_cannot_be_implicitly_changed(scenarios,change):
    from dataclasses import replace
    factory = scenarios[0].factory
    if change=='wrong_height':
        initial = factory.initial_state
        factory = replace(factory,initial_state=replace(initial,
            vehicle=replace(initial.vehicle,position=replace(initial.vehicle.position,y=.6)),
            suspension=replace(initial.suspension,height_m=.6)))
    else:
        factory = replace(factory,physics_config=replace(factory.physics_config,
            spawn_reference='legacy_vehicle_position_v1'))
    with pytest.raises(ValueError,match='spawn|navigation start'):
        factory.create()
