from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.exploratory.reward import REGISTERED, apply_training_curriculum, validate
from src.simulation.geometry.vector import Vec3

CONFIG = {'version':'straight_route_stability_demo_v1', 'training_corridor_m':2.,
          'lateral_penalty_per_m2_s':2., 'terminal_penalty':25.}


def test_registered_task_is_identical_and_demo_stop_keeps_original_outcome():
    original = (np.zeros(16), 1., False, False,
        {'research_outcome':{'kind':'active','reason':'accepted_step','valid_sample':True}})
    env = SimpleNamespace(_authored_route=SimpleNamespace(horizontal_distance=lambda position:2.1),
        _simulation=SimpleNamespace(snapshot=SimpleNamespace(world=SimpleNamespace(vehicle=SimpleNamespace(position=Vec3(2.1,.5,10.))))),
        _simulation_config=SimpleNamespace(fixed_dt=.1))
    assert apply_training_curriculum(env, original, REGISTERED) is original
    result = apply_training_curriculum(env, original, CONFIG)
    assert result[2] is True and result[3] is False
    assert result[1] == pytest.approx(1.-2.*2.1**2*.1-25.)
    assert result[4]['research_outcome'] == original[4]['research_outcome']
    assert result[4]['demo_training']['curriculum_terminal'] is True
    assert 'demo_training' not in original[4]


@pytest.mark.parametrize('changes', [{'version':'unregistered'}, {'training_corridor_m':4.},
    {'terminal_penalty':-1.}, {'lateral_penalty_per_m2_s':float('nan')}])
def test_demo_curriculum_rejects_unsupported_scope(changes):
    with pytest.raises(ValueError):
        validate({**CONFIG, **changes})


def test_demo_training_stops_before_reproducible_invalid_support_boundary():
    from src.validation.wp7_protocol import load_registration, prepare_scenarios
    root = Path(__file__).resolve().parents[2]
    factory = prepare_scenarios(root,load_registration(root,profile='spatial_rotating_owner_v3'),seed_index=0).scenarios[0].factory
    env = replace(factory,reset_config=replace(factory.reset_config,seed=10000)).create()
    env.reset(seed=10000)
    try:
        for step in range(1,29):
            result = apply_training_curriculum(env,env.step(np.array([.2,1.,0.],dtype=np.float32)),CONFIG)
            if result[2] or result[3]:
                assert step < 28
                assert result[4]['demo_training']['curriculum_terminal'] is True
                assert result[4]['research_outcome']['reason'] == 'accepted_step'
                break
        else:
            pytest.fail('curriculum did not stop before the known invalid support input')
    finally:
        env.close()


def test_curriculum_v2_requires_exactly_three_meters_and_records_versioned_scope():
    config = validate({**CONFIG, 'version':'straight_route_stability_demo_v2',
                       'training_corridor_m':3.})
    original = (np.zeros(16), 1., False, False,
        {'research_outcome':{'kind':'active','reason':'accepted_step','valid_sample':True}})
    env = SimpleNamespace(_authored_route=SimpleNamespace(horizontal_distance=lambda position:3.1),
        _simulation=SimpleNamespace(snapshot=SimpleNamespace(world=SimpleNamespace(vehicle=SimpleNamespace(position=Vec3(3.1,.5,10.))))),
        _simulation_config=SimpleNamespace(fixed_dt=.1))
    result = apply_training_curriculum(env, original, config)
    assert result[2] is True
    assert result[4]['demo_training']['version'] == 'straight_route_stability_demo_v2'
    assert result[4]['demo_training']['training_corridor_m'] == 3.
    assert result[4]['research_outcome']['reason'] == 'accepted_step'


def test_v2_does_not_reclassify_registered_four_meter_hard_route_failure():
    config = validate({**CONFIG, 'version':'straight_route_stability_demo_v2',
                       'training_corridor_m':3.})
    original_info = {'research_outcome':{'kind':'failure','reason':'hard_route_violation',
                                         'valid_sample':True}}
    original = (np.zeros(16), -1., True, False, original_info)
    env = SimpleNamespace(_authored_route=SimpleNamespace(horizontal_distance=lambda position:4.1),
        _simulation=SimpleNamespace(snapshot=SimpleNamespace(world=SimpleNamespace(vehicle=SimpleNamespace(position=Vec3(4.1,.5,10.))))),
        _simulation_config=SimpleNamespace(fixed_dt=.1))
    result = apply_training_curriculum(env, original, config)
    assert result[2] is True
    assert result[4]['research_outcome']['reason'] == 'hard_route_violation'
    assert result[4]['demo_training']['curriculum_terminal'] is False
    assert result[4]['demo_training']['training_corridor_m'] == 3.


@pytest.mark.parametrize('version,corridor', [
    ('straight_route_stability_demo_v1',3.),
    ('straight_route_stability_demo_v2',2.),
    ('straight_route_stability_demo_v2',3.0001),
    ('straight_route_stability_demo_v2',4.),
])
def test_demo_curriculum_versions_reject_cross_version_or_boundary_changes(version,corridor):
    with pytest.raises(ValueError):
        validate({**CONFIG, 'version':version, 'training_corridor_m':corridor})
