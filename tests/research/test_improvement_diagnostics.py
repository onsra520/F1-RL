import importlib.util
from pathlib import Path

import numpy as np
import pytest


def test_diagnostic_controller_corrects_target_direction_without_claiming_ppo():
    path = Path(__file__).resolve().parents[2]/'scripts/diagnose_driving.py'
    spec = importlib.util.spec_from_file_location('diagnostic', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    observation = np.zeros(16, dtype=np.float32)
    observation[12] = 1.
    observation[10] = .2
    action = module.diagnostic_action(observation, 0.)
    assert 0. < action[0] <= 1. and action[1] == pytest.approx(.6) and action[2] == 0.
    observation[10] = -.2
    assert module.diagnostic_action(observation, 0.)[0] < 0.


def test_timing_preserves_result_and_exception(monkeypatch):
    from f1rl.exploratory import pilot
    moments = iter([1., 3., 4., 7.])
    monkeypatch.setattr(pilot.time, 'perf_counter', lambda: next(moments))
    totals = {}
    assert pilot.timed_call(lambda value: value+1, totals, 'operation')(3) == 4
    assert totals == {'operation': 2.}
    def fail():
        raise ValueError('original failure')
    with pytest.raises(ValueError, match='original failure'):
        pilot.timed_call(fail, totals, 'operation')()
    assert totals == {'operation': 5.}


@pytest.mark.parametrize('axis', [0, 1, 2])
@pytest.mark.parametrize('invalid', [float('inf'), -float('inf'), float('nan')])
def test_vector_finite_validation_preserved(axis, invalid):
    from f1rl.sandbox.geometry.vector import Vec3
    values = [0., 1., -1.]
    values[axis] = invalid
    with pytest.raises(ValueError, match='components must be finite'):
        Vec3(*values)


def test_comparison_requires_progress_improvement_and_valid_samples():
    path = Path(__file__).resolve().parents[2]/'scripts/compare_driving.py'
    spec = importlib.util.spec_from_file_location('comparison', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    old = {'successes':0,'mean_progress':.15,'median_progress':.15,'worst_progress':.14}
    better = {'all_valid':True,'successes':0,'mean_progress':.4,'median_progress':.4,'worst_progress':.3}
    assert module.improved(old, better)
    assert not module.improved(old, {**better,'all_valid':False})
    assert not module.improved(old, {**better,'worst_progress':.1})
    assert not module.improved(old, {**better,'mean_progress':.19})
