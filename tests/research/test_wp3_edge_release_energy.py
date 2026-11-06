"""Explicit no-creation dissipative road-edge release audit."""
from dataclasses import replace
import pytest

from f1rl.sandbox.wheel_edge_release_energy import certify_wheel_edge_release_energy

A=frozenset({0,1,2,3})
B=frozenset({1,2,3})


def certificate(**overrides):
    info={
      "start_spring_energy_j":20.,"end_spring_energy_j":16.,
      "balance_residual_j":-4.,"balance_tolerance_j":.0001,
      "bracket_duration_s":1e-9}
    info.update(overrides)
    return certify_wheel_edge_release_energy((A,B,B,B,B),**info)


def test_potential_transferred_to_environment_during_monotone_release():
    x=certificate()
    assert x.removed_supports==(0,)
    assert x.transferred_to_environment_j==4.
    assert x.residual_after_transfer_j==0.
    assert x.bracket_duration_s==1e-9


@pytest.mark.parametrize("masks",[
    (A,A,A,A,A),(A,B,A,B,B),
    (A,frozenset({0,1,2}),frozenset({0,1,2,3})),
    (frozenset({0,1,4}),B)
])
def test_recontact_no_release_invalid_masks_rejected(masks):
    with pytest.raises(ValueError):
        certify_wheel_edge_release_energy(masks,
            start_spring_energy_j=20.,end_spring_energy_j=16.,
            balance_residual_j=-4.,balance_tolerance_j=.0001,
            bracket_duration_s=1e-9)


@pytest.mark.parametrize("update",[
    {"start_spring_energy_j":15.},
    {"end_spring_energy_j":22.},
    {"balance_residual_j":4.},
    {"balance_residual_j":-3.},
    {"balance_tolerance_j":0.},
    {"bracket_duration_s":1e-7},
    {"bracket_duration_s":0.},
    {"bracket_duration_s":float("nan")},
    {"bracket_duration_s":True},
])
def test_no_energy_creation_or_unregistered_event(update):
    with pytest.raises(ValueError):
        certificate(**update)
