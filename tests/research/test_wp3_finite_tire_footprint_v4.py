"""Finite tire footprint at polygon road edges must have conservative work."""
from math import isclose

import pytest

from src.simulation.finite_tire_footprint_v4 import (
    FOOTPRINT_RADIUS_M,_single_polygon_coverage,
)

SQUARE=((-1.,-1.),(1.,-1.),(1.,1.),(-1.,1.))
LEFT=((-2.,-1.),(0.,-1.),(0.,1.),(-2.,1.))
RIGHT=((0.,-1.),(2.,-1.),(2.,1.),(0.,1.))


def test_no_instantaneous_potential_jump_across_polygon_edge():
    r=FOOTPRINT_RADIUS_M
    for x in (-1.-r-1e-4,-1.-r,-1.,-1.+r,-1.+r+1e-4):
        alpha,gradient=_single_polygon_coverage(x,0.,SQUARE,r)
        assert 0.<=alpha<=1.
        assert gradient.y==0.
    a0,g0=_single_polygon_coverage(-1.-r,0.,SQUARE,r)
    a1,g1=_single_polygon_coverage(-1.,0.,SQUARE,r)
    a2,g2=_single_polygon_coverage(-1.+r,0.,SQUARE,r)
    assert a0==0. and g0.magnitude==0.
    assert a1==pytest.approx(.5)
    assert a2==pytest.approx(1.) and g2.magnitude<1e-12


def test_exact_analytic_energy_gradient_cancels_work():
    E=533.440979
    r=FOOTPRINT_RADIUS_M
    for x,z in [(-1.,0.),(-.96,.1),(.98,-.07),(-.96,-.95)]:
        a,g=_single_polygon_coverage(x,z,SQUARE,r)
        dx=1e-6
        axp,_=_single_polygon_coverage(x+dx,z,SQUARE,r)
        axm,_=_single_polygon_coverage(x-dx,z,SQUARE,r)
        azp,_=_single_polygon_coverage(x,z+dx,SQUARE,r)
        azm,_=_single_polygon_coverage(x,z-dx,SQUARE,r)
        assert g.x*E==pytest.approx(E*(axp-axm)/(2*dx),abs=1e-6)
        assert g.z*E==pytest.approx(E*(azp-azm)/(2*dx),abs=1e-6)


def test_adjacent_seams_have_continuous_full_coverage():
    for x in (-.05,0.,.05):
        l,lg=_single_polygon_coverage(x,0.,LEFT,FOOTPRINT_RADIUS_M)
        r,rg=_single_polygon_coverage(x,0.,RIGHT,FOOTPRINT_RADIUS_M)
        assert l+r==pytest.approx(1.,abs=1e-11)
        assert (lg+rg).magnitude==pytest.approx(0.,abs=1e-11)


@pytest.mark.parametrize("radius",[0.,-.1,.21])
def test_invalid_footprint_radius_rejected(radius):
    with pytest.raises(ValueError):
        _single_polygon_coverage(0.,0.,SQUARE,radius)
