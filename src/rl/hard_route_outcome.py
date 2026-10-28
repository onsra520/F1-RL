"""Owner-v2 hard-route geometry; never scientific training authorization.

The closed XZ tube is the union of finite-segment rectangles and endpoint
circles. Its path-parameter intervals identify first exit, including a path
that exits and returns before the tick ends. Exactly 4m remains inside.
"""
from math import copysign, isfinite, sqrt

from .route_navigation import AuthoredRoute
from ..simulation.geometry.vector import Vec3


def first_hard_route_exit(route: AuthoredRoute, previous: Vec3, candidate: Vec3) -> float | None:
    """First boundary fraction preceding witnessed >4m departure; None if inside.

    An outside initial state or unresolvable floating-point interval is invalid,
    not a valid terminal-failure sample. Ballistic centers have linear XZ paths.
    """
    if not isinstance(route,AuthoredRoute) or not isinstance(previous,Vec3) or not isinstance(candidate,Vec3):
        raise ValueError('authored route and finite path endpoints required')
    if route.horizontal_distance(previous)>4.:
        raise ValueError('initial route state is outside hard corridor')
    dx,dz=candidate.x-previous.x,candidate.z-previous.z
    speed2=dx*dx+dz*dz
    if not isfinite(speed2):
        raise ValueError('route path exceeds finite numerical domain')
    if speed2==0.:
        return None
    intervals=[]

    def clipped(lo,hi):
        lo,hi=max(0.,lo),min(1.,hi)
        return (lo,hi) if lo<=hi else None

    def band(value,rate,lower,upper):
        if rate==0.:
            return (0.,1.) if lower<=value<=upper else None
        a,b=(lower-value)/rate,(upper-value)/rate
        return clipped(min(a,b),max(a,b))

    def circle(point):
        x,z=previous.x-point.x,previous.z-point.z
        b=2.*(x*dx+z*dz)
        c=x*x+z*z-16.
        disc=b*b-4.*speed2*c
        if not all(isfinite(v) for v in (b,c,disc)):
            raise ValueError('route capsule exceeds finite numerical domain')
        if disc<0.:
            return
        q=-.5*(b+copysign(sqrt(disc),b))
        roots=(q/speed2,c/q) if q else (0.,0.)
        interval=clipped(min(roots),max(roots))
        if interval is not None:
            intervals.append(interval)

    for first,last in zip(route.points,route.points[1:]):
        circle(first)
        circle(last)
        ex,ez=last.x-first.x,last.z-first.z
        length2=ex*ex+ez*ez
        if not isfinite(length2):
            raise ValueError('route segment exceeds finite numerical domain')
        if length2==0.:
            continue
        x,z=previous.x-first.x,previous.z-first.z
        along=band(x*ex+z*ez,dx*ex+dz*ez,0.,length2)
        across=band(x*ez-z*ex,dx*ez-dz*ex,-4.*sqrt(length2),4.*sqrt(length2))
        if along is not None and across is not None:
            interval=clipped(max(along[0],across[0]),min(along[1],across[1]))
            if interval is not None:
                intervals.append(interval)
    covered=0.
    for lo,hi in sorted(intervals):
        if lo>covered:
            witness=previous+(candidate-previous)*((covered+lo)/2.)
            if route.horizontal_distance(witness)<=4.:
                raise ValueError('route capsule interval is numerically uncertain')
            return covered
        covered=max(covered,hi)
        if covered>=1.:
            return None
    witness=previous+(candidate-previous)*((covered+1.)/2.)
    if route.horizontal_distance(witness)<=4.:
        raise ValueError('route capsule exit is numerically uncertain')
    return covered


def owner_outcome_contract(path=None) -> dict:
    """Exact owner-v2 choices and canonical hash; never independent admission."""
    import json
    from hashlib import sha256
    expected=dict(schema_version='wp7_hard_route_owner_outcome_v2',version='wp7_hard_route_terminal_failure_v2',
        owner_decision_date='2026-10-17',
        parent_owner_contract_sha256='d9dd49cef664b083a30c98cf6b1d9205ca9a45522ed1ef431b49289edf72c0d1',
        soft_corridor_m=2.,hard_corridor_m=4.,violation_comparison='strict_greater_than',
        outcome='valid_terminal_failure',reward_policy='registered_terms_valid_prefix_no_terminal_adjustment',
        terminal_penalty=0.,new_success_credit=False,new_reward_shaping=False,
        training_permitted=False,heldout_permitted=False,scientific_freeze_permitted=False,wp8_permitted=False)
    from pathlib import Path
    import yaml
    from .owner_acceptance import _UniqueMappingLoader
    root=Path(__file__).resolve().parents[2]
    path=Path(path) if path is not None else root/'configs/research/wp7-hard-route-outcome-v2.json'
    source=path.read_bytes()
    document=yaml.load(source.decode('utf-8'),Loader=_UniqueMappingLoader)
    if (not isinstance(document,dict) or set(document)!=set(expected)
            or any(type(document[k]) is not type(v) or document[k]!=v for k,v in expected.items())):
        raise ValueError('owner-v2 outcome choices changed or unauthorized permission asserted')
    parent=root/'configs/recovery/wp7-owner-decisions-preregistered-v1.yml'
    _validate_owner_parent_identity(parent)
    return {'configuration':expected,'source_sha256':sha256(source).hexdigest(),'sha256':sha256(json.dumps(expected,sort_keys=True,
        separators=(',',':'),allow_nan=False).encode('utf-8')).hexdigest()}


def _validate_owner_parent_identity(parent):
    from hashlib import sha256
    # Pin both exact historical checkouts; accept no other source bytes.
    hashes={'d9dd49cef664b083a30c98cf6b1d9205ca9a45522ed1ef431b49289edf72c0d1',
            '04c7200f30a76460db2b5ca9a4b29ac8ccf2fe20b3d88556e2a14b2cb6315eeb'}
    if sha256(parent.read_bytes()).hexdigest() not in hashes:
        raise ValueError('owner-v2 historical parent identity changed')
