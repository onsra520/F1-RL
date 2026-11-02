"""Prospective WP7 continuous body-footprint containment in a flat road UNION.

Strictly a planar XZ road-coverage contract. Sloped/multi-height roads and
airborne motion are UNSUPPORTED until a versioned ramp/flight policy exists.
No centre-only or four-corner-only shortcuts: the projected 3D body convex
hull must be covered by the UNION of road patch rectangles.

For a moving body, an inflated midpoint footprint encloses every 3D body
point's XZ projection between two sampled poses under the same linear
translation + shortest-arc SLERP assumptions as body_collision.py.
Intervals not proven safe are recursively split; exhausting the budget
returns UNCERTAIN, never CLEAR.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import acos, isfinite

from .body_collision import VehicleBodyGeometry, _slerp
from .geometry.vector import Vec3
from .track.patch import PlanarSurfacePatch
from .vehicle import VehicleState


_XZ = tuple[float, float]
_POLYGON = tuple[_XZ, ...]
_AREA_EPS = 1e-9
_FLAT_EPS = 1e-9


def _cross(a: _XZ, b: _XZ, c: _XZ) -> float:
    return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])


def _area(poly: _POLYGON) -> float:
    if len(poly) < 3:
        return 0.0
    return abs(sum(
        x1*z2-x2*z1 for (x1,z1),(x2,z2)
        in zip(poly,poly[1:]+poly[:1],strict=True)
    )) / 2.0


def _hull(points: tuple[_XZ,...]) -> _POLYGON:
    """Monotone convex hull in CCW order; excludes redundant collinear points."""
    ordered=sorted(set(points))
    if len(ordered)<3:
        return ()
    lower:list[_XZ]=[]
    for p in ordered:
        while len(lower)>1 and _cross(lower[-2],lower[-1],p)<=0.0:
            lower.pop()
        lower.append(p)
    upper:list[_XZ]=[]
    for p in reversed(ordered):
        while len(upper)>1 and _cross(upper[-2],upper[-1],p)<=0.0:
            upper.pop()
        upper.append(p)
    return tuple(lower[:-1]+upper[:-1])


def _clip(poly:_POLYGON,a:_XZ,b:_XZ,*,inside:bool)->_POLYGON:
    """Clip polygon against one inclusive half-plane of a CCW polygon edge."""
    if not poly:
        return ()
    output:list[_XZ]=[]
    previous=poly[-1]
    ds=_cross(a,b,previous)
    previous_ok=ds>=0.0 if inside else ds<=0.0
    for current in poly:
        de=_cross(a,b,current)
        current_ok=de>=0.0 if inside else de<=0.0
        if current_ok!=previous_ok:
            frac=ds/(ds-de)
            output.append((
                previous[0]+frac*(current[0]-previous[0]),
                previous[1]+frac*(current[1]-previous[1]),
            ))
        if current_ok:
            output.append(current)
        previous,ds,previous_ok=current,de,current_ok
    return tuple(output)


def _subtract(subject:_POLYGON,patch:_POLYGON)->tuple[_POLYGON,...]:
    """Convex subject minus one convex road patch as disjoint area pieces."""
    remaining:list[_POLYGON]=[]
    inside_parts=[subject]
    for a,b in zip(patch,patch[1:]+patch[:1],strict=True):
        next_inside=[]
        for piece in inside_parts:
            outside=_clip(piece,a,b,inside=False)
            if _area(outside)>_AREA_EPS:
                remaining.append(outside)
            within=_clip(piece,a,b,inside=True)
            if _area(within)>_AREA_EPS:
                next_inside.append(within)
        inside_parts=next_inside
    return tuple(remaining)


def _uncovered_area(subject:_POLYGON,road_polygons:tuple[_POLYGON,...])->float:
    if not subject:
        return 0.0
    subject_bounds=_bounds(subject)
    fragments=(subject,)
    for road,bounds in zip(road_polygons,_polygon_bounds(road_polygons),strict=True):
        if not _bounds_overlap(subject_bounds,bounds):
            continue
        next_fragments=[]
        for fragment in fragments:
            next_fragments.extend(_subtract(fragment,road))
        fragments=tuple(next_fragments)
        if not fragments:
            return 0.0
        if len(fragments)>4096:
            # Complexity cannot be allowed to manufacture a CLEAR verdict.
            raise RuntimeError("road-union coverage subdivision budget exceeded")
    return sum(_area(fragment) for fragment in fragments)


def body_xz_polygon(state:VehicleState,body:VehicleBodyGeometry)->_POLYGON:
    """Whole horizontal projection of 3D OBB, not only tire/corner samples."""
    if not isinstance(state,VehicleState) or not isinstance(body,VehicleBodyGeometry):
        raise ValueError("body projection requires VehicleState and VehicleBodyGeometry")
    box=body.box_at(state.position,state.orientation)
    x=box.orientation.rotate(Vec3(1.,0.,0.))
    y=box.orientation.rotate(Vec3(0.,1.,0.))
    z=box.orientation.rotate(Vec3(0.,0.,1.))
    ext=box.half_extents
    points=[]
    for sx in (-1,1):
        for sy in (-1,1):
            for sz in (-1,1):
                v=box.center+x*(sx*ext.x)+y*(sy*ext.y)+z*(sz*ext.z)
                points.append((v.x,v.z))
    return _hull(tuple(points))


def _inflate(poly:_POLYGON,radius:float)->_POLYGON:
    if radius<0.0 or not isfinite(radius):
        raise ValueError("inflation radius must be finite non-negative")
    return _hull(tuple(
        (x+dx,z+dz)
        for x,z in poly
        for dx in (-radius,radius)
        for dz in (-radius,radius)
    ))


def _bounds(poly:_POLYGON)->tuple[float,float,float,float]:
    return min(x for x,_ in poly),max(x for x,_ in poly),min(z for _,z in poly),max(z for _,z in poly)


def _bounds_overlap(first:tuple[float,float,float,float],second:tuple[float,float,float,float])->bool:
    return not (first[1]<second[0] or second[1]<first[0]
                or first[3]<second[2] or second[3]<first[2])


@lru_cache(maxsize=32)
def _polygon_bounds(polygons:tuple[_POLYGON,...])->tuple[tuple[float,float,float,float],...]:
    return tuple(_bounds(polygon) for polygon in polygons)


def _flat_roads(surfaces:tuple[PlanarSurfacePatch,...])->tuple[_POLYGON,...]|None:
    if not surfaces or not isinstance(surfaces,tuple) or not all(
        isinstance(p,PlanarSurfacePatch) for p in surfaces
    ):
        return None
    reference_y=surfaces[0].origin.y
    result=[]
    for patch in surfaces:
        n=patch.normal
        if (
            abs(n.x)>_FLAT_EPS or abs(n.y-1.)>_FLAT_EPS
            or abs(n.z)>_FLAT_EPS
            or abs(patch.origin.y-reference_y)>_FLAT_EPS
        ):
            return None
        right=patch.right
        vertices=tuple(
            (
                patch.origin.x+sx*right.x*patch.width/2.+sz*patch.forward.x*patch.length/2.,
                patch.origin.z+sx*right.z*patch.width/2.+sz*patch.forward.z*patch.length/2.,
            )
            for sx in (-1.,1.) for sz in (-1.,1.)
        )
        hull=_hull(vertices)
        if _area(hull)<=_AREA_EPS:
            return None
        result.append(hull)
    return tuple(result)


@dataclass(frozen=True,slots=True)
class RoadSweepResult:
    status: str
    checks: int
    reason: str

    def __post_init__(self)->None:
        if self.status not in {"clear","outside","uncertain","unsupported"}:
            raise ValueError("unexpected road-sweep status")
        if type(self.checks) is not int or self.checks<0:
            raise ValueError("checks must be nonnegative")

    @property
    def valid(self)->bool:
        return self.status=="clear"


def swept_body_road_union(
    previous:VehicleState,
    candidate:VehicleState,
    body:VehicleBodyGeometry,
    surfaces:tuple[PlanarSurfacePatch,...],
    *,
    ride_height_m:float,
    max_subdivisions:int=12,
)->RoadSweepResult:
    """Certify continuous projected 3D body inside a union of FLAT road patches.

    Neither a projected road union nor this check proves support over ramps,
    flying, elevated overlaps, physical tire contact, or actual road curvature.
    Those require separate versioned policies and are intentionally blocked.
    """
    if not isinstance(previous,VehicleState) or not isinstance(candidate,VehicleState):
        raise ValueError("sweep endpoints must be VehicleState")
    if not isinstance(body,VehicleBodyGeometry):
        raise ValueError("body must be VehicleBodyGeometry")
    if type(max_subdivisions) is not int or not 0<=max_subdivisions<=20:
        raise ValueError("max_subdivisions must be integer in [0,20]")
    if isinstance(ride_height_m,bool) or not isinstance(ride_height_m,(int,float)) or not isfinite(ride_height_m) or ride_height_m<=0.:
        raise ValueError('ride_height_m must be finite positive')
    roads=_flat_roads(surfaces)
    if roads is None:
        return RoadSweepResult("unsupported",0,"requires coplanar horizontal road patches")
    if not previous.grounded or not candidate.grounded:
        return RoadSweepResult("unsupported",0,"airborne/ramp policy not registered")
    expected_y=surfaces[0].origin.y+ride_height_m
    if abs(previous.position.y-expected_y)>1e-6 or abs(candidate.position.y-expected_y)>1e-6:
        return RoadSweepResult("unsupported",0,"grounded reference pose disagrees with road elevation")
    q0,q1=previous.orientation,candidate.orientation
    dot=abs(q0.w*q1.w+q0.x*q1.x+q0.y*q1.y+q0.z*q1.z)
    theta=2.*acos(min(1.,max(0.,dot)))
    translation=(candidate.position-previous.position).magnitude
    max_motion=translation+theta*body.motion_radius_m
    checks=0

    def polygon_at(t:float)->_POLYGON:
        nonlocal checks
        checks+=1
        state=VehicleState(
            previous.position+(candidate.position-previous.position)*t,
            previous.velocity,
            _slerp(q0,q1,t),
            previous.angular_velocity,
            True,
        )
        return body_xz_polygon(state,body)

    if _uncovered_area(polygon_at(0.),roads)>_AREA_EPS:
        return RoadSweepResult("outside",checks,"start pose footprint outside road union")
    if _uncovered_area(polygon_at(1.),roads)>_AREA_EPS:
        return RoadSweepResult("outside",checks,"end pose footprint outside road union")

    def visit(lo:float,hi:float,remaining:int)->str:
        mid=(lo+hi)/2.
        poly=polygon_at(mid)
        if _uncovered_area(poly,roads)>_AREA_EPS:
            return "outside"
        envelope=_inflate(poly,max_motion*(hi-lo)/2.)
        if _uncovered_area(envelope,roads)<=_AREA_EPS:
            return "clear"
        if remaining==0:
            return "uncertain"
        left=visit(lo,mid,remaining-1)
        if left=="outside":
            return left
        right=visit(mid,hi,remaining-1)
        if right=="outside":
            return right
        return "uncertain" if "uncertain" in (left,right) else "clear"

    status=visit(0.,1.,max_subdivisions)
    return RoadSweepResult(status,checks,{
        "clear":"continuous whole-body projection inside flat road union",
        "outside":"sampled body footprint outside flat road union",
        "uncertain":"unable to prove continuous body footprint inside road union",
    }[status])
