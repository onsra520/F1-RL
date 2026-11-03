"""Finite planar tyre footprint contact regularization: development Physics v4.

The nominal point support at a polygon edge has a discontinuous preloaded
spring energy.  This model integrates a finite patch footprint with a smooth
C1 source-bound coverage and its exact horizontal energy-gradient force.
Without that conservative edge force, fading the spring normal causes a
numerically invisible creation/loss of mechanical energy. No artificial
dissipation, impulse, preload source or higher solver tolerance is added.

This is an explicit new experimental development contact law, *not* an
independently certified scientific force model.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import hypot, isfinite

from .geometry.vector import Vec3
from .independent_suspension_dynamics import SuspensionResolutionError
from .multi_surface_support import IndependentWheelContact, _height
from .vehicle import VehicleState

NAMES=("front_left","front_right","rear_left","rear_right")
FOOTPRINT_RADIUS_M=0.1  # explicit nominal projected contact half-width, dev


@dataclass(frozen=True,slots=True)
class FootprintWheel:
    contact:IndependentWheelContact|None
    coverage:float
    coverage_gradient:Vec3


def _single_polygon_coverage(x:float,z:float,polygon,radius:float):
    """C1 product kernel for a convex CCW road footprint polygon.

    Each one-sided edge uses smoothstep(signed_distance / (2*radius) + .5).
    The exact analytic in-plane gradient is returned for conservative work.
    """
    if not polygon or len(polygon)<3 or not 0.<radius<=0.2:
        raise ValueError("bounded physical tire footprint and convex patch required")
    factors=[]
    for a,b in zip(polygon,polygon[1:]+polygon[:1],strict=True):
        dx,dz=b[0]-a[0],b[1]-a[1]
        size=hypot(dx,dz)
        if size<=1e-10:raise ValueError("degenerate road boundary")
        signed=(dx*(z-a[1])-dz*(x-a[0]))/size
        u=(signed+radius)/(2.*radius)
        if u<=0: return 0.,Vec3(0.,0.,0.)
        if u>=1.:
            factors.append((1.,0.,Vec3(0.,0.,0.)))
        else:
            value=u*u*(3.-2.*u)
            derivative=(6.*u*(1.-u))/(2.*radius)
            factors.append((value,derivative,Vec3(-dz/size,0.,dx/size)))
    covered=1.
    for value,_,_ in factors:covered*=value
    grad=Vec3(0.,0.,0.)
    for i,(_,slope,axis) in enumerate(factors):
        if slope:
            other=1.
            for j,(factor,_,_) in enumerate(factors):
                if i!=j:other*=factor
            grad=grad+axis*(slope*other)
    if not all(isfinite(x) for x in (covered,grad.x,grad.z)):
        raise ValueError("nonfinite physical support geometry")
    return covered,grad


def sample_finite_wheel_coverage(
    vehicle:VehicleState, contact_policy, atlas, *,
    radius_m:float=FOOTPRINT_RADIUS_M,
)->tuple[FootprintWheel,...]:
    if not isinstance(vehicle,VehicleState) or not 0.<radius_m<=.2:
        raise ValueError("finite spatial tire footprint inputs required")
    patches,polygons=atlas
    out=[]
    for name,local in zip(NAMES,contact_policy.wheels.positions,strict=True):
        point=vehicle.position+vehicle.orientation.rotate(local)
        options=[]
        alpha,gradient=0.,Vec3(0.,0.,0.)
        for patch,polygon in zip(patches,polygons,strict=True):
            weight,normal=_single_polygon_coverage(point.x,point.z,polygon,radius_m)
            if weight<=0:continue
            height=_height(patch,point.x,point.z)
            deflection=height-point.y
            if not isfinite(deflection):raise SuspensionResolutionError("finite-tire wheel gap nonfinite")
            if deflection>contact_policy.max_bump_m+1e-9:
                fault=SuspensionResolutionError(
                    f"finite-tire wheel exceeds unchanged bump stop:"
                    f" {name} {patch.id} deflection={deflection:.9g}",
                    failure_code="over_bump_travel")
                # Preserve the original rejected RK4 trial as a diagnostic;
                # only the independent certified stop solver may accept it.
                fault.wheel_index=NAMES.index(name)
                fault.wheel_local=local
                fault.road_patch=patch
                fault.rejected_vehicle=vehicle
                raise fault
            if deflection < -contact_policy.max_droop_m-1e-9:
                continue
            options.append((weight,-abs(deflection),patch.id,height,deflection,
                            normal,patch.normal/patch.normal.y,patch.route_index))
            alpha+=weight
            gradient=gradient+normal
        if not options or alpha<=0.:
            out.append(FootprintWheel(None,0.,Vec3(0.,0.,0.)))
            continue
        # Neighbouring route patches overlap by design, but their planar
        # slopes differ. Picking the largest footprint weight changes the
        # spring potential discontinuously at the seam; RK4 cannot refine
        # away a finite energy jump. Use a differentiable partition of unity
        # for BOTH the potential height and its exact spatial derivative.
        # This is an explicit v4 contact law: the old v3 source is untouched.
        heights=[entry[3] for entry in options]
        if len(options)>1:
            from .multi_surface_support import _ADJACENT_PATCH_STEP_M
            same_route=all(entry[7] is not None for entry in options)
            seam_bound=(_ADJACENT_PATCH_STEP_M if same_route
                        else contact_policy.seam_height_tolerance_m)
            if max(heights)-min(heights)>seam_bound+1e-9:
                raise SuspensionResolutionError(
                    'finite-tire blended seam would combine distinct road decks')
        h=sum(entry[0]*entry[3] for entry in options)/alpha
        # h = sum(w_i h_i)/sum(w_i), hence grad h is the product-rule
        # derivative including the coverage-gradient term. Omitting the
        # latter would still create numerical energy at transition seams.
        dh=Vec3(0.,0.,0.)
        for weight,_,_,height,_,grad_weight,plane_gradient,_ in options:
            dh=dh+(Vec3(-plane_gradient.x,0.,-plane_gradient.z)*weight
                   +grad_weight*(height-h))
        dh=dh/alpha
        _,_,pid,_,_,_,_,_=max(options)
        blended_gradient=Vec3(-dh.x,1.,-dh.z)
        if alpha>=1.:
            alpha,gradient=1.,Vec3(0.,0.,0.)
        out.append(FootprintWheel(
            IndependentWheelContact(name,pid,h,h-point.y,blended_gradient),
            alpha,gradient))
    return tuple(out)
