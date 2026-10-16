"""Continuous planar candidate certificates, never dynamic or training admission.

Each segment is x(z)=x0+(x1-x0)*(10u^3-15u^4+6u^5). Its first two
derivatives vanish at joins. Global derivative bounds enclose EVERY body pose
in each interval; clear samples alone cannot earn a certificate. Failed finite
search is not an infeasibility proof. Sloped support remains a separate gate.
"""
from dataclasses import asdict
from math import atan, ceil, isfinite, sqrt, tan

from ..learning.pilot_preflight import TRAIN_IDS
from ..sandbox.body_collision import VehicleBodyGeometry, obb_separation_lower_bound
from ..sandbox.geometry.rotation import Quaternion
from ..sandbox.geometry.vector import Vec3
from ..sandbox.road_footprint import swept_body_road_union
from ..sandbox.vehicle import VehicleState
from ..track_assets.canonical import fingerprint_track_world
from .wp7_protocol import digest
from .wp7_track_feasibility import BODY, PARAMETERS

UP = Vec3(0., 1., 0.)
ZERO = Vec3(0., 0., 0.)


def _state(first, last, z, height):
    x0, z0 = first
    x1, z1 = last
    u = (z-z0)/(z1-z0)
    x = x0+(x1-x0)*(10*u**3-15*u**4+6*u**5)
    slope = (x1-x0)/(z1-z0)*30*u**2*(1-u)**2
    return VehicleState(Vec3(x, height, z), ZERO,
                        Quaternion.from_axis_angle(UP, atan(slope)), ZERO, True)


def certify_planar_trajectory(asset, knots, *, parameters=PARAMETERS, interval_m=.05):
    """Certify a finite C2 graph candidate against actual flat TRAIN geometry.

    Curvature <= |x''| <= 10/sqrt(3)*|dx|/dz^2; |x'| <= 15/8*|dx|/dz.
    Center displacement <= sqrt(1+max_slope^2)*h/2, yaw displacement
    <= max_second*h/2. Inflating the midpoint OBB by their sum (rotation
    multiplied by body motion radius) encloses the full continuous hull.
    This is a kinematic certificate, not a force/slip/speed certificate.
    """
    if asset.track_id not in TRAIN_IDS:
        raise ValueError("only four registered TRAIN tracks supported")
    if (type(interval_m) not in (int, float) or not isfinite(interval_m)
            or not .01 <= interval_m <= .25):
        raise ValueError("interval_m must be in [.01,.25]")
    if (not isinstance(knots, tuple) or not 2 <= len(knots) <= 64
            or any(not isinstance(k, tuple) or len(k) != 2
                   or any(type(v) not in (int, float) or not isfinite(v) for v in k) for k in knots)
            or any(not isfinite(b[0]-a[0]) or not isfinite(b[1]-a[1])
                   or b[1]-a[1] < 1e-6 for a, b in zip(knots, knots[1:]))):
        raise ValueError("finite ordered (x,z) knots with increasing z required")
    if (knots[-1][1]-knots[0][1] > 1000
            or any(abs(b[0]-a[0]) > 1000 for a,b in zip(knots,knots[1:]))):
        raise ValueError("trajectory extent exceeds bounded certification budget")
    blockers = []
    surfaces = asset.world.surfaces
    gates = (*sorted(asset.world.checkpoints, key=lambda g: g.order), *asset.world.finish_lines)
    if (not surfaces or any(p.normal != UP or p.origin.y != surfaces[0].origin.y for p in surfaces)):
        blockers.append("noncoplanar_or_sloped_support_unregistered")
    if any((g.volume.orientation.rotate(Vec3(0.,0.,1.))-Vec3(0.,0.,1.)).magnitude > 1e-10 for g in gates):
        blockers.append("graph_candidate_requires_positive_z_gates")
    limit = tan(parameters.max_steering_angle)/parameters.wheelbase
    maximum = max(10/sqrt(3)*abs(b[0]-a[0])/(b[1]-a[1])**2 for a,b in zip(knots,knots[1:]))
    if maximum > limit:
        blockers.append("steering_curvature_bound_exceeded")
    if not isfinite(maximum):
        raise ValueError("knots exceed finite derivative domain")
    # All graph-segment endpoint tangents are exactly +Z. A partial
    # center crossing cannot claim complete body passage at either end.
    if gates and knots[0][1]+BODY.length_m/2 >= gates[0].volume.center.z:
        blockers.append("body_not_fully_before_first_gate")
    if gates and knots[-1][1]-BODY.length_m/2 <= gates[-1].volume.center.z:
        blockers.append("body_does_not_clear_finish_plane")
    height = surfaces[0].origin.y+parameters.ride_height if surfaces else parameters.ride_height
    checks = 0
    if not blockers:
        for first, last in zip(knots, knots[1:]):
            dx, dz = abs(last[0]-first[0]), last[1]-first[1]
            slope = 15/8*dx/dz
            second = 10/sqrt(3)*dx/dz**2
            count = ceil(dz/interval_m)
            half = dz/count/2
            inflation = (sqrt(1+slope*slope)+BODY.motion_radius_m*second)*half+1e-9
            enclosing = VehicleBodyGeometry(BODY.width_m+2*inflation, BODY.height_m,
                                           BODY.length_m+2*inflation, BODY.local_center)
            for i in range(count):
                pose = _state(first,last,first[1]+(i+.5)*dz/count,height)
                road = swept_body_road_union(pose,pose,enclosing,surfaces,
                                             ride_height_m=parameters.ride_height)
                if not road.valid:
                    blockers.append(f"continuous_body_road:{road.status}")
                    break
                box = enclosing.box_at(pose.position,pose.orientation)
                if any(obb_separation_lower_bound(box,o.volume) <= 1e-9 for o in asset.world.obstacles):
                    blockers.append("continuous_body_obstacle_unresolved")
                    break
                axes = tuple(box.orientation.rotate(v) for v in (Vec3(1.,0.,0.),UP,Vec3(0.,0.,1.)))
                extents = (box.half_extents.x,box.half_extents.y,box.half_extents.z)
                for gate in gates:
                    # Enclose the entire body while any part could intersect
                    # the gate plane, not just its center at a sampled crossing.
                    delta = box.center-gate.volume.center
                    if abs(delta.z) > sum(e*abs(a.z) for e,a in zip(extents,axes)):
                        continue
                    for axis,half_aperture in ((gate.volume.orientation.rotate(Vec3(1.,0.,0.)),gate.volume.half_extents.x),
                                               (gate.volume.orientation.rotate(UP),gate.volume.half_extents.y)):
                        radius = sum(e*abs(a.dot(axis)) for e,a in zip(extents,axes))
                        if abs(delta.dot(axis))+radius > half_aperture+1e-9:
                            blockers.append(f"continuous_body_aperture:{gate.id}")
                checks += 1
            if blockers:
                break
    crossed = []
    cursor = knots[0][1]
    for gate in gates:
        z = gate.volume.center.z
        if not cursor < z < knots[-1][1]:
            blockers.append(f"directed_gate_not_ordered:{gate.id}")
            continue
        first,last = next((a,b) for a,b in zip(knots,knots[1:]) if a[1] <= z <= b[1])
        pose = _state(first,last,z,height)
        delta = pose.position-gate.volume.center
        if (abs(delta.dot(gate.volume.orientation.rotate(Vec3(1.,0.,0.)))) > gate.volume.half_extents.x
                or abs(delta.dot(gate.volume.orientation.rotate(UP))) > gate.volume.half_extents.y):
            blockers.append(f"directed_center_misses_aperture:{gate.id}")
        else:
            crossed.append(gate.id)
            cursor = z
    result = {
        "schema_version": "wp7_planar_trajectory_certificate_v1", "track_id": asset.track_id,
        "world_sha256": fingerprint_track_world(asset.world), "body": asdict(BODY),
        "vehicle_parameters": asdict(parameters), "knots_xz_m": knots,
        "curve": "quintic_smoothstep_x_of_z_c2_v1", "interval_m": interval_m,
        "intervals_certified": checks, "curvature_upper_bound_per_m": maximum,
        "steering_curvature_limit_per_m": limit, "directed_gate_ids": crossed,
        "geometry_status": "unresolved" if blockers else "certified_continuous_planar_path",
        "speed_contact_status": "unresolved", "training_authorized": False,
        "blockers": sorted(set(blockers)),
        "remaining_requirements": ["speed_braking_and_boost_schedule", "executed_physics_tracking_and_contact",
                                   "initial_spawn_heading_alignment", "owner_and_scientific_admission"],
    }
    return {**result,"sha256":digest(result)}


def search_planar_trajectory(asset, *, parameters=PARAMETERS):
    """Bounded deterministic candidate search; no assets rewritten or PPO run."""
    if asset.track_id not in TRAIN_IDS:
        raise ValueError("only four registered TRAIN tracks supported")
    if asset.track_id == "slalom-v1":
        candidates = (
            ((0.,4.),(-3.,25.),(-3.,43.),(3.,61.),(3.,70.),(-3.,85.),(1.,98.5),(-3.,110.),(-3.,114.),(0.,127.)),
            ((0.,4.),(-3.,25.),(-3.,42.),(3.,61.),(3.,70.),(-3.,85.),(1.1,98.5),(-3.,110.),(-3.,114.),(0.,127.)),
        )
    elif asset.track_id == "straight-sprint-v1":
        candidates = (((0.,4.),(0.,116.)),)
    elif asset.track_id == "right-turn-v1":
        # Explore a diagonal alternative, without falsely inferring that a
        # failed z-monotone family proves infeasibility of all planar curves.
        candidates = (((0.,4.),(0.,35.),(0.,62.),(20.,70.),(64.,74.)),)
    else:
        candidates = (((0.,4.),(0.,155.)),)
    attempts = []
    for knots in candidates:
        result = certify_planar_trajectory(asset,knots,parameters=parameters)
        attempts.append({"certificate_sha256":result["sha256"],"blockers":result["blockers"]})
        if result["geometry_status"] == "certified_continuous_planar_path":
            break
    document = {k:v for k,v in result.items() if k != "sha256"}
    document.update(search_attempts=attempts,search_scope="finite_graph_family_not_infeasibility_proof")
    return {**document,"sha256":digest(document)}
