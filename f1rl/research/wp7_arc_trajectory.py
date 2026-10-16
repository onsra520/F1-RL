"""C1 arc-length planar paths with bounded piecewise-constant curvature.

Every continuous body pose is enclosed, including between proof intervals.
Gate passage checks the plane section of the body, not its whole projection.
This is geometry/steering evidence, never speed, contact, learning or admission.
"""
from dataclasses import asdict, dataclass
from math import acos, asin, atan2, ceil, cos, floor, isfinite, pi, sin, tan

from ..learning.pilot_preflight import TRAIN_IDS
from ..learning.route_navigation import AuthoredRoute
from ..sandbox.body_collision import VehicleBodyGeometry, obb_separation_lower_bound
from ..sandbox.geometry.rotation import Quaternion
from ..sandbox.geometry.vector import Vec3
from ..sandbox.road_footprint import swept_body_road_union
from ..sandbox.vehicle import VehicleState, VehicleParameters
from ..track_assets.canonical import fingerprint_track_world
from .wp7_protocol import digest
from .wp7_track_feasibility import BODY, PARAMETERS

UP = Vec3(0.,1.,0.)
ZERO = Vec3(0.,0.,0.)
EPS = 1e-8


@dataclass(frozen=True, slots=True)
class ArcPath:
    start_xz_m: tuple[float,float]
    start_yaw_rad: float
    segments_length_curvature: tuple[tuple[float,float],...]

    def __post_init__(self):
        def finite(value):
            return type(value) in (int,float) and isfinite(value)
        segments = self.segments_length_curvature
        if (not isinstance(self.start_xz_m,tuple) or len(self.start_xz_m) != 2
                or any(not finite(v) or abs(v)>1000 for v in self.start_xz_m)
                or not finite(self.start_yaw_rad) or abs(self.start_yaw_rad)>2*pi
                or not isinstance(segments,tuple) or not 1 <= len(segments) <= 64
                or any(not isinstance(row,tuple) or len(row)!=2
                       or not all(finite(v) for v in row) or not 1e-6 <= row[0] <= 1000
                       or abs(row[1])>5 or (row[1]!=0 and abs(row[1])<1e-8)
                       or abs(row[0]*row[1])>pi for row in segments)
                or sum(row[0] for row in segments)>1000):
            raise ValueError('invalid bounded arc path')


def _advance(x,z,yaw,length,curvature):
    half = curvature*length/2
    sinc = sin(half)/half if half else 1.
    return (x+length*sin(yaw+half)*sinc,
            z+length*cos(yaw+half)*sinc, yaw+2*half)


def _pieces(path):
    x,z = path.start_xz_m
    yaw,s = path.start_yaw_rad,0.
    result = []
    for length,k in path.segments_length_curvature:
        result.append((s,x,z,yaw,length,k))
        x,z,yaw = _advance(x,z,yaw,length,k)
        s += length
    return result


def _pose(piece, local_s, height):
    _,x,z,yaw,_,k = piece
    x,z,yaw = _advance(x,z,yaw,local_s,k)
    return VehicleState(Vec3(x,height,z),ZERO,Quaternion.from_axis_angle(UP,yaw),ZERO,True)


def _path_pose(pieces,s,height):
    for piece in pieces:
        if piece[0]<=s<=piece[0]+piece[4]:
            return _pose(piece,s-piece[0],height)
    raise ValueError('path coordinate outside program')


def _enclosure(piece,a,b,height):
    pose = _pose(piece,(a+b)/2,height)
    inflation = (1+BODY.motion_radius_m*abs(piece[5]))*(b-a)/2+EPS
    body = VehicleBodyGeometry(BODY.width_m+2*inflation,BODY.height_m,
                               BODY.length_m+2*inflation,BODY.local_center)
    return pose,body,body.box_at(pose.position,pose.orientation)


def _plane_section(box,gate):
    """Exact rectangle/plane section for horizontal yaw-only boxes and gates."""
    right,forward = (box.orientation.rotate(v) for v in (Vec3(1.,0.,0.),Vec3(0.,0.,1.)))
    tangent,normal = (gate.orientation.rotate(v) for v in (Vec3(1.,0.,0.),Vec3(0.,0.,1.)))
    vertices = []
    for a,b in ((-1,-1),(1,-1),(1,1),(-1,1)):
        delta = box.center+right*(a*box.half_extents.x)+forward*(b*box.half_extents.z)-gate.center
        vertices.append((delta.dot(tangent),delta.dot(normal)))
    if min(v[1] for v in vertices)>0 or max(v[1] for v in vertices)<0:
        return None
    section = []
    for (xa,da),(xb,db) in zip(vertices,vertices[1:]+vertices[:1]):
        if da==0:
            section.append(xa)
        if da*db<0:
            section.append(xa-da*(xb-xa)/(db-da))
    if not section:
        return None
    return min(section),max(section)


def _roots(pieces,gate,height):
    normal = gate.orientation.rotate(Vec3(0.,0.,1.))
    tangent = gate.orientation.rotate(Vec3(1.,0.,0.))
    roots = []
    for piece in pieces:
        s,x,z,yaw,length,k = piece
        if k==0:
            rate = normal.x*sin(yaw)+normal.z*cos(yaw)
            local = [(normal.dot(gate.center)-normal.x*x-normal.z*z)/rate] if rate>1e-10 else []
        else:
            cx,cz = x+cos(yaw)/k,z-sin(yaw)/k
            value = k*(normal.dot(gate.center)-normal.x*cx-normal.z*cz)
            local = []
            if -1 <= value <= 1:
                phi = atan2(normal.x,normal.z)
                low,high = sorted((yaw,yaw+k*length))
                for base in (phi+asin(value),phi+pi-asin(value)):
                    for winding in range(ceil((low-base)/(2*pi)),floor((high-base)/(2*pi))+1):
                        angle = base+2*pi*winding
                        if normal.x*sin(angle)+normal.z*cos(angle)>1e-10:
                            local.append((angle-yaw)/k)
        for distance in local:
            if not 0 <= distance <= length:
                continue
            pose = _pose(piece,distance,height)
            delta = pose.position-gate.center
            if (abs(delta.dot(normal)) <= EPS and abs(delta.dot(tangent)) <= gate.half_extents.x-EPS
                    and abs(delta.y) <= gate.half_extents.y-EPS):
                roots.append(s+distance)
    return sorted(set(roots))


def certify_arc_trajectory(asset,path,*,parameters=PARAMETERS,interval_m=.05,max_subdivisions=7):
    """Prove continuous road/obstacle clearance and ordered full-body passages.

    |dp/ds|=1 and |dyaw/ds|=|curvature|. A midpoint OBB enlarged by
    (1+motion_radius*|curvature|)*h/2 encloses all poses on its interval.
    Adaptive subdivisions improve sufficient bounds; exhaustion stays unresolved.
    """
    if asset.track_id not in TRAIN_IDS or type(path) is not ArcPath or not isinstance(parameters,VehicleParameters):
        raise ValueError('registered TRAIN identity and exact ArcPath required')
    if (type(interval_m) not in (int,float) or not isfinite(interval_m) or not .01<=interval_m<=.1
            or type(max_subdivisions) is not int or not 0<=max_subdivisions<=10):
        raise ValueError('invalid certification budget')
    surfaces = asset.world.surfaces
    gates = (*sorted(asset.world.checkpoints,key=lambda g:g.order),*asset.world.finish_lines)
    blockers = []
    if not surfaces or any(p.normal!=UP or p.origin.y!=surfaces[0].origin.y for p in surfaces):
        blockers.append('noncoplanar_or_sloped_support_unregistered')
    if any((g.volume.orientation.rotate(UP)-UP).magnitude>1e-10 for g in gates):
        blockers.append('nonvertical_gate_plane_unregistered')
    limit = tan(parameters.max_steering_angle)/parameters.wheelbase
    if not isfinite(limit):
        raise ValueError('derived steering curvature exceeds finite numeric domain')
    maximum = max(abs(k) for _,k in path.segments_length_curvature)
    if maximum>limit:
        blockers.append('steering_curvature_bound_exceeded')
    pieces = _pieces(path)
    height = (surfaces[0].origin.y if surfaces else 0)+parameters.ride_height
    intervals = []
    checks = 0
    passages = []

    def prove(piece,a,b,gate=None,depth=0):
        nonlocal checks
        checks += 1
        if checks>200000:
            return False
        pose,body,box = _enclosure(piece,a,b,height)
        if gate is None:
            ok = swept_body_road_union(pose,pose,body,surfaces,ride_height_m=parameters.ride_height).valid
            ok &= all(obb_separation_lower_bound(box,o.volume)>EPS for o in asset.world.obstacles)
        else:
            section = _plane_section(box,gate)
            ok = section is None or (
                max(abs(v) for v in section) <= gate.half_extents.x-EPS
                and abs(box.center.y-gate.center.y)+box.half_extents.y <= gate.half_extents.y-EPS)
        if ok:
            return True
        return depth<max_subdivisions and prove(piece,a,(a+b)/2,gate,depth+1) and prove(piece,(a+b)/2,b,gate,depth+1)

    if not blockers:
        for piece in pieces:
            count = ceil(piece[4]/interval_m)
            for index in range(count):
                a,b = index*piece[4]/count,(index+1)*piece[4]/count
                intervals.append((piece,a,b,_enclosure(piece,a,b,height)[2]))
                if not prove(piece,a,b):
                    blockers.append('continuous_body_road_or_obstacle_unresolved')
                    break
            if blockers:
                break
    if not blockers:
        cursor = -EPS
        for gate in gates:
            accepted = None
            for root in _roots(pieces,gate.volume,height):
                bracket = (max(0.,root-1e-4),min(pieces[-1][0]+pieces[-1][4],root+1e-4))
                if bracket[0]<=cursor+EPS:
                    continue
                normal = gate.volume.orientation.rotate(Vec3(0.,0.,1.))
                before = (_path_pose(pieces,bracket[0],height).position-gate.volume.center).dot(normal)
                after = (_path_pose(pieces,bracket[1],height).position-gate.volume.center).dot(normal)
                if not before < -EPS or not after > EPS:
                    continue
                index = next((i for i,(p,a,b,_) in enumerate(intervals) if p[0]+a-EPS<=root<=p[0]+b+EPS),None)
                if index is None:
                    continue
                lo=hi=index
                while lo>0 and _plane_section(intervals[lo-1][3],gate.volume) is not None:
                    lo-=1
                while hi+1<len(intervals) and _plane_section(intervals[hi+1][3],gate.volume) is not None:
                    hi+=1
                first,a,_,_ = intervals[lo]
                last,_,b,_ = intervals[hi]
                def side(piece,local):
                    box = BODY.box_at(_pose(piece,local,height).position,_pose(piece,local,height).orientation)
                    radius = sum(e*abs(box.orientation.rotate(axis).dot(normal)) for e,axis in
                        ((box.half_extents.x,Vec3(1.,0.,0.)),(box.half_extents.z,Vec3(0.,0.,1.))))
                    distance = (box.center-gate.volume.center).dot(normal)
                    return distance-radius,distance+radius
                if side(first,a)[1]>=-EPS or side(last,b)[0]<=EPS:
                    continue
                if all(prove(p,a,b,gate.volume) for p,a,b,_ in intervals[lo:hi+1]):
                    accepted = dict(gate_id=gate.id,crossing_s_m=root,
                        center_crossing_bracket_s_m=bracket,
                        passage_start_s_m=first[0]+a,passage_end_s_m=last[0]+b,body_passage_certified=True)
                    break
            if accepted is None:
                blockers.append(f'full_body_passage_unresolved:{gate.id}')
                break
            passages.append(accepted)
            cursor = accepted['center_crossing_bracket_s_m'][1]
    route = AuthoredRoute(asset.navigation) if asset.navigation is not None else None
    route_status = 'clear' if route else 'missing_authored_route'
    witness = None
    max_upper = 0.
    if route:
        for piece in pieces:
            count = ceil(piece[4]/interval_m)
            for i in range(count):
                a,b = i*piece[4]/count,(i+1)*piece[4]/count
                position = _pose(piece,(a+b)/2,height).position
                distance = route.horizontal_distance(position)
                max_upper = max(max_upper,distance+(b-a)/2+EPS)
                if distance>4+EPS:
                    route_status = 'hard_violation'
                    if witness is None or distance>witness['deviation_m']:
                        witness = dict(path_s_m=piece[0]+(a+b)/2,position=asdict(position),deviation_m=distance)
                elif route_status=='clear' and distance+(b-a)/2+EPS>=4:
                    route_status = 'uncertain'
    execution_blockers = ['speed_braking_and_contact_unresolved','curvature_jump_steering_schedule_unverified']
    if blockers:
        execution_blockers.append('body_geometry_unresolved')
    if route_status!='clear':
        execution_blockers.append('hard_route_corridor_'+route_status)
    first = _pose(pieces[0],0.,height)
    aligned = ((first.position-Vec3(asset.spawn.position.x,height,asset.spawn.position.z)).magnitude<=EPS
        and (first.orientation.rotate(Vec3(0.,0.,1.))-asset.spawn.orientation.rotate(Vec3(0.,0.,1.))).magnitude<=EPS)
    if not aligned:
        execution_blockers.append('initial_spawn_pose_mismatch')
    document = dict(schema_version='wp7_arc_path_certificate_v1',track_id=asset.track_id,
        world_sha256=fingerprint_track_world(asset.world),body=asdict(BODY),vehicle_parameters=asdict(parameters),
        course_sha256=digest(dict(spawn=asdict(asset.spawn),navigation=asdict(asset.navigation) if asset.navigation else None)),
        path=asdict(path),curve='c1_piecewise_constant_curvature_arclength_v1',
        interval_m=interval_m,max_subdivisions=max_subdivisions,interval_checks=checks,
        curvature_upper_bound_per_m=maximum,steering_curvature_limit_per_m=limit,gate_passages=passages,
        geometry_status='unresolved' if blockers else 'certified_continuous_planar_path',
        geometry_scope='continuous_body_road_obstacle_and_gate_sections_not_route_or_dynamics',
        route_corridor=dict(status=route_status,hard_limit_m=4.,soft_limit_m=2.,
                            deviation_upper_bound_m=max_upper,witness=witness,
                            scope='provided_arc_path_only_not_general_infeasibility'),
        initial_spawn_xz_heading_aligned=aligned,executable_feasibility=False,
        execution_blockers=execution_blockers,
        speed_contact_status='unresolved',training_authorized=False,blockers=sorted(set(blockers)),
        remaining_requirements=['speed_braking_schedule','executed_physics_tracking_and_contact',
                                'curvature_jump_steering_schedule','owner_and_scientific_admission'])
    return {**document,'sha256':digest(document)}


def _right_turn_path(curvature,theta_degrees,straight_half):
    radius = 1/curvature
    theta = theta_degrees*pi/180
    height = 70-straight_half*cos(theta)-radius*(1-sin(theta))
    approach_x = 20-straight_half*sin(theta)-radius*cos(theta)
    after_height = 140-height
    cosine = 1-(after_height-height)/(2*radius)
    if not isfinite(cosine) or not -1<=cosine<=1:
        raise ValueError('Right-turn arc family exceeds supported numeric domain')
    bend = acos(cosine)
    after_x = 40-approach_x
    side_shift = 2*radius*sin(bend)
    return ArcPath((0.,4.),0.,(
        (height-radius-4,0.),(radius*pi/2,curvature),(approach_x-radius,0.),
        (radius*(pi/2-theta),-curvature),(2*straight_half,0.),
        (radius*(pi/2-theta),curvature),(radius*bend,curvature),(radius*bend,-curvature),
        (approach_x+44-after_x-side_shift,0.),(radius*(pi/2-theta),-curvature),
        (straight_half+3,0.)))


def search_right_turn_trajectory(asset,*,parameters=PARAMETERS):
    """Search a finite arc family, retaining every failed execution constraint."""
    if asset.track_id!='right-turn-v1' or not isinstance(parameters,VehicleParameters):
        raise ValueError('Right-turn TRAIN identity and vehicle parameters required')
    available = tan(parameters.max_steering_angle)/parameters.wheelbase
    if not isfinite(available) or available<=0:
        raise ValueError('Right-turn steering curvature outside finite positive numeric domain')
    curvature = min(available,tan(PARAMETERS.max_steering_angle)/PARAMETERS.wheelbase)
    if not isfinite(1/curvature):
        raise ValueError('Right-turn radius outside finite numeric domain')
    candidates = []
    for theta,straight_half in ((23.6,1.6),(24.,1.65),(25.,1.65)):
        try:
            path = _right_turn_path(curvature,theta,straight_half)
        except ValueError as exc:
            raise ValueError('Right-turn arc family outside supported numeric domain') from exc
        result = certify_arc_trajectory(asset,path,parameters=parameters)
        candidates.append(result)
        if result['geometry_status']=='certified_continuous_planar_path' and result['route_corridor']['status']=='clear':
            break
    def rank(result):
        return (result['geometry_status']!='certified_continuous_planar_path',
                result['route_corridor']['deviation_upper_bound_m'])
    best = min(candidates,key=rank)
    document = {k:v for k,v in best.items() if k!='sha256'}
    document.update(search_scope='finite_arc_family_not_general_infeasibility_proof',
        search_attempts=[dict(certificate_sha256=r['sha256'],geometry_status=r['geometry_status'],
                             route_corridor_status=r['route_corridor']['status'],
                             deviation_upper_bound_m=r['route_corridor']['deviation_upper_bound_m']) for r in candidates])
    return {**document,'sha256':digest(document)}
