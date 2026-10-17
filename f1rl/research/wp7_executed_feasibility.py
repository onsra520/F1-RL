"""Executable TRAIN-only numerical witnesses; never learned-policy evidence.

Quintic references are C2; arc references are C1 with bounded piecewise curvature.
Executed collision/contact and gate proofs cover numerical substep chords/SLERP, not a physical
continuous-force ODE or a tracking-error bound for that different trajectory.
"""
from dataclasses import asdict, replace
from hashlib import sha256
from math import acos, atan, atan2, cos, isfinite, pi, sin
from pathlib import Path
import json
import subprocess
import sys
import tempfile

from ..learning.episode import RaceProgressState, RaceRoute
from ..learning.pilot_preflight import TRAIN_IDS
from ..learning.provenance import _capture_runtime_provenance
from ..learning.route_navigation import AuthoredRoute
from ..learning.wp7_race import advance_directed_race
from ..learning.owner_acceptance import review_preregistered_contract
from ..runtime.simulation import Simulation
from ..sandbox.body_collision import _slerp
from ..sandbox.coupled_longitudinal_dynamics import LongitudinalContactPolicy
from ..sandbox.geometry.vector import Vec3
from ..sandbox.interactions.resolver import resolve_motion_modifiers
from ..sandbox.track.volume import OrientedBoxVolume
from ..sandbox.vehicle import VehicleControl
from ..track_assets import load_track_asset
from .wp7_planar_trajectory import search_planar_trajectory
from .wp7_arc_trajectory import search_right_turn_trajectory, ArcPath, _pieces, _advance
from .wp7_protocol import build_protocol, digest, load_registration, prepare_scenarios
from .wp7_track_feasibility import BODY

UP = Vec3(0.,1.,0.)
FORWARD = Vec3(0.,0.,1.)
CONTROLLER = dict(version='quintic_tracking_speed_stop_v1',speed_mps=3.,
                  heading_gain_per_m=.8,lateral_gain_per_m2=.3,brake_after_finish_m=2.5)
RAMP_CONTROLLER = dict(version='longitudinal_ramp_speed_stop_proposal_v1', horizontal_speed_mps=3.5,
                       brake_after_finish_m=2.5, max_abs_tilt_rad=.2, steering=0.)
ARC_CONTROLLER = dict(version='arc_center_gate_speed_stop_v1',speed_mps=3.,
                      heading_gain_per_m=.8,lateral_gain_per_m2=.3,brake_after_finish_m=1.)
_LOADED_SOURCE_SHA256 = sha256(Path(__file__).read_bytes()).hexdigest()


def _execution_sources(root):
    files = [*root.joinpath('f1rl').rglob('*.py'),*root.joinpath('scripts').glob('*.py'),
             *root.joinpath('tests').rglob('*.py'),*root.joinpath('configs').rglob('*.yml'),
             *root.joinpath('configs').rglob('*.yaml'),*root.joinpath('configs').rglob('*.json'),
             root/'pyproject.toml',root/'assets/tracks/manifest.json',
             *(root/f'assets/tracks/training/{identity}.json' for identity in TRAIN_IDS)]
    if any(p.is_symlink() or not p.resolve(strict=True).is_relative_to(root) for p in files):
        raise ValueError('execution source escapes registered root')
    return {p.relative_to(root).as_posix():sha256(p.read_bytes()).hexdigest() for p in sorted(files)}


def _verify_module_origins(root):
    for name,module in tuple(sys.modules.items()):
        if name=='f1rl' or name.startswith('f1rl.'):
            path = getattr(module,'__file__',None)
            if path is not None and not Path(path).resolve().is_relative_to(root):
                raise ValueError('worker imported foreign F1RL source')


def _validated_controls(controls):
    if controls is None:
        return None
    if (not isinstance(controls,(tuple,list)) or not 1<=len(controls)<=500
            or any(not isinstance(c,(tuple,list)) or len(c)!=3
                   or any(type(v) not in (int,float) or not isfinite(v) for v in c) for c in controls)):
        raise ValueError('bounded finite three-action control tape required')
    return tuple(VehicleControl(*c) for c in controls)


def certify_planar_execution(repository_root,track_id,*,controls=None):
    return _certify_execution(repository_root, track_id, controls=controls, ramp_profile=False)


def certify_ramp_execution(repository_root,*,controls=None):
    """Nominal explicit engineering proposal; not the registered PPO profile."""
    return _certify_execution(repository_root, 'ramp-v1', controls=controls, ramp_profile=True)


def certify_coupled_ramp_execution(repository_root,*,controls=None):
    """Unadmitted planar force/work proposal on the actual TRAIN Ramp."""
    return _certify_execution(repository_root,'ramp-v1',controls=controls,ramp_profile=True,coupled_profile=True)


def certify_registered_center_execution(repository_root,track_id,*,controls=None):
    """Registered physics and owner 5A center gates; never train permission."""
    return _certify_execution(repository_root,track_id,controls=controls,ramp_profile=False,center_gates=True)


def _certify_execution(repository_root,track_id,*,controls,ramp_profile,coupled_profile=False,center_gates=False):
    """Run fresh exact-root code; parent module state cannot mint a source pin."""
    if track_id not in TRAIN_IDS:
        raise ValueError('registered TRAIN identity required')
    validated = _validated_controls(controls)
    if ramp_profile and validated is not None and any(c.steering != 0. for c in validated):
        raise ValueError('ramp proposal requires zero steering')
    root = Path(repository_root).resolve(strict=True)
    if (root!=Path(__file__).resolve().parents[2]
            or sha256(Path(__file__).read_bytes()).hexdigest()!=_LOADED_SOURCE_SHA256):
        raise ValueError('execution source origin or loaded wrapper bytes differ')
    source = _execution_sources(root)
    payload = json.dumps(dict(track_id=track_id,controls=controls,source=source,ramp_profile=ramp_profile,
                              coupled_profile=coupled_profile,center_gates=center_gates),allow_nan=False)
    bootstrap = ('import sys;sys.path.insert(0,sys.argv[1]);'
                 'from f1rl.research.wp7_executed_feasibility import _worker;_worker()')
    # A fresh cache prefix prevents stale checkout pyc files from supplying code.
    with tempfile.TemporaryDirectory(prefix='wp7-execution-') as cache:
        completed = subprocess.run([sys.executable,'-I','-B','-X',f'pycache_prefix={cache}',
            '-c',bootstrap,str(root)],cwd=root,input=payload,text=True,capture_output=True,timeout=240)
    if completed.returncode!=0:
        raise RuntimeError(f'isolated feasibility worker failed: {completed.stderr[-1600:]}')
    result = json.loads(completed.stdout)
    if (_execution_sources(root)!=source or result.get('execution_source_sha256')!=digest(source)
            or result.get('sha256')!=digest({k:v for k,v in result.items() if k!='sha256'})):
        raise ValueError('execution source or returned certificate identity changed')
    return result


def _worker():
    """Internal JSON-only worker, never a model/unpickle or training entry point."""
    payload = json.loads(sys.stdin.read())
    root = Path(__file__).resolve().parents[2]
    if root!=Path.cwd() or _execution_sources(root)!=payload['source']:
        raise ValueError('worker source origin or import-time source bytes differ')
    _verify_module_origins(root)
    runtime = asdict(_capture_runtime_provenance())
    result = _execute_planar(root,payload['track_id'],controls=payload['controls'],ramp_profile=payload['ramp_profile'],
                             coupled_profile=payload['coupled_profile'],center_gates=payload['center_gates'])
    _verify_module_origins(root)
    if (_execution_sources(root)!=payload['source']
            or build_protocol(root)['sha256']!=result['protocol_sha256']
            or asdict(_capture_runtime_provenance())!=runtime):
        raise ValueError('source/configuration changed during feasibility execution')
    result.pop('sha256')
    result['execution_source_sha256'] = digest(payload['source'])
    result['execution_runtime'] = runtime
    result['sha256'] = digest(result)
    print(json.dumps(result,allow_nan=False))


def _curve(knots,z):
    first,last = next(((a,b) for a,b in zip(knots,knots[1:]) if a[1]<=z<=b[1]),
                      (knots[0],knots[1]) if z<knots[0][1] else (knots[-2],knots[-1]))
    x0,z0 = first
    x1,z1 = last
    u = max(0.,min(1.,(z-z0)/(z1-z0)))
    x = x0+(x1-x0)*(10*u**3-15*u**4+6*u**5)
    slope = (x1-x0)/(z1-z0)*30*u*u*(1-u)**2
    second = (x1-x0)/(z1-z0)**2*60*u*(1-u)*(1-2*u)
    return x,atan(slope),second/(1+slope*slope)**1.5


def _control(snapshot,factory,knots,finish_z):
    vehicle = snapshot.world.vehicle
    p = factory.vehicle_parameters
    dt = factory.simulation_config.fixed_dt
    forward = vehicle.orientation.rotate(FORWARD)
    speed = Vec3(vehicle.velocity.x,0.,vehicle.velocity.z).magnitude
    x,yaw,curvature = _curve(knots,vehicle.position.z+forward.z*speed*dt/2)
    heading_error = yaw-atan2(forward.x,forward.z)
    # These are proposal gains, not calibrated vehicle parameters or a policy.
    curvature += CONTROLLER['heading_gain_per_m']*heading_error
    curvature += CONTROLLER['lateral_gain_per_m2']*(x-vehicle.position.x)*cos(yaw)
    steering = max(-1.,min(1.,atan(p.wheelbase*curvature)/p.max_steering_angle))
    if vehicle.position.z>=finish_z+CONTROLLER['brake_after_finish_m']:
        return VehicleControl(steering,0.,1.)
    modifiers = resolve_motion_modifiers(snapshot.query)
    target_acceleration = (CONTROLLER['speed_mps']-speed)/dt
    drive = (target_acceleration+p.rolling_drag*modifiers.rolling_drag_multiplier*speed
             -modifiers.longitudinal_acceleration_bonus)
    return VehicleControl(steering,min(1.,max(0.,drive/p.max_acceleration)),
                          min(1.,max(0.,-drive/p.max_braking)))


def _ramp_control(snapshot,factory,finish_z):
    vehicle = snapshot.world.vehicle
    p = factory.vehicle_parameters
    contact = snapshot.query.contact
    if contact is None or abs(contact.normal.x)>1e-10:
        raise ValueError('ramp proposal requires longitudinal supported road')
    if vehicle.position.z>=finish_z+RAMP_CONTROLLER['brake_after_finish_m']:
        return VehicleControl(0.,0.,1.)
    forward = FORWARD.reject_from(contact.normal).normalized()
    speed = vehicle.velocity.reject_from(contact.normal).dot(forward)
    gravity = (UP*(-p.gravity)).reject_from(contact.normal).dot(forward)
    drive = ((RAMP_CONTROLLER['horizontal_speed_mps']/contact.normal.y-speed)/factory.simulation_config.fixed_dt
             +p.rolling_drag*speed-gravity)
    return VehicleControl(0.,min(1.,max(0.,drive/p.max_acceleration)),
                          min(1.,max(0.,-drive/p.max_braking)))


def _arc_control(snapshot,factory,pieces,finish_s):
    """Project onto line/circle pieces and track the bounded arc reference."""
    vehicle = snapshot.world.vehicle
    nearest = None
    for piece in pieces:
        start,x,z,yaw,length,k = piece
        if k == 0.:
            candidates = [(vehicle.position.x-x)*sin(yaw)+(vehicle.position.z-z)*cos(yaw)]
        else:
            cx,cz = x+cos(yaw)/k,z-sin(yaw)/k
            angle = atan2(k*(vehicle.position.z-cz),k*(cx-vehicle.position.x))
            winding = round((yaw-angle)/(2*pi))
            candidates = [(angle+2*pi*j-yaw)/k for j in (winding-1,winding,winding+1)]
        for candidate in candidates:
            local = max(0.,min(length,candidate))
            px,pz,_ = _advance(x,z,yaw,local,k)
            item = ((px-vehicle.position.x)**2+(pz-vehicle.position.z)**2,start+local)
            if nearest is None or item<nearest:
                nearest = item
    distance = nearest[1]
    parameters = factory.vehicle_parameters
    speed = Vec3(vehicle.velocity.x,0.,vehicle.velocity.z).magnitude
    lookahead = min(pieces[-1][0]+pieces[-1][4],distance+speed*factory.simulation_config.fixed_dt/2.)
    piece = next(p for p in pieces if p[0]<=lookahead<=p[0]+p[4])
    _,x,z,yaw,_,curvature = piece
    x,z,yaw = _advance(x,z,yaw,lookahead-piece[0],curvature)
    forward = vehicle.orientation.rotate(FORWARD)
    difference = yaw-atan2(forward.x,forward.z)
    curvature += ARC_CONTROLLER['heading_gain_per_m']*atan2(sin(difference),cos(difference))
    curvature += ARC_CONTROLLER['lateral_gain_per_m2']*((x-vehicle.position.x)*cos(yaw)-(z-vehicle.position.z)*sin(yaw))
    steering = max(-1.,min(1.,atan(parameters.wheelbase*curvature)/parameters.max_steering_angle))
    if distance>=finish_s+ARC_CONTROLLER['brake_after_finish_m']:
        return VehicleControl(steering,0.,1.)
    modifiers = resolve_motion_modifiers(snapshot.query)
    drive = ((ARC_CONTROLLER['speed_mps']-speed)/factory.simulation_config.fixed_dt
             +parameters.rolling_drag*modifiers.rolling_drag_multiplier*speed
             -modifiers.longitudinal_acceleration_bonus)
    return VehicleControl(steering,min(1.,max(0.,drive/parameters.max_acceleration)),
                          min(1.,max(0.,-drive/parameters.max_braking)))


def _plane_range(box,gate):
    normal = gate.orientation.rotate(FORWARD)
    axes = tuple(box.orientation.rotate(v) for v in (Vec3(1.,0.,0.),UP,FORWARD))
    extents = (box.half_extents.x,box.half_extents.y,box.half_extents.z)
    distance = (box.center-gate.center).dot(normal)
    radius = sum(e*abs(a.dot(normal)) for e,a in zip(extents,axes))
    return distance-radius,distance+radius


def _gate_segment_clear(first,last,gate):
    """Enclose every moving body point, then bound all possible plane sections."""
    a,b = first.orientation,last.orientation
    dot = min(1.,abs(a.w*b.w+a.x*b.x+a.y*b.y+a.z*b.z))
    angle = 2*acos(dot)
    # Full angle covers the tiny-rotation branch of the shared SLERP helper too.
    radius = (last.position-first.position).magnitude/2+BODY.motion_radius_m*angle+1e-8
    position = (first.position+last.position)*.5
    box = BODY.box_at(position,_slerp(a,b,.5))
    box = OrientedBoxVolume(box.center,box.half_extents+Vec3(radius,radius,radius),box.orientation)
    lo,hi = _plane_range(box,gate)
    if lo>0 or hi<0:
        return True
    axes = tuple(box.orientation.rotate(v) for v in (Vec3(1.,0.,0.),UP,FORWARD))
    extents = (box.half_extents.x,box.half_extents.y,box.half_extents.z)
    for local,limit in ((Vec3(1.,0.,0.),gate.half_extents.x),(UP,gate.half_extents.y)):
        axis = gate.orientation.rotate(local)
        bound = abs((box.center-gate.center).dot(axis))+sum(e*abs(a.dot(axis)) for e,a in zip(extents,axes))
        if bound>limit-1e-8:
            return False
    return True


def _execute_planar(repository_root,track_id,*,controls=None,ramp_profile=False,coupled_profile=False,center_gates=False):
    """Execute a reference controller or an exact control tape on a real TRAIN asset.

    Nominal spawn only, no RNG/policy/model, no redraw or asset changes. Execution
    continues after centre finish until whole-body passage and a verified stop.
    This is a Simulation feasibility witness, not a Gym episode-success metric.
    """
    if track_id not in TRAIN_IDS:
        raise ValueError('registered TRAIN identity required')
    controls = _validated_controls(controls)
    root = Path(repository_root).resolve(strict=True)
    protocol = build_protocol(root)
    if center_gates:
        review_preregistered_contract(root/'configs/recovery/wp7-owner-decisions-preregistered-v1.yml')
    factories = prepare_scenarios(root,load_registration(root),seed_index=0).scenarios
    factory = next(s.factory for s in factories if s.track_id==track_id)
    registered_physics_sha256 = factory.physics_config.sha256
    if ramp_profile:
        if track_id != 'ramp-v1':
            raise ValueError('ramp proposal requires the actual TRAIN ramp')
        suspension = replace(factory.physics_config.independent_suspension_policy,
            force_model='road_relative_vertical_v2',max_abs_tilt_rad=RAMP_CONTROLLER['max_abs_tilt_rad'])
        factory = replace(factory,physics_config=replace(factory.physics_config,
            independent_suspension_policy=suspension,
            ground_motion_version='coupled_longitudinal_v1' if coupled_profile else 'legacy_euler_v1',
            longitudinal_contact_policy=LongitudinalContactPolicy(.8,100000.) if coupled_profile else None))
    asset = load_track_asset(root/f'assets/tracks/training/{track_id}.json')
    expected = next(t for t in protocol['tracks'] if t['track_id']==track_id)
    if asset.world_sha256!=expected['world_sha256'] or asset.content_sha256!=expected['content_sha256']:
        raise ValueError('actual TRAIN identity differs from registered source')
    arc_reference = center_gates and track_id=='right-turn-v1'
    geometry = (search_right_turn_trajectory(asset,parameters=factory.vehicle_parameters,
                gate_contract='directed_center_crossing_v1') if arc_reference else
                search_planar_trajectory(asset,parameters=factory.vehicle_parameters))
    gates = (*sorted(factory.track.checkpoints,key=lambda g:g.order),*factory.track.finish_lines)
    result = dict(schema_version='wp7_executed_planar_feasibility_v1',track_id=track_id,
        protocol_sha256=protocol['sha256'],world_sha256=asset.world_sha256,content_sha256=asset.content_sha256,
        physics_sha256=factory.physics_config.sha256,physics=factory.physics_config.document(),
        initial_state_sha256=expected['initial_state_sha256'],vehicle_parameters=asdict(factory.vehicle_parameters),
        simulation=asdict(factory.simulation_config),race_sha256=expected['race_sha256'],
        controller=dict(CONTROLLER),reference_geometry_sha256=geometry['sha256'],
        scope='nominal_control_tape_registered_numerical_model_not_physical_or_learned_policy_validation',
        status='unresolved',blockers=[],controls=[],step_count=0,proof_substeps=0,
        gate_ids=[],expected_gate_ids=[g.id for g in gates],full_body_gate_passages=[],
        max_speed_mps=0.,max_sampled_route_deviation_m=0.,finished=False,stopped=False,
        training_authorized=False,scientific_evidence=False,trajectory_sha256=digest([]))
    if center_gates:
        result.pop('full_body_gate_passages')
        result.update(schema_version='wp7_executed_registered_center_feasibility_v1',
            gate_contract='directed_center_crossing_v1',whole_body_gate_aperture_required=False,
            owner_contract_sha256=sha256((root/'configs/recovery/wp7-owner-decisions-preregistered-v1.yml').read_bytes()).hexdigest(),
            scope='registered_numerical_full_body_road_obstacle_contact_and_owner_directed_center_gates_not_physical_or_PPO_validation')
        if arc_reference:
            result['controller'] = dict(ARC_CONTROLLER)
    if ramp_profile:
        result.update(schema_version='wp7_executed_ramp_proposal_v1',
            profile_registration_status='engineering_proposal_not_training_registration',
            registered_physics_sha256=registered_physics_sha256,controller=dict(RAMP_CONTROLLER),
            scope='nominal_grounded_reduced_suspension_and_legacy_longitudinal_proposal_not_registered_PPO_or_physical_validation',
            reference_geometry_sha256=digest(dict(navigation=asdict(factory.navigation),horizontal_curvature_per_m=0.)))
    if coupled_profile:
        result.update(schema_version='wp7_executed_coupled_ramp_proposal_v1',
            scope='nominal_coupled_world_Z_heave_pitch_numerical_proposal_not_registered_PPO_or_measured_physics',
            coupled_work_audit=dict(accepted_substeps=0,actuator_work_j=0.,dissipation_j=0.,
                energy_residual_j=0.,max_energy_residual_fraction=0.,
                max_linear_impulse_residual_ns=0.,max_angular_impulse_residual_nms=0.))
    if not ramp_profile and geometry['geometry_status']!='certified_continuous_planar_path':
        result['blockers'].append('reference_geometry_unresolved')
    if arc_reference and geometry['route_corridor']['status']!='clear':
        result['blockers'].append('reference_hard_4m_corridor_unresolved')
    if track_id=='right-turn-v1' and not center_gates:
        result['blockers'].append('reference_hard_4m_corridor_violation_no_executable_graph')
    if result['blockers']:
        return {**result,'sha256':digest(result)}
    sim = Simulation(factory.track,factory.initial_state,factory.vehicle_parameters,
                     factory.simulation_config,physics_config=factory.physics_config)
    route = RaceRoute.from_track(factory.track)
    authored = AuthoredRoute(factory.navigation)
    progress = RaceProgressState(0,False)
    if arc_reference:
        path = geometry['path']
        pieces = _pieces(ArcPath(tuple(path['start_xz_m']),path['start_yaw_rad'],
                                tuple(tuple(s) for s in path['segments_length_curvature'])))
    initial = sim.snapshot.world.vehicle
    if any((initial.position-g.volume.center).dot(g.volume.orientation.rotate(FORWARD))>=0
           if center_gates else _plane_range(BODY.box_at(initial.position,initial.orientation),g.volume)[1]>=0 for g in gates):
        result['blockers'].append('initial_center_not_before_gates' if center_gates else 'initial_body_not_before_gates')
    else:
        limit = len(controls) if controls is not None else min(500,factory.episode_config.max_steps)
        for tick in range(limit):
            control = controls[tick] if controls is not None else (
                _arc_control(sim.snapshot,factory,pieces,geometry['gate_passages'][-1]['crossing_s_m']) if arc_reference else
                _ramp_control(sim.snapshot,factory,gates[-1].volume.center.z) if ramp_profile else
                _control(sim.snapshot,factory,geometry['knots_xz_m'],gates[-1].volume.center.z))
            result['controls'].append([control.steering,control.throttle,control.brake])
            try:
                snapshot = sim.step(control)
                poses = snapshot.suspension_telemetry.poses
                if any(not p.grounded for p in poses):
                    raise RuntimeError('executed planar witness cannot include airborne states')
                race = advance_directed_race(factory.track,route,authored,progress,snapshot,factory.race_config)
                if not center_gates:
                    for first,last in zip(poses,poses[1:]):
                        if any(not _gate_segment_clear(first,last,g.volume) for g in gates):
                            raise RuntimeError('continuous executed body aperture unresolved')
                progress = race.progress.state
                result['gate_ids'].extend(race.directed_crossings)
                result['max_sampled_route_deviation_m'] = max(result['max_sampled_route_deviation_m'],race.max_sampled_deviation_m)
                peak = max(p.speed for p in poses)
                if peak>factory.vehicle_parameters.max_ground_speed+1e-8:
                    raise RuntimeError('executed speed exceeds registered limit')
                result['max_speed_mps'] = max(result['max_speed_mps'],peak)
                result['proof_substeps'] += len(poses)-1
                result['step_count'] += 1
                result['trajectory_sha256'] = digest([result['trajectory_sha256'],
                    json.loads(json.dumps(asdict(snapshot.world),default=sorted)),
                    asdict(snapshot.physics_transition),[asdict(p) for p in poses]])
                if coupled_profile:
                    audit = snapshot.coupled_ground_telemetry
                    result['trajectory_sha256'] = digest([result['trajectory_sha256'],asdict(audit)])
                    summary = result['coupled_work_audit']
                    summary['accepted_substeps'] += len(audit.substeps)
                    for name in ('actuator_work_j','dissipation_j','energy_residual_j'):
                        summary[name] += getattr(audit,name)
                    for name in ('max_linear_impulse_residual_ns','max_angular_impulse_residual_nms'):
                        summary[name] = max(summary[name],getattr(audit,name))
                    summary['max_energy_residual_fraction'] = max(summary['max_energy_residual_fraction'],
                        *(abs(s.energy_residual_j)/s.energy_tolerance_j for s in audit.substeps))
            except (ValueError,RuntimeError) as error:
                result['blockers'].append(f'{type(error).__name__}:{error}')
                break
            result['finished'] = progress.finished
            result['stopped'] = snapshot.world.vehicle.speed<=1e-8
            if controls is None and progress.finished and result['stopped']:
                break
    final = sim.snapshot.world.vehicle
    result['final_vehicle'] = asdict(final)
    result['elapsed_time_s'] = sim.snapshot.elapsed_time
    if not center_gates:
        result['full_body_gate_passages'] = [] if result['blockers'] else [g.id for g in gates
            if _plane_range(BODY.box_at(final.position,final.orientation),g.volume)[0]>0]
    if not result['finished'] or result['gate_ids']!=result['expected_gate_ids']:
        result['blockers'].append('ordered_finish_not_executed')
    if not center_gates and result['full_body_gate_passages']!=result['expected_gate_ids']:
        result['blockers'].append('whole_body_finish_passage_incomplete')
    if not result['stopped']:
        result['blockers'].append('terminal_braking_stop_incomplete')
    if not result['blockers']:
        result['status'] = ('executable_numerical_registered_center_path' if center_gates else
                            'executable_numerical_ramp_path' if ramp_profile else 'executable_numerical_planar_path')
    return {**result,'sha256':digest(result)}


def replay_planar_execution(repository_root,document,*,expected_sha256):
    """Re-execute every control; a rehashed self-assertion cannot replace the pin."""
    if (not isinstance(document,dict) or document.get('sha256')!=expected_sha256
            or digest({k:v for k,v in document.items() if k!='sha256'})!=expected_sha256):
        raise ValueError('independent certificate identity differs')
    if document.get('schema_version') == 'wp7_executed_registered_center_feasibility_v1':
        restored = certify_registered_center_execution(repository_root,document.get('track_id'),controls=document.get('controls'))
    elif document.get('schema_version') == 'wp7_executed_coupled_ramp_proposal_v1':
        restored = certify_coupled_ramp_execution(repository_root,controls=document.get('controls'))
    elif document.get('schema_version') == 'wp7_executed_ramp_proposal_v1':
        restored = certify_ramp_execution(repository_root,controls=document.get('controls'))
    else:
        restored = certify_planar_execution(repository_root,document.get('track_id'),controls=document.get('controls'))
    if restored!=document:
        raise ValueError('executed source, controls or physical evidence identity differs')
    return restored
