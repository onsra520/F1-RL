"""TRAIN-only authored-path body audit; neither route search nor admission.

Certificates cover linear translation/constant orientation on each segment.
Heading junctions, sloped support and absent speed profiles remain explicit
blockers. A failed authored path is not proof that every possible path fails.
"""
from dataclasses import asdict
from hashlib import sha256
from math import acos, degrees, tan
from pathlib import Path

from ..rl.experiments.definition import ExperimentDefinition
from ..rl.pilot_preflight import TRAIN_IDS
from ..simulation.body_collision import VehicleBodyGeometry, sweep_body_against_obstacle
from ..simulation.directed_center_crossing import directed_center_gate_crossing
from ..simulation.geometry.rotation import Quaternion
from ..simulation.geometry.vector import Vec3
from ..simulation.road_footprint import swept_body_road_union
from ..simulation.vehicle import VehicleParameters, VehicleState
from ..tracks import load_track_asset, load_track_manifest
from ..tracks.canonical import fingerprint_track_world

BODY = VehicleBodyGeometry(1.8, 1.2, 4., Vec3(0., .3, 0.))
PARAMETERS = VehicleParameters(3., .5, .5, 6., 10., .4, .1, 15., 9.81)
UP = Vec3(0., 1., 0.)
ZERO = Vec3(0., 0., 0.)


def audit_track(asset, *, parameters=PARAMETERS):
    if asset.track_id not in TRAIN_IDS or asset.navigation is None:
        raise ValueError("audit requires one of four TRAIN tracks and authored navigation")
    blockers = []
    segments = []
    poses = []
    directions = []
    for index, (first, last) in enumerate(zip(asset.navigation.points, asset.navigation.points[1:])):
        direction = (last-first).normalized()
        right = UP.cross(direction).normalized()
        rotation = Quaternion.from_basis(right, direction.cross(right), direction)
        a, b = (VehicleState(p+UP*parameters.ride_height, ZERO, rotation, ZERO, True)
                for p in (first, last))
        poses.append((a, b))
        directions.append(direction)
        road = swept_body_road_union(a, b, BODY, asset.world.surfaces,
                                    ride_height_m=parameters.ride_height)
        obstacles = []
        for obstacle in asset.world.obstacles:
            result = sweep_body_against_obstacle(a, b, BODY, obstacle.volume)
            status = "contact" if result.contact else "uncertain" if result.uncertain else "clear"
            obstacles.append({"obstacle_id": obstacle.id, "status": status, "checks": result.checks})
            if status != "clear":
                blockers.append(f"body_obstacle:{index}:{obstacle.id}:{status}")
        if not road.valid:
            blockers.append(f"body_road:{index}:{road.status}")
        segments.append({"index": index, "road": asdict(road), "obstacles": obstacles})
    angles = [acos(max(-1., min(1., a.dot(b)))) for a, b in zip(directions, directions[1:])]
    if any(angle > 1e-8 for angle in angles):
        blockers.append("authored_heading_discontinuity")
    gates = []
    gate_cursor = -1.
    for gate in (*sorted(asset.world.checkpoints, key=lambda c: c.order), *asset.world.finish_lines):
        crossings = []
        for index, (a, b) in enumerate(poses):
            crossing = directed_center_gate_crossing(a.position, b.position, gate.volume)
            if crossing is not None:
                # Affine projection along travel onto the gate plane gives the
                # exact full-passage section for a translating, fixed OBB.
                axes = [a.orientation.rotate(v) for v in (Vec3(1.,0.,0.), UP, Vec3(0.,0.,1.))]
                center = BODY.box_at(crossing.center, a.orientation).center
                normal = gate.volume.orientation.rotate(Vec3(0.,0.,1.))
                direction = (b.position-a.position).normalized()
                section = []
                for sx in (-1., 1.):
                    for sy in (-1., 1.):
                        for sz in (-1., 1.):
                            corner = center + axes[0]*(sx*BODY.width_m/2) + axes[1]*(sy*BODY.height_m/2) + axes[2]*(sz*BODY.length_m/2)
                            section.append(corner-direction*((corner-gate.volume.center).dot(normal)/direction.dot(normal)))
                fits = True
                margins = []
                for axis, half in ((gate.volume.orientation.rotate(Vec3(1.,0.,0.)), gate.volume.half_extents.x),
                                   (gate.volume.orientation.rotate(UP), gate.volume.half_extents.y)):
                    margin = half-max(abs((point-gate.volume.center).dot(axis)) for point in section)
                    margins.append(margin)
                    fits &= margin >= -1e-9
                crossings.append({"segment": index, "path_fraction": index+crossing.fraction,
                                  "body_fits": fits, "aperture_margins_m": margins})
                if not fits:
                    blockers.append(f"body_aperture:{gate.id}")
        if not crossings:
            blockers.append(f"directed_gate_not_crossed:{gate.id}")
        else:
            ordered = [c for c in crossings if c["path_fraction"] > gate_cursor+1e-9]
            if not ordered:
                blockers.append(f"directed_gate_order:{gate.id}")
            else:
                gate_cursor = ordered[0]["path_fraction"]
        gates.append({"gate_id": gate.id, "crossings": crossings})
    transitions = []
    surfaces = {p.id: p for p in asset.world.surfaces}
    ids = asset.navigation.surface_ids
    for i, (first, last) in enumerate(zip(ids, ids[1:])):
        if first != last:
            normal_change = degrees(acos(max(-1., min(1., surfaces[first].normal.dot(surfaces[last].normal)))))
            transitions.append({"point_index": i+1, "from": first, "to": last, "normal_change_deg": normal_change})
            if normal_change > 1e-8:
                blockers.append(f"surface_transition_uncertified:{i+1}")
    return {
        "track_id": asset.track_id, "world_sha256": fingerprint_track_world(asset.world),
        "geometry_status": "uncertified" if blockers else "certified_authored_sweep",
        "body": asdict(BODY), "vehicle_parameters": asdict(parameters),
        "segments": segments, "gates": gates, "surface_transitions": transitions,
        "turning": {"minimum_radius_m": parameters.wheelbase/tan(parameters.max_steering_angle),
                    "maximum_junction_deg": degrees(max(angles, default=0.)),
                    "scope": "exact_authored_polyline; alternative smooth paths not searched"},
        "braking": {"status": "unregistered_speed_profile", "policy_brake": 0.,
                    "reason": "no speed/time trajectory or lateral-force validation; no braking feasibility certificate"},
        "blockers": sorted(set(blockers)),
        "training_authorized": False,
    }


def audit_train_tracks(repository_root, *, track_ids=TRAIN_IDS, search_alternatives=False, execute_controls=False,
                       registered_center_gates=False,execution_profile="pv5_v1"):
    if type(execution_profile) is not str or execution_profile not in ('pv5_v1','spatial_ground_v1','spatial_owner_v2','spatial_rotating_owner_v3'):
        raise ValueError('unregistered execution profile')
    if execution_profile in ('spatial_ground_v1','spatial_owner_v2','spatial_rotating_owner_v3'):
        if execute_controls is not True:
            raise ValueError('spatial profile audit requires actual execution')
        registered_center_gates = True
    if type(search_alternatives) is not bool:
        raise ValueError("search_alternatives must be bool")
    if type(execute_controls) is not bool:
        raise ValueError('execute_controls must be bool')
    if type(registered_center_gates) is not bool or registered_center_gates and not (search_alternatives or execute_controls):
        raise ValueError('registered center gates require an alternative search or execution audit')
    if track_ids != TRAIN_IDS:
        raise ValueError("audit requires exactly the four TRAIN identities")
    root = Path(repository_root).resolve(strict=True)
    if registered_center_gates:
        from ..rl.owner_acceptance import review_preregistered_contract
        review_preregistered_contract(root/'configs/recovery/wp7-owner-decisions-preregistered-v1.yml')
    manifest_path = root / "assets/tracks/manifest.json"
    entries = {e.track_id: e for e in load_track_manifest(manifest_path).tracks}
    config_path = root / "configs/experiments/ppo-generalization-v1.yml"
    definition = ExperimentDefinition.from_yaml(config_path)
    rows = []
    for identity in TRAIN_IDS:
        entry = entries[identity]
        if entry.path != f"training/{identity}.json":
            raise ValueError("TRAIN manifest path drift")
        path = manifest_path.parent / entry.path
        if path.is_symlink() or not path.resolve(strict=True).is_relative_to(root / "assets/tracks/training"):
            raise ValueError("TRAIN source escapes registry")
        asset = load_track_asset(path)
        if asset.track_id != identity or asset.world_sha256 != entry.world_sha256 or asset.content_sha256 != entry.content_sha256:
            raise ValueError("TRAIN asset identity/hash drift")
        row = audit_track(asset, parameters=VehicleParameters(**asdict(definition.vehicle)))
        if search_alternatives:
            from .wp7_planar_trajectory import search_planar_trajectory
            if registered_center_gates and identity=='right-turn-v1':
                from .wp7_arc_trajectory import search_right_turn_trajectory
                row['alternative_trajectory'] = search_right_turn_trajectory(asset,
                    parameters=VehicleParameters(**asdict(definition.vehicle)),gate_contract='directed_center_crossing_v1')
            else:
                row["alternative_trajectory"] = search_planar_trajectory(
                    asset, parameters=VehicleParameters(**asdict(definition.vehicle)))
        if execute_controls:
            from .wp7_executed_feasibility import certify_planar_execution, certify_registered_center_execution
            executor = certify_registered_center_execution if registered_center_gates else certify_planar_execution
            if execution_profile in ('spatial_ground_v1','spatial_owner_v2','spatial_rotating_owner_v3'):
                from .wp7_executed_feasibility import certify_spatial_profile_execution
                row['executed_spatial_trajectory'] = certify_spatial_profile_execution(root,identity,profile=execution_profile)
            else:
                row['executed_planar_trajectory'] = executor(root,identity)
        row.update(source_path=path.relative_to(root).as_posix(), source_sha256=sha256(path.read_bytes()).hexdigest())
        rows.append(row)
    report = {"schema_version": "wp7_body_track_audit_v1", "tracks": rows,
            "manifest_sha256": sha256(manifest_path.read_bytes()).hexdigest(),
            "vehicle_config_sha256": sha256(config_path.read_bytes()).hexdigest(),
            "scope": ('authored_geometry_and_executed_numerical_planar_witnesses; no_physical_validation_or_admission'
                      if execute_controls else "authored_geometry_and_bounded_planar_candidates; no dynamic_feasibility_or_admission"
                      if search_alternatives else "authored_geometry_only; no rollout, calibration or alternative-path proof"),
            "training_authorized": False}
    if registered_center_gates:
        report.update(schema_version='wp7_body_track_audit_v2',gate_contract='directed_center_crossing_v1',
            owner_contract_sha256=sha256((root/'configs/recovery/wp7-owner-decisions-preregistered-v1.yml').read_bytes()).hexdigest())
    if execution_profile in ('spatial_ground_v1','spatial_owner_v2','spatial_rotating_owner_v3'):
        from .wp7_protocol import build_protocol
        protocol = build_protocol(root,profile=execution_profile)
        if any(row['executed_spatial_trajectory']['protocol_sha256']!=protocol['sha256']
               or row['executed_spatial_trajectory']['physics_sha256']!=protocol['physics_sha256'] for row in rows):
            raise ValueError('spatial audit source/profile identity changed between executions')
        report.update(schema_version='wp7_body_track_audit_v3',execution_profile=execution_profile,
            protocol_sha256=protocol['sha256'],physics_sha256=protocol['physics_sha256'],
            scope='authored_findings_separate_from_nominal_registered_spatial_execution_not_physical_validation',
            all_train_executable=all(row['executed_spatial_trajectory']['status']=='executable_numerical_registered_spatial_path'
                                     for row in rows))
    return report
