from __future__ import annotations
import dataclasses, hashlib, json, math, sys, traceback
from pathlib import Path
import numpy as np
import torch

ROOT = Path(r"D:\Projects\.worktrees\F1-RL\physics-v4-phase1-audit").resolve()
OUT = Path(r"D:\Projects\.evidence\F1-RL\phase2a-root-cause-audit-20261106")
PHASE1 = Path(r"D:\Projects\.evidence\F1-RL\phase1-physics-v4-audit-20261106")
sys.path.insert(0, str(ROOT))
torch.set_num_threads(1)
from f1rl.learning.algorithms.ppo import PPOModel
from f1rl.learning.action import map_policy_action
from f1rl.research.wp7_protocol import load_registration, prepare_scenarios
from f1rl.sandbox import spatial_contact_dynamics as scd
from f1rl.sandbox import multi_surface_support as mss
from f1rl.sandbox.road_footprint import body_xz_polygon


def safe(v):
    if dataclasses.is_dataclass(v):
        return safe(dataclasses.asdict(v))
    if isinstance(v, dict):
        return {str(k): safe(x) for k, x in v.items()}
    if isinstance(v, (tuple, list)):
        return [safe(x) for x in v]
    if isinstance(v, np.ndarray):
        return safe(v.tolist())
    if isinstance(v, (np.floating, float)):
        x = float(v)
        return x if math.isfinite(x) else str(x)
    if isinstance(v, (np.integer, int)):
        return int(v)
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    if v is None or isinstance(v, (str, bool)):
        return v
    if hasattr(v, "__dict__"):
        return safe(vars(v))
    return str(v)


def vec(v):
    return [float(v.x), float(v.y), float(v.z)]


def vehicle_state(v):
    q = v.orientation
    return {"position_m": vec(v.position), "velocity_mps": vec(v.velocity),
            "speed_mps": float(v.speed), "orientation_wxyz": [q.w, q.x, q.y, q.z],
            "angular_velocity_radps": vec(v.angular_velocity), "grounded": bool(v.grounded)}


def suspension_state(v):
    return safe(scd._suspension_state(v))


def support_at_state(state, policy, atlas, *, free_extension):
    patches, polygons = atlas
    report = []
    all_ok = True
    for name, local in zip(mss._WHEEL_NAMES, policy.wheels.positions, strict=True):
        point = state.position + state.orientation.rotate(local)
        candidates = []
        patch_rows = []
        for patch, polygon in zip(patches, polygons, strict=True):
            inside = mss._within_shape(point.x, point.z, polygon)
            height = mss._height(patch, point.x, point.z)
            deflection = height - point.y
            within_droop = free_extension or deflection >= -policy.max_droop_m - 1e-9
            within_bump = deflection <= policy.max_bump_m + 1e-9
            row = {"patch_id": patch.id, "inside_polygon": inside,
                   "surface_height_m": height, "deflection_m": deflection,
                   "within_droop": within_droop, "within_bump": within_bump,
                   "bump_margin_m": policy.max_bump_m - deflection,
                   "droop_margin_m": deflection + policy.max_droop_m}
            if inside:
                patch_rows.append(row)
                if within_droop and within_bump:
                    candidates.append((abs(deflection), patch.id, height, deflection))
        chosen = min(candidates) if candidates else None
        all_ok &= chosen is not None
        report.append({"wheel": name, "local_position_m": vec(local),
                       "world_position_m": vec(point), "candidate_count": len(candidates),
                       "selected_contact": None if chosen is None else {
                           "patch_id": chosen[1], "ground_height_m": chosen[2], "deflection_m": chosen[3]},
                       "patches_under_wheel": patch_rows})
    return {"all_wheels_have_reference_contact": all_ok, "wheels": report}


def endpoint_geometry(state, body, surfaces, policy, *, contact_domain):
    atlas = mss._atlas(surfaces, policy.seam_height_tolerance_m)
    if atlas is None:
        return {"atlas_valid": False}
    patches, polygons = atlas
    free_extension = contact_domain == "unilateral_free_extension_v1"
    support = support_at_state(state, policy, atlas, free_extension=free_extension)
    contacts = scd._PHASE2A_ORIGINAL_CONTACTS(state, policy, atlas, free_extension=free_extension)
    hull = body_xz_polygon(state, body)
    envelope = mss._inflate(hull, 0.0)
    uncovered = mss._uncovered_area(envelope, polygons)
    clearance_checks = []
    for patch, polygon in zip(patches, polygons, strict=True):
        if not mss._intersection(envelope, polygon):
            continue
        clearance = mss._bounded_patch_body_clearance(state, body, patch, polygon, 0.0)
        clearance_checks.append({"patch_id": patch.id, "clearance_m": clearance,
                                 "penetration_or_unresolved": clearance is None or clearance < -1e-9})
    return {"atlas_valid": True,
            "wheel_contacts": None if contacts is None else safe(contacts),
            "support_geometry": support,
            "body_footprint_uncovered_area_m2": uncovered,
            "body_clearance_checks": clearance_checks,
            "body_corners_m": [vec(v) for v in mss._body_corners(state, body)]}


PHASE1_SUMMARY = json.loads((PHASE1 / "smoke-summary.json").read_text(encoding="utf-8"))
INVALIDS = [row for row in PHASE1_SUMMARY["runs"] if row["classification"] == "invalid_physical_transition"]
EXPECTED = {(row["model_id"], row["track_id"]): row for row in INVALIDS}
CASES = [(row["model_id"], row["track_id"]) for row in INVALIDS]
registration = load_registration(ROOT, profile="spatial_rotating_owner_v3")
scenarios = prepare_scenarios(ROOT, registration, seed_index=0)
scenario_by_id = {row.track_id: row for row in scenarios.scenarios}

scd._PHASE2A_ORIGINAL_CONTACTS = scd._contacts
scd._PHASE2A_ORIGINAL_SWEEP = scd.certify_multisurface_support
monitor = {"active": None}


def watched_contacts(state, policy, atlas, *, free_extension=False):
    result = scd._PHASE2A_ORIGINAL_CONTACTS(state, policy, atlas, free_extension=free_extension)
    active = monitor["active"]
    if active is not None and result is None:
        active["contact_missing_calls"] += 1
        if active.get("last_contact_missing") is None or active["contact_missing_calls"] <= 3:
            active["contact_missing_first_or_early"] = active.get("contact_missing_first_or_early", [])
            if len(active["contact_missing_first_or_early"]) < 3:
                active["contact_missing_first_or_early"].append({
                    "vehicle": vehicle_state(state),
                    "suspension": suspension_state(state),
                    "wheel_support": support_at_state(state, policy, atlas, free_extension=free_extension)})
        active["last_contact_missing"] = {
            "vehicle": vehicle_state(state),
            "suspension": suspension_state(state),
            "wheel_support": support_at_state(state, policy, atlas, free_extension=free_extension)}
    return result


def watched_sweep(previous, candidate, body, policy, surfaces, **kwargs):
    verdict = scd._PHASE2A_ORIGINAL_SWEEP(previous, candidate, body, policy, surfaces, **kwargs)
    active = monitor["active"]
    if active is not None and verdict.status != "clear":
        active["sweep_nonclear_calls"] += 1
        if verdict.status == "outside":
            atlas = mss._atlas(surfaces, policy.seam_height_tolerance_m)
            free_extension = kwargs.get("contact_domain") == "unilateral_free_extension_v1"
            active["last_sweep_outside"] = {
                "status": verdict.status, "reason": verdict.reason, "checks": verdict.checks,
                "verdict_contacts": safe(verdict.contacts),
                "previous_substep_state": vehicle_state(previous),
                "candidate_substep_state": vehicle_state(candidate),
                "previous_substep_suspension": suspension_state(previous),
                "candidate_substep_suspension": suspension_state(candidate),
                "previous_endpoint_geometry": endpoint_geometry(previous, body, surfaces, policy,
                                                                  contact_domain=kwargs.get("contact_domain", "four_supported_v1")),
                "candidate_endpoint_geometry": endpoint_geometry(candidate, body, surfaces, policy,
                                                                   contact_domain=kwargs.get("contact_domain", "four_supported_v1")),
            }
    return verdict


scd._contacts = watched_contacts
scd.certify_multisurface_support = watched_sweep

results = []
model_cache = {}
for model_id, track_id in CASES:
    expected = EXPECTED[(model_id, track_id)]
    if model_id not in model_cache:
        model_cache[model_id] = PPOModel.load(ROOT / "models" / model_id / "model-final.zip", device="cpu")
    factory = scenario_by_id[track_id].factory
    env = factory.create()
    env.set_training_telemetry_enabled(True)
    observation, reset_info = env.reset(seed=PHASE1_SUMMARY["reset_seed"])
    case = {"model_id": model_id, "track_id": track_id,
            "source_commit": PHASE1_SUMMARY["source_commit"],
            "reset_seed": PHASE1_SUMMARY["reset_seed"],
            "inference_mode": "deterministic", "step_limit": PHASE1_SUMMARY["step_limit"],
            "phase1_accepted_steps": expected["steps"],
            "phase1_failed_attempt_step": expected["steps"] + 1,
            "physics_config": safe(factory.physics_config.document()),
            "race_config": safe(factory.race_config),
            "reset_info": safe(reset_info),
            "asset_sha256": hashlib.sha256((ROOT / "assets" / "tracks" / "training" / f"{track_id}.json").read_bytes()).hexdigest(),
            "road_geometry_patches": safe(factory.track.surfaces),
            "replay_status": "not_started"}
    monitor["active"] = {"contact_missing_calls": 0, "sweep_nonclear_calls": 0}
    try:
        for step in range(1, PHASE1_SUMMARY["step_limit"] + 1):
            before = env._simulation.snapshot
            before_vehicle = before.world.vehicle
            action = np.asarray(model_cache[model_id].predict(observation, deterministic=True), dtype=np.float32)
            controls = {"steering": float(action[0]), "throttle": float(action[1]), "brake": float(action[2]) if len(action) > 2 else 0.0}
            attempt_monitor = {"contact_missing_calls": 0, "sweep_nonclear_calls": 0}
            monitor["active"] = attempt_monitor
            try:
                next_observation, reward, terminated, truncated, info = env.step(action)
            except Exception as error:
                accepted_audit = before.spatial_ground_telemetry
                last_contacts = []
                if accepted_audit and accepted_audit.substeps:
                    stages = accepted_audit.substeps[-1].contact_stages
                    if stages:
                        last_contacts = safe(stages[-1])
                case.update({
                    "replay_status": "exception_reproduced",
                    "attempt_step": step,
                    "pre_transition_vehicle": vehicle_state(before_vehicle),
                    "pre_transition_suspension": suspension_state(before_vehicle),
                    "pre_transition_snapshot_tick": before.tick,
                    "pre_transition_elapsed_s": before.elapsed_time,
                    "pre_transition_last_successful_wheel_support_stage": last_contacts,
                    "action": [float(x) for x in action], "controls": controls,
                    "exception": {"class": type(error).__name__, "message": str(error),
                                  "failure_code": getattr(error, "failure_code", None),
                                  "traceback": traceback.format_exc()},
                    "research_outcome_after_exception": safe(getattr(env, "last_research_outcome", None)),
                    "internal_contact_missing_calls": attempt_monitor["contact_missing_calls"],
                    "internal_sweep_nonclear_calls": attempt_monitor["sweep_nonclear_calls"],
                    "internal_contact_missing_first_or_early": attempt_monitor.get("contact_missing_first_or_early", []),
                    "internal_contact_missing_final": attempt_monitor.get("last_contact_missing"),
                    "internal_sweep_final_outside_witness": attempt_monitor.get("last_sweep_outside"),
                })
                break
            observation = next_observation
            if terminated or truncated:
                case.update({"replay_status": "valid_terminal", "terminal_step": step,
                             "terminated": bool(terminated), "truncated": bool(truncated),
                             "terminal_info": safe(info)})
                break
        else:
            case["replay_status"] = "step_limit"
    except Exception as outer_error:
        case.update({"replay_status": "replay_driver_error", "driver_error": str(outer_error),
                     "traceback": traceback.format_exc()})
    finally:
        monitor["active"] = None
        case["observed_attempt_step"] = case.get("attempt_step")
        case["matches_phase1_identity_and_step"] = (
            case.get("replay_status") == "exception_reproduced"
            and case.get("attempt_step") == case["phase1_failed_attempt_step"]
            and case.get("exception", {}).get("class") == expected["exception"]["class"]
            and case.get("exception", {}).get("message") == expected["exception"]["message"])
        results.append(case)
        env.close()

summary = {"status": "PASS" if len(results) == 8 and all(row["matches_phase1_identity_and_step"] for row in results) else "FAIL",
           "replayed_errors": len(results), "exact_reproductions": sum(bool(row["matches_phase1_identity_and_step"]) for row in results),
           "case_results": [{"model_id": row["model_id"], "track_id": row["track_id"],
                             "failed_attempt_step": row.get("attempt_step"), "status": row["replay_status"],
                             "exception": row.get("exception", {}).get("message"),
                             "failure_code": row.get("exception", {}).get("failure_code"),
                             "contact_missing_calls": row.get("internal_contact_missing_calls"),
                             "sweep_nonclear_calls": row.get("internal_sweep_nonclear_calls"),
                             "matches_phase1": row["matches_phase1_identity_and_step"]}
                            for row in results]}
(OUT / "v3-eight-case-instrumented-replay.json").write_text(json.dumps({"summary": summary, "cases": results}, indent=2, allow_nan=False) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2))

