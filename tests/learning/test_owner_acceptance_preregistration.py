"""No HELD-OUT rollouts, no TRAIN operations: owner decisions only."""
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from src.learning.owner_acceptance import OWNER_CODES, review_preregistered_contract
from src.sandbox.body_collision import VehicleBodyGeometry
from src.sandbox.geometry.vector import Vec3
from src.sandbox.ramp_support import WheelContactPoints


CONTRACT = Path("configs/recovery/wp7-owner-decisions-preregistered-v1.yml")


def _modified(tmp_path: Path, mutate):
    data = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    mutate(data)
    output = tmp_path / "tampered-contract.yml"
    output.write_text(yaml.safe_dump(data,sort_keys=False),encoding="utf-8")
    return output


def test_owner_decisions_registered_and_no_scientific_approval() -> None:
    report = review_preregistered_contract(CONTRACT)
    assert report.codes == OWNER_CODES
    assert report.status == "design_locked_implementation_pending"
    assert report.blockers
    assert not report.training_permitted
    assert not report.heldout_permitted
    assert not report.scientific_freeze_permitted
    assert not report.wp8_permitted


def test_owner_vehicle_hull_and_contact_points_are_coherent() -> None:
    doc = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    body = doc["simulation"]["body"]
    v = VehicleBodyGeometry(
        body["width_m"], body["height_m"], body["length_m"],
        Vec3(*body["local_center_xyz_m"]),
    )
    assert v.width_m == 1.8 and v.height_m == 1.2 and v.length_m == 4.
    points=body["wheel_points_xyz_m"]
    wheels = WheelContactPoints(*(Vec3(*points[name]) for name in (
        "front_left","front_right","rear_left","rear_right"
    )))
    assert wheels.front_left.x == pytest.approx(-.75)
    assert wheels.front_right.x == pytest.approx(.75)
    assert wheels.front_left.z - wheels.rear_left.z == pytest.approx(3.)
    assert body["ride_height_m"] + v.local_center.y - v.height_m/2 == pytest.approx(.2)


@pytest.mark.parametrize("mutation", [
    lambda d: d.update(decision_codes=["1A",*d["decision_codes"][1:]]),
    lambda d: d.update(wp0="official_wp0"),
    lambda d: d.update(status="frozen"),
    lambda d: d["simulation"]["body"].update(length_m=3.9),
    lambda d: d["simulation"]["body"].update(width_m=2.0),
    lambda d: d["simulation"]["body"].update(height_m=1.4),
    lambda d: d["simulation"]["body"].update(local_center_xyz_m=[0.,0.,0.]),
    lambda d: d["simulation"]["body"].update(wheelbase_m=5.0),
    lambda d: d["simulation"]["body"]["wheel_points_xyz_m"]["front_left"].__setitem__(2,1.9),
    lambda d: d["simulation"]["suspension"].update(mode="rigid_four_wheel"),
    lambda d: d["simulation"]["suspension"].update(max_bump_m=.8),
    lambda d: d["simulation"]["airborne"].update(mode="no_flight"),
    lambda d: d["driving"].update(soft_deviation_m=2.1),
    lambda d: d["driving"].update(hard_deviation_m=5.),
    lambda d: d["driving"].update(route_reference="dynamic_route"),
    lambda d: d["driving"]["checkpoint_finish"].update(mode="any_body_overlap"),
    lambda d: d["driving"]["checkpoint_finish"].update(continuous_sweep_required=False),
    lambda d: d["reliability"].update(minimum_independent_ppo_seed_pairs=2),
    lambda d: d["reliability"].update(confidence_level=.9),
    lambda d: d["reliability"].update(minimum_lower_success_bound_per_train_track=.8),
    lambda d: d["reliability"].update(episodes_per_track_per_model_per_condition=59),
    lambda d: d["reliability"].update(evaluation_conditions=["nominal"]),
    lambda d: d["reliability"].update(spawn_lateral_offset_m=[-1.,1.]),
    lambda d: d["reliability"].update(spawn_heading_jitter_deg=[-10.,10.]),
    lambda d: d["backup"].update(provider="local_only"),
    lambda d: d["backup"].update(require_sha256_manifest=False),
    lambda d: d["backup"].update(require_independent_restore_test=False),
    lambda d: d["evidence_gates"].update(geometry_feasibility="verified"),
    lambda d: d["heldout"].update(status="open"),
    lambda d: d["heldout"].update(evaluation_permitted=True),
    lambda d: d.update(scientific_freeze_permitted=True),
    lambda d: d.update(wp8_permitted=True),
    lambda d: d["simulation"].update(extra="unreviewed"),
])
def test_preregistered_decisions_cannot_be_silently_rewritten(tmp_path,mutation) -> None:
    with pytest.raises(ValueError):
        review_preregistered_contract(_modified(tmp_path,mutation))


def test_yaml_duplicate_key_cannot_override_owner_threshold(tmp_path) -> None:
    text=CONTRACT.read_text(encoding="utf-8")
    text=text.replace("  soft_deviation_m: 2.0","  soft_deviation_m: 2.0\n  soft_deviation_m: 8.0")
    path=tmp_path/"duplicate.yml"
    path.write_text(text,encoding="utf-8")
    with pytest.raises(ValueError,match="duplicate preregistration key"):
        review_preregistered_contract(path)


def test_unknown_evidence_fields_are_rejected(tmp_path) -> None:
    def add(d): d["evidence_gates"]["trust_me"]=True
    with pytest.raises(ValueError,match="evidence_gates"):
        review_preregistered_contract(_modified(tmp_path,add))
