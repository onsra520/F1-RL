"""Fail-closed WP7 physical vehicle review contract, not a simulator override."""
from __future__ import annotations
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import yaml


@dataclass(frozen=True)
class GeometryReview:
    source_sha256: str
    status: str
    baseline_wheelbase_m: float
    baseline_ride_height_m: float
    missing_body_fields: tuple[str,...]
    missing_corridor_fields: tuple[str,...]
    collision_decision: str
    heldout_evaluation_authorized: bool


def _keys(value:object,expected:set[str],label:str)->dict:
    if not isinstance(value,dict) or set(value)!=expected:
        raise ValueError(label+" fields do not match the registered schema")
    return value


def load_geometry_review(path:str|Path)->GeometryReview:
    file=Path(path).resolve(strict=True)
    raw=file.read_bytes()
    doc=_keys(yaml.safe_load(raw.decode("utf-8")),{
        "schema_version","status","baseline","body","route_corridor",
        "collision_decision","training_change_authorized",
        "heldout_evaluation_authorized",
    },"geometry review")
    if doc["schema_version"]!="wp7-recovery-geometry-review-v1" or doc["status"]!="unapproved":
        raise ValueError("geometry review must remain unapproved until independent review")
    base=_keys(doc["baseline"],{"wheelbase_m","ride_height_m"},"baseline")
    if base!={"wheelbase_m":3.0,"ride_height_m":0.5}:
        raise ValueError("original wheelbase/ride_height must not change")
    body=_keys(doc["body"],{
        "length_m","width_m","height_m","local_center_xyz_m",
        "collision_hull","tire_footprint",
    },"body")
    corridor=_keys(doc["route_corridor"],{
        "max_lateral_deviation_m","max_outside_road_body_distance_m",
        "approved_design_rationale",
    },"route corridor")
    if any(x is not None for x in (*body.values(),*corridor.values())):
        raise ValueError("unapproved geometric values may not be promoted to official dimensions")
    if (doc["collision_decision"]!="undecided"
        or doc["training_change_authorized"] is not False
        or doc["heldout_evaluation_authorized"] is not False):
        raise ValueError("unreviewed geometry cannot approve training/HELD-OUT")
    return GeometryReview(
        sha256(raw).hexdigest(),"unapproved",3.0,0.5,
        tuple(sorted(body)),tuple(sorted(corridor)),"undecided",False
    )
