"""Validated immutable development-only Auto Drive tuning (never Physics overrides)."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

DrivingStyle = Literal["right", "centerline", "racing"]
SpeedProfile = Literal["normal", "racing", "hyper"]
SPEED_PRESET_KMH = {"normal": 45.0, "racing": 90.0, "hyper": 240.0}


class AutoDriveConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)

    driving_style: DrivingStyle = "right"
    speed_profile: SpeedProfile = "normal"
    target_max_speed_kmh: float = Field(default=45.0, gt=0.0, le=300.0)
    vehicle_width_m: float = Field(default=1.5, gt=0.4, le=4.0)
    lane_offset_m: float = Field(default=0.0, ge=0.0, le=5.0)
    edge_margin_m: float = Field(default=0.6, ge=0.2, le=3.0)
    min_lookahead_m: float = Field(default=4.0, ge=1.0, le=15.0)
    max_lookahead_m: float = Field(default=14.0, ge=2.0, le=35.0)
    speed_lookahead_s: float = Field(default=0.65, ge=0.0, le=3.0)
    corner_acceleration_mps2: float = Field(default=2.2, gt=0.2, le=12.0)
    braking_factor: float = Field(default=0.65, gt=0.1, le=1.0)
    acceleration_factor: float = Field(default=0.65, gt=0.1, le=1.0)
    steering_gain: float = Field(default=1.0, gt=0.25, le=2.0)
    recovery_speed_mps: float = Field(default=3.0, gt=0.5, le=10.0)
    grade_lookahead_m: float = Field(default=20.0, ge=2.0, le=60.0)
    moderate_grade_threshold: float = Field(default=0.1, ge=0.02, le=0.5)
    steep_grade_threshold: float = Field(default=0.2, ge=0.03, le=0.8)
    moderate_grade_speed_mps: float = Field(default=6.0, gt=0.5, le=20.0)
    steep_grade_speed_mps: float = Field(default=4.0, gt=0.5, le=15.0)
    corner_anticipation_m: float = Field(default=1.5, ge=0.0, le=10.0)

    @model_validator(mode="after")
    def consistent(self):
        if self.min_lookahead_m > self.max_lookahead_m:
            raise ValueError("minimum lookahead cannot exceed maximum lookahead")
        if self.moderate_grade_threshold >= self.steep_grade_threshold:
            raise ValueError("steep grade threshold must exceed moderate threshold")
        if self.steep_grade_speed_mps > self.moderate_grade_speed_mps:
            raise ValueError("steep grade speed must not exceed moderate grade speed")
        return self
