"""Explicit development-only Physics v4 + Race v4 bind for CUSTOM Auto Drive.

Never mutate registered TRAIN factories or the frozen PPO profile.
"""
from dataclasses import replace

from src.rl.crash_terminal_registered import RACE_V4
from src.rl.wp7_race import WP7RaceConfig

V3 = "spatial_rotating_compliant_lifecycle_v3"
V4 = "spatial_road_edge_partial_lifecycle_v4"


def bind_custom_auto_v4(factory):
    """Bind matching source-audited Physics/Race contracts to a new factory."""
    if (factory.reset_config is not None
            or factory.physics_config is None
            or factory.physics_config.mode != "wp7_lifecycle_v1"
            or factory.physics_config.ground_motion_version != V3
            or factory.race_config is None
            or not factory.race_config.owner_terminal
            or factory.navigation is None):
        raise ValueError("v4 development profile requires a custom v3 owner-route factory")
    bound = replace(
        factory,
        physics_config=replace(factory.physics_config, ground_motion_version=V4),
        race_config=WP7RaceConfig(version=RACE_V4, hard_violation="terminate_valid_failure"),
    )
    env = bound.create()
    try:
        _, _info = env.reset(seed=0)
        if env.runtime_snapshot()["tick"] != 0:
            raise ValueError("v4 custom Auto Drive spawn did not initialize")
    finally:
        env.close()
    return bound
