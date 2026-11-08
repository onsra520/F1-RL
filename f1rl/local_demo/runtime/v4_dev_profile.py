"""Explicit development-only Physics v4 + Race v4 bind for CUSTOM Auto Drive.

Never mutate registered TRAIN factories or the frozen PPO profile.
"""
from f1rl.learning.crash_terminal_registered import RACE_V4
from f1rl.local_demo.runtime.adapter import RuntimeEnvironmentAdapter

V4 = "spatial_road_edge_partial_lifecycle_v4"


def bind_custom_auto_v4(factory):
    """Accept only the registered v4 Physics/Race contracts on a custom factory."""
    if (factory.reset_config is not None
            or factory.physics_config is None
            or factory.physics_config.mode != "wp7_lifecycle_v1"
            or factory.physics_config.ground_motion_version != V4
            or factory.race_config is None
            or factory.race_config.version != RACE_V4
            or not factory.race_config.owner_terminal
            or factory.navigation is None):
        raise ValueError("v4 development profile requires an already registered custom v4 owner-route factory")
    env = factory.create()
    try:
        _, _info = env.reset(seed=0)
        if RuntimeEnvironmentAdapter(env).runtime_snapshot()["tick"] != 0:
            raise ValueError("v4 custom Auto Drive spawn did not initialize")
    finally:
        env.close()
    return factory
