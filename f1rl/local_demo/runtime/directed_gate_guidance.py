from f1rl.local_demo.runtime.drive_config import AutoDriveConfig
from f1rl.sandbox.geometry.rotation import Quaternion
from f1rl.sandbox.geometry.vector import Vec3
from f1rl.track_assets.model import AuthoredNavigation

RIGHT_TURN_WORLD_SHA256 = "d79133bc9a0599f9b056c7eb78658446631f564f038bc4afa91f3325fd4f9fd2"
RIGHT_TURN_AUTHORED_POINTS = ((0, 4), (0, 35), (0, 64), (0, 70), (20, 70), (64, 70))
RIGHT_TURN_AUTHORED_SURFACES = ("road-entry", "road-entry", "road-entry", "road-exit", "road-exit")
RIGHT_TURN_GUIDANCE_POINTS = (
    (0, 4), (0, 35), (0, 60), (2, 70), (10, 68.7), (17, 67.7), (20, 69.7),
    (24, 71.7), (30, 71.4), (40, 68), (56, 68.8), (61, 68.8), (64, 70.8), (68, 72.8),
)
RIGHT_TURN_GUIDANCE_MAX_KMH = 25.0


def directed_gate_guidance(track_id: str, world_sha256: str, navigation: AuthoredNavigation,
                           config: AutoDriveConfig, spawn_position: Vec3,
                           spawn_orientation: Quaternion) -> tuple[AuthoredNavigation, AutoDriveConfig, bool]:
    expected_points = tuple(Vec3(x, 0, z) for x, z in RIGHT_TURN_AUTHORED_POINTS)
    if (track_id != "right-turn-v1" or world_sha256 != RIGHT_TURN_WORLD_SHA256
            or navigation.points != expected_points or navigation.surface_ids != RIGHT_TURN_AUTHORED_SURFACES
            or navigation.checkpoint_point_indices != (1, 4)
            or spawn_position != Vec3(0, .5, 4) or spawn_orientation != Quaternion(1, 0, 0, 0)
            or config.driving_style != "right" or config.lane_offset_m != 0.0):
        return navigation, config, False
    points = tuple(Vec3(x, 0, z) for x, z in RIGHT_TURN_GUIDANCE_POINTS)
    surfaces = tuple("road-entry" if index < 3 else "road-exit" for index in range(len(points) - 1))
    guidance = AuthoredNavigation(points, surfaces, (1, 6))
    effective = config.model_copy(update={"target_max_speed_kmh": min(config.target_max_speed_kmh,
                                                                     RIGHT_TURN_GUIDANCE_MAX_KMH)})
    return guidance, effective, True
