from pathlib import Path

import numpy as np

from f1rl.local_demo.runtime.adapter import RuntimeEnvironmentAdapter
from f1rl.local_demo.runtime.auto_drive import AutoDriveController
from f1rl.local_demo.runtime.catalog import RuntimeCatalog
from f1rl.local_demo.runtime.custom_tracks import convert_custom_track
from f1rl.local_demo.runtime.drive_config import AutoDriveConfig
from f1rl.local_demo.runtime.v4_dev_profile import bind_custom_auto_layered_v5
from f1rl.sandbox.spatial_contact_dynamics import LAYERED_ROAD_EDGE_LIFECYCLE_V5

ROOT = Path(__file__).resolve().parents[2]


def test_custom_layered_crossing_auto_drives_to_valid_finish():
    draft = {
        "schema": "f1rl_racing_sandbox_draft_v1",
        "id": "layered-v5-crossing",
        "name": "Layered V5 crossing",
        "width": 11,
        "points": [
            {"x": x, "z": z}
            for x, z in ((0, 0), (100, 100), (0, 100), (100, 0), (160, 0))
        ],
        "props": [],
    }
    converted = convert_custom_track(draft, RuntimeCatalog(ROOT))
    factory = bind_custom_auto_layered_v5(converted.factory)
    environment = RuntimeEnvironmentAdapter(factory.create())
    try:
        observation, _ = environment.reset(seed=412)
        controller = AutoDriveController(
            environment,
            factory.navigation,
            factory.vehicle_parameters,
            min(surface.width for surface in factory.track.surfaces),
            AutoDriveConfig(),
            obstacles=factory.track.obstacles,
            checkpoints=factory.track.checkpoints,
            body_geometry=factory.body_geometry or factory.physics_config.body_geometry,
            allow_compact_corridor=True,
        )
        for _ in range(1000):
            action = np.asarray(controller.predict(observation), dtype=np.float32)
            observation, _, terminated, truncated, info = environment.step(action)
            if terminated or truncated:
                break
        else:
            raise AssertionError("layered custom route did not reach a terminal outcome")

        snapshot = environment.runtime_snapshot()
        assert terminated and not truncated
        assert info["finished"] is True
        assert info["research_outcome"]["kind"] == "successful_finish"
        assert info["research_outcome"]["valid_sample"] is True
        assert info["research_outcome"]["reason"] == "finish"
        assert info["physics"]["configuration"]["ground_motion_version"] == LAYERED_ROAD_EDGE_LIFECYCLE_V5
        assert snapshot["route_progress_fraction"] == 1.0
        assert not snapshot["physical_event_types"]
    finally:
        environment.close()
