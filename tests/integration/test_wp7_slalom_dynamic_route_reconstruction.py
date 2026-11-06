"""TRAIN-only deterministic heuristic diagnostic for slalom route driveability.

NOT a learned policy evaluation or scientific proof of PPO performance.
This regression prevents reintroducing a final waypoint that causes a simple
steering controller to miss checkpoint-2 and exit the road.
"""
from math import atan2
from pathlib import Path

import numpy as np
import pytest

from f1rl.learning.reconstruction import RecoveryCandidate, build_training_scenarios
from f1rl.sandbox.geometry.vector import Vec3


@pytest.mark.parametrize(("gain", "throttle"), [
    (1.0, 1.0),
    (2.0, 1.0),
    (2.0, 0.4),
    (4.0, 0.4),
])
def test_slalom_reaches_all_checkpoints_and_finishes_without_offtrack(
    gain: float, throttle: float,
) -> None:
    candidate=RecoveryCandidate.from_yaml(
        Path("configs/recovery/wp7-c2-candidate.yml")
    )
    factory=next(
        s.factory for s in build_training_scenarios(candidate).scenarios
        if s.track_id == "slalom-v1"
    )
    env=factory.create()
    forward=Vec3(0.0,0.0,1.0)
    max_lateral=0.0
    collisions=0
    try:
        env.reset(seed=0)
        for _ in range(500):
            vehicle=env._simulation.snapshot.world.vehicle
            endpoint=env._route_target(vehicle,env._progress)
            assert endpoint is not None
            facing=vehicle.orientation.rotate(forward)
            dx=endpoint.x-vehicle.position.x
            dz=endpoint.z-vehicle.position.z
            error=atan2(
                facing.z*dx-facing.x*dz,
                facing.x*dx+facing.z*dz,
            )
            steering=max(-1.0,min(1.0,gain*error/0.5))
            _,_,terminated,truncated,info=env.step(
                np.array([steering,throttle],dtype=np.float32)
            )
            collisions+=len(info["telemetry"]["obstacle_hit_ids"])
            max_lateral=max(max_lateral,info["offtrack_lateral_meters"])
            if terminated or truncated:
                break
        assert terminated is True
        assert truncated is False
        assert info["finished"] is True
        assert info["next_checkpoint_index"] == 3
        assert collisions == 0
        assert max_lateral < 6.0
    finally:
        env.close()


def test_slalom_route_corner_reconstruction_preserves_world_geometry() -> None:
    from f1rl.track_assets import load_track_asset, load_track_manifest
    manifest=load_track_manifest(Path("assets/tracks/manifest.json"))
    entry=next(e for e in manifest.tracks if e.track_id=="slalom-v1")
    asset=load_track_asset(Path("assets/tracks")/entry.path)
    assert asset.navigation is not None
    points=asset.navigation.points
    assert tuple((p.x,p.y,p.z) for p in points[-4:]) == (
        (0.0,0.0,96.0),
        (1.5,0.0,102.0),
        (-3.0,0.0,110.0),
        (0.0,0.0,124.0),
    )
    assert asset.world_sha256 == "6071829f2eb790ada04c5e59b22efb205e3a20d59fbe670e68fb78ed3eb304e0"
    assert asset.content_sha256 == entry.content_sha256
    assert asset.navigation.checkpoint_point_indices == (2,5,10)
