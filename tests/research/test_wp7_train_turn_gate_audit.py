"""Directed TRAIN gates must not be silently reinterpreted from route headings."""
from pathlib import Path
import pytest

from f1rl.track_assets import load_track_asset
from f1rl.sandbox.directed_center_crossing import directed_center_gate_crossing
from f1rl.sandbox.center_trajectory import CenterSegment
from f1rl.sandbox.geometry.vector import Vec3

ROOT=Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("name,sign", [
    ("right-turn-v1",1),
    ("left-turn-train-v1",-1),
])
def test_real_train_turn_gate_direction_contract_before_episode(name,sign):
    asset=load_track_asset(ROOT/f"assets/tracks/training/{name}.json")
    assert [x.id for x in asset.world.checkpoints]==["checkpoint-0","checkpoint-1"]
    gates=[*asset.world.checkpoints,*asset.world.finish_lines]
    for gate in gates:
        vol=gate.volume
        forward=vol.orientation.rotate(Vec3(0,0,1))
        assert forward==Vec3(0,0,1)
        x,y,z=vol.center.x,vol.center.y,vol.center.z
        # A geometric line segment parallel to intended outgoing +/-X
        # does not cross the currently identity-oriented checkpoint plane.
        if gate.id!="checkpoint-0":
            start=Vec3(x-3*sign,y+.5,z)
            end=Vec3(x+3*sign,y+.5,z)
            assert directed_center_gate_crossing(start,end,vol) is None
            assert CenterSegment(0,1,start,end).directed_crossings(vol)==()
        # A +Z segment through the same gate centre is a directed crossing.
        start=Vec3(x,y+.5,z-2)
        end=Vec3(x,y+.5,z+2)
        crossing=directed_center_gate_crossing(start,end,vol)
        assert crossing is not None
        assert crossing.fraction==pytest.approx(.5)
        assert CenterSegment(0,1,start,end).directed_crossings(vol)==pytest.approx((.5,))


def test_mirrored_navigation_is_geometrically_equal_not_a_physics_certificate():
    from f1rl.track_assets import load_track_asset
    original=load_track_asset(ROOT/"assets/tracks/training/right-turn-v1.json")
    mirrored=load_track_asset(ROOT/"assets/tracks/training/left-turn-train-v1.json")
    a=original.navigation.points
    b=mirrored.navigation.points
    assert len(a)==len(b)
    assert all(abs(p.x+q.x)<1e-10 and abs(p.y-q.y)<1e-10
               and abs(p.z-q.z)<1e-10 for p,q in zip(a,b))
    assert original.navigation.surface_ids==mirrored.navigation.surface_ids
