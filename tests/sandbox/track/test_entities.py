from dataclasses import FrozenInstanceError

import pytest

from f1rl.sandbox.entities.boost_pad import BoostPad
from f1rl.sandbox.entities.checkpoint import Checkpoint
from f1rl.sandbox.entities.finish_line import FinishLine
from f1rl.sandbox.entities.obstacle import Obstacle
from f1rl.sandbox.entities.score_pickup import ScorePickup
from f1rl.sandbox.entities.slow_zone import SlowZone
from f1rl.sandbox.geometry.rotation import Quaternion
from f1rl.sandbox.geometry.vector import Vec3
from f1rl.sandbox.track.volume import OrientedBoxVolume


def volume() -> OrientedBoxVolume:
    return OrientedBoxVolume(Vec3(0.0, 0.5, 0.0), Vec3(1.0, 0.5, 1.0), Quaternion.identity())


def test_entities_construct_with_expected_values() -> None:
    box = volume()

    assert Obstacle("wall", box).volume == box
    assert BoostPad("boost", box, 8.0).acceleration_bonus == 8.0
    assert SlowZone("mud", box, 2.5).drag_multiplier == 2.5
    assert Checkpoint("cp-1", box, 1).order == 1
    assert FinishLine("finish", box).id == "finish"
    assert ScorePickup("score-1", box, 100).points == 100


def test_entity_ids_must_be_non_empty() -> None:
    constructors = (
        lambda: Obstacle("", volume()),
        lambda: BoostPad(" ", volume(), 1.0),
        lambda: SlowZone("", volume(), 1.0),
        lambda: Checkpoint("", volume(), 0),
        lambda: FinishLine("", volume()),
        lambda: ScorePickup("", volume(), 1),
    )

    for constructor in constructors:
        with pytest.raises(ValueError):
            constructor()


def test_entity_specific_parameters_are_validated() -> None:
    with pytest.raises(ValueError):
        BoostPad("boost", volume(), 0.0)
    with pytest.raises(ValueError):
        SlowZone("slow", volume(), 0.99)
    with pytest.raises(ValueError):
        Checkpoint("cp", volume(), -1)
    with pytest.raises(ValueError):
        ScorePickup("score", volume(), 0)


def test_entity_numeric_values_must_be_finite_and_typed() -> None:
    with pytest.raises(ValueError):
        BoostPad("boost", volume(), float("inf"))
    with pytest.raises(ValueError):
        SlowZone("slow", volume(), float("nan"))
    with pytest.raises(ValueError):
        Checkpoint("cp", volume(), True)
    with pytest.raises(ValueError):
        ScorePickup("score", volume(), 1.5)  # type: ignore[arg-type]


def test_entities_are_immutable() -> None:
    pickup = ScorePickup("score", volume(), 100)

    with pytest.raises(FrozenInstanceError):
        pickup.points = 200  # type: ignore[misc]
