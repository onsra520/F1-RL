from pathlib import Path

import pytest

from f1rl.learning.experiments import ExperimentDefinition


VALID_DRAFT = """
experiment:
  id: ppo-generalization-v1
  protocol_version: 1
  status: draft
semantics:
  algorithm: ppo
  policy: MlpPolicy
  action: steering-throttle-brake0-v1
  observation: rangefinders-local-navigation-v1
  reward: target-progress-collision-time-v1
  race: single-lap-ordered-checkpoints-v1
environment:
  vehicle:
    wheelbase: null
    ride_height: null
    max_steering_angle: null
    max_acceleration: null
    max_braking: null
    rolling_drag: null
    air_drag: null
    max_ground_speed: null
    gravity: null
  simulation:
    fixed_dt: null
    max_surface_distance: null
  observation:
    velocity_scale: null
    target_distance_scale: null
    rangefinders: []
  reward:
    target_progress:
      weight_per_meter: null
    collision:
      penalty: null
    time:
      penalty_per_second: null
  episode:
    max_steps: null
training:
  runtime_config: ../../config.yml
  protocol_status: pilot_owned
seeds:
  ppo: []
  sampling: []
tracks:
  manifest: ../../assets/tracks/manifest.json
  training_ids:
    - train-b
    - train-a
  heldout_ids:
    - heldout-a
evaluation:
  deterministic_policy: true
"""


def _write(tmp_path: Path, text: str = VALID_DRAFT) -> Path:
    path = tmp_path / "configs" / "experiments" / "definition.yml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_valid_draft_yaml_loads_and_resolves_relative_references(tmp_path: Path) -> None:
    path = _write(tmp_path)

    definition = ExperimentDefinition.from_yaml(path)

    assert definition.metadata.id == "ppo-generalization-v1"
    assert definition.metadata.protocol_version == 1
    assert definition.metadata.status == "draft"
    assert definition.vehicle.wheelbase is None
    assert definition.observation.rangefinders == ()
    assert definition.seeds.ppo == ()
    assert definition.seeds.sampling == ()
    assert definition.training.runtime_config == (path.parent / "../../config.yml").resolve()
    assert definition.tracks.manifest == (path.parent / "../../assets/tracks/manifest.json").resolve()
    assert definition.tracks.training_ids == ("train-b", "train-a")
    assert definition.tracks.heldout_ids == ("heldout-a",)


def test_loading_does_not_change_current_working_directory(tmp_path: Path) -> None:
    path = _write(tmp_path)
    before = Path.cwd()
    ExperimentDefinition.from_yaml(path)
    assert Path.cwd() == before


@pytest.mark.parametrize(
    "text",
    [
        "[]\n",
        "- broken\n- document\n",
        "experiment: [unterminated\n",
        "",
    ],
)
def test_yaml_root_must_be_a_valid_mapping(tmp_path: Path, text: str) -> None:
    path = _write(tmp_path, text)
    with pytest.raises(ValueError):
        ExperimentDefinition.from_yaml(path)


def test_unknown_or_missing_root_keys_are_rejected(tmp_path: Path) -> None:
    path = _write(tmp_path, VALID_DRAFT + "unknown: {}\n")
    with pytest.raises(ValueError):
        ExperimentDefinition.from_yaml(path)

    missing = VALID_DRAFT.replace("evaluation:\n  deterministic_policy: true\n", "")
    path = _write(tmp_path, missing)
    with pytest.raises(ValueError):
        ExperimentDefinition.from_yaml(path)


def test_unknown_nested_keys_are_rejected(tmp_path: Path) -> None:
    text = VALID_DRAFT.replace("    gravity: null\n", "    gravity: null\n    fixture_default: 9.81\n")
    path = _write(tmp_path, text)
    with pytest.raises(ValueError):
        ExperimentDefinition.from_yaml(path)


@pytest.mark.parametrize("value", ["true", ".nan", ".inf", "-.inf"])
def test_invalid_numeric_yaml_values_are_rejected(tmp_path: Path, value: str) -> None:
    text = VALID_DRAFT.replace("    wheelbase: null", f"    wheelbase: {value}")
    path = _write(tmp_path, text)
    with pytest.raises(ValueError):
        ExperimentDefinition.from_yaml(path)


def test_rangefinder_yaml_uses_domain_validation(tmp_path: Path) -> None:
    invalid_vector = VALID_DRAFT.replace(
        "    rangefinders: []",
        "    rangefinders:\n      - id: front\n        local_direction: [0.0, 0.0]\n        max_distance: 30.0",
    )
    with pytest.raises(ValueError):
        ExperimentDefinition.from_yaml(_write(tmp_path, invalid_vector))

    duplicate = VALID_DRAFT.replace(
        "    rangefinders: []",
        "    rangefinders:\n"
        "      - id: front\n        local_direction: [0.0, 0.0, 1.0]\n        max_distance: 30.0\n"
        "      - id: front\n        local_direction: [1.0, 0.0, 0.0]\n        max_distance: 30.0",
    )
    with pytest.raises(ValueError):
        ExperimentDefinition.from_yaml(_write(tmp_path, duplicate))


def test_invalid_seed_yaml_is_rejected(tmp_path: Path) -> None:
    for replacement in ("  ppo: [true]", "  ppo: [-1]", f"  ppo: [{2**32}]"):
        text = VALID_DRAFT.replace("  ppo: []", replacement)
        with pytest.raises(ValueError):
            ExperimentDefinition.from_yaml(_write(tmp_path, text))


def test_track_split_yaml_requires_exact_list_fields_and_valid_membership(tmp_path: Path) -> None:
    missing = VALID_DRAFT.replace("  heldout_ids:\n    - heldout-a\n", "")
    with pytest.raises(ValueError):
        ExperimentDefinition.from_yaml(_write(tmp_path, missing))

    unknown = VALID_DRAFT.replace(
        "  heldout_ids:\n    - heldout-a\n",
        "  heldout_ids:\n    - heldout-a\n  role: research\n",
    )
    with pytest.raises(ValueError):
        ExperimentDefinition.from_yaml(_write(tmp_path, unknown))

    non_list = VALID_DRAFT.replace(
        "  training_ids:\n    - train-b\n    - train-a\n",
        "  training_ids: train-a\n",
    )
    with pytest.raises(ValueError):
        ExperimentDefinition.from_yaml(_write(tmp_path, non_list))

    blank = VALID_DRAFT.replace("    - train-a", '    - "   "')
    with pytest.raises(ValueError):
        ExperimentDefinition.from_yaml(_write(tmp_path, blank))

    duplicate = VALID_DRAFT.replace("    - train-a", "    - train-b")
    with pytest.raises(ValueError):
        ExperimentDefinition.from_yaml(_write(tmp_path, duplicate))

    overlap = VALID_DRAFT.replace("    - heldout-a", "    - train-a")
    with pytest.raises(ValueError):
        ExperimentDefinition.from_yaml(_write(tmp_path, overlap))


def test_input_path_must_identify_an_existing_yaml_file(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        ExperimentDefinition.from_yaml(tmp_path / "missing.yml")
    with pytest.raises(ValueError):
        ExperimentDefinition.from_yaml(tmp_path)
    with pytest.raises(ValueError):
        ExperimentDefinition.from_yaml(123)  # type: ignore[arg-type]
