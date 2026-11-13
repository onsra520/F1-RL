import json

from f1rl.learning.evaluation.metrics import EpisodeMetrics
from scripts.m5_c0_diagnostic import predict_policy_action, serialize_episode_metrics, terminal_summary


def test_episode_metrics_dataclass_serializes_to_json_without_model_dump():
    metrics = EpisodeMetrics(
        track_id="ramp-v1", success=False, truncated=True, steps=500,
        episode_time=50.0, completion_time=None, episode_return=-5.0,
        reward_component_totals=(("route_progress", 0.0), ("time", -5.0)),
        collision_count=0, path_length=50.0, final_normalized_route_progress=0.8,
        mean_speed=2.0, max_speed=4.0, airborne_steps=0,
        boost_applied_steps=0, slow_zone_applied_steps=0,
    )

    serialized = serialize_episode_metrics(metrics)
    encoded = json.dumps(serialized, sort_keys=True, allow_nan=False)

    assert json.loads(encoded)["steps"] == 500
    assert json.loads(encoded)["reward_component_totals"] == [["route_progress", 0.0], ["time", -5.0]]


def test_simulation_step_limit_is_not_reported_as_wall_timeout():
    metrics = EpisodeMetrics(
        track_id="ramp-v1", success=False, truncated=True, steps=500,
        episode_time=50.0, completion_time=None, episode_return=-5.0,
        reward_component_totals=(("route_progress", 0.0), ("time", -5.0)),
        collision_count=0, path_length=50.0, final_normalized_route_progress=0.8,
        mean_speed=2.0, max_speed=4.0, airborne_steps=0,
        boost_applied_steps=0, slow_zone_applied_steps=0,
    )
    terminal = terminal_summary(metrics, {
        "research_outcome": None, "physical_crash_terminal": None,
        "next_checkpoint_index": 1, "completed_checkpoint_ids": [],
        "collision_obstacle_ids": [],
    })

    assert terminal["reason"] == "step_limit"
    assert terminal["simulation_timeout"] is True
    assert terminal["wall_timeout"] is False


def test_ppo_model_prediction_is_used_as_the_action_array():
    class Model:
        def predict(self, observation, *, deterministic):
            return observation

    action = [0.25, 0.75, 0.0]

    assert predict_policy_action(Model(), action, deterministic=False) is action
