"""Synthetic integrated v4 crash: Simulation -> Gym -> reward/event prefix."""
from dataclasses import replace
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import numpy as np
from src.simulation.ballistic_flight import FlightState
from src.simulation.entities.obstacle import Obstacle
from src.simulation.track.volume import OrientedBoxVolume
from src.simulation.geometry.vector import Vec3
from src.simulation.spatial_contact_dynamics import _suspension_state
from src.rl.wp7_race import WP7RaceConfig
from src.rl.rewards import RewardComposer,TimePenalty
from src.evaluation.evaluator import _valid_terminal_failure
from tests.integration.test_spatial_ground_motion import spatial_factory
f=spatial_factory()
p=replace(f.physics_config,mode="wp7_lifecycle_v1",
          ground_motion_version="spatial_road_edge_partial_lifecycle_v4",
          independent_suspension_policy=replace(f.physics_config.independent_suspension_policy,
                                                force_model="road_relative_unilateral_v3"))
v=replace(f.initial_state.vehicle,velocity=Vec3(0.,0.,5.))
s=replace(f.initial_state,vehicle=v,suspension=_suspension_state(v),
          flight=FlightState("grounded"))
obstacle=Obstacle("crash-1",OrientedBoxVolume(Vec3(0.,.8,6.4),
        Vec3(2.,2.,.1),v.orientation))
f=replace(f,physics_config=p,initial_state=s,
    track=replace(f.track,obstacles=(obstacle,)),
    race_config=WP7RaceConfig(version="wp7_directed_lifecycle_crash_dev_v3",
                             hard_violation="terminate_valid_failure"),
    reward_composer=RewardComposer((TimePenalty(2.),)))
env=f.create()
try:
    obs,info=env.reset()
    observation,reward,terminated,truncated,info=env.step(np.zeros(3,dtype=np.float32))
    print("CRASH_GYM",terminated,truncated,round(reward,9),
          info["research_outcome"],info["race"]["terminal_reason"])
    print("CHECK",_valid_terminal_failure(info),
          "time",info["race"]["terminal_time_s"],
          "prefix",info["race"]["reward_valid_prefix_s"],
          "crossings",info["race"]["directed_crossings"])
    print("IMPACT",info["physical_crash_terminal"])
finally:env.close()
