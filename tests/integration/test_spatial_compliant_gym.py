"""Actual Simulation/Gym lifecycle with continuous compliant touchdown."""
from dataclasses import replace
import numpy as np
from src.simulation.ballistic_flight import FlightState
from src.simulation.geometry.vector import Vec3
from src.rl.wp7_race import WP7RaceConfig
from tests.integration.test_spatial_ground_motion import spatial_factory


def lifecycle_factory():
    f=spatial_factory()
    physics=replace(f.physics_config,mode='wp7_lifecycle_v1',
        ground_motion_version='spatial_compliant_lifecycle_v1',spawn_reference='legacy_vehicle_position_v1',
        independent_suspension_policy=replace(f.physics_config.independent_suspension_policy,
                                             force_model='road_relative_unilateral_v3'))
    vehicle=replace(f.initial_state.vehicle,velocity=Vec3(0.,2.,0.))
    state=replace(f.initial_state,vehicle=vehicle,flight=FlightState('grounded'),
                  suspension=replace(f.initial_state.suspension,heave_velocity_mps=2.))
    return replace(f,physics_config=physics,initial_state=state,
        simulation_config=replace(f.simulation_config,fixed_dt=.01),
        navigation=replace(f.navigation,points=(vehicle.position,*f.navigation.points[1:])),
        race_config=WP7RaceConfig(version='wp7_directed_lifecycle_research_v1'))


def test_compliant_lifecycle_reaches_actual_gym_without_legacy_impulse_fallback():
    env=lifecycle_factory().create()
    try:
        observation,info=env.reset()
        transitions=[]
        for _ in range(120):
            observation,_,terminated,truncated,info=env.step(np.zeros(3,dtype=np.float32))
            assert env.observation_space.contains(observation)
            assert not terminated and not truncated
            audit=info['physics']['transition']
            transitions.extend(e['kind'] for e in audit['events'])
            assert 'spatial_compliant_lifecycle_rk4_v1' in audit['solvers']
            assert info['physics']['flight'] is None
        assert transitions[:2]==['takeoff','landing']
        assert audit['final_phase']=='grounded'
        assert env.requires_scientific_admission
    finally:
        env.close()


def test_lifecycle_configuration_rejects_missing_suspension_and_legacy_force_model():
    import pytest
    config=lifecycle_factory().physics_config
    for changes in (dict(independent_suspension_policy=None),
                    dict(independent_suspension_policy=replace(config.independent_suspension_policy,
                                                              force_model='road_relative_vertical_v2'))):
        with pytest.raises(ValueError):
            replace(config,**changes)


def moving_race_factory(*, checkpoints=(.4,.9), finish=3., max_steps=70):
    from tests.integration.test_wp7_race_contract import race_factory
    base=race_factory(checkpoints=checkpoints,finish=finish,speed=5.,max_steps=max_steps)
    config=lifecycle_factory().physics_config
    vehicle=replace(base.initial_state.vehicle,velocity=Vec3(0.,2.,5.))
    return replace(base,physics_config=config,initial_state=replace(base.initial_state,
        vehicle=vehicle,flight=FlightState('grounded'),
        suspension=replace(base.initial_state.suspension,heave_velocity_mps=2.)),
        vehicle_parameters=replace(base.vehicle_parameters,rolling_drag=0.),
        simulation_config=replace(base.simulation_config,fixed_dt=.01),
        action_contract=lifecycle_factory().action_contract,
        race_config=WP7RaceConfig(version='wp7_directed_lifecycle_research_v1'))


def test_airborne_checkpoints_then_landing_and_valid_finish_share_one_episode():
    import pytest
    env=moving_race_factory().create()
    try:
        env.reset()
        events=[]
        crossings=[]
        airborne_checkpoints=[]
        for _ in range(70):
            observation,_,terminated,truncated,info=env.step(np.zeros(3,dtype=np.float32))
            assert env.observation_space.contains(observation)
            events.extend(e['kind'] for e in info['physics']['transition']['events'])
            for name,time in zip(info['race']['directed_crossings'],info['race']['tick_local_crossing_times_s']):
                crossings.append((name,info['race']['crossing_time_origin_s']+time))
                if name.startswith('cp-'):
                    airborne_checkpoints.append(not info['telemetry']['grounded'])
            if terminated or truncated:
                break
        assert terminated and not truncated
        assert events==['takeoff','landing']
        assert [name for name,_ in crossings]==['cp-0','cp-1','finish']
        assert [time for _,time in crossings]==pytest.approx([.08,.18,.6])
        assert all(airborne_checkpoints) and info['telemetry']['grounded']
    finally:
        env.close()


def test_compliant_airborne_finish_precedes_same_tick_timeout():
    env=moving_race_factory(checkpoints=(.32,.36),finish=.39,max_steps=8).create()
    try:
        env.reset()
        for _ in range(8):
            _,_,terminated,truncated,info=env.step(np.zeros(3,dtype=np.float32))
        assert terminated and not truncated
        assert not info['telemetry']['grounded']
        assert info['race']['directed_crossings'][-1]=='finish'
    finally:
        env.close()


def test_compliant_reset_replays_and_preserves_vector_rank_identity():
    from tests.integration.test_wp7_train_reset import randomized_factory
    from src.training.env_factory import TrainingScenario,TrainingScenarioSet
    from src.training.vector_env import make_multi_track_vector_env
    from src.training.config import VectorEnvConfig
    base=randomized_factory()
    candidate=replace(base,physics_config=lifecycle_factory().physics_config,
        initial_state=replace(base.initial_state,flight=FlightState('grounded')),
        race_config=WP7RaceConfig(version='wp7_directed_lifecycle_research_v1'),
        reset_config=replace(base.reset_config,version='wp7_train_reset_lifecycle_research_v1'),
        action_contract=lifecycle_factory().action_contract)
    scenarios=TrainingScenarioSet((TrainingScenario(candidate.reset_track_id,candidate),))
    recorded=[]
    for backend in ('dummy','subproc'):
        vec=make_multi_track_vector_env(scenarios,VectorEnvConfig(n_envs=2,backend=backend),23)
        try:
            vec.seed(99)
            vec.reset()
            info=vec.step(np.zeros((2,3),dtype=np.float32))[-1]
            draws=[row['train_reset'] for row in info]
            assert [row['rank'] for row in draws]==[0,1]
            assert draws[0]['draw_seed']!=draws[1]['draw_seed']
            assert all(row['physics']['transition']['final_phase']=='grounded' for row in info)
            recorded.append(draws)
        finally:
            vec.close()
    assert recorded[0]==recorded[1]


def airborne_step_fixture():
    from src.simulation.spatial_contact_dynamics import _suspension_state
    f=lifecycle_factory()
    vehicle=replace(f.initial_state.vehicle,position=Vec3(0.,5.,4.),
                    velocity=Vec3(0.,.4905,10.),grounded=False)
    config=replace(f.physics_config,independent_suspension_policy=replace(
        f.physics_config.independent_suspension_policy,integration_substeps=1))
    state=replace(f.initial_state,vehicle=vehicle,suspension=_suspension_state(vehicle),flight=FlightState('airborne'))
    return replace(f,physics_config=config,initial_state=state)


def test_ballistic_body_cannot_tunnel_through_thin_overhead_obstacle():
    import pytest
    from src.simulation.entities.obstacle import Obstacle
    from src.simulation.track.volume import OrientedBoxVolume
    from src.simulation.ballistic_flight import FlightPhysicsError
    from src.simulation.world import step_world
    from src.simulation.vehicle import VehicleControl
    f=airborne_step_fixture()
    obstacle=Obstacle('overhead',OrientedBoxVolume(Vec3(0.,5.9112625,4.5),Vec3(.05,.0005,.05),f.initial_state.vehicle.orientation))
    track=replace(f.track,obstacles=(obstacle,))
    with pytest.raises(FlightPhysicsError,match='ballistic obstacle'):
        step_world(track,f.initial_state,VehicleControl(0.,0.,0.),f.vehicle_parameters,.1,1.,
                   **f.physics_config.world_options())


def test_transition_rejects_a_forged_ballistic_interior_with_identical_endpoints():
    import pytest
    from src.core.physics import PhysicsTransitionError
    from src.simulation.world import step_world
    from src.simulation.vehicle import VehicleControl
    f=airborne_step_fixture()
    r=step_world(f.track,f.initial_state,VehicleControl(0.,0.,0.),f.vehicle_parameters,.1,1.,
                 **f.physics_config.world_options())
    segment=r.center_segments[0]
    h=segment.end_time_s-segment.start_time_s
    gravity=2.*segment.gravity_mps2
    forged=replace(segment,gravity_mps2=gravity,
                   initial_velocity=(segment.end-segment.start)/h+Vec3(0.,.5*gravity*h,0.))
    with pytest.raises(PhysicsTransitionError,match='ballistic'):
        f.physics_config.validate_transition(replace(r,center_segments=(forged,)),
            previous=f.initial_state,parameters=f.vehicle_parameters,dt=.1,start_time_s=0.,end_time_s=.1)
