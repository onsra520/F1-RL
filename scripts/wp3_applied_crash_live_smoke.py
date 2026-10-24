from dataclasses import replace
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from f1rl.runtime.simulation import Simulation
from f1rl.sandbox.ballistic_flight import FlightState
from f1rl.sandbox.entities.obstacle import Obstacle
from f1rl.sandbox.track.volume import OrientedBoxVolume
from f1rl.sandbox.geometry.vector import Vec3
from f1rl.sandbox.vehicle import VehicleControl
from f1rl.sandbox.spatial_contact_dynamics import _suspension_state
from tests.integration.test_spatial_ground_motion import spatial_factory
f=spatial_factory()
p=replace(f.physics_config,mode="wp7_lifecycle_v1",ground_motion_version="spatial_road_edge_partial_lifecycle_v4",independent_suspension_policy=replace(f.physics_config.independent_suspension_policy,force_model="road_relative_unilateral_v3"))
v=replace(f.initial_state.vehicle,velocity=Vec3(0.,0.,5.))
s=replace(f.initial_state,vehicle=v,suspension=_suspension_state(v),flight=FlightState("grounded"))
obs=Obstacle("crash-1",OrientedBoxVolume(Vec3(0.,.8,6.4),Vec3(2.,2.,.1),v.orientation))
t=replace(f.track,obstacles=(obs,))
sim=Simulation(t,s,f.vehicle_parameters,f.simulation_config,physics_config=p)
try:
    res=sim.step(VehicleControl(0.,0.,0.))
    print("SIM_SUCCESS","crash",res.physical_crash,"end",res.world.vehicle,"solvers",res.physics_transition.solvers)
except Exception as e:
    print("SIM_ERROR",type(e).__name__,str(e))
    import traceback;traceback.print_exc()
