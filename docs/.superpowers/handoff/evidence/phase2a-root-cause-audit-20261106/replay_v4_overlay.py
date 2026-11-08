from __future__ import annotations
import dataclasses, hashlib, json, math, sys, traceback
from dataclasses import replace
from pathlib import Path
import numpy as np
import torch
ROOT=Path(r'D:\Projects\.worktrees\F1-RL\physics-v4-phase1-audit').resolve()
OUT=Path(r'D:\Projects\.evidence\F1-RL\phase2a-root-cause-audit-20261106')
PHASE1=Path(r'D:\Projects\.evidence\F1-RL\phase1-physics-v4-audit-20261106')
sys.path.insert(0,str(ROOT)); torch.set_num_threads(1)
from f1rl.learning.algorithms.ppo import PPOModel
from f1rl.research.wp7_protocol import load_registration,prepare_scenarios

def safe(v):
 if dataclasses.is_dataclass(v): return safe(dataclasses.asdict(v))
 if isinstance(v,dict): return {str(k):safe(x) for k,x in v.items()}
 if isinstance(v,(tuple,list)): return [safe(x) for x in v]
 if isinstance(v,np.ndarray): return safe(v.tolist())
 if isinstance(v,(np.floating,float)):
  x=float(v); return x if math.isfinite(x) else str(x)
 if isinstance(v,(np.integer,int)): return int(v)
 if isinstance(v,(np.bool_,bool)): return bool(v)
 if v is None or isinstance(v,(str,bool)): return v
 if hasattr(v,'__dict__'): return safe(vars(v))
 return str(v)
def vec(v): return [float(v.x),float(v.y),float(v.z)]
def state(v):
 q=v.orientation
 return {'position_m':vec(v.position),'velocity_mps':vec(v.velocity),'speed_mps':float(v.speed),'orientation_wxyz':[q.w,q.x,q.y,q.z],'angular_velocity_radps':vec(v.angular_velocity),'grounded':bool(v.grounded)}
summary=json.loads((PHASE1/'smoke-summary.json').read_text(encoding='utf-8'))
invalid=[r for r in summary['runs'] if r['classification']=='invalid_physical_transition']
reg=load_registration(ROOT,profile='spatial_rotating_owner_v3')
scenarios=prepare_scenarios(ROOT,reg,seed_index=0)
by_track={x.track_id:x for x in scenarios.scenarios}
models={}; rows=[]
for original in invalid:
 mid,tid=original['model_id'],original['track_id']
 model=models.setdefault(mid,PPOModel.load(ROOT/'models'/mid/'model-final.zip',device='cpu'))
 factory=by_track[tid].factory
 suspension=replace(factory.physics_config.independent_suspension_policy,max_abs_tilt_rad=.7)
 factory=replace(factory,physics_config=replace(factory.physics_config,ground_motion_version='spatial_road_edge_partial_lifecycle_v4',independent_suspension_policy=suspension),race_config=replace(factory.race_config,version='wp7_directed_lifecycle_crash_contract_v4'))
 env=factory.create(); env.set_training_telemetry_enabled(True)
 obs,reset_info=env.reset(seed=summary['reset_seed'])
 row={'model_id':mid,'track_id':tid,'reset_seed':summary['reset_seed'],'inference_mode':'deterministic','counterfactual_overlay':'FULL-06 physics/race/tilt runtime overlay','physics_version':factory.physics_config.ground_motion_version,'race_version':factory.race_config.version,'max_abs_tilt_rad':factory.physics_config.independent_suspension_policy.max_abs_tilt_rad,'phase1_failed_attempt_step':original['steps']+1,'asset_sha256':hashlib.sha256((ROOT/'assets'/'tracks'/'training'/f'{tid}.json').read_bytes()).hexdigest(),'outcome':'not_started','reset_info':safe(reset_info)}
 try:
  for step in range(1,summary['step_limit']+1):
   before=env._simulation.snapshot
   action=model.predict(obs,deterministic=True)
   try: out=env.step(action)
   except Exception as e:
    row.update({'outcome':'exception','step':step,'pre_transition_vehicle':state(before.world.vehicle),'action':safe(action),'exception':{'class':type(e).__name__,'message':str(e),'failure_code':getattr(e,'failure_code',None),'traceback':traceback.format_exc()},'research_outcome':safe(env.last_research_outcome)})
    break
   obs,reward,terminated,truncated,info=out
   if terminated or truncated:
    row.update({'outcome':'terminal','step':step,'terminated':bool(terminated),'truncated':bool(truncated),'terminal_info':safe(info),'final_vehicle':state(env._simulation.snapshot.world.vehicle),'research_outcome':safe(env.last_research_outcome)})
    break
  else: row['outcome']='step_limit'
 except Exception as e:
  row.update({'outcome':'driver_error','exception':{'class':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()}})
 finally: env.close()
 rows.append(row)
res={'summary':{'cases':len(rows),'counts':{s:sum(r['outcome']==s for r in rows) for s in sorted({r['outcome'] for r in rows})},'rows':[{'model_id':r['model_id'],'track_id':r['track_id'],'outcome':r['outcome'],'step':r.get('step'),'terminal_reason':r.get('terminal_info',{}).get('terminal_reason') if isinstance(r.get('terminal_info'),dict) else None,'exception':r.get('exception',{}).get('message')} for r in rows]},'cases':rows}
(OUT/'v4-overlay-counterfactual.json').write_text(json.dumps(res,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
print(json.dumps(res['summary'],indent=2,ensure_ascii=False))
