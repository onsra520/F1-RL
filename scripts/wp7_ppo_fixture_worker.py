"""Internal trusted-pipe PPO fixture worker; not a public evidence decoder.

The parent passes a pickled, locally constructed software factory and verified
archive bytes over stdin. Never feed untrusted data to this internal IPC entry.
Framework-specific RNG operations stay outside the dependency-free core.
"""
import json
import pickle
import sys
from hashlib import sha256
from pathlib import Path
from f1rl.research.wp7_cell_evaluation import (
    _execute_cell, _fixture_identity, _registered_cell_admission, _registered_model_snapshot,
)
from f1rl.research.wp7_protocol import digest, evaluation_seed, prepare_evaluation_factory, validate_protocol
from f1rl.learning.training.env_factory import F1RLEnvFactory


def run_fixture(payload):
    return _run_cell(payload,registered=False)


def run_registered(payload):
    return _run_cell(payload,registered=True)


def _run_cell(payload,*,registered):
    """The child checks actual admission; pipe booleans cannot authorize research."""
    try:
        root=Path(__file__).resolve().parents[1]
        protocol=payload['protocol']
        environments={}
        if registered:
            evaluation_seed(protocol,payload['policy_id'],payload['track_id'],payload['stratum'],0)
            admission=_registered_cell_admission(root,protocol,payload['policy_id'],
                model_sha256=payload.get('expected_model_sha256'),
                run_manifest_sha256=payload.get('expected_run_manifest_sha256'))
            blob,model=_registered_model_snapshot(protocol=protocol,policy_id=payload['policy_id'],
                model_path=Path(payload['model_path']),expected_model_sha256=payload['expected_model_sha256'],
                run_manifest=payload['run_manifest'],expected_run_manifest_sha256=payload['expected_run_manifest_sha256'],
                artifact_directory=Path(payload['artifact_directory']))
            if blob!=payload['model_bytes'] or model!=payload['model']:
                raise ValueError('registered model snapshot changed across worker boundary')
            if (type(payload['episodes']) is not int
                    or payload['episodes']!=protocol['registration']['evaluation']['episodes_per_cell']
                    or payload['device']!=protocol['requested_device']):
                raise ValueError('registered episode count or device differs')
            def factory_for_episode(index):
                factory,identity=prepare_evaluation_factory(root,protocol,payload['policy_id'],
                    payload['track_id'],payload['stratum'],index)
                if type(factory) is not F1RLEnvFactory or not factory.requires_scientific_admission:
                    raise ValueError('registered evaluator requires its exact research factory')
                environments[index]=identity
                return factory
            scope='registered_ppo_inference_not_independent_scientific_acceptance'
        else:
            factory=payload['factory']
            if type(factory) is not F1RLEnvFactory or factory.requires_scientific_admission:
                raise ValueError('research physics forbidden in software worker')
            if _fixture_identity(factory)!=payload['fixture']:
                raise ValueError('worker fixture identity mismatch')
            factory_for_episode=lambda index:factory
            blob=payload['model_bytes']
            scope='synthetic_saved_ppo_fixture_not_scientific_evidence'
        if sha256(blob).hexdigest()!=payload['model']['sha256']:
            raise ValueError('worker model identity mismatch')
        if payload['device'] not in ('cpu','cuda'):
            raise ValueError('explicit worker cpu or cuda device required')
        from io import BytesIO
        import random
        import numpy as np
        import torch
        from f1rl.learning.algorithms.ppo import PPO, PPOModel

        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        if payload['device']=='cuda' and not torch.cuda.is_available():
            raise RuntimeError('explicit CUDA device unavailable in worker')
        traces = []
        loaded_models = []

        class TracedPolicy:
            def __init__(self, model):
                self.model = model
                self.sha = sha256()
                self.count = 0
                self.rng_start = self.rng_digest()
                self.rng_end = self.rng_start

            def rng_digest(self):
                state = (torch.cuda.get_rng_state() if payload["device"] == "cuda"
                         else torch.random.get_rng_state())
                return sha256(state.cpu().numpy().tobytes()).hexdigest()

            def predict(self, observation, *, deterministic=True):
                if deterministic:
                    raise ValueError("software worker requires stochastic prediction")
                action = self.model.predict(observation,deterministic=False)
                self.rng_end = self.rng_digest()
                self.sha.update(json.dumps(action.tolist(),separators=(",",":"),allow_nan=False).encode())
                self.count += 1
                return action

        def policy_for_seed(seed):
            # Load first: SB3 setup may reset global RNGs from the archive.
            backend = PPO.load(BytesIO(blob),device=payload["device"])
            if payload["device"] == "cuda" and backend.device.type != "cuda":
                raise RuntimeError("explicit CUDA device was not honored")
            if registered:
                policy=next(p for p in protocol['policies'] if p['policy_id']==payload['policy_id'])
                if (backend.seed!=policy['seeds']['ppo']
                        or backend.observation_space.shape!=(len(protocol['observation']['features']),)
                        or not np.array_equal(backend.action_space.low,np.asarray(protocol['action']['low'],dtype=np.float32))
                        or not np.array_equal(backend.action_space.high,np.asarray(protocol['action']['high'],dtype=np.float32))):
                    raise ValueError('loaded PPO seed or observation/action contract differs')
                loaded_models.append(dict(ppo_seed=backend.seed,num_timesteps=int(backend.num_timesteps),
                    observation_shape=list(backend.observation_space.shape),
                    action_shape=list(backend.action_space.shape)))
            model = PPOModel(backend)
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if payload["device"] == "cuda":
                torch.cuda.manual_seed_all(seed)
            traced = TracedPolicy(model)
            traces.append(traced)
            return traced

        record = _execute_cell(protocol=payload["protocol"],policy_id=payload["policy_id"],
            track_id=payload["track_id"],stratum=payload["stratum"],model=payload["model"],
            episodes=payload["episodes"],factory_for_episode=factory_for_episode,
            policy_for_seed=policy_for_seed,scope=scope)
        for row,trace in zip(record["episodes"],traces):
            row.update(policy_actions_sha256=trace.sha.hexdigest(),policy_action_count=trace.count,
                       policy_rng_start_sha256=trace.rng_start,policy_rng_end_sha256=trace.rng_end)
        record["schema_version"] = 'wp7_isolated_registered_ppo_cell_v1' if registered else 'wp7_isolated_ppo_fixture_cell_v1'
        if registered:
            for row in record['episodes']:
                row['environment_identity']=environments.get(row['index'])
            record['run_manifest_sha256']=payload['expected_run_manifest_sha256']
            validate_protocol(protocol,root)
        else:
            record['fixture']=payload['fixture']
        record["worker"] = {"isolation":"spawn_process","device":payload["device"],
                            "torch_version":str(torch.__version__),"rng":"python_numpy_torch_seed_after_load_v1"}
        if registered:
            record['worker']['runtime']=admission['runtime']
            record['worker']['loaded_models']=loaded_models
        valid = [r for r in record["episodes"] if r['valid']]
        successes = sum(r['metrics']['success'] for r in valid)
        from f1rl.research.wp7_robustness_gate import one_sided_cp_lower
        lower = one_sided_cp_lower(successes,len(valid)) if valid else None
        record["numerical_summary"] = {
            "valid_trials":len(valid),"successes":successes,"one_sided_cp95_lower":lower,
            "minimum_trials":60,"required_lower_bound":.90,
            "meets_cell_numeric_criterion":record['status']=='completed' and len(valid)>=60 and lower>=.90,
            "cohort_multiplicity_review_required":True,"scientific_acceptance":False,
        }
        record["sha256"] = digest({k:v for k,v in record.items() if k != "sha256"})
        return ("ok",record)
    except Exception as exc:
        return ("error",type(exc).__name__,str(exc))



if __name__ == "__main__":
    request=pickle.load(sys.stdin.buffer)
    kind=request.get('worker_kind','fixture')
    result=(run_registered(request) if kind=='registered' else run_fixture(request) if kind=='fixture'
            else ('error','ValueError','unknown PPO worker kind'))
    pickle.dump(result,sys.stdout.buffer)
