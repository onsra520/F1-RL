"""Internal trusted-pipe PPO fixture worker; not a public evidence decoder.

The parent passes a pickled, locally constructed software factory and verified
archive bytes over stdin. Never feed untrusted data to this internal IPC entry.
Framework-specific RNG operations stay outside the dependency-free core.
"""
import json
import pickle
import sys
from hashlib import sha256
from f1rl.research.wp7_cell_evaluation import _execute_cell, _fixture_identity
from f1rl.research.wp7_protocol import digest
from f1rl.learning.training.env_factory import F1RLEnvFactory


def run_fixture(payload):
    """Only software fixtures enter this process; all policy RNG state is local."""
    try:
        from io import BytesIO
        import random
        import numpy as np
        import torch
        from f1rl.learning.algorithms.ppo import PPO, PPOModel

        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        factory = payload["factory"]
        if type(factory) is not F1RLEnvFactory or factory.requires_scientific_admission:
            raise ValueError("research physics forbidden in software worker")
        if _fixture_identity(factory) != payload['fixture']:
            raise ValueError("worker fixture identity mismatch")
        blob = payload["model_bytes"]
        if sha256(blob).hexdigest() != payload["model"]["sha256"]:
            raise ValueError("worker model identity mismatch")
        traces = []

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
            episodes=payload["episodes"],factory_for_episode=lambda i: factory,
            policy_for_seed=policy_for_seed,scope="synthetic_saved_ppo_fixture_not_scientific_evidence")
        for row,trace in zip(record["episodes"],traces):
            row.update(policy_actions_sha256=trace.sha.hexdigest(),policy_action_count=trace.count,
                       policy_rng_start_sha256=trace.rng_start,policy_rng_end_sha256=trace.rng_end)
        record["schema_version"] = "wp7_isolated_ppo_fixture_cell_v1"
        record["fixture"] = payload["fixture"]
        record["worker"] = {"isolation":"spawn_process","device":payload["device"],
                            "torch_version":str(torch.__version__),"rng":"python_numpy_torch_seed_after_load_v1"}
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
    pickle.dump(run_fixture(pickle.load(sys.stdin.buffer)),sys.stdout.buffer)
