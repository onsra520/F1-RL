"""Run one unregistered TRAIN-development smoke with the bundled A-seed0 PPO."""
from pathlib import Path
import sys

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from src.learning.algorithms.ppo import PPOModel
from src.research.wp3_eval_only_coordinator import _production_episode

model_path = root / 'docs/research/evidence/wp3-issue51-frozen-final-models-20261028/models/A-seed0/model-final.zip'
model = PPOModel.load(model_path, device='cpu')
case = {'model_id': 'A-seed0', 'seed_index': 0, 'track': 'straight-sprint-v1', 'stratum': 'nominal', 'env_seed': 123, 'policy_seed': 456}
print('checkpoint timesteps:', model.num_timesteps, flush=True)
print('episode:', _production_episode(root, root, case, {'A-seed0': model}))
