"""Construct a non-authorizing, source-bound paired trial proposal."""
from hashlib import sha256
import json
from pathlib import Path
import subprocess

from f1rl.research.wp7_train5_v4_admission import development_policy_seeds, development_seeds, load_v4_protocol
from f1rl.research.wp7_train5_v4_checkpoint import FROZEN_BY_REPLICATE, FROZEN_CHECKPOINTS

DOMAIN = 'F1RL-TERMINAL-PENALTY-TRIAL-20261111-v2'
EXTRA_SOURCES = ('scripts/wp7_terminal_crash_penalty.py', 'scripts/wp7_penalty_trial_protocol.py',
                 'scripts/wp7_penalty_trial_authorization.py', 'tests/research/test_wp7_penalty_trial_authorization.py',
                 'scripts/wp7_penalty_trial_runner.py', 'tests/research/test_wp7_penalty_trial_runner.py',
                 'scripts/wp7_penalty_trial_capability.py',
                 'scripts/wp7_penalty_trial_cli.py', 'scripts/wp7_penalty_trial_evaluate.py',
                 'tests/research/test_wp7_penalty_trial_evaluate.py',
                 'tests/research/test_wp7_penalty_trial_fake_batch.py',
                 'tests/research/test_wp7_terminal_crash_penalty.py', 'tests/research/test_wp7_penalty_trial_protocol.py',
                 'configs/research/wp7-terminal-crash-penalty-trial-draft.json')


def _digest(path):
    data = path.read_bytes()
    return sha256(data if path.suffix.lower() == '.zip' else data.replace(b'\r\n', b'\n')).hexdigest()


def _seed(*parts):
    return int.from_bytes(sha256(('/'.join((DOMAIN, *map(str, parts)))).encode('ascii')).digest()[:4], 'big')


def build_proposal(root):
    root = Path(root).resolve(strict=True)
    protocol, old_sha = load_v4_protocol(root)
    registration = json.loads((root / 'configs/research/wp7-train5-paired-v2-registration.json').read_text())
    registry = json.loads((root / 'configs/research/wp7-train5-global-seed-registry-owner-exception-20261107.json').read_text())
    reserved = set(registry['candidate_seeds'])
    reserved.update(development_seeds(root, protocol, old_sha).values())
    reserved.update(development_policy_seeds(root, protocol, old_sha).values())
    journal = root.parent / 'TRAIN5-v4-internal-20261109' / 'evaluation-journal.jsonl'
    if not journal.is_file():
        raise ValueError('prior evaluation seed evidence missing')
    completed = [json.loads(line) for line in journal.read_text(encoding='utf8').splitlines()]
    completed = [row for row in completed if row.get('event') == 'completed']
    if len(completed) != 3600:
        raise ValueError('prior evaluation seed evidence incomplete')
    for row in completed:
        reserved.update((row['environment_seed'], row['policy_seed']))
    for row in protocol['runs']:
        reserved.update(row['seeds'].values())
        reserved.update(range(row['seeds']['ppo'], row['seeds']['ppo'] + 8))
    for rows in registration['proposed_training_seeds'].values():
        for row in rows:
            reserved.update(row.values())
            reserved.update(range(row['ppo'], row['ppo'] + 8))
    # Also exclude the immediately preceding exploratory replay seeds.
    audit = root.parent / 'slalom-fresh-audit.json'
    if not audit.is_file():
        raise ValueError('fresh diagnostic seed evidence missing')
    for row in json.loads(audit.read_text(encoding='utf8')):
        reserved.update((row['environment_seed'], row['policy_seed']))
    pairs, allocated = [], set()
    for cohort in ('train5_from_scratch', 'train5_from_frozen_checkpoint'):
        for rep in range(3):
            seeds = {role: _seed('train', cohort, rep, role) for role in ('ppo', 'sampling', 'perturbation')}
            ranks = set(range(seeds['ppo'], seeds['ppo'] + 8))
            all_values = set(seeds.values()) | ranks
            if (max(ranks) >= 2**32 or len(all_values) != 10 or all_values & (allocated | reserved)):
                raise ValueError('trial training seeds overlap')
            allocated.update(all_values)
            pairs.append({'cohort': cohort, 'replicate': rep, 'seeds': seeds,
                          'initial_checkpoint': FROZEN_BY_REPLICATE[rep] if cohort == 'train5_from_frozen_checkpoint' else None})
    evaluations = []
    for pair in pairs:
        for track in protocol['track_ids']:
            for condition in ('nominal', 'perturbation'):
                for episode in range(60):
                    identity = (pair['cohort'], pair['replicate'], track, condition, episode)
                    e, p = _seed('env', *identity), _seed('policy', *identity)
                    if e == p or {e, p} & (reserved | allocated):
                        raise ValueError('evaluation seed collision')
                    allocated.update((e, p))
                    evaluations.append([*identity, e, p])
    files = subprocess.run(['git', '-C', str(root), 'ls-files', '-z'], capture_output=True, check=True).stdout.decode('utf8').split('\0')
    roster = sorted(set(filter(None, files)) | set(EXTRA_SOURCES))
    bindings = {}
    for name in roster:
        path = root / name
        if path.is_symlink() or not path.resolve(strict=True).is_relative_to(root):
            raise ValueError('source path invalid')
        bindings[name] = _digest(path)
    for name, expected in FROZEN_CHECKPOINTS.items():
        if _digest(root / 'models' / name / 'model-final.zip') != expected:
            raise ValueError('original checkpoint changed')
    head = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip()
    return {'schema': 'train5_penalty_paired_proposal_v1', 'status': 'DRAFT_NOT_AUTHORIZED', 'source_commit': head,
            'old_protocol_sha256': old_sha, 'source_bindings': bindings,
            'diagnostic_seed_evidence_sha256': _digest(audit),
            'prior_evaluation_seed_evidence_sha256': _digest(journal),
            'interventions': {'baseline': 0, 'terminal_crash_penalty': 5},
            'pairs': pairs, 'training_runs': 12,
            'budget_per_run': protocol['budget'], 'total_actual_training_steps': 12 * 200704,
            'evaluation_seed_rows': evaluations, 'evaluation_episodes': len(evaluations) * 2,
            'evaluation_contract': protocol['development'],
            'tracks': protocol['track_ids'], 'track_sampling': 'unchanged_uniform',
            'limitations': ['15 historical logs missing', 'historical C2 replay missing',
                            'new seed global non-use not established', 'independent review not established'],
            'execution_permitted': False, 'heldout_permitted': False}


def require_execution_capability(*args, **kwargs):
    raise ValueError('penalty-trial runner and verified owner capability are not implemented')


def validate_proposal(root, proposal):
    if proposal != build_proposal(root):
        raise ValueError('paired trial proposal or source bindings changed')
