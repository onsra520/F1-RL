"""Strict validator for the draft, TRAIN-only A/B/C/D experiment manifest."""
import json
from hashlib import sha256
from pathlib import Path
import subprocess

from .reward import validate as validate_demo
from ..research.wp7_protocol import digest


MANIFEST = 'configs/research/straight-demo-abcd-v1.json'
STATUS = 'DRAFT — NOT PREREGISTERED — NOT AUTHORIZED'
EXPECTED = (
    ('A', 2.0, 2.0, 'straight_route_stability_demo_v1'),
    ('B', 4.0, 2.0, 'straight_route_stability_demo_v1'),
    ('C', 2.0, 3.0, 'straight_route_stability_demo_v2'),
    ('D', 4.0, 3.0, 'straight_route_stability_demo_v2'),
)


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate experiment manifest field')
        result[key] = value
    return result


def _nonfinite(value):
    raise ValueError('nonfinite experiment manifest value')


def validate_abcd_manifest(value):
    fields = {'schema_version', 'status', 'split', 'evaluation', 'training',
              'statistical_protocol', 'candidates'}
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError('invalid A/B/C/D manifest schema')
    if (value['schema_version'] != 'f1rl_abcd_experiment_manifest_v1'
            or value['status'] != STATUS or value['split'] != 'TRAIN'):
        raise ValueError('A/B/C/D manifest is not the registered draft TRAIN contract')
    evaluation = value['evaluation']
    if (not isinstance(evaluation, dict)
            or evaluation != {'task': 'registered_original_task', 'hard_route_boundary_m': 4.0,
                              'semantics_unchanged': True}):
        raise ValueError('registered evaluation boundary and semantics must remain unchanged')
    training = value['training']
    expected_training = {
        'learning_timesteps_per_candidate_seed': 16384,
        'checkpoint_every_timesteps': 1024,
        'physics_unchanged': True,
        'observation_action_contracts_unchanged': True,
        'selection': 'final_checkpoint_only',
        'approval_state': 'proposal_only',
    }
    if not isinstance(training, dict) or training != expected_training:
        raise ValueError('unapproved or scope-escalated A/B/C/D training settings')
    statistics = value['statistical_protocol']
    expected_statistics = {
        'version': 'wp2_abcd_statistical_protocol_v1',
        'tuning_seed_blocks': 3,
        'confirmation_seed_blocks': 12,
        'tuning_seed_bundles': [[0, 0, 1000], [1, 1, 1001], [2, 2, 1002]],
        'confirmation_seed_bundle_rule': {'ppo_start': 100, 'sampling_start': 200,
                                          'training_reset_start': 1100, 'count': 12},
        'episodes_per_track_stratum_per_model': 60,
        'evaluation_tracks': ['straight-sprint-v1', 'slalom-v1', 'right-turn-v1', 'ramp-v1'],
        'evaluation_strata': ['nominal', 'randomized_train'],
        'primary_endpoint': 'valid_autonomous_finish_probability',
        'primary_contrasts': ['reward_main_effect', 'corridor_main_effect', 'interaction'],
        'effect_interval': 'two_sided_95_percent_seed_block_cluster_bootstrap_10000',
        'multiplicity': 'holm_three_primary_contrasts_alpha_0.05',
        'evaluation_seed_derivation': 'sha256_manifest_track_stratum_episode_candidate_independent',
        'robustness_rule': 'each_model_track_stratum_one_sided_95_percent_clopper_pearson_lower_at_least_0.90',
        'invalid_episode_rule': 'abort_cell_no_redraw',
        'approval_state': 'proposal_only',
    }
    if not isinstance(statistics, dict) or statistics != expected_statistics:
        raise ValueError('unapproved or scope-escalated statistical protocol')
    candidates = value['candidates']
    if not isinstance(candidates, list) or len(candidates) != len(EXPECTED):
        raise ValueError('exact A/B/C/D candidate roster required')
    for row, (candidate_id, lateral, corridor, version) in zip(candidates, EXPECTED, strict=True):
        if (not isinstance(row, dict)
                or set(row) != {'candidate_id', 'lateral_penalty_per_m2_s', 'demo_training'}
                or row['candidate_id'] != candidate_id
                or type(row['lateral_penalty_per_m2_s']) not in (int, float)
                or row['lateral_penalty_per_m2_s'] != lateral):
            raise ValueError('candidate identity or reward factor differs from draft definition')
        demo = validate_demo(row['demo_training'])
        if (demo['version'] != version or demo['training_corridor_m'] != corridor
                or demo['lateral_penalty_per_m2_s'] != lateral or demo['terminal_penalty'] != 25.0):
            raise ValueError('candidate demo reward/corridor differs from versioned definition')
    return {**value, 'sha256': digest(value)}


def load_abcd_manifest(repository_root):
    root = Path(repository_root).resolve(strict=True)
    path = root / MANIFEST
    if path.is_symlink() or not path.resolve(strict=True).is_relative_to(root):
        raise ValueError('experiment manifest must be a repository file')
    value = json.loads(path.read_text(encoding='utf-8'), object_pairs_hook=_unique,
                       parse_constant=_nonfinite)
    return validate_abcd_manifest(value)


def _source_identity(root):
    def git(*args):
        return subprocess.run(['git', *args], cwd=root, check=True, capture_output=True).stdout.decode().strip()
    if git('status', '--porcelain', '--untracked-files=all'):
        raise ValueError('clean source tree required for a run binding')
    commit = git('rev-parse', '--verify', 'HEAD')
    tree = git('rev-parse', '--verify', 'HEAD^{tree}')
    if (len(commit) != 40 or any(c not in '0123456789abcdef' for c in commit)
            or len(tree) != 40 or any(c not in '0123456789abcdef' for c in tree)):
        raise ValueError('canonical Git source identity required')
    return commit, sha256(b'f1rl-abcd-git-tree-v1\x00' + bytes.fromhex(tree)).hexdigest()


def build_candidate_run_manifest(repository_root, *, candidate_id, run_id, seed_bundle):
    """Bind the draft factors, statistical design and exact clean source to one run ID."""
    root = Path(repository_root).resolve(strict=True)
    manifest = load_abcd_manifest(root)
    row = next((item for item in manifest['candidates'] if item['candidate_id'] == candidate_id), None)
    if row is None or not isinstance(run_id, str) or not run_id.strip() or len(run_id) > 128:
        raise ValueError('registered candidate and bounded run identity required')
    if (not isinstance(seed_bundle, dict) or set(seed_bundle) != {'ppo', 'sampling', 'training_reset'}
            or any(type(seed_bundle[key]) is not int or not 0 <= seed_bundle[key] <= 2**32-1
                   for key in seed_bundle)):
        raise ValueError('complete nonnegative seed bundle required')
    statistics = manifest['statistical_protocol']
    tuning = {tuple(row) for row in statistics['tuning_seed_bundles']}
    rule = statistics['confirmation_seed_bundle_rule']
    confirmation = {(rule['ppo_start'] + i, rule['sampling_start'] + i,
                     rule['training_reset_start'] + i) for i in range(rule['count'])}
    bundle = (seed_bundle['ppo'], seed_bundle['sampling'], seed_bundle['training_reset'])
    if bundle not in tuning | confirmation:
        raise ValueError('seed bundle is outside the proposed disjoint tuning/confirmation roster')
    commit, tree_sha256 = _source_identity(root)
    body = dict(schema_version='wp2_abcd_run_manifest_v1', state='prepared_not_authorized',
        run_id=run_id, split='TRAIN', candidate_id=candidate_id, candidate=row,
        source_commit=commit, source_tree_sha256=tree_sha256,
        experiment_manifest_sha256=manifest['sha256'], statistical_protocol=manifest['statistical_protocol'],
        seed_role='tuning' if bundle in tuning else 'confirmation',
        seed_bundle=dict(seed_bundle), training=manifest['training'], evaluation=manifest['evaluation'],
        training_executed=False, training_authorized=False)
    return {**body, 'sha256': digest(body)}


def verify_candidate_run_manifest(repository_root, run_manifest):
    """Validate a prepared binding and require it still names current clean source."""
    root = Path(repository_root).resolve(strict=True)
    manifest = load_abcd_manifest(root)
    fields = {'schema_version', 'state', 'run_id', 'split', 'candidate_id', 'candidate',
        'source_commit', 'source_tree_sha256', 'experiment_manifest_sha256', 'statistical_protocol',
        'seed_role', 'seed_bundle', 'training', 'evaluation', 'training_executed', 'training_authorized', 'sha256'}
    if not isinstance(run_manifest, dict) or set(run_manifest) != fields:
        raise ValueError('invalid A/B/C/D run manifest schema')
    if run_manifest['sha256'] != digest({key: value for key, value in run_manifest.items() if key != 'sha256'}):
        raise ValueError('A/B/C/D run manifest checksum mismatch')
    commit, tree_sha256 = _source_identity(root)
    candidate = next((row for row in manifest['candidates']
                      if row['candidate_id'] == run_manifest['candidate_id']), None)
    if (not isinstance(run_manifest['run_id'], str) or not run_manifest['run_id'].strip()
            or len(run_manifest['run_id']) > 128
            or run_manifest['schema_version'] != 'wp2_abcd_run_manifest_v1'
            or run_manifest['state'] != 'prepared_not_authorized'
            or run_manifest['split'] != 'TRAIN'
            or run_manifest['training_executed'] is not False
            or run_manifest['training_authorized'] is not False
            or run_manifest['source_commit'] != commit
            or run_manifest['source_tree_sha256'] != tree_sha256
            or run_manifest['experiment_manifest_sha256'] != manifest['sha256']
            or run_manifest['statistical_protocol'] != manifest['statistical_protocol']
            or run_manifest['training'] != manifest['training']
            or run_manifest['evaluation'] != manifest['evaluation']
            or run_manifest['candidate'] != candidate):
        raise ValueError('A/B/C/D run binding is stale or inconsistent with current source/manifest')
    if not isinstance(run_manifest['seed_bundle'], dict) or set(run_manifest['seed_bundle']) != {
            'ppo', 'sampling', 'training_reset'} or any(
            type(seed) is not int or not 0 <= seed <= 2**32-1 for seed in run_manifest['seed_bundle'].values()):
        raise ValueError('run manifest seed bundle malformed')
    bundle = (run_manifest['seed_bundle']['ppo'], run_manifest['seed_bundle']['sampling'],
              run_manifest['seed_bundle']['training_reset'])
    tuning = {tuple(row) for row in manifest['statistical_protocol']['tuning_seed_bundles']}
    rule = manifest['statistical_protocol']['confirmation_seed_bundle_rule']
    confirmation = {(rule['ppo_start'] + i, rule['sampling_start'] + i,
                     rule['training_reset_start'] + i) for i in range(rule['count'])}
    expected_role = 'tuning' if bundle in tuning else 'confirmation' if bundle in confirmation else None
    if run_manifest['seed_role'] != expected_role:
        raise ValueError('run seed is not in the disjoint preregistered seed roster')
    return {key: run_manifest[key] for key in ('source_commit', 'source_tree_sha256',
        'experiment_manifest_sha256', 'run_id', 'candidate_id', 'split')} | {
            'run_manifest_sha256': run_manifest['sha256']}


def authorize_candidate_run(repository_root, run_manifest, envelope, replay_directory, *, now=None):
    """First-side-effect gate for a future A/B/C/D worker; no runner is enabled here."""
    from ..research.wp2_execution_permit import claim_repository_wp2_execution_permit
    expected = verify_candidate_run_manifest(repository_root, run_manifest)
    return claim_repository_wp2_execution_permit(repository_root, envelope, expected=expected,
        replay_directory=replay_directory, now=now)
