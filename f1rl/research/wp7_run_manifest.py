"""Run-bound preparation and byte verification, without asserting a run occurred.

Restore verification compares recorded identities, not the verifier host's
runtime. Retain the expected manifest digest through an independent channel.
"""
from copy import deepcopy
import json

from ..learning.evidence_backup import build_evidence_manifest, verify_evidence_manifest
from .wp7_protocol import build_protocol, capture_runtime_manifest, digest


def read_json(path):
    """Decode unambiguous finite evidence JSON for both CLI entrypoints."""
    def unique(pairs):
        document = {}
        for key, value in pairs:
            if key in document:
                raise ValueError('duplicate JSON key')
            document[key] = value
        return document
    def nonfinite(value):
        raise ValueError('nonfinite JSON value')
    return json.loads(path.read_text(encoding='utf-8'), object_pairs_hook=unique, parse_constant=nonfinite)


def _checked_digest(document):
    if not isinstance(document, dict) or document.get('sha256') != digest({k:v for k,v in document.items() if k != 'sha256'}):
        raise ValueError('manifest identity differs')


def capture_registered_run(binding, *, run_id):
    """Capture at the registered runner boundary before artifacts/model allocation."""
    from .wp7_execution import PolicyBoundTrainingConfig
    if type(binding) is not PolicyBoundTrainingConfig:
        raise ValueError('exact policy binding required')
    if not isinstance(run_id, str) or not run_id.strip() or len(run_id) > 128:
        raise ValueError('explicit bounded run identity required')
    binding.validate(binding.configuration, binding.scenarios)
    protocol = build_protocol(binding.root,profile=binding.profile)
    if protocol['sha256'] != binding.protocol_sha256:
        raise ValueError('protocol changed during run preparation')
    runtime = capture_runtime_manifest(binding.root, protocol)
    document = dict(schema_version='wp7_run_manifest_v1', run_id=run_id,
        state='prepared_not_started', protocol_sha256=binding.protocol_sha256,
        protocol=protocol,
        receipt=deepcopy(binding.receipt), runtime_manifest=runtime,
        artifacts=None, prepared_sha256=None, training_executed=False,
        training_authorized=False)
    record = {**document, 'sha256': digest(document)}
    verify_run_manifest(record, None, expected_sha256=record['sha256'])
    return record


def bind_run_artifacts(prepared, directory):
    """Bind existing artifact bytes. Does not infer training or timestep history."""
    verify_run_manifest(prepared, None, expected_sha256=prepared.get('sha256'))
    if prepared['state'] != 'prepared_not_started':
        raise ValueError('only a preparation manifest can bind artifacts')
    document = {**deepcopy(prepared), 'state':'artifacts_bound',
                'prepared_sha256':prepared['sha256'], 'artifacts':build_evidence_manifest(directory)}
    document.pop('sha256')
    return {**document, 'sha256':digest(document)}


def verify_run_preparation(document, *, expected_sha256):
    """Check the prepared identity without opening model or restored artifact bytes.

    An artifacts-bound envelope still needs verify_run_manifest with its actual
    directory at the later artifact boundary. This is not restore verification.
    """
    _checked_digest(document)
    if document['sha256'] != expected_sha256:
        raise ValueError('run manifest external identity differs')
    prepared = deepcopy(document)
    bound = document.get('state') == 'artifacts_bound'
    if bound:
        if not isinstance(document.get('artifacts'), dict):
            raise ValueError('bound artifact inventory required')
        prepared.update(state='prepared_not_started', artifacts=None, prepared_sha256=None)
        prepared.pop('sha256')
        prepared['sha256'] = digest(prepared)
    verify_run_manifest(prepared, None,
        expected_sha256=document['prepared_sha256'] if bound else expected_sha256)
    return dict(state='preparation_verified_artifacts_unchecked' if bound else 'preparation_verified',
        run_id=document['run_id'], policy_id=document['receipt']['policy_id'],
        protocol_sha256=document['protocol_sha256'], run_manifest_sha256=document['sha256'],
        artifact_bytes_verified=False, training_authorized=False)


def verify_run_manifest(document, directory, *, expected_sha256):
    """Independently check the hash chain and a restored exact artifact set.

    SHA checks establish byte consistency. Authentic origin requires separately
    trusted signatures and cannot be inferred from self-contained JSON.
    """
    fields = {'schema_version','run_id','state','protocol_sha256','protocol','receipt','runtime_manifest',
              'artifacts','prepared_sha256','training_executed','training_authorized','sha256'}
    if not isinstance(document, dict) or set(document) != fields:
        raise ValueError('invalid run manifest schema')
    _checked_digest(document)
    if (not isinstance(expected_sha256, str) or document['sha256'] != expected_sha256
            or document['schema_version'] != 'wp7_run_manifest_v1'
            or document['training_executed'] is not False or document['training_authorized'] is not False
            or not isinstance(document['run_id'], str) or not document['run_id'].strip()
            or len(document['run_id']) > 128):
        raise ValueError('run manifest scope or external identity differs')
    receipt, runtime = document['receipt'], document['runtime_manifest']
    protocol = document['protocol']
    _checked_digest(protocol)
    if protocol['sha256'] != document['protocol_sha256'] or protocol.get('training_authorized') is not False:
        raise ValueError('recorded protocol identity or scope differs')
    for row in (receipt, runtime):
        _checked_digest(row)
        if row.get('protocol_sha256') != document['protocol_sha256']:
            raise ValueError('protocol identity differs across manifest chain')
    if (receipt.get('schema_version') != 'wp7_policy_execution_binding_v2'
            or receipt.get('training_authorized') is not False
            or runtime.get('schema_version') != 'wp7_runtime_manifest_v1'
            or runtime.get('training_executed') is not False
            or receipt.get('policy_id') not in [p['policy_id'] for p in runtime['policies']]):
        raise ValueError('unsupported receipt/runtime scope')
    for key in ('source','source_files','policies','physics_sha256','tracks','resets',
                'requested_device','split_identity','evaluator'):
        if runtime.get(key) != protocol.get(key):
            raise ValueError('runtime and protocol identities differ')
    policy = next(p for p in protocol['policies'] if p['policy_id'] == receipt['policy_id'])
    reset = next(r for r in protocol['resets'] if r['seeds'] == policy['seeds'])
    expected = dict(seed_bundle=policy['seeds'], ppo=policy['ppo'], sampling_seed=policy['seeds']['sampling'],
        training_track_ids=policy['training_ids'], physics_sha256=protocol['physics_sha256'],
        reset_sha256=reset['sha256'], vector_environment=protocol['vector_environment'],
        race_sha256_by_track={r['track_id']:r['race_sha256'] for r in protocol['tracks']
                              if r['track_id'] in policy['training_ids']},
        action_contract={'version':protocol['action']['version']},
        observation_sha256=digest(protocol['observation']), simulation_sha256=digest(protocol['simulation']),
        checkpoint_plan=dict(training_steps=protocol['training_steps'],
            checkpoint_every_steps=protocol['checkpoint_every_steps'], selection='final_model_primary'))
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise ValueError('receipt and recorded execution protocol differ')
    count = 0
    if document['state'] == 'prepared_not_started':
        if document['artifacts'] is not None or document['prepared_sha256'] is not None or directory is not None:
            raise ValueError('preparation cannot assert artifacts')
        state = 'preparation_verified'
    elif document['state'] == 'artifacts_bound':
        if directory is None:
            raise ValueError('independent artifact directory required')
        original = {**document, 'state':'prepared_not_started', 'artifacts':None, 'prepared_sha256':None}
        original.pop('sha256')
        if document['prepared_sha256'] != digest(original):
            raise ValueError('preparation hash chain differs')
        count = verify_evidence_manifest(directory, document['artifacts'])
        state = 'artifact_bytes_verified'
    else:
        raise ValueError('unknown run manifest state')
    return dict(state=state, manifest_sha256=document['sha256'], verified_files=count,
                source_clean=runtime['source']['dirty'] is False, training_authorized=False)
