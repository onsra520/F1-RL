"""Prepare, personally sign, or execute one paired internal trial."""
import argparse
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import getpass
import json
from pathlib import Path
import secrets

from cryptography.hazmat.primitives.serialization import load_ssh_private_key
from f1rl.research.wp7_protocol import _unique
from f1rl.research.wp7_train5_internal_execution import OWNER_KEY, _owner_public_key
from scripts.wp7_penalty_trial_authorization import authorization_message, canonical, verify_trial_authorization
from scripts.wp7_penalty_trial_protocol import build_proposal, validate_proposal
from scripts.wp7_penalty_trial_runner import _require_clean_source, run_trial


def external(root, raw):
    path = Path(raw).resolve()
    if path == root or path.is_relative_to(root):
        raise ValueError('path must be outside checkout')
    return path


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='action', required=True)
    prep = sub.add_parser('prepare'); prep.add_argument('--output', required=True); prep.add_argument('--plan', required=True); prep.add_argument('--evidence', required=True)
    sign = sub.add_parser('sign'); sign.add_argument('--plan', required=True); sign.add_argument('--private-key', required=True); sign.add_argument('--packet', required=True)
    execute = sub.add_parser('execute'); execute.add_argument('--packet', required=True); execute.add_argument('--evidence', required=True)
    args = parser.parse_args()
    root = Path.cwd().resolve(strict=True)
    if args.action == 'prepare':
        _require_clean_source(root)
        output, plan = external(root, args.output), external(root, args.plan)
        if output.exists() or plan.exists(): raise ValueError('new output and plan paths required')
        from scripts.wp7_train5_batch import preflight_checkpoints
        preflight_checkpoints(root, Path(args.evidence))
        proposal = build_proposal(root)
        with plan.open('x', encoding='utf8') as f:
            json.dump({'proposal': proposal, 'output_root': str(output)}, f, indent=2); f.write('\n')
        print('Unsigned plan saved:', plan)
    elif args.action == 'sign':
        _require_clean_source(root)
        plan = json.loads(external(root, args.plan).read_text(encoding='utf8'), object_pairs_hook=_unique)
        if set(plan) != {'proposal', 'output_root'}: raise ValueError('invalid plan')
        proposal, output = plan['proposal'], external(root, plan['output_root'])
        validate_proposal(root, proposal)
        if output.exists(): raise ValueError('output already exists')
        packet = external(root, args.packet)
        if packet.exists(): raise ValueError('packet already exists')
        print(json.dumps({k: proposal[k] for k in ('source_commit', 'interventions', 'pairs', 'training_runs',
              'budget_per_run', 'total_actual_training_steps', 'evaluation_episodes', 'evaluation_contract', 'tracks', 'track_sampling', 'limitations')}, indent=2))
        print('Output:', output, '\nScope: internal simulation only; HELD-OUT is not authorized.')
        print('This packet authorizes 12 training runs followed by 7,200 paired TRAIN evaluation episodes.')
        if input('Type DONG Y TRAIN5 PENALTY to sign, or anything else to cancel: ').strip() != 'DONG Y TRAIN5 PENALTY':
            raise ValueError('owner did not confirm')
        password = getpass.getpass('Private-key passphrase (leave blank if unencrypted): ')
        try:
            key = load_ssh_private_key(Path(args.private_key).read_bytes(), password=password.encode() if password else None)
            if key.public_key().public_bytes_raw() != _owner_public_key(root, proposal['source_bindings'][OWNER_KEY]):
                raise ValueError('private key does not match enrolled owner public key')
            now = datetime.now(timezone.utc)
            payload = {'schema': 'train5_penalty_trial_authorization_v1', 'purpose': 'paired_internal_train_only',
                       'proposal_sha256': sha256(canonical(proposal)).hexdigest(), 'output_root': str(output),
                       'nonce': secrets.token_hex(16), 'issued_at_utc': now.isoformat(),
                       'expires_at_utc': (now + timedelta(hours=168)).isoformat(),
                       'accept_missing_seed_history': True, 'accept_missing_c2_replay': True,
                       'accept_internal_review_pending': True}
            envelope = {'payload': payload, 'signature_hex': key.sign(authorization_message(payload)).hex()}
        finally:
            password = None
        verify_trial_authorization(root, proposal, output, envelope)
        with packet.open('x', encoding='utf8') as f:
            json.dump({'proposal': proposal, 'output_root': str(output), 'envelope': envelope}, f, indent=2); f.write('\n')
        print('Signed packet saved:', packet, '\nTraining has not started.')
    else:
        packet = json.loads(external(root, args.packet).read_text(encoding='utf8'), object_pairs_hook=_unique)
        if set(packet) != {'proposal', 'output_root', 'envelope'}: raise ValueError('invalid packet')
        run_trial(root=root, proposal=packet['proposal'], output=external(root, packet['output_root']),
                  envelope=packet['envelope'], evidence_root=Path(args.evidence))
        from scripts.wp7_penalty_trial_evaluate import evaluate_trial
        evaluate_trial(root=root, proposal=packet['proposal'], output=external(root, packet['output_root']),
                       envelope=packet['envelope'], evidence_root=Path(args.evidence))


if __name__ == '__main__':
    main()
