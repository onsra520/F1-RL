"""Verify a scoped signature for review; this module does not grant training."""
from datetime import datetime, timezone
import json

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from f1rl.research.wp7_trusted_reviews import _fields, _hex, _utc


def authorization_message(payload):
    return b"F1-RL/TRAIN5/internal-execution-review/v1\x00" + json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("ascii")


def verify_authorization(envelope, public_key, *, protocol_sha256, manifest_sha256,
                         source_commit, cohort, replicate, output_root, now=None):
    """Check bytes and scope only; a caller-supplied key is not a trust anchor."""
    _fields(envelope, "payload signature_hex")
    payload = envelope["payload"]
    _fields(payload, "schema purpose source_commit protocol_sha256 manifest_sha256 cohort replicate output_root nonce issued_at_utc expires_at_utc accept_missing_seed_history accept_missing_c2_replay accept_internal_review_pending")
    _hex(protocol_sha256, 32)
    _hex(manifest_sha256, 32)
    _hex(source_commit, 20)
    if (payload["schema"] != "train5_local_execution_review_v1"
            or payload["purpose"] != "simulation_only_train5"
            or payload["source_commit"] != source_commit
            or payload["protocol_sha256"] != protocol_sha256
            or payload["manifest_sha256"] != manifest_sha256
            or cohort not in ("train5_from_scratch", "train5_from_frozen_checkpoint")
            or payload["cohort"] != cohort or type(replicate) is not int or replicate not in range(3)
            or type(payload["replicate"]) is not int or payload["replicate"] != replicate
            or not isinstance(output_root, str) or not output_root or payload["output_root"] != output_root
            or payload["accept_missing_seed_history"] is not True
            or payload["accept_missing_c2_replay"] is not True
            or payload["accept_internal_review_pending"] is not True):
        raise ValueError("TRAIN-5 authorization scope differs")
    _hex(payload["nonce"], 16)
    now = datetime.now(timezone.utc) if now is None else now
    if not isinstance(now, datetime) or now.utcoffset() is None or now.utcoffset().total_seconds() != 0:
        raise ValueError("UTC verification clock required")
    issued, expires = _utc(payload["issued_at_utc"]), _utc(payload["expires_at_utc"])
    if not issued <= now < expires:
        raise ValueError("TRAIN-5 authorization outside validity window")
    if not isinstance(public_key, bytes) or len(public_key) != 32:
        raise ValueError("Ed25519 public key required")
    try:
        Ed25519PublicKey.from_public_bytes(public_key).verify(
            _hex(envelope["signature_hex"], 64), authorization_message(payload))
    except InvalidSignature as exc:
        raise ValueError("invalid TRAIN-5 owner signature") from exc
    return {"signature_verified": True, "identity_authenticated": False, "training_permitted": False}
