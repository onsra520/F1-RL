"""Scoped local TRAIN-5 capability; application guard, not a Python sandbox."""
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
import json
import hashlib
import random
from pathlib import Path
from weakref import WeakSet

from stable_baselines3.common.vec_env import DummyVecEnv

from .wp7_protocol import _unique
from .wp7_trusted_reviews import _hex, _utc

_active = ContextVar("train5_internal_capability", default=None)
_claims = WeakSet()
OWNER_KEY = "configs/research/wp7-train5-internal-owner-key-v1.json"


def _owner_public_key(root, expected_sha256):
    root = Path(root).resolve(strict=True)
    path = root / OWNER_KEY
    if path.is_symlink() or not path.resolve(strict=True).is_relative_to(root):
        raise ValueError("owner key registry must remain inside the checkout")
    data = path.read_bytes().replace(b"\r\n", b"\n")
    if hashlib.sha256(data).hexdigest() != expected_sha256:
        raise ValueError("TRAIN-5 owner key registry changed")
    registry = json.loads(data, object_pairs_hook=_unique)
    if set(registry) != {"schema", "owner_public_key_hex"} or registry["schema"] != "train5_internal_owner_key_v1":
        raise ValueError("invalid TRAIN-5 owner key registry")
    return _hex(registry["owner_public_key_hex"], 32)


def verify_run(root, output, cohort, replicate, envelope):
    from .wp7_train5_v4_admission import load_v4_protocol, validate_v4_protocol
    from scripts.wp7_train5_v4_review_manifest import load_review_manifest
    from scripts.wp7_train5_execution_authorization import verify_authorization

    root, output = Path(root).resolve(strict=True), Path(output).resolve()
    if output == root or root in output.parents:
        raise ValueError("TRAIN-5 output must be outside source checkout")
    protocol, protocol_sha = load_v4_protocol(root)
    validate_v4_protocol(root, protocol)
    manifest, manifest_sha = load_review_manifest(root)
    if not any(row["cohort"] == cohort and row["replicate"] == replicate for row in protocol["runs"]):
        raise ValueError("unregistered TRAIN-5 run")
    owner_key = _owner_public_key(root, manifest["review_source_bindings"][OWNER_KEY])
    verify_authorization(envelope, owner_key, protocol_sha256=protocol_sha,
                         manifest_sha256=manifest_sha, cohort=cohort, replicate=replicate,
                         output_root=str(output))
    return protocol


def admitted(env):
    """Recheck a live, exact eight-selector capability at each PPO boundary."""
    from ..learning.training.vector_env import _SampledF1RLEnv

    context = _active.get()
    if context is None or env is not context["env"] or type(env) is not DummyVecEnv:
        return False
    if (env.num_envs != 8 or len(env.envs) != 8
            or any(type(child.unwrapped) is not _SampledF1RLEnv
                   or child.unwrapped._scenarios is not context["scenarios"]
                   or child.unwrapped._rank != rank
                   or child.unwrapped._sampling_seed != context["sampling_seed"]
                   for rank, child in enumerate(env.envs))):
        return False
    try:
        verify_run(context["root"], context["output"], context["cohort"],
                   context["replicate"], context["envelope"])
    except (OSError, ValueError, TypeError):
        return False
    return True


def check_active_step(child):
    """Stop an active local run when its owner-signed window expires."""
    context = _active.get()
    if context is None:
        return
    if not any(item.unwrapped is child for item in context["env"].envs):
        raise ValueError("environment is outside the active TRAIN-5 run")
    if datetime.now(timezone.utc) >= _utc(context["envelope"]["payload"]["expires_at_utc"]):
        raise ValueError("TRAIN-5 authorization expired during execution")
    if context["steps"] >= context["max_steps"]:
        raise ValueError("TRAIN-5 signed step budget exhausted")
    context["steps"] += 1


def require_completed_steps(env, actual):
    context = _active.get()
    if (context is None or env is not context["env"] or type(actual) is not int
            or actual != context["steps"] or actual != context["max_steps"]):
        raise ValueError("TRAIN-5 recorded model and environment steps differ")


@contextmanager
def activate(env, *, root, output, cohort, replicate, envelope, scenarios, evidence_root, claim_token):
    from scripts.wp7_train5_v4_factory_preview import validate_registered_factories

    protocol = verify_run(root, output, cohort, replicate, envelope)
    validate_registered_factories(root, evidence_root, scenarios, cohort=cohort, replicate=replicate)
    if (not isinstance(claim_token, _Claim) or claim_token not in _claims or claim_token.used
            or claim_token.output != Path(output).resolve()
            or claim_token.nonce != envelope["payload"]["nonce"]):
        raise ValueError("fresh TRAIN-5 execution claim required")
    claim_token.used = True
    _claims.remove(claim_token)
    run = next(row for row in protocol["runs"] if row["cohort"] == cohort and row["replicate"] == replicate)
    context = dict(env=env, root=root, output=output, cohort=cohort, replicate=replicate,
                   envelope=envelope, scenarios=scenarios, sampling_seed=run["seeds"]["sampling"],
                   steps=0, max_steps=protocol["budget"]["actual_final_steps"])
    if _active.get() is not None:
        raise ValueError("nested TRAIN-5 capabilities are not allowed")
    token = _active.set(context)
    try:
        if not admitted(env):
            raise ValueError("TRAIN-5 vector environment is not admitted")
        for rank, child in enumerate(env.envs):
            sampled = child.unwrapped
            seed = int.from_bytes(hashlib.sha256(
                f"f1rl/multi-track-sampling/v1/{context['sampling_seed']}/{rank}".encode("utf8")
            ).digest(), "big")
            if (sampled._reset_sequence != 0 or sampled._active_env is not None
                    or sampled._selection_rng.getstate() != random.Random(seed).getstate()):
                raise ValueError("TRAIN-5 selectors must be fresh and unmodified")
        yield
    finally:
        _active.reset(token)


class _Claim:
    def __init__(self, output, nonce):
        self.output, self.nonce, self.used = output, nonce, False


def claim(output, nonce):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    path = output / f".train5-claim-{nonce}"
    with path.open("x", encoding="ascii") as marker:
        marker.write("TRAIN-5 internal execution claim\n")
    token = _Claim(output, nonce)
    _claims.add(token)
    return token
