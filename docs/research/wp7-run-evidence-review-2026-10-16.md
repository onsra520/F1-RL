# WP7 run evidence and trusted review infrastructure

Parent: Draft PR #25, `13189b76a3068e62f5d3c40b354d7362a8c2b817`.
Engineering implementation only; no research run or scientific admission occurs.

## Run identity and restore

`TrainingRunner.run_registered` validates the existing policy binding and captures
`wp7_run_manifest_v1` before artifact/model allocation and the existing research
guard. `runner.registered_run_manifest` returns a defensive copy, including for
denied attempts. Its state is `prepared_not_started`; it cannot claim that training
occurred. Dirty source is captured honestly and remains an admission blocker.
Runtime capture retains the actual CPU/GPU, OS, Python and package records from
the existing runtime manifest. Policy identity, PPO/sampling/perturbation seeds,
vector/physics/race/reset/action identities and checkpoint plan remain bound by
the existing V2 execution receipt. A source/protocol change during capture fails.

`bind_run_artifacts` hashes an existing exact file set through the existing backup
manifest implementation and links it to the preparation digest. It does not load
models, infer learned timesteps, or reconstruct past runtime provenance. A future
authorized runner still needs observer integration for automatic checkpoint and
completion publication; this version cannot make that claim from opaque bytes.

`verify_run_manifest` checks nested digests, cross-bound protocol/policy identity,
the preparation chain and the exact restored artifact set. It requires an expected
digest retained independently. It does not recapture the verifier host as the
training host. Hash consistency alone establishes neither authentic origin nor
scientific correctness. Existing `wp7_evidence_manifest.py create/verify` remains
the versioned directory backup/restore interface; no cloud upload or independent
Drive recovery is claimed.

Cell JSON publication now hard-links a fully flushed temporary file in the same
directory, atomically creating a destination without overwriting previous bytes.
Unsupported filesystems fail explicitly; no partial-copy fallback is used. This
is publication atomicity, not a guarantee against every filesystem/power failure.

## Trusted review identity

`wp7_trusted_reviewers_v1` is a separate owner-controlled trust registry. The
checked-in registry is deliberately empty: no real owner/reviewer identities
have been independently bootstrapped. Fixture signing keys never enter it.

`verify_run_reviews` verifies Ed25519 signatures using `cryptography` (optional
`science` extra; also installed by `dev` for CI). It requires exactly two distinct
trusted keys and identities with owner and independent-reviewer roles. Strict
envelopes bind domain-separated canonical JSON to the exact protocol and run
manifest, a fixed scientific-review purpose, UTC issue/expiry windows, registry
validity and revocation. Unknown keys, duplicate roles/keys, replay against other
runs, extra booleans, unsigned and altered payloads are rejected. Trust must be
established outside the review envelope; a supplied registry is not self-proving.

`trusted_reviews_verified` authenticates review signatures. It never authorizes
training, HELD-OUT, freeze or WP8. All four permissions remain false, including
when both fixture signatures verify. Empirical calibration, full track feasibility,
independent restoration and scientific decisions remain external/blocked.
The existing admission API retains its closed boundary and legacy blocker ID;
it now reports that signature-verifier implementation exists while trusted
bootstrap and scientific approval are missing.

## Independent verifier CLI

Install `.[science]`, retain the expected manifest digest independently, then run:

```powershell
python scripts/wp7_verify_run_evidence.py --manifest restored-manifest.json --expected-sha256 <digest> --artifacts restored-run
```

Optional `--review review.json --trust-registry independently-trusted-registry.json`
authenticates the two signatures with the current UTC clock. Duplicate JSON keys
and nonfinite values are rejected. No pickle or model archive is deserialized.
Successful output always reports `training_authorized: false`.

## Compatibility limits

V1 runtime captures and V2 policy receipts remain readable through their existing
APIs. The new run manifest does not silently migrate historical exploratory runs.
Legacy runner entrypoints retain their behavior. Optional cryptography imports
are lazy; installing base/rl does not require science dependencies. This host
has CPU-only PyTorch; CUDA hardware execution remains unverified. Windows and
Ubuntu exact-HEAD CI, saved-model inference, both vector backends, historical C2
and public `prepare_scenarios(seed_index=...)` regression are required separately.
