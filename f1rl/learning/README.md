# F1-RL learning status and limitations

**Vectorized PPO** and local legacy model interfaces are **implemented** and covered by development smoke/regression tests. Multi-track sampling is implemented for explicitly constructed scenario sets, including deterministic rank-specific scenario selection. The vector wrapper has `dummy` and `subproc` backends.

Training orchestration includes configuration, checkpointing, manifests and source checks. Evaluation includes deterministic and seeded stochastic development mechanisms; formal scientific admission remains independent of those engineering capabilities. Currently the 9 WP7 compatibility gates and three scientific capabilities are not certified by the new TRAIN-5 protocol.

Unseen-track generalization experiments and HELD-OUT evaluation remain blocked. The original C2 draft is historical metadata and tests do not establish a reproduced historical trajectory. Successful legacy vectorized PPO tests are not a claim of calibrated real-world physics, safety or verified new-science authorization.
