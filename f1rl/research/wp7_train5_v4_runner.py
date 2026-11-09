"""Final-only v4 runner; the historical training runner remains unchanged."""
from io import BytesIO
from contextlib import ExitStack
from copy import deepcopy
from threading import Lock

from stable_baselines3.common.save_util import load_from_zip_file

from ..learning.training import TrainingRunResult, TrainingRunner, TrainingScenarioSet
from ..learning.training import runner as core


def initialize_policy_from_archive(model, archive: bytes) -> None:
    if not isinstance(archive, bytes) or not archive:
        raise ValueError("checkpoint archive bytes are required")
    if model.num_timesteps != 0:
        raise ValueError("policy initialization requires a fresh model")
    data, parameters, _ = load_from_zip_file(BytesIO(archive), device="cpu")
    backend = model._model
    if (not isinstance(data, dict) or not isinstance(parameters, dict)
            or data.get("observation_space") != backend.observation_space
            or data.get("action_space") != backend.action_space
            or "policy" not in parameters):
        raise ValueError("checkpoint spaces or policy parameters do not match")
    backend.policy.load_state_dict(parameters["policy"], strict=True)
    if not model.policy_weights_finite():
        raise ValueError("checkpoint policy contains non-finite weights")


class V4TrainingRunner(TrainingRunner):
    def __init__(self, config, *, final_model_observer, model_initializer=None, initial_checkpoint=None):
        super().__init__(config, final_model_observer=final_model_observer)
        self._model_initializer = model_initializer
        self._initial_checkpoint = initial_checkpoint
        self._registered_run = None
        self._authorized_context = None
        self._authorized_lock = Lock()

    def run_single(self, factory):
        if self._registered_run is not None:
            raise ValueError("registered v4 runs require the five-track scenario set")
        return super().run_single(factory)

    def run_multi(self, scenarios):
        if self._registered_run is not None:
            expected = self._registered_run
            if not isinstance(scenarios, TrainingScenarioSet):
                raise ValueError("scenarios must be TrainingScenarioSet")
            if tuple(s.track_id for s in scenarios.scenarios) != tuple(expected["track_ids"]):
                raise ValueError("registered v4 training track roster differs")
            for scenario in scenarios.scenarios:
                reset = scenario.factory.reset_config
                if (reset is None or reset.seed != expected["perturbation_seed"]
                        or reset.version != "wp7_train_reset_lifecycle_research_v1"
                        or reset.max_attempts != 32 or scenario.factory.reset_track_id != scenario.track_id):
                    raise ValueError("registered v4 reset configuration differs")
            from .wp7_train5_v4_admission import require_v4_training_admission
            require_v4_training_admission(expected["root"])
        return super().run_multi(scenarios)

    def run_authorized_multi(self, scenarios, *, envelope, evidence_root):
        """Run only a source-bound, owner-signed internal TRAIN-5 request."""
        if not self._authorized_lock.acquire(blocking=False):
            raise ValueError("TRAIN-5 runner already has an active execution")
        try:
            return self._run_authorized_multi(scenarios, envelope=envelope, evidence_root=evidence_root)
        finally:
            self._authorized_lock.release()

    def _run_authorized_multi(self, scenarios, *, envelope, evidence_root):
        from .wp7_train5_internal_execution import claim, verify_run
        from .wp7_train5_v4_checkpoint import prepare_v4_registered_runner
        from scripts.wp7_train5_v4_factory_preview import validate_registered_factories

        if self._registered_run is None or self._authorized_context is not None:
            raise ValueError("registered TRAIN-5 runner required")
        envelope = deepcopy(envelope)
        run = self._registered_run
        root, cohort, replicate = run["root"], run["cohort"], run["replicate"]
        output = self._config.artifacts_root.resolve()
        verify_run(root, output, cohort, replicate, envelope)
        reference = prepare_v4_registered_runner(root, output, cohort=cohort, replicate=replicate)
        if (self._config != reference._config or self._initial_checkpoint != reference._initial_checkpoint
                or self._registered_run != reference._registered_run):
            raise ValueError("registered TRAIN-5 runner configuration differs")
        validate_registered_factories(root, evidence_root, scenarios, cohort=cohort, replicate=replicate)
        claim_token = claim(output, envelope["payload"]["nonce"])
        self._model_initializer = reference._model_initializer
        self._final_model_observer = reference._final_model_observer
        self._authorized_context = dict(root=root, output=output, cohort=cohort, replicate=replicate,
                                        envelope=envelope, scenarios=scenarios, evidence_root=evidence_root,
                                        claim_token=claim_token)
        try:
            return self._run(mode="multitrack", training_track_ids=tuple(run["track_ids"]),
                             build_env=lambda: core.make_multi_track_vector_env(
                                 scenarios, self._config.vector_env, self._config.sampling_seed))
        finally:
            self._authorized_context = None

    def _run(self, *, mode, training_track_ids, build_env):
        artifacts = core._prepare_run(
            self._config, started_at=core._now(), mode=mode,
            training_track_ids=training_track_ids, provenance_file=None,
        )
        env = model = final_path = None
        stack = ExitStack()
        primary_error = None
        try:
            if self._initial_checkpoint is not None:
                core._update_state(artifacts, initial_checkpoint=self._initial_checkpoint)
            if self._registered_run is not None:
                core._update_state(artifacts, training_reset_seed=self._registered_run["perturbation_seed"])
            if self._authorized_context is not None:
                core._update_state(artifacts, execution_scope="INTERNAL_SIMULATION_ONLY",
                                   internal_authorization=self._authorized_context["envelope"])
            env = build_env()
            if self._authorized_context is not None:
                from .wp7_train5_internal_execution import activate
                stack.enter_context(activate(env, **self._authorized_context))
            model = core.PPOModel.create(env, self._config.ppo)
            if self._model_initializer is not None:
                self._model_initializer(model)
            if self._authorized_context is not None:
                next_report = 8192

                def report(steps):
                    nonlocal next_report
                    if steps >= next_report:
                        print(f"TRAIN-5 {self._registered_run['cohort']} replicate {self._registered_run['replicate']}: {steps}/200704 steps", flush=True)
                        next_report = ((steps // 8192) + 1) * 8192

                model.learn(self._config.total_timesteps, progress_hook=report)
            else:
                model.learn(self._config.total_timesteps)
            actual = model.num_timesteps
            if self._authorized_context is not None:
                from .wp7_train5_internal_execution import require_completed_steps
                require_completed_steps(env, actual)
            final_path = model.save(artifacts.final_model_path)
            self._final_model_observer(final_path, actual)
            core._update_state(artifacts, status="completed", actual_total_timesteps=actual,
                               final_model=core._relative_artifact_path(artifacts, final_path))
            return TrainingRunResult(
                run_id=artifacts.run_id, run_directory=artifacts.run_directory,
                final_model_path=final_path, checkpoint_paths=(),
                requested_total_timesteps=self._config.total_timesteps,
                actual_total_timesteps=actual,
            )
        except Exception as error:
            primary_error = error
            cleanup_error = None
            if final_path is not None:
                try:
                    final_path.unlink()
                except Exception as exc:
                    cleanup_error = f"{type(exc).__name__}: {exc}"
            try:
                actual = model.num_timesteps if model is not None else 0
            except Exception:
                actual = 0
            try:
                core._update_state(artifacts, status="failed", actual_total_timesteps=actual,
                                   final_model=None, final_model_cleanup_error=cleanup_error)
            except Exception:
                pass
            raise
        finally:
            stack.close()
            if env is not None:
                try:
                    env.close()
                except Exception:
                    if primary_error is None:
                        raise
