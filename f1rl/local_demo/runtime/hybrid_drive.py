from __future__ import annotations

from threading import Lock

import numpy as np

from f1rl.local_demo.runtime.auto_drive import MAX_STEERING_CHANGE_PER_SECOND


class HybridDrive:
    def __init__(self, ppo, auto, ppo_steering_weight: float):
        if not 0.0 <= ppo_steering_weight <= 1.0:
            raise ValueError("hybrid PPO steering weight must be in [0, 1]")
        self.ppo = ppo
        self.auto = auto
        self.ppo_steering_weight = ppo_steering_weight
        self._lock = Lock()
        self._steering = 0.0
        self._tick = None
        self._steps = 0
        self._interventions = 0
        self._takeover_steps = 0
        self._takeover_active = False

    def predict(self, observation, *, deterministic: bool = True):
        ppo = np.asarray(self.ppo.predict(observation, deterministic=deterministic), dtype=np.float32)
        auto = np.asarray(self.auto.predict(observation, deterministic=deterministic), dtype=np.float32)
        for action in (ppo, auto):
            if (action.shape != (3,) or not np.isfinite(action).all() or not -1 <= action[0] <= 1
                    or not 0 <= action[1] <= 1 or not 0 <= action[2] <= 1):
                raise ValueError("hybrid controller received an invalid action")
        snapshot = self.auto.env.runtime_snapshot()
        takeover = self.auto.requires_obstacle_assist(snapshot)
        tick, dt = int(snapshot["tick"]), float(snapshot["dt_s"])
        with self._lock:
            if self._tick is None or tick < self._tick:
                self._steering = 0.0
                self._steps = 0
                self._interventions = 0
                self._takeover_steps = 0
                elapsed_steps = 1
            else:
                elapsed_steps = tick - self._tick
            weight = 0.0 if takeover else self.ppo_steering_weight
            target = weight * float(ppo[0]) + (1.0 - weight) * float(auto[0])
            maximum_change = MAX_STEERING_CHANGE_PER_SECOND * dt * elapsed_steps
            self._steering = float(np.clip(target, self._steering - maximum_change, self._steering + maximum_change))
            if elapsed_steps:
                self._steps += 1
                self._interventions += int(abs(float(ppo[0]) - self._steering) > 0.05)
                self._takeover_steps += int(takeover)
            self._tick = tick
            self._takeover_active = takeover
            steering = self._steering
        brake = float(auto[2])
        throttle = 0.0 if brake > 0 else min(float(ppo[1]), float(auto[1]))
        return np.asarray((steering, throttle, brake), dtype=np.float32)

    def status(self):
        with self._lock:
            return {
                "ppo_steering_weight": self.ppo_steering_weight,
                "steps": self._steps,
                "steering_interventions": self._interventions,
                "obstacle_takeover_steps": self._takeover_steps,
                "obstacle_takeover_active": self._takeover_active,
            }
