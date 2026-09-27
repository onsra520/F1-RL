# `f1rl.api.routes`

## Purpose

Feature-oriented FastAPI HTTP route modules.

Likely route groups include tracks, runtime sessions, policies, and evaluation. Exact modules should be created only when their endpoint contracts are implemented.

Routes should validate/translate requests and delegate work to runtime or learning services rather than implementing domain behavior directly.
