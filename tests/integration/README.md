# Integration tests

Cross-boundary tests proving components work together, for example:

- FastAPI/WebSocket to runtime session.
- Runtime to sandbox stepping.
- Exported policy through `policy_driver` into vehicle control.
- Track data loaded from `assets/tracks` into a runnable session.

Keep full training runs out of the normal integration suite unless explicitly marked slow.
