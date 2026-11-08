import pytest
from pydantic import ValidationError

from f1rl.local_demo.runtime.protocol import WS_COMMAND_ADAPTER


def test_versioned_ws_protocol_accepts_controls_and_validated_actions():
    common = {"protocol_version": "f1rl_runtime_ws_v1"}
    assert WS_COMMAND_ADAPTER.validate_python({**common, "type": "start"}).type == "start"
    speed = WS_COMMAND_ADAPTER.validate_python({**common, "type": "speed", "multiplier": 2.5})
    assert speed.multiplier == 2.5
    action = WS_COMMAND_ADAPTER.validate_python({
        **common, "type": "action", "steering": -1, "throttle": 0.5, "brake": 0,
    })
    assert action.steering == -1


@pytest.mark.parametrize("payload", [
    {"protocol_version": "v0", "type": "start"},
    {"protocol_version": "f1rl_runtime_ws_v1", "type": "speed", "multiplier": 20},
    {"protocol_version": "f1rl_runtime_ws_v1", "type": "action", "steering": 2, "throttle": 0, "brake": 0},
    {"protocol_version": "f1rl_runtime_ws_v1", "type": "pause", "extra": True},
])
def test_ws_protocol_rejects_wrong_version_unbounded_actions_and_extra_fields(payload):
    with pytest.raises(ValidationError):
        WS_COMMAND_ADAPTER.validate_python(payload)
