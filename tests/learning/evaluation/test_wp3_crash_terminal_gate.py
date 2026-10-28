"""Historical evaluator must never score arbitrary unregistered crash claims."""
import pytest
from src.learning.evaluation.evaluator import _step_evidence


@pytest.mark.parametrize("claimed",[
    None,True,{"status":"certified","impulse_n_s":2000},
    {"physics_version":"spatial_road_edge_partial_lifecycle_v4",
     "owner_signed":True,"energy_loss_j":99999.},
])
def test_evaluator_does_not_infer_crash_terminal_from_unregistered_rows(claimed):
    with pytest.raises(ValueError,match="unregistered physical crash terminal"):
        _step_evidence({"physical_crash_terminal":claimed},executed_steps=1,
                       fixed_dt=.05)
