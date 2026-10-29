"""Non-executing compatibility checks for archived TRAIN-only audit entrypoints."""
from pathlib import Path
import subprocess
import sys
import pytest

ROOT=Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("script,action", [
    ("wp7_audit_3d_swept_volume.py","audit-swept-volume"),
    ("wp7_audit_right_turn_corridor.py","audit-road-corridor"),
    ("wp7_audit_train_trajectories.py","audit-trajectories"),
])
def test_legacy_and_versionless_help(script,action):
    old=subprocess.run([sys.executable,str(ROOT/"scripts"/script),"--help"],
                       cwd=ROOT,capture_output=True,text=True,timeout=30)
    new=subprocess.run([sys.executable,"-m","scripts.workflows","evidence",action,"--help"],
                       cwd=ROOT,capture_output=True,text=True,timeout=30)
    assert old.returncode==new.returncode==0,(old.stderr,new.stderr)
    assert old.stdout==new.stdout
    assert "--run" in new.stdout and "--output" in new.stdout


@pytest.mark.parametrize("module",[
    "src.validation.swept_volume_audit_cli",
    "src.validation.road_corridor_audit_cli",
    "src.validation.trajectory_audit_cli",
])
def test_audit_rejects_missing_run_before_writing(module,tmp_path):
    import importlib
    runner=importlib.import_module(module)
    output=tmp_path/"audit.json"
    with pytest.raises((FileNotFoundError, SystemExit)):
        runner.main(ROOT,["--run",str(tmp_path/"absent"),"--output",str(output)])
    assert not output.exists()
