from pathlib import Path


def test_demo_factory_never_opens_heldout_assets(monkeypatch):
    from f1rl.research.wp7_protocol import load_registration, prepare_scenarios
    original = Path.open

    def guarded(path, *args, **kwargs):
        assert not ('tracks' in path.parts and 'evaluation' in path.parts), path
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'open', guarded)
    root = Path(__file__).resolve().parents[2]
    registration = load_registration(root, profile='spatial_rotating_owner_v3')
    scenarios = prepare_scenarios(root, registration, seed_index=0)
    assert scenarios.scenarios[0].track_id == 'straight-sprint-v1'
