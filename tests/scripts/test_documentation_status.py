from pathlib import Path
import json
import re

ROOT = Path(__file__).resolve().parents[2]

def test_active_doc_links_exist():
    paths = [ROOT / 'README.md', ROOT / 'scripts/README.md',
             ROOT / 'docs/README.md', ROOT / 'docs/architecture/README.md',
             ROOT / 'docs/research/README.md', ROOT / 'docs/superpowers/README.md']
    paths += sorted((ROOT / 'docs/development').glob('*.md'))
    paths += sorted((ROOT / 'docs/operations').glob('*.md'))
    for page in paths:
        assert page.is_file()
        for link in re.findall(r'(?<!!)\[[^\]]+\]\(([^)]+)\)', page.read_text(encoding='utf-8')):
            relative = link.split('#')[0].strip('<>')
            if not relative or relative.startswith(('https://', 'http://', 'mailto:')): continue
            target = (page.parent / relative).resolve()
            assert target.is_relative_to(ROOT.resolve())
            assert target.exists(), (page.relative_to(ROOT), relative)

def test_entrypoint_inventory_covers_all_commands():
    data = json.loads((ROOT / 'docs/development/SCRIPT-STATUS.json').read_text(encoding='utf-8'))
    assert data['counts'] == {
        'catalogued': 48, 'grouped_entrypoints': 48,
        'root_legacy_entrypoints': 0, 'compatibility_wrappers': 0,
    }
    assert len(data['workflows']) == 48
    for command in data['workflows']:
        assert command['scope']
        assert (ROOT / command['entrypoint']).is_file()
        assert not (ROOT / command['former_root_path']).exists()
