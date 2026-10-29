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

def test_dispositions_cover_all_commands():
    data = json.loads((ROOT / 'docs/development/SCRIPT-STATUS.json').read_text(encoding='utf-8'))
    expected = {'extracted-implementation': 14, 'existing-delegator': 15,
                'shared-dependency': 2, 'guarded-historical-executable': 17}
    assert data['review_disposition_counts'] == expected
    assert sum(expected.values()) == len(data['workflows']) == 48
    for command in data['workflows']:
        assert command['review_disposition'] in expected
        assert command['rationale'] and (ROOT / command['legacy_path']).is_file()
