"""Read-only analysis of penalty-v3-20261111 paired evaluation.

Reads the original v3 output; writes only to a NEW folder (must not exist).
"""
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, stdev

RUN = Path('D:/REL301m/F1RL-runs/penalty-v3-20261111')
OUT = Path('D:/REL301m/F1RL-runs/penalty-v3-analysis-20261112')
REPO = Path('D:/REL301m/F1-RL-par')
FIN, CR, HRV = 'finished', 'certified_physical_crash', 'hard_route_violation'
BASE, PEN = 'baseline', 'terminal_crash_penalty'
TRACKS = ['straight-sprint-v1', 'slalom-v1', 'right-turn-train-v2', 'ramp-v1', 'left-turn-train-v2']
CONDS = ['nominal', 'perturbation']
IVS = [BASE, PEN]

OUT.mkdir(parents=True, exist_ok=False)
summary = {}


def rate(n, d):
    return n / d if d else None


# ---------- STEP 0.1: pairing ----------
rows = [json.loads(l) for l in (RUN / 'paired-evaluation-journal.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]
summary['journal_rows'] = len(rows)
summary['events'] = dict(Counter(r['event'] for r in rows))
summary['outcomes'] = dict(Counter(r.get('outcome') for r in rows))
summary['valid_physics_not_true'] = sum(1 for r in rows if r.get('valid_physics') is not True)
summary['row_keys'] = sorted(rows[0].keys())

groups = defaultdict(dict)
dup = 0
for r in rows:
    k = (r['cohort'], r['replicate'], r['track_id'], r['condition'], r['episode_index'])
    if r['intervention'] in groups[k]:
        dup += 1
    groups[k][r['intervention']] = r
REPS = sorted({k[1] for k in groups})
summary['seed_keys'] = len(groups)
summary['duplicate_intervention_rows'] = dup
summary['keys_with_exactly_baseline_and_penalty'] = sum(1 for g in groups.values() if set(g) == {BASE, PEN})
summary['pairs_same_env_seed'] = sum(1 for g in groups.values() if len({x['environment_seed'] for x in g.values()}) == 1)
summary['pairs_same_policy_seed'] = sum(1 for g in groups.values() if len({x['policy_seed'] for x in g.values()}) == 1)
env_c = Counter(next(iter(g.values()))['environment_seed'] for g in groups.values())
summary['environment_seeds_reused_across_keys'] = sum(c - 1 for c in env_c.values() if c > 1)
summary['cohort_replicates'] = sorted({f"{r['cohort']}-{r['replicate']}" for r in rows})
summary['models_in_journal'] = len({(r['intervention'], r['cohort'], r['replicate']) for r in rows})


def records(iv, cohort=None, rep=None, track=None, cond=None):
    out = []
    for k, g in groups.items():
        c, rp, t, cd, _ = k
        if cohort is not None and c != cohort:
            continue
        if rep is not None and rp != rep:
            continue
        if track is not None and t != track:
            continue
        if cond is not None and cd != cond:
            continue
        out.append(g[iv])
    return out


def agg(recs):
    n = len(recs)
    c = Counter(r['outcome'] for r in recs)
    f, cr, h = c.get(FIN, 0), c.get(CR, 0), c.get(HRV, 0)
    return {'n': n, 'finished': f, 'certified_crash': cr, 'hard_route': h, 'other': n - f - cr - h,
            'finish_rate': rate(f, n), 'crash_rate': rate(cr, n)}


COHORTS = sorted({k[0] for k in groups})

# ---------- cells (per intervention x cohort x replicate x track x condition) ----------
cells = []
for iv in IVS:
    for c in COHORTS:
        for rp in REPS:
            for t in TRACKS:
                for cd in CONDS:
                    cells.append({'intervention': iv, 'cohort': c, 'replicate': rp, 'track': t, 'condition': cd,
                                  **agg(records(iv, c, rp, t, cd))})

# cross-check against the run's own result file
res = json.loads((RUN / 'paired-evaluation-result.json').read_text(encoding='utf-8'))
idx = {(c['intervention'], c['cohort'], c['replicate'], c['track'], c['condition']): c for c in cells}
mism = []
for rc in res['cells']:
    key = (rc['intervention'], rc['cohort'], rc['replicate'], rc['track'], rc['condition'])
    o = rc['outcomes']
    m = idx.get(key)
    if m is None or (m['n'], m['finished'], m['certified_crash'], m['hard_route']) != (
            rc['n'], o.get(FIN, 0), o.get(CR, 0), o.get(HRV, 0)):
        mism.append(key)
summary['result_file_cells'] = len(res['cells'])
summary['result_file_cell_mismatches'] = len(mism)
summary['result_file_schema'] = res.get('schema')

# ---------- gates (thresholds copied from plan D2 / handoff; see report) ----------
def gates_for(iv):
    tot = agg(records(iv))
    nom = agg(records(iv, cond='nominal'))
    pt = {t: agg(records(iv, track=t)) for t in TRACKS}
    ptc = {f'{t}|{cd}': agg(records(iv, track=t, cond=cd)) for t in TRACKS for cd in CONDS}
    return {
        'overall': tot, 'nominal': nom, 'per_track_pooled_conditions': pt, 'per_track_condition': ptc,
        'G1_track_finish_ge_0.80_pooled_conditions': {t: pt[t]['finish_rate'] >= 0.80 for t in TRACKS},
        'G2_overall_certified_crash_le_0.01': tot['crash_rate'] <= 0.01,
        'G3_nominal_finish_ge_0.90': nom['finish_rate'] >= 0.90,
        'ALT_track_condition_finish_ge_0.80': {k: v['finish_rate'] >= 0.80 for k, v in ptc.items()},
        'ALT_track_crash_le_0.01_pooled_conditions': {t: pt[t]['crash_rate'] <= 0.01 for t in TRACKS},
        'ALT_track_condition_crash_le_0.01': {k: v['crash_rate'] <= 0.01 for k, v in ptc.items()},
    }


summary['gates'] = {iv: gates_for(iv) for iv in IVS}


# ---------- paired comparison (same seed row, both interventions) ----------
def paired(track=None, cond=None):
    c = Counter()
    for k, g in groups.items():
        _, _, t, cd, _ = k
        if track is not None and t != track:
            continue
        if cond is not None and cd != cond:
            continue
        b, p = g[BASE]['outcome'], g[PEN]['outcome']
        c['n'] += 1
        c['base_fin'] += b == FIN
        c['pen_fin'] += p == FIN
        c['base_crash'] += b == CR
        c['pen_crash'] += p == CR
        c['fin_both'] += (b == FIN and p == FIN)
        c['fin_base_only'] += (b == FIN and p != FIN)
        c['fin_pen_only'] += (b != FIN and p == FIN)
        c['crash_both'] += (b == CR and p == CR)
        c['crash_base_only'] += (b == CR and p != CR)
        c['crash_pen_only'] += (b != CR and p == CR)
    n = c['n']
    return {**dict(c), 'finish_base': c['base_fin'] / n, 'finish_pen': c['pen_fin'] / n,
            'finish_diff_pen_minus_base': (c['pen_fin'] - c['base_fin']) / n,
            'crash_base': c['base_crash'] / n, 'crash_pen': c['pen_crash'] / n,
            'crash_diff_pen_minus_base': (c['pen_crash'] - c['base_crash']) / n}


paired_scopes = {'overall': paired(), 'nominal': paired(cond='nominal'), 'perturbation': paired(cond='perturbation')}
for t in TRACKS:
    paired_scopes[f'track:{t}'] = paired(track=t)
    for cd in CONDS:
        paired_scopes[f'track:{t}|{cd}'] = paired(track=t, cond=cd)
summary['paired_scopes'] = paired_scopes

# ---------- per model (12 trained models) and model-unit paired differences ----------
models = []
for c in COHORTS:
    for rp in REPS:
        for iv in IVS:
            row = {'intervention': iv, 'cohort': c, 'replicate': rp}
            a = agg(records(iv, c, rp))
            row.update({f'overall_{k}': v for k, v in a.items()})
            for t in TRACKS:
                ta = agg(records(iv, c, rp, t))
                row[f'{t}_finish'] = ta['finish_rate']
                row[f'{t}_crash'] = ta['crash_rate']
            row['nominal_finish'] = agg(records(iv, c, rp, cond='nominal'))['finish_rate']
            models.append(row)
summary['models'] = models


def get_model(iv, c, rp):
    return next(m for m in models if m['intervention'] == iv and m['cohort'] == c and m['replicate'] == rp)


diff_keys = ['overall_finish_rate', 'overall_crash_rate', 'slalom-v1_finish', 'slalom-v1_crash', 'nominal_finish']
pair_rows = []
for c in COHORTS:
    for rp in REPS:
        b, p = get_model(BASE, c, rp), get_model(PEN, c, rp)
        r = {'cohort': c, 'replicate': rp}
        for k in diff_keys:
            r[f'{k}_base'] = b[k]
            r[f'{k}_pen'] = p[k]
            r[f'{k}_diff'] = p[k] - b[k]
        pair_rows.append(r)
unit = {}
for k in diff_keys:
    d = [r[f'{k}_diff'] for r in pair_rows]
    unit[k] = {'mean_diff': mean(d), 'sd_diff': stdev(d), 'min_diff': min(d), 'max_diff': max(d), 'n_pairs': len(d)}
summary['model_unit_paired_diffs'] = unit

# per-model spread for each intervention
spread = {}
for iv in IVS:
    for k in ['overall_finish_rate', 'overall_crash_rate', 'slalom-v1_finish', 'slalom-v1_crash']:
        v = [m[k] for m in models if m['intervention'] == iv]
        spread[f'{iv}|{k}'] = {'min': min(v), 'max': max(v), 'mean': mean(v), 'sd': stdev(v)}
summary['model_spread_12'] = spread

# ---------- source & artifact checks ----------
prop = json.loads((RUN.parent / 'penalty-v3-20261111-plan.json').read_text(encoding='utf-8'))['proposal']
bad = []
for rel, h in prop['source_bindings'].items():
    p = REPO / rel
    if not p.is_file():
        bad.append((rel, 'missing'))
        continue
    b = p.read_bytes()
    if not rel.endswith('.zip'):
        b = b.replace(b'\r\n', b'\n')
    if hashlib.sha256(b).hexdigest() != h:
        bad.append((rel, 'hash differs'))
summary['source_bindings_checked'] = len(prop['source_bindings'])
summary['source_binding_problems'] = bad[:20]
summary['source_binding_problem_count'] = len(bad)
summary['plan_source_commit'] = prop.get('source_commit')
summary['plan_seed_domain'] = prop.get('seed_domain')
summary['plan_retired_seed_domains'] = prop.get('retired_seed_domains')

tr = json.loads((RUN / 'training-result.json').read_text(encoding='utf-8'))
summary['training_result_keys'] = sorted(tr.keys())
runs = tr.get('runs') or []
model_check = []
for r in runs:
    p = Path(r['model'])
    ok = p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest() == r['sha256']
    model_check.append({'intervention': r['intervention'], 'cohort': r['cohort'], 'replicate': r['replicate'],
                        'steps': r.get('steps'), 'sha_ok': ok})
summary['model_sha_check'] = model_check
summary['model_sha_ok_count'] = sum(1 for m in model_check if m['sha_ok'])

# ---------- calibration against the v4 evidence journal (definition check) ----------
cal = {}
cp = Path('D:/REL301m/TRAIN5-v4-internal-20261109/evaluation-journal.jsonl')
if cp.is_file():
    crow = [json.loads(l) for l in cp.read_text(encoding='utf-8').splitlines() if l.strip()]
    cal['rows'] = len(crow)
    cal['keys'] = sorted(crow[0].keys()) if crow else []
    cal['events'] = dict(Counter(r.get('event') for r in crow))
summary['calibration_v4_journal'] = cal

# ---------- write outputs ----------
(OUT / 'analysis-summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding='utf-8')


def write_csv(name, data):
    with open(OUT / name, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(data[0].keys()))
        w.writeheader()
        w.writerows(data)


write_csv('cells.csv', cells)
write_csv('models.csv', models)
write_csv('model_pair_diffs.csv', pair_rows)
print(json.dumps({k: summary[k] for k in summary if k not in ('gates', 'models', 'paired_scopes', 'model_check', 'models')},
                 indent=1, ensure_ascii=False, default=str)[:6000])
