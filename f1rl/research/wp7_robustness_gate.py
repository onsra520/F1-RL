"""WP7 preregistered robustness NUMERIC gate, without running training.

This module checks a supplied, externally evidenced TRAIN evaluation table
and NEVER marks scientific acceptance or unseals HELD-OUT. The one-sided 95%
exact binomial (Clopper-Pearson) lower confidence bound solves

    P_{p=lower}[Binomial(n,p) >= successes] = 0.05

for successes > 0. Evidence provenance, owner decision, split sealing and
external validation remain separate independent gates.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import comb,isfinite


@dataclass(frozen=True,slots=True)
class RobustnessCell:
    policy_id: str
    train_track_id: str
    stratum_id: str
    paired_seed_ids: tuple[int,...]
    successes: int
    episodes: int
    evidence_id: str

    def __post_init__(self):
        if any(not isinstance(t,str) or not t.strip() for t in (
            self.policy_id,self.train_track_id,self.stratum_id,self.evidence_id,
        )):
            raise ValueError("each robustness cell needs identifiers and evidence provenance")
        if (type(self.episodes) is not int or not 1<=self.episodes<=1000
                or type(self.successes) is not int
                or not 0<=self.successes<=self.episodes):
            raise ValueError("invalid robustness binomial observations")
        if (not isinstance(self.paired_seed_ids,tuple)
                or any(type(s) is not int or s<0 for s in self.paired_seed_ids)
                or len(set(self.paired_seed_ids))!=len(self.paired_seed_ids)):
            raise ValueError("paired seeds must be distinct nonnegative identifiers")


def one_sided_cp_lower(successes:int,episodes:int,alpha:float=.05)->float:
    if (type(successes) is not int or type(episodes) is not int
            or not 0<=successes<=episodes<=1000 or episodes<=0
            or type(alpha) not in (int,float) or not isfinite(alpha)
            or not 0.<alpha<1.):
        raise ValueError("invalid exact-binomial confidence-bound inputs")
    if successes==0:
        return 0.
    if successes==episodes:
        return alpha**(1./episodes)
    def tail(p):
        return sum(
            comb(episodes,k)*(p**k)*((1.-p)**(episodes-k))
            for k in range(successes,episodes+1)
        )
    low,high=0.,1.
    for _ in range(70):
        middle=(low+high)/2.
        if tail(middle)<alpha:
            low=middle
        else:
            high=middle
    return (low+high)/2.


@dataclass(frozen=True,slots=True)
class RobustnessNumericalVerdict:
    meets_preregistered_numeric_criterion: bool
    reasons: tuple[str,...]
    lower_bounds: tuple[tuple[str,float], ...]

    @property
    def scientifically_accepted(self)->bool:
        # Calibration, sealed splits, independent reproducibility, evidence
        # restore and owner scientific freeze are all external to this gate.
        return False


def evaluate_train_robustness(
    cells: tuple[RobustnessCell,...],
    *,
    required_strata: tuple[tuple[str,str,str], ...],
    minimum_episodes:int=60,
    minimum_seeds:int=3,
    required_lower_bound:float=.9,
)->RobustnessNumericalVerdict:
    if (not isinstance(cells,tuple)
            or any(not isinstance(c,RobustnessCell) for c in cells)
            or not isinstance(required_strata,tuple)
            or not required_strata
            or any(not isinstance(k,tuple) or len(k)!=3 for k in required_strata)):
        raise ValueError("explicit complete TRAIN strata and evidence rows required")
    if (type(minimum_episodes) is not int or minimum_episodes<60
            or type(minimum_seeds) is not int or minimum_seeds<3
            or type(required_lower_bound) not in (int,float)
            or not .9<=required_lower_bound<1.):
        raise ValueError("owner robustness criteria cannot be weakened")
    keys=[(c.policy_id,c.train_track_id,c.stratum_id) for c in cells]
    if len(set(keys))!=len(keys) or len(set(required_strata))!=len(required_strata):
        raise ValueError("duplicate or conflicting policy/track/stratum cells")
    by_key=dict(zip(keys,cells,strict=True))
    reasons=[]
    bounds=[]
    seed_sets={}
    for key in required_strata:
        cell=by_key.get(key)
        if cell is None:
            reasons.append(f"missing required TRAIN cell {key!r}")
            continue
        if cell.episodes<minimum_episodes:
            reasons.append(f"insufficient episodes in {key!r}")
        if len(cell.paired_seed_ids)<minimum_seeds:
            reasons.append(f"insufficient paired seeds in {key!r}")
        seed_sets.setdefault(cell.policy_id,[]).append(set(cell.paired_seed_ids))
        lower=one_sided_cp_lower(cell.successes,cell.episodes)
        bounds.append(("/".join(key),lower))
        if lower<required_lower_bound:
            reasons.append(f"Clopper-Pearson lower bound below target in {key!r}")
    for policy,sets in seed_sets.items():
        if any(group!=sets[0] for group in sets[1:]):
            reasons.append(f"paired seed sets differ across TRAIN strata for {policy}")
    unexpected=set(keys)-set(required_strata)
    if unexpected:
        reasons.append("unregistered TRAIN strata were supplied")
    return RobustnessNumericalVerdict(not reasons,tuple(reasons),tuple(bounds))
