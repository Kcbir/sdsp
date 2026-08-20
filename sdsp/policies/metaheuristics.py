"""Metaheuristics over treatment schedules: Simulated Annealing and VNS.

The search space is the set of open-loop schedules a in A^T. In the
deterministic setting the best open-loop schedule attains the closed-loop
optimum, so these are directly comparable to `AlphaDP` and to Gurobi -- which
means every heuristic here can be reported with a *certified* optimality gap
rather than the usual "it converged".

Both operate on structured neighbourhoods that respect what a schedule is:
change one stage's drug, change dose while holding the drug fixed, swap the
order of two stages, reverse or re-assign a block, rotate. Order-preserving
moves matter because the entire question of the paper is whether order matters:
`swap_two` and `reverse_block` explore *permutations* of a multiset of stage
actions, so the improvement they find is attributable to ordering alone. That
decomposition is reported by `order_sensitivity`.

All objective evaluations go through `sdsp.fastsim.FastSim`, which evaluates a
whole neighbourhood in one tensor contraction.
"""
from __future__ import annotations

import math
import time

import numpy as np

from ..fastsim import get_sim


# ----------------------------------------------------------------------
def evaluate_schedule(inst, seq, s0=None, cfg=None, n_mc=1, seed=0):
    """Objective of an open-loop schedule (mean over n_mc episodes if noisy)."""
    if cfg is None or (cfg.kind == "none" and not cfg.delivery and cfg.capacity is None):
        return get_sim(inst).eval_one(seq, s0)
    from ..dynamics import rollout
    vals = [rollout(inst, list(seq), s0=s0, rng=np.random.default_rng(seed + i), cfg=cfg)
            for i in range(n_mc)]
    return float(np.mean(vals))


def _same_drugs_diff_dose(inst, j):
    """Indices of actions using the same drug set as action j but other doses."""
    drugs_j = tuple(sorted(i for i, _ in inst.actions[j]))
    return [k for k, a in enumerate(inst.actions)
            if tuple(sorted(i for i, _ in a)) == drugs_j and k != j]


class MoveSet:
    """Structured neighbourhood moves on a schedule."""

    def __init__(self, inst, rng):
        self.inst, self.rng = inst, rng
        self.A = inst.n_actions
        self._dose_nbrs = {j: _same_drugs_diff_dose(inst, j) for j in range(self.A)}

    def change_one(self, seq):
        s = list(seq)
        s[self.rng.integers(len(s))] = int(self.rng.integers(self.A))
        return s

    def change_dose(self, seq):
        s = list(seq)
        t = int(self.rng.integers(len(s)))
        nb = self._dose_nbrs[s[t]]
        if nb:
            s[t] = int(self.rng.choice(nb))
        return s

    def swap_two(self, seq):
        s = list(seq)
        if len(s) < 2:
            return s
        i, j = self.rng.choice(len(s), size=2, replace=False)
        s[i], s[j] = s[j], s[i]
        return s

    def reverse_block(self, seq):
        s = list(seq)
        if len(s) < 2:
            return s
        i, j = sorted(self.rng.choice(len(s), size=2, replace=False))
        s[i:j + 1] = s[i:j + 1][::-1]
        return s

    def block_assign(self, seq):
        s = list(seq)
        i, j = sorted(self.rng.choice(len(s), size=2, replace=True))
        a = int(self.rng.integers(self.A))
        for t in range(i, j + 1):
            s[t] = a
        return s

    def rotate(self, seq):
        s = list(seq)
        k = int(self.rng.integers(1, max(2, len(s))))
        return s[k:] + s[:k]

    def perturb_k(self, seq, k):
        s = list(seq)
        for t in self.rng.choice(len(s), size=min(k, len(s)), replace=False):
            s[t] = int(self.rng.integers(self.A))
        return s

    def all_moves(self):
        return [self.change_one, self.change_dose, self.swap_two,
                self.reverse_block, self.block_assign, self.rotate]


# ----------------------------------------------------------------------
def simulated_annealing(inst, iters=20000, T0=None, Tend=None, seed=0, s0=None,
                        cfg=None, n_mc=1, init=None, restarts=1, verbose=False,
                        time_limit=None, record_every=1):
    """Geometric-cooling SA over schedules.

    T0 defaults to the standard deviation of a sample of random schedules, so
    the initial acceptance rate is ~0.5 regardless of the instance's scale. That
    auto-calibration is not cosmetic: objective magnitudes here span many orders
    of magnitude between the subcritical and supercritical regimes, and a fixed
    T0 would make SA behave like pure descent in one regime and pure random
    walk in the other.
    """
    rng = np.random.default_rng(seed)
    sim = get_sim(inst)
    mv = MoveSet(inst, rng)
    moves = mv.all_moves()
    noisy = not (cfg is None or (cfg.kind == "none" and not cfg.delivery
                                 and cfg.capacity is None))
    ev = ((lambda s: evaluate_schedule(inst, s, s0, cfg, n_mc, seed)) if noisy
          else (lambda s: sim.eval_one(s, s0)))
    t_start = time.time()

    if T0 is None:
        samp = sim.eval_batch(rng.integers(inst.n_actions, size=(32, inst.T)), s0)
        T0 = max(float(np.std(samp)), 1e-12)
    if Tend is None:
        Tend = T0 * 1e-4

    best_all, best_seq_all, hist_all = np.inf, None, []
    for r in range(restarts):
        cur = (list(init) if (init is not None and r == 0)
               else list(rng.integers(inst.n_actions, size=inst.T)))
        fcur = ev(cur)
        best, best_seq, hist = fcur, list(cur), [fcur]
        n = max(1, iters // restarts)
        for k in range(n):
            temp = T0 * (Tend / T0) ** (k / max(n - 1, 1))
            cand = moves[rng.integers(len(moves))](cur)
            fc = ev(cand)
            if fc <= fcur or rng.random() < math.exp(
                    -min((fc - fcur) / max(temp, 1e-300), 700.0)):
                cur, fcur = cand, fc
                if fc < best:
                    best, best_seq = fc, list(cand)
            if k % record_every == 0:
                hist.append(best)
            if time_limit and (time.time() - t_start) > time_limit:
                break
        if best < best_all:
            best_all, best_seq_all = best, best_seq
        hist_all.append(hist)
        if verbose:
            print(f"  SA restart {r}: {best:.6g}")
    return {"J": float(best_all), "schedule": best_seq_all, "history": hist_all,
            "runtime": time.time() - t_start, "n_eval": iters, "method": "SA"}


# ----------------------------------------------------------------------
def variable_neighborhood_search(inst, k_max=5, max_iters=300, seed=0, s0=None,
                                 cfg=None, n_mc=1, init=None, verbose=False,
                                 time_limit=None):
    """Basic VNS (Mladenovic-Hansen): shake in N_k, local search, move or k+1.

    Neighbourhood ladder N_1..N_kmax = "randomly reassign k stages". The local
    search is a vectorised best-improvement descent over single-stage flips and
    then pairwise swaps -- each descent step costs one batched evaluation of the
    whole neighbourhood rather than |A|*T separate rollouts.

    VNS suits this problem because the schedule landscape has strong block
    structure (long runs of one drug punctuated by switches) that single-flip
    descent escapes badly: flipping one stage of an optimal alternating pattern
    is always worse, so descent stalls at exactly the schedules we care about.
    """
    rng = np.random.default_rng(seed)
    sim = get_sim(inst)
    mv = MoveSet(inst, rng)
    noisy = not (cfg is None or (cfg.kind == "none" and not cfg.delivery
                                 and cfg.capacity is None))
    t_start = time.time()
    n_eval = 0

    def ev(s):
        nonlocal n_eval
        n_eval += 1
        return (evaluate_schedule(inst, s, s0, cfg, n_mc, seed) if noisy
                else sim.eval_one(s, s0))

    def local(seq, fseq):
        """Vectorised best-improvement descent."""
        nonlocal n_eval
        cur, fcur = list(seq), fseq
        while True:
            f1, c1 = sim.neighbourhood_single_flip(cur, s0)
            n_eval += inst.T * inst.n_actions
            if noisy and f1 < fcur:
                f1 = ev(c1)
            if f1 < fcur - 1e-15:
                cur, fcur = c1, f1
                continue
            f2, c2 = sim.neighbourhood_swap(cur, s0)
            n_eval += inst.T * (inst.T - 1) // 2
            if noisy and f2 < fcur:
                f2 = ev(c2)
            if f2 < fcur - 1e-15:
                cur, fcur = c2, f2
                continue
            return cur, fcur

    cur = list(init) if init is not None else list(rng.integers(inst.n_actions, size=inst.T))
    fcur = ev(cur)
    cur, fcur = local(cur, fcur)
    best, best_seq, hist = fcur, list(cur), [fcur]

    it = 0
    while it < max_iters:
        k = 1
        while k <= k_max and it < max_iters:
            cand = mv.perturb_k(best_seq, k)
            fc = ev(cand)
            cand, fc = local(cand, fc)
            if fc < best - 1e-15:
                best, best_seq, k = fc, list(cand), 1
            else:
                k += 1
            it += 1
            hist.append(best)
            if time_limit and time.time() - t_start > time_limit:
                break
        if time_limit and time.time() - t_start > time_limit:
            break
        if verbose and it % 50 == 0:
            print(f"  VNS it={it} best={best:.6g}")
    return {"J": float(best), "schedule": best_seq, "history": hist,
            "runtime": time.time() - t_start, "n_eval": n_eval, "method": "VNS"}


# ----------------------------------------------------------------------
def gvns(inst, warm="perron", **kw):
    """General VNS, warm-started from a theory-motivated heuristic.

    warm in {"perron", "myopic", "spectral", None}. Seeding a metaheuristic with
    the Perron-greedy rollout is the cheapest large win available, and it is
    exactly the kind of hybridisation the spectral theory is supposed to buy:
    the heuristic supplies the switching *pattern*, VNS repairs the details.
    """
    from ..dynamics import rollout
    from .baselines import (myopic_policy, perron_greedy_policy,
                            spectral_greedy_policy)
    if warm is not None and kw.get("init") is None:
        pol = {"perron": perron_greedy_policy, "myopic": myopic_policy,
               "spectral": spectral_greedy_policy}[warm](inst)
        _, _, acts = rollout(inst, pol, return_traj=True)
        kw["init"] = acts
    out = variable_neighborhood_search(inst, **kw)
    out["method"] = f"GVNS({warm})"
    return out


# ----------------------------------------------------------------------
def order_sensitivity(inst, schedule, n_perm=2000, seed=0, s0=None):
    """How much of a schedule's value comes from *order* rather than composition?

    Holds the multiset of stage-actions fixed and samples random permutations of
    it. Returns the schedule's percentile among its own permutations, plus the
    best and worst achievable by reordering. This is the paper's headline
    question -- "does the order in which drugs are administered matter, and by
    how much?" -- reduced to a directly measurable quantity, and it is
    completely absent from the current draft.
    """
    rng = np.random.default_rng(seed)
    sim = get_sim(inst)
    base = np.asarray(schedule, dtype=np.int64)
    perms = np.array([rng.permutation(base) for _ in range(n_perm)])
    vals = sim.eval_batch(perms, s0)
    j0 = sim.eval_one(base, s0)
    return {
        "J": j0,
        "best_permutation": float(vals.min()),
        "worst_permutation": float(vals.max()),
        "mean_permutation": float(vals.mean()),
        "percentile": float((vals < j0).mean()),
        "order_range_ratio": float(vals.max() / max(vals.min(), 1e-300)),
        "best_perm_schedule": perms[int(np.argmin(vals))].tolist(),
    }


METAHEURISTIC_REGISTRY = {
    "SA": simulated_annealing,
    "VNS": variable_neighborhood_search,
    "GVNS": gvns,
}
