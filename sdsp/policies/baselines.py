"""Baseline and heuristic policies.

Includes the paper's `spectral_greedy` -- and demonstrates the defect found in
review: on Construction 3.2 it reduces to the myopic policy, because it never
looks at s_t at all. `perron_greedy` is the proposed replacement: greedy on the
*left* Perron vector (Fisher's reproductive value), which is the correct
potential function for a linear multitype system and does use the state.
"""
from __future__ import annotations

import numpy as np

from ..dynamics import rollout
from ..spectral import perron_vectors, rho


def _cost_weights(inst):
    return inst.c + inst.lam_res * inst.resistant_nodes.astype(float)


# ----------------------------------------------------------------------
def myopic_policy(inst):
    """argmin_a  c'(W D_a s) + lam_tox tox(a).  The clinical default."""
    cw = _cost_weights(inst)
    WT_c = inst.W.T @ cw                      # (N,)

    def pi(s, t):
        # c'(W D_a s) = ((1-kappa_a) o (W' c))' s
        vals = ((1.0 - inst.kill) * WT_c[None, :]) @ s + inst.lam_tox * inst.tox
        return int(np.argmin(vals))
    return pi


def spectral_greedy_policy(inst):
    """argmin_a rho(W diag(1 - kappa^eff(a))) -- the paper's Algorithm.

    Note it is independent of s and of t, i.e. it is an *open-loop constant*
    policy despite being described as closed-loop.
    """
    Ws = inst.all_W_eff()
    vals = np.array([rho(Ws[j]) for j in range(len(Ws))]) + 1e-9 * inst.tox
    j = int(np.argmin(vals))
    return lambda s, t: j


def perron_greedy_policy(inst, weight="W", horizon_blend=1.0):
    """argmin_a  v'(W D_a s) + lam_tox tox(a), with v the left Perron vector.

    v_i is the long-run contribution of one unit of burden at node i to all
    future burden, so this is a one-step lookahead on the correct asymptotic
    value function rather than on immediate observable damage. Its optimality
    gap is controlled by the spectral gap |lambda_2|/|lambda_1|.

    weight="Weff" uses the left Perron vector of the best controllable operator
    W_eff(a*), which is sharper when the formulary can reshape the dominant mode.
    horizon_blend in [0,1] interpolates v -> c (0 recovers the myopic policy).
    """
    cw = _cost_weights(inst)
    if weight == "Weff":
        from ..spectral import controllability_radius
        _, j = controllability_radius(inst)
        base = inst.W_eff(j)
    else:
        base = inst.W
    _, v, _ = perron_vectors(base)
    v = np.abs(v)
    denom = float(v @ inst.s0)
    if denom > 1e-12:
        v = v * (float(cw @ inst.s0) / denom)      # put v on the scale of c
    wvec = horizon_blend * v + (1.0 - horizon_blend) * cw
    WT_v = inst.W.T @ wvec

    def pi(s, t):
        vals = ((1.0 - inst.kill) * WT_v[None, :]) @ s + inst.lam_tox * inst.tox
        return int(np.argmin(vals))
    return pi


def rolling_horizon_policy(inst, h=3):
    """Exact h-step lookahead with a Perron terminal value. Interpolates
    between myopic (h=1) and optimal (h=T); useful for the ablation that shows
    how much of the gap is closed by how much lookahead."""
    cw = _cost_weights(inst)
    _, v, _ = perron_vectors(inst.W)
    v = np.abs(v)
    d = float(v @ inst.s0)
    if d > 1e-12:
        v = v * (float(cw @ inst.s0) / d)
    surv = 1.0 - inst.kill

    def value(s, depth):
        if depth == 0:
            return float(v @ s)
        best = np.inf
        for j in range(inst.n_actions):
            sn = inst.W @ (surv[j] * s)
            val = float(cw @ sn) + inst.lam_tox * inst.tox[j] + inst.gamma * value(sn, depth - 1)
            best = min(best, val)
        return best

    def pi(s, t):
        hh = min(h, inst.T - t)
        best, arg = np.inf, 0
        for j in range(inst.n_actions):
            sn = inst.W @ (surv[j] * s)
            val = float(cw @ sn) + inst.lam_tox * inst.tox[j] + inst.gamma * value(sn, hh - 1)
            if val < best:
                best, arg = val, j
        return arg
    return pi


def random_cycling_policy(inst, seed=0, exclude_holiday=True):
    """Uniform random drug choice: the standard empirical stewardship heuristic."""
    rng = np.random.default_rng(seed)
    cands = [j for j in range(inst.n_actions)
             if (not exclude_holiday) or len(inst.actions[j]) > 0]
    return lambda s, t: int(rng.choice(cands))


def round_robin_policy(inst, exclude_holiday=True):
    """Deterministic drug cycling."""
    cands = [j for j in range(inst.n_actions)
             if (not exclude_holiday) or len(inst.actions[j]) > 0]
    return lambda s, t: cands[t % len(cands)]


def fixed_policy(inst, j):
    return lambda s, t: j


def mtd_policy(inst):
    """Maximum tolerated dose: the most aggressive feasible action."""
    j = int(np.argmax(inst.tox))
    return lambda s, t: j


def best_fixed_policy(inst, **rollout_kw):
    """The best constant action, found by rollout. This is 'fixed-schedule
    therapy' and is the right thing to beat, not a random baseline."""
    vals = [rollout(inst, [j] * inst.T, **rollout_kw) for j in range(inst.n_actions)]
    j = int(np.argmin(vals))
    return fixed_policy(inst, j), j, float(vals[j])


def adaptive_therapy_policy(inst, on_frac=0.5, off_frac=0.25):
    """Gatenby-style containment: treat at MTD while burden is above `on_frac`
    of its initial value, withdraw below `off_frac`. Requires drug holidays to
    be in the action set (and requires density dependence in the dynamics to
    make sense -- see NoiseConfig.capacity)."""
    j_on = int(np.argmax(inst.tox))
    j_off = 0 if len(inst.actions[0]) == 0 else int(np.argmin(inst.tox))
    b0 = float(inst.c @ inst.s0)
    state = {"on": True}

    def pi(s, t):
        b = float(inst.c @ s)
        if state["on"] and b < off_frac * b0:
            state["on"] = False
        elif (not state["on"]) and b > on_frac * b0:
            state["on"] = True
        return j_on if state["on"] else j_off
    return pi


POLICY_REGISTRY = {
    "myopic":          lambda inst, **k: myopic_policy(inst),
    "spectral_greedy": lambda inst, **k: spectral_greedy_policy(inst),
    "perron_greedy":   lambda inst, **k: perron_greedy_policy(inst),
    "perron_greedy_eff": lambda inst, **k: perron_greedy_policy(inst, weight="Weff"),
    "lookahead3":      lambda inst, **k: rolling_horizon_policy(inst, h=3),
    "random_cycle":    lambda inst, seed=0, **k: random_cycling_policy(inst, seed),
    "round_robin":     lambda inst, **k: round_robin_policy(inst),
    "mtd":             lambda inst, **k: mtd_policy(inst),
    "best_fixed":      lambda inst, **k: best_fixed_policy(inst)[0],
    "adaptive_therapy": lambda inst, **k: adaptive_therapy_policy(inst),
}
