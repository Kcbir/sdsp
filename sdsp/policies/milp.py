"""Mixed-integer formulations of the multi-stage control problem (Gurobi).

Two formulations:

`solve_milp_actions`    binaries over the *enumerated* action set. Compact,
                        and the cleanest thing to certify optimality against.

`solve_milp_structured` the paper's formulation: binaries over (drug, dose
                        level) with the toxicity knapsack and combination cap
                        written out, so each stage is a Multiple-Choice
                        Multi-Dimensional Knapsack Problem. Scales when |A| is
                        combinatorially large.

Both linearise the bilinear state-action products exactly by McCormick/big-M
using the *valid* componentwise state bound

    s_t <= W^t s_0      (because 0 <= 1 - kappa^eff <= 1 elementwise),

so the linearisation is exact rather than a relaxation. In the deterministic
setting the optimal open-loop schedule attains the closed-loop optimum, so these
values coincide with `AlphaDP`; that agreement is the cross-check in tests.
"""
from __future__ import annotations

import numpy as np

try:
    import gurobipy as gp
    from gurobipy import GRB
    HAVE_GUROBI = True
except Exception:                                    # pragma: no cover
    HAVE_GUROBI = False


def state_bounds(inst, s0=None, T=None, slack=1.0 + 1e-9, tight=True):
    """Valid componentwise upper bounds U[t] >= s_t for every admissible policy.

    The loose bound U[t] = W^t s0 ignores the fact that *every* action kills
    something, so for a fast-replicating instance it overshoots the reachable
    state by many orders of magnitude. With big-M that is fatal: Gurobi's
    feasibility tolerance times M then exceeds the whole objective, and the
    solver returns a schedule whose reported cost is not the cost it actually
    incurs. The tight bound takes the componentwise max over actions,

        U[t+1][v] = max_a sum_u W[v,u] (1-kappa_a[u]) U[t][u],

    which is still valid (each component is maximised independently) and is
    typically orders of magnitude smaller.
    """
    s0 = inst.s0 if s0 is None else np.asarray(s0, float)
    T = inst.T if T is None else T
    surv = 1.0 - inst.kill
    U = [s0.copy()]
    for _ in range(T):
        if tight:
            cand = np.einsum("vu,au->av", inst.W, surv * U[-1][None, :])
            U.append(cand.max(axis=0) * slack)
        else:
            U.append(inst.W @ U[-1] * slack)
    return np.array(U)


# ----------------------------------------------------------------------
def solve_milp_actions(inst, s0=None, t0=0, T=None, time_limit=60.0, mip_gap=0.0,
                       verbose=False, warm_start=None, use_indicators=True):
    """Exact schedule over the enumerated action set.

    Returns dict with keys: J, schedule, gap, runtime, status, bound.
    """
    if not HAVE_GUROBI:
        raise RuntimeError("gurobipy not available")
    s0 = inst.s0 if s0 is None else np.asarray(s0, float)
    T = (inst.T - t0) if T is None else T
    N, A, g = inst.N, inst.n_actions, inst.gamma
    c_eff = inst.c + inst.lam_res * inst.resistant_nodes.astype(float)
    surv = 1.0 - inst.kill                       # (A,N)
    U = state_bounds(inst, s0, T)

    m = gp.Model("sdsp_actions")
    m.Params.OutputFlag = 1 if verbose else 0
    m.Params.TimeLimit = time_limit
    m.Params.MIPGap = mip_gap
    m.Params.NumericFocus = 3
    m.Params.IntFeasTol = 1e-9
    m.Params.FeasibilityTol = 1e-9

    y = m.addVars(T, A, vtype=GRB.BINARY, name="y")
    s = [m.addVars(N, lb=0.0, ub=[float(U[t][v]) for v in range(N)], name=f"s{t}")
         for t in range(T + 1)]
    # z[t,a,v] = y[t,a] * s[t,v]
    z = m.addVars(T, A, N, lb=0.0, name="z")

    for v in range(N):
        m.addConstr(s[0][v] == float(s0[v]))
    for t in range(T):
        m.addConstr(gp.quicksum(y[t, a] for a in range(A)) == 1)
        for a in range(A):
            for v in range(N):
                Uv = float(U[t][v])
                z[t, a, v].UB = Uv
                if use_indicators:
                    # exact, big-M free: y=1 => z = s ; y=0 => z = 0
                    m.addGenConstrIndicator(y[t, a], True, z[t, a, v] - s[t][v],
                                            GRB.EQUAL, 0.0)
                    m.addGenConstrIndicator(y[t, a], False, z[t, a, v],
                                            GRB.EQUAL, 0.0)
                else:
                    m.addConstr(z[t, a, v] <= Uv * y[t, a])
                    m.addConstr(z[t, a, v] <= s[t][v])
                    m.addConstr(z[t, a, v] >= s[t][v] - Uv * (1 - y[t, a]))
        for v in range(N):
            m.addConstr(gp.quicksum(z[t, a, v] for a in range(A)) == s[t][v])
        # s_{t+1} = sum_a W diag(surv_a) z[t,a]
        for v in range(N):
            m.addConstr(
                s[t + 1][v] == gp.quicksum(
                    float(inst.W[v, u] * surv[a, u]) * z[t, a, u]
                    for a in range(A) for u in range(N)
                    if inst.W[v, u] * surv[a, u] != 0.0))

    obj = gp.quicksum(
        (g ** t) * (gp.quicksum(float(c_eff[v]) * s[t + 1][v] for v in range(N))
                    + inst.lam_tox * gp.quicksum(float(inst.tox[a]) * y[t, a]
                                                 for a in range(A)))
        for t in range(T))
    m.setObjective(obj, GRB.MINIMIZE)

    if warm_start is not None:
        for t, a in enumerate(warm_start[:T]):
            for aa in range(A):
                y[t, aa].Start = 1.0 if aa == a else 0.0

    m.optimize()
    sched = [int(np.argmax([y[t, a].X for a in range(A)])) for t in range(T)]
    return {"J": float(m.ObjVal), "schedule": sched, "gap": float(m.MIPGap),
            "runtime": float(m.Runtime), "status": int(m.Status),
            "bound": float(m.ObjBound), "n_vars": m.NumVars, "n_cons": m.NumConstrs}


# ----------------------------------------------------------------------
def solve_milp_structured(inst, s0=None, T=None, time_limit=120.0, mip_gap=0.0,
                          verbose=False, use_indicators=True):
    """The paper's MCMDKP-per-stage formulation, written out explicitly.

    Decision variables u[t,i,l] = 1 iff drug i is given at dose level l in
    stage t, subject to
        sum_l u[t,i,l] <= 1                         (one dose level per drug)
        sum_{i,l} u[t,i,l] <= m                     (combination cap)
        sum_{i,l} f_i(r_l) u[t,i,l] <= B            (toxicity knapsack)
    with kappa^eff capped at 1 so survival stays in [0,1].
    """
    if not HAVE_GUROBI:
        raise RuntimeError("gurobipy not available")
    s0 = inst.s0 if s0 is None else np.asarray(s0, float)
    T = inst.T if T is None else T
    N, D, g = inst.N, inst.d, inst.gamma
    c_eff = inst.c + inst.lam_res * inst.resistant_nodes.astype(float)
    L = [len(dr.doses) for dr in inst.drugs]
    U = state_bounds(inst, s0, T)
    # phi_i(r_l) * kappa_iv
    KILL = [[inst.drugs[i].kill(inst.drugs[i].doses[l]) for l in range(L[i])]
            for i in range(D)]
    TOX = [[inst.drugs[i].tox(inst.drugs[i].doses[l]) for l in range(L[i])]
           for i in range(D)]

    m = gp.Model("sdsp_structured")
    m.Params.OutputFlag = 1 if verbose else 0
    m.Params.TimeLimit = time_limit
    m.Params.MIPGap = mip_gap
    m.Params.NumericFocus = 3
    m.Params.IntFeasTol = 1e-9
    m.Params.FeasibilityTol = 1e-9

    if inst.kill_model != "additive_capped":
        raise ValueError(
            "solve_milp_structured is exact only for kill_model='additive_capped' "
            f"(got {inst.kill_model!r}). The structured formulation writes "
            "kappa^eff as a linear sum of (drug,dose) indicators, which matches "
            "the additive model with an explicit kappa^eff <= 1 constraint. Use "
            "solve_milp_actions for the Bliss-independence model.")

    u = {(t, i, l): m.addVar(vtype=GRB.BINARY, name=f"u{t}_{i}_{l}")
         for t in range(T) for i in range(D) for l in range(L[i])}
    s = [m.addVars(N, lb=0.0, ub=[float(U[t][v]) for v in range(N)], name=f"s{t}")
         for t in range(T + 1)]
    p = {(t, i, l, v): m.addVar(lb=0.0, name=f"p{t}_{i}_{l}_{v}")
         for t in range(T) for i in range(D) for l in range(L[i]) for v in range(N)}
    shat = [m.addVars(N, lb=0.0, name=f"sh{t}") for t in range(T)]

    for v in range(N):
        m.addConstr(s[0][v] == float(s0[v]))
    for t in range(T):
        for i in range(D):
            m.addConstr(gp.quicksum(u[t, i, l] for l in range(L[i])) <= 1)
        m.addConstr(gp.quicksum(u[t, i, l] for i in range(D) for l in range(L[i]))
                    <= inst.m)
        m.addConstr(gp.quicksum(TOX[i][l] * u[t, i, l]
                                for i in range(D) for l in range(L[i])) <= inst.B)
        for v in range(N):
            # kappa^eff_v <= 1 so that survival stays non-negative
            m.addConstr(gp.quicksum(float(KILL[i][l][v]) * u[t, i, l]
                                    for i in range(D) for l in range(L[i])) <= 1.0)
            Uv = float(U[t][v])
            for i in range(D):
                for l in range(L[i]):
                    p[t, i, l, v].UB = Uv
                    if use_indicators:
                        # exact and big-M free (see solve_milp_actions)
                        m.addGenConstrIndicator(u[t, i, l], True,
                                                p[t, i, l, v] - s[t][v], GRB.EQUAL, 0.0)
                        m.addGenConstrIndicator(u[t, i, l], False,
                                                p[t, i, l, v], GRB.EQUAL, 0.0)
                    else:
                        m.addConstr(p[t, i, l, v] <= Uv * u[t, i, l])
                        m.addConstr(p[t, i, l, v] <= s[t][v])
                        m.addConstr(p[t, i, l, v] >= s[t][v] - Uv * (1 - u[t, i, l]))
            m.addConstr(shat[t][v] == s[t][v] - gp.quicksum(
                float(KILL[i][l][v]) * p[t, i, l, v]
                for i in range(D) for l in range(L[i])))
        for v in range(N):
            m.addConstr(s[t + 1][v] == gp.quicksum(float(inst.W[v, w_]) * shat[t][w_]
                                                   for w_ in range(N)
                                                   if inst.W[v, w_] != 0.0))

    obj = gp.quicksum(
        (g ** t) * (gp.quicksum(float(c_eff[v]) * s[t + 1][v] for v in range(N))
                    + inst.lam_tox * gp.quicksum(TOX[i][l] * u[t, i, l]
                                                 for i in range(D) for l in range(L[i])))
        for t in range(T))
    m.setObjective(obj, GRB.MINIMIZE)
    m.optimize()

    sched = []
    for t in range(T):
        act = tuple(sorted((i, l) for i in range(D) for l in range(L[i])
                           if u[t, i, l].X > 0.5))
        sched.append(act)
    # map back to enumerated-action indices where possible
    idx = []
    for act in sched:
        idx.append(inst.actions.index(act) if act in inst.actions else -1)
    return {"J": float(m.ObjVal), "schedule_raw": sched, "schedule": idx,
            "gap": float(m.MIPGap), "runtime": float(m.Runtime),
            "status": int(m.Status), "bound": float(m.ObjBound),
            "n_vars": m.NumVars, "n_cons": m.NumConstrs}


# ----------------------------------------------------------------------
def milp_expert(inst, time_limit=10.0, cache=True):
    """A callable (s, t) -> optimal action index, solved fresh by Gurobi.

    This is the DAgger expert: it can be queried at *any* state the learner
    visits, not just states on the optimal trajectory, which is exactly what
    behaviour cloning alone cannot give you.
    """
    memo: dict = {}

    def expert(s, t):
        key = (round(float(np.sum(s)), 9), t, tuple(np.round(s, 6)))
        if cache and key in memo:
            return memo[key]
        T = inst.T - t
        if T <= 0:
            return 0
        r = solve_milp_actions(inst, s0=s, T=T, time_limit=time_limit)
        a = r["schedule"][0]
        if cache:
            memo[key] = a
        return a
    return expert
