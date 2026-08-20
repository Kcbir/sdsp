"""Exact closed-loop solvers.

The key structural fact (and the main algorithmic upgrade over the paper's
MILP): with sigma = 0, linear dynamics, a linear stage cost and a
state-independent finite action set, the finite-horizon value function

    V_t(s) = min_a [ c'(W D_a s) + lam_tox tox(a) + gamma V_{t+1}(W D_a s) ]

is **concave, piecewise-linear and positively homogeneous** in s. Hence
V_t(s) = min over a finite set Gamma_t of affine functionals, and the exact
backward recursion is an alpha-vector backup:

    Gamma_t = { ( (1-kappa_a) o (W'(c + gamma w)),  lam_tox tox(a) + gamma b )
                : a in A, (w,b) in Gamma_{t+1} }

Because the state lives in the non-negative orthant, domination pruning is just
Pareto pruning -- no LP required. This solves *every* initial state at once and
runs at N ~ 30, T ~ 15, versus the paper's "exhaustive DP for N <= 5, T <= 3".
"""
from __future__ import annotations

import itertools

import numpy as np

from ..dynamics import rollout


# ----------------------------------------------------------------------
def _pareto_prune(A: np.ndarray, tol: float = 1e-12, cap: int | None = None,
                  probes: np.ndarray | None = None) -> np.ndarray:
    """Keep only the Pareto-minimal rows of A (all comparisons componentwise).

    Row i is redundant if some row j satisfies A[j] <= A[i] everywhere: then
    A[j] @ s <= A[i] @ s for every s >= 0, so row i never attains the min.
    """
    if len(A) <= 1:
        return A
    order = np.argsort(A.sum(axis=1))
    A = A[order]
    keep: list[int] = []
    for i in range(len(A)):
        dominated = False
        for j in keep:
            if np.all(A[j] <= A[i] + tol):
                dominated = True
                break
        if not dominated:
            keep.append(i)
    out = A[keep]
    if cap is not None and len(out) > cap:
        # Value-directed fallback: keep the vectors that are best at some probe
        # state. Marks the result as an upper bound rather than exact.
        if probes is None or len(probes) == 0:
            out = out[:cap]
        else:
            scores = probes @ out[:, :probes.shape[1]].T          # (P, K)
            best = np.unique(np.argmin(scores, axis=1))
            extra = [k for k in np.argsort(out.sum(axis=1)) if k not in set(best)]
            sel = list(best) + extra[: max(0, cap - len(best))]
            out = out[np.array(sel[:cap])]
    return out


class AlphaDP:
    """Exact closed-loop DP by alpha-vector backup.

    Attributes
    ----------
    Gamma : list of (K_t, N+1) arrays, one per t = 0..T. Last column is the
            affine constant.
    exact : False if the alpha-set cap ever bound (then values are upper bounds).
    """

    def __init__(self, inst, cap: int | None = 20000, n_probes: int = 256, seed: int = 0):
        self.inst = inst
        self.cap = cap
        self.exact = True
        rng = np.random.default_rng(seed)
        N = inst.N
        # probe states used only if the cap binds
        self.probes = np.abs(rng.normal(size=(n_probes, N))) * inst.s0.mean()
        self.probes = np.vstack([self.probes, inst.s0[None, :], np.eye(N)])
        self._solve()

    def _solve(self):
        inst = self.inst
        N, W, g = inst.N, inst.W, inst.gamma
        c_eff = inst.c + inst.lam_res * inst.resistant_nodes.astype(float)
        surv = 1.0 - inst.kill                     # (A,N)
        tox = inst.tox                             # (A,)
        A = surv.shape[0]

        Gamma = [np.zeros((1, N + 1))]             # Gamma_T
        for _ in range(inst.T):
            nxt = Gamma[0]
            w, b = nxt[:, :N], nxt[:, N]           # (K,N), (K,)
            # backup: alpha_{a,k} = surv_a o (W' (c_eff + g * w_k))
            M = (c_eff[None, :] + g * w) @ W       # (K,N)  == (W'(c+g w))'
            alphas = surv[:, None, :] * M[None, :, :]          # (A,K,N)
            consts = (inst.lam_tox * tox)[:, None] + g * b[None, :]   # (A,K)
            new = np.concatenate(
                [alphas.reshape(A * len(w), N), consts.reshape(A * len(w), 1)], axis=1)
            before = len(new)
            new = _pareto_prune(new, cap=self.cap,
                                probes=np.hstack([self.probes,
                                                  np.ones((len(self.probes), 1))]))
            if self.cap is not None and before > self.cap and len(new) >= self.cap:
                self.exact = False
            Gamma.insert(0, new)
        self.Gamma = Gamma

    # ------------------------------------------------------------------
    def value(self, s: np.ndarray, t: int = 0) -> float:
        sa = np.append(np.asarray(s, float), 1.0)
        return float(np.min(self.Gamma[t] @ sa))

    def act(self, s: np.ndarray, t: int) -> int:
        """Optimal action at (s,t): one-step backup against Gamma_{t+1}."""
        inst = self.inst
        c_eff = inst.c + inst.lam_res * inst.resistant_nodes.astype(float)
        s = np.asarray(s, float)
        s_hat = (1.0 - inst.kill) * s[None, :]              # (A,N)
        s_nxt = s_hat @ inst.W.T                            # (A,N)
        nxt = self.Gamma[min(t + 1, inst.T)]
        tail = np.min(np.hstack([s_nxt, np.ones((len(s_nxt), 1))]) @ nxt.T, axis=1)
        tot = s_nxt @ c_eff + inst.lam_tox * inst.tox + inst.gamma * tail
        return int(np.argmin(tot))

    def policy(self):
        return lambda s, t: self.act(s, t)

    @property
    def J(self) -> float:
        return self.value(self.inst.s0, 0)

    def optimal_schedule(self) -> list[int]:
        """The action sequence the optimal closed-loop policy actually realises
        on the nominal (noiseless) trajectory from s0."""
        _, _, acts = rollout(self.inst, self.policy(), return_traj=True)
        return acts

    def n_alphas(self) -> list[int]:
        return [len(G) for G in self.Gamma]


# ----------------------------------------------------------------------
def brute_force(inst, s0=None, T=None):
    """Enumerate all |A|^T open-loop schedules. Validation only.

    In the deterministic case the best open-loop schedule attains the
    closed-loop optimum, so this is the ground truth AlphaDP is checked against.
    """
    T = inst.T if T is None else T
    A = inst.n_actions
    if A ** T > 4_000_000:
        raise ValueError(f"brute force too large: {A}^{T}")
    best, arg = np.inf, None
    for seq in itertools.product(range(A), repeat=T):
        J = rollout(inst, list(seq), s0=s0, T=T)
        if J < best:
            best, arg = J, list(seq)
    return best, arg


def value_iteration_sampled(inst, n_states=4000, seed=0):
    """Fitted-value sanity check on random states (diagnostic, not used in results)."""
    rng = np.random.default_rng(seed)
    S = np.abs(rng.normal(size=(n_states, inst.N))) * inst.s0.mean()
    dp = AlphaDP(inst)
    return float(np.mean([dp.value(s) for s in S]))
