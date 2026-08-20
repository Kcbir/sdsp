"""Vectorised batch simulation.

`sdsp.dynamics.rollout` is the readable reference implementation; it evaluates
one schedule at a time in a Python loop. That is far too slow for the inner loop
of VNS (which sweeps |A| * T neighbours per descent step) and for PPO rollouts.

Everything here evaluates *many* schedules at once with a single einsum per
stage. Results are numerically identical to `rollout` in the deterministic case;
`tests/test_consistency.py` asserts that.
"""
from __future__ import annotations

import numpy as np


class FastSim:
    """Pre-compiled simulator for one instance.

    Caches the (A, N, N) stack of post-drug operators W_eff(a) = W diag(1-kappa_a)
    so that a batch of M schedules costs T einsums of shape (M,N,N) x (M,N).
    """

    __slots__ = ("inst", "Weff", "c_eff", "tox", "gamma", "N", "A", "disc")

    def __init__(self, inst):
        self.inst = inst
        self.Weff = np.ascontiguousarray(inst.all_W_eff())          # (A,N,N)
        self.c_eff = inst.c + inst.lam_res * inst.resistant_nodes.astype(float)
        self.tox = inst.lam_tox * inst.tox                          # (A,)
        self.gamma = inst.gamma
        self.N, self.A = inst.N, inst.n_actions
        self.disc = self.gamma ** np.arange(max(inst.T, 1) + 64)

    # ------------------------------------------------------------------
    def eval_batch(self, seqs, s0=None) -> np.ndarray:
        """(M,T) integer schedules -> (M,) discounted costs."""
        seqs = np.atleast_2d(np.asarray(seqs, dtype=np.int64))
        M, T = seqs.shape
        s0 = self.inst.s0 if s0 is None else np.asarray(s0, float)
        S = np.broadcast_to(s0, (M, self.N)).copy()
        J = np.zeros(M)
        for t in range(T):
            a = seqs[:, t]
            S = np.einsum("mij,mj->mi", self.Weff[a], S, optimize=True)
            J += self.disc[t] * (S @ self.c_eff + self.tox[a])
        return J

    def eval_one(self, seq, s0=None) -> float:
        return float(self.eval_batch(np.asarray(seq)[None, :], s0)[0])

    def traj_batch(self, seqs, s0=None) -> np.ndarray:
        """(M,T) schedules -> (M,T+1,N) state trajectories."""
        seqs = np.atleast_2d(np.asarray(seqs, dtype=np.int64))
        M, T = seqs.shape
        s0 = self.inst.s0 if s0 is None else np.asarray(s0, float)
        S = np.broadcast_to(s0, (M, self.N)).copy()
        out = np.empty((M, T + 1, self.N))
        out[:, 0] = S
        for t in range(T):
            S = np.einsum("mij,mj->mi", self.Weff[seqs[:, t]], S, optimize=True)
            out[:, t + 1] = S
        return out

    # ------------------------------------------------------------------
    def step_all_actions(self, S: np.ndarray) -> np.ndarray:
        """(M,N) states -> (M,A,N) successor states under every action.

        This is the workhorse for one-step-lookahead policies and for the
        vectorised local search: it turns the neighbourhood sweep into one
        tensor contraction.
        """
        S = np.atleast_2d(S)
        return np.einsum("aij,mj->mai", self.Weff, S, optimize=True)

    def onestep_costs(self, S: np.ndarray, weights=None) -> np.ndarray:
        """(M,N) states -> (M,A) one-step costs under a weight vector.

        weights=None uses c_eff (the myopic criterion); pass the left Perron
        vector to get the reproductive-value criterion.
        """
        w = self.c_eff if weights is None else np.asarray(weights, float)
        nxt = self.step_all_actions(S)                       # (M,A,N)
        return nxt @ w + self.tox[None, :]

    # ------------------------------------------------------------------
    def neighbourhood_single_flip(self, seq, s0=None):
        """All T*(A-1) single-stage reassignments of `seq`, evaluated in one go.

        Returns (best_value, best_schedule). This replaces the O(T*A) Python
        rollouts per descent step in the VNS local search.
        """
        seq = np.asarray(seq, dtype=np.int64)
        T = len(seq)
        cand = np.repeat(seq[None, :], T * self.A, axis=0)
        rows = np.arange(T * self.A)
        cand[rows, np.repeat(np.arange(T), self.A)] = np.tile(np.arange(self.A), T)
        vals = self.eval_batch(cand, s0)
        k = int(np.argmin(vals))
        return float(vals[k]), cand[k].tolist()

    def neighbourhood_swap(self, seq, s0=None):
        """All pairwise stage swaps of `seq`, evaluated in one go."""
        seq = np.asarray(seq, dtype=np.int64)
        T = len(seq)
        pairs = [(i, j) for i in range(T) for j in range(i + 1, T) if seq[i] != seq[j]]
        if not pairs:
            return np.inf, list(seq)
        cand = np.repeat(seq[None, :], len(pairs), axis=0)
        for r, (i, j) in enumerate(pairs):
            cand[r, i], cand[r, j] = seq[j], seq[i]
        vals = self.eval_batch(cand, s0)
        k = int(np.argmin(vals))
        return float(vals[k]), cand[k].tolist()


_CACHE: dict[int, FastSim] = {}


def get_sim(inst) -> FastSim:
    """Memoised FastSim per instance object (keyed by identity)."""
    key = id(inst)
    sim = _CACHE.get(key)
    if sim is None or sim.inst is not inst:
        sim = FastSim(inst)
        _CACHE[key] = sim
    return sim


def clear_cache():
    _CACHE.clear()
