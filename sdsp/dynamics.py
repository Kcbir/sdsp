"""Within-host dynamics and rollout.

Implements eqs. (2)-(5) of the paper, plus two fixes discussed in the review:

* `noise="branching"` replaces the Gaussian perturbation with a Poisson/binomial
  multitype branching process. The Gaussian model of eq. (3) puts mass on
  s_{t+1} < 0, contradicting s in R^N_+; the branching model keeps the state
  non-negative and integral and hands you extinction probability for free.
* `density_dependence` adds a carrying capacity, so that competitive release --
  the mechanism the adaptive-therapy story actually rests on -- exists at all.
  With the linear model there is no competition, hence no competitive release.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class NoiseConfig:
    kind: str = "none"           # none | gaussian | branching
    sigma: float = 0.0           # demographic noise scale (gaussian)
    delivery: bool = False       # multiplicative Beta delivery noise on kappa^eff
    beta_a: float = 8.0
    beta_b: float = 2.0          # mean a/(a+b) = 0.8, sd ~ 0.12
    capacity: float | None = None  # carrying capacity K; None = linear model


def apply_drug(inst, s: np.ndarray, j: int, rng=None, cfg: NoiseConfig | None = None):
    """Eq. (2): s_hat = (1 - kappa^eff) * s, with optional delivery noise."""
    kill = inst.kill[j]
    if cfg is not None and cfg.delivery and rng is not None:
        eta = rng.beta(cfg.beta_a, cfg.beta_b)
        kill = np.clip(kill * eta, 0.0, 1.0)
    return (1.0 - kill) * s


def propagate(inst, s_hat: np.ndarray, rng=None, cfg: NoiseConfig | None = None,
              W: np.ndarray | None = None):
    """Eq. (3): s_{t+1} = W s_hat + noise."""
    cfg = cfg or NoiseConfig()
    Wm = inst.W if W is None else W
    mean = Wm @ s_hat

    if cfg.capacity is not None:
        # logistic damping: growth is throttled as total burden approaches K.
        tot = float(s_hat.sum())
        mean = s_hat + (mean - s_hat) * max(0.0, 1.0 - tot / cfg.capacity)

    if cfg.kind == "none" or rng is None:
        out = mean
    elif cfg.kind == "gaussian":
        out = mean + cfg.sigma * rng.normal(0.0, np.sqrt(np.maximum(s_hat, 0.0)))
    elif cfg.kind == "branching":
        # Multitype branching: each node u sends Poisson(W_vu * s_hat_u) to v.
        lam = Wm * s_hat[None, :]
        out = rng.poisson(np.maximum(lam, 0.0)).sum(axis=1).astype(float)
    else:
        raise ValueError(f"unknown noise kind {cfg.kind!r}")
    return np.maximum(out, 0.0)


def step(inst, s, j, rng=None, cfg=None, W=None):
    cfg = cfg or NoiseConfig()
    s_hat = apply_drug(inst, s, j, rng, cfg)
    s_next = propagate(inst, s_hat, rng, cfg, W)
    stage_cost = float(inst.c @ s_next) + inst.lam_tox * float(inst.tox[j])
    if inst.lam_res:
        stage_cost += inst.lam_res * inst.res_penalty(s_next)
    return s_next, stage_cost


def rollout(inst, policy, s0=None, T=None, rng=None, cfg=None, return_traj=False):
    """Run a policy and return its discounted cost J.

    `policy` is either a callable (s, t) -> action index, or a sequence of
    action indices of length T (an open-loop schedule).
    """
    cfg = cfg or NoiseConfig()
    s = (inst.s0 if s0 is None else np.asarray(s0, float)).copy()
    T = inst.T if T is None else T
    seq = None if callable(policy) else list(policy)
    J, traj, acts = 0.0, [s.copy()], []
    for t in range(T):
        j = policy(s, t) if seq is None else seq[t]
        s, cost = step(inst, s, j, rng, cfg)
        J += (inst.gamma ** t) * cost
        traj.append(s.copy())
        acts.append(j)
    if return_traj:
        return J, np.array(traj), acts
    return J


def evaluate(inst, policy, n_episodes=1, seed=0, cfg=None, s0=None):
    """Monte-Carlo evaluation. Returns (mean, std, all_values)."""
    cfg = cfg or NoiseConfig()
    if cfg.kind == "none" and not cfg.delivery:
        n_episodes = 1
    vals = []
    for e in range(n_episodes):
        rng = np.random.default_rng(seed + e)
        vals.append(rollout(inst, policy, s0=s0, rng=rng, cfg=cfg))
    v = np.array(vals)
    return float(v.mean()), float(v.std()), v


def extinction_probability(inst, policy, n_episodes=200, seed=0, thresh=0.5,
                           cfg: NoiseConfig | None = None):
    """P(cure): fraction of branching-process trajectories that hit zero burden.

    This is the clinically meaningful output that the expected-burden objective
    hides, and it is exactly the classical sub/supercritical branching quantity.
    """
    cfg = cfg or NoiseConfig(kind="branching")
    if cfg.kind != "branching":
        cfg = NoiseConfig(kind="branching", delivery=cfg.delivery,
                          beta_a=cfg.beta_a, beta_b=cfg.beta_b, capacity=cfg.capacity)
    hits = 0
    for e in range(n_episodes):
        rng = np.random.default_rng(seed + e)
        _, traj, _ = rollout(inst, policy, rng=rng, cfg=cfg, return_traj=True)
        if traj[-1].sum() < thresh:
            hits += 1
    return hits / n_episodes


# ----------------------------------------------------------------------
# PK memory (section 4.2)
# ----------------------------------------------------------------------
class PKMemory:
    """Exponential carry-in: C_i(t) = sum_{s<=t} r_{i,s} exp(-(t-s)*Delta/tau_i).

    Wrapping a policy in this makes the effective kill at stage t a function of
    the whole dosing history, so the relevant object becomes the *joint*
    spectral radius of {W_eff(t)} rather than a single rho.
    """

    def __init__(self, inst):
        self.inst = inst
        self.C = np.zeros(inst.d)

    def decay(self):
        dt = self.inst.stage_hours
        for i, drug in enumerate(self.inst.drugs):
            self.C[i] *= np.exp(-dt / max(drug.half_life_h, 1e-9))

    def dose(self, j: int):
        for i, l in self.inst.actions[j]:
            self.C[i] += self.inst.drugs[i].doses[l]

    def effective_kill(self) -> np.ndarray:
        k = np.zeros(self.inst.N)
        for i, drug in enumerate(self.inst.drugs):
            if self.C[i] > 0:
                k += drug.phi(float(self.C[i])) * drug.kappa
        return np.clip(k, 0.0, 1.0)

    def rollout(self, policy, s0=None, T=None, rng=None, cfg=None):
        """Rollout honouring PK carry-in rather than instantaneous kill."""
        inst, cfg = self.inst, (cfg or NoiseConfig())
        s = (inst.s0 if s0 is None else np.asarray(s0, float)).copy()
        T = inst.T if T is None else T
        self.C[:] = 0.0
        seq = None if callable(policy) else list(policy)
        J, traj, Weffs = 0.0, [s.copy()], []
        for t in range(T):
            j = policy(s, t) if seq is None else seq[t]
            self.decay()
            self.dose(j)
            kill = self.effective_kill()
            s_hat = (1.0 - kill) * s
            s = propagate(inst, s_hat, rng, cfg)
            Weffs.append(inst.W * (1.0 - kill)[None, :])
            J += (inst.gamma ** t) * (float(inst.c @ s) + inst.lam_tox * inst.tox[j])
            traj.append(s.copy())
        return J, np.array(traj), Weffs


def joint_spectral_radius_ub(mats, p=8):
    """Upper bound on the joint spectral radius of a finite matrix family,
    via the p-th root of the max norm of all length-p products (sub-additivity).
    Used to make the PK-memory statement in section 4.2 quantitative.
    """
    import itertools as it
    best = 0.0
    n = len(mats)
    if n == 0:
        return 0.0
    p = min(p, max(1, 12 // max(1, n // 3 + 1)))
    for combo in it.product(range(n), repeat=p):
        P = np.eye(mats[0].shape[0])
        for k in combo:
            P = P @ mats[k]
        best = max(best, float(np.linalg.norm(P, 2)))
    return best ** (1.0 / p)
