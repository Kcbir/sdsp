"""Spectral diagnostics.

Everything the paper hangs on, plus the quantities the write-up argues are the
*real* drivers: the spectral gap, the departure from normality, the Kreiss
constant, and the controllability radius

    rho* := min_{a in A^D} rho( W diag(1 - kappa^eff(a)) ),

which is the decision-relevant analogue of rho(W): not "is the pathogen
growing?" but "can any admissible action stop it growing?".
"""
from __future__ import annotations

import numpy as np


# ----------------------------------------------------------------------
# basic
# ----------------------------------------------------------------------
def rho(A: np.ndarray) -> float:
    """Spectral radius."""
    return float(np.max(np.abs(np.linalg.eigvals(A))))


def eigs_sorted(A: np.ndarray):
    w, V = np.linalg.eig(A)
    idx = np.argsort(-np.abs(w))
    return w[idx], V[:, idx]


def perron_vectors(A: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Right and left Perron vectors (u, v) and the Perron root.

    Normalised so that u >= 0, v >= 0, v @ u = 1. For a non-negative
    irreducible A this is the Perron-Frobenius pair; for reducible A we take
    the dominant eigenvector and fix the sign, which is what the greedy policy
    needs anyway.

    v is Fisher's *reproductive value*: v_i is the long-run contribution of one
    unit of burden at node i to total future burden. It is the correct weight
    for a one-step-lookahead policy, and using c instead of v is precisely the
    myopic error.
    """
    w, V = eigs_sorted(A)
    u = np.real(V[:, 0])
    wl, Vl = eigs_sorted(A.T)
    v = np.real(Vl[:, 0])
    if u.sum() < 0:
        u = -u
    if v.sum() < 0:
        v = -v
    u = np.abs(u) if np.all(u <= 1e-12) else u
    v = np.abs(v) if np.all(v <= 1e-12) else v
    nrm = float(v @ u)
    if abs(nrm) > 1e-14:
        v = v / nrm
    return u, v, float(np.abs(w[0]))


def spectral_gap(A: np.ndarray) -> float:
    """|lambda_2| / |lambda_1|. Near 1 means a slow, badly-conditioned split;
    near 0 means the dominant mode takes over immediately."""
    w, _ = eigs_sorted(A)
    if len(w) < 2 or abs(w[0]) < 1e-14:
        return 0.0
    return float(abs(w[1]) / abs(w[0]))


# ----------------------------------------------------------------------
# non-normality
# ----------------------------------------------------------------------
def henrici(A: np.ndarray) -> float:
    """Henrici's departure from normality, normalised by ||A||_F.

    dep(A) = sqrt(||A||_F^2 - sum_i |lambda_i|^2) / ||A||_F  in [0,1].
    Zero iff A is normal. The write-up's claim is that this, not rho alone,
    is what makes the myopic policy fail: the extremal instance of
    Construction 3.2 is a *defective* (Jordan) matrix in the alpha,beta -> 0
    limit, i.e. maximally non-normal.
    """
    w = np.linalg.eigvals(A)
    fro2 = float(np.sum(np.abs(A) ** 2))
    if fro2 < 1e-300:
        return 0.0
    dep2 = max(fro2 - float(np.sum(np.abs(w) ** 2)), 0.0)
    return float(np.sqrt(dep2 / fro2))


def kreiss_lower_bound(A: np.ndarray, n_radii: int = 40, n_theta: int = 180) -> float:
    """Lower bound on the discrete-time Kreiss constant

        K(A) = sup_{|z| > 1} (|z| - 1) || (zI - A)^{-1} ||_2,

    evaluated on a polar grid, after rescaling A so rho(A) = 1 (so the constant
    measures transient amplification *relative to* the asymptotic rate). By the
    Kreiss matrix theorem K <= sup_n ||A^n|| <= e*N*K, so K is a two-sided proxy
    for how much damage a non-normal operator can do before its asymptotics bite.
    """
    r = rho(A)
    if r < 1e-14:
        return 1.0
    B = A / r
    N = B.shape[0]
    best = 1.0
    for eps in np.geomspace(1e-3, 3.0, n_radii):
        rad = 1.0 + eps
        for th in np.linspace(0, 2 * np.pi, n_theta, endpoint=False):
            z = rad * np.exp(1j * th)
            try:
                R = np.linalg.inv(z * np.eye(N) - B)
            except np.linalg.LinAlgError:
                continue
            best = max(best, eps * float(np.linalg.norm(R, 2)))
    return float(best)


def transient_amplification(A: np.ndarray, n_max: int = 60) -> tuple[float, int]:
    """max_n ||A^n||_2 / rho(A)^n and the n attaining it.

    A normal matrix gives 1 at n=0. Large values mean the population can surge
    far above its asymptotic trajectory -- the mechanism by which a myopically
    'controlled' infection rebounds.
    """
    r = rho(A)
    if r < 1e-14:
        return 1.0, 0
    B = A / r
    P = np.eye(B.shape[0])
    best, arg = 1.0, 0
    for n in range(1, n_max + 1):
        P = P @ B
        val = float(np.linalg.norm(P, 2))
        if val > best:
            best, arg = val, n
        if val < 1e-8 and n > 5:
            break
    return best, arg


# ----------------------------------------------------------------------
# control-relevant
# ----------------------------------------------------------------------
def controllability_radius(inst) -> tuple[float, int]:
    """rho* = min_a rho(W_eff(a)) and the minimising action index."""
    Ws = inst.all_W_eff()
    vals = np.array([rho(Ws[j]) for j in range(Ws.shape[0])])
    j = int(np.argmin(vals))
    return float(vals[j]), j


def regime(gamma_rho: float, tol: float = 5e-2) -> str:
    if gamma_rho < 1.0 - tol:
        return "subcritical"
    if gamma_rho > 1.0 + tol:
        return "supercritical"
    return "critical"


def evolvability(inst, s: np.ndarray) -> float:
    """Diversity-based Res(.): Shannon entropy of the strain distribution,
    weighted by burden. High diversity = many independent shots at escape.
    Offered as the principled alternative to 'mass on resistant nodes'.
    """
    if inst.strain_of is None:
        return 0.0
    tot = s.sum()
    if tot <= 0:
        return 0.0
    ks = np.unique(inst.strain_of)
    p = np.array([s[inst.strain_of == k].sum() for k in ks]) / tot
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def spectral_report(inst) -> dict:
    """One-stop diagnostic dictionary for an instance."""
    W = inst.W
    r = rho(W)
    rs, js = controllability_radius(inst)
    amp, amp_n = transient_amplification(W)
    return {
        "rho": r,
        "gamma_rho": inst.gamma * r,
        "regime": regime(inst.gamma * r),
        "gap_ratio": spectral_gap(W),
        "henrici": henrici(W),
        "kreiss_lb": kreiss_lower_bound(W),
        "amp": amp,
        "amp_n": amp_n,
        "rho_star": rs,
        "gamma_rho_star": inst.gamma * rs,
        "regime_star": regime(inst.gamma * rs),
        "rho_star_action": inst.action_label(js),
        "rho_star_j": js,
    }
