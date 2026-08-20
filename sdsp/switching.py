"""Switching stabilizability: when does drug ORDER matter, exactly?

The claim
---------
Let {A_1, ..., A_d} be the per-stage population operators, one per drug. Two
numbers decide everything:

    rho_fix  := min_a rho(A_a)            best achievable growth with ONE fixed drug
    rho_low  := lower spectral radius     best achievable growth over ALL schedules
                = inf_k min_{a_1..a_k} rho(A_{a_1} ... A_{a_k})^{1/k}

and the assertion is:

    **Drug order matters if and only if   rho_low < 1 <= rho_fix.**

Below that band, any single drug already cures and sequencing is a refinement.
Above it, nothing cures and sequencing only changes how fast you lose. *Inside*
it, sequencing is the difference between cure and failure -- a qualitative phase
boundary, not a quantitative gap. This is the "stabilizable by switching but not
by any fixed mode" phenomenon from switched-systems theory (Jungers; Blondel &
Tsitsiklis), and the biological reading is exactly collateral sensitivity: two
drugs that each fail can compose into a product that contracts.

Note what this repairs. A criterion of the form gamma*rho(W) < 1 collapses at
gamma = 1 to "the pathogen is not growing", which is false for every untreated
infection and therefore discriminates nothing. The criterion above is about the
*controlled* system and stays sharp at gamma = 1.

What is and is not computable
-----------------------------
rho_low is NOT computable in general -- it is not even algorithmically decidable
whether it is < 1 for general matrix families. So we never claim to compute it.
We compute certified two-sided bounds and report the band:

  UPPER  an explicit periodic schedule with rho(prod)^{1/k} < 1 is a *witness*:
         it is a concrete protocol, and its contraction factor is just an
         eigenvalue, so the certificate is exact up to floating point.
  LOWER  (i) determinant bound: rho_low >= min_a |det A_a|^{1/n}, exactly, free.
         (ii) SDP bound: if some P > 0 satisfies A_a' P A_a >= g^2 P for every a,
              then no schedule can contract faster than g, so rho_low >= g.

When UPPER < 1 <= rho_fix we have *proved* order matters for that family.
When LOWER >= 1 we have *proved* it does not. In between, the answer is open and
we say so.

The learning problem lives in the UPPER bound: finding a short contractive
product is a combinatorial search over d^k schedules, and that is what the
policy network in `sdsp.policies.rl` is trained to do.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np

try:
    import cvxpy as cp
    HAVE_CVXPY = True
except Exception:                                        # pragma: no cover
    HAVE_CVXPY = False
    cp = None


# ======================================================================
# Elementary quantities
# ======================================================================
def spectral_radius(A: np.ndarray) -> float:
    return float(np.max(np.abs(np.linalg.eigvals(A))))


def fixed_best(mats) -> tuple[float, int]:
    """rho_fix = min_a rho(A_a), and the minimising drug index.

    "The best you can do with one drug, forever." Note this is *not* the same
    as the best constant policy for a finite horizon, but for the cure/no-cure
    question it is the right object: rho >= 1 means the population persists.
    """
    vals = [spectral_radius(A) for A in mats]
    j = int(np.argmin(vals))
    return float(vals[j]), j


def periodic_radius(mats, seq) -> float:
    """rho(A_{a_1} ... A_{a_k})^{1/k}: the per-stage growth of a cyclic schedule.

    This is an exact, certifiable number. If it is < 1 the cyclic schedule
    drives the population to extinction, whatever the initial state.
    """
    seq = list(seq)
    P = np.eye(mats[0].shape[0])
    for a in seq:
        P = mats[a] @ P
    r = spectral_radius(P)
    return float(r ** (1.0 / len(seq))) if r > 0 else 0.0


def product_of(mats, seq) -> np.ndarray:
    P = np.eye(mats[0].shape[0])
    for a in seq:
        P = mats[a] @ P
    return P


# ======================================================================
# Certified lower bounds on the lower spectral radius
# ======================================================================
def det_lower_bound(mats) -> float:
    """rho_low >= min_a |det A_a|^{1/n}.  Exact, O(d n^3), no solver needed.

    Proof: for any product P of length k, |det P| = prod_i |det A_{a_i}| >=
    (min_a |det A_a|)^k, and rho(P)^n >= |det P|, so
    rho(P)^{1/k} >= (min_a |det A_a|)^{1/n}.

    Biologically this says: a drug family cannot sterilise faster than its
    least-volume-contracting member allows. It is the reason a family of drugs
    that each merely *slow* growth can never be composed into a cure.
    """
    n = mats[0].shape[0]
    dets = [abs(float(np.linalg.det(A))) for A in mats]
    return float(min(dets) ** (1.0 / n))


def sdp_lower_bound(mats, tol=1e-4, lo=0.0, hi=None, max_iter=40, solver=None):
    """SDP lower bound on rho_low by bisection on a common expanding metric.

    Feasibility problem at level g:  find P >= I  such that
        A_a' P A_a  >=  g^2 P    for all a.
    If feasible, then ||A_a x||_P >= g ||x||_P for every a and every x, so no
    product of length k can shrink the P-norm below g^k, hence rho_low >= g.

    This is the exact dual of the standard common-quadratic *upper* bound on the
    joint spectral radius, and like it, it is tight only up to a sqrt(n) factor.
    Requires cvxpy; returns None if unavailable.
    """
    if not HAVE_CVXPY:
        return None
    n = mats[0].shape[0]
    if hi is None:
        hi = min(spectral_radius(A) for A in mats) + 1e-9

    def feasible(g):
        P = cp.Variable((n, n), symmetric=True)
        cons = [P >> np.eye(n)]
        for A in mats:
            cons.append(A.T @ P @ A - (g ** 2) * P >> 0)
        prob = cp.Problem(cp.Minimize(cp.trace(P)), cons)
        try:
            prob.solve(solver=solver)
        except Exception:
            return False
        return prob.status in ("optimal", "optimal_inaccurate")

    if not feasible(lo):
        return 0.0
    for _ in range(max_iter):
        if hi - lo < tol:
            break
        mid = 0.5 * (lo + hi)
        if feasible(mid):
            lo = mid
        else:
            hi = mid
    return float(lo)


def jsr_upper_cqlf(mats, tol=1e-4, max_iter=40, solver=None):
    """Common-quadratic (degree-2 SOS) UPPER bound on the *joint* spectral radius.

    min g s.t. exists P > 0 with A_a' P A_a <= g^2 P for all a.
    JSR <= g <= sqrt(n) * JSR   (Parrilo & Jadbabaie 2008, degree-2 case).

    This bounds the WORST case: if g < 1, *every* schedule cures, including a
    random or non-adherent one. That is a much stronger and more clinically
    meaningful guarantee than a single good schedule, and it is the right thing
    to report when arguing a regimen is robust to patient non-adherence.
    """
    if not HAVE_CVXPY:
        return None
    n = mats[0].shape[0]
    lo, hi = 0.0, max(spectral_radius(A) for A in mats) * n ** 0.5 + 1.0

    def feasible(g):
        P = cp.Variable((n, n), symmetric=True)
        cons = [P >> np.eye(n)]
        for A in mats:
            cons.append((g ** 2) * P - A.T @ P @ A >> 0)
        prob = cp.Problem(cp.Minimize(cp.trace(P)), cons)
        try:
            prob.solve(solver=solver)
        except Exception:
            return False
        return prob.status in ("optimal", "optimal_inaccurate")

    if not feasible(hi):
        return None
    for _ in range(max_iter):
        if hi - lo < tol:
            break
        mid = 0.5 * (lo + hi)
        if feasible(mid):
            hi = mid
        else:
            lo = mid
    return float(hi)


# ======================================================================
# Searching for a contractive periodic schedule (the UPPER bound)
# ======================================================================
@dataclass
class ScheduleResult:
    radius: float
    schedule: list[int]
    length: int
    n_evaluated: int
    method: str
    certified: bool = True          # radius is an exact eigenvalue computation
    extra: dict = field(default_factory=dict)


def search_periodic_exhaustive(mats, k_max=4, k_min=1, prune=True,
                               max_products=2_000_000) -> ScheduleResult:
    """Exhaustively minimise rho(prod)^{1/k} over all schedules of length <= k_max.

    Exhaustive search is d^k, so this is the ground truth for small d,k and the
    reference the learned searcher must be measured against. Cyclic rotations
    are equivalent (rho(AB) = rho(BA)), so we canonicalise to the
    lexicographically smallest rotation, which cuts the search by ~k.
    """
    d = len(mats)
    best, best_seq, best_k, n_eval = np.inf, None, 0, 0
    seen: set[tuple] = set()
    for k in range(k_min, k_max + 1):
        if d ** k > max_products:
            break
        for seq in itertools.product(range(d), repeat=k):
            rots = [tuple(seq[i:] + seq[:i]) for i in range(k)]
            canon = min(rots)
            if prune and canon in seen:
                continue
            seen.add(canon)
            r = periodic_radius(mats, seq)
            n_eval += 1
            if r < best:
                best, best_seq, best_k = r, list(seq), k
    return ScheduleResult(best, best_seq, best_k, n_eval, "exhaustive")


def search_periodic_beam(mats, k_max=12, beam=64, seed=0, restarts=1,
                         objective="radius") -> ScheduleResult:
    """Beam search over schedule prefixes.

    Scoring a *prefix* is the difficulty: rho of a partial product is not
    monotone in the way a beam search wants. We therefore score prefixes by the
    spectral norm of the partial product (which IS submultiplicative, so it
    bounds what any completion can achieve) and re-rank finished cycles by the
    exact periodic radius. This is the baseline the RL searcher must beat.
    """
    d, n = len(mats), mats[0].shape[0]
    rng = np.random.default_rng(seed)
    best, best_seq, best_k, n_eval = np.inf, None, 0, 0

    for _ in range(restarts):
        frontier = [([a], mats[a]) for a in range(d)]
        for k in range(1, k_max + 1):
            scored = []
            for seq, P in frontier:
                r = spectral_radius(P)
                n_eval += 1
                rad = float(r ** (1.0 / len(seq))) if r > 0 else 0.0
                if rad < best:
                    best, best_seq, best_k = rad, list(seq), len(seq)
                scored.append((float(np.linalg.norm(P, 2)) ** (1.0 / len(seq)),
                               seq, P))
            scored.sort(key=lambda t: t[0])
            keep = scored[:beam]
            if k == k_max:
                break
            frontier = [(seq + [a], mats[a] @ P) for _, seq, P in keep
                        for a in range(d)]
    return ScheduleResult(best, best_seq, best_k, n_eval, f"beam({beam})")


def search_periodic_annealed(mats, k=6, iters=20000, seed=0, T0=None,
                             restarts=4) -> ScheduleResult:
    """Simulated annealing over cyclic schedules of fixed length k.

    Included because it scales to d and k where exhaustive is hopeless, and
    because it is the honest non-learned baseline: any claim that a neural
    searcher helps must beat this at equal evaluation budget.
    """
    rng = np.random.default_rng(seed)
    d = len(mats)
    best_all, best_seq_all, n_eval = np.inf, None, 0
    for _ in range(restarts):
        cur = list(rng.integers(d, size=k))
        f = periodic_radius(mats, cur); n_eval += 1
        T0_ = T0 if T0 is not None else max(0.1 * abs(f), 1e-6)
        best, best_seq = f, list(cur)
        for i in range(iters // restarts):
            temp = T0_ * (1e-3 ** (i / max(iters // restarts - 1, 1)))
            cand = list(cur)
            if rng.random() < 0.7:
                cand[rng.integers(k)] = int(rng.integers(d))
            else:
                i1, i2 = rng.choice(k, size=2, replace=False)
                cand[i1], cand[i2] = cand[i2], cand[i1]
            fc = periodic_radius(mats, cand); n_eval += 1
            if fc <= f or rng.random() < np.exp(-(fc - f) / max(temp, 1e-300)):
                cur, f = cand, fc
                if fc < best:
                    best, best_seq = fc, list(cand)
        if best < best_all:
            best_all, best_seq_all = best, best_seq
    return ScheduleResult(best_all, best_seq_all, k, n_eval, "annealed")


# ======================================================================
# The verdict
# ======================================================================
@dataclass
class StabilizabilityVerdict:
    rho_fix: float
    best_fixed_drug: int
    rho_low_upper: float             # witnessed by an explicit schedule
    rho_low_lower: float             # certified lower bound
    witness: list[int]
    witness_length: int
    jsr_upper: float | None
    verdict: str
    gain: float                      # rho_fix / rho_low_upper
    detail: dict = field(default_factory=dict)

    def __str__(self):
        w = "->".join(map(str, self.witness)) if self.witness else "-"
        s = [
            f"rho_fix (best single drug #{self.best_fixed_drug}) = {self.rho_fix:.4f}",
            f"rho_low in [{self.rho_low_lower:.4f}, {self.rho_low_upper:.4f}]  "
            f"witness = [{w}] (period {self.witness_length})",
        ]
        if self.jsr_upper is not None:
            s.append(f"JSR upper (all schedules cure if < 1) = {self.jsr_upper:.4f}")
        s.append(f"switching gain rho_fix/rho_low = {self.gain:.3f}x")
        s.append(f"VERDICT: {self.verdict}")
        return "\n".join(s)


def stabilizability_verdict(mats, k_max=6, search="exhaustive", use_sdp=True,
                            dilution_free=False, **kw) -> StabilizabilityVerdict:
    """The headline computation: does order matter for this drug family?

    Two modes, because the right question depends on whether the experimenter
    controls an overall scale.

    `dilution_free=False` (absolute).  Thresholds against 1, i.e. the operators
    already carry every loss term. Four verdicts:

      ORDER MATTERS (proved)      rho_low_upper < 1 <= rho_fix.
      NO SWITCHING NEEDED         rho_fix < 1; one drug already cures.
      HOPELESS (proved)           rho_low_lower >= 1; no schedule can cure.
      UNDETERMINED                bounds straddle 1.

    `dilution_free=True` (scale-free).  For serial passage the dilution 1/D is a
    free positive scalar that multiplies *every* operator identically, so every
    radius scales as 1/D and thresholding against 1 is meaningless: at D=1 all
    of Mira's operators expand (growth rates are non-negative, so nothing can
    contract) and the absolute verdict is unreachable *by construction*. The
    scale-free question is the one the experimenter actually faces:

        is there a dilution at which switching cures and no fixed drug does?

    and the answer is exactly `rho_low_upper < rho_fix`, with the admissible
    dilutions being the interval (rho_low_upper, rho_fix]. Verdicts:

      ORDER MATTERS AT SOME DILUTION (proved)   rho_low_upper < rho_fix
      NO GAIN AT ANY DILUTION                   rho_low_upper == rho_fix

    Both are certified: rho_fix and rho_low_upper are eigenvalues of explicitly
    constructed matrices, so the window is a proved-nonempty interval, not an
    estimate. The gain rho_fix/rho_low_upper is invariant to D, and log10(gain)
    is exactly the width of that window in decades of dilution.
    """
    rho_fix, jfix = fixed_best(mats)
    if search == "exhaustive":
        res = search_periodic_exhaustive(mats, k_max=k_max, **kw)
    elif search == "beam":
        res = search_periodic_beam(mats, k_max=k_max, **kw)
    elif search == "annealed":
        res = search_periodic_annealed(mats, k=k_max, **kw)
    else:
        raise ValueError(f"unknown search {search!r}")

    lower = det_lower_bound(mats)
    sdp_lo = sdp_lower_bound(mats) if use_sdp else None
    if sdp_lo is not None:
        lower = max(lower, sdp_lo)
    jsr = jsr_upper_cqlf(mats) if use_sdp else None

    upper = min(res.radius, rho_fix)         # a fixed drug is a period-1 schedule
    gain = float(rho_fix / max(upper, 1e-300))

    if dilution_free:
        if gain > 1.0 + 1e-12:
            verdict = ("ORDER MATTERS AT SOME DILUTION (proved): the cyclic "
                       "schedule beats every fixed drug")
        else:
            verdict = "NO GAIN AT ANY DILUTION: no schedule found beats the best fixed drug"
    elif rho_fix < 1.0:
        verdict = "NO SWITCHING NEEDED (a single drug already contracts)"
    elif upper < 1.0 <= rho_fix:
        verdict = "ORDER MATTERS (proved): switching cures, no fixed drug does"
    elif lower >= 1.0:
        verdict = "HOPELESS (proved): no schedule can contract this family"
    else:
        verdict = "UNDETERMINED: bounds straddle 1; report the band, do not guess"

    return StabilizabilityVerdict(
        rho_fix=rho_fix, best_fixed_drug=jfix,
        rho_low_upper=float(upper), rho_low_lower=float(lower),
        witness=res.schedule, witness_length=res.length, jsr_upper=jsr,
        verdict=verdict, gain=gain,
        detail={"search": res.method, "n_evaluated": res.n_evaluated,
                "det_bound": det_lower_bound(mats), "sdp_bound": sdp_lo,
                "dilution_free": dilution_free,
                "D_low": float(upper), "D_high": float(rho_fix)},
    )


# ======================================================================
# Robustness of a found schedule
# ======================================================================
def robustness_margin(mats, seq, n_grid=40):
    """Largest uniform multiplicative growth perturbation a schedule survives.

    Replaces every A_a by (1+eps) A_a and finds the largest eps with the cyclic
    schedule still contracting. Since scaling all matrices by (1+eps) scales the
    periodic radius by exactly (1+eps), this is closed form:

        eps* = 1/rho_periodic - 1

    which is why we report it: it converts "the schedule works" into "the
    schedule tolerates an X% error in every growth rate", the only form in which
    a wet-lab or clinical reader should trust a model-derived protocol.
    """
    r = periodic_radius(mats, seq)
    return float(1.0 / max(r, 1e-300) - 1.0)


def perturbation_sweep(mats, seq, rel_noise=0.10, n_draws=500, seed=0):
    """Does the schedule still contract when every growth rate is jittered?

    Draws multiplicative log-normal noise on each matrix entry and reports the
    fraction of draws in which the schedule still has periodic radius < 1. This
    is the parameter-uncertainty analogue of a power calculation, and it is the
    number to quote when the underlying growth rates were measured with error --
    as they always are.
    """
    rng = np.random.default_rng(seed)
    ok, radii = 0, []
    for _ in range(n_draws):
        pert = [A * np.exp(rng.normal(0, rel_noise, size=A.shape)) for A in mats]
        r = periodic_radius(pert, seq)
        radii.append(r)
        ok += int(r < 1.0)
    radii = np.array(radii)
    return {"p_contract": ok / n_draws,
            "median_radius": float(np.median(radii)),
            "q05": float(np.quantile(radii, 0.05)),
            "q95": float(np.quantile(radii, 0.95)),
            "rel_noise": rel_noise}


def protocol_power(mats, seq, resample=None, rel_noise=0.10, n_draws=500,
                   seed=0, dilution=None):
    """Would the bench experiment still demonstrate the effect, given noisy rates?

    `perturbation_sweep` asks whether the schedule still contracts *at the scale
    the operators were built with*. For serial passage that question is empty:
    the operators are built dilution-free, nothing contracts at D=1, and the
    answer is always "no" regardless of the data. The experimenter's question is
    different and has two arms:

        pick D from the nominal data, commit to the protocol, then run it in a
        world where the growth rates are 10% off. Does the cycle still clear the
        culture, AND does the control arm (every single drug) still fail?

    Both arms matter. A draw where the cycle cures but some fixed drug also
    cures is a failed *demonstration* even though the treatment worked, because
    it no longer isolates order as the cause.

    D defaults to the geometric midpoint sqrt(rho_switch * rho_fix) of the
    nominal window, which is the dilution furthest (in log) from both failure
    modes and therefore the one an experimenter should actually pick.

    `resample(rng) -> mats` regenerates the operators from perturbed *measured
    inputs*. Prefer it: the fallback jitters matrix entries directly, which also
    corrupts the mutation structure and is only a crude proxy. The witness
    schedule is held fixed across draws, as it must be -- you commit to a
    protocol before you run it.
    """
    rng = np.random.default_rng(seed)
    rho_fix0, _ = fixed_best(mats)
    rho_sw0 = periodic_radius(mats, seq)
    if dilution is None:
        dilution = float(np.sqrt(max(rho_sw0, 1e-300) * rho_fix0))

    n_cure = n_ctrl = n_demo = n_gain = 0
    gains, sw_r = [], []
    for _ in range(n_draws):
        if resample is not None:
            m = resample(rng)
        else:
            m = [A * np.exp(rng.normal(0, rel_noise, size=A.shape)) for A in mats]
        sw = periodic_radius(m, seq)
        fx, _ = fixed_best(m)
        cure = sw < dilution                  # schedule contracts at this D
        ctrl = fx >= dilution                 # no single drug contracts at this D
        n_cure += cure
        n_ctrl += ctrl
        n_demo += (cure and ctrl)
        n_gain += (fx > sw + 1e-12)
        gains.append(fx / max(sw, 1e-300))
        sw_r.append(sw / dilution)

    gains = np.array(gains)
    return {
        "dilution": dilution,
        "rho_switch_nominal": rho_sw0,
        "rho_fix_nominal": rho_fix0,
        "p_cure": n_cure / n_draws,           # treatment arm works
        "p_control_fails": n_ctrl / n_draws,  # control arm fails, as predicted
        "p_demonstrates": n_demo / n_draws,   # BOTH -- the experiment succeeds
        "p_gain_positive": n_gain / n_draws,  # scale-free: switching still wins
        "median_gain": float(np.median(gains)),
        "gain_q05": float(np.quantile(gains, 0.05)),
        "gain_q95": float(np.quantile(gains, 0.95)),
        "median_switch_radius_at_D": float(np.median(sw_r)),
        "rel_noise": rel_noise,
        "n_draws": n_draws,
        "resampled_inputs": resample is not None,
    }


# ======================================================================
# Pairwise / subset screening
# ======================================================================
def screen_subsets(mats, names=None, sizes=(2, 3), k_max=6, use_sdp=False,
                   progress=False):
    """Which SUBSETS of the formulary are switching-stabilizable?

    This is the deliverable on real data: given 15 measured beta-lactams, which
    pairs and triples can be cycled into a cure that no member achieves alone?
    Returns a list of dicts sorted by switching gain, so the top of the list is
    a ranked, testable set of candidate protocols.
    """
    names = names or [str(i) for i in range(len(mats))]
    out = []
    for size in sizes:
        for combo in itertools.combinations(range(len(mats)), size):
            sub = [mats[i] for i in combo]
            v = stabilizability_verdict(sub, k_max=k_max, use_sdp=use_sdp)
            out.append({
                "drugs": [names[i] for i in combo],
                "idx": list(combo),
                "rho_fix": v.rho_fix,
                "rho_low_upper": v.rho_low_upper,
                "rho_low_lower": v.rho_low_lower,
                "gain": v.gain,
                "verdict": v.verdict,
                "witness": [names[combo[a]] for a in v.witness] if v.witness else [],
                "period": v.witness_length,
                "margin": robustness_margin(sub, v.witness) if v.witness else np.nan,
            })
            if progress and len(out) % 25 == 0:
                print(f"    screened {len(out)} subsets")
    out.sort(key=lambda r: (-int("ORDER MATTERS" in r["verdict"]), -r["gain"]))
    return out
