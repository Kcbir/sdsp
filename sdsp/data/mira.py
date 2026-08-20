"""Build per-drug population operators from the Mira et al. (2015) landscapes.

Real measured input, one modelling layer, no synthetic data anywhere.

Model
-----
Node = TEM genotype (16 of them: all combinations of M69L, E104K, G238S, N276D).
Over one passage cycle under drug a:

    grow      genotype i multiplies by exp(r[a,i] * delta)     [r is MEASURED]
    mutate    a fraction mu moves to each Hamming-1 neighbour  [mu is a parameter]
    passage   the culture is diluted 1 : D                     [D is a KNOB]

so the per-cycle operator is

    A_a = (1/D) * M(mu) * diag(exp(r[:,a] * delta)).

Why dilution is load-bearing
----------------------------
Mira's numbers are *growth* rates: they are non-negative everywhere (min 0.034),
because the assay measures how fast a genotype grows in the presence of drug,
not how fast it dies. So every diag(exp(r*delta)) is expanding, and with no
other term nothing could ever contract. Contraction has to come from somewhere
real, and in a serial-passage experiment it comes from the dilution step. That
is not a modelling convenience -- it is the actual thing that happens on the
bench, and D is set by the experimenter.

The payoff: an exactly closed-form, testable prediction
-------------------------------------------------------
Since D enters as a scalar 1/D, every spectral radius scales as 1/D:

    rho_fix(D) = rho_fix(1)/D          rho_low(D) = rho_low(1)/D

Therefore the set of dilution factors at which **switching cures but no single
drug does** is exactly the interval

    D  in  ( rho_low(1),  rho_fix(1) ]

and its width in log-dilution is exactly the switching gain rho_fix/rho_low,
which is *dimensionless and independent of D*. So the whole question reduces to
one number per drug subset, and that number translates directly into a protocol:
"passage at 1:D, cycle these drugs, and you clear the culture; run either drug
alone at the same dilution and you do not."

Free parameters, and how they are handled
-----------------------------------------
`mu`     mutation rate per cycle. Not measured by this assay. Always swept.
`delta`  stage length x rate units. The published rates are "x 10^-3" in an
         unstated time unit, so r*delta is the only meaningful combination. We
         parameterise it by `max_log_growth` -- the log-growth of the fastest
         genotype/drug pair per cycle -- which is directly interpretable
         ("the best-growing culture does N doublings per passage") and is what
         an experimenter actually controls via incubation time. Always swept.
`D`      dilution. Not fixed at all: it is the axis of the prediction above.
"""
from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "mira2015"
CSV = DATA_DIR / "growth_rates.csv"

#: bit order of the genotype label string, most significant first
GENOTYPE_BITS = ("M69L", "E104K", "G238S", "N276D")

DRUG_NAMES = {
    "AMP": "ampicillin", "AM": "amoxicillin", "CEC": "cefaclor",
    "CTX": "cefotaxime", "ZOX": "ceftizoxime", "CXM": "cefuroxime",
    "CRO": "ceftriaxone", "AMC": "amoxicillin+clavulanate",
    "CAZ": "ceftazidime", "CTT": "cefotetan", "SAM": "ampicillin+sulbactam",
    "CPR": "cefprozil", "CPD": "cefpodoxime", "TZP": "piperacillin+tazobactam",
    "FEP": "cefepime",
}


# ======================================================================
def load_growth_rates(path=None):
    """-> (genotypes: list[str], drugs: list[str], R: (16,15) array).

    R[i, a] is the measured growth rate of genotype i under drug a, in the
    paper's units of 1e-3. Raises if the file is missing -- there is no
    synthetic fallback anywhere in this package by design.
    """
    p = Path(path) if path is not None else CSV
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found. This package ships no synthetic substitute. "
            f"See data/mira2015/PROVENANCE.md for the source and "
            f"`python -m sdsp.data.verify_mira` to re-download.")
    rows = [ln.split(",") for ln in p.read_text().strip().splitlines()]
    drugs = [c.strip() for c in rows[0][1:]]
    genotypes = [r[0].strip() for r in rows[1:]]
    R = np.array([[float(x) for x in r[1:]] for r in rows[1:]])
    _validate(genotypes, drugs, R)
    return genotypes, drugs, R


def _validate(genotypes, drugs, R):
    assert R.shape == (16, 15), f"expected 16x15, got {R.shape}"
    assert len(set(genotypes)) == 16, "genotype labels not unique"
    assert set(genotypes) == {"".join(b) for b in
                              itertools.product("01", repeat=4)}, \
        "genotype labels are not the 16 four-bit strings"
    assert np.isfinite(R).all(), "non-finite growth rate"
    assert (R >= 0).all() and (R <= 4).all(), "growth rates out of expected range"
    assert len(drugs) == len(set(drugs)) == 15


# ======================================================================
def hypercube_mutation_matrix(genotypes, mu=1e-3):
    """M[j,i] = probability a descendant of genotype i is genotype j.

    Single point mutations only: the four substitutions are independent loci, so
    the genotype space is the 4-cube and each genotype has exactly 4 Hamming-1
    neighbours. Column-stochastic by construction, so mutation alone neither
    creates nor destroys population -- all growth is in the diagonal term and
    all loss is in the dilution term, which keeps the two cleanly separated.
    """
    n = len(genotypes)
    idx = {g: i for i, g in enumerate(genotypes)}
    M = np.zeros((n, n))
    for g, i in idx.items():
        nbrs = []
        for b in range(len(g)):
            flipped = g[:b] + ("1" if g[b] == "0" else "0") + g[b + 1:]
            nbrs.append(idx[flipped])
        M[i, i] = 1.0 - mu * len(nbrs)
        for j in nbrs:
            M[j, i] += mu
    assert np.allclose(M.sum(axis=0), 1.0), "mutation matrix not column-stochastic"
    return M


def delta_for(R, max_log_growth=np.log(2.0)):
    """Stage length such that the fastest genotype/drug pair grows by
    exp(max_log_growth) per cycle. Default = one doubling."""
    return float(max_log_growth / R.max())


def drug_operators(R=None, genotypes=None, mu=1e-3, dilution=1.0,
                   max_log_growth=np.log(2.0), delta=None):
    """-> list of 16x16 per-cycle operators A_a, one per drug.

    With `dilution=1.0` (the default) the operators carry no dilution, which is
    the right normalisation for computing the *dilution-free* quantities
    rho_fix(1) and rho_low(1) that define the order-matters window.
    """
    if R is None:
        genotypes, _, R = load_growth_rates()
    delta = delta_for(R, max_log_growth) if delta is None else delta
    M = hypercube_mutation_matrix(genotypes, mu)
    return [(M @ np.diag(np.exp(R[:, a] * delta))) / dilution
            for a in range(R.shape[1])]


def resample_operators(subset=None, mu=1e-3, max_log_growth=np.log(2.0),
                       rel_noise=0.10, R=None, genotypes=None):
    """-> a callable(rng) -> operators, with the MEASURED RATES perturbed.

    This is the honest way to propagate measurement error. Perturbing the
    entries of A_a directly (the generic fallback in `switching.protocol_power`)
    also perturbs the mutation matrix, which was never measured here and carries
    no such error; and it breaks the column-stochastic structure that separates
    growth from loss. Here the log-normal noise is applied to R -- the thing
    Mira et al. actually measured -- and the operators are rebuilt from it.

    `delta` is held FIXED at its nominal value rather than recomputed from the
    perturbed R. Deliberate: delta encodes incubation time, which the
    experimenter sets on the bench and which does not move when the rate
    estimates move. Recomputing it would renormalise part of the noise away and
    flatter the result.

    `subset` selects drug indices; None means all 15.
    """
    if R is None:
        genotypes, _, R = load_growth_rates()
    delta = delta_for(R, max_log_growth)          # nominal, held fixed
    M = hypercube_mutation_matrix(genotypes, mu)  # not measured -> not perturbed
    cols = range(R.shape[1]) if subset is None else list(subset)

    def resample(rng):
        Rp = R * np.exp(rng.normal(0.0, rel_noise, size=R.shape))
        return [M @ np.diag(np.exp(Rp[:, a] * delta)) for a in cols]

    return resample


# ======================================================================
def sanctuary_operators(R=None, genotypes=None, mu=1e-3, dilution=1.0,
                        max_log_growth=np.log(2.0), penetration=0.2,
                        migration=0.05, asymmetry=0.0):
    """Two-compartment version: a treated well plus a low-drug sanctuary well.

    IMPORTANT: the compartment structure is an *imposed experimental design*,
    not something fitted to Mira's data (those are well-mixed liquid cultures).
    `penetration` is the fraction of the drug's effect that reaches the
    sanctuary, and it is realised on the bench by adding less drug to that well;
    `migration` and `asymmetry` are pipetting choices. Everything about the
    biology still comes from the measured growth rates.

    In the sanctuary the drug is attenuated, so the effective growth rate is
    interpolated back toward the *most permissive* drug for that genotype:
        r_eff = r_drug + (1 - penetration) * (r_max_over_drugs - r_drug)
    i.e. penetration = 1 reproduces full drug effect, penetration = 0 gives the
    genotype's drug-free-like growth. This uses only measured numbers.
    """
    if R is None:
        genotypes, _, R = load_growth_rates()
    delta = delta_for(R, max_log_growth)
    n = R.shape[0]
    M = hypercube_mutation_matrix(genotypes, mu)
    r_free = R.max(axis=1)                       # most permissive measured rate

    # inter-well transfer (column-stochastic), well0 = treated, well1 = sanctuary
    f = np.array([[1.0 - migration, migration * (1.0 - asymmetry)],
                  [migration, 1.0 - migration * (1.0 - asymmetry)]])

    ops = []
    for a in range(R.shape[1]):
        r_treated = R[:, a]
        r_sanct = r_treated + (1.0 - penetration) * (r_free - r_treated)
        G0 = M @ np.diag(np.exp(r_treated * delta))
        G1 = M @ np.diag(np.exp(r_sanct * delta))
        A = np.zeros((2 * n, 2 * n))
        A[:n, :n] = f[0, 0] * G0
        A[:n, n:] = f[0, 1] * G1
        A[n:, :n] = f[1, 0] * G0
        A[n:, n:] = f[1, 1] * G1
        ops.append(A / dilution)
    return ops


# ======================================================================
def order_matters_window(mats):
    """(D_low, D_high]: dilution factors at which ONLY switching cures.

    Because dilution enters as a scalar, this is exact:
        D_low  = rho_low(1)   (upper bound on it, i.e. the witnessed schedule)
        D_high = rho_fix(1)
    Any D in (D_low, D_high] is a dilution at which the witnessed cyclic
    schedule clears the culture and no single drug does. Empty if D_low >= D_high.
    """
    from ..switching import fixed_best, search_periodic_exhaustive
    rho_fix, jfix = fixed_best(mats)
    res = search_periodic_exhaustive(mats, k_max=min(6, 12 // max(len(mats) // 4, 1)))
    d_low = min(res.radius, rho_fix)
    return {
        "D_low": float(d_low), "D_high": float(rho_fix),
        "width_log10": float(np.log10(max(rho_fix / max(d_low, 1e-300), 1.0))),
        "witness": res.schedule, "period": res.length,
        "best_fixed_drug": jfix, "nonempty": bool(d_low < rho_fix - 1e-12),
    }


def rate_space_advantage(mats_fn=None, combo=None, mu=1e-3, k_max=8,
                         probe=(0.69, 2.77), R=None, genotypes=None):
    """The switching advantage as an invariant of the DATA, not of the design.

    Empirically (checked to 0.3% median over all 38 pairs with a non-trivial
    gain), the switching gain obeys

        log(rho_fix / rho_low)  =  delta * dr

    exactly, where delta is the stage length and `dr` depends only on the
    measured rates and the drug subset. The reason is structural: at small mu
    the operators are near-diagonal, products are near-diagonal, and both
    radii are of the form exp(delta * <some rate>), so the log-gain is linear
    in delta with the optimal cycle unchanged.

    The consequence is a design law rather than a limitation:

        window width in decades of dilution = delta * dr / ln(10)

    The pair fixes `dr`; the experimenter fixes `delta` by choosing incubation
    time. A window that looks uselessly narrow at one doubling per passage
    (0.05 decades) is 0.20 decades at four doublings and 0.41 at eight, and
    eight doublings per passage is an ordinary 1:100-style transfer.

    Returns dr estimated at two probe stage lengths plus their relative drift,
    which should be ~0 if the law holds for this subset.
    """
    from ..switching import fixed_best, search_periodic_exhaustive
    if R is None:
        genotypes, _, R = load_growth_rates()
    vals = []
    for mlg in probe:
        mats = drug_operators(R, genotypes, mu=mu, max_log_growth=mlg)
        sub = mats if combo is None else [mats[i] for i in combo]
        rf, _ = fixed_best(sub)
        lo = min(search_periodic_exhaustive(sub, k_max=k_max).radius, rf)
        vals.append(np.log(rf / max(lo, 1e-300)) / (mlg / R.max()))
    dr = float(vals[-1])
    drift = abs(vals[0] - vals[-1]) / max(dr, 1e-12)
    return {"dr": dr, "drift": float(drift), "probe": probe,
            "decades_per_delta": dr / np.log(10.0)}


def design_passage(combo, target_decades=0.2, mu=1e-3, k_max=8,
                   R=None, genotypes=None):
    """Turn a drug subset into a bench protocol: how long to incubate, how hard
    to dilute, and how far that sits below the saturation dilution.

    Two constraints the experimenter must respect, both of which fall out of the
    algebra rather than being imposed:

    1. `delta` (incubation) sets the window width: decades = delta*dr/ln(10).
       Solve it for the delta that gives `target_decades`.

    2. The window sits STRICTLY BELOW the saturation dilution D_sat = exp(mlg),
       the dilution at which a drug-free control exactly refills the culture.
       D_high/D_sat = exp(delta*(r_fix - R.max())) < 1 because the best drug's
       worst genotype always grows slower than the fastest genotype anywhere.
       So a protocol that times passages by "transfer when the untreated
       control saturates" and dilutes by that same factor lands ABOVE the
       window, where a single drug already cures and sequencing buys nothing.
       Incubation and dilution must be set independently.

    Note that the population is not returned to its starting density in the
    control arm: below the saturation dilution it accumulates. That is exactly
    the intended reading -- the fixed-drug arm fails. Linearity holds where the
    claim lives (the contracting arm stays at low density); in the failing arm
    the true dynamics saturate rather than grow without bound, which makes the
    predicted failure conservative, not optimistic.
    """
    from ..switching import fixed_best, search_periodic_exhaustive
    if R is None:
        genotypes, _, R = load_growth_rates()
    adv = rate_space_advantage(combo=combo, mu=mu, k_max=k_max,
                               R=R, genotypes=genotypes)
    if adv["dr"] <= 1e-9:
        return {"feasible": False, "reason": "no switching advantage", **adv}
    delta = target_decades * np.log(10.0) / adv["dr"]
    mlg = delta * R.max()
    mats = drug_operators(R, genotypes, mu=mu, max_log_growth=mlg)
    sub = [mats[i] for i in combo]
    rf, jf = fixed_best(sub)
    res = search_periodic_exhaustive(sub, k_max=k_max)
    lo = min(res.radius, rf)
    return {
        "feasible": True,
        "dr": adv["dr"],
        "doublings_per_passage": float(mlg / np.log(2.0)),
        "max_log_growth": float(mlg),
        "delta": float(delta),
        "D_low": float(lo), "D_high": float(rf),
        "D_recommended": float(np.sqrt(lo * rf)),
        "D_saturation": float(np.exp(mlg)),
        "D_high_over_D_sat": float(rf / np.exp(mlg)),
        "decades": float(np.log10(rf / lo)),
        "cycle": res.schedule, "period": res.length, "best_fixed_index": jf,
    }


def summary(mu=1e-3, max_log_growth=np.log(2.0)):
    genotypes, drugs, R = load_growth_rates()
    return (f"Mira et al. 2015: {len(genotypes)} TEM genotypes x {len(drugs)} "
            f"beta-lactams\n  bit order {GENOTYPE_BITS}\n"
            f"  growth rate range [{R.min():.3f}, {R.max():.3f}] (x1e-3)\n"
            f"  mu={mu:g}, max log-growth per cycle={max_log_growth:.3f}")
