"""Structured microbial populations under serial passage: the *validation* system.

Why this module exists
----------------------
The disease modules in `sdsp.instances` are extrapolation targets. They cannot
validate anything, because their load-bearing compartments -- granuloma, latent
reservoir, sequestered RBC pool -- are exactly the ones you cannot measure in a
patient, and a full N x N transition matrix is not identifiable from one blood
draw. Any claim of the form "we fitted W to clinical data" is not supportable.

A structured microbial serial-passage experiment inverts that completely:

    W is not estimated. It is imposed by the experimenter, one pipetting step
    at a time.

Per transfer cycle t:
    1. dose        add drug i at concentration r; well v receives an effective
                   concentration p[i,v] * r, where p < 1 makes a *sanctuary well*
                   (the physical analogue of a granuloma -- you build it by
                   adding less drug, or by adding a drug-degrading condition).
    2. grow        incubate for `cycle_hours`; well u multiplies by g[u].
    3. transfer    move a defined fraction f[v,u] of well u into well v and
                   dilute `dilution`-fold.

which is exactly    s_{t+1} = W diag(1 - kappa(a)) s_t   with

    W[v,u] = f[v,u] * g[u] / dilution.

If f is column-stochastic and growth is uniform, rho(W) = g / dilution: the
spectral radius is the ratio of growth per cycle to dilution per cycle, and both
are dials on the bench. Note the corollary -- standard 1:100 daily passage of a
culture that grows ~100-fold per day sits at rho = 1 *exactly*, i.e. every
serial-passage lab is already running at the critical point.

Readout: per-well, per-strain abundance by flow cytometry (distinct fluorescent
markers) or amplicon barcode sequencing. Unlike the clinical setting, the full
state vector s_t is observable, so the POMDP caveat of section 4.3 does not
apply and the theory can be tested as stated.

Suggested organism/drug systems (all standard, all cheap):
    E. coli MG1655 + beta-lactam / aminoglycoside / quinolone, resistance by
    known plasmid or chromosomal marker, strains distinguished by GFP/mCherry;
    or a defined collateral-sensitivity pair from the published networks
    (Imamovic & Sommer 2013; Nichol et al. 2015; Maltas & Wood 2019).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .instance import Drug, Instance
from .spectral import henrici, rho, spectral_gap


# ======================================================================
# Pharmacodynamics from MICs (the parameters a microbiologist actually has)
# ======================================================================
def hill_kill(conc, mic, emax=0.99, hill=2.0):
    """Fraction killed per cycle at concentration `conc` given MIC.

    Standard sigmoid PD: kill = emax * (c/MIC)^H / (1 + (c/MIC)^H), so kill =
    emax/2 at c = MIC. MICs are the one drug parameter that is measured
    routinely and reported in absolute units, which is what makes this module
    parameterisable from a real bench protocol rather than from guesses.
    """
    x = np.maximum(np.asarray(conc, float), 0.0) / np.maximum(mic, 1e-12)
    xh = x ** hill
    return emax * xh / (1.0 + xh)


# ======================================================================
# Instance construction from wet-lab knobs
# ======================================================================
@dataclass
class PassageDesign:
    """Everything the experimenter sets. Every field is a bench parameter."""

    n_wells: int = 2
    n_strains: int = 2
    doubling_hours: float = 0.5          # E. coli in rich medium ~30 min
    cycle_hours: float = 12.0            # time between transfers
    dilution: float = 100.0              # 1:100 passage
    migration: float = 0.05              # fraction of a well transferred sideways
    asymmetry: float = 0.0               # 0 = symmetric exchange, 1 = one-way only
    carrying_capacity: float = 1e9       # cells/mL; caps growth per cycle
    inoculum: float = 1e6                # cells/mL at t=0
    fitness_cost: float = 0.05           # per resistance step, per cycle
    mutation_rate: float = 1e-8          # per cell per cycle, S -> R
    penetration: np.ndarray | None = None    # (n_drugs, n_wells) in [0,1]
    mic: np.ndarray | None = None            # (n_drugs, n_strains) absolute units
    concentrations: tuple[float, ...] = (1.0, 4.0)   # dose levels, in xMIC of WT
    emax: float = 0.99
    hill: float = 2.0
    T: int = 20
    gamma: float = 1.0                   # in vitro: total burden, no discounting
    drug_names: tuple[str, ...] = ("drugA", "drugB")
    combination_cap: int = 1
    tox_budget: float = 1e9              # no toxicity limit in vitro
    sanctuary_penetration: float = 0.1   # drug level in the sanctuary well

    def growth_per_cycle(self) -> float:
        """Fold increase per cycle if unconstrained: 2^(cycle/doubling)."""
        return float(2.0 ** (self.cycle_hours / max(self.doubling_hours, 1e-9)))


def transfer_matrix(n_wells, migration, asymmetry=0.0, dilution=100.0):
    """Column-stochastic transfer fractions f[v,u], then divided by dilution.

    `asymmetry` interpolates between symmetric exchange (0) and strictly
    one-way flow along the well index (1). At asymmetry = 1 with a linear well
    chain the matrix becomes triangular -- and a triangular matrix with equal
    diagonal entries is *defective*, the Jordan structure that makes the
    myopic-optimal gap extremal. This single scalar is therefore the knob for
    the non-normality experiment: hold rho fixed, sweep asymmetry, and see
    whether the gap tracks rho (paper's thesis) or non-normality (the stronger
    one).
    """
    f = np.zeros((n_wells, n_wells))
    for u in range(n_wells):
        out_fwd = migration
        out_bwd = migration * (1.0 - asymmetry)
        stay = 1.0 - 0.0
        tot_out = 0.0
        if u + 1 < n_wells:
            f[u + 1, u] = out_fwd
            tot_out += out_fwd
        if u - 1 >= 0:
            f[u - 1, u] = out_bwd
            tot_out += out_bwd
        f[u, u] = max(stay - tot_out, 0.0)
    return f / dilution


def serial_passage_instance(design: PassageDesign, name="serial_passage") -> Instance:
    """Build a game instance from a bench protocol.

    Node ordering is (well, strain), matching the rest of the package.
    """
    D = design
    Wn, K = D.n_wells, D.n_strains
    N = Wn * K
    idx = lambda w, k: w * K + k

    # ---- growth per cycle, with resistance fitness cost ----
    g0 = D.growth_per_cycle()
    g = np.array([g0 * (1.0 - D.fitness_cost) ** k for k in range(K)])

    # ---- transfer ----
    f = transfer_matrix(Wn, D.migration, D.asymmetry, D.dilution)

    # ---- assemble W[v,u] = f[well_v, well_u] * g[strain_u] * 1{same strain} ----
    Wm = np.zeros((N, N))
    for wu in range(Wn):
        for wv in range(Wn):
            if f[wv, wu] == 0.0:
                continue
            for k in range(K):
                Wm[idx(wv, k), idx(wu, k)] = f[wv, wu] * g[k]
    # mutation, within well, forward-biased
    for w in range(Wn):
        for k in range(1, K):
            Wm[idx(w, k), idx(w, k - 1)] += D.mutation_rate * f[w, w] * g[k - 1]
            Wm[idx(w, k - 1), idx(w, k)] += D.mutation_rate * 1e-2 * f[w, w] * g[k]

    # ---- drugs ----
    n_drugs = len(D.drug_names)
    pen = D.penetration
    if pen is None:
        pen = np.ones((n_drugs, Wn))
        pen[:, -1] = D.sanctuary_penetration        # last well is the sanctuary
    mic = D.mic
    if mic is None:
        # drug A: WT sensitive, strain 1 resistant. drug B: the reverse
        # (a collateral-sensitivity pair -- the case where order provably matters)
        mic = np.ones((n_drugs, K))
        for i in range(n_drugs):
            for k in range(K):
                mic[i, k] = 32.0 if (k % n_drugs) == i else 1.0

    drugs = []
    for i in range(n_drugs):
        mask = np.zeros(N)
        kappa = np.zeros(N)
        for w in range(Wn):
            for k in range(K):
                kill_at_ref = hill_kill(pen[i, w] * 1.0, mic[i, k], D.emax, D.hill)
                if kill_at_ref > 1e-6:
                    mask[idx(w, k)] = 1.0
                kappa[idx(w, k)] = kill_at_ref
        # dose levels are multiples of the WT MIC; phi rescales the Hill curve
        drugs.append(Drug(D.drug_names[i], mask=mask, kappa=kappa,
                          doses=D.concentrations, kind="hill",
                          ec50=1.0, hill=D.hill, emax=1.0,
                          tox_slope=0.0, half_life_h=D.cycle_hours))

    s0 = np.zeros(N)
    for w in range(Wn):
        s0[idx(w, 0)] = D.inoculum
    c = np.ones(N)
    strain_of = np.tile(np.arange(K), Wn)
    well_of = np.repeat(np.arange(Wn), K)
    names = [f"well{well_of[v]}|strain{strain_of[v]}" for v in range(N)]

    return Instance(
        W=Wm, drugs=drugs, c=c, s0=s0, T=D.T, gamma=D.gamma,
        B=D.tox_budget, m=D.combination_cap, lam_tox=0.0,
        resistant_nodes=(strain_of > 0), node_names=names,
        compartment_of=well_of, strain_of=strain_of,
        observable=np.ones(N, dtype=bool),       # fully observable on the bench
        stage_hours=D.cycle_hours, allow_holiday=True, name=name,
    )


# ======================================================================
# Inverse design: set the regime you want
# ======================================================================
def dilution_for_rho(design: PassageDesign, target_rho: float,
                     tol=1e-10, max_iter=200) -> float:
    """The dilution factor that puts the *untreated* system at a target rho(W).

    rho(W) is exactly inversely proportional to the dilution factor (dilution
    enters W as a scalar 1/D), so this is closed form -- but it is computed by
    construction here so it stays correct if the assembly changes.

    Use it to pre-register a regime: "we set D = 63 so that rho = 1.6, i.e.
    supercritical, where the theory predicts an exponentially growing gap."
    """
    probe = PassageDesign(**{**design.__dict__, "dilution": 1.0})
    r1 = rho(serial_passage_instance(probe).W)          # rho at dilution = 1
    if r1 <= 0:
        raise ValueError("degenerate design: zero growth")
    return float(r1 / target_rho)


def design_for_regime(design: PassageDesign, target_rho: float) -> PassageDesign:
    """Copy of `design` retuned to hit `target_rho` via the dilution factor."""
    d = PassageDesign(**design.__dict__)
    d.dilution = dilution_for_rho(design, target_rho)
    return d


def matched_rho_family(base: PassageDesign, target_rho=1.6,
                       asymmetries=(0.0, 0.25, 0.5, 0.75, 1.0)):
    """Instances with *identical* rho(W) but increasing non-normality.

    This is the decisive experiment. The paper claims the myopic-optimal gap is
    governed by rho(W). If that is the whole story, every member of this family
    behaves identically. If the gap instead grows with asymmetry at fixed rho,
    then non-normality is the driver and the theory needs restating -- which is
    both more interesting and, on the two-compartment construction, what the
    Jordan-block limit already suggests.

    Returns list of (asymmetry, instance, diagnostics).
    """
    out = []
    for a in asymmetries:
        d = PassageDesign(**{**base.__dict__, "asymmetry": float(a)})
        d.dilution = dilution_for_rho(d, target_rho)
        inst = serial_passage_instance(d, name=f"passage_asym{a:.2f}")
        out.append((float(a), inst, {
            "rho": rho(inst.W),
            "henrici": henrici(inst.W),
            "gap_ratio": spectral_gap(inst.W),
            "dilution": d.dilution,
        }))
    return out


# ======================================================================
# Power analysis
# ======================================================================
def power_analysis(inst, policy_a, policy_b, n_reps=6, n_sim=400, seed=0,
                   noise=None, alpha=0.05, readout="total", log_readout=True,
                   measurement_cv=0.10):
    """Empirical power to detect the predicted arm difference, by simulation.

    Rather than assuming an effect size, this simulates the actual experiment:
    `n_reps` replicate cultures per arm under the demographic-noise branching
    model plus log-normal measurement error, then runs a two-sample Welch t-test
    on the log final burden, repeated `n_sim` times.

    Returns the empirical power, the median log10 effect, and the implied
    minimum replicates -- i.e. the numbers a pre-registration needs.

    `measurement_cv` is the coefficient of variation of the plate-reader/flow
    readout (10% is typical); it is applied on top of the biological variance
    the branching model already generates.
    """
    from .dynamics import NoiseConfig, rollout
    noise = noise or NoiseConfig(kind="branching")
    rng = np.random.default_rng(seed)

    def arm(policy, r):
        _, traj, _ = rollout(inst, policy, rng=np.random.default_rng(r),
                             cfg=noise, return_traj=True)
        val = traj[-1].sum() if readout == "total" else float(inst.c @ traj[-1])
        val = max(val, 0.5)                              # floor at detection limit
        val *= float(np.exp(rng.normal(0, measurement_cv)))
        return np.log10(val) if log_readout else val

    from math import sqrt
    hits, effects = 0, []
    for sim in range(n_sim):
        A = np.array([arm(policy_a, seed + 100000 * sim + i) for i in range(n_reps)])
        B = np.array([arm(policy_b, seed + 200000 * sim + i) for i in range(n_reps)])
        effects.append(float(np.median(A) - np.median(B)))
        va, vb = A.var(ddof=1), B.var(ddof=1)
        se = sqrt(va / n_reps + vb / n_reps)
        if se <= 0:
            hits += int(abs(A.mean() - B.mean()) > 0)
            continue
        t = abs(A.mean() - B.mean()) / se
        df = (va / n_reps + vb / n_reps) ** 2 / max(
            (va / n_reps) ** 2 / (n_reps - 1) + (vb / n_reps) ** 2 / (n_reps - 1), 1e-300)
        try:
            from scipy import stats
            crit = stats.t.ppf(1 - alpha / 2, df)
        except Exception:
            crit = 2.2                                    # ~n=6 two-sided 0.05
        hits += int(t > crit)
    return {
        "power": hits / n_sim,
        "median_log10_effect": float(np.median(effects)),
        "n_reps": n_reps,
        "n_sim": n_sim,
        "readout": readout,
    }


def min_replicates(inst, policy_a, policy_b, target_power=0.9, max_reps=24, **kw):
    """Smallest replicate count reaching `target_power`. Returns (n, power)."""
    for n in range(3, max_reps + 1):
        r = power_analysis(inst, policy_a, policy_b, n_reps=n, **kw)
        if r["power"] >= target_power:
            return n, r["power"]
    return max_reps, r["power"]


# ======================================================================
# Protocol report
# ======================================================================
def protocol_summary(design: PassageDesign, inst=None) -> str:
    """Human-readable bench protocol plus the spectral regime it realises."""
    inst = inst or serial_passage_instance(design)
    r = rho(inst.W)
    g = design.growth_per_cycle()
    lines = [
        f"SERIAL-PASSAGE PROTOCOL  ({design.n_wells} wells x {design.n_strains} strains "
        f"= {inst.N} nodes, {design.T} transfers)",
        f"  incubate            {design.cycle_hours:g} h per cycle "
        f"(doubling {design.doubling_hours:g} h -> {g:.3g}-fold growth)",
        f"  passage             1:{design.dilution:.3g}",
        f"  sideways transfer   {design.migration:.3g} of each well "
        f"(asymmetry {design.asymmetry:.2f})",
        f"  sanctuary well      #{design.n_wells - 1}, drug at "
        f"{design.sanctuary_penetration:.0%} of nominal",
        f"  inoculum            {design.inoculum:.2g} cells/mL",
        f"  drugs               {', '.join(design.drug_names)} at "
        f"{design.concentrations} x MIC(WT)",
        "",
        f"  REALISED rho(W)     {r:.4f}   (growth/dilution = {g / design.dilution:.4f})",
        f"  gamma*rho           {design.gamma * r:.4f}  -> "
        f"{'SUPERCRITICAL' if design.gamma * r > 1.05 else 'SUBCRITICAL' if design.gamma * r < 0.95 else 'CRITICAL'}",
        f"  non-normality       Henrici {henrici(inst.W):.4f}, "
        f"spectral gap |l2/l1| {spectral_gap(inst.W):.4f}",
        f"  total wall-clock    {design.T * design.cycle_hours / 24:.1f} days",
    ]
    return "\n".join(lines)


# ======================================================================
# Standard pre-registered designs
# ======================================================================
def regime_sweep_designs(base: PassageDesign = None,
                         rhos=(0.6, 0.8, 0.95, 1.05, 1.3, 1.8, 2.5)):
    """One design per target rho, spanning the predicted phase transition.

    This is the primary experiment: the theory says the myopic-optimal gap is
    bounded below 1 for gamma*rho < 1, linear at 1, and exponential above. A
    dilution series is the cheapest possible way to sweep that axis, because
    dilution is the only thing that changes between arms.
    """
    base = base or PassageDesign()
    out = []
    for r in rhos:
        d = PassageDesign(**base.__dict__)
        d.dilution = dilution_for_rho(base, r)
        out.append((r, d, serial_passage_instance(d, name=f"passage_rho{r:g}")))
    return out
