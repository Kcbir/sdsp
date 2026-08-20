"""Instance generators: the paper's construction, random families, disease modules.

IMPORTANT (honesty note carried over from the review): the disease modules below
are *illustrative parameterisations* assembled from published order-of-magnitude
values. They are not fitted to patient-level data. Any claim of the form
"disease X is supercritical" made from them should be reported as a literature-
based placement, not an estimate. `sdsp.calibrate` does real fitting when given
real trajectories.
"""
from __future__ import annotations

import numpy as np

from .instance import Drug, Instance


# ----------------------------------------------------------------------
# Construction 3.2 of the paper
# ----------------------------------------------------------------------
def construction_two_compartment(rho=3.0, alpha=0.2, beta=0.5, T=6, gamma=0.9,
                                 allow_holiday=False, s0=(1.0, 1.0)) -> Instance:
    """The two-compartment family W = [[rho-a, 1], [a*b, rho-b]].

    Node 0 = peripheral blood P (observable, the only one in the cost c).
    Node 1 = deep reservoir D (invisible, feeds P).

    Eigenvalues are rho and rho-alpha-beta, so rho(W) = rho whenever
    alpha + beta > 0; as alpha,beta -> 0 the matrix becomes the *defective*
    Jordan block [[rho,1],[0,rho]], which is where the paper's gap bound is
    extremal. That is the non-normality point.
    """
    assert alpha + beta < rho - 1 + 1e-12, "constraint alpha+beta < rho-1 violated"
    W = np.array([[rho - alpha, 1.0], [alpha * beta, rho - beta]])
    mk = lambda v: np.array(v, dtype=float)
    drugs = [
        Drug("A_blood", mask=mk([1, 0]), kappa=mk([1, 0]), kind="step", tox_slope=1.0),
        Drug("B_reservoir", mask=mk([0, 1]), kappa=mk([0, 1]), kind="step", tox_slope=1.0),
    ]
    return Instance(
        W=W, drugs=drugs, c=mk([1, 0]), s0=mk(s0), T=T, gamma=gamma,
        B=1.0, m=1, node_names=["P_blood", "D_reservoir"],
        compartment_of=np.array([0, 1]), strain_of=np.array([0, 0]),
        observable=np.array([True, False]), allow_holiday=allow_holiday,
        name=f"construction(rho={rho},a={alpha},b={beta})",
    )


# ----------------------------------------------------------------------
# Structured random family with controllable spectral geometry
# ----------------------------------------------------------------------
def random_instance(n_comp=3, n_strain=2, n_drugs=3, rho_target=2.0, gamma=0.9,
                    T=6, asymmetry=0.8, mu=1e-3, coverage=0.6, seed=0,
                    n_doses=2, m=1, B=1.0, sanctuary=True) -> Instance:
    """Random (compartment x strain) instance with a tunable spectral geometry.

    `asymmetry` in [0,1] controls how one-directional the inter-compartment
    fluxes are. Asymmetric flux plus (almost always) forward-only mutation makes
    W structurally non-normal -- which the review argues is the true driver of
    the myopic/optimal gap, and which this knob lets you test directly.

    `sanctuary=True` forces at least one compartment that no drug covers well,
    reproducing the granuloma/latent-reservoir situation.
    """
    rng = np.random.default_rng(seed)
    C, K = n_comp, n_strain
    N = C * K
    idx = lambda ci, ki: ci * K + ki
    comp_of = np.repeat(np.arange(C), K)
    strain_of = np.tile(np.arange(K), C)

    W = np.zeros((N, N))
    # intra-compartment replication, decreasing with resistance (fitness cost)
    for ci in range(C):
        base = rng.uniform(0.8, 1.4)
        for ki in range(K):
            fit_cost = 1.0 - 0.12 * ki
            W[idx(ci, ki), idx(ci, ki)] = base * fit_cost
    # inter-compartment flux, within strain
    for ci in range(C):
        for cj in range(C):
            if ci == cj:
                continue
            f = rng.uniform(0.02, 0.35)
            # asymmetry: forward edges keep full weight, backward edges shrink
            if cj < ci:
                f *= (1.0 - asymmetry)
            for ki in range(K):
                W[idx(ci, ki), idx(cj, ki)] = f
    # mutation, within compartment, forward-biased (upper-triangular in strain)
    for ci in range(C):
        for ki in range(K):
            for kj in range(K):
                if ki == kj:
                    continue
                fwd = ki > kj
                W[idx(ci, ki), idx(ci, kj)] += mu * (1.0 if fwd else 1e-2)

    # rescale to hit the requested spectral radius exactly
    cur = float(np.max(np.abs(np.linalg.eigvals(W))))
    W *= rho_target / max(cur, 1e-12)

    # drug menu: each drug covers a random set of (compartment, strain) nodes
    sanctuary_comp = C - 1 if sanctuary else -1
    drugs = []
    for i in range(n_drugs):
        mask = np.zeros(N)
        for ci in range(C):
            if ci == sanctuary_comp and rng.random() < 0.85:
                continue                      # sanctuary: usually unreachable
            for ki in range(K):
                # resistant strains are progressively harder to cover
                p = coverage * (0.75 ** ki)
                if rng.random() < p:
                    mask[idx(ci, ki)] = 1.0
        if mask.sum() == 0:
            mask[rng.integers(N)] = 1.0
        kappa = mask * rng.uniform(0.55, 0.95, size=N)
        doses = tuple(np.linspace(0.5, 1.0, n_doses))
        drugs.append(Drug(f"D{i}", mask=mask, kappa=kappa, doses=doses,
                          ec50=rng.uniform(0.35, 0.6), hill=2.0,
                          tox_slope=rng.uniform(0.6, 1.0),
                          half_life_h=float(rng.choice([6, 24, 72, 336]))))

    c = np.ones(N)
    s0 = rng.uniform(0.5, 1.5, size=N)
    obs = comp_of == 0                        # only compartment 0 is measurable
    return Instance(W=W, drugs=drugs, c=c, s0=s0, T=T, gamma=gamma, B=B, m=m,
                    resistant_nodes=(strain_of > 0),
                    node_names=[f"c{comp_of[v]}k{strain_of[v]}" for v in range(N)],
                    compartment_of=comp_of, strain_of=strain_of, observable=obs,
                    name=f"random(seed={seed},rho={rho_target},asym={asymmetry})")


def random_family(n=40, rho_range=(0.6, 3.0), gamma=0.9, seed=0, **kw):
    """A batch of random instances spanning the sub/critical/supercritical range."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        r = float(rng.uniform(*rho_range))
        out.append(random_instance(rho_target=r, gamma=gamma, seed=seed * 1000 + i, **kw))
    return out


# ----------------------------------------------------------------------
# Disease modules (illustrative parameterisations -- see module docstring)
# ----------------------------------------------------------------------
def _module(name, comps, strains, W, drugs, c, s0, T, gamma, obs_comp=0,
            stage_hours=48.0, m=2, B=2.0):
    C, K = len(comps), len(strains)
    comp_of = np.repeat(np.arange(C), K)
    strain_of = np.tile(np.arange(K), C)
    names = [f"{comps[comp_of[v]]}|{strains[strain_of[v]]}" for v in range(C * K)]
    return Instance(W=np.asarray(W, float), drugs=drugs, c=np.asarray(c, float),
                    s0=np.asarray(s0, float), T=T, gamma=gamma, B=B, m=m,
                    resistant_nodes=(strain_of > 0), node_names=names,
                    compartment_of=comp_of, strain_of=strain_of,
                    observable=(comp_of == obs_comp), stage_hours=stage_hours,
                    name=name)


def _block_W(C, K, repl, flux, mu, fitness_cost=0.10):
    """Assemble a (compartment x strain) transition matrix from block parts."""
    N = C * K
    idx = lambda ci, ki: ci * K + ki
    W = np.zeros((N, N))
    for ci in range(C):
        for ki in range(K):
            W[idx(ci, ki), idx(ci, ki)] = repl[ci] * (1.0 - fitness_cost * ki)
    for (ci, cj), f in flux.items():
        for ki in range(K):
            W[idx(ci, ki), idx(cj, ki)] = f
    for ci in range(C):
        for ki in range(1, K):
            W[idx(ci, ki), idx(ci, ki - 1)] += mu           # forward mutation
            W[idx(ci, ki - 1), idx(ci, ki)] += mu * 1e-2    # rare reversion
    return W


def malaria(gamma=0.9, T=6, kelch13=False) -> Instance:
    """P. falciparum: hepatocyte / peripheral RBC / sequestered RBC / gametocyte
    x {sensitive, partial-R, full-R}. Stage = 48 h erythrocytic cycle.
    Replication numbers of order 8-16 per cycle follow Saralamba et al. (2011)
    and White et al. (2014) in magnitude only.
    """
    comps = ["hepatocyte", "peripheral_RBC", "sequestered_RBC", "gametocyte"]
    strains = ["sensitive", "partial_R", "full_R"]
    C, K = 4, 3
    repl = [1.02, 9.0 if not kelch13 else 11.0, 7.5, 0.35]
    flux = {(1, 0): 0.30, (2, 1): 1.10, (1, 2): 0.55, (3, 1): 0.02, (1, 3): 0.001}
    W = _block_W(C, K, repl, flux, mu=1e-4, fitness_cost=0.08)
    N = C * K
    m_all = np.ones(N)
    art_mask = np.zeros(N)          # artemisinin: strong in blood, weak in sequestered
    art_kappa = np.zeros(N)
    for ci, kap in [(1, 0.97), (2, 0.88), (3, 0.60)]:
        for ki in range(K):
            art_mask[ci * K + ki] = 1
            # Kelch13 = ring-stage survival: it degrades artemisinin kill most
            # in the sequestered compartment, which is where rings hide.
            pen = 1.0
            if kelch13 and ki > 0:
                pen = 0.30 if ci == 2 else 0.45
            art_kappa[ci * K + ki] = kap * pen
    lum_mask, lum_kappa = np.zeros(N), np.zeros(N)
    for ci, kap in [(1, 0.80), (2, 0.55)]:
        for ki in range(K):
            lum_mask[ci * K + ki] = 1
            lum_kappa[ci * K + ki] = kap * (0.6 if ki == 2 else 1.0)
    pq_mask, pq_kappa = np.zeros(N), np.zeros(N)    # primaquine: liver + gametocyte
    for ci, kap in [(0, 0.90), (3, 0.85)]:
        for ki in range(K):
            pq_mask[ci * K + ki] = 1
            pq_kappa[ci * K + ki] = kap
    drugs = [
        Drug("artemisinin", art_mask, art_kappa, doses=(0.5, 1.0), ec50=0.35,
             tox_slope=0.5, half_life_h=1.0),
        Drug("lumefantrine", lum_mask, lum_kappa, doses=(0.5, 1.0), ec50=0.45,
             tox_slope=0.7, half_life_h=96.0),
        Drug("primaquine", pq_mask, pq_kappa, doses=(0.5, 1.0), ec50=0.5,
             tox_slope=0.9, half_life_h=6.0),
    ]
    c = np.ones(N)
    c[1 * K:(1 + 1) * K] = 1.0        # peripheral: symptomatic
    c[2 * K:(2 + 1) * K] = 3.0        # sequestered: drives severe disease
    s0 = np.zeros(N)
    s0[1 * K] = 1.0                   # start: sensitive parasites in blood
    s0[2 * K] = 0.4
    s0[0 * K] = 0.05
    return _module(f"malaria{'_kelch13' if kelch13 else ''}", comps, strains, W,
                   drugs, c, s0, T, gamma, obs_comp=1, stage_hours=48.0, m=2, B=1.6)


def hiv(gamma=0.95, T=8) -> Instance:
    """HIV: blood / GALT / CNS / lymph node / latent CD4 x 4 genotypes.
    The latent compartment has replication ~1.0 and near-zero drug kill, so
    rho(W) = max over blocks stays at ~1 -> *critical*, not subcritical.
    (This is the internal inconsistency flagged in the section 7 table.)
    """
    comps = ["blood", "GALT", "CNS", "lymph", "latent_CD4"]
    strains = ["WT", "NRTI_R", "NNRTI_R", "multi_R"]
    C, K = 5, 4
    repl = [1.6, 1.8, 1.3, 1.7, 1.0]        # latent: pure persistence
    flux = {(1, 0): 0.25, (0, 1): 0.20, (2, 0): 0.05, (0, 2): 0.02,
            (3, 0): 0.30, (0, 3): 0.25, (4, 0): 0.01, (0, 4): 0.004}
    W = _block_W(C, K, repl, flux, mu=3e-3, fitness_cost=0.12)
    N = C * K

    def mk(cov: dict, name, tox, hl, ec50=0.4):
        mask, kap = np.zeros(N), np.zeros(N)
        for ci, (base, res_prof) in cov.items():
            for ki in range(K):
                mask[ci * K + ki] = 1
                kap[ci * K + ki] = base * res_prof[ki]
        return Drug(name, mask, kap, doses=(0.5, 1.0), ec50=ec50,
                    tox_slope=tox, half_life_h=hl)

    full, nrti_res, nnrti_res = [1, 1, 1, 1], [1, 0.15, 1, 0.1], [1, 1, 0.1, 0.1]
    drugs = [
        mk({0: (0.9, nrti_res), 1: (0.8, nrti_res), 3: (0.8, nrti_res),
            2: (0.3, nrti_res), 4: (0.02, full)}, "NRTI", 0.5, 12.0),
        mk({0: (0.9, nnrti_res), 1: (0.7, nnrti_res), 3: (0.75, nnrti_res),
            2: (0.5, nnrti_res), 4: (0.02, full)}, "NNRTI", 0.6, 40.0),
        mk({0: (0.92, full), 1: (0.85, full), 3: (0.85, full),
            2: (0.15, full), 4: (0.02, full)}, "INSTI", 0.4, 14.0),
        mk({0: (0.6, full), 1: (0.5, full), 2: (0.05, full), 3: (0.5, full),
            4: (0.25, full)}, "LRA_latency_reversal", 1.0, 8.0),
    ]
    c = np.ones(N)
    c[4 * K:] = 0.2                    # latent cells cause no immediate symptoms
    s0 = np.zeros(N)
    s0[0] = 1.0
    s0[1 * K] = 0.8
    s0[3 * K] = 0.6
    s0[4 * K] = 0.3
    return _module("hiv", comps, strains, W, drugs, c, s0, T, gamma,
                   obs_comp=0, stage_hours=24.0 * 30, m=3, B=2.2)


def tuberculosis(gamma=0.95, T=8, rifampicin=True) -> Instance:
    """TB: cavity / granuloma-caseum / intracellular x {DS, INH-R, MDR}.
    Toggling rifampicin off removes the only well-penetrating agent, which is
    the mechanism the paper's TB paragraph claims.
    """
    comps = ["cavity", "granuloma_caseum", "intracellular"]
    strains = ["DS", "INH_R", "MDR"]
    C, K = 3, 3
    repl = [1.35, 1.22, 1.15]
    flux = {(1, 0): 0.10, (0, 1): 0.05, (2, 0): 0.12, (0, 2): 0.08, (1, 2): 0.04}
    W = _block_W(C, K, repl, flux, mu=2e-4, fitness_cost=0.15)
    N = C * K

    def mk(cov, name, tox, hl, res_prof):
        mask, kap = np.zeros(N), np.zeros(N)
        for ci, base in cov.items():
            for ki in range(K):
                mask[ci * K + ki] = 1
                kap[ci * K + ki] = base * res_prof[ki]
        return Drug(name, mask, kap, doses=(0.5, 1.0), ec50=0.45,
                    tox_slope=tox, half_life_h=hl)

    drugs = [mk({0: 0.85, 2: 0.75}, "isoniazid", 0.5, 3.0, [1, 0.05, 0.05])]
    if rifampicin:
        # the one agent that both penetrates caseum and kills non-replicators
        drugs.append(mk({0: 0.80, 1: 0.60, 2: 0.70}, "rifampicin", 0.6, 3.0,
                        [1, 1, 0.05]))
    drugs += [
        # PZA is caseum-active but PZA resistance is common in MDR strains
        mk({1: 0.42, 2: 0.30}, "pyrazinamide", 0.7, 10.0, [1, 0.9, 0.35]),
        mk({0: 0.55, 2: 0.35}, "ethambutol", 0.5, 3.0, [1, 1, 0.8]),
    ]
    c = np.ones(N)
    c[0 * K:K] = 2.0                   # cavitary disease drives transmission
    s0 = np.zeros(N)
    s0[0] = 1.0
    s0[1 * K] = 0.7
    s0[2 * K] = 0.5
    if not rifampicin:                 # MDR scenario: seed the resistant strain
        s0[0 * K + 2] = 0.6
        s0[1 * K + 2] = 0.5
    return _module(f"tb{'_RIPE' if rifampicin else '_noRIF'}", comps, strains, W,
                   drugs, c, s0, T, gamma, obs_comp=0, stage_hours=24.0 * 14,
                   m=3, B=2.0)


def melanoma(gamma=0.9, T=8) -> Instance:
    """Melanoma: skin lesion / lymph node / visceral micrometastasis
    x {BRAF-mut sensitive, BRAFi-R, MEKi-R, phenotype-switched invasive}.

    The fourth 'strain' is a *reversible* state, so unlike the other modules the
    strain graph has a substantial backward edge: this is phenotypic plasticity,
    and it is exactly what makes drug holidays valuable here.
    """
    comps = ["skin_lesion", "lymph_node", "visceral_micromet"]
    strains = ["BRAFmut_S", "BRAFi_R", "MEKi_R", "switched_invasive"]
    C, K = 3, 4
    repl = [1.30, 1.20, 1.15]
    flux = {(1, 0): 0.18, (0, 1): 0.06, (2, 1): 0.14, (1, 2): 0.03, (2, 0): 0.05}
    W = _block_W(C, K, repl, flux, mu=5e-3, fitness_cost=0.08)
    # phenotypic switching S <-> invasive is fast and reversible (not a mutation)
    idx = lambda ci, ki: ci * K + ki
    for ci in range(C):
        W[idx(ci, 3), idx(ci, 0)] += 0.08     # switch out under stress
        W[idx(ci, 0), idx(ci, 3)] += 0.05     # switch back when drug is withdrawn
    N = C * K

    def mk(cov, name, tox, hl, res_prof):
        mask, kap = np.zeros(N), np.zeros(N)
        for ci, base in cov.items():
            for ki in range(K):
                mask[ci * K + ki] = 1
                kap[ci * K + ki] = base * res_prof[ki]
        return Drug(name, mask, kap, doses=(0.4, 0.7, 1.0), ec50=0.45,
                    tox_slope=tox, half_life_h=hl)

    drugs = [
        mk({0: 0.90, 1: 0.75, 2: 0.55}, "BRAFi_vemurafenib", 0.6, 50.0,
           [1, 0.05, 0.9, 0.15]),
        mk({0: 0.85, 1: 0.70, 2: 0.50}, "MEKi_trametinib", 0.7, 100.0,
           [1, 0.8, 0.05, 0.2]),
        mk({0: 0.45, 1: 0.60, 2: 0.55}, "anti_PD1", 0.4, 500.0,
           [0.8, 0.8, 0.8, 0.9]),
    ]
    c = np.ones(N)
    c[2 * K:] = 2.5                    # visceral disease is what kills
    s0 = np.zeros(N)
    s0[0] = 1.0
    s0[1 * K] = 0.4
    s0[2 * K] = 0.1
    return _module("melanoma", comps, strains, W, drugs, c, s0, T, gamma,
                   obs_comp=0, stage_hours=24.0 * 7, m=2, B=1.6)


DISEASE_MODULES = {
    "malaria": malaria,
    "malaria_kelch13": lambda **kw: malaria(kelch13=True, **kw),
    "hiv": hiv,
    "tb_ripe": tuberculosis,
    "tb_norif": lambda **kw: tuberculosis(rifampicin=False, **kw),
    "melanoma": melanoma,
}
