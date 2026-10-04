<h1 align="center">Spectral Dominance in Sequential Control</h1>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10+-blue?logo=python&logoColor=white" alt="python">
  <img src="https://img.shields.io/badge/numpy-1.26+-013243?logo=numpy&logoColor=white" alt="numpy">
  <img src="https://img.shields.io/badge/data-Mira%202015-green" alt="data">
  <img src="https://img.shields.io/badge/licence-MIT-lightgrey" alt="licence">
</p>

---

This treats sequential scheduling as what it actually is: a switched linear
system. Each  is a matrix acting on the population once per passage, a
treatment plan is a product of those matrices, and the question of whether the
order of s matters becomes a question about spectral radii.

The other half of the project is honesty about what can be computed. The
quantity that decides the question is not computable in general, so nothing here
claims to compute it — everything reported is a certified two-sided bound, and
an answer is allowed to come back undetermined.

---

## The criterion

Write `A_a` for the per-passage operator under  `a` — growth, then mutation,
then dilution. Two numbers settle everything:

```
rho_fix  = min_a rho(A_a)                            best a single fixed  can do
rho_low  = inf_k min rho(A_a1 ... A_ak)^(1/k)        best any schedule can do
```

Order matters if and only if `rho_low < 1 <= rho_fix`. Below that band a single
 already clears the population and sequencing is a refinement; above it
nothing clears; inside it, sequencing is the difference between cure and
failure. That is the switched-systems statement of collateral sensitivity — two
s that each fail on their own composing into a product that contracts.

Because dilution enters `A_a` as a scalar `1/D`, the band has a closed form: only
switching cures for `D` in `(rho_low(1), rho_fix(1)]`, a window whose width in
log-dilution is `log10(rho_fix / rho_low)`. That is a bench protocol, not a
metaphor — passage at 1:D, cycle the s, and either  alone at the same
dilution should fail.

## What gets computed

| bound | how |
|---|---|
| upper on `rho_low` | an explicit periodic schedule with `rho(prod)^(1/k) < 1`, a witness whose contraction factor is just an eigenvalue |
| lower on `rho_low` | `min_a abs(det A_a)^(1/n)`, exact and free; plus an SDP certificate when a common `P > 0` exists |
| worst case | a common-quadratic SOS bound on the joint spectral radius — under 1 means every schedule cures, including a non-adherent one |

Four verdicts come out: `ORDER MATTERS (proved)`, `NO SWITCHING NEEDED`,
`HOPELESS (proved)`, `UNDETERMINED`. Undetermined stays undetermined.

Searching for a short contractive product is combinatorial in the number of
s, which is where the learned searcher earns its place. It is benchmarked
against exhaustive search, beam search and simulated annealing at an equal
evaluation budget.

## Data

`data/mira2015/` holds 16 TEM β-lactamase genotypes measured against 15
β-lactams (Mira et al. 2015, *PLoS ONE* 10:e0122283, Table 4, CC-BY) — the same
landscapes used by Nichol et al. 2015 and Weaver et al. 2024, so numbers here are
directly comparable to theirs.

There is no synthetic data and no fallbacks anywhere: every loader raises rather
than fabricating. The table was extracted from PMC HTML rather than a
machine-readable supplement, so run the verifier before trusting any derived
number. Free parameters — mutation rate, log-growth per cycle — are always swept.
Dilution is not a parameter at all; it is the axis of the prediction.

## Example

```bash
pip install -r requirements.txt

python -m sdsp.data.verify_mira

python -m sdsp.experiments.exp_order_matters --sizes 2 3 --out results/
python -m sdsp.experiments.exp_order_matters --sanctuary --penetration 0.2
```

That writes certified bounds and switching gains per  subset, robustness to
measurement error in the growth rates, a sweep over the free parameters, and the
top candidates written out as bench protocols.

## Layout

```
sdsp/
  switching.py     the criterion and its certificates
  spectral.py      spectral gap, non-normality, Kreiss constant, controllability radius
  dynamics.py      growth, mutation, dilution
  microbial.py     population model on measured landscapes
  fastsim.py       vectorised rollouts
  instance.py      problem instances
  policies/        exact DP, MILP, metaheuristics, imitation, RL
  data/            Mira loader and verifier
  experiments/     the order-matters study
```

## Limitations

Real growth saturates, and the linear model is defensible only at low density —
which happens to be the regime where cure and relapse are decided, but it is a
restriction and not a detail. Mira's assay is one concentration per , so a
 is binary here; a real dose axis needs seascape data. The compartments in
the sanctuary variant are an experimental design, not something fitted to
well-mixed cultures. And collateral sensitivity is often not reproducible across
replicate lineages (Barbosa et al. 2018) — that is the strongest caveat in the
field and it belongs here rather than in a footnote.

---

Guidance by Dr. Connor Jerzak (UT Austin).
