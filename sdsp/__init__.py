"""SDSP — sequential drug scheduling as a switched-system control problem.

Central claim
-------------
Let A_a be the per-passage population operator under drug a. Then

    **drug order matters iff   rho_low({A_a}) < 1 <= min_a rho(A_a)**

i.e. iff the family is stabilizable by switching but not by any fixed drug.
Inside that band, sequencing is the difference between clearing a population and
not clearing it -- a qualitative phase boundary, not a quantitative improvement.

Layout
------
    switching.py    the core: certified bounds on the lower spectral radius,
                    periodic-schedule search, SDP (Lyapunov) certificates
    data/           real measured datasets only; no synthetic fallbacks
    instance.py     graph / formulary / action-space definitions
    dynamics.py     rollout, branching + Gaussian noise, PK memory
    fastsim.py      vectorised batch simulation
    spectral.py     rho, Perron vectors, spectral gap, non-normality, Kreiss
    microbial.py    serial-passage experimental designs and power analysis
    policies/       exact.py (alpha-vector DP), milp.py (Gurobi),
                    metaheuristics.py (SA/VNS), baselines.py,
                    rl.py (MLP + GAT + PPO), imitation.py (BC + DAgger)
    experiments/    exp_order_matters.py is the main one

Validated
---------
The alpha-vector DP, brute force and both Gurobi MILP formulations agree to
1e-6 on every instance tested. Everything else is written but unrun -- see
README.md, which tracks this honestly.
"""
__version__ = "0.2.0"
