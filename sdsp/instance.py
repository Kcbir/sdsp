"""Problem instance: the within-host graph, the drug formulary, the action space.

An instance is the tuple (V, W, D, c, s0, T, gamma) of the paper, plus the
toxicity budget B and combination cap m that define the per-stage action set

    A^D = { (x,r) : x in {0,1}^d, ||x||_1 <= m, sum_i x_i f_i(r_i) <= B }.

Doses are discretised into L levels so that A^D becomes a finite set; every
solver in this package (DP, MILP, SA/VNS, RL) then acts on the same finite
action set, which is what makes their values directly comparable.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Drug:
    """A single agent in the formulary.

    name    : label
    mask    : {0,1}^N spectrum mask (which (compartment,strain) nodes it reaches)
    kappa   : R^N_+ baseline per-stage kill fraction, zero off the mask
    doses   : the L admissible dose levels (r), ascending, r=0 excluded
    phi     : PK-PD efficacy phi(r) in [0,1], non-decreasing, phi(0)=0
    tox     : toxicity f(r) >= 0
    half_life_h : PK half-life in hours (used by the PK-memory extension)
    """

    name: str
    mask: np.ndarray
    kappa: np.ndarray
    doses: tuple[float, ...] = (1.0,)
    kind: str = "hill"       # "hill" | "step"
    emax: float = 1.0        # phi(r) = emax * r^h / (ec50^h + r^h)   (Hill)
    ec50: float = 0.5
    hill: float = 2.0
    tox_slope: float = 1.0   # f(r) = tox_slope * r
    half_life_h: float = 24.0

    def phi(self, r: float) -> float:
        if r <= 0.0:
            return 0.0
        if self.kind == "step":          # phi(r) = emax * 1{r > 0}
            return float(self.emax)
        rh = r ** self.hill
        return float(self.emax * rh / (self.ec50 ** self.hill + rh))

    def tox(self, r: float) -> float:
        return float(self.tox_slope * r) if r > 0 else 0.0

    def kill(self, r: float) -> np.ndarray:
        return self.phi(r) * self.kappa


@dataclass
class Instance:
    """A full problem instance."""

    W: np.ndarray                     # (N,N) non-negative transition matrix
    drugs: list[Drug]
    c: np.ndarray                     # (N,) clinical importance weights
    s0: np.ndarray                    # (N,) initial burden
    T: int = 6                        # horizon (number of stages)
    gamma: float = 0.9                # discount
    B: float = 1.0                    # toxicity budget per stage
    m: int = 1                        # combination cap
    lam_tox: float = 0.0              # lambda_tox in eq. (5)
    lam_res: float = 0.0              # lambda_res in eq. (5)
    resistant_nodes: np.ndarray | None = None   # for Res(.)
    node_names: list[str] = field(default_factory=list)
    compartment_of: np.ndarray | None = None    # (N,) compartment index per node
    strain_of: np.ndarray | None = None         # (N,) strain index per node
    observable: np.ndarray | None = None        # (N,) bool, for the POMDP variant
    stage_hours: float = 48.0
    allow_holiday: bool = True   # include the empty action (drug holiday)
    kill_model: str = "bliss"    # bliss | additive_capped | additive_clipped
    name: str = "instance"

    # ---- derived, cached ----
    _actions: list[tuple] | None = field(default=None, repr=False, compare=False)
    _kill: np.ndarray | None = field(default=None, repr=False, compare=False)
    _tox: np.ndarray | None = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        self.W = np.asarray(self.W, dtype=float)
        self.c = np.asarray(self.c, dtype=float).reshape(-1)
        self.s0 = np.asarray(self.s0, dtype=float).reshape(-1)
        N = self.W.shape[0]
        assert self.W.shape == (N, N), "W must be square"
        assert (self.W >= 0).all(), "W must be non-negative"
        assert self.c.shape == (N,) and self.s0.shape == (N,)
        for d in self.drugs:
            assert d.mask.shape == (N,) and d.kappa.shape == (N,)
            assert np.all(d.kappa[d.mask == 0] == 0), f"{d.name}: kappa nonzero off mask"
        if self.observable is None:
            self.observable = np.ones(N, dtype=bool)
        if self.resistant_nodes is None:
            self.resistant_nodes = np.zeros(N, dtype=bool)
        if not self.node_names:
            self.node_names = [f"v{i}" for i in range(N)]

    # ------------------------------------------------------------------
    @property
    def N(self) -> int:
        return self.W.shape[0]

    @property
    def d(self) -> int:
        return len(self.drugs)

    # ------------------------------------------------------------------
    def _build_actions(self):
        """Enumerate the feasible finite action set.

        An action is a tuple of (drug_index, dose_level_index) pairs, of size
        <= m, that respects the toxicity budget. The empty tuple (drug holiday)
        is always included -- it is a real clinical option and it is exactly
        what adaptive therapy exploits.
        """
        acts: list[tuple] = [()] if self.allow_holiday else []
        for k in range(1, self.m + 1):
            for combo in itertools.combinations(range(self.d), k):
                lvl_ranges = [range(len(self.drugs[i].doses)) for i in combo]
                for lvls in itertools.product(*lvl_ranges):
                    a = tuple(sorted(zip(combo, lvls)))
                    tox = sum(self.drugs[i].tox(self.drugs[i].doses[l]) for i, l in a)
                    if tox <= self.B + 1e-12:
                        acts.append(a)
        # Pre-compute effective kill vectors and toxicities.
        #
        # kill_model:
        #   "bliss"            survival = prod_i (1 - phi_i kappa_i)   [default]
        #                      Pharmacologically standard (Bliss independence),
        #                      automatically in [0,1]. Eq. (4) of the paper sums
        #                      kill rates, which can exceed 1 for combinations
        #                      and then needs an ad-hoc clip.
        #   "additive_capped"  eq. (4) additive, but combinations that would push
        #                      kappa^eff above 1 anywhere are declared infeasible.
        #                      Keeps the structured MILP exact.
        #   "additive_clipped" eq. (4) verbatim, clipped at 1.
        keep, K, Tx = [], [], []
        for j, a in enumerate(acts):
            raw = np.zeros(self.N)
            surv = np.ones(self.N)
            for i, l in a:
                ki = self.drugs[i].kill(self.drugs[i].doses[l])
                raw += ki
                surv *= (1.0 - np.clip(ki, 0.0, 1.0))
            if self.kill_model == "bliss":
                k = 1.0 - surv
            elif self.kill_model == "additive_capped":
                if raw.max() > 1.0 + 1e-12:
                    continue                       # infeasible combination
                k = raw
            elif self.kill_model == "additive_clipped":
                k = np.clip(raw, 0.0, 1.0)
            else:
                raise ValueError(f"unknown kill_model {self.kill_model!r}")
            keep.append(a)
            K.append(k)
            Tx.append(sum(self.drugs[i].tox(self.drugs[i].doses[l]) for i, l in a))
        self._actions = keep
        self._kill = np.array(K).reshape(len(keep), self.N)
        self._tox = np.array(Tx)

    @property
    def actions(self) -> list[tuple]:
        if self._actions is None:
            self._build_actions()
        return self._actions

    @property
    def kill(self) -> np.ndarray:
        """(A,N) effective kill fraction kappa^eff for each action."""
        if self._kill is None:
            self._build_actions()
        return self._kill

    @property
    def tox(self) -> np.ndarray:
        """(A,) per-stage toxicity of each action."""
        if self._tox is None:
            self._build_actions()
        return self._tox

    @property
    def n_actions(self) -> int:
        return len(self.actions)

    def action_label(self, j: int) -> str:
        a = self.actions[j]
        if not a:
            return "holiday"
        return "+".join(f"{self.drugs[i].name}@{self.drugs[i].doses[l]:g}" for i, l in a)

    # ------------------------------------------------------------------
    def W_eff(self, j: int) -> np.ndarray:
        """Post-drug transition operator W diag(1 - kappa^eff(a_j))."""
        return self.W * (1.0 - self.kill[j])[None, :]

    def all_W_eff(self) -> np.ndarray:
        """(A,N,N) stack of post-drug operators."""
        return self.W[None, :, :] * (1.0 - self.kill)[:, None, :]

    def res_penalty(self, s: np.ndarray) -> float:
        """Res(.) of eq. (5): burden carried on resistant nodes.

        Left deliberately simple and swappable -- see `sdsp.spectral.evolvability`
        for the diversity-based alternative discussed in the write-up.
        """
        return float(s[self.resistant_nodes].sum())

    def summary(self) -> str:
        from .spectral import spectral_report
        r = spectral_report(self)
        return (
            f"{self.name}: N={self.N} d={self.d} |A|={self.n_actions} T={self.T} "
            f"gamma={self.gamma}\n"
            f"  rho(W)={r['rho']:.4f}  gamma*rho={r['gamma_rho']:.4f} -> {r['regime']}\n"
            f"  spectral gap |l2/l1|={r['gap_ratio']:.4f}  non-normality "
            f"(Henrici)={r['henrici']:.4f}  Kreiss>={r['kreiss_lb']:.3f}\n"
            f"  controllability radius rho*={r['rho_star']:.4f} "
            f"(gamma*rho*={r['gamma_rho_star']:.4f} -> {r['regime_star']}) "
            f"via {r['rho_star_action']}"
        )
