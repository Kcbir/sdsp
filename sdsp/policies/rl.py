"""Reinforcement learning policies: MLP and Graph Attention (GAT), trained by PPO.

Architecture note (this is the part that matters for the paper)
---------------------------------------------------------------
A naive policy network maps a fixed-length state vector to logits over a fixed
number of actions. That policy is welded to one instance: it cannot transfer
between diseases, because N, d and |A| all change. So instead both networks here
*score action embeddings*:

    node encoder      : node features  -> H in R^{N x h}        (MLP or GAT)
    graph embedding   : pool(H)        -> g in R^{h}
    action embedding  : e_a = MLP([ action_features(a) , sum_v kappa_a[v] H_v ])
    logit(a)          = MLP([g, e_a])

Because an action is represented by *what it kills* rather than by an index,
one trained network runs on any instance with any formulary and any graph size.
That is what makes the "train on random instances, test on held-out disease
modules" experiment possible, and it is the honest way to claim the learned
policy has learned something about the structure rather than about a lookup
table.

The only difference between `MLPEncoder` and `GATEncoder` is whether node
representations exchange information along the edges of W. Everything else --
features, action head, value head, PPO hyperparameters -- is identical, so the
comparison is a clean ablation of *graph structure*, not of capacity.

The GAT layer is implemented from scratch (Velickovic et al. 2018) with edge
weights as edge features; torch_geometric is deliberately not a dependency, so
this runs anywhere torch runs.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAVE_TORCH = True
except Exception:                                        # pragma: no cover
    HAVE_TORCH = False
    torch = None
    nn = object

from ..fastsim import get_sim
from ..spectral import perron_vectors, rho


# ======================================================================
# Feature extraction
# ======================================================================
N_STATIC_NODE_FEATS = 11
N_DYNAMIC_NODE_FEATS = 6
N_NODE_FEATS = N_STATIC_NODE_FEATS + N_DYNAMIC_NODE_FEATS
N_ACTION_FEATS = 9


class InstanceGraph:
    """All instance-level tensors the networks need, computed once.

    Everything is normalised so that features are O(1) across instances whose
    raw scales differ by orders of magnitude -- otherwise a network trained on
    malaria (rho ~ 9) sees nothing but saturated units on TB (rho ~ 1.4).
    """

    def __init__(self, inst, device="cpu"):
        self.inst = inst
        self.device = device
        W, N = inst.W, inst.N
        r = max(rho(W), 1e-12)
        u, v, _ = perron_vectors(W)
        u = np.abs(u) / max(np.abs(u).sum(), 1e-12)
        v = np.abs(v) / max(np.abs(v).sum(), 1e-12)
        self.perron_left = v

        kap = inst.kill                                   # (A,N)
        cov_frac = (kap > 1e-9).mean(axis=0)              # fraction of actions hitting v
        cmax = max(inst.c.max(), 1e-12)

        self.x_static = np.stack([
            inst.c / cmax,
            inst.observable.astype(float),
            inst.resistant_nodes.astype(float),
            np.diag(W) / r,
            W.sum(axis=1) / r,                            # in-strength  (row sums)
            W.sum(axis=0) / r,                            # out-strength (col sums)
            u, v,
            kap.max(axis=0),
            kap.mean(axis=0),
            cov_frac,
        ], axis=1).astype(np.float32)                     # (N, 11)

        src, dst = np.nonzero(W.T)                        # edge u -> v when W[v,u] > 0
        self.edge_index = np.stack([src, dst]).astype(np.int64)      # (2,E)
        w = W.T[src, dst]
        self.edge_attr = np.stack([w / r, np.log1p(w)], axis=1).astype(np.float32)

        # ---- static action features ----
        Weff = inst.all_W_eff()
        rho_eff = np.array([rho(Weff[a]) for a in range(inst.n_actions)])
        self.rho_eff = rho_eff
        nsel = np.array([len(a) for a in inst.actions], dtype=float)
        self.a_static = np.stack([
            inst.tox / max(inst.B, 1e-12),
            nsel / max(inst.m, 1),
            kap.mean(axis=1),
            kap.max(axis=1),
            rho_eff / r,                                  # the paper's diagnostic
            (inst.gamma * rho_eff < 1.0).astype(float),   # "this action is stabilising"
            kap @ v,                                      # reproductive-value coverage
            kap @ (inst.c / cmax),                        # clinical-weight coverage
            np.log1p(rho_eff),
        ], axis=1).astype(np.float32)                     # (A, 9)

        self.kappa = kap.astype(np.float32)               # (A,N)
        self.N, self.A = N, inst.n_actions
        self.sim = get_sim(inst)

    # ------------------------------------------------------------------
    def dynamic_node_feats(self, S: np.ndarray, t: int, T: int) -> np.ndarray:
        """(M,N) states -> (M,N,6) dynamic features, scale-free by construction."""
        S = np.atleast_2d(S)
        tot = S.sum(axis=1, keepdims=True) + 1e-12
        z = np.log1p(np.maximum(S, 0.0))
        zm = z.mean(axis=1, keepdims=True)
        zs = z.std(axis=1, keepdims=True) + 1e-6
        M, N = S.shape
        return np.stack([
            (z - zm) / zs,
            S / tot,
            np.repeat(np.log1p(tot), N, axis=1) / 10.0,
            np.repeat(np.full((M, 1), t / max(T, 1)), N, axis=1),
            np.repeat(np.full((M, 1), (T - t) / max(T, 1)), N, axis=1),
            np.repeat(np.log1p(S.sum(axis=1, keepdims=True) /
                               max(self.inst.s0.sum(), 1e-12)), N, axis=1) / 5.0,
        ], axis=2).astype(np.float32)                     # (M,N,6)

    def dynamic_action_feats(self, S: np.ndarray) -> np.ndarray:
        """(M,N) states -> (M,A,2): how much of the current burden / current
        reproductive value each action removes. These are the only genuinely
        state-dependent action features, and they are what the paper's spectral
        greedy is missing."""
        S = np.atleast_2d(S)
        tot = S.sum(axis=1, keepdims=True) + 1e-12
        burden_cov = (S / tot) @ self.kappa.T                       # (M,A)
        rv = self.perron_left[None, :] * S
        rv = rv / (rv.sum(axis=1, keepdims=True) + 1e-12)
        rv_cov = rv @ self.kappa.T
        return np.stack([burden_cov, rv_cov], axis=2).astype(np.float32)


# ======================================================================
# Networks
# ======================================================================
if HAVE_TORCH:

    def _mlp(sizes, act=nn.SiLU, out_act=None):
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            if i < len(sizes) - 2:
                layers.append(act())
        if out_act is not None:
            layers.append(out_act())
        return nn.Sequential(*layers)

    def _segment_softmax(logits, index, n_seg):
        """Softmax over entries sharing the same `index` (edge softmax by dst)."""
        m = torch.full((n_seg,) + logits.shape[1:], float("-inf"),
                       device=logits.device, dtype=logits.dtype)
        m = m.scatter_reduce(0, index.view(-1, *([1] * (logits.dim() - 1)))
                             .expand_as(logits), logits, reduce="amax",
                             include_self=True)
        m = torch.nan_to_num(m, neginf=0.0)
        e = torch.exp(logits - m.index_select(0, index))
        s = torch.zeros_like(m).index_add_(0, index, e)
        return e / (s.index_select(0, index) + 1e-16)

    def _segment_sum(src, index, n_seg):
        out = torch.zeros((n_seg,) + src.shape[1:], device=src.device, dtype=src.dtype)
        return out.index_add_(0, index, src)

    def _segment_mean(src, index, n_seg):
        s = _segment_sum(src, index, n_seg)
        cnt = torch.zeros(n_seg, device=src.device, dtype=src.dtype).index_add_(
            0, index, torch.ones(index.shape[0], device=src.device, dtype=src.dtype))
        return s / cnt.clamp(min=1).view(-1, *([1] * (src.dim() - 1)))

    def _segment_max(src, index, n_seg):
        out = torch.full((n_seg,) + src.shape[1:], float("-inf"),
                         device=src.device, dtype=src.dtype)
        out = out.scatter_reduce(0, index.view(-1, *([1] * (src.dim() - 1)))
                                 .expand_as(src), src, reduce="amax", include_self=True)
        return torch.nan_to_num(out, neginf=0.0)

    class GATLayer(nn.Module):
        """Multi-head graph attention with edge features (Velickovic et al. 2018).

        Edge features carry the transition weight W_vu, which is the whole point:
        attention that ignores edge weight cannot distinguish a sanctuary
        compartment from a well-mixed one.
        """

        def __init__(self, in_dim, out_dim, heads=4, edge_dim=2, dropout=0.0,
                     concat=True):
            super().__init__()
            self.heads, self.out_dim, self.concat = heads, out_dim, concat
            self.lin = nn.Linear(in_dim, heads * out_dim, bias=False)
            self.att_src = nn.Parameter(torch.empty(heads, out_dim))
            self.att_dst = nn.Parameter(torch.empty(heads, out_dim))
            self.lin_edge = nn.Linear(edge_dim, heads, bias=False)
            self.bias = nn.Parameter(torch.zeros(heads * out_dim if concat else out_dim))
            self.dropout = dropout
            nn.init.xavier_uniform_(self.lin.weight)
            nn.init.xavier_uniform_(self.att_src)
            nn.init.xavier_uniform_(self.att_dst)
            nn.init.xavier_uniform_(self.lin_edge.weight)

        def forward(self, x, edge_index, edge_attr):
            Nn = x.size(0)
            src, dst = edge_index[0], edge_index[1]
            h = self.lin(x).view(Nn, self.heads, self.out_dim)
            a = ((h * self.att_src).sum(-1)[src]
                 + (h * self.att_dst).sum(-1)[dst]
                 + self.lin_edge(edge_attr))                       # (E,heads)
            a = F.leaky_relu(a, 0.2)
            alpha = _segment_softmax(a, dst, Nn)                   # (E,heads)
            if self.dropout > 0 and self.training:
                alpha = F.dropout(alpha, p=self.dropout)
            msg = h[src] * alpha.unsqueeze(-1)                     # (E,heads,out)
            out = _segment_sum(msg, dst, Nn)                       # (N,heads,out)
            out = out.reshape(Nn, -1) if self.concat else out.mean(1)
            return out + self.bias

    class MLPEncoder(nn.Module):
        """Node encoder with no message passing. The ablation control."""

        def __init__(self, in_dim, hidden=64, layers=3, **kw):
            super().__init__()
            self.net = _mlp([in_dim] + [hidden] * layers, out_act=nn.SiLU)
            self.out_dim = hidden

        def forward(self, x, edge_index, edge_attr):
            return self.net(x)

    class GATEncoder(nn.Module):
        """Node encoder that passes messages along the within-host graph."""

        def __init__(self, in_dim, hidden=64, layers=3, heads=4, edge_dim=2,
                     dropout=0.0):
            super().__init__()
            assert hidden % heads == 0, "hidden must be divisible by heads"
            self.inp = nn.Linear(in_dim, hidden)
            self.gats = nn.ModuleList([
                GATLayer(hidden, hidden // heads, heads=heads, edge_dim=edge_dim,
                         dropout=dropout) for _ in range(layers)])
            self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])
            self.out_dim = hidden

        def forward(self, x, edge_index, edge_attr):
            h = F.silu(self.inp(x))
            for gat, norm in zip(self.gats, self.norms):
                h = norm(h + F.silu(gat(h, edge_index, edge_attr)))   # residual
            return h

    class ActorCritic(nn.Module):
        """Shared encoder, action-scoring actor head, graph-level critic head."""

        def __init__(self, encoder="gat", hidden=64, layers=3, heads=4,
                     node_in=N_NODE_FEATS, act_in=N_ACTION_FEATS + 2, dropout=0.0):
            super().__init__()
            Enc = {"gat": GATEncoder, "mlp": MLPEncoder}[encoder]
            self.encoder_kind = encoder
            self.enc = Enc(node_in, hidden=hidden, layers=layers, heads=heads,
                           dropout=dropout)
            h = self.enc.out_dim
            self.graph_proj = _mlp([2 * h, h, h], out_act=nn.SiLU)
            self.act_proj = _mlp([act_in + h, h, h], out_act=nn.SiLU)
            self.score = _mlp([2 * h, h, 1])
            self.value = _mlp([h, h, 1])

        def forward(self, batch):
            """`batch` is the dict produced by `collate`."""
            H = self.enc(batch["x"], batch["edge_index"], batch["edge_attr"])
            n2g, a2g = batch["node2graph"], batch["act2graph"]
            G = int(batch["n_graphs"])
            g = self.graph_proj(torch.cat(
                [_segment_mean(H, n2g, G), _segment_max(H, n2g, G)], dim=-1))

            # kappa-weighted pooling of node embeddings for each action
            kap = batch["kappa"]                       # (nnz_a, ) sparse-ish dense
            # kappa is stored dense per action row over that graph's nodes:
            #   act_node_index maps each (action,node) pair to a global node id
            pooled = _segment_sum(
                H.index_select(0, batch["act_node_index"]) * kap.unsqueeze(-1),
                batch["act_node_seg"], batch["n_actions_total"])
            denom = _segment_sum(kap, batch["act_node_seg"],
                                 batch["n_actions_total"]).clamp(min=1e-6)
            pooled = pooled / denom.unsqueeze(-1)

            e = self.act_proj(torch.cat([batch["a_feat"], pooled], dim=-1))
            logits = self.score(torch.cat([g.index_select(0, a2g), e], dim=-1)).squeeze(-1)
            values = self.value(g).squeeze(-1)
            return logits, values, a2g, G

        # --------------------------------------------------------------
        @torch.no_grad()
        def act(self, batch, greedy=False):
            logits, values, a2g, G = self(batch)
            acts, logps, ents = [], [], []
            for gi in range(G):
                sel = (a2g == gi).nonzero(as_tuple=True)[0]
                lg = logits.index_select(0, sel)
                dist = torch.distributions.Categorical(logits=lg)
                a = torch.argmax(lg) if greedy else dist.sample()
                acts.append(a)
                logps.append(dist.log_prob(a))
                ents.append(dist.entropy())
            return (torch.stack(acts), torch.stack(logps), torch.stack(ents), values)

        def evaluate_actions(self, batch, actions):
            logits, values, a2g, G = self(batch)
            logps, ents = [], []
            for gi in range(G):
                sel = (a2g == gi).nonzero(as_tuple=True)[0]
                dist = torch.distributions.Categorical(logits=logits.index_select(0, sel))
                logps.append(dist.log_prob(actions[gi]))
                ents.append(dist.entropy())
            return torch.stack(logps), torch.stack(ents), values

        def logits_per_graph(self, batch):
            """List of per-graph logit vectors (used by behaviour cloning)."""
            logits, _, a2g, G = self(batch)
            return [logits[(a2g == gi).nonzero(as_tuple=True)[0]] for gi in range(G)]


# ======================================================================
# Batching
# ======================================================================
def collate(graphs, states, ts, device="cpu"):
    """Build one batched graph from a list of (InstanceGraph, state, t).

    Graphs are concatenated block-diagonally: node ids are offset per graph so
    the disjoint union is a single graph, which is why the segment softmax over
    edges works without any per-graph masking.
    """
    if not HAVE_TORCH:
        raise RuntimeError("torch not available")
    xs, eis, eas, n2g, afs, a2g = [], [], [], [], [], []
    an_idx, an_seg, kaps = [], [], []
    n_off, a_off = 0, 0
    for gi, (G, s, t) in enumerate(zip(graphs, states, ts)):
        dyn = G.dynamic_node_feats(np.asarray(s)[None, :], t, G.inst.T)[0]
        xs.append(np.concatenate([G.x_static, dyn], axis=1))
        eis.append(G.edge_index + n_off)
        eas.append(G.edge_attr)
        n2g.append(np.full(G.N, gi, dtype=np.int64))
        dyn_a = G.dynamic_action_feats(np.asarray(s)[None, :])[0]     # (A,2)
        afs.append(np.concatenate([G.a_static, dyn_a], axis=1))
        a2g.append(np.full(G.A, gi, dtype=np.int64))
        # dense (action,node) pairs for kappa-weighted pooling
        aa, vv = np.meshgrid(np.arange(G.A), np.arange(G.N), indexing="ij")
        an_idx.append(vv.reshape(-1) + n_off)
        an_seg.append(aa.reshape(-1) + a_off)
        kaps.append(G.kappa.reshape(-1))
        n_off += G.N
        a_off += G.A

    T_ = lambda a, dt: torch.as_tensor(np.concatenate(a), dtype=dt, device=device)
    return {
        "x": T_(xs, torch.float32),
        "edge_index": torch.as_tensor(np.concatenate(eis, axis=1),
                                      dtype=torch.long, device=device),
        "edge_attr": T_(eas, torch.float32),
        "node2graph": T_(n2g, torch.long),
        "a_feat": T_(afs, torch.float32),
        "act2graph": T_(a2g, torch.long),
        "act_node_index": T_(an_idx, torch.long),
        "act_node_seg": T_(an_seg, torch.long),
        "kappa": T_(kaps, torch.float32),
        "n_graphs": len(graphs),
        "n_actions_total": a_off,
    }


# ======================================================================
# Environment
# ======================================================================
class Env:
    """Single-instance episodic environment, vectorised over parallel episodes.

    Rewards are *normalised per instance* by the myopic policy's total cost. In
    the supercritical regime raw costs grow like (gamma*rho)^T and span many
    orders of magnitude, which destroys any shared value function; dividing by
    the myopic baseline puts every instance on the same scale and makes the
    learned quantity directly interpretable -- a return of 1.0 means "matched
    the clinical default", above 1.0 means "beat it".
    """

    def __init__(self, inst, graph=None, reward_scale="myopic", randomize_s0=0.0,
                 noise=None, seed=0):
        from ..dynamics import rollout
        from .baselines import myopic_policy
        self.inst = inst
        self.graph = graph or InstanceGraph(inst)
        self.sim = get_sim(inst)
        self.rng = np.random.default_rng(seed)
        self.randomize_s0 = randomize_s0
        self.noise = noise
        self.T = inst.T
        if reward_scale == "myopic":
            self.ref = abs(rollout(inst, myopic_policy(inst))) + 1e-12
        elif reward_scale == "s0":
            self.ref = abs(float(inst.c @ inst.s0)) + 1e-12
        else:
            self.ref = 1.0
        self.reward_scale = reward_scale

    def reset(self, n_env=1):
        s0 = np.broadcast_to(self.inst.s0, (n_env, self.inst.N)).copy()
        if self.randomize_s0 > 0:
            s0 *= np.exp(self.rng.normal(0, self.randomize_s0, size=s0.shape))
        self.S, self.t = s0, 0
        return self.S.copy()

    def step(self, actions):
        """actions: (n_env,) ints -> (next_states, rewards, done)."""
        a = np.asarray(actions, dtype=np.int64)
        if self.noise is None or self.noise.kind == "none":
            self.S = np.einsum("mij,mj->mi", self.sim.Weff[a], self.S, optimize=True)
        else:
            from ..dynamics import apply_drug, propagate
            out = np.empty_like(self.S)
            for i in range(len(a)):
                sh = apply_drug(self.inst, self.S[i], int(a[i]), self.rng, self.noise)
                out[i] = propagate(self.inst, sh, self.rng, self.noise)
            self.S = out
        cost = self.S @ self.sim.c_eff + self.sim.tox[a]
        r = -(self.inst.gamma ** self.t) * cost / self.ref
        self.t += 1
        return self.S.copy(), r, self.t >= self.T


# ======================================================================
# PPO
# ======================================================================
@dataclass
class PPOConfig:
    iters: int = 300
    n_env: int = 16                 # parallel episodes per instance per iteration
    epochs: int = 4
    minibatch: int = 256
    lr: float = 3e-4
    gamma: float = 1.0              # discounting already lives in the reward
    lam: float = 0.95
    clip: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    ent_final: float = 0.001
    max_grad_norm: float = 0.5
    device: str = "cpu"
    seed: int = 0
    log_every: int = 10
    instances_per_iter: int = 4     # sampled from the training pool each iteration
    hidden: int = 64
    layers: int = 3
    heads: int = 4
    encoder: str = "gat"
    reward_scale: str = "myopic"
    randomize_s0: float = 0.0


def _explained_variance(y_pred, y_true):
    var = np.var(y_true)
    return float("nan") if var == 0 else float(1 - np.var(y_true - y_pred) / var)


class PPOTrainer:
    """PPO over a *pool* of instances.

    Sampling a fresh subset of instances each iteration is what turns this from
    "fit one disease" into "learn a policy class"; with `instances_per_iter=1`
    and a single-instance pool it degenerates to standard single-task PPO.
    """

    def __init__(self, instances, cfg: PPOConfig = None, net=None, noise=None):
        if not HAVE_TORCH:
            raise RuntimeError("torch not available")
        self.cfg = cfg or PPOConfig()
        self.instances = list(instances)
        self.device = torch.device(self.cfg.device)
        torch.manual_seed(self.cfg.seed)
        self.graphs = [InstanceGraph(i) for i in self.instances]
        self.envs = [Env(i, g, reward_scale=self.cfg.reward_scale,
                         randomize_s0=self.cfg.randomize_s0, noise=noise,
                         seed=self.cfg.seed + k)
                     for k, (i, g) in enumerate(zip(self.instances, self.graphs))]
        self.net = (net or ActorCritic(encoder=self.cfg.encoder, hidden=self.cfg.hidden,
                                       layers=self.cfg.layers, heads=self.cfg.heads)
                    ).to(self.device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=self.cfg.lr)
        self.rng = np.random.default_rng(self.cfg.seed)
        self.history = []

    # ------------------------------------------------------------------
    def collect(self, env_ids):
        """Roll out n_env episodes on each of the given instances."""
        buf = []
        for ei in env_ids:
            env, G = self.envs[ei], self.graphs[ei]
            S = env.reset(self.cfg.n_env)
            steps = []
            for t in range(env.T):
                batch = collate([G] * len(S), list(S), [t] * len(S), self.device)
                a, lp, _, v = self.net.act(batch)
                S2, r, _ = env.step(a.cpu().numpy())
                steps.append({"graph": G, "s": S.copy(), "t": t,
                              "a": a.cpu(), "logp": lp.cpu(), "v": v.cpu(),
                              "r": torch.as_tensor(r, dtype=torch.float32)})
                S = S2
            # GAE with a zero terminal value (finite horizon, no bootstrap)
            adv = torch.zeros(self.cfg.n_env)
            last_v = torch.zeros(self.cfg.n_env)
            for t in reversed(range(env.T)):
                nv = steps[t + 1]["v"] if t + 1 < env.T else last_v
                delta = steps[t]["r"] + self.cfg.gamma * nv - steps[t]["v"]
                adv = delta + self.cfg.gamma * self.cfg.lam * adv
                steps[t]["adv"] = adv.clone()
                steps[t]["ret"] = adv + steps[t]["v"]
            buf.extend(steps)
        return buf

    def _flatten(self, buf):
        recs = []
        for st in buf:
            for k in range(len(st["a"])):
                recs.append((st["graph"], st["s"][k], st["t"], int(st["a"][k]),
                             float(st["logp"][k]), float(st["adv"][k]),
                             float(st["ret"][k])))
        return recs

    def train(self, verbose=True):
        cfg = self.cfg
        for it in range(cfg.iters):
            k = min(cfg.instances_per_iter, len(self.envs))
            env_ids = self.rng.choice(len(self.envs), size=k, replace=False)
            buf = self.collect(env_ids)
            recs = self._flatten(buf)
            adv_all = np.array([r[5] for r in recs], dtype=np.float32)
            adv_n = (adv_all - adv_all.mean()) / (adv_all.std() + 1e-8)
            ent_coef = cfg.ent_coef + (cfg.ent_final - cfg.ent_coef) * (it / max(cfg.iters - 1, 1))

            stats = {"pl": [], "vl": [], "ent": [], "kl": []}
            for _ in range(cfg.epochs):
                order = self.rng.permutation(len(recs))
                for st in range(0, len(order), cfg.minibatch):
                    idx = order[st:st + cfg.minibatch]
                    mb = [recs[i] for i in idx]
                    batch = collate([m[0] for m in mb], [m[1] for m in mb],
                                    [m[2] for m in mb], self.device)
                    acts = torch.as_tensor([m[3] for m in mb], dtype=torch.long,
                                           device=self.device)
                    old_lp = torch.as_tensor([m[4] for m in mb], dtype=torch.float32,
                                             device=self.device)
                    A = torch.as_tensor(adv_n[idx], dtype=torch.float32, device=self.device)
                    R = torch.as_tensor([m[6] for m in mb], dtype=torch.float32,
                                        device=self.device)

                    lp, ent, v = self.net.evaluate_actions(batch, acts)
                    ratio = torch.exp(lp - old_lp)
                    pl = -torch.min(ratio * A,
                                    torch.clamp(ratio, 1 - cfg.clip, 1 + cfg.clip) * A).mean()
                    vl = F.mse_loss(v, R)
                    loss = pl + cfg.vf_coef * vl - ent_coef * ent.mean()
                    self.opt.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.net.parameters(), cfg.max_grad_norm)
                    self.opt.step()
                    stats["pl"].append(float(pl)); stats["vl"].append(float(vl))
                    stats["ent"].append(float(ent.mean()))
                    stats["kl"].append(float((old_lp - lp).mean()))

            # mean undiscounted episode return, averaged over the sampled instances
            ep_ret = float(np.mean([float(s["r"].mean()) for s in buf]) * self.envs[0].T)
            rec = {"iter": it,
                   "return": ep_ret,
                   "mean_target": float(np.mean([r[6] for r in recs])),
                   "policy_loss": float(np.mean(stats["pl"])),
                   "value_loss": float(np.mean(stats["vl"])),
                   "entropy": float(np.mean(stats["ent"])),
                   "approx_kl": float(np.mean(stats["kl"]))}
            self.history.append(rec)
            if verbose and (it % cfg.log_every == 0 or it == cfg.iters - 1):
                print(f"  ppo it={it:4d} ret={rec['return']:9.4f} "
                      f"pl={rec['policy_loss']:+.4f} vl={rec['value_loss']:.4f} "
                      f"ent={rec['entropy']:.3f} kl={rec['approx_kl']:+.4f}")
        return self.history


# ======================================================================
# Deployment
# ======================================================================
def torch_policy(net, inst, graph=None, greedy=True, device="cpu"):
    """Wrap a trained network as a `(s, t) -> action` policy for `rollout`."""
    G = graph or InstanceGraph(inst)
    net = net.to(device).eval()

    def pi(s, t):
        batch = collate([G], [np.asarray(s, float)], [int(t)], device)
        with torch.no_grad():
            a, _, _, _ = net.act(batch, greedy=greedy)
        return int(a[0])
    return pi


def save(net, path, cfg=None):
    torch.save({"state_dict": net.state_dict(),
                "encoder": net.encoder_kind,
                "cfg": cfg.__dict__ if cfg is not None else None}, path)


def load(path, device="cpu", **kw):
    ck = torch.load(path, map_location=device, weights_only=False)
    c = ck.get("cfg") or {}
    net = ActorCritic(encoder=ck.get("encoder", "gat"),
                      hidden=c.get("hidden", kw.get("hidden", 64)),
                      layers=c.get("layers", kw.get("layers", 3)),
                      heads=c.get("heads", kw.get("heads", 4)))
    net.load_state_dict(ck["state_dict"])
    return net.to(device)
