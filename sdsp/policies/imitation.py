"""Imitation learning from an optimal expert (Gurobi MILP or exact alpha-vector DP).

Three stages, in the order you should run them:

1. `behavior_cloning`  supervised cross-entropy on (state, expert action) pairs
                       drawn from expert trajectories. Cheap, and it gets you
                       most of the way, but it suffers the standard covariate
                       shift: the learner is only ever trained on states the
                       *expert* visits, and a single early mistake takes it
                       somewhere it has never seen. In this domain that failure
                       is severe, because a supercritical instance amplifies any
                       deviation geometrically -- which is exactly what
                       Theorem 3.4 says.

2. `dagger`            roll out the *learner*, query the expert at the states the
                       learner actually reaches, aggregate, retrain. This is the
                       principled fix for (1) and it is only possible because our
                       experts are queryable at arbitrary states: `AlphaDP` gives
                       the optimal action at any (s,t) in microseconds, and
                       `milp_expert` re-solves Gurobi from any (s,t).

3. PPO fine-tuning     (see `sdsp.policies.rl`) initialised from the DAgger
                       weights. Imitation supplies a strong prior; PPO then
                       optimises the actual objective, including the stochastic
                       dynamics that the deterministic expert never saw.

Expert choice: `AlphaDP` is exact, ~1e4x faster than Gurobi, and gives the same
answer (validated in tests/test_solvers.py), so it is the default. Use
`milp_expert` when you want the headline "learned from Gurobi" claim, or when
the instance is large enough that the alpha set has to be capped.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAVE_TORCH = True
except Exception:                                        # pragma: no cover
    HAVE_TORCH = False

from .exact import AlphaDP
from .rl import ActorCritic, InstanceGraph, collate, torch_policy


# ======================================================================
# Experts
# ======================================================================
def dp_expert(inst, **kw):
    """Exact optimal (s,t) -> action from the alpha-vector DP. Fast; default."""
    dp = AlphaDP(inst, **kw)
    return dp.policy(), dp


def gurobi_expert(inst, time_limit=10.0):
    """Exact optimal (s,t) -> action by re-solving the MILP from that state."""
    from .milp import milp_expert
    return milp_expert(inst, time_limit=time_limit), None


EXPERTS = {"dp": dp_expert, "gurobi": gurobi_expert}


# ======================================================================
# Dataset
# ======================================================================
@dataclass
class Sample:
    graph: "InstanceGraph"
    s: np.ndarray
    t: int
    a: int                     # expert action
    q: np.ndarray | None = None   # optional: expert Q-values over all actions


class ExpertDataset:
    """(graph, state, t) -> expert action, pooled across instances."""

    def __init__(self):
        self.samples: list[Sample] = []

    def __len__(self):
        return len(self.samples)

    def add(self, *a, **kw):
        self.samples.append(Sample(*a, **kw))

    def extend(self, other):
        self.samples.extend(other.samples)

    def split(self, frac=0.9, seed=0):
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(self.samples))
        k = int(frac * len(idx))
        a, b = ExpertDataset(), ExpertDataset()
        a.samples = [self.samples[i] for i in idx[:k]]
        b.samples = [self.samples[i] for i in idx[k:]]
        return a, b


def collect_expert_data(instances, graphs=None, n_traj=8, expert="dp",
                        jitter=0.35, seed=0, with_q=True, verbose=False,
                        expert_kw=None):
    """Roll the expert out from perturbed initial states and record its choices.

    `jitter` log-normally perturbs s0 on each trajectory. Without it every
    trajectory from a deterministic instance is identical and the dataset has
    exactly T distinct points -- a classic and easy-to-miss way to train a
    behaviour-cloning policy that memorises one trajectory.

    `with_q=True` also records the expert's full action-value vector, enabling
    the soft/Q-weighted cloning loss in `behavior_cloning(loss="kl")`, which is
    much more informative than a one-hot target when several actions are nearly
    tied -- common here, since many drug combinations are near-equivalent.
    """
    from ..fastsim import get_sim
    rng = np.random.default_rng(seed)
    graphs = graphs or [InstanceGraph(i) for i in instances]
    ds = ExpertDataset()
    expert_kw = expert_kw or {}
    for inst, G in zip(instances, graphs):
        if expert == "dp":
            dp = AlphaDP(inst, **expert_kw)
            pi = dp.policy()
        else:
            pi, dp = EXPERTS[expert](inst, **expert_kw)
        sim = get_sim(inst)
        for _ in range(n_traj):
            s = inst.s0 * np.exp(rng.normal(0, jitter, size=inst.N)) if jitter > 0 \
                else inst.s0.copy()
            for t in range(inst.T):
                a = int(pi(s, t))
                q = None
                if with_q and expert == "dp":
                    q = _dp_action_values(dp, s, t)
                ds.add(G, s.copy(), t, a, q)
                s = sim.Weff[a] @ s
        if verbose:
            print(f"  collected {inst.name}: {len(ds)} samples")
    return ds


def _dp_action_values(dp, s, t):
    """Exact Q(s,t,a) for every action, from the alpha-vector value function."""
    inst = dp.inst
    c_eff = inst.c + inst.lam_res * inst.resistant_nodes.astype(float)
    s_nxt = ((1.0 - inst.kill) * np.asarray(s, float)[None, :]) @ inst.W.T
    nxt = dp.Gamma[min(t + 1, inst.T)]
    tail = np.min(np.hstack([s_nxt, np.ones((len(s_nxt), 1))]) @ nxt.T, axis=1)
    return s_nxt @ c_eff + inst.lam_tox * inst.tox + inst.gamma * tail


# ======================================================================
# Behaviour cloning
# ======================================================================
def behavior_cloning(dataset, net=None, epochs=30, batch_size=128, lr=1e-3,
                     device="cpu", loss="ce", tau=None, val=None, seed=0,
                     verbose=True, encoder="gat", hidden=64, layers=3, heads=4,
                     weight_decay=0.0):
    """Supervised cloning of the expert.

    loss="ce"  cross-entropy against the expert's argmin action.
    loss="kl"  KL against a Boltzmann policy over the expert's exact Q-values,
               softmax(-Q/tau). Requires `with_q=True` at collection time. This
               transfers the expert's indifference structure, not just its
               argmin, and in a domain with many near-tied combinations that is
               a much denser training signal.

    `tau` defaults to a per-sample scale (the interquartile range of Q), which
    keeps the target distribution meaningful across instances whose objective
    magnitudes differ by orders of magnitude.
    """
    if not HAVE_TORCH:
        raise RuntimeError("torch not available")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    dev = torch.device(device)
    net = (net or ActorCritic(encoder=encoder, hidden=hidden, layers=layers,
                              heads=heads)).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    hist = []

    for ep in range(epochs):
        net.train()
        order = rng.permutation(len(dataset))
        tot, nb, acc = 0.0, 0, 0.0
        for st in range(0, len(order), batch_size):
            mb = [dataset.samples[i] for i in order[st:st + batch_size]]
            batch = collate([m.graph for m in mb], [m.s for m in mb],
                            [m.t for m in mb], dev)
            logits = net.logits_per_graph(batch)
            if loss == "kl":
                terms = []
                for lg, m in zip(logits, mb):
                    q = torch.as_tensor(m.q, dtype=torch.float32, device=dev)
                    scale = tau if tau is not None else \
                        float(np.subtract(*np.percentile(m.q, [75, 25])) + 1e-9)
                    tgt = F.softmax(-q / max(scale, 1e-9), dim=0)
                    terms.append(F.kl_div(F.log_softmax(lg, dim=0), tgt,
                                          reduction="sum"))
                L = torch.stack(terms).mean()
            else:
                L = torch.stack([
                    F.cross_entropy(lg.unsqueeze(0),
                                    torch.tensor([m.a], device=dev))
                    for lg, m in zip(logits, mb)]).mean()
            opt.zero_grad(set_to_none=True)
            L.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            tot += float(L); nb += 1
            acc += float(np.mean([int(torch.argmax(lg)) == m.a
                                  for lg, m in zip(logits, mb)]))
        rec = {"epoch": ep, "loss": tot / max(nb, 1), "train_acc": acc / max(nb, 1)}
        if val is not None and len(val):
            rec["val_acc"] = expert_agreement(net, val, device=device)
        hist.append(rec)
        if verbose and (ep % 5 == 0 or ep == epochs - 1):
            extra = f" val_acc={rec.get('val_acc', float('nan')):.3f}" if val else ""
            print(f"  bc ep={ep:3d} loss={rec['loss']:.4f} "
                  f"acc={rec['train_acc']:.3f}{extra}")
    return net, hist


@torch.no_grad() if HAVE_TORCH else (lambda f: f)
def expert_agreement(net, dataset, device="cpu", batch_size=256):
    """Fraction of held-out states where the net's argmax equals the expert's."""
    net.eval()
    dev = torch.device(device)
    ok, n = 0, 0
    for st in range(0, len(dataset), batch_size):
        mb = dataset.samples[st:st + batch_size]
        batch = collate([m.graph for m in mb], [m.s for m in mb],
                        [m.t for m in mb], dev)
        for lg, m in zip(net.logits_per_graph(batch), mb):
            ok += int(int(torch.argmax(lg)) == m.a)
            n += 1
    return ok / max(n, 1)


# ======================================================================
# DAgger
# ======================================================================
def dagger(instances, graphs=None, rounds=8, n_traj=8, epochs=10, expert="dp",
           beta0=1.0, beta_decay=0.6, device="cpu", seed=0, net=None,
           batch_size=128, lr=1e-3, verbose=True, encoder="gat", hidden=64,
           layers=3, heads=4, jitter=0.35, expert_kw=None, eval_every=1):
    """Dataset Aggregation (Ross, Gordon & Bagnell 2011).

    Round i rolls out the mixture policy  beta_i * expert + (1-beta_i) * learner,
    labels *every* visited state with the expert's action, appends to the
    aggregate dataset, and retrains from scratch on the aggregate. beta decays
    geometrically so control passes to the learner.

    Returns (net, dataset, history) where history records, per round, the
    learner's realised cost against the certified optimum -- i.e. a true
    optimality gap, not a proxy.
    """
    if not HAVE_TORCH:
        raise RuntimeError("torch not available")
    from ..dynamics import rollout
    from ..fastsim import get_sim

    rng = np.random.default_rng(seed)
    instances = list(instances)
    graphs = graphs or [InstanceGraph(i) for i in instances]
    expert_kw = expert_kw or {}

    experts, opt_vals = [], []
    for inst in instances:
        if expert == "dp":
            dp = AlphaDP(inst, **expert_kw)
            experts.append((dp.policy(), dp))
            opt_vals.append(dp.J)
        else:
            pi, _ = EXPERTS[expert](inst, **expert_kw)
            experts.append((pi, None))
            opt_vals.append(AlphaDP(inst).J)

    # round 0: pure expert data
    ds = collect_expert_data(instances, graphs, n_traj=n_traj, expert=expert,
                             jitter=jitter, seed=seed, with_q=(expert == "dp"),
                             expert_kw=expert_kw)
    net, _ = behavior_cloning(ds, net=net, epochs=epochs, batch_size=batch_size,
                              lr=lr, device=device, verbose=False, seed=seed,
                              encoder=encoder, hidden=hidden, layers=layers,
                              heads=heads)
    history = []

    for r in range(rounds):
        beta = beta0 * (beta_decay ** r)
        new = ExpertDataset()
        for k, (inst, G) in enumerate(zip(instances, graphs)):
            pi_e, dp = experts[k]
            pi_l = torch_policy(net, inst, G, greedy=False, device=device)
            sim = get_sim(inst)
            for _ in range(n_traj):
                s = inst.s0 * np.exp(rng.normal(0, jitter, size=inst.N)) \
                    if jitter > 0 else inst.s0.copy()
                for t in range(inst.T):
                    a_e = int(pi_e(s, t))                      # label: always expert
                    q = _dp_action_values(dp, s, t) if dp is not None else None
                    new.add(G, s.copy(), t, a_e, q)
                    a_take = a_e if rng.random() < beta else int(pi_l(s, t))
                    s = sim.Weff[a_take] @ s
        ds.extend(new)
        net, _ = behavior_cloning(ds, net=net, epochs=epochs, batch_size=batch_size,
                                  lr=lr, device=device, verbose=False, seed=seed + r,
                                  encoder=encoder, hidden=hidden, layers=layers,
                                  heads=heads)

        rec = {"round": r, "beta": beta, "n_samples": len(ds)}
        if eval_every and r % eval_every == 0:
            gaps, agree = [], expert_agreement(net, ds, device=device)
            for k, (inst, G) in enumerate(zip(instances, graphs)):
                J = rollout(inst, torch_policy(net, inst, G, greedy=True, device=device))
                gaps.append(J / max(abs(opt_vals[k]), 1e-12))
            rec.update({"mean_ratio_to_opt": float(np.mean(gaps)),
                        "median_ratio_to_opt": float(np.median(gaps)),
                        "expert_agreement": agree})
        history.append(rec)
        if verbose:
            print(f"  dagger round {r}: beta={beta:.3f} n={len(ds)} "
                  f"J/J*={rec.get('mean_ratio_to_opt', float('nan')):.4f} "
                  f"agree={rec.get('expert_agreement', float('nan')):.3f}")
    return net, ds, history


# ======================================================================
# Full pipeline
# ======================================================================
def bc_then_ppo(train_instances, ppo_cfg=None, dagger_rounds=5, bc_epochs=20,
                expert="dp", device="cpu", seed=0, verbose=True, noise=None,
                **kw):
    """The pipeline: DAgger warm-start, then PPO fine-tuning on the real objective.

    Returns (net, info). `info["stage"]` records values after each stage so you
    can report the marginal contribution of imitation versus RL -- which is the
    ablation a reviewer will ask for.
    """
    from .rl import PPOConfig, PPOTrainer
    from ..dynamics import rollout

    cfg = ppo_cfg or PPOConfig(device=device, seed=seed)
    net, ds, dag_hist = dagger(train_instances, rounds=dagger_rounds,
                               epochs=bc_epochs, expert=expert, device=device,
                               seed=seed, verbose=verbose, encoder=cfg.encoder,
                               hidden=cfg.hidden, layers=cfg.layers,
                               heads=cfg.heads, **kw)
    after_il = [rollout(i, torch_policy(net, i, greedy=True, device=device))
                for i in train_instances]
    trainer = PPOTrainer(train_instances, cfg=cfg, net=net, noise=noise)
    ppo_hist = trainer.train(verbose=verbose)
    after_rl = [rollout(i, torch_policy(trainer.net, i, greedy=True, device=device))
                for i in train_instances]
    opt = [AlphaDP(i).J for i in train_instances]
    info = {
        "dagger_history": dag_hist,
        "ppo_history": ppo_hist,
        "stage": {
            "after_imitation": {"J": after_il,
                                "ratio": [a / max(abs(o), 1e-12)
                                          for a, o in zip(after_il, opt)]},
            "after_ppo": {"J": after_rl,
                          "ratio": [a / max(abs(o), 1e-12)
                                    for a, o in zip(after_rl, opt)]},
            "optimal": opt,
        },
        "n_expert_samples": len(ds),
    }
    return trainer.net, info
