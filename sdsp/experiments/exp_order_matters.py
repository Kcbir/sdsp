"""MAIN EXPERIMENT: on real data, which beta-lactam subsets require switching?

Run:  python -m sdsp.experiments.exp_order_matters --sizes 2 3 --out results/

Everything here is computed from Mira et al. (2015) measured growth rates. There
is no synthetic instance in this file.

What it produces
----------------
1. `order_matters_pairs.csv`   every drug pair/triple, with certified bounds, the
                               switching gain, the witness schedule, and the
                               dilution window (D_low, D_high] at which only
                               switching cures.
2. `sensitivity.csv`           the same headline numbers swept over the two free
                               parameters (mutation rate mu, log-growth per
                               cycle), because a result that survives only one
                               parameter choice is not a result.
3. `robustness.csv`            for the top candidates, the fraction of
                               log-normal perturbations of the measured growth
                               rates under which the witness schedule still
                               contracts -- the number to quote given that the
                               input rates carry measurement error.
4. `protocol.md`               the top candidates written as bench protocols.

Reading the output
------------------
The column that matters is `gain` = rho_fix / rho_low. It is dimensionless and
independent of dilution, and log10(gain) is exactly the width in log-dilution of
the window where sequencing is the difference between cure and failure. A pair
with gain 1.0 is a pair for which order does not matter at all.
"""
from __future__ import annotations

import argparse
import csv
import itertools
from pathlib import Path

import numpy as np

from ..data.mira import (DRUG_NAMES, drug_operators, load_growth_rates,
                         order_matters_window, resample_operators,
                         sanctuary_operators)
from ..switching import (det_lower_bound, fixed_best, jsr_upper_cqlf,
                         protocol_power, robustness_margin,
                         search_periodic_exhaustive, stabilizability_verdict)


# ----------------------------------------------------------------------
def screen(mats, names, sizes=(2, 3), k_max=6, use_sdp=False, verbose=True,
           dilution_free=True):
    rows = []
    for size in sizes:
        combos = list(itertools.combinations(range(len(mats)), size))
        if verbose:
            print(f"  screening {len(combos)} subsets of size {size} "
                  f"(k_max={k_max})")
        for n, combo in enumerate(combos):
            sub = [mats[i] for i in combo]
            v = stabilizability_verdict(sub, k_max=k_max, use_sdp=use_sdp,
                                        dilution_free=dilution_free)
            wit = [names[combo[a]] for a in v.witness] if v.witness else []
            rows.append({
                "size": size,
                "drugs": "+".join(names[i] for i in combo),
                "rho_fix": round(v.rho_fix, 6),
                "rho_low_upper": round(v.rho_low_upper, 6),
                "rho_low_lower": round(v.rho_low_lower, 6),
                "gain": round(v.gain, 6),
                "log10_dilution_window": round(float(np.log10(max(v.gain, 1.0))), 6),
                "period": v.witness_length,
                "witness": "->".join(wit),
                "best_fixed_drug": names[combo[v.best_fixed_drug]],
                "D_low": round(v.rho_low_upper, 6),
                "D_high": round(v.rho_fix, 6),
                "margin": round(robustness_margin(sub, v.witness), 6) if v.witness else "",
                "verdict": v.verdict.split(":")[0],
            })
            if verbose and (n + 1) % 20 == 0:
                print(f"    {n + 1}/{len(combos)}")
    rows.sort(key=lambda r: -r["gain"])
    return rows


def sensitivity(names, mus=(1e-4, 1e-3, 1e-2), growths=(0.35, 0.69, 1.39),
                sizes=(2,), k_max=5, top=5, verbose=True):
    """Sweep the two free parameters and report whether the top hits are stable.

    If the set of switching-stabilizable pairs changes qualitatively across this
    grid, that is the headline finding and must be reported as such, not buried.
    """
    genotypes, drugs, R = load_growth_rates()
    out = []
    for mu in mus:
        for mlg in growths:
            mats = drug_operators(R, genotypes, mu=mu, max_log_growth=mlg)
            rows = screen(mats, names, sizes=sizes, k_max=k_max, use_sdp=False,
                          verbose=False)
            for r in rows[:top]:
                out.append({"mu": mu, "max_log_growth": round(mlg, 4),
                            **{k: r[k] for k in
                               ("drugs", "rho_fix", "rho_low_upper", "gain",
                                "witness", "verdict")}})
            if verbose:
                best = rows[0] if rows else None
                print(f"  mu={mu:<8g} logg={mlg:<5.2f} top={best['drugs'] if best else '-'} "
                      f"gain={best['gain'] if best else float('nan'):.3f}")
    return out


def robustness(mats, names, rows, top=10, rel_noise=0.10, n_draws=500,
               mu=1e-3, max_log_growth=float(np.log(2.0)), sanctuary=False):
    """For each top candidate: would the bench experiment still work at 10% error?

    Perturbs the MEASURED GROWTH RATES and rebuilds the operators (see
    `mira.resample_operators`), then asks the two-armed question in
    `switching.protocol_power`: at the dilution chosen from nominal data, does
    the cycle still clear the culture AND does every single drug still fail?

    Falls back to entrywise matrix jitter for the sanctuary variant, where the
    operator is not a direct function of one R matrix.
    """
    out = []
    for r in rows[:top]:
        idx = [names.index(d) for d in r["drugs"].split("+")]
        sub = [mats[i] for i in idx]
        seq = [names.index(d) for d in r["witness"].split("->")] if r["witness"] else []
        seq = [idx.index(s) for s in seq] if seq else []
        if not seq:
            continue
        resample = None if sanctuary else resample_operators(
            subset=idx, mu=mu, max_log_growth=max_log_growth, rel_noise=rel_noise)
        pr = protocol_power(sub, seq, resample=resample, rel_noise=rel_noise,
                            n_draws=n_draws)
        out.append({"drugs": r["drugs"], "witness": r["witness"],
                    "gain": r["gain"],
                    **{k: (round(v, 6) if isinstance(v, float) else v)
                       for k, v in pr.items()}})
    return out


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {path} ({len(rows)} rows)")


def write_protocol(path, mats, names, rows, top=5):
    lines = ["# Candidate switching protocols", "",
             "Derived from Mira et al. (2015) measured growth rates. Each entry is",
             "a falsifiable prediction: at any dilution in the stated window, the",
             "cycle clears the culture and the best single drug does not.", ""]
    for r in rows[:top]:
        if "ORDER MATTERS" not in r["verdict"]:
            continue
        idx = [names.index(d) for d in r["drugs"].split("+")]
        win = order_matters_window([mats[i] for i in idx])
        lines += [
            f"## {r['drugs']}",
            f"- cycle: **{r['witness']}** (period {r['period']}, one drug per passage)",
            f"- passage at 1:D for D in ({win['D_low']:.4g}, {win['D_high']:.4g}]",
            f"- control arm: {r['best_fixed_drug']} alone at the same dilution "
            f"(predicted to fail)",
            f"- switching gain {r['gain']:.3f}x "
            f"(= {r['log10_dilution_window']:.2f} decades of dilution)",
            f"- tolerates a uniform growth-rate error of {float(r['margin']):.1%}"
            if r["margin"] != "" else "",
            "",
        ]
    Path(path).write_text("\n".join(lines))
    print(f"  wrote {path}")


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sizes", type=int, nargs="+", default=[2, 3])
    ap.add_argument("--k-max", type=int, default=6)
    ap.add_argument("--mu", type=float, default=1e-3)
    ap.add_argument("--max-log-growth", type=float, default=float(np.log(2.0)))
    ap.add_argument("--sdp", action="store_true", help="also compute SDP bounds")
    ap.add_argument("--sanctuary", action="store_true",
                    help="two-compartment variant with a low-drug sanctuary well")
    ap.add_argument("--penetration", type=float, default=0.2)
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--skip-sensitivity", action="store_true")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    genotypes, drugs, R = load_growth_rates()
    print(f"Mira et al. 2015: {len(genotypes)} genotypes x {len(drugs)} drugs")
    print(f"  growth rates in [{R.min():.3f}, {R.max():.3f}] (x1e-3)")

    if args.sanctuary:
        mats = sanctuary_operators(R, genotypes, mu=args.mu,
                                   max_log_growth=args.max_log_growth,
                                   penetration=args.penetration)
        tag = f"_sanctuary{args.penetration:g}"
        print(f"  two-compartment, sanctuary penetration {args.penetration}")
    else:
        mats = drug_operators(R, genotypes, mu=args.mu,
                              max_log_growth=args.max_log_growth)
        tag = ""

    rfix, jfix = fixed_best(mats)
    print(f"  best single drug: {drugs[jfix]}  rho={rfix:.4f}")
    print(f"  determinant lower bound on rho_low (all 15): {det_lower_bound(mats):.4f}")
    if args.sdp:
        j = jsr_upper_cqlf(mats)
        print(f"  CQLF upper bound on JSR (all 15): "
              f"{j if j is None else round(j, 4)}")

    print("\n[1/4] screening subsets")
    rows = screen(mats, drugs, sizes=tuple(args.sizes), k_max=args.k_max,
                  use_sdp=args.sdp)
    write_csv(args.out / f"order_matters{tag}.csv", rows)

    n_order = sum("ORDER MATTERS" in r["verdict"] for r in rows)
    widths = np.array([r["log10_dilution_window"] for r in rows])
    print(f"\n  {n_order}/{len(rows)} subsets have a NON-EMPTY dilution window")
    print(f"  (i.e. some cyclic schedule strictly beats every fixed drug)")
    print(f"  window width, decades of dilution: max={widths.max():.4f}  "
          f"median={np.median(widths):.4f}  frac>0.1={np.mean(widths > 0.1):.2%}")
    print(f"  top by switching gain:")
    for r in rows[:8]:
        print(f"    {r['drugs']:<12} gain={r['gain']:6.3f}  "
              f"rho_fix={r['rho_fix']:.4f} -> rho_low<={r['rho_low_upper']:.4f}  "
              f"cycle {r['witness']}")

    print("\n[2/4] robustness to growth-rate measurement error")
    rob = robustness(mats, drugs, rows, mu=args.mu,
                     max_log_growth=args.max_log_growth,
                     sanctuary=args.sanctuary)
    write_csv(args.out / f"robustness{tag}.csv", rob)
    for r in rob[:8]:
        print(f"    {r['drugs']:<12} P(demonstrates | 10% rate error) = "
              f"{r['p_demonstrates']:.2f}   "
              f"P(switching still wins) = {r['p_gain_positive']:.2f}   "
              f"median gain = {r['median_gain']:.3f}")

    if not args.skip_sensitivity:
        print("\n[3/4] sensitivity to the two free parameters")
        sens = sensitivity(drugs, sizes=tuple(args.sizes[:1]),
                           k_max=min(args.k_max, 5))
        write_csv(args.out / f"sensitivity{tag}.csv", sens)

    print("\n[4/4] writing bench protocols")
    write_protocol(args.out / f"protocol{tag}.md", mats, drugs, rows)
    print("\ndone.")


if __name__ == "__main__":
    main()
