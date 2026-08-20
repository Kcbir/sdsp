"""Re-download Mira et al. Table 4 from the publisher and diff it against the CSV.

    python -m sdsp.data.verify_mira

The stored copy in `data/mira2015/growth_rates.csv` was transcribed by hand, so
it MUST be verified before any derived number is published. This script
re-fetches the article and reports any cell that differs.

Source of record is the **PLoS ONE JATS XML**, not the PMC HTML mirror: PMC now
serves a reCAPTCHA interstitial to scripted clients, and the publisher's XML is
the authoritative CC-BY copy anyway.

Table 4 in the published layout is *transposed* relative to our CSV (published
rows are drugs, columns are genotypes) and is split into two blocks of eight
genotypes. This script handles both, so a clean diff means the stored matrix
agrees cell-for-cell with the published one under the correct orientation.
"""
from __future__ import annotations

import re
import ssl
import sys
import urllib.request

import numpy as np

from .mira import CSV, load_growth_rates

DOI = "10.1371/journal.pone.0122283"
XML_URL = ("https://journals.plos.org/plosone/article/file"
           f"?id={DOI}&type=manuscript")
TABLE_ID = "pone.0122283.t004"


def _ssl_context():
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def fetch_xml(url=XML_URL, timeout=90) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "sdsp-verify/2.0"})
    with urllib.request.urlopen(req, timeout=timeout,
                                context=_ssl_context()) as r:
        return r.read().decode("utf-8", errors="replace")


def _cells(row: str) -> list[str]:
    return [" ".join(re.sub(r"<[^>]+>", " ", c).replace("−", "-").split())
            for c in re.findall(r"<t[dh]\b[^>]*>(.*?)</t[dh]>", row, re.S)]


def parse_table4(xml: str) -> dict[tuple[str, str], float]:
    """-> {(genotype, drug): growth rate} from the published table."""
    m = re.search(rf'<table-wrap\b[^>]*id="{re.escape(TABLE_ID)}".*?</table-wrap>',
                  xml, re.S)
    if m is None:
        raise LookupError(f"table {TABLE_ID} not found; layout may have changed")
    grid = [_cells(r) for r in
            re.findall(r"<tr\b[^>]*>(.*?)</tr>", m.group(0), re.S)]

    ref: dict[tuple[str, str], float] = {}
    for i, row in enumerate(grid):
        if not row or not all(re.fullmatch(r"[01]{4}", c) for c in row):
            continue                        # not a genotype header row
        header = row
        for r in grid[i + 1:]:
            if not r or re.fullmatch(r"[01]{4}", r[0]):
                break                       # next block starts
            drug, vals = r[0], r[1:]
            if len(vals) != len(header):
                raise ValueError(f"row {drug!r}: {len(vals)} values, "
                                 f"{len(header)} genotypes")
            for g, v in zip(header, vals):
                ref[(g, drug)] = float(v)
    if not ref:
        raise LookupError("no genotype header row found in the table")
    return ref


def main() -> int:
    print(f"stored : {CSV}")
    genos, drugs, R = load_growth_rates()
    print(f"         {R.shape[0]} genotypes x {R.shape[1]} drugs")
    print(f"fetching {XML_URL}")
    try:
        ref = parse_table4(fetch_xml())
    except Exception as e:
        print(f"VERIFY FAILED: {type(e).__name__}: {e}")
        print("Cannot verify. Do not publish numbers derived from this file.")
        return 2

    print(f"published: {len(ref)} cells parsed from Table 4")
    missing = [(g, d) for g in genos for d in drugs if (g, d) not in ref]
    if missing:
        print(f"MISSING {len(missing)} cells in the published table, e.g. {missing[:5]}")
        return 1

    bad = [(g, d, R[i, j], ref[(g, d)])
           for i, g in enumerate(genos) for j, d in enumerate(drugs)
           if abs(R[i, j] - ref[(g, d)]) > 1e-9]
    if not bad:
        print(f"MATCH: all {R.size} cells identical to the published Table 4.")
        print(f"       (orientation: published rows = drugs, columns = genotypes)")
        return 0

    print(f"MISMATCH in {len(bad)} cells:")
    for g, d, a, b in bad[:40]:
        print(f"   {g} {d:<4} stored={a:.3f}  published={b:.3f}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
