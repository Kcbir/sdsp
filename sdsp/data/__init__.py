"""Real datasets. This package contains no synthetic data and no fallbacks.

Every loader raises FileNotFoundError rather than fabricating numbers. If a
dataset is missing, fetch it (see `data/SOURCES.md`) -- do not substitute.
"""
from .mira import (DRUG_NAMES, GENOTYPE_BITS, drug_operators,
                   hypercube_mutation_matrix, load_growth_rates,
                   order_matters_window, sanctuary_operators, summary)

__all__ = ["load_growth_rates", "drug_operators", "sanctuary_operators",
           "hypercube_mutation_matrix", "order_matters_window", "summary",
           "DRUG_NAMES", "GENOTYPE_BITS"]
