"""VillageSQL 0.0.7 on its MySQL 8.4 base + vsql_vector, driven by
ann-benchmarks.

Identical to the 9.7 module in every respect except the name, for the reason
`algorithms/mariadb123/module.py` gives: ann-benchmarks keys its result files
on the algorithm name and its index parameters and knows nothing about which
server produced them, so sharing `villagesql`'s name would hand an 8.4 run the
9.7 build's result files. The class is subclassed rather than copied so the
two server lines cannot drift apart in how the index is built or queried --
including the plan check, which refuses to time a full scan on either.
"""

from ..villagesql.module import VillageSQL


class VillageSQL84(VillageSQL):
    vb_engine = "villagesql84"

    def __str__(self) -> str:
        return (f"VillageSQL84(M={self._m}, "
                f"ef_construction={self._ef_construction}, "
                f"ef_search={self._ef_search})")
