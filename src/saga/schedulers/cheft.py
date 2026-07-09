"""Conditional HEFT (CHEFT) scheduler.

CHEFT is HEFT with its upward-rank priority computed on the probability-weighted
task graph, so tasks on likely branches are prioritized. It is expressed as a
parametric scheduler: a ``ProbabilityWeighted(UpwardRanking)`` priority plus
greedy earliest-finish-time insertion. On a non-conditional graph the weighting
is a no-op and CHEFT reduces to standard HEFT.
"""

from saga.schedulers.parametric import ParametricScheduler
from saga.schedulers.parametric.components import (
    GreedyInsert,
    GreedyInsertCompareFuncs,
    ProbabilityWeighted,
    UpwardRanking,
)


class CheftScheduler(ParametricScheduler):
    """HEFT with probability-weighted upward rank for conditional task graphs."""

    def __init__(self) -> None:
        super().__init__(
            initial_priority=ProbabilityWeighted(base=UpwardRanking()),
            insert_task=GreedyInsert(
                append_only=False, compare=GreedyInsertCompareFuncs.EFT
            ),
        )

    @property
    def name(self) -> str:
        return "CheftScheduler"
