"""Conditional CPoP (CCPoP) scheduler.

CCPoP is CPoP with its rank priority computed on the probability-weighted task
graph. It is expressed as a parametric scheduler: a ``ProbabilityWeighted(CPoPRanking)``
priority plus greedy earliest-finish-time insertion pinned to the critical path.
On a non-conditional graph the weighting is a no-op and CCPoP reduces to CPoP.

Note: the parametric critical-path insertion determines the critical task from
the unweighted CPoP ranks and pins it to the fastest node, which differs slightly
from a fully probability-weighted critical-path choice.
"""

from saga.schedulers.parametric import ParametricScheduler
from saga.schedulers.parametric.components import (
    CPoPRanking,
    GreedyInsert,
    GreedyInsertCompareFuncs,
    ProbabilityWeighted,
)


class CCpopScheduler(ParametricScheduler):
    """CPoP with probability-weighted ranking for conditional task graphs."""

    def __init__(self) -> None:
        super().__init__(
            initial_priority=ProbabilityWeighted(base=CPoPRanking()),
            insert_task=GreedyInsert(
                append_only=False,
                compare=GreedyInsertCompareFuncs.EFT,
                critical_path=True,
            ),
        )

    @property
    def name(self) -> str:
        return "CCpopScheduler"
