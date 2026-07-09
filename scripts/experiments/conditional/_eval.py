"""Shared evaluation helpers for conditional-scheduling experiments.

Metric (per instance): a scheduler's EXPECTED MAKESPAN RATIO against the clairvoyant
offline schedule,

    ratio_S = sum_T p_T * (realized_makespan_S(T) / offline_makespan(T)),

where a trace's realized makespan uses the scheduler's node assignments with times
recomputed for that trace (recalculate_trace_times), and offline_makespan(T) is HEFT
scheduling just T's tasks (knowing T will run). All schedulers are evaluated on the
same instance against the same offline baseline.
"""

from typing import Dict, Optional

from saga import Network, Schedule
from saga.conditional import (
    ConditionalTaskGraph,
    recalculate_trace_times,
    schedule_trace_standalone,
)
from saga.schedulers.cheft import CheftScheduler
from saga.schedulers.ccpop import CCpopScheduler
from saga.schedulers.parametric import ParametricScheduler
from saga.schedulers.parametric.components import (
    CPoPRanking,
    GreedyInsert,
    GreedyInsertCompareFuncs,
    UpwardRanking,
)

_EFT = GreedyInsertCompareFuncs.EFT

# Conditional schedulers under test: unweighted (HEFT/CPoP) vs probability-reweighted
# (CHEFT/CCPoP). All use the same greedy EFT insert; CPoP variants pin the critical path.
SCHEDULERS: Dict[str, ParametricScheduler] = {
    "HEFT": ParametricScheduler(UpwardRanking(), GreedyInsert(compare=_EFT)),
    "CHEFT": CheftScheduler(),
    "CPoP": ParametricScheduler(CPoPRanking(), GreedyInsert(compare=_EFT, critical_path=True)),
    "CCPoP": CCpopScheduler(),
}

# Clairvoyant per-trace baseline (knows which trace runs; schedules it as a plain DAG).
OFFLINE = ParametricScheduler(UpwardRanking(), GreedyInsert(compare=_EFT))


def realized_makespan(schedule: Schedule, trace_tasks) -> float:
    """Makespan of one trace under a committed schedule, times recomputed for that trace."""
    task_set = set(trace_tasks)
    mapping = {
        node: [t for t in tasks if t.name in task_set]
        for node, tasks in schedule.mapping.items()
    }
    recalced = recalculate_trace_times(mapping, schedule)
    return max((t.end for tasks in recalced.values() for t in tasks), default=0.0)


def expected_makespan_ratios(
    network: Network,
    ctg: ConditionalTaskGraph,
    schedulers: Optional[Dict[str, ParametricScheduler]] = None,
) -> Dict[str, float]:
    """Per-instance expected makespan ratio vs offline, for each scheduler."""
    schedulers = schedulers if schedulers is not None else SCHEDULERS
    traces = ctg.identify_traces_detailed()
    offline = [
        schedule_trace_standalone(tr["tasks"], ctg, network, OFFLINE).makespan
        for tr in traces
    ]
    ratios: Dict[str, float] = {}
    for name, scheduler in schedulers.items():
        schedule = scheduler.schedule(network, ctg)
        ratio = 0.0
        for tr, m_off in zip(traces, offline):
            if m_off <= 0:
                continue
            ratio += tr["probability"] * realized_makespan(schedule, tr["tasks"]) / m_off
        ratios[name] = ratio
    return ratios
