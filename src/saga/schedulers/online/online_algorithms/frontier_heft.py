from typing import Any, Callable, Dict, Optional, Tuple

from saga import Network, Schedule, TaskGraph, Scheduler, TaskGraphNode
from saga.schedulers.online.environments import FrontierEnvironment
from saga.schedulers.online.policies import FrontierFillPolicy
from saga.schedulers.parametric.components import GreedyInsert, GreedyInsertCompareFuncs
from saga.schedulers.cpop import upward_rank


class FrontierHeftEnvironment(FrontierEnvironment):
    """Frontier environment that dispatches ready tasks in HEFT upward-rank order."""

    def __init__(self, network: Network, task_graph: TaskGraph, **kwargs: Any) -> None:
        super().__init__(
            network=network,
            task_graph=task_graph,
            policy=FrontierFillPolicy(
                insertion_strategy=GreedyInsert(
                    append_only=False,
                    compare=GreedyInsertCompareFuncs.EFT,
                    critical_path=False,
                ),
            ),
            **kwargs
        )
        self.ready_condition: str = "p_committed"
        self.ready_node_only: bool = False
        self._bootstrap_insert = GreedyInsert(
            append_only=False,
            compare=GreedyInsertCompareFuncs.EFT,
            critical_path=False,
        )
        self.ranks: Dict[str, Tuple[float, int]] = self.compute_upward_ranks()
        # Returns a (-(urank), -topo) tuple for lexicographic ordering in the frontier heap.
        self.priority_condition: Callable[[TaskGraphNode], Any] = lambda x: self.ranks[x.name]

    def compute_upward_ranks(self):
        urank = upward_rank(self.network, self.task_graph)
        topological_sort = {
            node.name: i for i, node in enumerate(reversed(self.task_graph.topological_sort()))
        }
        return {node: (-(urank[node]), (-topological_sort[node])) for node in urank}


class FrontierHeftScheduler(Scheduler):
    def schedule(
        self,
        network: Network,
        task_graph: TaskGraph,
        schedule: Optional[Schedule] = None,
        min_start_time: float = 0.0,
    ) -> Schedule:
        environment = FrontierHeftEnvironment(network, task_graph)
        return environment.run()
