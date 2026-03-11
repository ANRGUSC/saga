from typing import Callable, Union, Optional

from saga.utils.random_variable import RandomVariable
from saga import (
    Network,
    NetworkNode,
    NetworkEdge,
    TaskGraph,
    TaskGraphNode,
    TaskGraphEdge,
    Scheduler,
    Schedule
)
from saga.stochastic import (
    StochasticNetwork,
    StochasticTaskGraph,
    StochasticSchedule,
    StochasticScheduledTask,
    StochasticScheduler,
)

#using this approach, we should be able to make an all-in one online scheduler
class EstimateStochasticScheduler(StochasticScheduler):
    def __init__(
        self, 
        scheduler: StochasticScheduler,
        estimate: Callable[[RandomVariable], float],
        seed: Optional[int]= None
    ) -> None:
        self.scheduler = scheduler
        self._estimate = estimate
        self.seed = seed
    




    @property
    def name(self) -> str:
        return f"Determinizer({self.scheduler.name})"

    def determinize(self, rv: Union[RandomVariable, int, float]) -> float:
        if isinstance(rv, RandomVariable):
            return self._estimate(rv)
        return float(rv)

    def schedule(
        self, 
        network: StochasticNetwork, 
        task_graph: StochasticTaskGraph,  
        schedule: Optional[Schedule] = None, 
        min_start_time: float = 0.0 
    ) -> StochasticSchedule:
        """Schedule the tasks on the network.

        Args:
            network (StochasticNetwork): The network on which to schedule the tasks.
            task_graph (StochasticTaskGraph): The task graph to be scheduled.

        Returns:
            StochasticSchedule: The resulting schedule.
        """
        det_network = Network.create(
            nodes=[
                NetworkNode(name=node.name, speed=self.determinize(node.speed))
                for node in network.nodes
            ],
            edges=[
                NetworkEdge(
                    source=edge.source,
                    target=edge.target,
                    speed=self.determinize(edge.speed),
                )
                for edge in network.edges
            ],
        )

        det_task_graph = TaskGraph.create(
            tasks=[
                TaskGraphNode(name=task.name, cost=self.determinize(task.cost))
                for task in task_graph.tasks
            ],
            dependencies=[
                TaskGraphEdge(
                    source=edge.source,
                    target=edge.target,
                    size=self.determinize(edge.size),
                )
                for edge in task_graph.dependencies
            ],
        )
        
        
        if schedule is None and min_start_time == 0.0:
            det_schedule = self.scheduler.schedule(
                network=det_network,
                task_graph=det_task_graph
            )
        else:
            det_schedule = self.scheduler.schedule(
                network=det_network,
                task_graph=det_task_graph,
                schedule=schedule,
                min_start_time=min_start_time
            )
        schedule = StochasticSchedule(task_graph=task_graph, network=network)

        for node_name, tasks in det_schedule.mapping.items():
            for i, det_task in enumerate(tasks):
                new_task = StochasticScheduledTask(
                    node=node_name, name=det_task.name, rank=i
                )
                schedule.add_task(new_task)

        return schedule, det_network, det_task_graph
