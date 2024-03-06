from typing import Any, Callable, Dict, Hashable, List, Optional
from networkx import DiGraph, Graph
from exp_parametric import ParametricScheduler, GreedyInsert, InsertTask, GREEDY_INSERT_COMPARE_FUNCS, UpwardRanking, get_insert_loc, cpop_ranks
from saga.scheduler import Task
from saga.schedulers.parametric import ScheduleType
import networkx as nx
import numpy as np
from saga.utils.draw import draw_gantt

from saga.utils.random_graphs import add_random_weights, get_branching_dag, get_network

class ConstrainedGreedyInsert(InsertTask):
    def __init__(self,
                 append_only: bool = False,
                 compare: Callable[[nx.Graph, nx.DiGraph, ScheduleType, Task, Task], float] = lambda new, cur: new.end - cur.end,
                 critical_path: bool = False):
        """Initialize the GreedyInsert class.
        
        Args:
            append_only (bool, optional): Whether to only append the task to the schedule. Defaults to False.
            compare (Callable[[Task, Task], float], optional): The comparison function to use. Defaults to lambda new, cur: new.end - cur.end.
                Must be one of "EFT", "EST", or "Quickest".
            critical_path (bool, optional): Whether to only schedule tasks on the critical path. Defaults to False.
        """
        self.append_only = append_only
        self._compare = compare
        self.critical_path = critical_path

    def __call__(self,
                 network: nx.Graph,
                 task_graph: nx.DiGraph,
                 schedule: ScheduleType,
                 task: Hashable,
                 node: Optional[Hashable] = None,
                 dry_run: bool = False) -> Task:
        """Insert a task into the schedule.

        Args:
            network (nx.Graph): The network of resources.
            task_graph (nx.DiGraph): The task graph.
            schedule (ScheduleType): The current schedule.
            task (Hashable): The task to insert.
            node (Hashable, optional): The node to insert the task onto. Defaults to None.
            dry_run (bool, optional): Whether to actually insert the task. Defaults to False.

        Returns:
            Task: The inserted task.
        """ 
        best_insert_loc, best_task = None, None
        # second_best_task = None
        # TODO: This is a bit inefficient adds O(n) per iteration,
        #       so it doesn't affect asymptotic complexity, but it's still bad
        if self.critical_path and node is None:
            # if task_graph doesn't have critical_path attribute, then we need to calculate it
            if "critical_path" not in task_graph.nodes[task]:
                ranks = cpop_ranks(network, task_graph)
                critical_rank = max(ranks.values())
                fastest_node = max(
                    network.nodes,
                    key=lambda node: network.nodes[node]['weight']
                )
                nx.set_node_attributes(
                    task_graph,
                    {
                        task: fastest_node if np.isclose(rank, critical_rank) else None
                     for task, rank in ranks.items()
                    },
                    "critical_path"
                )

            # if task is on the critical path, then we only consider the fastest node
            node = task_graph.nodes[task]["critical_path"]

        considered_nodes = network.nodes if node is None else [node]
        for node in considered_nodes:
            exec_time = task_graph.nodes[task]['weight'] / network.nodes[node]['weight']

            min_start_time = 0
            for parent in task_graph.predecessors(task):
                parent_task: Task = task_graph.nodes[parent]['scheduled_task']
                parent_node = parent_task.node
                data_size = task_graph.edges[parent, task]["weight"]
                comm_strength = network.edges[parent_node, node]["weight"]
                comm_time = data_size / comm_strength
                min_start_time = max(min_start_time, parent_task.end + comm_time)

            if self.append_only:
                start_time = max(
                    min_start_time,
                    0.0 if not schedule[node] else schedule[node][-1].end
                )
                insert_loc = len(schedule[node])
            else:
                insert_loc, start_time  = get_insert_loc(schedule[node], min_start_time, exec_time)

            new_task = Task(node, task, start_time, start_time + exec_time)
            if best_task is None or self._compare(network, task_graph, schedule, new_task, best_task) < 0:
                best_insert_loc, best_task = insert_loc, new_task

        if not dry_run: # if not a dry run, then insert the task into the schedule
            schedule[best_task.node].insert(best_insert_loc, best_task)
            task_graph.nodes[task]['scheduled_task'] = best_task
        return best_task


    def serialize(self) -> Dict[str, Any]:
        return {
            "name": "ConstrainedGreedyInsert",
            "append_only": self.append_only,
            "compare": self._compare,
            "critical_path": self.critical_path
        }
    
    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> "GreedyInsert":
        return cls(
            append_only=data["append_only"],
            compare=data["compare"],
            critical_path=data["critical_path"],
        )
    
def main():
    task_graph = add_random_weights(get_branching_dag(levels=3, branching_factor=2))
    network = add_random_weights(get_network())
    # schedule_restrictions = {
    #     1: {1, 2, 3},
    #     2: {2, 3},
    #     3: {3}
    # }
    # don't allow any tasks to be scheduled on node 1
    schedule_restrictions = {
        task_name: {0, 1, 2}
        for task_name in task_graph.nodes
    }
    insert_task = ConstrainedGreedyInsert(
        append_only=True,
        compare=lambda network, task_graph, schedule, new, cur: new.end - cur.end if new.name in schedule_restrictions[new.node] else 1e9,
        critical_path=True
    )

    scheduler = ParametricScheduler(
        initial_priority=UpwardRanking(),
        insert_task=insert_task
    )


    schedule = scheduler.schedule(network, task_graph)
    ax = draw_gantt(schedule)
    ax.set_title("Constrained Greedy Insert")
    ax.set_xlabel("Time")
    ax.set_ylabel("Node")
    ax.figure.savefig("constrained_greedy_insert.png")


if __name__ == "__main__":
    main()