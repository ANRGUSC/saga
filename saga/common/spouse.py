from saga.base import Scheduler, Task
from typing import Dict, Hashable, List, Tuple
import networkx as nx

from saga.common.heft import heft_rank_sort

class SpouseScheduler(Scheduler):
    def __init__(self) -> None:
        super().__init__()

    def schedule(self, network: nx.Graph, task_graph: nx.DiGraph) -> Dict[Hashable, List[Task]]:
        schedule_order = heft_rank_sort(network, task_graph)

        schedule: Dict[Hashable, List[Task]] = {node: [] for node in network.nodes}  # Initialize list for each node
        scheduled_tasks: Dict[Hashable, Task] = {}
        for task_name in schedule_order:
            candidate_tasks: List[Tuple[Task, int]] = []
            for node in network.nodes:
                execution_time = task_graph.nodes[task_name]['weight'] / network.nodes[node]['weight']
                input_availablity_time = max([0, *(
                    scheduled_tasks[parent].end  + task_graph.edges[parent, task_name]['weight'] / network.edges[scheduled_tasks[parent].node, node]['weight']
                    for parent in task_graph.predecessors(task_name)
                )])
                    
                # first gap in schedule after availability time that is big enough
                start_time, insert_loc = None, None
                for i, scheduled_task in enumerate(schedule[node]):
                    # see if new task can be inserted before scheduled_task
                    prev_end = 0 if i == 0 else schedule[node][i - 1].end
                    if prev_end < input_availablity_time and max(prev_end, input_availablity_time) + execution_time <= scheduled_task.start:
                        start_time = max(prev_end, input_availablity_time)
                        insert_loc = i
                        break
                if insert_loc is None: # append to end of schedule
                    insert_loc = len(schedule[node])
                    start_time = max(schedule[node][-1].end, input_availablity_time) if schedule[node] else input_availablity_time
                insert_loc = len(schedule[node]) if insert_loc is None else insert_loc
                new_task = Task(node, task_name, start_time, start_time + execution_time)
                candidate_tasks.append((new_task, insert_loc))

            def avg_availablity_on_neighbors(task: Task) -> float:
                # return task.end
                parent_nodes = {
                    scheduled_tasks[parent].node for parent in task_graph.predecessors(task.name)
                    if parent in scheduled_tasks
                }
                return sum(
                    task.end + task_graph.edges[task.name, child]['weight'] / network.edges[task.node, parent_node]['weight']
                    for child in task_graph.successors(task.name)
                    for parent_node in parent_nodes
                )

            # Find node with minimum completion time for the task
            sched_task, insert_loc = min(candidate_tasks, key=lambda task: avg_availablity_on_neighbors(task[0]))
            schedule[sched_task.node].insert(insert_loc, sched_task)
            scheduled_tasks[task_name] = sched_task

        return schedule
                    
                    
                



