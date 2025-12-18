import itertools

from saga import Scheduler, ScheduledTask, TaskGraph, Network, Schedule


class BruteForceScheduler(Scheduler):
    """Brute force scheduler"""

    def schedule(self, network: Network, task_graph: TaskGraph) -> Schedule:
        """Returns the best schedule (minimizing makespan) for a problem
           instance using brute force

        Args:
            network: Network
            task_graph: Task graph

        Returns:
            A dictionary of the schedule
        """
        # get all topological sorts of the task graph
        topological_sorts = list(task_graph.all_topological_sorts())
        # get all valid mappings of the task graph nodes to the network nodes
        mappings = [
            dict(zip(task_graph.tasks, mapping))
            for mapping in itertools.product(
                network.nodes, repeat=len(task_graph.tasks)
            )
        ]

        best_schedule = None
        best_makespan = float("inf")
        for mapping in mappings:
            for top_sort in topological_sorts:
                schedule: Schedule = Schedule(task_graph, network)
                for task in top_sort:
                    node = mapping[task]
                    ready_time = schedule.get_earliest_start_time(
                        task.name, node.name, append_only=True
                    )
                    new_task = ScheduledTask(
                        node=node.name,
                        name=task.name,
                        start=ready_time,
                        end=ready_time + task.cost / node.speed,
                    )
                    schedule.add_task(new_task)

                if schedule.makespan < best_makespan:
                    best_makespan = schedule.makespan
                    best_schedule = schedule

        if best_schedule is None:
            raise RuntimeError("Brute force scheduler failed to find a schedule")
        return best_schedule
