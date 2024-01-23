# different ranking functions
# different selection metrics (INSERT_EARLIEST_START, APPEND_EARLIEST_START, INSERT_EARLIEST_FINISH, APPEND_EARLIEST_FINISH)
# lookahead length k (ignore for now)




def schedule(self, network: nx.Graph,
             task_graph: nx.DiGraph,
             get_priority_queue: Callable[[nx.Graph, nx.DiGraph, Optional[PriorityQueue], Optional[Dict[Hashable, List[Task]]]], PriorityQueue],
             insert_task: Callable[[nx.Graph, nx.DiGraph, Dict[Hashable, List[Task]], Hashable], None]
    ) -> Dict[Hashable, List[Task]]:
        """
        Schedule a task graph onto a network by parameters given in the constructor.

        Args:
            network (nx.Graph): The network to schedule onto.
            task_graph (nx.DiGraph): The task graph to schedule.

        Returns:
            Dict[Hashable, List[Task]]: The schedule.
        """
        schedule = {node: [] for node in network.nodes}
        priority_queue = get_priority_queue(network, task_graph)
        while priority_queue:
            task_name = priority_queue.pop()
            insert_task(network, task_graph, schedule, task_name)
            priority_queue = get_priority_queue(network, task_graph, priority_queue, schedule)

        return schedule

# HEFT
schedule(network, task_graph, upward_rank, 