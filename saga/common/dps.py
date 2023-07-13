from typing import Dict, Hashable, List, Tuple
import networkx as nx

from ..base import Scheduler, Task
from ..utils.tools import check_instance_simple

class DpsScheduler(Scheduler):
    def __init__(self):
        super(DpsScheduler, self).__init__()

    def schedule(self, network: nx.Graph, task_graph: nx.DiGraph) -> Dict[Hashable, List[Task]]:
        check_instance_simple(network, task_graph)

        schedule: Dict[Hashable, List[Task]] = {}
        scheduled_tasks: Dict[Hashable, Task] = {}
        
        def find_max_TEC(task: Hashable, network: nx.Graph) -> float:
            return max(task_graph.nodes[task]['weight'] / network.nodes[node]['weight'] for node in network.nodes)

        def TL(task: Hashable, network: nx.Graph) -> float:
            if(len(task_graph.predecessors(task_graph.nodes[task]))) == 0:
                return 0
            return max(TL(pred) + find_max_TEC(task, network) + task_graph.edges[pred, task]['weight'] for pred in task_graph.predecessors(task_graph.nodes[task]))

        def BL(task:Hashable, network: nx.Graph) -> float:
            if(len(task_graph.successors(task_graph.nodes[task]))) == 0:
                return find_max_TEC(task, network)
            return max(BL(succ) + find_max_TEC(task, network) + task_graph.edges[task, succ]['weight'] for succ in task_graph.successors)
        
        def PRIORITY(TL: float, BL: float) -> float:
            return BL - TL

        def COMMTIME(task1: Hashable, task2: Hashable, node1: Hashable, node2: Hashable) -> float:
            return task_graph.edges[task1, task2]['weight'] / network.edges[node1, node2]['weight']

        def EAT(node: Hashable) -> float:
            eat = schedule[node][-1].end if schedule.get(node) else 0
            return eat
        
        def FAT(task: Hashable, node: Hashable) -> float:
            fat = 0 if task_graph.in_degree(task) <= 0 else max([
                scheduled_tasks[pred_task].end +
                COMMTIME(pred_task, task, scheduled_tasks[pred_task].node, node)
                for pred_task in task_graph.predecessors(task)
            ]) 
            return fat

        def EET(task: Hashable, node: Hashable) -> float:
            return task_graph.nodes[task]['weight'] / network.nodes[node]['weight']

        def ECT(task: Hashable, node: Hashable) -> float:
            return EET(task, node) + max(EAT(node), FAT(task, node))

        def EFT(task:Hashable, network: nx.Graph) -> Tuple[float,Hashable]:
            if(len(task_graph.predecessors(task_graph.nodes[task]))) == 0:
                return min(Tuple[task_graph.nodes[task]['weight'] / network.nodes[node]['weight'],node] for node in network.nodes)
            else :
                return min(Tuple[max(EFT(pred) + task_graph.edges[pred,task]for pred in task_graph.predecessors(task_graph.nodes[task]))+ task_graph.nodes[task]['weight'] / network.nodes[node]['weight'],node] for node in network.nodes)

        task_list = list(task_graph.nodes)
        task_list.sort(key=lambda task: PRIORITY(TL(task,network), BL(task,network)), reverse=True)
        
        processors = list(network.nodes())

        ready_list = task_list.copy()
        
        while ready_list:
            task_index = ready_list[0]['name']
            task = nx.get_node_attributes(task_graph,{'name':task_index})
            processor = EFT(task,network)[1]
            schedule.setdefault(processor, [])
            new_task = Task(node=processor,
            name=task,
            start=max(EAT(processor),FAT(task, processor)),
            end=ECT(task, processor)
            )
            schedule[processor].append(new_task)
            ready_list.remove(task)
            ready_list.sort(key=lambda task: PRIORITY(TL(task,network), BL(task,network)), reverse=True)


        return schedule