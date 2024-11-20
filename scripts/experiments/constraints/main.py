from copy import deepcopy
from functools import partial
from itertools import product
import random
from typing import Any, Callable, Dict, Hashable, Optional, Tuple
from matplotlib import pyplot as plt
from saga.schedulers.parametric.components import (
    ArbitraryTopological, CPoPRanking, ParametricScheduler, GreedyInsert, InsertTask,
    ParametricSufferageScheduler, UpwardRanking, get_insert_loc
)
from saga.schedulers.smt import SMTScheduler
from saga.schedulers.brute_force import BruteForceScheduler
from saga.scheduler import Task
from saga.schedulers.parametric import ScheduleType
from saga.utils.tools import validate_simple_schedule
import networkx as nx
from saga.utils.draw import draw_gantt, draw_network, draw_task_graph
import plotly.express as px
import pandas as pd
import pathlib
import seaborn as sns

thisdir = pathlib.Path(__file__).parent
resultsdir = thisdir / "results"
outputdir = thisdir / "output"

SCHEDULER_RENAMES = {
    "Cpop": "CPoP",
    "Heft": "HEFT",
}

# Enable LaTeX fonts
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

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

        # If critical path is enabled and node is None, then we need to find the fastest node
        # but with the constraint-based comparison functions some of the nodes may be invalid
        if self.critical_path and node is None:
            fastest_node = min(network.nodes, key=lambda node: network.nodes[node]['weight'])
            schedule_restrictions = task_graph.graph.get("schedule_restrictions", {})
            if fastest_node not in schedule_restrictions[task]:
                node = fastest_node

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
    
def generate_instance(workflow_type: str) -> Tuple[nx.Graph, nx.DiGraph]:
    """Generate a random problem instance based on the scenarios described in Mattia's email.

    Task Graph:
        Two types of task graphs. The first is linear and the second is tree-like.

        Linear: Camera -> License Plate Recognition (fast) -> Message Broker -> License Plate Recognition (slow)
        Tree-like: 
            Sensor 1 -> Data Fusion 1
            Sensor 2 -> Data Fusion 1
            Sensor 3 -> Data Fusion 2
            Sensor 4 -> Data Fusion 2
            Data Fusion 1 -> Data Fusion 3
            Data Fusion 2 -> Data Fusion 3

    Network:
        LAN (Bandwidth/Packet Loss/Latency): No restrictions on bandwidth, packet loss, or latency.
        Best: 4 Mbit/s, 5% packet loss, 200 ms latency
        Average: 2 Mbit/s, 10% packet loss, 400 ms latency
        Worst: 0.65 kbit/s, 15% packet loss, 3000 ms latency

        We'll just assume there are 7 nodes. This is 42 communication links which we can set randomly as 
        best, average, or worst case.
    
    Returns:
        Tuple[nx.DiGraph, nx.Graph]: The task graph and network.
    """

    workflow_types = {'chain', 'heirarchy'}
    if workflow_type not in workflow_types:
        raise ValueError(f"Invalid workflow type: {workflow_type}. Must be one of {workflow_types}")

    task_graph = nx.DiGraph()
    if workflow_type == 'chain':
        task_graph.add_node("Camera", weight=2.0)
        task_graph.add_node("LPR (stream)", weight=1.0)
        task_graph.add_node("Broker", weight=1.0)
        task_graph.add_node("LPR (model)", weight=5.0)

        task_graph.add_edge("Camera", "LPR (stream)", weight=1.0)
        task_graph.add_edge("LPR (stream)", "Broker", weight=1.0)
        task_graph.add_edge("Broker", "LPR (model)", weight=1.0)
    elif workflow_type == 'heirarchy':
        task_graph.add_node("Sensor 1", weight=1.0)
        task_graph.add_node("Sensor 2", weight=1.0)
        task_graph.add_node("Sensor 3", weight=1.0)
        task_graph.add_node("Sensor 4", weight=1.0)
        task_graph.add_node("Fusion 1", weight=2.0)
        task_graph.add_node("Fusion 2", weight=2.0)
        task_graph.add_node("Fusion 3", weight=4.0)

        task_graph.add_edge("Sensor 1", "Fusion 1", weight=1.0)
        task_graph.add_edge("Sensor 2", "Fusion 1", weight=1.0)
        task_graph.add_edge("Sensor 3", "Fusion 2", weight=1.0)
        task_graph.add_edge("Sensor 4", "Fusion 2", weight=1.0)
        task_graph.add_edge("Fusion 1", "Fusion 3", weight=1.0)
        task_graph.add_edge("Fusion 2", "Fusion 3", weight=1.0)

    # add src and sink nodes
    task_graph.add_node("src", weight=1e-9)
    task_graph.add_node("sink", weight=1e-9)
    for node in task_graph.nodes:
        if node != "src" and task_graph.in_degree(node) == 0:
            task_graph.add_edge("src", node, weight=1e-9)
        if node != "sink" and task_graph.out_degree(node) == 0:
            task_graph.add_edge(node, "sink", weight=1e-9)
    
    network = nx.Graph()
    bandwidths = [4, 2, 0.65]
    for i in range(7):
        network.add_node(i, weight=1.0)
    for u, v in product(network.nodes, network.nodes):
        if u != v:
            network.add_edge(u, v, weight=random.choice(bandwidths))
        else:
            network.add_edge(u, v, weight=1e9)

    return network, task_graph

def get_makespan(network: nx.Graph, task_graph: nx.DiGraph, schedule: ScheduleType) -> float:
    return max(task.end for tasks in schedule.values() for task in tasks)

def get_throughput(network: nx.Graph, task_graph: nx.DiGraph, schedule: ScheduleType) -> float:
    tasks = {task.name: task for _, tasks in schedule.items() for task in tasks}
    task_bottleneck = max(sum(task.end - task.start for task in tasks) for node, tasks in schedule.items())
    comm_bottleneck = max(
        sum(
            task_graph.edges[parent, child]["weight"] / network.edges[tasks[child].node, tasks[child].node]["weight"]
            for parent, child in task_graph.edges
            if parent in tasks and child in tasks and tasks[parent].node == src and tasks[child].node == dst
        )
        for src, dst in network.edges
    )
    return max(task_bottleneck, comm_bottleneck)

def compare_etf(network: nx.Graph, task_graph: nx.DiGraph, schedule: ScheduleType, new: Task, cur: Task) -> float:
    schedule_restrictions = task_graph.graph.get("schedule_restrictions", {})
    if new.node in schedule_restrictions[new.name]:
        return 1e9
    elif cur.node in schedule_restrictions[cur.name]:
        return -1e9
    return new.end - cur.end

def compare_est(network: nx.Graph, task_graph: nx.DiGraph, schedule: ScheduleType, new: Task, cur: Task) -> float:
    schedule_restrictions = task_graph.graph.get("schedule_restrictions", {})
    if new.node in schedule_restrictions[new.name]:
        return 1e9
    elif cur.node in schedule_restrictions[cur.name]:
        return -1e9
    return new.start - cur.start

def compare_quickest(network: nx.Graph, task_graph: nx.DiGraph, schedule: ScheduleType, new: Task, cur: Task) -> float:
    schedule_restrictions = task_graph.graph.get("schedule_restrictions", {})
    if new.node in schedule_restrictions[new.name]:
        return 1e9
    elif cur.node in schedule_restrictions[cur.name]:
        return -1e9
    return (new.end - new.start) - (cur.end - cur.start)

def compare_arbitrary(network: nx.Graph, task_graph: nx.DiGraph, schedule: ScheduleType, new: Task, cur: Task) -> float:
    schedule_restrictions = task_graph.graph.get("schedule_restrictions", {})
    if new.node in schedule_restrictions[new.name]:
        return 1e9
    elif cur.node in schedule_restrictions[cur.name]:
        return -1e9
    return 0

def compare_max_proc_time(network: nx.Graph, task_graph: nx.DiGraph, schedule: ScheduleType, new: Task, cur: Task) -> float:
    schedule_restrictions = task_graph.graph.get("schedule_restrictions", {})
    if new.node in schedule_restrictions[new.name]:
        return 1e9
    elif cur.node in schedule_restrictions[cur.name]:
        return -1e9
    # This is the "simple_greedy" approach from https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=5161045
    new_node_tp = max([
        # runtime of new task on new node
        task_graph.nodes[new.name]["weight"] / network.nodes[new.node]["weight"],
        # runtime of previously scheduled tasks on new node
        *[task_graph.nodes[task.name]["weight"] / network.nodes[task.node]["weight"] for task in schedule[new.node]]
    ])
    cur_node_tp = max([
        # runtime of new task on current node
        task_graph.nodes[cur.name]["weight"] / network.nodes[cur.node]["weight"],
        # runtime of previously scheduled tasks on current node
        *[task_graph.nodes[task.name]["weight"] / network.nodes[task.node]["weight"] for task in schedule[cur.node]]
    ])
    return new_node_tp - cur_node_tp

def compare_throughput(network: nx.Graph, task_graph: nx.DiGraph, schedule: ScheduleType, new: Task, cur: Task) -> float:
    schedule_restrictions = task_graph.graph.get("schedule_restrictions", {})
    if new.node in schedule_restrictions[new.name]:
        return 1e9
    elif cur.node in schedule_restrictions[cur.name]:
        return -1e9
    schedule_1 = deepcopy(schedule)
    schedule_2 = deepcopy(schedule)
    schedule_1[new.node].append(new) # doesn't matter if this is valid or not because it's just for computing throughput
    schedule_2[cur.node].append(cur) # doesn't matter if this is valid or not because it's just for computing throughput

    return get_throughput(network, task_graph, schedule_1) - get_throughput(network, task_graph, schedule_2)

def example(workflow_type: str = 'heirarchy',
            num_examples: int = 1,
            savedir: pathlib.Path = outputdir,
            mode: str = "makespan",
            filetype: str = "png"):
    """Produce an example where the HEFT scheduling algorithm doesn't perform well"""

    worst_network = None
    worst_task_graph = None
    worst_schedule = None
    worst_smt_schedule = None
    worst_qr = float("-inf")

    savedir.mkdir(parents=True, exist_ok=True)

    for i in range(num_examples):
        print(f"Example {i+1}/{num_examples}")
        network, task_graph = generate_instance(workflow_type)
        
        schedule_restrictions = {
            # random constraints
            task: set(random.sample(list(network.nodes), random.randint(1, len(network.nodes)-1)))
            for task in task_graph.nodes
        }
        def compare(network: nx.Graph,
                    task_graph: nx.DiGraph,
                    schedule: ScheduleType,
                    new: Task, cur: Task) -> float:
            if new.node in schedule_restrictions[new.name]:
                return 1e9 # return high number, indicating new is bad
            elif cur.node in schedule_restrictions[cur.name]:
                return -1e9 # return low number, indicating cur is bad
            return new.end - cur.end # return the difference in end times

        # HEFT w/ constraints
        scheduler = ParametricScheduler(
            initial_priority=UpwardRanking(),
            insert_task=ConstrainedGreedyInsert(
                append_only=False,
                compare=compare,
                critical_path=False
            )
        )
        
        schedule = scheduler.schedule(network.copy(), task_graph.copy())

        # verify schedule satisfies constraints
        if not all(task.node not in schedule_restrictions[task.name] for tasks in schedule.values() for task in tasks):
            raise ValueError("Schedule does not satisfy constraints")
        
        # Verify schedule is valid
        validate_simple_schedule(network, task_graph, schedule)

        # SMT schedule
        task_graph.graph["schedule_restrictions"] = schedule_restrictions
        smt_scheduler = SMTScheduler(epsilon=0.1, mode=mode)
        smt_schedule = smt_scheduler.schedule(network, task_graph)

        # verify schedule satisfies constraints
        if not all(task.node not in schedule_restrictions[task.name] for tasks in smt_schedule.values() for task in tasks):
            raise ValueError("Schedule does not satisfy constraints")
        
        # Verify schedule is valid
        validate_simple_schedule(network, task_graph, smt_schedule)

        if mode == "makespan":
            quality = get_makespan(network, task_graph, schedule)
            smt_quality = get_makespan(network, task_graph, smt_schedule)
        elif mode == "throughput":
            quality = get_throughput(network, task_graph, schedule)
            smt_quality = get_throughput(network, task_graph, smt_schedule)
        qr = quality / smt_quality
        if qr > worst_qr:
            worst_qr = qr
            worst_network = network
            worst_task_graph = task_graph
            worst_schedule = schedule
            worst_smt_schedule = smt_schedule
    
    savedir.mkdir(parents=True, exist_ok=True)

    max_makespan = max(
        max(task.end for tasks in worst_schedule.values() for task in tasks),
        max(task.end for tasks in worst_smt_schedule.values() for task in tasks)
    )

    worst_task_graph = worst_task_graph.subgraph([node for node in worst_task_graph.nodes if node not in {"src", "sink"}])
    # rename nodes in \text{} to enable LaTeX rendering
    # worst_task_graph = nx.relabel_nodes(worst_task_graph, {node: f"node}}}" for node in worst_task_graph.nodes})
    ax = draw_task_graph(
        worst_task_graph,
        figsize=(10, 10),
        use_latex=True,
        draw_edge_weights=False,
        draw_node_weights=False
    )
    plt.tight_layout()
    ax.figure.savefig(savedir / f"task_graph.{filetype}")

    ax = draw_network(
        worst_network,
        figsize=(10, 10),
        use_latex=True,
        draw_edge_weights=False,
        draw_node_weights=False,
        draw_colors=False
    )
    plt.tight_layout()
    ax.figure.savefig(savedir / f"network.{filetype}")

    ax = draw_gantt(worst_schedule, figsize=(15, 6), use_latex=True, xmax=max_makespan)
    ax.set_xlabel("Time")
    ax.set_ylabel("Node")
    plt.tight_layout()
    ax.figure.savefig(savedir / f"schedule.{filetype}")

    ax = draw_gantt(worst_smt_schedule, figsize=(15, 6), use_latex=True, xmax=max_makespan)
    ax.set_xlabel("Time")
    ax.set_ylabel("Node")
    plt.tight_layout()
    ax.figure.savefig(savedir / f"smt_schedule.{filetype}")


COMPARE_FUNCS = {
    "EFT": compare_etf,
    "EST": compare_est,
    "Qck": compare_quickest,
    "Arb": compare_arbitrary,
    "MET": compare_max_proc_time,
    "Thp": compare_throughput
}

PRIORITY_FUNCS = {
    "Upwd": UpwardRanking(),
    "CPoP": CPoPRanking(),
    "ArbT": ArbitraryTopological()
}

def experiment(workflow_type: str,
               savedir: pathlib.Path,
               mode: str = "makespan"):
    savedir.mkdir(parents=True, exist_ok=True)

    schedulers: Dict[str, ParametricScheduler] = {}
    for append_only, compare_func, critical_path in product([True, False], COMPARE_FUNCS.keys(), [False, True]):
        for intial_priority_name, initial_priority_func in PRIORITY_FUNCS.items():
            name = f"{compare_func}_{intial_priority_name}_{'App' if append_only else 'Ins'}{'_CP' if critical_path else ''}"
            reg_scheduler = ParametricScheduler(
                initial_priority=initial_priority_func,
                insert_task=ConstrainedGreedyInsert(
                    append_only=append_only,
                    compare=COMPARE_FUNCS[compare_func],
                    critical_path=critical_path
                )
            )
            reg_scheduler.name = name
            schedulers[reg_scheduler.name] = reg_scheduler

            sufferage_scheduler = ParametricSufferageScheduler(
                scheduler=reg_scheduler,
                top_n=2
            )
            sufferage_scheduler.name = f"{reg_scheduler.name}_Suff"
            schedulers[sufferage_scheduler.name] = sufferage_scheduler

    def compare_arb(network: nx.Graph,
                    task_graph: nx.DiGraph,
                    schedule: ScheduleType,
                    new: Task, cur: Task) -> float:
        schedule_restrictions = task_graph.graph.get("schedule_restrictions", {})
        if new.node in schedule_restrictions[new.name]:
            return 1e9
        elif cur.node in schedule_restrictions[cur.name]:
            return -1e9
        return 0

    arb_scheduler = ParametricScheduler(
        initial_priority=ArbitraryTopological(),
        insert_task=ConstrainedGreedyInsert(
            append_only=True,
            compare=compare_arb,
            critical_path=False
        )
    )

    smt_scheduler = SMTScheduler(epsilon=0.1, mode=mode)
    schedulers = {"Arbitrary": arb_scheduler, "SMT": smt_scheduler, **schedulers}

    NUM_EVALS = 100
    rows = []
    total = len(schedulers) * NUM_EVALS
    count = 0
    for i in range(NUM_EVALS):
        # random constraints allowing each task to be scheduled on at least one node
        network, task_graph = generate_instance(workflow_type)
        schedule_restrictions = {
            task: set(random.sample(list(network.nodes), random.randint(1, len(network.nodes))-1))
            for task in task_graph.nodes
        }
        task_graph.graph["schedule_restrictions"] = schedule_restrictions
        for scheduler_name, scheduler in schedulers.items():
            count += 1
            print(f"Progress: {count/total*100:.2f}%" + " "*100) #, end="\r")
            start = pd.Timestamp.now()
            schedule = scheduler.schedule(network.copy(), task_graph.copy())
            dt = pd.Timestamp.now() - start

            # verify schedule satisfies constraints
            if not all(task.node not in schedule_restrictions[task.name] for tasks in schedule.values() for task in tasks):
                raise ValueError(f"Schedule does not satisfy constraints for scheduler {scheduler_name}")
            
            # Verify schedule is valid
            try:
                validate_simple_schedule(network, task_graph, schedule)
            except Exception as e:
                print(f"Error validating schedule for scheduler {scheduler_name}: {e}")
                raise e

            quality = get_makespan(network, task_graph, schedule) if mode == "makespan" else get_throughput(network, task_graph, schedule)
            rows.append({
                "scheduler": scheduler_name,                
                "problem": i,
                "quality": quality,
                "time": dt.total_seconds()
            })

    df = pd.DataFrame(rows)

    for key, value in SCHEDULER_RENAMES.items():
        df["scheduler"] = df["scheduler"].str.replace(key, value, regex=True)

    df.to_csv(savedir / "data.csv", index=False)

def generate_plots(savedir: pathlib.Path, mode: str = "makespan", filetype: str = "png"):
    df = pd.read_csv(savedir / "data.csv")
    arb_mean = df[df["scheduler"] == "Arbitrary"]["quality"].median()
    quality_label = "Makespan" if mode == "makespan" else "Throughput"

    min_quality = df["quality"].min()
    max_quality = df["quality"].max()

    print(min_quality, max_quality)

    for compare_func in COMPARE_FUNCS.keys():
        _df = df[(df["scheduler"].str.startswith(compare_func) | df["scheduler"].str.contains("SMT")) | (df["scheduler"] == "Arbitrary")]

        plt.figure(figsize=(12, 6))
        sns.boxplot(data=_df, x="scheduler", y="quality", color="gray", showfliers=False)
        plt.axhline(y=arb_mean, linestyle="--", color="black", label="Arbitrary Median")
        plt.xlabel("Scheduler", fontsize=14)
        plt.ylabel(quality_label, fontsize=14)
        padding = (max_quality - min_quality) * 0.1
        plt.ylim(min_quality - padding, max_quality + padding)
        plt.xticks(rotation=45, ha="right", fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend(fontsize=12)
        plt.tight_layout()

        output_path = savedir / f"quality_{compare_func}.{filetype}"
        plt.savefig(output_path)
        plt.close()
        print(f"Saved to {output_path}")

    # Plot the time taken for each scheduler
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x="scheduler", y="time", color="gray", showfliers=False)
    plt.xlabel(r"Scheduler", fontsize=14)
    plt.ylabel(r"Time (s)", fontsize=14)
    plt.xticks(rotation=45, ha="right", fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()

    output_path = savedir / f"time_vs_scheduler.{filetype}"
    plt.savefig(output_path)
    plt.close()
    print(f"Saved to {output_path}")

    # On average, how many times slower is SMT than other schedulers
    smt_times = df[df["scheduler"] == "SMT"]["time"]
    other_times = df[df["scheduler"] != "SMT"]["time"]
    avg_smt_time = smt_times.mean()
    avg_other_time = other_times.mean()
    print(f"SMT average time / Other average time: {avg_smt_time / avg_other_time:.2f}")

def main():
    filetype = "pdf"

    for mode in ["makespan", "throughput"]:
        print(f"Mode: {mode}")
        for workflow_type in ['chain', 'heirarchy']:
            savedir = resultsdir / workflow_type
            example(workflow_type=workflow_type, num_examples=20, savedir=savedir / mode / "example", mode=mode, filetype=filetype)
            # experiment(workflow_type=workflow_type, savedir=savedir / mode / "comparison", mode=mode)
            # generate_plots(savedir=savedir / mode / "comparison", mode=mode, filetype=filetype)

if __name__ == "__main__":
    main()