from functools import partial
from itertools import product
import random
from typing import Any, Callable, Dict, Hashable, Optional, Tuple
from matplotlib import pyplot as plt
from saga.schedulers.parametric.components import (
    ArbitraryTopological, CPoPRanking, ParametricScheduler,
    GreedyInsert, InsertTask, GREEDY_INSERT_COMPARE_FUNCS,
    ParametricSufferageScheduler, UpwardRanking, get_insert_loc,
    initial_priority_funcs, insert_funcs
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

thisdir = pathlib.Path(__file__).parent
resultsdir = thisdir / "results"
outputdir = thisdir / "output"

SCHEDULER_RENAMES = {
    "Cpop": "CPoP",
    "Heft": "HEFT",
}

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

def example(workflow_type: str = 'heirarchy',
            num_examples: int = 1,
            savedir: pathlib.Path = outputdir):
    worst_network = None
    worst_task_graph = None
    worst_schedule = None
    worst_smt_schedule = None
    worst_mr = float("-inf")

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

        # # HEFT w/ constraints
        scheduler = ParametricScheduler(
            initial_priority=UpwardRanking(),
            insert_task=ConstrainedGreedyInsert(
                append_only=False,
                compare=compare,
                critical_path=False
            )
        )

        # # CPoP w/ constraints
        # scheduler = ParametricScheduler(
        #     initial_priority=CPoPRanking(),
        #     insert_task=ConstrainedGreedyInsert(
        #         append_only=False,
        #         compare=compare,
        #         critical_path=True
        #     )
        # )
        
        schedule = scheduler.schedule(network.copy(), task_graph.copy())

        # verify schedule satisfies constraints
        if not all(task.node not in schedule_restrictions[task.name] for tasks in schedule.values() for task in tasks):
            raise ValueError("Schedule does not satisfy constraints")
        
        # Verify schedule is valid
        validate_simple_schedule(network, task_graph, schedule)

        # SMT schedule
        task_graph.graph["schedule_restrictions"] = schedule_restrictions
        smt_scheduler = SMTScheduler(epsilon=0.1)
        smt_schedule = smt_scheduler.schedule(network, task_graph)

        # verify schedule satisfies constraints
        if not all(task.node not in schedule_restrictions[task.name] for tasks in smt_schedule.values() for task in tasks):
            raise ValueError("Schedule does not satisfy constraints")
        
        # Verify schedule is valid
        validate_simple_schedule(network, task_graph, smt_schedule)

        makespan = max(task.end for tasks in schedule.values() for task in tasks)
        smt_makespan = max(task.end for tasks in smt_schedule.values() for task in tasks)
        mr = makespan / smt_makespan
        if mr > worst_mr:
            worst_mr = mr
            worst_network = network
            worst_task_graph = task_graph
            worst_schedule = schedule
            worst_smt_schedule = smt_schedule
    
    savedir.mkdir(parents=True, exist_ok=True)

    ax = draw_task_graph(worst_task_graph, figsize=(10, 10))
    ax.set_title("Task Graph")
    plt.tight_layout()
    ax.figure.savefig(savedir / "task_graph.png")

    ax = draw_network(worst_network, figsize=(10, 10))
    ax.set_title("Network")
    plt.tight_layout()
    ax.figure.savefig(savedir / "network.png")

    ax = draw_gantt(worst_schedule, figsize=(15, 6))
    ax.set_title("Constrained Greedy Insert")
    ax.set_xlabel("Time")
    ax.set_ylabel("Node")
    plt.tight_layout()
    ax.figure.savefig(savedir / "schedule.png")

    ax = draw_gantt(worst_smt_schedule, figsize=(15, 6))
    ax.set_title("SMT Schedule")
    ax.set_xlabel("Time")
    ax.set_ylabel("Node")
    plt.tight_layout()
    ax.figure.savefig(savedir / "smt_schedule.png")

def experiment(workflow_type: str,
               savedir: pathlib.Path):
    savedir.mkdir(parents=True, exist_ok=True)
    
    schedulers: Dict[str, ParametricScheduler] = {}
    for name, insert_task in insert_funcs.items():
        for intial_priority_name, initial_priority_func in initial_priority_funcs.items():
            def compare(network: nx.Graph,
                        task_graph: nx.DiGraph,
                        schedule: ScheduleType,
                        new: Task, cur: Task) -> float:
                schedule_restrictions = task_graph.graph.get("schedule_restrictions", {})
                if new.node in schedule_restrictions[new.name]:
                    return 1e9
                elif cur.node in schedule_restrictions[cur.name]:
                    return -1e9
                return insert_task._compare(new, cur)
            reg_scheduler = ParametricScheduler(
                initial_priority=initial_priority_func,
                insert_task=ConstrainedGreedyInsert(
                    append_only=insert_task.append_only,
                    compare=compare,
                    critical_path=insert_task.critical_path
                )
            )
            reg_scheduler.name = f"{name}_{intial_priority_name}"
            schedulers[reg_scheduler.name] = reg_scheduler

            sufferage_scheduler = ParametricSufferageScheduler(
                scheduler=reg_scheduler,
                top_n=2
            )
            sufferage_scheduler.name = f"{reg_scheduler.name}_Sufferage"
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

    bf_scheduler = BruteForceScheduler()
    smt_scheduler = SMTScheduler(epsilon=0.1)

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

            makespan = max(task.end for tasks in schedule.values() for task in tasks)
            rows.append({
                "scheduler": scheduler_name,
                "problem": i,
                "makespan": makespan,
                "time": dt.total_seconds()
            })

    df = pd.DataFrame(rows)

    for key, value in SCHEDULER_RENAMES.items():
        df["scheduler"] = df["scheduler"].str.replace(key, value, regex=True)

    arb_mean = df[df["scheduler"] == "Arbitrary"]["makespan"].median()

    fig = px.box(
        df,
        x="scheduler", y="makespan",
        title="Makespan vs. Scheduler",
        template="plotly_white",
        # exclude outliers
        # boxmode="overlay",
        points=False,
        # make black and white
        color_discrete_sequence=["black"],
    )
    # draw horizontal line at the mean of the Arbitrary scheduler
    fig.add_hline(y=arb_mean, line_dash="dot", line_color="black")

    # make the plot really wide to accommodate the 72 schedulers
    fig.update_layout(width=3000)
    fig.update_xaxes(title_text="Scheduler")
    fig.update_yaxes(title_text="Makespan")

    savedir.mkdir(parents=True, exist_ok=True)
    fig.write_image(savedir / "makespan_vs_scheduler.png")
    print(f"Saved to {savedir / 'makespan_vs_scheduler.png'}")
    fig.write_html(savedir / "makespan_vs_scheduler.html")
    print(f"Saved to {savedir / 'makespan_vs_scheduler.html'}")

    num_plots = 4
    schedulers_without_arb = [scheduler for scheduler in df["scheduler"].unique() if scheduler != "Arbitrary"]
    scheduler_groups = [schedulers_without_arb[i::num_plots] for i in range(num_plots)]
    upper_threshold = df["makespan"].max() + 1
    lower_threshold = df["makespan"].min() - 1
    figs = []
    for i in range(num_plots):
        _df = df[df["scheduler"].isin({"Arbitrary", *scheduler_groups[i]})]
        fig = px.box(
            _df,
            x="scheduler", y="makespan",
            title="Makespan vs. Scheduler",
            template="plotly_white",
            # exclude outliers
            # boxmode="overlay",
            points=False,
            # make black and white
            color_discrete_sequence=["black"],
        )
        # draw horizontal line at the mean of the Arbitrary scheduler
        fig.add_hline(y=arb_mean, line_dash="dot", line_color="black")

        fig.update_layout(width=1500, height=800)
        fig.update_xaxes(title_text="Scheduler")
        fig.update_yaxes(title_text="Makespan")
        # make font larger
        fig.update_layout(font=dict(size=18))
        # set the y-axis range to be the same for all plots
        fig.update_yaxes(range=[lower_threshold, upper_threshold])

        fig.write_image(savedir / f"makespan_vs_scheduler_{i}.png")
        print(f"Saved to {savedir / f'makespan_vs_scheduler_{i}.png'}")
        fig.write_html(savedir / f"makespan_vs_scheduler_{i}.html")
        print(f"Saved to {savedir / f'makespan_vs_scheduler_{i}.html'}")
        figs.append(fig)

    # generate plot with only Arbitrary and main schedulers: CPoP, HEFT, Sufferage, MET, MCT
    main_schedulers = ["CPoP", "HEFT", "Sufferage", "MET", "MCT"]
    _df = df[df["scheduler"].isin({"Arbitrary", *main_schedulers})]
    fig = px.box(
        _df,
        x="scheduler", y="makespan",
        template="plotly_white",
        # exclude outliers
        # boxmode="overlay",
        points=False,
        # make black and white
        color_discrete_sequence=["black"],
    )
    # draw horizontal line at the mean of the Arbitrary scheduler
    fig.add_hline(y=arb_mean, line_dash="dot", line_color="black")

    fig.update_layout(width=1500, height=800)
    fig.update_xaxes(title_text="Scheduler")
    fig.update_yaxes(title_text="Makespan")
    fig.update_layout(font=dict(size=28, family="serif"))

    savedir.mkdir(parents=True, exist_ok=True)
    fig.write_image(savedir / f"constraints.pdf")
    print(f"Saved to {savedir / 'constraints.pdf'}")


    # Plot the time taken for each scheduler
    fig = px.box(
        df,
        x="scheduler", y="time",
        title="Time vs. Scheduler",
        template="plotly_white",
        # exclude outliers
        # boxmode="overlay",
        points=False,
        # make black and white
        color_discrete_sequence=["black"],
    )
    fig.update_layout(width=1500, height=800)
    fig.update_xaxes(title_text="Scheduler")
    fig.update_yaxes(title_text="Time (s)")
    fig.write_image(savedir / "time_vs_scheduler.png")


    # On average how many times slower is SMT than other schedulers
    smt_times = df[df["scheduler"] == "SMT"]["time"]
    other_times = df[df["scheduler"] != "SMT"]["time"]
    avg_smt_time = smt_times.mean()
    avg_other_time = other_times.mean()
    print(f"SMT average time / Other average time: {avg_smt_time / avg_other_time:.2f}")

def main():
    for workflow_type in ['chain', 'heirarchy']:
        savedir = resultsdir / workflow_type
        example(workflow_type=workflow_type, num_examples=20, savedir=savedir / "example")
        # experiment(workflow_type=workflow_type, savedir=savedir / "comparison")

if __name__ == "__main__":
    main()