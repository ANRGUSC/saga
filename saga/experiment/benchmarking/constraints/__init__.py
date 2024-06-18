from functools import partial
import random
from sched import scheduler
from typing import Any, Callable, Dict, Hashable, List, Optional
from matplotlib import pyplot as plt
from saga.experiment.benchmarking.parametric.components import (
    ArbitraryTopological, CPoPRanking, ParametricScheduler,
    GreedyInsert, InsertTask, GREEDY_INSERT_COMPARE_FUNCS,
    ParametricSufferageScheduler, UpwardRanking, get_insert_loc,
    initial_priority_funcs, insert_funcs
)
from saga.scheduler import Task
from saga.schedulers.parametric import ScheduleType
import networkx as nx
from saga.utils.draw import draw_gantt
import plotly.express as px
import pandas as pd

from saga.utils.random_graphs import add_random_weights, get_branching_dag, get_network
from saga.experiment import resultsdir, outputdir, datadir
from saga.experiment.benchmarking.parametric.analyze import SCHEDULER_RENAMES


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
        # so we need to find the fastest valid node
        if self.critical_path and node is None:
            fastest_node = list(network.nodes)[0]
            for _node in network.nodes:
                fnode_task = Task(fastest_node, task, 0, task_graph.nodes[task]['weight'] / network.nodes[fastest_node]['weight'])
                cnode_task = Task(_node, task, 0, task_graph.nodes[task]['weight'] / network.nodes[_node]['weight'])
                if self._compare(network, task_graph, schedule, fnode_task, cnode_task) < 0:
                    node = _node
                    break

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
    
def example():
    task_graph = add_random_weights(get_branching_dag(levels=3, branching_factor=2))
    network = add_random_weights(get_network())

    # convert task nodes from numbers to letters A, B, C, ...
    task_graph = nx.relabel_nodes(task_graph, {i: chr(i + 65) for i in task_graph.nodes})
    
    schedule_restrictions = {
        "A": {0, 1, 2},
        "B": {1, 2},
        "C": {2},
        "D": {2, 3},
        "E": {3},
        "F": {1, 3},
        "G": {0, 1, 2},
        "H": {1, 2},
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
    
    schedule = scheduler.schedule(network, task_graph)
    
    savedir = outputdir / "constraints"
    savedir.mkdir(parents=True, exist_ok=True)
    ax = draw_gantt(schedule)
    ax.set_title("Constrained Greedy Insert")
    ax.set_xlabel("Time")
    ax.set_ylabel("Node")
    plt.tight_layout()
    ax.figure.savefig(savedir / "task_types_schedule.png")

def experiment_1():
    def compare(network: nx.Graph,
                task_graph: nx.DiGraph,
                schedule: ScheduleType,
                new: Task, cur: Task,
                max_tasks_per_node: int) -> float:
        if len(schedule[new.node]) >= max_tasks_per_node:
            return 1e9
        if len(schedule[cur.node]) >= max_tasks_per_node:
            return -1e9
        return new.end - cur.end
    
    NUM_EVALS = 100
    LEVELS = 3
    BRANCHING_FACTOR = 2
    rows = []
    for max_tasks in range(2, 9):
        for i in range(NUM_EVALS):
            task_graph = add_random_weights(
                get_branching_dag(levels=LEVELS, branching_factor=BRANCHING_FACTOR),
                weight_range=(0.2, 1.0)
            )
            network = add_random_weights(
                get_network(),
                weight_range=(0.2, 1.0)
            )

            # HEFT w/ constraints
            scheduler = ParametricScheduler(
                initial_priority=UpwardRanking(),
                insert_task=ConstrainedGreedyInsert(
                    append_only=False,
                    compare=partial(compare, max_tasks_per_node=max_tasks),
                    critical_path=False
                )
            )
            schedule = scheduler.schedule(network, task_graph)
            makespan = max(task.end for tasks in schedule.values() for task in tasks)
            rows.append({
                "max_tasks_per_node": max_tasks,
                "run": i,
                "makespan": makespan
            })

    df = pd.DataFrame(rows)

    savedir = outputdir / "constraints"
    savedir.mkdir(parents=True, exist_ok=True)

    fig = px.box(
        df,
        x="max_tasks_per_node", y="makespan",
        title="Makespan vs. Max Tasks Per Node",
        template="plotly_white",
        # exclude outliers
        # boxmode="overlay",
        points=False,
        # make black and white
        color_discrete_sequence=["black"],
    )
    # make X axes label "Max Tasks Per Node"
    fig.update_xaxes(title_text="Max Tasks Per Node")
    fig.update_yaxes(title_text="Makespan")
    fig.write_image(savedir / "makespan_vs_max_tasks_per_node.png")
    fig.write_html(savedir / "makespan_vs_max_tasks_per_node.html")

def experiment_2():
    LEVELS = 3
    BRANCHING_FACTOR = 2
    NUM_NODES = 4

    NUM_RUNS = 1000
    CCR = 1
    rows = []
    for run_num in range(NUM_RUNS):
        task_graph = add_random_weights(
            get_branching_dag(levels=LEVELS, branching_factor=BRANCHING_FACTOR),
            weight_range=(0.2, 1.0)
        )
        network = add_random_weights(
            get_network(num_nodes=NUM_NODES),
            weight_range=(0.2, 1.0)
        )
        
        # multiply all task graph edge weights by CCR
        for u, v in task_graph.edges:
            task_graph.edges[u, v]["weight"] *= CCR

        # random constraints allowing each task to be scheduled on at least one node
        schedule_restrictions = {
            task: set(random.sample(network.nodes, random.randint(1, NUM_NODES)))
            for task in task_graph.nodes
        }
        def compare_eft(network: nx.Graph,
                        task_graph: nx.DiGraph,
                        schedule: ScheduleType,
                        new: Task, cur: Task) -> float:
            if new.node in schedule_restrictions[new.name]:
                return 1e9
            elif cur.node in schedule_restrictions[cur.name]:
                return -1e9
            return new.end - cur.end
        
        def compare_est(network: nx.Graph,
                        task_graph: nx.DiGraph,
                        schedule: ScheduleType,
                        new: Task, cur: Task) -> float:
            if new.node in schedule_restrictions[new.name]:
                return 1e9
            elif cur.node in schedule_restrictions[cur.name]:
                return -1e9
            return new.start - cur.start
        
        scheduler_heft = ParametricScheduler(
            initial_priority=UpwardRanking(),
            insert_task=ConstrainedGreedyInsert(
                append_only=False,
                compare=compare_eft,
                critical_path=False
            )
        )

        scheduler_cpop = ParametricScheduler(
            initial_priority=CPoPRanking(),
            insert_task=ConstrainedGreedyInsert(
                append_only=False,
                compare=compare_eft,
                critical_path=True
            )
        )

        scheduler_est = ParametricScheduler(
            initial_priority=ArbitraryTopological(),
            insert_task=ConstrainedGreedyInsert(
                append_only=False,
                compare=compare_est,
                critical_path=False
            )
        )

        def compare_arbitrary(network: nx.Graph,
                              task_graph: nx.DiGraph,
                              schedule: ScheduleType,
                              new: Task, cur: Task) -> float:
            if new.node in schedule_restrictions[new.name]:
                return 1e9
            elif cur.node in schedule_restrictions[cur.name]:
                return -1e9
            return random.random() - 0.5 # return a random number between -0.5 and 0.5
        scheduler_baseline = ParametricScheduler(
            initial_priority=ArbitraryTopological(),
            insert_task=ConstrainedGreedyInsert(
                append_only=False,
                compare=compare_arbitrary,
                critical_path=False
            )
        )

        schedule_heft = scheduler_heft.schedule(network, task_graph)
        schedule_cpop = scheduler_cpop.schedule(network, task_graph)
        schedule_baseline = scheduler_baseline.schedule(network, task_graph)

        makespan_heft = max(task.end for tasks in schedule_heft.values() for task in tasks)
        makespan_cpop = max(task.end for tasks in schedule_cpop.values() for task in tasks)
        makespan_baseline = max(task.end for tasks in schedule_baseline.values() for task in tasks)

        rows.append([run_num, makespan_heft, "HEFT"])
        rows.append([run_num, makespan_baseline, "Baseline"])
        rows.append([run_num, makespan_cpop, "CPoP"])

    df = pd.DataFrame(rows, columns=["run", "makespan", "scheduler"])
    
    savedir = outputdir / "constraints"
    savedir.mkdir(parents=True, exist_ok=True)
    
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
    # make X axes label "Scheduler"
    fig.update_xaxes(title_text="Scheduler")
    fig.update_yaxes(title_text="Makespan")
    fig.write_image(savedir / "makespan_vs_scheduler.png")
    fig.write_html(savedir / "makespan_vs_scheduler.html")

def experiment_full():
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

    arb_scheduler = ParametricScheduler(
        initial_priority=ArbitraryTopological(),
        insert_task=ConstrainedGreedyInsert(
            append_only=True,
            compare=lambda *_: random.random() - 0.5,
            critical_path=False
        )
    )
    schedulers = {"Arbitrary": arb_scheduler, **schedulers}

    NUM_NODES = 4
    LEVELS = 3
    BRANCHING_FACTOR = 2
    NUM_EVALS = 100
    rows = []
    total = len(schedulers) * NUM_EVALS
    count = 0
    for i in range(NUM_EVALS):
        # random constraints allowing each task to be scheduled on at least one node
            
        task_graph = add_random_weights(
            get_branching_dag(levels=LEVELS, branching_factor=BRANCHING_FACTOR),
            weight_range=(0.2, 1.0)
        )
        network = add_random_weights(
            get_network(NUM_NODES),
            weight_range=(0.2, 1.0)
        )
        schedule_restrictions = {
            task: set(random.sample(network.nodes, random.randint(1, NUM_NODES)))
            for task in task_graph.nodes
        }
        task_graph.graph["schedule_restrictions"] = schedule_restrictions
        for scheduler_name, scheduler in schedulers.items():
            count += 1
            print(f"Progress: {count/total*100:.2f}%" + " "*100, end="\r")
            schedule = scheduler.schedule(network, task_graph)
            makespan = max(task.end for tasks in schedule.values() for task in tasks)
            rows.append({
                "scheduler": scheduler_name,
                "problem": i,
                "makespan": makespan
            })

    df = pd.DataFrame(rows)
    savedir = resultsdir / "constraints"
    savedir.mkdir(parents=True, exist_ok=True)
    df.to_csv(savedir / "results.csv", index=False)

def experiment_full_plots():
    df = pd.read_csv(resultsdir / "constraints" / "results.csv")
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
    savedir = outputdir / "constraints"

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
        savedir = outputdir / "constraints"
        # make font larger
        fig.update_layout(font=dict(size=18))
        # set the y-axis range to be the same for all plots
        fig.update_yaxes(range=[lower_threshold, upper_threshold])

        savedir.mkdir(parents=True, exist_ok=True)
        fig.write_image(savedir / f"makespan_vs_scheduler_{i}.png")
        print(f"Saved to {savedir / f'makespan_vs_scheduler_{i}.png'}")
        fig.write_html(savedir / f"makespan_vs_scheduler_{i}.html")
        print(f"Saved to {savedir / f'makespan_vs_scheduler_{i}.html'}")
        figs.append(fig)