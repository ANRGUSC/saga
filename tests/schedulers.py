import logging
import pprint
import traceback
from typing import Callable, Dict, List, Tuple

from saga.base import Scheduler, Task
import networkx as nx
import pathlib
from saga.common.cpop import CPOPScheduler
from saga.common.fastest_node import FastestNodeScheduler
from saga.common.heft import HeftScheduler
from saga.common.brute_force import BruteForceScheduler
from saga.stochastic.sheft import SheftScheduler
from saga.stochastic.stoch_heft import StochHeftScheduler
from saga.stochastic.improved_sheft import ImprovedSheftScheduler
from saga.utils.draw import draw_gantt, draw_network, draw_task_graph
from saga.utils.tools import validate_simple_schedule

from random_graphs import (
    get_diamond_dag, get_chain_dag, get_fork_dag, get_branching_dag, 
    get_network, add_random_weights, add_rv_weights
)

thisdir = pathlib.Path(__file__).resolve().parent

class ListHandler(logging.Handler):
    def __init__(self, log_list):
        super().__init__()
        self.log_list = log_list

    def emit(self, record):
        log_entry = self.format(record)
        self.log_list.append(log_entry)

class Test:
    def __init__(self,
                 name: str,
                 scheduler: Scheduler,
                 network: nx.Graph,
                 task_graph: nx.DiGraph,
                 path: pathlib.Path,
                 save_passing: bool = False,
                 simplify_instance: Callable[[nx.Graph, nx.DiGraph], Tuple[nx.Graph, nx.DiGraph]] = lambda x, y: (x, y)) -> None:
        self.scheduler = scheduler
        self.network = network
        self.task_graph = task_graph

        self.scheduler_name = scheduler.__class__.__name__
        self.name = name
        self.save_passing = save_passing
        self.path = path
        self.simplify_instance = simplify_instance

    def save_output(self, 
                    details: Dict[str, str], 
                    schedule: Dict[str, List[Task]],
                    log_entries: List[str],
                    path: pathlib.Path) -> None:
        path = path / self.name
        path.mkdir(parents=True, exist_ok=True)
        details_str = "\n".join([f"# {key}\n{value}\n\n" for key, value in details.items()])
        path.joinpath("details.md").write_text(details_str)
        
        # draw network, task graph, and gantt chart (if schedule is not None)
        ax = draw_network(self.network)
        ax.figure.savefig(path / "network.png")
        ax = draw_task_graph(self.task_graph, schedule=schedule)
        ax.figure.savefig(path / "task_graph.png")
        if schedule is not None:
            fig = draw_gantt(schedule)
            # plotly Figure
            fig.write_image(path / "gantt.png")

        path.joinpath("log.txt").write_text("\n".join(log_entries))

    def run(self) -> bool:
        # capture logging output to a tempfile
        log_entries = []
        handler = ListHandler(log_entries)
        handler.setLevel(logging.DEBUG)
        current_handlers = logging.getLogger().handlers
        # remove all handlers
        for h in current_handlers:
            logging.getLogger().removeHandler(h)
        # get current config
        current_config = logging.getLogger().getEffectiveLevel()
        # set to debug
        logging.getLogger().setLevel(logging.DEBUG)
        logging.getLogger().addHandler(handler)

        logging.info(f"Running test for {self.name}")
        schedule = None
        try:
            schedule = self.scheduler.schedule(self.network, self.task_graph)
            simple_network, simple_task_graph = self.simplify_instance(self.network, self.task_graph)
            validate_simple_schedule(simple_network, simple_task_graph, schedule)
            if self.save_passing:
                details = {
                    "scheduler": str(self.scheduler_name),
                    "schedule": pprint.pformat(schedule),
                }
                self.save_output(details, schedule, log_entries, self.path / "pass")
        except Exception as e:
            details = {
                "scheduler": str(self.scheduler_name),
                "error": str(e),
                "stacktrace": traceback.format_exc(),
                "schedule": pprint.pformat(schedule),
            }
            self.save_output(details, schedule, log_entries, self.path / "fail")
            return False
        finally:
            logging.getLogger().removeHandler(handler)
            # add back all handlers
            for h in current_handlers:
                logging.getLogger().addHandler(h)
            # set back to original config
            logging.getLogger().setLevel(current_config)

        return True

def test_common_schedulers():
    task_graphs = {
        "diamond": add_random_weights(get_diamond_dag()),
        "chain": add_random_weights(get_chain_dag()),
        "fork": add_random_weights(get_fork_dag()),
        "branching": add_random_weights(get_branching_dag()),
    }
    network = add_random_weights(get_network())
    schedulers = [
        HeftScheduler(),
        CPOPScheduler(),
        FastestNodeScheduler(),
        BruteForceScheduler(),
    ]

    for scheduler in schedulers:
        for task_graph_name, task_graph in task_graphs.items():
            test_name = f"common/{task_graph_name}/{scheduler.__class__.__name__}"
            test = Test(
                name=test_name,
                scheduler=scheduler,
                network=network.copy(),
                task_graph=task_graph.copy(),
                path=thisdir / "output" / "schedulers"
            )
            passed = test.run()
            print(f"{test.name} passed: {passed}")
            if not passed:
                print(f"Failed: {test.name} - see output in {test.path.joinpath(task_graph_name)}")

def test_reweighting_stochastic_schedulers():
    task_graphs = {
        "diamond": add_rv_weights(get_diamond_dag()),
        "chain": add_rv_weights(get_chain_dag()),
        "fork": add_rv_weights(get_fork_dag()),
        "branching": add_rv_weights(get_branching_dag()),
    }
    network = add_rv_weights(get_network())
    schedulers = [
        SheftScheduler()
    ]

    for scheduler in schedulers:
        for task_graph_name, task_graph in task_graphs.items():
            test_name = f"sheft/{task_graph_name}/{scheduler.__class__.__name__}"
            test = Test(
                name=test_name,
                scheduler=scheduler,
                network=network.copy(),
                task_graph=task_graph.copy(),
                path=thisdir / "output" / "schedulers",
                simplify_instance=lambda network, task_graph: scheduler.reweight_instance(network, task_graph)
            )
            passed = test.run()
            print(f"{test.name} passed: {passed}")
            if not passed:
                print(f"Failed: {test.name} - see output in {test.path.joinpath(task_graph_name)}")

def test_stochastic_schedulers():
    task_graphs = {
        "diamond": add_rv_weights(get_diamond_dag()),
        "chain": add_rv_weights(get_chain_dag()),
        "fork": add_rv_weights(get_fork_dag()),
        "branching": add_rv_weights(get_branching_dag()),
    }
    network = add_rv_weights(get_network())
    schedulers = [
        StochHeftScheduler(),
        ImprovedSheftScheduler(),
    ]

    for scheduler in schedulers:
        for task_graph_name, task_graph in task_graphs.items():
            test_name = f"stoch/{task_graph_name}/{scheduler.__class__.__name__}"
            schedule = scheduler.schedule(network, task_graph)
            print(f"{test_name} schedule:")
            pprint.pprint(schedule)
            print()
            # TODO: add test functionality for stochastic schedulers

def test():
    test_common_schedulers()
    test_reweighting_stochastic_schedulers()
    test_stochastic_schedulers()
    
if __name__ == "__main__":
    test()