from itertools import product
import logging
import pprint
import traceback
from typing import Callable, Dict, List, Tuple

import numpy as np
from scipy.stats import norm
from saga.base import Scheduler, Task
import networkx as nx
import pathlib
from saga.common.cpop import CPOPScheduler
from saga.common.fastest_node import FastestNodeScheduler
from saga.common.heft import HeftScheduler
from saga.stochastic.sheft import SheftScheduler
from saga.stochastic.stoch_heft import StochHeftScheduler
from saga.stochastic.improved_sheft import ImprovedSheftScheduler
from saga.utils.draw import draw_gantt, draw_network, draw_task_graph
from saga.utils.random_variable import RandomVariable
from saga.utils.tools import validate_simple_schedule

from uuid import uuid4

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

    def run(self) -> None:
        # capture logging output to a tempfile
        log_entries = []
        handler = ListHandler(log_entries)
        handler.setLevel(logging.DEBUG)
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
        finally:
            logging.getLogger().removeHandler(handler)

        

def get_random_network(num_nodes: int = 4):
    network = nx.Graph()
    network.add_nodes_from(range(num_nodes))
    network.add_edges_from(list(product(network.nodes, repeat=2)))
    for node in network.nodes:
        network.nodes[node]["weight"] = np.random.random()
    for edge in network.edges:
        if edge[0] == edge[1]:
            network.edges[edge]["weight"] = 1e9
        else:
            network.edges[edge]["weight"] = np.random.random()
    return network

common_schedulers = [HeftScheduler(), CPOPScheduler(), FastestNodeScheduler()]
def test_common_chain_dag():
    task_graph = nx.DiGraph()
    task_graph.add_nodes_from(["A", "B", "C", "D", "E", "F"])
    task_graph.add_edges_from([("A", "B"), ("B", "C"), ("C", "D"), ("D", "E"), ("E", "F")])

    for node in task_graph.nodes:
        task_graph.nodes[node]["weight"] = np.random.random()

    for edge in task_graph.edges:
        task_graph.edges[edge]["weight"] = np.random.random()

    network = get_random_network()
    tests = [
        Test(
            f"chain_{scheduler.__class__.__name__}",
            scheduler, network.copy(),
            task_graph.copy(), thisdir / "output" / "schedulers"
        )
        for scheduler in common_schedulers
    ]

    for test in tests:
        test.run()

def test_common_diamond_dag():
    task_graph = nx.DiGraph()
    task_graph.add_nodes_from(["A", "B", "C", "D", "E"])
    task_graph.add_edges_from([("A", "B"), ("A", "C"), ("A", "D"), ("B", "E"), ("C", "E"), ("D", "E")])

    for node in task_graph.nodes:
        task_graph.nodes[node]["weight"] = np.random.random()

    for edge in task_graph.edges:
        task_graph.edges[edge]["weight"] = np.random.random()

    network = get_random_network()
    tests = [
        Test(
            f"diamond_{scheduler.__class__.__name__}",
            scheduler, network.copy(), 
            task_graph.copy(), thisdir / "output" / "schedulers"
        )
        for scheduler in common_schedulers
    ]

    for test in tests:
        test.run()

def test_common_fork_dag():
    task_graph = nx.DiGraph()
    task_graph.add_nodes_from(["A", "B", "C", "D", "E", "F"])
    task_graph.add_edges_from([("A", "B"), ("A", "C"), ("B", "D"), ("C", "E"), ("D", "F"), ("E", "F")])

    for node in task_graph.nodes:
        task_graph.nodes[node]["weight"] = np.random.random()

    for edge in task_graph.edges:
        task_graph.edges[edge]["weight"] = np.random.random()

    network = get_random_network()
    tests = [
        Test(
            f"fork_{scheduler.__class__.__name__}",
            scheduler, network.copy(),
            task_graph.copy(), thisdir / "output" / "schedulers"
        )
        for scheduler in common_schedulers
    ]

    for test in tests:
        test.run()

def test_common():
    test_common_chain_dag()
    test_common_diamond_dag()
    test_common_fork_dag()

def test_diamond_sheft():
    task_graph = nx.DiGraph()
    task_graph.add_nodes_from(["A", "B", "C", "D", "E"])
    task_graph.add_edges_from([("A", "B"), ("A", "C"), ("A", "D"), ("B", "E"), ("C", "E"), ("D", "E")])

    for node in task_graph.nodes:
        task_graph.nodes[node]["weight"] = norm(0, 1)

    for edge in task_graph.edges:
        task_graph.edges[edge]["weight"] = norm(0, 1)

    network = get_random_network()
    for node in network.nodes:
        network.nodes[node]["weight"] = norm(0, 1)
    for edge in network.edges:
        if edge[0] == edge[1]:
            network.edges[edge]["weight"] = norm(1e9, 1e-9)
        else:
            network.edges[edge]["weight"] = norm(0, 1)
    scheduler = SheftScheduler()
    test = Test(
        f"diamond_{scheduler.__class__.__name__}",
        scheduler, network.copy(),
        task_graph.copy(), thisdir / "output" / "schedulers",
        simplify_instance=lambda network, task_graph: SheftScheduler.reweight_instance(network, task_graph)
    )
    test.run()

def test_fork_sheft():
    task_graph = nx.DiGraph()
    task_graph.add_nodes_from(["A", "B", "C", "D", "E", "F"])
    task_graph.add_edges_from([("A", "B"), ("A", "C"), ("B", "D"), ("C", "E"), ("D", "F"), ("E", "F")])

    for node in task_graph.nodes:
        task_graph.nodes[node]["weight"] = norm(0, 1)

    for edge in task_graph.edges:
        task_graph.edges[edge]["weight"] = norm(0, 1)

    network = get_random_network()
    for node in network.nodes:
        network.nodes[node]["weight"] = norm(0, 1)
    for edge in network.edges:
        if edge[0] == edge[1]:
            network.edges[edge]["weight"] = norm(1e9, 1e-9)
        else:
            network.edges[edge]["weight"] = norm(0, 1)
    scheduler = SheftScheduler()
    test = Test(
        f"fork_{scheduler.__class__.__name__}",
        scheduler, network.copy(),
        task_graph.copy(), thisdir / "output" / "schedulers",
        simplify_instance=lambda network, task_graph: SheftScheduler.reweight_instance(network, task_graph)
    )
    test.run()

def test_sheft():
    test_diamond_sheft()
    test_fork_sheft()

def test_chain_stoch_heft():
    task_graph = nx.DiGraph()
    task_graph.add_nodes_from(["A", "B", "C", "D", "E"])
    task_graph.add_edges_from([("A", "B"), ("B", "C"), ("C", "D"), ("D", "E")])

    x = np.linspace(0, 10, 100000)
    for node in task_graph.nodes:
        task_graph.nodes[node]["weight"] = RandomVariable.from_pdf(x, norm.pdf(x, loc=5, scale=1))

    for edge in task_graph.edges:
        task_graph.edges[edge]["weight"] = RandomVariable.from_pdf(x, norm.pdf(x, loc=5, scale=1))

    network = get_random_network()
    for node in network.nodes:
        network.nodes[node]["weight"] = RandomVariable.from_pdf(x, norm.pdf(x, loc=5, scale=1))
    for edge in network.edges:
        if edge[0] == edge[1]:
            network.edges[edge]["weight"] = RandomVariable([1e9], num_samples=1)
        else:
            network.edges[edge]["weight"] = RandomVariable.from_pdf(x, norm.pdf(x, loc=5, scale=1))
    scheduler = StochHeftScheduler()
    schedule = scheduler.schedule(network, task_graph)
    pprint.pprint(schedule)
    # TODO: validate schedule

def test_diamond_stoch_heft():
    task_graph = nx.DiGraph()
    task_graph.add_nodes_from(["A", "B", "C", "D", "E"])
    task_graph.add_edges_from([("A", "B"), ("A", "C"), ("A", "D"), ("B", "E"), ("C", "E"), ("D", "E")])

    x = np.linspace(0, 10, 100000)
    for node in task_graph.nodes:
        task_graph.nodes[node]["weight"] = RandomVariable.from_pdf(x, norm.pdf(x, loc=5, scale=1))

    for edge in task_graph.edges:
        task_graph.edges[edge]["weight"] = RandomVariable.from_pdf(x, norm.pdf(x, loc=5, scale=1))

    network = get_random_network()
    for node in network.nodes:
        network.nodes[node]["weight"] = RandomVariable.from_pdf(x, norm.pdf(x, loc=5, scale=1))
    for edge in network.edges:
        if edge[0] == edge[1]:
            network.edges[edge]["weight"] = RandomVariable([1e9], num_samples=1)
        else:
            network.edges[edge]["weight"] = RandomVariable.from_pdf(x, norm.pdf(x, loc=5, scale=1))
    scheduler = StochHeftScheduler()
    schedule = scheduler.schedule(network, task_graph)
    pprint.pprint(schedule)
    # TODO: validate schedule

def test_fork_stoch_heft():
    task_graph = nx.DiGraph()
    task_graph.add_nodes_from(["A", "B", "C", "D", "E", "F"])
    task_graph.add_edges_from([("A", "B"), ("A", "C"), ("B", "D"), ("C", "E"), ("D", "F"), ("E", "F")])

    x = np.linspace(0, 10, 100000)
    for node in task_graph.nodes:
        task_graph.nodes[node]["weight"] = RandomVariable.from_pdf(x, norm.pdf(x, loc=5, scale=1))

    for edge in task_graph.edges:
        task_graph.edges[edge]["weight"] = RandomVariable.from_pdf(x, norm.pdf(x, loc=5, scale=1))

    network = get_random_network()
    for node in network.nodes:
        network.nodes[node]["weight"] = RandomVariable.from_pdf(x, norm.pdf(x, loc=5, scale=1))
    for edge in network.edges:
        if edge[0] == edge[1]:
            network.edges[edge]["weight"] = RandomVariable([1e9], num_samples=1)
        else:
            network.edges[edge]["weight"] = RandomVariable.from_pdf(x, norm.pdf(x, loc=5, scale=1))
    scheduler = StochHeftScheduler()
    schedule = scheduler.schedule(network, task_graph)
    pprint.pprint(schedule)
    # TODO: validate schedule

def test_stoch_heft():
    test_chain_stoch_heft()
    test_diamond_stoch_heft()
    test_fork_stoch_heft()

def test_chain_improved_sheft():
    task_graph = nx.DiGraph()
    task_graph.add_nodes_from(["A", "B", "C", "D", "E"])
    task_graph.add_edges_from([("A", "B"), ("B", "C"), ("C", "D"), ("D", "E")])

    x = np.linspace(0, 10, 100000)
    for node in task_graph.nodes:
        task_graph.nodes[node]["weight"] = RandomVariable.from_pdf(x, norm.pdf(x, loc=5, scale=1))

    for edge in task_graph.edges:
        task_graph.edges[edge]["weight"] = RandomVariable.from_pdf(x, norm.pdf(x, loc=5, scale=1))

    network = get_random_network()
    for node in network.nodes:
        network.nodes[node]["weight"] = RandomVariable.from_pdf(x, norm.pdf(x, loc=5, scale=1))
    for edge in network.edges:
        if edge[0] == edge[1]:
            network.edges[edge]["weight"] = RandomVariable([1e9], num_samples=1)
        else:
            network.edges[edge]["weight"] = RandomVariable.from_pdf(x, norm.pdf(x, loc=5, scale=1))
    scheduler = ImprovedSheftScheduler()
    schedule = scheduler.schedule(network, task_graph)
    pprint.pprint(schedule)
    # TODO: validate schedule


def test_stochastic():
    test_sheft()
    test_stoch_heft()
    test_chain_improved_sheft()

def test():
    logging.basicConfig(level=logging.DEBUG)
    # test_common()
    test_stochastic()
    
    

if __name__ == "__main__":
    test()