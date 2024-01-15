import logging
import pathlib
import pprint
import traceback
from typing import Callable, Dict, List, Tuple

import networkx as nx

from matplotlib import pyplot as plt

from saga.general import GeneralScheduler
from saga.general.InsertTask import EarliestFinishTimeInsert, CriticalPathInsert, LookAheadInsert
from saga.general.RankingHeuristics import UpwardRankSort, CriticalPathSort, DownwardRankSort
from saga.general.TieBreaker import Sufferage, RandomTieBreaker, KDepthTieBreaker
from saga.general.Filters import KFirstFilter
from saga.scheduler import Scheduler, Task
from saga.schedulers import (
    BruteForceScheduler, CpopScheduler, DuplexScheduler, ETFScheduler,
    FastestNodeScheduler, FCPScheduler, HeftScheduler, MaxMinScheduler,
    METScheduler, MinMinScheduler, SMTScheduler, WBAScheduler, HybridScheduler,
    BILScheduler, FLBScheduler, DPSScheduler, GDLScheduler, MsbcScheduler,
    SufferageScheduler
)
from saga.schedulers.stochastic.improved_sheft import ImprovedSheftScheduler
from saga.schedulers.stochastic.sheft import SheftScheduler
from saga.schedulers.stochastic.stoch_heft import StochHeftScheduler
from saga.utils.draw import draw_gantt, draw_network, draw_task_graph
from saga.utils.random_graphs import (
    add_random_weights,
    add_rv_weights,
    get_branching_dag,
    get_chain_dag,
    get_diamond_dag,
    get_fork_dag,
    get_network,
)
from saga.utils.tools import validate_simple_schedule

thisdir = pathlib.Path(__file__).resolve().parent


class ListHandler(logging.Handler):
    """A logging handler that appends log entries to a list."""

    def __init__(self, log_list: List[str]) -> None:
        """Initializes the handler.

        Args:
            log_list (List[str]): The list to append log entries to.
        """
        super().__init__()
        self.log_list = log_list

    def emit(self, record: logging.LogRecord) -> None:
        """Appends a log entry to the list.

        Args:
            record (logging.LogRecord): The log record.
        """
        log_entry = self.format(record)
        self.log_list.append(log_entry)


class Test:
    """A test case for a scheduler."""

    def __init__(
        self,
        name: str,
        scheduler: Scheduler,
        network: nx.Graph,
        task_graph: nx.DiGraph,
        path: pathlib.Path,
        save_passing: bool = False,
        simplify_instance: Callable[
            [nx.Graph, nx.DiGraph], Tuple[nx.Graph, nx.DiGraph]
        ] = lambda x, y: (x, y),
    ) -> None:
        """Initializes the test case.

        Args:
            name (str): The name of the test.
            scheduler (Scheduler): The scheduler to test.
            network (nx.Graph): The network.
            task_graph (nx.DiGraph): The task graph.
            path (pathlib.Path): The path to save the test results to.
            save_passing (bool, optional): Whether to save passing tests. Defaults to False.
            simplify_instance (Callable[[nx.Graph, nx.DiGraph], Tuple[nx.Graph, nx.DiGraph]], optional): A
                function to simplify the instance. Defaults to lambda x, y: (x, y).
        """
        self.scheduler = scheduler
        self.network = network
        self.task_graph = task_graph

        self.scheduler_name = scheduler.__class__.__name__
        self.name = name
        self.save_passing = save_passing
        self.path = path
        self.simplify_instance = simplify_instance

    def save_output(
        self,
        details: Dict[str, str],
        schedule: Dict[str, List[Task]],
        log_entries: List[str],
        path: pathlib.Path,
    ) -> None:
        """Saves the output of the test.

        Args:
            details (Dict[str, str]): The details of the test.
            schedule (Dict[str, List[Task]]): The schedule.
            log_entries (List[str]): The log entries.
            path (pathlib.Path): The path to save the output to.
        """
        path = path / self.name
        path.mkdir(parents=True, exist_ok=True)
        details_str = "\n".join(
            [f"# {key}\n{value}\n\n" for key, value in details.items()]
        )
        path.joinpath("details.md").write_text(details_str)

        # draw network, task graph, and gantt chart (if schedule is not None)
        axis = draw_network(self.network)
        axis.figure.savefig(path / "network.png")
        axis = draw_task_graph(self.task_graph, schedule=schedule)
        axis.figure.savefig(path / "task_graph.png")
        if schedule is not None:
            ax: plt.Axes = draw_gantt(schedule)
            ax.get_figure().savefig(path / "gantt.png")

        path.joinpath("log.txt").write_text("\n".join(log_entries))
        # close all figures
        plt.close("all")

    def run(self) -> bool:
        """Runs the test.

        Returns:
            bool: Whether the test passed.
        """
        # capture logging output to a tempfile
        log_entries = []
        handler = ListHandler(log_entries)
        handler.setLevel(logging.DEBUG)
        current_handlers = logging.getLogger().handlers
        # remove all handlers
        for current_handler in current_handlers:
            logging.getLogger().removeHandler(current_handler)
        # get current config
        current_config = logging.getLogger().getEffectiveLevel()
        # set to debug
        logging.getLogger().setLevel(logging.DEBUG)
        logging.getLogger().addHandler(handler)

        logging.info("Running test for %s", self.name)
        schedule = None
        try:
            schedule = self.scheduler.schedule(self.network, self.task_graph)
            simple_network, simple_task_graph = self.simplify_instance(
                self.network, self.task_graph
            )
            validate_simple_schedule(simple_network, simple_task_graph, schedule)
            if self.save_passing:
                details = {
                    "scheduler": str(self.scheduler_name),
                    "schedule": pprint.pformat(schedule),
                }
                self.save_output(details, schedule, log_entries, self.path / "pass")
                print(max([schedule[node][-1].end for node in schedule if schedule[node]]))
        except Exception as exp:  # pylint: disable=broad-except
            details = {
                "scheduler": str(self.scheduler_name),
                "error": str(exp),
                "stacktrace": traceback.format_exc(),
                "schedule": pprint.pformat(schedule),
            }
            self.save_output(details, schedule, log_entries, self.path / "fail")
            return False
        finally:
            logging.getLogger().removeHandler(handler)
            # add back all handlers
            for current_handler in current_handlers:
                logging.getLogger().addHandler(current_handler)
            # set back to original config
            logging.getLogger().setLevel(current_config)

        return True


# class KDepth:
#     def __init__(self, scheduler, k: int = 1):
#         self.scheduler = scheduler

#     def __call__(self, network, task_graph, runtimes, commtimes, comp_schedule, task_schedule, priority_queue):
        
#         for task in priority_queue:
#             _task_graph = None # trim task_graph to only include child tasks within k distance from task
#             _task_schedule = deepcopy(task_schedule)

#             self.scheduler.insert_task(network, _task_graph, runtimes, commtimes, comp_schedule, _task_schedule, task)
#             schedule = self.scheduler(network, _task_graph, runtimes, commtimes, comp_schedule, _task_schedule, priority_queue)
#            # TODO: return option w/ smallest makespan

def test_common_schedulers():
    """Tests schedulers schedulers on schedulers task graphs."""
    task_graphs = {
        # "diamond": add_random_weights(get_diamond_dag()),
        # "chain": add_random_weights(get_chain_dag()),
        # "fork": add_random_weights(get_fork_dag()),
        "branching": add_random_weights(get_branching_dag(levels=4, branching_factor=3)),
    }
    network = add_random_weights(get_network())

    schedulers = [
        HeftScheduler(),
        # CpopScheduler(),
        # FastestNodeScheduler(),
        # BruteForceScheduler(),
        # MinMinScheduler(),
        # ETFScheduler(),
        # MaxMinScheduler(),
        # DuplexScheduler(),
        # METScheduler(),
        # FCPScheduler(),
        # SMTScheduler(solver_name="z3"),
        # HybridScheduler(schedulers=[HeftScheduler(), CpopScheduler()]),
        # WBAScheduler(),
        # BILScheduler(),
        # FLBScheduler(),
        # GDLScheduler()
        # HbmctScheduler()
        # MsbcScheduler()
        # DPSScheduler(),
        # GDLScheduler(),
        # SufferageScheduler(),
        GeneralScheduler(
            UpwardRankSort(), None, KDepthTieBreaker(k=2), EarliestFinishTimeInsert()
        ),
        # GeneralScheduler(
        #     UpwardRankSort(), None, None, LookAheadInsert(k=2)
        # ),
    ]

    for scheduler in schedulers:
        for task_graph_name, task_graph in task_graphs.items():
            test_name = f"schedulers/{task_graph_name}/{scheduler.__class__.__name__}"
            test = Test(
                name=test_name,
                scheduler=scheduler,
                network=network.copy(),
                task_graph=task_graph.copy(),
                path=thisdir / "output" / "schedulers",
                save_passing=True,
            )
            passed = test.run()
            print(f"{test.name} passed: {passed}")
            if not passed:
                print(
                    f"Failed: {test.name} - see output in {test.path.joinpath('fail', test_name, 'details.md')}"
                )


def test_reweighting_stochastic_schedulers():
    """Tests the stochastic schedulers with reweighting."""
    task_graphs = {
        "diamond": add_rv_weights(get_diamond_dag()),
        "chain": add_rv_weights(get_chain_dag()),
        "fork": add_rv_weights(get_fork_dag()),
        "branching": add_rv_weights(get_branching_dag()),
    }
    network = add_rv_weights(get_network())
    schedulers = [SheftScheduler()]

    for scheduler in schedulers:
        for task_graph_name, task_graph in task_graphs.items():
            test_name = f"sheft/{task_graph_name}/{scheduler.__class__.__name__}"
            test = Test(
                name=test_name,
                scheduler=scheduler,
                network=network.copy(),
                task_graph=task_graph.copy(),
                path=thisdir / "output" / "schedulers",
                simplify_instance=scheduler.reweight_instance,
            )
            passed = test.run()
            print(f"{test.name} passed: {passed}")
            if not passed:
                print(
                    f"Failed: {test.name} - see output in {test.path.joinpath(task_graph_name)}"
                )


def test_stochastic_schedulers():
    """Tests stochastic schedulers."""
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


def test_all():
    """Runs all tests."""
    test_common_schedulers()
    # test_reweighting_stochastic_schedulers()
    # test_stochastic_schedulers()


if __name__ == "__main__":
    test_all()
