from collections import deque
import copy
import heapq
import logging
import math
import os
import pathlib
import random
from dataclasses import dataclass
from typing import Any, Callable, Dict, Generator, Hashable, List, Optional, Set, Tuple, Type
import uuid

import networkx as nx
from saga.schedulers.base import Task, Scheduler

from common import standardize_instance

from changes import (
    Change,
    TaskGraphDeleteDependency,
    TaskGraphAddDependency,
    TaskGraphChangeDependencyWeight,
    TaskGraphChangeTaskWeight,
    NetworkChangeEdgeWeight,
    NetworkChangeNodeWeight
)

thisdir = pathlib.Path(__file__).parent.absolute()

# data class for simulated annealing iteration
@dataclass
class SimulatedAnnealingIteration:
    """Data class for simulated annealing iteration"""
    iteration: int
    temperature: float
    current_energy: float
    neighbor_energy: float
    best_energy: float

    current_schedule: Dict[str, List[Task]]
    current_base_schedule: Dict[str, List[Task]]

    neighbor_schedule: Dict[str, List[Task]]
    neighbor_base_schedule: Dict[str, List[Task]]

    best_schedule: Dict[str, List[Task]]
    best_base_schedule: Dict[str, List[Task]]

    accept_probability: float
    accepted: bool

    change: Change
    current_network: nx.Graph
    current_task_graph: nx.DiGraph
    neighbor_network: nx.Graph
    neighbor_task_graph: nx.DiGraph
    best_network: nx.Graph
    best_task_graph: nx.DiGraph

class SimulatedAnnealing:
    DEFAULT_CHANGE_TYPES = [
        TaskGraphDeleteDependency,
        TaskGraphAddDependency,
        TaskGraphChangeDependencyWeight,
        TaskGraphChangeTaskWeight,
        NetworkChangeEdgeWeight,
        NetworkChangeNodeWeight
    ]
    DEFAULT_GET_MAKESPAN = lambda network, task_graph, schedule: max([
        max([t.end for t in schedule[node]])
        for node in schedule if len(schedule[node]) > 0
    ])
    def __init__(self,
                 task_graph: nx.DiGraph,
                 network: nx.Graph,
                 scheduler: Scheduler,
                 base_scheduler: Scheduler,
                 get_makespan: Callable[[nx.Graph, nx.DiGraph, Dict[Hashable, List[Task]]], float] = DEFAULT_GET_MAKESPAN,
                 max_iterations: int = 1000,
                 max_temp: float = 100,
                 min_temp: float = 0.1,
                 cooling_rate: float = 0.99,
                 change_types: List[Type[Change]] = DEFAULT_CHANGE_TYPES):
        """Simulated annealing algorithm for scheduling task graphs on networks

        Tries to find an instance where scheduler performs as poorly as possible
        compared to base_scheduler. This is done by randomly changing the
        network and task graph and accepting changes that make the scheduler
        perform worse. The probability of accepting a change is determined by
        the temperature and the energy difference between the current and
        neighbor solution. The temperature is decreased over time, so that
        the algorithm is more likely to accept changes that make the scheduler
        perform worse. This is done to avoid getting stuck in local minima.

        Args:
            task_graph (nx.DiGraph): Initial task graph
            network (nx.Graph): Initial network
            scheduler (Scheduler): Scheduler to use
            base_scheduler (Scheduler): Scheduler to compare to
            get_makespan (Callable[[nx.Graph, nx.DiGraph, Dict[Hashable, List[Task]]], float], optional): Function to get makespan from schedule.
                Defaults to DEFAULT_GET_MAKESPAN, which returns the makespan of the schedule supposing it is deterministic.
            max_iterations (int, optional): Maximum number of iterations. Defaults to 1000.
            max_temp (float, optional): Maximum temperature. Defaults to 100.
            min_temp (float, optional): Minimum temperature. Defaults to 0.1.
            cooling_rate (float, optional): Cooling rate. Defaults to 0.99.
            change_types (List[Change], optional): List of change types to use. Defaults to DEFAULT_CHANGE_TYPES.
        """
        self.task_graph = task_graph
        self.network = network
        self.scheduler = scheduler
        self.base_scheduler = base_scheduler
        self.get_makespan = get_makespan
        self.max_iterations = max_iterations
        self.max_temp = max_temp
        self.min_temp = min_temp
        self.cooling_rate = cooling_rate
        self.change_types = change_types

        self.iterations: List[SimulatedAnnealingIteration] = []

    def reset(self):
        """Reset simulated annealing algorithm"""
        self.iterations = []

    def get_schedule(self, network: nx.Graph, task_graph: nx.DiGraph) -> Dict[str, List[Task]]:
        """Get schedule for network and task graph

        Args:
            network (nx.Graph): Network
            task_graph (nx.DiGraph): Task graph

        Returns:
            Dict[str, List[Task]]: Schedule
        """
        network, task_graph = standardize_instance(network.copy(), task_graph.copy())
        return self.scheduler.schedule(network, task_graph)

    def get_base_schedule(self, network: nx.Graph, task_graph: nx.DiGraph) -> Dict[str, List[Task]]:
        """Get base schedule for network and task graph

        Args:
            network (nx.Graph): Network
            task_graph (nx.DiGraph): Task graph

        Returns:
            Dict[str, List[Task]]: Schedule
        """
        network, task_graph = standardize_instance(network.copy(), task_graph.copy())
        return self.base_scheduler.schedule(network, task_graph)

    def run_iter(self) -> Generator[SimulatedAnnealingIteration, None, None]:
        """Run the simulated annealing algorithm, yielding the current iteration

        energy = makespan / base_makespan

        Yields:
            Generator[SimulatedAnnealingIteration, None, None]: Current iteration
        """
        # initialize temperature
        temp = self.max_temp
        # initialize current solution
        current_network, current_task_graph = self.network.copy(), self.task_graph.copy()
        best_network, best_task_graph = self.network.copy(), self.task_graph.copy()

        current_schedule = self.get_schedule(current_network, current_task_graph)
        current_base_schedule = self.get_base_schedule(current_network, current_task_graph)

        # calculate makespan
        current_makespan = self.get_makespan(current_network, current_task_graph, current_schedule)
        current_base_makespan = self.get_makespan(current_network, current_task_graph, current_base_schedule)

        # calculate energy
        current_energy = current_makespan / current_base_makespan

        # initialize best solution
        best_energy = current_energy
        best_schedule = current_schedule
        best_base_schedule = current_base_schedule

        iteration = 0 if len(self.iterations) == 0 else self.iterations[-1].iteration + 1
        # initialize iteration counter
        first_iteration = SimulatedAnnealingIteration(
            iteration=iteration,
            temperature=temp,
            current_energy=current_energy,
            neighbor_energy=current_energy,
            best_energy=best_energy,
            current_schedule=current_schedule,
            current_base_schedule=current_base_schedule,
            neighbor_schedule=current_schedule,
            neighbor_base_schedule=current_base_schedule,
            best_schedule=best_schedule,
            best_base_schedule=best_base_schedule,
            accept_probability=1,
            accepted=True,
            change=None,
            current_network=current_network,
            current_task_graph=current_task_graph,
            neighbor_network=current_network,
            neighbor_task_graph=current_task_graph,
            best_network=best_network,
            best_task_graph=best_task_graph,
        )
        self.iterations.append(first_iteration)
        yield first_iteration
        # loop until max iterations or min temperature is reached
        while iteration < self.max_iterations and temp > self.min_temp:
            log_prefix = f"[Iter {iteration}/{self.max_iterations} | Temp {temp:.2f}]"
            logging.info(f"{log_prefix} Running")

            ChangeType = random.choice(self.change_types)
            neighbor_network, neighbor_task_graph = current_network.copy(), current_task_graph.copy()
            change = ChangeType.apply_random(neighbor_network, neighbor_task_graph)
            logging.debug(f"{log_prefix} Applying {change}")

            neighbor_schedule = self.get_schedule(neighbor_network, neighbor_task_graph)
            neighbor_makespan = self.get_makespan(neighbor_network, neighbor_task_graph, neighbor_schedule)

            neighbor_base_schedule = self.get_base_schedule(neighbor_network, neighbor_task_graph)
            neighbor_base_makespan = self.get_makespan(neighbor_network, neighbor_task_graph, neighbor_base_schedule)

            neighbor_energy = neighbor_makespan / neighbor_base_makespan

            # calculate energy ratio
            # energy_ratio > 1 means neighbor has higher energy (better solution)
            # energy_ratio < 1 means neighbor has lower energy (worse solution)
            energy_ratio = neighbor_energy / current_energy

            # 1 if neighbor has higher energy, e^((1-energy_ratio)/temp) if neighbor has lower energy
            accept_probability = math.exp(-energy_ratio/temp) if energy_ratio <= 1 else 1
            accepted = random.random() < accept_probability

            # yield iteration
            new_iteration = SimulatedAnnealingIteration(
                iteration=iteration,
                temperature=temp,
                current_energy=current_energy,
                neighbor_energy=neighbor_energy,
                best_energy=best_energy,
                current_schedule=current_schedule,
                current_base_schedule=current_base_schedule,
                neighbor_schedule=neighbor_schedule,
                neighbor_base_schedule=neighbor_base_schedule,
                best_schedule=best_schedule,
                best_base_schedule=best_base_schedule,
                accept_probability=accept_probability,
                accepted=accepted,
                change=change,
                current_network=current_network.copy(),
                current_task_graph=current_task_graph.copy(),
                neighbor_network=neighbor_network.copy(),
                neighbor_task_graph=neighbor_task_graph.copy(),
                best_network=best_network.copy(),
                best_task_graph=best_task_graph.copy(),
            )
            self.iterations.append(new_iteration)
            yield new_iteration

            # if energy difference is positive, accept neighbor
            if accepted:
                current_network, current_task_graph = neighbor_network, neighbor_task_graph
                current_energy = neighbor_energy
                current_schedule = neighbor_schedule
                current_base_schedule = neighbor_base_schedule

                text_comp = 'better' if energy_ratio > 1 else 'worse'
                logging.debug(f"{log_prefix} accepted {text_comp} neighbor with energy {neighbor_energy:.3f} (prob {accept_probability:.3f}))")
                # update best solution
                if neighbor_energy > best_energy:
                    best_network, best_task_graph = neighbor_network, neighbor_task_graph
                    best_energy = neighbor_energy
                    best_schedule = neighbor_schedule
                    best_base_schedule = neighbor_base_schedule
                    logging.debug(f"{log_prefix} found new best solution with energy {best_energy:.3f}")

            # update temperature
            temp *= self.cooling_rate

            # update iteration counter
            iteration += 1

        logging.debug(f"Best solution energy: {best_energy}")

    def run(self) -> SimulatedAnnealingIteration:
        """Run the simulated annealing algorithm, returning the best iteration"""
        best_iteration: SimulatedAnnealingIteration = None
        for iteration in self.run_iter():
            if best_iteration is None or iteration.current_energy > best_iteration.current_energy:
                best_iteration = iteration
        return best_iteration

class MostPromisingOptimizer:
    def __init__(self,
                 num_optimizers: int = os.cpu_count() - 1,
                 num_cache: int = None,
                 promise_func: Callable[[str, float], float] = lambda instance_name, energy: energy) -> None:
        """Initialize the most promising optimizer

        Args:
            num_optimizers (int, optional): Number of optimizers to run in parallel. Defaults to os.cpu_count() - 1.
            num_cache (int, optional): Number of solutions to cache. Defaults to 2 * num_optimizers.
            promise_func (Callable[[str, float], float], optional): Function to calculate the promise of a solution. Defaults to lambda instance_name, makespan: 1/makespan.
        """
        self.num_optimizers = num_optimizers
        if num_cache is None:
            num_cache = 2 * num_optimizers
        self.num_cache = num_cache
        self.promise_func = promise_func

    def run_iter(self, optimizer: SimulatedAnnealing) -> Generator[List[Tuple[int, SimulatedAnnealing]], None, None]:
        # create optimizers
        optimizers = [(i, copy.deepcopy(optimizer)) for i in range(self.num_optimizers)]
        optimizer_iterators = [o.run_iter() for oid, o in optimizers]

        optimizer_max_id = self.num_optimizers - 1
        # run_iter for all optimizers together one iteration at a time
        for i in range(optimizer.max_iterations):
            # run one iteration for each optimizer
            for i, iteration in enumerate([next(iterator, None) for iterator in optimizer_iterators]):
                if iteration is None: # no more iterations - reached min temperature
                    continue
                # if a worse solution was accepted, create a new optimizer without the last iteration
                if iteration.accepted and iteration.accept_probability < 1: # accepted a worse solution
                    new_optimizer = copy.deepcopy(optimizers[i][1])
                    new_optimizer.iterations.pop()

                    optimizer_max_id += 1
                    optimizers.append((optimizer_max_id, new_optimizer))

            # sort optimizers by promise
            optimizers = sorted(optimizers, key=lambda o: self.promise_func(o[0], o[1].iterations[-1].current_energy), reverse=True)

            # keep top num_cache optimizers
            optimizers = optimizers[:self.num_cache]
            yield optimizers

        # yield final optimizers
        yield optimizers

    def run(self, optimizer: SimulatedAnnealing) -> SimulatedAnnealing:
        """Run the most promising optimizer, returning the best iteration"""
        optimizers = deque(self.run_iter(optimizer), maxlen=1).pop()
        best_run: SimulatedAnnealing = None
        for oid, o in optimizers:
            if best_run is None or o.iterations[-1].current_energy > best_run.iterations[-1].current_energy:
                best_run = o

        return best_run

