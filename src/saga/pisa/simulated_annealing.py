"""Simulated annealing for finding adversarial scheduling instances."""
import math
import os
import pathlib
import random
from typing import Dict, Generator, List, Literal, Optional, Type, cast

from pydantic import BaseModel, Field

from saga import Network, TaskGraph, Scheduler
from saga.pisa.changes import Change, ChangeType, DEFAULT_CHANGE_TYPES
from saga.schedulers import (
    BILScheduler, CpopScheduler, DuplexScheduler, ETFScheduler, FCPScheduler,
    FLBScheduler, FastestNodeScheduler, GDLScheduler, HeftScheduler,
    MCTScheduler, METScheduler, MaxMinScheduler, MinMinScheduler,
    OLBScheduler, WBAScheduler, SufferageScheduler
)

SchedulerName = Literal[
    "BIL", "CPoP", "Duplex", "ETF", "FCP", "FLB", "FastestNode", "GDL",
    "HEFT", "MCT", "MET", "MaxMin", "MinMin", "OLB", "WBA", "Sufferage"
]

SCHEDULERS: Dict[SchedulerName, Scheduler] = {
    "BIL": BILScheduler(),
    "CPoP": CpopScheduler(),
    "Duplex": DuplexScheduler(),
    "ETF": ETFScheduler(),
    "FCP": FCPScheduler(),
    "FLB": FLBScheduler(),
    "FastestNode": FastestNodeScheduler(),
    "GDL": GDLScheduler(),
    "HEFT": HeftScheduler(),
    "MCT": MCTScheduler(),
    "MET": METScheduler(),
    "MaxMin": MaxMinScheduler(),
    "MinMin": MinMinScheduler(),
    "OLB": OLBScheduler(),
    "WBA": WBAScheduler(),
    "Sufferage": SufferageScheduler(),
}


def get_pisa_dir() -> pathlib.Path:
    """Get the PISA data directory.

    Returns:
        pathlib.Path: The PISA data directory.
    """
    data_dir = pathlib.Path(os.getenv("SAGA_PISA_DIR", pathlib.Path.home() / ".saga" / "pisa"))
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


class SimulatedAnnealingConfig(BaseModel):
    """Configuration for simulated annealing."""
    max_iterations: int = Field(default=1000, description="Maximum number of iterations.")
    max_temp: float = Field(default=100.0, description="Maximum (starting) temperature.")
    min_temp: float = Field(default=0.1, description="Minimum temperature (stopping condition).")
    cooling_rate: float = Field(default=0.99, description="Cooling rate per iteration.")
    change_types: List[str] = Field(
        default_factory=lambda: [c.__name__ for c in DEFAULT_CHANGE_TYPES],
        description="List of change type names to use."
    )


class SimulatedAnnealingIteration(BaseModel):
    """Data for a single simulated annealing iteration."""
    iteration: int = Field(..., description="The iteration number.")
    temperature: float = Field(..., description="The current temperature.")

    current_energy: float = Field(..., description="Energy of the current solution.")
    neighbor_energy: float = Field(..., description="Energy of the neighbor solution.")
    best_energy: float = Field(..., description="Best energy found so far.")

    current_makespan: float = Field(..., description="Makespan of current schedule.")
    current_base_makespan: float = Field(..., description="Makespan of current base schedule.")
    neighbor_makespan: float = Field(..., description="Makespan of neighbor schedule.")
    neighbor_base_makespan: float = Field(..., description="Makespan of neighbor base schedule.")

    accept_probability: float = Field(..., description="Probability of accepting the neighbor.")
    accepted: bool = Field(..., description="Whether the neighbor was accepted.")

    change: Optional[ChangeType] = Field(default=None, description="The change applied.")

    current_network: Network = Field(..., description="The current network.")
    current_task_graph: TaskGraph = Field(..., description="The current task graph.")
    neighbor_network: Network = Field(..., description="The neighbor network.")
    neighbor_task_graph: TaskGraph = Field(..., description="The neighbor task graph.")


class SimulatedAnnealingRun(BaseModel):
    """Metadata and results for a simulated annealing run."""
    name: str = Field(..., description="Name of this run.")
    scheduler: SchedulerName = Field(..., description="Scheduler being tested.")
    base_scheduler: SchedulerName = Field(..., description="Base scheduler for comparison.")
    config: SimulatedAnnealingConfig = Field(..., description="Configuration used.")

    initial_network: Network = Field(..., description="Initial network.")
    initial_task_graph: TaskGraph = Field(..., description="Initial task graph.")

    best_energy: float = Field(default=1.0, description="Best energy found.")
    best_network: Optional[Network] = Field(default=None, description="Best network found.")
    best_task_graph: Optional[TaskGraph] = Field(default=None, description="Best task graph found.")

    num_iterations: int = Field(default=0, description="Number of iterations completed.")
    completed: bool = Field(default=False, description="Whether the run completed.")


class SimulatedAnnealing:
    """Simulated annealing for finding adversarial scheduling instances.

    Persists iterations to disk for resumability and post-analysis.

    Directory structure:
        {data_dir}/{name}/
            run.json          - SimulatedAnnealingRun metadata
            iterations/
                0000.json     - SimulatedAnnealingIteration for iteration 0
                0001.json     - SimulatedAnnealingIteration for iteration 1
                ...
    """

    def __init__(
        self,
        name: str,
        scheduler: SchedulerName,
        base_scheduler: SchedulerName,
        initial_network: Network,
        initial_task_graph: TaskGraph,
        config: Optional[SimulatedAnnealingConfig] = None,
        data_dir: Optional[pathlib.Path] = None,
    ):
        """Initialize simulated annealing.

        Args:
            name: Name of this run (used for directory).
            scheduler: Name of scheduler to test.
            base_scheduler: Name of base scheduler for comparison.
            initial_network: Starting network.
            initial_task_graph: Starting task graph.
            config: Configuration options.
            data_dir: Directory to store results.
        """
        self.name = name
        self.data_dir = data_dir or get_pisa_dir()
        self._run_dir = self.data_dir / name
        self._run_dir.mkdir(parents=True, exist_ok=True)
        self._iterations_dir = self._run_dir / "iterations"
        self._iterations_dir.mkdir(parents=True, exist_ok=True)

        self._scheduler_name = scheduler
        self._base_scheduler_name = base_scheduler
        self._scheduler = SCHEDULERS[scheduler]
        self._base_scheduler = SCHEDULERS[base_scheduler]

        self._config = config or SimulatedAnnealingConfig()
        self._change_types = self._resolve_change_types(self._config.change_types)

        # Initialize or load run metadata
        run_path = self._run_dir / "run.json"
        if run_path.exists():
            self._run = SimulatedAnnealingRun.model_validate_json(run_path.read_text())
        else:
            self._run = SimulatedAnnealingRun(
                name=name,
                scheduler=scheduler,
                base_scheduler=base_scheduler,
                config=self._config,
                initial_network=initial_network,
                initial_task_graph=initial_task_graph,
            )
            self._save_run()

    def _resolve_change_types(self, change_type_names: List[str]) -> List[Type[Change]]:
        """Resolve change type names to classes."""
        from saga.pisa.changes import (
            TaskGraphDeleteDependency, TaskGraphAddDependency,
            TaskGraphChangeDependencyWeight, TaskGraphChangeTaskWeight,
            NetworkChangeEdgeWeight, NetworkChangeNodeWeight
        )
        name_to_class = {
            "TaskGraphDeleteDependency": TaskGraphDeleteDependency,
            "TaskGraphAddDependency": TaskGraphAddDependency,
            "TaskGraphChangeDependencyWeight": TaskGraphChangeDependencyWeight,
            "TaskGraphChangeTaskWeight": TaskGraphChangeTaskWeight,
            "NetworkChangeEdgeWeight": NetworkChangeEdgeWeight,
            "NetworkChangeNodeWeight": NetworkChangeNodeWeight,
        }
        return [name_to_class[name] for name in change_type_names]

    def _save_run(self) -> None:
        """Save run metadata."""
        run_path = self._run_dir / "run.json"
        run_path.write_text(self._run.model_dump_json(indent=2))

    def _save_iteration(self, iteration: SimulatedAnnealingIteration) -> None:
        """Save an iteration to disk."""
        iteration_path = self._iterations_dir / f"{iteration.iteration:06d}.json"
        iteration_path.write_text(iteration.model_dump_json(indent=2))

    def get_iteration(self, iteration_num: int) -> Optional[SimulatedAnnealingIteration]:
        """Load an iteration from disk."""
        iteration_path = self._iterations_dir / f"{iteration_num:06d}.json"
        if not iteration_path.exists():
            return None
        return SimulatedAnnealingIteration.model_validate_json(iteration_path.read_text())

    def iter_iterations(self) -> Generator[SimulatedAnnealingIteration, None, None]:
        """Iterate over all saved iterations."""
        for path in sorted(self._iterations_dir.glob("*.json")):
            yield SimulatedAnnealingIteration.model_validate_json(path.read_text())

    @property
    def run(self) -> SimulatedAnnealingRun:
        """Get run metadata."""
        return self._run

    @property
    def num_iterations(self) -> int:
        """Number of iterations completed."""
        return self._run.num_iterations

    @classmethod
    def load(cls, name: str, data_dir: Optional[pathlib.Path] = None) -> "SimulatedAnnealing":
        """Load an existing run from disk.

        Args:
            name: Name of the run to load.
            data_dir: Directory where runs are stored.

        Returns:
            SimulatedAnnealing instance.
        """
        data_dir = data_dir or get_pisa_dir()
        run_path = data_dir / name / "run.json"
        if not run_path.exists():
            raise FileNotFoundError(f"Run '{name}' not found at {run_path}")

        run = SimulatedAnnealingRun.model_validate_json(run_path.read_text())
        return cls(
            name=run.name,
            scheduler=run.scheduler,
            base_scheduler=run.base_scheduler,
            initial_network=run.initial_network,
            initial_task_graph=run.initial_task_graph,
            config=run.config,
            data_dir=data_dir,
        )

    def run_iter(self) -> Generator[SimulatedAnnealingIteration, None, None]:
        """Run simulated annealing, yielding each iteration.

        Resumes from last saved iteration if interrupted.

        Yields:
            SimulatedAnnealingIteration for each step.
        """
        config = self._config

        # Resume from last iteration or start fresh
        if self._run.num_iterations > 0:
            last_iter = self.get_iteration(self._run.num_iterations - 1)
            if last_iter is None:
                raise ValueError("Could not load last iteration for resume")

            iteration = self._run.num_iterations
            temp = last_iter.temperature * config.cooling_rate
            current_network = last_iter.current_network
            current_task_graph = last_iter.current_task_graph
            current_energy = last_iter.current_energy
            best_energy = self._run.best_energy
            best_network = self._run.best_network
            best_task_graph = self._run.best_task_graph
        else:
            iteration = 0
            temp = config.max_temp
            current_network = self._run.initial_network
            current_task_graph = self._run.initial_task_graph

            # Calculate initial energy
            current_schedule = self._scheduler.schedule(current_network, current_task_graph)
            current_base_schedule = self._base_scheduler.schedule(current_network, current_task_graph)
            current_energy = current_schedule.makespan / current_base_schedule.makespan

            best_energy = current_energy
            best_network = current_network
            best_task_graph = current_task_graph

            # Save initial iteration
            initial_iter = SimulatedAnnealingIteration(
                iteration=0,
                temperature=temp,
                current_energy=current_energy,
                neighbor_energy=current_energy,
                best_energy=best_energy,
                current_makespan=current_schedule.makespan,
                current_base_makespan=current_base_schedule.makespan,
                neighbor_makespan=current_schedule.makespan,
                neighbor_base_makespan=current_base_schedule.makespan,
                accept_probability=1.0,
                accepted=True,
                change=None,
                current_network=current_network,
                current_task_graph=current_task_graph,
                neighbor_network=current_network,
                neighbor_task_graph=current_task_graph,
            )
            self._save_iteration(initial_iter)
            self._run.num_iterations = 1
            self._run.best_energy = best_energy
            self._run.best_network = best_network
            self._run.best_task_graph = best_task_graph
            self._save_run()
            yield initial_iter
            iteration = 1

        # Main loop
        while iteration < config.max_iterations and temp > config.min_temp:
            # Apply random change
            ChangeClass = random.choice(self._change_types)
            change_result, neighbor_network, neighbor_task_graph = ChangeClass.apply_random(
                current_network, current_task_graph
            )
            change = cast(Optional[ChangeType], change_result)

            # Compute schedules and energy
            neighbor_schedule = self._scheduler.schedule(neighbor_network, neighbor_task_graph)
            neighbor_base_schedule = self._base_scheduler.schedule(neighbor_network, neighbor_task_graph)
            neighbor_energy = neighbor_schedule.makespan / neighbor_base_schedule.makespan

            # Acceptance probability
            energy_ratio = neighbor_energy / current_energy
            accept_probability = math.exp(-energy_ratio / temp) if energy_ratio <= 1 else 1.0
            accepted = random.random() < accept_probability

            # Get current makespans for logging
            current_schedule = self._scheduler.schedule(current_network, current_task_graph)
            current_base_schedule = self._base_scheduler.schedule(current_network, current_task_graph)

            # Create and save iteration
            iter_data = SimulatedAnnealingIteration(
                iteration=iteration,
                temperature=temp,
                current_energy=current_energy,
                neighbor_energy=neighbor_energy,
                best_energy=best_energy,
                current_makespan=current_schedule.makespan,
                current_base_makespan=current_base_schedule.makespan,
                neighbor_makespan=neighbor_schedule.makespan,
                neighbor_base_makespan=neighbor_base_schedule.makespan,
                accept_probability=accept_probability,
                accepted=accepted,
                change=change,
                current_network=current_network,
                current_task_graph=current_task_graph,
                neighbor_network=neighbor_network,
                neighbor_task_graph=neighbor_task_graph,
            )
            self._save_iteration(iter_data)

            # Update state if accepted
            if accepted:
                current_network = neighbor_network
                current_task_graph = neighbor_task_graph
                current_energy = neighbor_energy

                if neighbor_energy > best_energy:
                    best_energy = neighbor_energy
                    best_network = neighbor_network
                    best_task_graph = neighbor_task_graph

            # Update run metadata
            self._run.num_iterations = iteration + 1
            self._run.best_energy = best_energy
            self._run.best_network = best_network
            self._run.best_task_graph = best_task_graph
            self._save_run()

            yield iter_data

            # Cool down
            temp *= config.cooling_rate
            iteration += 1

        # Mark as completed
        self._run.completed = True
        self._save_run()

    def execute(self, progress: bool = True) -> SimulatedAnnealingRun:
        """Run simulated annealing to completion.

        Args:
            progress: Whether to print progress.

        Returns:
            The final run metadata.
        """
        for iter_data in self.run_iter():
            if progress:
                print(
                    f"\r[Iter {iter_data.iteration}/{self._config.max_iterations}] "
                    f"Temp: {iter_data.temperature:.2f} | "
                    f"Energy: {iter_data.current_energy:.4f} | "
                    f"Best: {iter_data.best_energy:.4f}",
                    end=""
                )
        if progress:
            print()  # Newline after progress
        return self._run
