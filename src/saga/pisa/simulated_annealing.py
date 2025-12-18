"""Simulated annealing for finding adversarial scheduling instances."""

import math
import os
import pathlib
import random
from typing import Dict, Generator, List, Literal, Optional, Type, cast

from pydantic import BaseModel, Field

from saga import Network, Schedule, TaskGraph, Scheduler
from saga.pisa.changes import Change, ChangeType, DEFAULT_CHANGE_TYPES
from saga.schedulers import (
    BILScheduler,
    CpopScheduler,
    DuplexScheduler,
    ETFScheduler,
    FCPScheduler,
    FLBScheduler,
    FastestNodeScheduler,
    GDLScheduler,
    HeftScheduler,
    MCTScheduler,
    METScheduler,
    MaxMinScheduler,
    MinMinScheduler,
    OLBScheduler,
    WBAScheduler,
    SufferageScheduler,
)

SchedulerName = Literal[
    "BIL",
    "CPoP",
    "Duplex",
    "ETF",
    "FCP",
    "FLB",
    "FastestNode",
    "GDL",
    "HEFT",
    "MCT",
    "MET",
    "MaxMin",
    "MinMin",
    "OLB",
    "WBA",
    "Sufferage",
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
    data_dir = pathlib.Path(
        os.getenv("SAGA_PISA_DIR", pathlib.Path.home() / ".saga" / "pisa")
    )
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


class SimulatedAnnealingConfig(BaseModel):
    """Configuration for simulated annealing."""

    max_iterations: int = Field(
        default=1000, description="Maximum number of iterations."
    )
    max_temp: float = Field(
        default=100.0, description="Maximum (starting) temperature."
    )
    min_temp: float = Field(
        default=0.1, description="Minimum temperature (stopping condition)."
    )
    cooling_rate: float = Field(default=0.99, description="Cooling rate per iteration.")
    change_types: List[str] = Field(
        default_factory=lambda: [c.__name__ for c in DEFAULT_CHANGE_TYPES],
        description="List of change type names to use.",
    )


class SimulatedAnnealingIteration(BaseModel):
    """Data for a single simulated annealing iteration."""

    iteration: int = Field(..., description="The iteration number.")
    temperature: float = Field(..., description="The current temperature.")

    change: Optional[ChangeType] = Field(
        default=None, description="The change applied."
    )

    current_schedule: Schedule = Field(..., description="The current schedule")
    current_base_schedule: Schedule = Field(
        ..., description="The current base schedule"
    )
    neighbor_schedule: Schedule = Field(..., description="The neighbor schedule")
    neighbor_base_schedule: Schedule = Field(
        ..., description="The neighbor base schedule"
    )

    @property
    def current_network(self) -> Network:
        return self.current_schedule.network

    @property
    def neighbor_network(self) -> Network:
        return self.neighbor_schedule.network

    @property
    def current_task_graph(self) -> TaskGraph:
        return self.current_schedule.task_graph

    @property
    def neighbor_task_graph(self) -> TaskGraph:
        return self.neighbor_schedule.task_graph

    @property
    def current_makespan(self) -> float:
        return self.current_schedule.makespan

    @property
    def current_base_makespan(self) -> float:
        return self.current_base_schedule.makespan

    @property
    def neighbor_makespan(self) -> float:
        return self.neighbor_schedule.makespan

    @property
    def neighbor_base_makespan(self) -> float:
        return self.neighbor_base_schedule.makespan

    @property
    def current_energy(self) -> float:
        return self.current_makespan / self.current_base_makespan

    @property
    def neighbor_energy(self) -> float:
        return self.neighbor_makespan / self.neighbor_base_makespan

    @property
    def accept_probability(self) -> float:
        energy_ratio = self.neighbor_energy / self.current_energy
        return math.exp(-energy_ratio / self.temperature) if energy_ratio <= 1 else 1.0


class SimulatedAnnealingRun(BaseModel):
    """Metadata and results for a simulated annealing run."""

    name: str = Field(..., description="Name of this run.")
    path: pathlib.Path = Field(..., description="Path to the run directory.")

    scheduler: SchedulerName = Field(..., description="Scheduler being tested.")
    base_scheduler: SchedulerName = Field(
        ..., description="Base scheduler for comparison."
    )
    config: SimulatedAnnealingConfig = Field(..., description="Configuration used.")

    initial_network: Network = Field(..., description="Initial network.")
    initial_task_graph: TaskGraph = Field(..., description="Initial task graph.")

    def iter_iterations(self) -> Generator[SimulatedAnnealingIteration, None, None]:
        """Iterate over all saved iterations for this run."""
        iterations_dir = self.path / "iterations"
        for path in sorted(iterations_dir.glob("*.json")):
            yield SimulatedAnnealingIteration.model_validate_json(path.read_text())

    @property
    def num_iterations(self) -> int:
        """Number of iterations completed."""
        iterations_dir = self.path / "iterations"
        return len(list(iterations_dir.glob("*.json")))

    @property
    def best_iteration(self) -> SimulatedAnnealingIteration:
        """Get the iteration with the best energy."""
        best_energy = -math.inf
        best_iter = None
        for iter_data in self.iter_iterations():
            if iter_data.current_energy > best_energy:
                best_energy = iter_data.current_energy
                best_iter = iter_data
        if best_iter is None:
            raise ValueError("No iterations found to determine best iteration")
        return best_iter


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

        self._scheduler_name: SchedulerName = scheduler
        self._base_scheduler_name: SchedulerName = base_scheduler
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
                path=self._run_dir,
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
            TaskGraphDeleteDependency,
            TaskGraphAddDependency,
            TaskGraphChangeDependencyWeight,
            TaskGraphChangeTaskWeight,
            NetworkChangeEdgeWeight,
            NetworkChangeNodeWeight,
        )

        name_to_class: Dict[str, Type[Change]] = {
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

    def get_iteration(
        self, iteration_num: int
    ) -> Optional[SimulatedAnnealingIteration]:
        """Load an iteration from disk."""
        iteration_path = self._iterations_dir / f"{iteration_num:06d}.json"
        if not iteration_path.exists():
            return None
        return SimulatedAnnealingIteration.model_validate_json(
            iteration_path.read_text()
        )

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
    def load(
        cls, name: str, data_dir: Optional[pathlib.Path] = None
    ) -> "SimulatedAnnealing":
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
        else:
            iteration = 0
            temp = config.max_temp
            current_network = self._run.initial_network
            current_task_graph = self._run.initial_task_graph

            # Calculate initial energy
            current_schedule = self._scheduler.schedule(
                current_network, current_task_graph
            )
            current_base_schedule = self._base_scheduler.schedule(
                current_network, current_task_graph
            )
            current_energy = current_schedule.makespan / current_base_schedule.makespan

            # Save initial iteration
            initial_iter = SimulatedAnnealingIteration(
                iteration=iteration,
                temperature=temp,
                change=None,
                current_schedule=current_schedule,
                current_base_schedule=current_base_schedule,
                neighbor_schedule=current_schedule,
                neighbor_base_schedule=current_base_schedule,
            )
            self._save_iteration(initial_iter)
            self._save_run()
            yield initial_iter
            iteration = 1

        # Main loop
        while iteration < config.max_iterations and temp > config.min_temp:
            # Apply random change
            ChangeClass = random.choice(self._change_types)
            change_result, neighbor_network, neighbor_task_graph = (
                ChangeClass.apply_random(current_network, current_task_graph)
            )
            change = cast(Optional[ChangeType], change_result)

            # Compute schedules and energy
            neighbor_schedule = self._scheduler.schedule(
                neighbor_network, neighbor_task_graph
            )
            neighbor_base_schedule = self._base_scheduler.schedule(
                neighbor_network, neighbor_task_graph
            )
            neighbor_energy = (
                neighbor_schedule.makespan / neighbor_base_schedule.makespan
            )

            # Acceptance probability
            energy_ratio = neighbor_energy / current_energy
            accept_probability = (
                math.exp(-energy_ratio / temp) if energy_ratio <= 1 else 1.0
            )
            accepted = random.random() < accept_probability

            # Get current makespans for logging
            current_schedule = self._scheduler.schedule(
                current_network, current_task_graph
            )
            current_base_schedule = self._base_scheduler.schedule(
                current_network, current_task_graph
            )

            # Create and save iteration
            iter_data = SimulatedAnnealingIteration(
                iteration=iteration,
                temperature=temp,
                change=change,
                current_schedule=current_schedule,
                current_base_schedule=current_base_schedule,
                neighbor_schedule=neighbor_schedule,
                neighbor_base_schedule=neighbor_base_schedule,
            )
            self._save_iteration(iter_data)

            # Update state if accepted
            if accepted:
                current_network = neighbor_network
                current_task_graph = neighbor_task_graph
                current_energy = neighbor_energy

            self._save_run()

            yield iter_data

            # Cool down
            temp *= config.cooling_rate
            iteration += 1

        self._save_run()

    def execute(self, progress: bool = True) -> SimulatedAnnealingRun:
        """Run simulated annealing to completion.

        Args:
            progress: Whether to print progress.

        Returns:
            The final run metadata.
        """
        best_energy = -math.inf
        for iter_data in self.run_iter():
            if progress:
                best_energy = max(best_energy, iter_data.current_energy)
                print(
                    f"\r[Iter {iter_data.iteration}/{self._config.max_iterations}] "
                    f"Temp: {iter_data.temperature:.2f} | "
                    f"Energy: {iter_data.current_energy:.4f} | "
                    f"Best: {best_energy:.4f}",
                    end="",
                )
        if progress:
            print()  # Newline after progress
        return self._run
