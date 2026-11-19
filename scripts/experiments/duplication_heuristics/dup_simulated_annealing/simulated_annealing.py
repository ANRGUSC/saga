from scripts.experiments.duplication_heuristics.heuristics import task_score, compute_schedule_features, task_can_be_scored, is_super_node
from saga import Schedule, Scheduler, TaskGraph, Network
from saga.schedulers import HeftScheduler, CpopScheduler
from dataclasses import dataclass, field
from pydantic import BaseModel, Field, ConfigDict, model_validator, PrivateAttr
import pathlib
from copy import deepcopy
import random
import math
from typing import Literal, Generator
import json
import base64
import pickle

# immutable rep of a dup plan used for caching
StateKey = tuple[tuple[str, tuple[str, ...]], ...]
# allows for tasks w/ a score of 0.0 to still have a chance to be selected, if it has positive estimated dup benefit
EPS = 1e-6

SchedulerName = Literal["HEFT", "CPoP"]
SCHEDULERS: dict[SchedulerName, type[Scheduler]] = {
    "HEFT": HeftScheduler,
    "CPoP": CpopScheduler
}

@dataclass
class DupState:
    """ Stores everything associated with one simulated annealing state. """
    dup_plan: dict[str, list[str]]
    schedule: Schedule
    makespan: float
    task_scores: dict[str, float] = field(default_factory=dict)
    task_pool: dict[str, list[tuple[str, float]]] = field(default_factory=dict)

class DupSimulatedAnnealingConfig(BaseModel):
    """Configuration for simulated annealing."""
    max_iterations: int = Field(default=1000, description="Maximum number of iterations.")
    max_temp: float = Field(default=100.0, description="Maximum (starting) temperature.")
    min_temp: float = Field(default=0.1, description="Minimum temperature (stopping condition).")
    cooling_rate: float = Field(default=0.99, description="Cooling rate per iteration.")

# "increase" & "decrease" are only used when runnning heuristic_free simulated annealing
MoveType = Literal["increase", "decrease", "add", "remove", "swap"]

class DupSimulatedAnnealingSavedIteration(BaseModel):
    """Data saved after a completed SA iteration."""
    iteration: int
    temperature: float
    move: MoveType
    accepted: bool
    current_dup_plan: dict[str, list[str]]
    current_makespan: float
    resulting_dup_plan: dict[str, list[str]]
    resulting_makespan: float
    makespan_ratio: float
    best_dup_plan: dict[str, list[str]]
    best_makespan: float
    best_iteration: int | None = None
    # heuristic-guided SA only 
    task_name: str | None = None
    task_score: float | None = None
    task_rank: int | None = None
    candidate_pool_size: int | None = None
    # track all tasks involved in a move
    added_tasks: list[str] = Field(default_factory=list)
    removed_tasks: list[str] = Field(default_factory=list)
    random_state: str

class DupSimulatedAnnealingIteration(BaseModel):
    """Data for a single simulated annealing iteration."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    iteration: int = Field(..., description="The iteration number.")
    temperature: float = Field(..., description="The current temperature.")
    current_state: DupState = Field(..., description="The current state.")
    neighbor_state: DupState = Field(..., description="The neighbor state.")
    move: MoveType = Field(..., description="The move applied.")
    accepted: bool | None = Field(default=None, description="Stores whether moves were accepted or not.")

    @property
    def current_makespan(self) -> float: 
        return self.current_state.makespan

    @property
    def neighbor_makespan(self) -> float: 
        return self.neighbor_state.makespan

    @property
    def current_num_dups(self) -> int: 
        return len(self.current_state.dup_plan)

    @property
    def neighbor_num_dups(self) -> int: 
        return len(self.neighbor_state.dup_plan)

    @property
    def delta(self) -> float:
        makespan_ratio = self.neighbor_makespan / self.current_makespan
        return makespan_ratio - 1

    @property
    def accept_probability(self) -> float: 
        # delta allows neighbor makespan that is slightly worse to have a higher chance to be selected as opposed to one who is not
        return 1.0 if self.delta <= 0 else math.exp(-self.delta / self.temperature)

def encode_random_state() -> str:
    """Encode Python's random state as a JSON compatible string."""
    return base64.b64encode(pickle.dumps(random.getstate())).decode("ascii")

def restore_random_state(encoded_state: str) -> None:
    """Restore Python's random state from a saved string."""
    random.setstate(pickle.loads(base64.b64decode(encoded_state)))


class DupSimulatedAnnealing(BaseModel):
    """"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str = Field(..., description="Name of this SA run.")
    data_dir: pathlib.Path = Field(
        default_factory=lambda: pathlib.Path(__file__).parent / "results", 
        description="Directory to store SA results."
    )
    task_graph: TaskGraph
    network: Network
    baseline_state: DupState | None = Field(default=None, description="The baseline state with no task duplication.")
    best_state: DupState | None = Field(default=None, description="The best state found.")
    best_iteration: int | None = None 
    scheduler_class: type[Scheduler]
    config: DupSimulatedAnnealingConfig = Field(default_factory=DupSimulatedAnnealingConfig)
    state_cache: dict[StateKey, DupState] = Field(default_factory=dict)
    instance_seed: int | None = None
    sa_seed: int | None = None

    _run_dir: pathlib.Path = PrivateAttr()
    _iterations_dir: pathlib.Path = PrivateAttr()

    @model_validator(mode="after")
    def _setup_runtime(self) -> "DupSimulatedAnnealing":
        """Initialize runtime directories."""
        self._run_dir = self.data_dir / self.name
        self._iterations_dir = self._run_dir / "iterations"
        return self

    @property
    def path(self) -> pathlib.Path:
        """Return the run directory."""
        return self._run_dir

    def _ensure_dirs(self) -> None:
        """Create directories for this SA run."""
        self._run_dir.mkdir(parents=True, exist_ok=True)
        self._iterations_dir.mkdir(parents=True, exist_ok=True)

    def save(self) -> None: 
        """Save SA run metadata."""
        run_path = self._run_dir / "run.json"
        if run_path.exists():
            raise FileExistsError(
                f"Experiment '{self.name}' already exists at {self._run_dir}. "
                "Use DupSimulatedAnnealing.load() to resume it."
            )
        self._ensure_dirs()
        metadata = {
            "name": self.name, 
            "scheduler": next(
                name for name, scheduler_class in SCHEDULERS.items() if scheduler_class is self.scheduler_class
            ),
            "config": self.config.model_dump(mode="json"),
            "num_tasks": len(self.task_graph.tasks),
            "num_processors": len(self.network.nodes),
            "instance_seed": self.instance_seed,
            "sa_seed": self.sa_seed
        }
        run_path = self._run_dir / "run.json"
        run_path.write_text(json.dumps(metadata, indent=2))
        network_path = self._run_dir / "network.json"
        network_path.write_text(self.network.model_dump_json(indent=2))
        task_graph_path = self._run_dir / "task_graph.json"
        task_graph_path.write_text(self.task_graph.model_dump_json(indent=2))

    def _save_iteration(self, iter: DupSimulatedAnnealingSavedIteration) -> None:
        """Save a completed SA iteration to disk."""
        self._ensure_dirs()
        iter_path = self._iterations_dir / f"{iter.iteration:06d}.json"
        if iter_path.exists():
            raise FileExistsError(
                f"Iteration {iter.iteration} already exists at {iter_path}."
            )
        
        temp_path = iter_path.with_suffix(".json.tmp")
        temp_path.write_text(iter.model_dump_json(indent=2))
        temp_path.replace(iter_path)

    def get_iteration(self, iter_num: int) -> DupSimulatedAnnealingSavedIteration | None: 
        """Load a saved SA iteration from disk."""
        iter_path = self._iterations_dir / f"{iter_num:06d}.json"
        if not iter_path.exists():
            return None
        return DupSimulatedAnnealingSavedIteration.model_validate_json(iter_path.read_text())

    def get_last_iteration(self) -> DupSimulatedAnnealingSavedIteration | None:
        """Return the most recent saved SA iteration."""
        iter_files = sorted(self._iterations_dir.glob("*json"))
        if not iter_files:
            return None
        return DupSimulatedAnnealingSavedIteration.model_validate_json(iter_files[-1].read_text())
    
    @classmethod
    def load(cls, name: str, data_dir: pathlib.Path) -> "DupSimulatedAnnealing":
        """Load a previously saved SA experiment."""
        run_dir = data_dir / name
        metadata = json.loads((run_dir / "run.json").read_text())
        network = Network.model_validate_json((run_dir / "network.json").read_text())
        task_graph = TaskGraph.model_validate_json((run_dir / "task_graph.json").read_text())
        return cls(
            name=name, data_dir=data_dir, network=network, task_graph=task_graph,
            scheduler_class=SCHEDULERS[metadata["scheduler"]], 
            config=DupSimulatedAnnealingConfig.model_validate(metadata["config"]),
            instance_seed=metadata.get("instance_seed"),
            sa_seed=metadata.get("sa_seed")
        )

    def get_changed_tasks(self, old_plan: dict[str, list[str]], new_plan: dict[str, list[str]]) -> tuple[list[str], list[str]]:
        """Identify tasks added to or removed from a duplication plan."""
        added = [task for task in new_plan if task not in old_plan]
        removed = [task for task in old_plan if task not in new_plan]
        return added, removed

    def make_state_key(self, dup_plan: dict[str, list[str]]) -> StateKey:
        """ Creates a immutable representation of a duplication plan. This allows states to be cached. """
        return tuple(
            sorted((task_name, tuple(sorted(target_processors))) for task_name, target_processors in dup_plan.items())
        )

    def schedule_dup_plan(self, dup_plan: dict[str, list[str]]) -> Schedule:
        """Duplicates the selected task(s) from the duplication plan and generates a schedule."""
        dup_factor = 1 + max((len(targets) for targets in dup_plan.values()), default=0)
        scheduler = self.scheduler_class(duplication_factor=dup_factor, duplication_targets=dup_plan)
        return scheduler.schedule(self.network, self.task_graph)

    
    def compute_state_features(self, dup_plan: dict[str, list[str]], schedule: Schedule) -> tuple[dict[str, float], dict[str, list[tuple[str, float]]]]:
        """Compute heuristic scores and eligible duplication candidates for a state."""
        schedule_features = compute_schedule_features(self.task_graph, schedule)
        scores: dict[str, float] = {}
        task_pool: dict[str, list[tuple[str, float]]] = {}

        for task_name, raw_features in schedule_features["raw"].items():
            if task_name in dup_plan:
                continue

            existing_processors = {
                task.node
                for task in schedule.get_scheduled_tasks(task_name)
            }

            available_processors = [
                proc.name
                for proc in self.network.nodes
                if proc.name not in existing_processors
            ]

            if not available_processors:
                continue

            # Prioritize processors currently executing the task's children.
            child_processors = [
                (proc_name, benefit)
                for proc_name, benefit in raw_features["target_processors"]
                if proc_name not in existing_processors
            ]

            if child_processors:
                candidate_processors = child_processors
            else:
                # Fallback when no child-hosting processor is available.
                candidate_processors = [
                    (proc_name, 0.0)
                    for proc_name in available_processors
                ]

            scores[task_name] = task_score(
                task_name, self.task_graph, self.network,
                schedule, schedule_features
            )
            task_pool[task_name] = candidate_processors

        return scores, task_pool


    def get_state(self, dup_plan: dict[str, list[str]]) -> DupState:
        """Returns the state for a duplication plan, reusing a cached state when available."""
        key = self.make_state_key(dup_plan)
        if key in self.state_cache:
            return self.state_cache[key]

        schedule = self.schedule_dup_plan(dup_plan)
        # commented out for now in order to use heuristic-free simulated annealing approach 
        task_scores, task_pool = self.compute_state_features(dup_plan, schedule) 
        state = DupState(
            dup_plan=deepcopy(dup_plan), 
            schedule=schedule, 
            makespan=schedule.makespan, 
            task_scores=task_scores, 
            task_pool=task_pool
        )
        self.state_cache[key] = state
        return state

    def select_task_from_pool(self, state: DupState, excluded_tasks: set[str] | None = None) -> str:
        """
        Selects a task from the task pool to be added to the duplication plan. 
        The probability a task is selected is heuristic-weighted.
        """
        if not state.task_pool: 
            raise ValueError("The task pool is currently empty.")

        excluded_tasks = excluded_tasks or set()
        tasks = [task_name for task_name in state.task_pool if task_name not in excluded_tasks]
        if not tasks: 
            raise ValueError("No eligible tasks available for selection.")
        
        scores = [state.task_scores[task_name] + EPS for task_name in tasks]
        task = random.choices(population=tasks, weights=scores, k=1)[0]
        return task
    
    def add_task_to_dup_plan(self, state: DupState, excluded_tasks: set[str] | None = None) -> DupState:
        """Add a task to the duplication plan using estimated processor benefits."""
        if not state.task_pool:
            raise ValueError("Cannot add a task from an empty task pool.")

        selected_task = self.select_task_from_pool(state, excluded_tasks)
        processor_candidates = state.task_pool[selected_task]

        # select every processor with a positive estimated benefit
        selected_processors = [
            proc_name
            for proc_name, benefit in processor_candidates
            if benefit > 0.0
        ]

        # if none have positive benefits, randomly select one candidate
        if not selected_processors:
            selected_processors = [
                random.choice(processor_candidates)[0]
            ]

        new_dup_plan = deepcopy(state.dup_plan)
        new_dup_plan[selected_task] = selected_processors
        return self.get_state(new_dup_plan)

    def remove_task_from_dup_plan(self, state: DupState) -> tuple[DupState, str]:
        """
        Randomly removes a currently duplicated task from the duplication plan and 
        returns the resulting state and removed task.
        """
        # after removing a task, remove it from the previous dup plan's task pool
        if not state.dup_plan: 
            raise ValueError("A task removal cannot be done to an empty duplication plan.")

        selected_task = random.choice(list(state.dup_plan.keys()))
        new_dup_plan = deepcopy(state.dup_plan)
        del new_dup_plan[selected_task]
        return self.get_state(new_dup_plan), selected_task

    def swap_task_from_dup_plan(self, state: DupState) -> DupState:
        """Swaps a task from the duplication plan with a task from the reduced state."""
        if not state.dup_plan: 
            raise ValueError("A task swap cannot be done to an empty duplication.")

        tasks_to_remove = list(state.dup_plan.keys())
        random.shuffle(tasks_to_remove)

        for removed_task in tasks_to_remove:
            new_dup_plan = deepcopy(state.dup_plan)
            del new_dup_plan[removed_task]
            reduced_state = self.get_state(new_dup_plan)
            available_tasks = [task_name for task_name in reduced_state.task_pool if task_name != removed_task]

            if not available_tasks: 
                continue
        
            return self.add_task_to_dup_plan(reduced_state, {removed_task})
        return state

    def propose_neighbor(self, state: DupState) -> tuple[MoveType, DupState]:
        """Selects a valid move to create a neighbor duplication state."""
        available_moves: list[MoveType] = []

        if state.task_pool: 
            available_moves.append("add")

        if state.dup_plan: 
            available_moves.append("remove")

        if state.dup_plan and state.task_pool:
            available_moves.append("swap")

        if not available_moves:
            raise ValueError("No valid moves are available based off the current state.")

        move = random.choice(available_moves)

        if move == "add":
            neighbor_state = self.add_task_to_dup_plan(state)

        elif move == "remove":
            neighbor_state, _ = self.remove_task_from_dup_plan(state)

        else: 
            neighbor_state = self.swap_task_from_dup_plan(state)

        return move, neighbor_state


    #######################################
    # helpers for heuristic-free approach
    ######################################

    def get_valid_tasks(self, state: DupState) -> list[str]:
        """
        Returns all valid tasks available for duplication. 
        A task is considered valid if they have at least one child and its 
        duplication factor must be below the number of processors or children, whichever is smaller.
        """
        valid_tasks = []
        for task in self.task_graph.tasks:
            task_name = task.name
            if is_super_node(task_name):
                continue

            children = [edge.target for edge in self.task_graph.out_edges(task_name) if not is_super_node(edge.target)]
            if not children: 
                continue

            current_dup_factor = 1 + len(state.dup_plan.get(task_name, []))
            max_dup_factor = len(self.network.nodes)
            if current_dup_factor < max_dup_factor:
                valid_tasks.append(task_name)
        return valid_tasks

    def find_best_dup_processor(self, state: DupState, task_name: str) -> DupState:
        """
        Exhaustively finds the best processor (improves makespan the most) for the selected task to duplicate on.
        """

        task_instances = state.schedule.get_scheduled_tasks(task_name)
        occupied_processors = {task_copy.node for task_copy in task_instances}
        free_processors = [node.name for node in self.network.nodes if node.name not in occupied_processors]
        if not free_processors:
            raise ValueError(f"Task {task_name} is already scheduled on every processor.")

        best_state: DupState | None = None

        for proc_name in free_processors:
            candidate_dup_plan = deepcopy(state.dup_plan)
            candidate_dup_plan.setdefault(task_name, [])
            candidate_dup_plan[task_name].append(proc_name)
            candidate_state = self.get_state(candidate_dup_plan)

            if best_state is None or candidate_state.makespan < best_state.makespan:
                best_state = candidate_state

        if best_state is None: 
            raise ValueError(f"No valid duplicate placement found for task {task_name}.")
        return best_state

    def find_best_dup_removal(self, state: DupState, task_name: str) -> DupState:
        target_processors = state.dup_plan.get(task_name, [])
        if not target_processors:
            raise ValueError(f"Task {task_name} has no duplicates to remove.")

        best_state: DupState | None = None

        for proc_name in target_processors:
            candidate_dup_plan = deepcopy(state.dup_plan)
            candidate_dup_plan[task_name].remove(proc_name)
            # dup factor is now = 1, so task can be removed from dup plan
            if not candidate_dup_plan[task_name]:
                del candidate_dup_plan[task_name]

            candidate_state = self.get_state(candidate_dup_plan)
            if best_state is None or candidate_state.makespan < best_state.makespan: 
                best_state = candidate_state

        if best_state is None: 
            raise ValueError(f"No valid duplicate removal found for task {task_name}.")

        return best_state

    def select_random_task(self, state: DupState) -> str:
        """Randomly selects a valid task to be duplicated."""

        valid_tasks = []
        for task in self.task_graph.tasks:
            task_name = task.name
            if is_super_node(task_name) or not task_can_be_scored(task_name, self.task_graph):
                continue

            task_instances = state.schedule.get_scheduled_tasks(task_name)
            occupied_processors = {task.node for task in task_instances}
            if len(occupied_processors) >= len(self.network.nodes):
                continue

            valid_tasks.append(task_name)

        if not valid_tasks:
            raise ValueError("No valid tasks eligible for duplication.")

        return random.choice(valid_tasks)

    def propose_random_neighbor(self, state: DupState) -> tuple[MoveType, DupState]:
        available_moves: list[MoveType] = []
        move_weights = []
        valid_tasks = self.get_valid_tasks(state)

        if valid_tasks:
            available_moves.append("increase")
            move_weights.append(0.7)

        if state.dup_plan:
            available_moves.append("decrease")
            move_weights.append(0.3)

        if not available_moves:
            raise ValueError("No valid neighbor moves are available.")

        move = random.choices(population=available_moves, weights=move_weights, k=1)[0]
        if move == "increase":
            task_name = random.choice(valid_tasks)
            neighbor_state = self.find_best_dup_processor(state, task_name)
        else: 
            task_name = random.choice(list(state.dup_plan.keys()))
            neighbor_state = self.find_best_dup_removal(state, task_name)

        return move, neighbor_state

    # heuristic-free simulated annealing approach
    def run_heuristic_free_iter(self) -> Generator[DupSimulatedAnnealingIteration, None, None]:
        """"""
        config = self.config
        self.baseline_state = self.get_state({}) # baseline schedule
        last_iter = self.get_last_iteration()

        if last_iter is None: 
            # start new experiment 
            iteration = 0 
            temp = config.max_temp
            current_state = self.baseline_state
            self.best_state = current_state
            self.best_iteration = None
        else: 
            # continue last iteration
            iteration = last_iter.iteration + 1
            temp = last_iter.temperature * config.cooling_rate
            current_state = self.get_state(last_iter.resulting_dup_plan)
            self.best_state = self.get_state(last_iter.best_dup_plan)
            self.best_iteration = last_iter.best_iteration
            restore_random_state(last_iter.random_state)

        while iteration < config.max_iterations and temp > config.min_temp:
            valid_tasks = self.get_valid_tasks(current_state)
            if not valid_tasks and not current_state.dup_plan:
                break

            move, neighbor_state = self.propose_random_neighbor(current_state)
            iter_data = DupSimulatedAnnealingIteration(
                iteration=iteration, 
                temperature=temp, 
                current_state=current_state, 
                neighbor_state=neighbor_state, 
                move=move
            )
            iter_data.accepted = random.random() < iter_data.accept_probability
            if iter_data.accepted:
                current_state = neighbor_state

                if self.best_state is None or current_state.makespan < self.best_state.makespan:
                    self.best_state = current_state
                    self.best_iteration = iteration
                    
            added_tasks, removed_tasks = self.get_changed_tasks(iter_data.current_state.dup_plan, neighbor_state.dup_plan)
            saved_iteration = DupSimulatedAnnealingSavedIteration(
                iteration=iteration, temperature=temp, move=move, accepted=iter_data.accepted,
                current_dup_plan=iter_data.current_state.dup_plan, current_makespan=iter_data.current_state.makespan,
                resulting_dup_plan=current_state.dup_plan, resulting_makespan=current_state.makespan,
                makespan_ratio=current_state.makespan / self.baseline_state.makespan, 
                best_dup_plan=self.best_state.dup_plan, best_makespan=self.best_state.makespan, best_iteration=self.best_iteration,
                added_tasks=added_tasks, removed_tasks=removed_tasks,
                random_state=encode_random_state()
            )
            self._save_iteration(saved_iteration)
            temp *= config.cooling_rate
            iteration += 1
            yield iter_data

    def get_task_selection_info(self, state: DupState, task_name: str) -> dict:
        """Get a task's score and rank within the current candidate pool."""

        if task_name not in state.task_scores:
            return {
                "task_name": task_name,
                "task_score": None,
                "task_rank": None,
                "candidate_pool_size": len(state.task_pool)
            }

        score = state.task_scores[task_name]

        rank = 1 + sum(
            other_score > score
            for other_score in state.task_scores.values()
        )

        return {
            "task_name": task_name,
            "task_score": score,
            "task_rank": rank,
            "candidate_pool_size": len(state.task_pool)
        }

    # heuristic simulated annealing approach
    def run_heuristic_iter(self) -> Generator[DupSimulatedAnnealingIteration, None, None]:
        """Run simulated annealing, yielding each iteration.
        
        Resumes from last saved iteration if interrupted.

        Yields:
            DupSimulatedAnnealingIteration for each step.
        """
        config = self.config
        self.baseline_state = self.get_state({}) # baseline schedule
        last_iter = self.get_last_iteration()
        if last_iter is None: 
            # start new experiment 
            iteration = 0 
            temp = config.max_temp
            current_state = self.baseline_state
            self.best_state = current_state
            self.best_iteration = None
        else: 
            # continue last iteration
            iteration = last_iter.iteration + 1
            temp = last_iter.temperature * config.cooling_rate
            current_state = self.get_state(last_iter.resulting_dup_plan)
            self.best_state = self.get_state(last_iter.best_dup_plan)
            self.best_iteration = last_iter.best_iteration
            restore_random_state(last_iter.random_state)

        while iteration < config.max_iterations and temp > config.min_temp:
            if not current_state.task_pool and not current_state.dup_plan:
                break

            move, neighbor_state = self.propose_neighbor(current_state)
            added_tasks, removed_tasks = self.get_changed_tasks(current_state.dup_plan, neighbor_state.dup_plan)
            selection_info = {
                "task_name": None,
                "task_score": None,
                "task_rank": None,
                "candidate_pool_size": None,
            }

            if move == "add" and added_tasks:
                selection_info = self.get_task_selection_info(
                    current_state, added_tasks[0]
                )

            elif move == "remove" and removed_tasks:
                selection_info["task_name"] = removed_tasks[0]

            elif move == "swap" and added_tasks:
                reduced_plan = deepcopy(current_state.dup_plan)

                for task in removed_tasks:
                    reduced_plan.pop(task, None)
                reduced_state = self.get_state(reduced_plan)
                selection_info = self.get_task_selection_info(reduced_state, added_tasks[0])

            iter_data = DupSimulatedAnnealingIteration(
                iteration=iteration, 
                temperature=temp, 
                current_state=current_state, 
                neighbor_state=neighbor_state, 
                move=move
            )

            prev_best_makespan = self.best_state.makespan
            iter_data.accepted = random.random() < iter_data.accept_probability
            if iter_data.accepted:
                current_state = neighbor_state

                if self.best_state is None or current_state.makespan < self.best_state.makespan:
                    self.best_state = current_state
                    self.best_iteration = iteration

            saved_iteration = DupSimulatedAnnealingSavedIteration(
                iteration=iteration, temperature=temp, move=move, accepted=iter_data.accepted,
                current_dup_plan=iter_data.current_state.dup_plan, current_makespan=iter_data.current_state.makespan,
                resulting_dup_plan=current_state.dup_plan, resulting_makespan=current_state.makespan,
                makespan_ratio=current_state.makespan / self.baseline_state.makespan,
                best_dup_plan=self.best_state.dup_plan, best_makespan=self.best_state.makespan, best_iteration=self.best_iteration,
                task_name=selection_info["task_name"], task_score=selection_info["task_score"], 
                task_rank=selection_info["task_rank"], candidate_pool_size=selection_info["candidate_pool_size"],
                added_tasks=added_tasks, removed_tasks=removed_tasks,
                random_state=encode_random_state()
            )
            if self.best_state.makespan < prev_best_makespan:
                ranking_state = iter_data.current_state
                if move == "swap":
                    reduced_plan = deepcopy(ranking_state.dup_plan)
                    for task in removed_tasks:
                        reduced_plan.pop(task, None)
                    ranking_state = self.get_state(reduced_plan)

                rankings = sorted(
                    ranking_state.task_scores.items(),
                    key=lambda item: (-item[1], item[0])
                )

                ranking_data = {
                    "iteration": iteration,
                    "selected_task": selection_info["task_name"],
                    "candidate_pool_size": len(ranking_state.task_pool),
                    "candidates": [
                        {
                            "task_name": task_name,
                            "task_score": score,
                            "task_rank": 1 + sum(
                                other_score > score
                                for other_score in ranking_state.task_scores.values()
                            )
                        }
                        for task_name, score in rankings
                    ]
                }

                rankings_dir = self._run_dir / "best_rankings"
                rankings_dir.mkdir(parents=True, exist_ok=True)

                ranking_path = rankings_dir / f"{iteration:06d}.json"
                ranking_path.write_text(json.dumps(ranking_data, indent=2))

            self._save_iteration(saved_iteration)
            temp *= config.cooling_rate
            iteration += 1
            yield iter_data


    def execute(self, progress: bool = True) -> "DupSimulatedAnnealing":
        """Run simulated annealing to completion."""

        for iter_data in self.run_heuristic_iter():
            if progress:
                print(
                    f"\n[Iter {iter_data.iteration}/{self.config.max_iterations}]"
                )

        completion_path = self._run_dir / "completed.json"
        completion_path.write_text(json.dumps({"completed": True}, indent=2))

        if progress: print()
        return self