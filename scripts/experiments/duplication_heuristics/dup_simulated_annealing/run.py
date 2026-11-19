from saga import Scheduler, Network, TaskGraph
import pathlib
from .simulated_annealing import DupSimulatedAnnealingConfig, DupSimulatedAnnealing, DupSimulatedAnnealingSavedIteration
from saga.schedulers import HeftScheduler, CpopScheduler
from saga.schedulers.data.wfcommons import get_wfcommons_instance


def run_experiments(
        scheduler: type[Scheduler],
        max_iterations: int = 100,
        max_temp: float = 100.0,
        min_temp: float = 0.1,
        cooling_rate: float = 0.99,
        output_path: pathlib.Path | None = None
    ) -> None:
    
    config = DupSimulatedAnnealingConfig(
        max_iterations=max_iterations, max_temp=max_temp, min_temp=min_temp, cooling_rate=cooling_rate
    )
    network, task_graph = get_wfcommons_instance(
        recipe_name="epigenomics", ccr=5.0, max_size_multiplier=5, num_nodes=4
    )
    sa = DupSimulatedAnnealing(
        name="testing", task_graph=task_graph, network=network, scheduler_class=scheduler, config=config
    )
    sa.save()
    sa.execute(progress=True)

def main():
    run_experiments(scheduler=HeftScheduler, max_iterations=2)

if __name__ == "__main__":
    main()