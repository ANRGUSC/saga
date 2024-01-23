import logging
import pathlib
import pickle
import tempfile

from exp_compare_all import SCHEDULERS, run_experiments
from simulated_annealing import SimulatedAnnealing

from saga.schedulers import HybridScheduler

thisdir = pathlib.Path(__file__).parent.absolute()

def run(savepath): # pylint: disable=too-many-locals, too-many-statements
    """Run ensemble experiment.

    Iteratively construct the best hybrid algorithm.
    1. Start with alg with least max MR (MinMinScheduler)
    2. For each remaining alg:
    4.   Run alg vs. hybrid
    5. Add best alg to hybrid
    6. Repeat from 2 until all algs are in hybrid

    Intuitively, this is adding algorithms that cover problem instances the hybrid doesn't
    currently cover.
    """
    logging.basicConfig(level=logging.WARNING)

    hybrid_algs = ["MinMin"]
    remaining_algs = [s for s in SCHEDULERS.keys() if s not in hybrid_algs]
    round_i = 0
    while remaining_algs:
        round_i += 1
        print(f"Iteration {round_i}")
        print(f"  Hybrid: {hybrid_algs}")
        print(f"  Remaining: {remaining_algs}")

        hybrid_alg = HybridScheduler([SCHEDULERS[s] for s in hybrid_algs])
        pairs = [(("Hybrid", hybrid_alg), (s, SCHEDULERS[s])) for s in remaining_algs]
        # create temp directory to store results
        with tempfile.TemporaryDirectory() as tempdir:
            tempdir = pathlib.Path(tempdir)
            run_experiments(
                scheduler_pairs=pairs,
                max_iterations=1000,
                num_tries=5,
                max_temp=10,
                min_temp=0.1,
                cooling_rate=0.99,
                skip_existing=True,
                output_path=tempdir
            )

            # load results
            max_makespan_ratio, best_alg = 0, None
            for scheduler_name in remaining_algs:
                res: SimulatedAnnealing = pickle.loads(
                    tempdir.joinpath(f"{scheduler_name}/Hybrid.pkl").read_bytes()
                )
                makespan_ratio = res.iterations[-1].best_energy
                if makespan_ratio > max_makespan_ratio:
                    max_makespan_ratio = makespan_ratio
                    best_alg = scheduler_name

            print(f"Adding {best_alg} to hybrid (MR={max_makespan_ratio})")
            hybrid_algs.append(best_alg)
            remaining_algs.remove(best_alg)

if __name__ == "__main__":
    savedir = thisdir.joinpath("results/ensemble")
    savedir.mkdir(exist_ok=True, parents=True)
    run(savedir)