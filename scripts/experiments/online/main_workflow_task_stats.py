from typing import Dict, List
from matplotlib import pyplot as plt
import pathlib

from saga.schedulers.data.wfcommons import generate_rvs, get_workflow_task_info, recipes
from saga.utils.random_variable import RandomVariable

THISDIR = pathlib.Path(__file__).resolve().parent
SAVEDIR = THISDIR / "plot-workflow-stats"

def main():
    for workflow_name in recipes.keys():
        task_info = get_workflow_task_info(workflow_name)
        task_rvs: Dict[str, RandomVariable] = {}
        for task, details in task_info.items():
            print(f"Processing task: {task}")
            rvs = generate_rvs(
                distribution=details["runtime"]["distribution"],
                min_value=details["runtime"]["min"],
                max_value=details["runtime"]["max"],
                num=1000
            )
            task_rvs[task] = RandomVariable(rvs)

        # Plot grid of histograms for task RVs
        ntasks = len(task_rvs)
        fig, axes = plt.subplots(ncols=2, nrows=(ntasks + 1) // 2, figsize=(12, 6))
        axes: List[plt.Axes] = axes.flatten()
        for i, (task, rv) in enumerate(task_rvs.items()):
            print(f"Plotting task: {task}")
            axes[i].hist(
                rv.samples,
                bins=30, alpha=0.5,
                density=False, log=True,
                label=f"{task} samples"
            )
            axes[i].set_title(f"Task: {task}")
            axes[i].set_xlabel("Runtime")
            axes[i].set_ylabel("Density")
            # add variance to text
            max_val = max(rv.samples)
            min_val = min(rv.samples)
            axes[i].text(
                0.05, 0.95, 
                f"STD: {rv.std():.2f}, Min: {min_val:.2f}, Max: {max_val:.2f}, Mean: {rv.mean():.2f}",
                transform=axes[i].transAxes, fontsize=10,
                verticalalignment='top', 
                bbox=dict(facecolor='white', alpha=0.5)
            )
        plt.tight_layout()
        savepath = SAVEDIR / workflow_name / "task_rvs_grid.png"
        savepath.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(savepath)
        plt.close()


if __name__ == "__main__":
    main()