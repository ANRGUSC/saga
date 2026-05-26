from functools import lru_cache
import pathlib
import matplotlib.pyplot as plt

from saga import Network, Schedule, TaskGraph
from saga.schedulers import CpopScheduler, HeftScheduler
from saga.utils.duplication import should_duplicate
from saga.utils.draw import draw_gantt
import saga.schedulers.heft as heft_mod
import saga.schedulers.cpop as cpop_mod


thisdir = pathlib.Path(__file__).parent.absolute()
savedir = thisdir / "outputs"
savedir.mkdir(exist_ok=True)


def select_should_duplicate(task_name: str, task_graph: TaskGraph, network: Network) -> bool:
    return task_name in ALLOWED_DUPLICATES


def build_uniform_network() -> Network:
	"""Create a small homogeneous network for the diamond example."""
	nodes = [
		("processor1", 1.0),
		("processor2", 1.0),
	]
	edges = [
		("processor1", "processor2", 1.0)
	]
	return Network.create(nodes=nodes, edges=edges)


def build_diamond_task_graph() -> TaskGraph:
	"""
               A
             /   \
            B     C
           / \    |
          D   E   F 
		    \ | /
		      G 
			 / \
			H   I  
      
    """
	tasks = [
        ("A", 2.0),
        ("B", 2.0), ("C", 2.0), 
		("D", 2.0), ("E", 2.0), ("F", 2.0),
		("G", 2.0), 
		("H", 2.0), ("I", 2.0)
    ]

	dependencies = [
        ("A", "B", 5.0), 
        ("B", "D", 8.0), ("B", "E", 6.0), 
		("C", "F", 20.0), 
		("D", "G", 5.0), 
		("E", "G", 5.0),
		("F", "G", 20.0),
		("G", "H", 20.0), ("G", "I", 2.0)
    ]
	return TaskGraph.create(tasks=tasks, dependencies=dependencies)


@lru_cache(maxsize=1)
def get_instance() -> tuple[Network, TaskGraph]:
	network = build_uniform_network()
	task_graph = build_diamond_task_graph()
	return network, task_graph


def draw_schedule(schedule: Schedule, name: str, xmax: float | None = None) -> None:
	ax = draw_gantt(schedule.mapping, use_latex=False, xmax=xmax)
	fig = ax.get_figure()
	if fig is not None:
		fig.savefig(str(savedir / f"{name}.png")) # type: ignore


def draw_side_by_side_schedules(
	scheduler_name: str,
	schedule_df1: Schedule,
	schedule_df2: Schedule,
) -> None:
	"""Save one image with duplication factors 1 and 2 side by side."""
	xmax = max(schedule_df1.makespan, schedule_df2.makespan)
	fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharey=True)

	draw_gantt(schedule_df1.mapping, use_latex=False, xmax=xmax, axis=axes[0])
	axes[0].set_title(f"{scheduler_name.upper()} (duplication_factor=1)")

	draw_gantt(schedule_df2.mapping, use_latex=False, xmax=xmax, axis=axes[1])
	axes[1].set_title(f"{scheduler_name.upper()} (duplication_factor=2)")

	fig.tight_layout()
	fig.savefig(str(savedir / f"{scheduler_name}_duplication_comparison.png"))
	plt.close(fig)

def run_experiment():
	network, task_graph = get_instance()

	scheduler_classes = [
		("heft", HeftScheduler),
		("cpop", CpopScheduler),
	]

	for name, scheduler_class in scheduler_classes:
		schedules_by_factor: dict[int, Schedule] = {}
		for duplication_factor in (1,2):
			scheduler = scheduler_class(duplication_factor=duplication_factor)
			schedule = scheduler.schedule(network, task_graph)
			schedules_by_factor[duplication_factor] = schedule
			print(
				f"\n{name.upper()} duplication_factor={duplication_factor} makespan: {schedule.makespan:.2f}"
			)
			for node_name, scheduled_tasks in sorted(schedule.items()):
				if not scheduled_tasks:
					continue
				print(f"  {node_name}:")
				for task in scheduled_tasks:
					print(f"    {task.name}: [{task.start:.2f}, {task.end:.2f}]")

		draw_side_by_side_schedules(
			scheduler_name=name,
			schedule_df1=schedules_by_factor[1],
			schedule_df2=schedules_by_factor[2],
		)

	print(f"\nSaved Gantt charts to: {savedir}")

def main() -> None:
    global ALLOWED_DUPLICATES

    test_cases = {
		"ABD_only": {"A", "B"},

    }

    heft_mod.should_duplicate = select_should_duplicate
    cpop_mod.should_duplicate = select_should_duplicate

    for case_name, duplicate_set in test_cases.items():
        print("\n" + "=" * 50)
        print(f"TEST CASE: {case_name}")
        print(f"ALLOWED_DUPLICATES = {duplicate_set}")

        ALLOWED_DUPLICATES = duplicate_set

        network, task_graph = get_instance()
        for task_name in ["A", "B", "C", "D", "E", "F", "G", "H", "I"]:
            duplicate = select_should_duplicate(task_name, task_graph, network)
            print(f"should_duplicate('{task_name}'): {duplicate}")

        run_experiment()


if __name__ == "__main__":
	main()
