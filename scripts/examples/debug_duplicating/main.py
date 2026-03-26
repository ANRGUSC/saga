from functools import lru_cache
import pathlib
import matplotlib.pyplot as plt

from saga import Network, Schedule, TaskGraph
from saga.schedulers import CpopScheduler, HeftScheduler
from saga.utils.duplication import should_duplicate
from saga.utils.draw import draw_gantt


thisdir = pathlib.Path(__file__).parent.absolute()
savedir = thisdir / "outputs"
savedir.mkdir(exist_ok=True)


def build_uniform_network() -> Network:
	"""Create a small homogeneous network for the diamond example."""
	nodes = [
		("v_1", 1.0),
		("v_2", 1.1),
	]
	edges = [
		("v_1", "v_2", 1.0)
	]
	return Network.create(nodes=nodes, edges=edges)


def build_diamond_task_graph(
) -> TaskGraph:
	"""Build diamond DAG: t_1 -> {t_2, t_3} -> t_4."""
	tasks = [
		("t_1", 4.9),
		("t_2", 4.9),
		("t_3", 4.9),
		("t_4", 4.9),
	]
	dependencies = [
		("t_1", "t_2", 5.0),
		("t_1", "t_3", 5.0),
		("t_2", "t_4", 5.0),
		("t_3", "t_4", 5.0),
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
	network, task_graph = get_instance()

	duplicate = should_duplicate("t_1", task_graph, network)
	print(f"\nshould_duplicate('t_1'): {duplicate}")
	run_experiment()


if __name__ == "__main__":
	main()
