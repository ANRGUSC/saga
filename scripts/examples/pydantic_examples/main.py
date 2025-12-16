from typing import List
from saga import Schedule, ScheduledTask, Network, NetworkEdge, NetworkNode, TaskGraph
from saga.schedulers.cpop import CpopScheduler

def main():
    schedule = Schedule.create({"1"})
    schedule.add_task(
        ScheduledTask(node="1", name="task1", start=0, end=10)
    )
    print(schedule.makespan)
    schedule.add_task(
        ScheduledTask(node="1", name="task2", start=15, end=20)
    )
    schedule.add_task(
        ScheduledTask(node="1", name="task3", start=10, end=15)
    )
    print(schedule.makespan)  # 20
    # schedule["1"] = [Task(node="1", name="task4", start=20, end=25)]

    network = Network.create(
        nodes=[("1", 1.0), ("2", 2.0), ("3", 3.0)],
        edges=[("1", "2", 10.0), ("2", "3", 20.0), ("1", "3", 2.0)]
    )
    # network.nodes.add(NetworkNode(name="4", speed=4.0))
    print(network)

    task_graph = TaskGraph.create(
        tasks=[("A", 10.0), ("B", 20.0), ("C", 30.0)],
        dependencies=[("A", "B", 5.0), ("B", "C", 15.0), ("A", "C", 25.0)]
    )

    scheduler = CpopScheduler()
    schedule = scheduler.schedule(network, task_graph)
    print(schedule)

if __name__ == "__main__":
    main()