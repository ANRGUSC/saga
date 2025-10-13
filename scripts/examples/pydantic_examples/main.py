from typing import List
from saga.scheduler import Schedule, Task, Network, NetworkEdge, NetworkNode

def main():
    schedule = Schedule(nodes={"1"})
    schedule.add_task(
        Task(node="1", name="task1", start=0, end=10)
    )
    print(schedule.makespan)
    schedule.add_task(
        Task(node="1", name="task2", start=15, end=20)
    )
    schedule.add_task(
        Task(node="1", name="task3", start=10, end=15)
    )
    print(schedule.makespan)  # 20
    # schedule["1"] = [Task(node="1", name="task4", start=20, end=25)]

    network = Network(
        nodes=[("1", 1.0), ("2", 2.0), ("3", 3.0)],
        edges=[("1", "2", 10.0), ("2", "3", 20.0), ("1", "3", 2.0)]
    )
    # network.nodes.add(NetworkNode(name="4", speed=4.0))
    print(network)

if __name__ == "__main__":
    main()