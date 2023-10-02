import json
import logging
from saga.schedulers.flb import FLBScheduler
from saga.utils.draw import draw_task_graph, draw_network
import networkx as nx
import pathlib

thisdir = pathlib.Path(__file__).parent.absolute()
logging.basicConfig(level=logging.DEBUG)

def main():
    task_graph = nx.readwrite.json_graph.node_link_graph(
        json.loads(thisdir.joinpath("task_graph.json").read_text(encoding="utf-8"))
    )
    axis = draw_task_graph(task_graph)
    axis.figure.savefig(thisdir.joinpath("task_graph.png"))

    network = nx.readwrite.json_graph.node_link_graph(
        json.loads(thisdir.joinpath("network.json").read_text(encoding="utf-8"))
    )
    axis = draw_network(network)
    axis.figure.savefig(thisdir.joinpath("network.png"))

    scheduler = FLBScheduler()
    schedule = scheduler.schedule(network, task_graph)
    print(schedule)


if __name__ == "__main__":
    main()