import openai
import os
import json
from dotenv import load_dotenv
from openai import OpenAI
import networkx as nx
import matplotlib.pyplot as plt
from saga.utils.draw import draw_gantt, draw_network, draw_task_graph
from saga.utils.tools import check_instance_simple
from typing import Dict, List, Tuple
import pathlib
import logging
from saga.schedulers import (BILScheduler, CpopScheduler, DuplexScheduler,
                             ETFScheduler, FastestNodeScheduler, FCPScheduler,
                             FLBScheduler, GDLScheduler, HeftScheduler,
                             MaxMinScheduler, MCTScheduler, METScheduler,
                             MinMinScheduler, OLBScheduler, WBAScheduler, SDBATSScheduler)

# scheduling
def schedule(TASK_GRAPH: nx.DiGraph, NETWORK_GRAPH: nx.Graph):

    cpop_scheduler = CpopScheduler()
    heft_scheduler = HeftScheduler()

    heft_schedule = heft_scheduler.schedule(NETWORK_GRAPH, TASK_GRAPH)
    cpop_schedule = cpop_scheduler.schedule(NETWORK_GRAPH, TASK_GRAPH)

    cpop_makespan = max([0 if not tasks else tasks[-1].end for tasks in cpop_schedule.values()])
    heft_makespan = max([0 if not tasks else tasks[-1].end for tasks in heft_schedule.values()])

    print(f'CPoP makespan: {cpop_makespan}')
    print(f'HEFT makespan: {heft_makespan}')
    print(f"Makespan Ratio: {cpop_makespan/heft_makespan}")

# get prompt
def getPrompt(algorithm1: str, algorithm2: str) -> str:
    return (
        "Can you generate a detailed network graph (G = (T, D), where T is the set of tasks and D contains "
        "the directed edges or dependencies between these tasks? An edge (t, t′) ∈ D implies that the output "
        "from task t is required input for task t′.) and task graph (N = (V, E) denote the compute node network, "
        "where N is a complete undirected graph. V is the set of nodes and E is the set of edges. The compute speed "
        "of a node v ∈ V is s(v) ∈ R+ and the communication strength between nodes (v,v′) ∈ E is s(v,v′) ∈ R+). "
        f"An example where {algorithm1} performs dramatically worse compared to {algorithm2} in a scheduling makespan (we want the maximum difference in the execution time between the two algorithms). "
        "Name nodes in task graph A, B, and C etc and name nodes in network graph 1, 2, and 3 etc "
        "(no limitations for the number of nodes). We want no cycle, and exactly one source (start node) and one sink (end node) for task graph."
    )

# get networkx graph object
def getGraphs(task_graph: json, network_graph: json) -> Tuple[nx.DiGraph, nx.Graph]:
    
    # task graph
    TASK_GRAPH = nx.DiGraph()
    
    for node in task_graph["nodes"]:
        TASK_GRAPH.add_node(node["name"], weight=node["weight"])

    for edge in task_graph["edges"]:
        TASK_GRAPH.add_edge(edge["source"], edge["target"], weight=edge["weight"])
    
    # network graph
    NETWORK_GRAPH = nx.Graph()

    for node in network_graph["nodes"]:
        NETWORK_GRAPH.add_node(node["name"], weight=node["weight"])

        # for each node in Network Graph, we must add self edge
        NETWORK_GRAPH.add_edge(node["name"], node["name"], weight=1e9)

    for edge in network_graph["edges"]:
        NETWORK_GRAPH.add_edge(edge["node1"], edge["node2"], weight=edge["weight"])
    
    return TASK_GRAPH, NETWORK_GRAPH

# visualize the graphs generated
def visualizeGraphs(TASK_GRAPH: nx.DiGraph, NETWORK_GRAPH: nx.Graph):
    thisdir = pathlib.Path(__file__).parent.absolute()

    savepath = thisdir / 'results'
    savepath.mkdir(exist_ok=True)
    axis = draw_task_graph(TASK_GRAPH, use_latex=True)
    axis.get_figure().savefig(savepath / 'task_graph.pdf')
    plt.close(axis.get_figure())

    axis = draw_network(NETWORK_GRAPH, draw_colors=False, use_latex=True)
    axis.get_figure().savefig(savepath / 'network_graph.pdf')
    plt.close(axis.get_figure())

# get chatgpt answer
def query(algorithm1: str, algorithm2: str) -> Tuple[nx.DiGraph, nx.Graph]:
    
    # load key
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": getPrompt(algorithm1, algorithm2)
            }
        ],
        functions = [
            {
                "name": "return_graphs",
                "description": "Return one directed (task) and one undirected (network) graph, each with nodes and weighted edges.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "task_graph": {
                            "type": "object",
                            "properties": {
                                "nodes": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "name": {"type": "string"},
                                            "weight": {"type": "number"}
                                        },
                                        "required": ["name", "weight"]
                                    }
                                },
                                "edges": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "source": {"type": "string"},
                                            "target": {"type": "string"},
                                            "weight": {"type": "number"}
                                        },
                                        "required": ["source", "target", "weight"]
                                    }
                                }
                            },
                            "required": ["nodes", "edges"]
                        },
                        "network_graph": {
                            "type": "object",
                            "properties": {
                                "nodes": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "name": {"type": "string"},
                                            "weight": {"type": "number"}
                                        },
                                        "required": ["name", "weight"]
                                    }
                                },
                                "edges": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "node1": {"type": "string"},
                                            "node2": {"type": "string"},
                                            "weight": {"type": "number"}
                                        },
                                        "required": ["node1", "node2", "weight"]
                                    }
                                }
                            },
                            "required": ["nodes", "edges"]
                        }
                    },
                    "required": ["directed_graph", "undirected_graph"]
                }
            }
        ]
    )
    
    graph_data = json.loads(response.choices[0].message.function_call.arguments)

    print(json.dumps(graph_data, indent=2))

    task_graph = graph_data["task_graph"]
    network_graph = graph_data ["network_graph"]
    
    [TASK_GRAPH, NETWORK_GRAPH] = getGraphs(task_graph, network_graph)

    return TASK_GRAPH, NETWORK_GRAPH


if __name__ == "__main__":
    
    # get graphs from api
    [TASK_GRAPH, NETWORK_GRAPH] = query("HEFT", "CPOP")

    # visualize graphs
    visualizeGraphs(TASK_GRAPH, NETWORK_GRAPH)

    # check to see if graphs are valid
    check_instance_simple(NETWORK_GRAPH, TASK_GRAPH)

    # used for debug purposes
    # logging.basicConfig(level=logging.DEBUG)

    # make schedules
    schedule(TASK_GRAPH, NETWORK_GRAPH)


