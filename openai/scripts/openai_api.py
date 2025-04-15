import openai
import os
import json
from dotenv import load_dotenv
from openai import OpenAI
import networkx as nx
import matplotlib.pyplot as plt
from saga.utils.draw import draw_gantt, draw_network, draw_task_graph
from typing import Dict, List, Tuple
from saga.scheduler import Task
import pathlib
from src.data import (SCHEDULER_MAP, 
                      SCHEDULER_NAME_MAP,
                      NETWORK_GRAPH_DESCRIPTION, 
                      TASK_GRAPH_DESCRIPTION,
                      SCHEDULER_DESCRIPTION_MAP
)

# scheduling
def schedule(algorithm_1 : int, algorithm_2 : int, TASK_GRAPH: nx.DiGraph, NETWORK_GRAPH: nx.Graph, visualize: bool) -> None:
    
    scheduler_1 = SCHEDULER_MAP[algorithm_1]
    scheduler_2 = SCHEDULER_MAP[algorithm_2]

    schedule_1 = scheduler_1.schedule(NETWORK_GRAPH, TASK_GRAPH)
    schedule_2 = scheduler_2.schedule(NETWORK_GRAPH, TASK_GRAPH)

    schedule_1_makespan = max([0 if not tasks else tasks[-1].end for tasks in schedule_1.values()])
    schedule_2_makespan = max([0 if not tasks else tasks[-1].end for tasks in schedule_2.values()])

    if visualize:
        draw_schedule(schedule_1, 'scheduler_1_scaled', xmax=schedule_1_makespan)
        draw_schedule(schedule_2, 'scheduler_2_scaled', xmax=schedule_2_makespan)

        print(f'{SCHEDULER_NAME_MAP[algorithm_1]} makespan: {schedule_1_makespan}')
        print(f'{SCHEDULER_NAME_MAP[algorithm_2]} makespan: {schedule_2_makespan}')
        print(f"Makespan Ratio: {schedule_1_makespan/schedule_2_makespan}")

    

# get prompt
def getPrompt(algorithm_1: str, algorithm_2: str, prompt_level: int) -> str:
    return (
        f"Can you generate a detailed network graph {NETWORK_GRAPH_DESCRIPTION if prompt_level >= 1 else ''} "
        f"and task graph {TASK_GRAPH_DESCRIPTION if prompt_level >=1 else ''} "
        f"An example where {SCHEDULER_NAME_MAP[algorithm_1]} {SCHEDULER_DESCRIPTION_MAP[algorithm_1] if prompt_level >= 2 else ''} performs dramatically different compared to "
        f"{SCHEDULER_NAME_MAP[algorithm_2]} {SCHEDULER_DESCRIPTION_MAP[algorithm_2] if prompt_level >= 2 else ''} in a scheduling makespan (we want the maximum difference in the execution time between the two algorithms). "
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
    thisdir = pathlib.Path(__file__).resolve().parent

    savepath = thisdir / 'results'
    savepath.mkdir(exist_ok=True)
    axis = draw_task_graph(TASK_GRAPH, use_latex=True)
    axis.get_figure().savefig(savepath / 'task_graph.pdf')
    plt.close(axis.get_figure())

    axis = draw_network(NETWORK_GRAPH, draw_colors=False, use_latex=True)
    axis.get_figure().savefig(savepath / 'network_graph.pdf')
    plt.close(axis.get_figure())

# draw schedule
def draw_schedule(schedule: Dict[str, List[Task]], name: str, xmax: float = None):
    thisdir = pathlib.Path(__file__).resolve().parent
    savepath = thisdir / 'results'
    savepath.mkdir(exist_ok=True)
    ax: plt.Axes = draw_gantt(schedule, use_latex=True, xmax=xmax)
    ax.get_figure().savefig(str(savepath / f'{name}.png'))

# get chatgpt answer
def query(algorithm1: str, algorithm2: str, prompt_level: int) -> Tuple[nx.DiGraph, nx.Graph]:
    
    # load key
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": getPrompt(algorithm1, algorithm2, prompt_level)
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
                        },
                        "explanation": {
                            "type": "string",
                            "description": "An detailed explanation & reasoning behind the graphs generated"
                        }
                    },
                    "required": ["task_graph", "network_graph", "explanation"]
                }
            }
        ]
    )
    
    graph_data = json.loads(response.choices[0].message.function_call.arguments)

    print(json.dumps(graph_data, indent=2))

    task_graph = graph_data["task_graph"]
    network_graph = graph_data ["network_graph"]
    explanation = graph_data["explanation"]
    
    [TASK_GRAPH, NETWORK_GRAPH] = getGraphs(task_graph, network_graph)

    return TASK_GRAPH, NETWORK_GRAPH, explanation