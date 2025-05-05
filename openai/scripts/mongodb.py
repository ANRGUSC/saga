from pymongo import MongoClient
from datetime import datetime
from networkx.readwrite import json_graph
from networkx import Graph
import os
from dotenv import load_dotenv

"""
To run MongoDB locally:
 --> On macOS (homebrew)
        brew tap mongodb/brew
        brew install mongodb-community
        brew services start mongodb-community

        mongosh             # Drops you into the shell

 --> On Linux
        sudo apt install -y mongodb
        sudo systemctl start mongod

To run MongoDB in Docker
        docker run -d 
        --name test-mongo 
        -p 27017:27017 
        mongo:latest

To view the database
  --> Using the Mongo Shell
        mongosh
        use scheduler_experiments
        db.experiment_runs.find().pretty()

"""

load_dotenv()
_mongo_uri = (
    "mongodb://andrewsykes04:1MA94EsUBU7L6DwN"
    "@cluster0-shard-00-00.555trkl.mongodb.net:27017,"
    "cluster0-shard-00-01.555trkl.mongodb.net:27017,"
    "cluster0-shard-00-02.555trkl.mongodb.net:27017/"
    "?ssl=true&replicaSet=atlas-555trk-shard-0&authSource=admin&retryWrites=true&w=majority"
)
_client     = MongoClient(_mongo_uri)
_db         = _client["scheduler_experiments"]
_collection = _db["experiment_runs"]

def store_experiment(prompt: str, alg1_name: str, alg2_name: str, task_graph: Graph, network_graph: Graph, makespan_diff: float, explanation: str):
    """
    Serialize graphs and insert one document into MongoDB.
    """
    doc = {
        "timestamp":     datetime.utcnow().isoformat(),
        "prompt_used":   prompt,
        "algorithms": {
            "algorithm1": alg1_name,
            "algorithm2": alg2_name
        },
        "explanation": explanation,
        "task_graph":    json_graph.node_link_data(task_graph),
        "network_graph": json_graph.node_link_data(network_graph),
        "alg1_makespan": 1,
        "alg2_makespan": 1,
        "makespan_diff": makespan_diff
    }
    res = _collection.insert_one(doc)
    print(f"[MongoDB] stored experiment _id={res.inserted_id}")

def delete_all_experiments() -> int:
    """
    Remove all documents from the experiment_runs collection.
    Returns the number of documents deleted.
    """
    result = _collection.delete_many({})
    count = result.deleted_count
    print(f"[MongoDB] deleted {count} experiment(s)")
    return count

