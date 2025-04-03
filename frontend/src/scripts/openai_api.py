# pip install openai
# connect to Open AI API to generate special cases for two user proposed algorithms

import openai
import os
from dotenv import load_dotenv
from openai import OpenAI
import json

# get chatgpt answer
def query(algorithm_worse, algorithn_best):
    
    # load key
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")

    client = OpenAI()
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": "Can you generate an example directed graph?"
            }
        ],
        functions=[
            {
                "name": "return_graph",
                "description": "Return a graph structure with nodes and weighted edges.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "nodes": {
                            "type": "array",
                            "items": {"type": "string"}
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
                }
            }
        ]
    )
    
    graph_data = json.loads(response.choices[0].message.function_call.arguments)

    print(json.dumps(graph_data, indent=2))

if __name__ == "__main__":
    query(1,2)


