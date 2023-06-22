# Example usage
tasks = [
    {'id': 'T1', 'processor': 0, 'cost': [5, 10, 15], 'predecessors': [], 'successors': ['T2', 'T3'], 'data_transfer': [0, 5, 10]},
    {'id': 'T2', 'processor': 0, 'cost': [5, 10, 15], 'predecessors': ['T1'], 'successors': ['T4'], 'data_transfer': [0, 5, 10]},
    {'id': 'T3', 'processor': 0, 'cost': [5, 10, 15], 'predecessors': ['T1'], 'successors': ['T4'], 'data_transfer': [0, 5, 10]},
    {'id': 'T4', 'processor': 0, 'cost': [5, 10, 15], 'predecessors': ['T2', 'T3'], 'successors': [], 'data_transfer': [0, 5, 10]}
]

processors = ['P1', 'P2', 'P3']

result = heuristic_dps(tasks, processors)

for task, processor in result:
    print(f"Task {task} assigned to Processor {processor}")