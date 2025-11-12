"""
Test conditional ranking functions and compare with standard HEFT.
"""
import networkx as nx
import numpy as np
from typing import List, Dict, Any, Callable, Hashable, Tuple
from saga.schedulers.heft import heft_rank_sort, HeftScheduler
from saga.schedulers.cpop import upward_rank
from saga.scheduler import Task
from generate_dags import create_test_conditional_dags

#NOT IN USE 

# step 1 Scale conditional edge weights by their probabilities ----------------------------


def scale_conditional_weights(task_graph: nx.DiGraph) -> nx.DiGraph:
    """
    Scale edge weights by their probabilities for conditional branches.
    This should make HEFT ranking consistent across execution paths.
    
    Args:
        task_graph: Original task graph with conditional edges
        
    Returns:
        New task graph with scaled edge weights
    """

# step 2 Generate all possible static execution paths from conditional DAG ----------------

def get_all_execution_paths(task_graph: nx.DiGraph) -> Dict[str, nx.DiGraph]:
    """
    Generate all static task graphs using BFS queue with the task graphs being the inside the queue
    """
    from collections import deque
    
    # Creates queue where we can add remove paths from both ends 
    queue = deque([(task_graph.copy(), "path")])
    
    # Store our final Dict of paths
    completed_paths: Dict[str, nx.DiGraph] = {}
    
    # COunter used to give each static path an ID 
    path_counter = 0
    
    # For infinite loop checking
    iteration = 0
    
    def find_first_conditional_node(graph: nx.DiGraph):
        """
        Input: Task graph
        Output: if no conditional branches in task graph = None
                else: return parent node and conditional children: ("parent Node", ["Conditional node", "Conditional node 2.."]) 
        e.g. for node A that has child B, C, D where D is deterministic
        it will return ("A", ["B", "C"])

        """

        # topo sort 
        topo_order = list(nx.topological_sort(graph))   
        
        # Check each node in order
        for node in topo_order:
            conditional_children = []
            successors = list(graph.successors(node))

            # Check if child\s is conditional 
            for child in successors:
                is_conditional = graph.edges[node, child].get("conditional", False)
                if is_conditional:
                    conditional_children.append(child)
            
            # If node has multiple conditional childs then this node is a branch point thus return parent node with conditional children in list
            if len(conditional_children) > 1:
                return (node, conditional_children)
        
        # If no conditional children found then return None -> lead to saving this graph as a path as nothing left
        return None
            
    
    def resolve_conditional_choice(graph: nx.DiGraph, parent: str, chosen_child: str) -> nx.DiGraph:
        """
        Input: digraph/task graph e.g. ("A", ["B", "C"])
               parent: "A"
               chosen_child: "B"
        
        Output: digraph removing the unchosen conditional children e.g. ("A", ["B"])
        """
        new_graph = graph.copy()
        
        # Find all conditional siblings of parent
        conditional_siblings = []
        for child in graph.successors(parent):
            # Simplified if statement from chat, if conditional is true then execute otherwise if conditional attribute does not exist or is equal to false dont execute
            if graph.edges[parent, child].get("conditional", False):
                conditional_siblings.append(child)
        
        # Loop through conditional siblings and if not chosen_child then remove it + all its descendants
        for sibling in conditional_siblings:
            if sibling != chosen_child:
                # Remove the edge from parent to this unchosen sibling
                if new_graph.has_edge(parent, sibling):
                    new_graph.remove_edge(parent, sibling)
                
                # Create a set with the descendants and sibling and then removes them
                nodes_to_remove = set(nx.descendants(graph, sibling)) | {sibling}
                for node in nodes_to_remove:
                    if node in new_graph:
                        new_graph.remove_node(node)
                #might have to do another check if we are trying to remove sibling that has already been removed? could happen if two parents link to same child?
        return new_graph
    

    # Main loop    
    while queue:
        iteration += 1
        
        # Infinite loop check
        if iteration > 100:
            print("infinite loop check hit")
            break
        
        # Pop queue 
        current_graph, current_path_name = queue.popleft()
        
        # Check if has conditional childs
        result = find_first_conditional_node(current_graph)
        
        if result is None:
            # If none save path
            path_counter += 1
            final_name = "path_" + str(path_counter)
            completed_paths[final_name] = current_graph.copy()
        else:
            # Get the parent node and conditional children from the find_first_conditional_node() function
            parent_node, conditional_children = result
            
            #loop through conditional childs 
            for i, chosen_child in enumerate(conditional_children):               
                # Create new graph removing unchosen conditions
                new_graph = resolve_conditional_choice(current_graph, parent_node, chosen_child)
                new_path_name = str(current_path_name) + "_" + str(parent_node) + "->" + str(chosen_child)
                
                # Add partially complete graph back into queue, will be resolved of different conditions as it keeps being explored
                queue.append((new_graph, new_path_name))
    
    # Debugging output
    print("queue empty")
    print("iteration: ")
    print(iteration)
    print("completed paths: ")
    print(len(completed_paths))
    
    return completed_paths


# step 3 Run HEFT on each static path and extract schedules + rankings --------------------

# step 4 Check consistency of rankings and schedules across all paths --------------------

# step 5 Compare standard HEFT vs conditional HEFT (with weight scaling) ---------------


def compare_ranking_approaches(task_graph: nx.DiGraph, network: nx.Graph) -> Dict[str, Any]:
    """
    Compare standard HEFT vs conditional HEFT with weight scaling.
    
    This is the main function that ties everything together:
    1. Generate all execution paths from conditional DAG
    2. Run standard HEFT on each path
    3. Scale weights and run HEFT again
    4. Check consistency in both cases
    
    Args:
        task_graph: Conditional task graph
        network: Network graph
        
    Returns:
        Comparison report showing which approach maintains consistency
    """
    # Generate all possible execution paths
    execution_paths = get_all_execution_paths(task_graph)
    
    print(f"Generated {len(execution_paths)} execution paths")
    
    #Standard HEFT (no weight scaling)

    
    #Conditional HEFT (with weight scaling)



# stepo 6  printing and visualisation ---------------------------------------------------
