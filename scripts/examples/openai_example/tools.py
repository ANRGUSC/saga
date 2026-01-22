"""
Tool functions for the agentic loop.

This module provides:
- Source code retrieval for schedulers
- PISA experiment execution
- LLM-based algorithm comparison
"""

import inspect
import pathlib
import uuid
from typing import TYPE_CHECKING, Optional

import numpy as np

from saga.pisa.simulated_annealing import (
    SCHEDULERS,
    SchedulerName,
    SimulatedAnnealing,
    SimulatedAnnealingConfig,
)
from saga.utils.random_graphs import get_chain_dag, get_network
from saga.utils.random_variable import UniformRandomVariable

from logger import ExperimentLogger

if TYPE_CHECKING:
    from pydantic_ai import Agent
    from models import AlgorithmComparison, CodeHypothesis


def get_scheduler_source_code(scheduler_name: SchedulerName) -> str:
    """Get the source code of a scheduling algorithm."""
    scheduler = SCHEDULERS.get(scheduler_name)
    if scheduler is None:
        return f"Unknown scheduler: {scheduler_name}"

    try:
        module = inspect.getmodule(scheduler.__class__)
        if module:
            return inspect.getsource(module)
        return inspect.getsource(scheduler.__class__)
    except (TypeError, OSError) as e:
        return f"Could not retrieve source code: {e}"


def run_pisa_experiment(
    results_dir: pathlib.Path,
    scheduler: SchedulerName,
    base_scheduler: SchedulerName,
    max_iterations: int = 1000,
    num_tasks: int = 6,
    num_nodes: int = 4,
) -> str:
    """
    Run PISA to find adversarial instances.

    Returns a summary string with details about the adversarial instance found.
    """
    run_name = f"pisa_{scheduler}_vs_{base_scheduler}_{uuid.uuid4().hex[:6]}"

    initial_network = get_network(
        num_nodes=num_nodes,
        node_weight_distribution=UniformRandomVariable(0.1, 1.0),
        edge_weight_distribution=UniformRandomVariable(0.1, 1.0),
    )
    initial_task_graph = get_chain_dag(
        num_nodes=num_tasks,
        node_weight_distribution=UniformRandomVariable(0.1, 1.0),
        edge_weight_distribution=UniformRandomVariable(0.1, 1.0),
    )

    sa = SimulatedAnnealing(
        name=run_name,
        scheduler=scheduler,
        base_scheduler=base_scheduler,
        config=SimulatedAnnealingConfig(max_iterations=max_iterations),
        initial_network=initial_network,
        initial_task_graph=initial_task_graph,
        data_dir=results_dir,
    )

    sa.execute(progress=False)
    best = sa.best_iteration

    tg = best.current_task_graph
    net = best.current_network

    # Compute statistics
    num_tasks_result = len(tg.tasks)
    num_deps = len(tg.dependencies)
    avg_task_cost = np.mean([t.cost for t in tg.tasks])
    avg_data_size = np.mean([d.size for d in tg.dependencies]) if tg.dependencies else 0

    in_degrees = [tg.in_degree(t.name) for t in tg.tasks]
    out_degrees = [tg.out_degree(t.name) for t in tg.tasks]
    max_in = max(in_degrees)
    max_out = max(out_degrees)


    # Build task details
    task_details = []
    for task in sorted(tg.tasks, key=lambda t: t.name):
        predecessors = [d.source for d in tg.dependencies if d.target == task.name]
        successors = [d.target for d in tg.dependencies if d.source == task.name]
        task_details.append(
            f"    {task.name}: cost={task.cost:.3f}, "
            f"in={predecessors if predecessors else '[]'}, "
            f"out={successors if successors else '[]'}"
        )

    # Build edge details
    edge_details = [
        f"    {dep.source} -> {dep.target}: size={dep.size:.3f}"
        for dep in tg.dependencies
    ]

    # Build network node details
    node_details = [
        f"    {node.name}: speed={node.speed:.3f}"
        for node in sorted(net.nodes, key=lambda n: n.name)
    ]

    # Build network link details
    link_details = []
    seen_links = set()
    for edge in net.edges:
        link_key = tuple(sorted([edge.source, edge.target]))
        if link_key not in seen_links and edge.source != edge.target:
            seen_links.add(link_key)
            link_details.append(
                f"    {edge.source} <-> {edge.target}: bandwidth={edge.speed:.3f}"
            )


    return f"""PISA Results ({scheduler} vs {base_scheduler}):
- Energy (makespan ratio): {best.current_energy:.4f}
- {scheduler} makespan: {best.current_makespan:.4f}, {base_scheduler} makespan: {best.current_base_makespan:.4f}
- Tasks: {num_tasks_result}, Dependencies: {num_deps}
- Max in-degree: {max_in}, Max out-degree: {max_out}

TASK GRAPH:
{chr(10).join(task_details)}

DATA DEPENDENCIES:
{chr(10).join(edge_details) if edge_details else '    (none)'}

NETWORK:
{chr(10).join(node_details)}

LINKS:
{chr(10).join(link_details) if link_details else '    (fully connected)'}"""


def test_single_instance(
    hypothesis: "CodeHypothesis",
    target_scheduler: SchedulerName,
    baseline_scheduler: SchedulerName,
) -> str:
    """
    Test a hypothesis on a single instance and return detailed schedule comparison.

    This shows the agent EXACTLY what each scheduler produces:
    - Which task is assigned to which processor
    - Start and end times for each task
    - Why one makespan is longer than the other
    """
    from hypothesis import execute_code_hypothesis

    # Execute the hypothesis code to generate one instance
    network, task_graph, error = execute_code_hypothesis(hypothesis)
    if error:
        return f"ERROR generating instance: {error}"

    # Get the scheduler classes
    target_sched = SCHEDULERS.get(target_scheduler)
    baseline_sched = SCHEDULERS.get(baseline_scheduler)

    if target_sched is None or baseline_sched is None:
        return f"ERROR: Unknown scheduler(s): {target_scheduler}, {baseline_scheduler}"

    # Run both schedulers
    target_schedule = target_sched.schedule(network, task_graph)
    baseline_schedule = baseline_sched.schedule(network, task_graph)

    target_makespan = target_schedule.makespan
    baseline_makespan = baseline_schedule.makespan
    ratio = target_makespan / baseline_makespan if baseline_makespan > 0 else float('inf')

    # Build detailed output
    lines = [
        f"SINGLE INSTANCE TEST: {target_scheduler} vs {baseline_scheduler}",
        f"=" * 60,
        f"",
        f"RESULT: {target_scheduler} makespan={target_makespan:.4f}, {baseline_scheduler} makespan={baseline_makespan:.4f}",
        f"RATIO: {ratio:.4f} ({'WORSE' if ratio > 1.0 else 'BETTER' if ratio < 1.0 else 'SAME'})",
        f"",
    ]

    # Task graph info
    lines.append("TASK GRAPH:")
    for task in sorted(task_graph.tasks, key=lambda t: t.name):
        predecessors = [d.source for d in task_graph.dependencies if d.target == task.name]
        successors = [d.target for d in task_graph.dependencies if d.source == task.name]
        lines.append(f"  {task.name}: cost={task.cost:.3f}, in={predecessors}, out={successors}")

    # Data dependencies
    lines.append("")
    lines.append("DATA DEPENDENCIES:")
    for dep in task_graph.dependencies:
        lines.append(f"  {dep.source} -> {dep.target}: size={dep.size:.3f}")

    # Network info
    lines.append("")
    lines.append("NETWORK:")
    for node in sorted(network.nodes, key=lambda n: n.name):
        lines.append(f"  {node.name}: speed={node.speed:.3f}")

    lines.append("")
    lines.append("LINKS:")
    seen_links = set()
    for edge in network.edges:
        link_key = tuple(sorted([edge.source, edge.target]))
        if link_key not in seen_links and edge.source != edge.target:
            seen_links.add(link_key)
            lines.append(f"  {edge.source} <-> {edge.target}: bandwidth={edge.speed:.3f}")

    # Target scheduler schedule
    lines.append("")
    lines.append(f"{target_scheduler} SCHEDULE (makespan={target_makespan:.4f}):")
    for node_name in sorted(network.graph.nodes()):
        tasks_on_node = target_schedule[node_name]
        if tasks_on_node:
            for st in sorted(tasks_on_node, key=lambda t: t.start):
                lines.append(f"  {node_name}: [{st.start:.3f} - {st.end:.3f}] {st.name}")
        else:
            lines.append(f"  {node_name}: (idle)")

    # Baseline scheduler schedule
    lines.append("")
    lines.append(f"{baseline_scheduler} SCHEDULE (makespan={baseline_makespan:.4f}):")
    for node_name in sorted(network.graph.nodes()):
        tasks_on_node = baseline_schedule[node_name]
        if tasks_on_node:
            for st in sorted(tasks_on_node, key=lambda t: t.start):
                lines.append(f"  {node_name}: [{st.start:.3f} - {st.end:.3f}] {st.name}")
        else:
            lines.append(f"  {node_name}: (idle)")

    # Analysis hints
    lines.append("")
    lines.append("ANALYSIS:")
    if ratio > 1.0:
        lines.append(f"  {target_scheduler} took {(ratio - 1) * 100:.1f}% longer than {baseline_scheduler}")

        # Find what task finished last in each schedule
        target_last = max(
            (st for node_tasks in target_schedule.mapping.values() for st in node_tasks),
            key=lambda t: t.end
        )
        baseline_last = max(
            (st for node_tasks in baseline_schedule.mapping.values() for st in node_tasks),
            key=lambda t: t.end
        )
        lines.append(f"  {target_scheduler} last task: {target_last.name} on {target_last.node} (ends at {target_last.end:.3f})")
        lines.append(f"  {baseline_scheduler} last task: {baseline_last.name} on {baseline_last.node} (ends at {baseline_last.end:.3f})")

        # Find differences in task assignments
        lines.append("")
        lines.append("  TASK ASSIGNMENT DIFFERENCES:")
        for task in task_graph.tasks:
            target_st = target_schedule.get_scheduled_task(task.name)
            baseline_st = baseline_schedule.get_scheduled_task(task.name)
            if target_st.node != baseline_st.node:
                lines.append(f"    {task.name}: {target_scheduler}={target_st.node}, {baseline_scheduler}={baseline_st.node}")
    elif ratio < 1.0:
        lines.append(f"  {target_scheduler} was {(1 - ratio) * 100:.1f}% FASTER than {baseline_scheduler} (not adversarial)")
    else:
        lines.append(f"  Both schedulers produced identical makespans (not adversarial)")

    return "\n".join(lines)


def compare_algorithms_with_llm(
    scheduler1: SchedulerName,
    scheduler2: SchedulerName,
    comparison_agent: "Agent[None, AlgorithmComparison]",
    model_name: str,
    logger: Optional[ExperimentLogger] = None,
) -> str:
    """Use an LLM to compare two scheduling algorithms based on their source code.

    Args:
        scheduler1: First scheduler to compare
        scheduler2: Second scheduler to compare
        comparison_agent: The Pydantic AI agent to use for comparison
        model_name: Name of the model (for logging)
        logger: Optional experiment logger
    """
    source1 = get_scheduler_source_code(scheduler1)
    source2 = get_scheduler_source_code(scheduler2)

    prompt = f"""Analyze and compare these two scheduling algorithms:

## {scheduler1} Source Code:
```python
{source1[:8000]}
```

## {scheduler2} Source Code:
```python
{source2[:8000]}
```

Provide a detailed comparison focusing on their approaches, strengths, weaknesses,
and predict task graph patterns where {scheduler1} might underperform compared to {scheduler2}.
"""

    result = comparison_agent.run_sync(prompt)
    comparison = result.output

    if logger:
        logger.log_token_usage("comparison", result.usage(), model_name)

    return f"""
=== ALGORITHM COMPARISON: {scheduler1} vs {scheduler2} ===

{scheduler1} APPROACH:
{comparison.algorithm1_approach}

{scheduler2} APPROACH:
{comparison.algorithm2_approach}

KEY DIFFERENCES:
{chr(10).join(f'  - {d}' for d in comparison.key_differences)}

{scheduler1} STRENGTHS:
{chr(10).join(f'  + {s}' for s in comparison.algorithm1_strengths)}

{scheduler1} WEAKNESSES:
{chr(10).join(f'  - {w}' for w in comparison.algorithm1_weaknesses)}

{scheduler2} STRENGTHS:
{chr(10).join(f'  + {s}' for s in comparison.algorithm2_strengths)}

{scheduler2} WEAKNESSES:
{chr(10).join(f'  - {w}' for w in comparison.algorithm2_weaknesses)}

PREDICTED ADVERSARIAL PATTERNS (where {scheduler1} underperforms):
{chr(10).join(f'  * {p}' for p in comparison.predicted_adversarial_patterns)}
"""
