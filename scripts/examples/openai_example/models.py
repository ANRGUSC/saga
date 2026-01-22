"""
Pydantic models for the hypothesis generation system.

This module contains all data models used for:
- Hypothesis representation and validation results
- Agent actions and decisions
- Strategic planning and reflection
- Algorithm comparison
"""

import uuid
from typing import List, Literal, Optional

from pydantic import BaseModel, Field

from saga.pisa.simulated_annealing import SchedulerName


# =============================================================================
# Hypothesis Validation
# =============================================================================

class HypothesisValidationResult(BaseModel):
    """Results from validating a hypothesis by generating and testing instances."""

    hypothesis_id: str
    num_instances_tested: int
    num_instances_confirmed: int = Field(
        description="Number of instances where worse_scheduler actually performed worse"
    )
    avg_makespan_ratio: float = Field(
        description="Average ratio of worse_scheduler makespan / better_scheduler makespan"
    )
    max_makespan_ratio: float
    min_makespan_ratio: float
    confirmation_rate: float = Field(
        description="Percentage of instances that confirmed the hypothesis"
    )
    is_validated: bool = Field(description="True if confirmation_rate > 0.5")


# =============================================================================
# Code-Based Hypothesis
# =============================================================================

class CodeHypothesis(BaseModel):
    """
    A hypothesis expressed as executable Python code that generates task graph instances.

    The code should define a function `get_instance()` that returns a (Network, TaskGraph) tuple.
    This gives the LLM complete freedom to create any pathological case it can imagine.
    """

    hypothesis_id: str = Field(
        default_factory=lambda: str(uuid.uuid4())[:8],
        description="Unique identifier for this hypothesis",
    )
    name: str = Field(
        ...,
        description="Short descriptive name for this hypothesis (e.g., 'imbalanced_diamond')",
    )
    worse_scheduler: SchedulerName = Field(
        ..., description="The scheduler predicted to perform worse"
    )
    better_scheduler: SchedulerName = Field(
        ..., description="The scheduler predicted to perform better (baseline)"
    )
    pisa_pattern: str = Field(
        ...,
        description="""BEFORE writing code, describe the EXACT pattern from PISA you are replicating:
- How many tasks? How many have no dependencies (isolated)?
- What is the graph structure? (chains, fan-in, fan-out, disconnected components)
- What are the approximate task cost ranges?
- What are the processor speed differences?
Example: "PISA found 6 tasks: 2 chains (E→A, F→C) and 2 isolated tasks (B, D). Costs range 0.2-0.8. Network has 4 processors with 5x speed difference."
""",
    )
    reasoning: str = Field(
        ...,
        description="Explanation of WHY this specific structure causes worse_scheduler to underperform",
    )
    code: str = Field(
        ...,
        description="""Python code that defines a get_instance() function.

The function signature must be:
    def get_instance() -> Tuple[Network, TaskGraph]:

IMPORTANT: Use randomization! The function will be called MANY times to generate a FAMILY
of problem instances. Each call should return a DIFFERENT instance with the same
pathological structure but varied weights/sizes.

Available imports (already imported):
    - networkx as nx
    - numpy as np
    - random
    - from itertools import product
    - Network, TaskGraph (from saga)

Example code:
```python
def get_instance() -> Tuple[Network, TaskGraph]:
    num_branches = random.randint(2, 5)
    dag = nx.DiGraph()
    dag.add_node("source", weight=random.uniform(0.1, 0.5))
    dag.add_node("sink", weight=random.uniform(0.1, 0.5))

    heavy_branch = random.randint(0, num_branches - 1)
    for i in range(num_branches):
        branch_name = f"branch_{i}"
        if i == heavy_branch:
            dag.add_node(branch_name, weight=random.uniform(3.0, 8.0))
            dag.add_edge(branch_name, "sink", weight=random.uniform(2.0, 5.0))
        else:
            dag.add_node(branch_name, weight=random.uniform(0.1, 0.5))
            dag.add_edge(branch_name, "sink", weight=random.uniform(0.1, 0.3))
        dag.add_edge("source", branch_name, weight=random.uniform(0.1, 0.3))

    task_graph = TaskGraph.from_nx(dag)

    num_processors = random.randint(2, 4)
    net = nx.Graph()
    for i in range(num_processors):
        speed = random.uniform(5.0, 10.0) if i == 0 else random.uniform(0.5, 2.0)
        net.add_node(f"p{i}", weight=speed)

    for i in range(num_processors):
        for j in range(num_processors):
            bw = 1e9 if i == j else random.uniform(0.5, 2.0)
            net.add_edge(f"p{i}", f"p{j}", weight=bw)

    network = Network.from_nx(net)
    return network, task_graph
```
""",
    )
    confidence: float = Field(
        default=0.5,
        description="Confidence level in this hypothesis (0.0 to 1.0)",
        ge=0.0,
        le=1.0,
    )


# =============================================================================
# Agent Decision Models
# =============================================================================

class AgentAction(BaseModel):
    """The agent's decision about what action to take next."""

    action: Literal[
        "compare_algorithms",
        "read_source_code",
        "run_pisa",
        "test_code_hypothesis",
        "submit_code_hypothesis",
    ] = Field(..., description="The action to take")
    reasoning: str = Field(..., description="Why this action was chosen")
    scheduler_to_read: Optional[SchedulerName] = Field(
        default=None, description="Which scheduler's source code to read"
    )
    code_hypothesis: Optional[CodeHypothesis] = Field(
        default=None,
        description="Code-based hypothesis with get_instance() function that generates (Network, TaskGraph) instances.",
    )


class StrategicPlan(BaseModel):
    """A strategic plan for the current iteration."""

    current_understanding: str = Field(
        ..., description="Summary of what we currently understand about the algorithms"
    )
    key_unknowns: List[str] = Field(
        ...,
        description="What we still don't know that would help form a hypothesis",
    )
    working_hypothesis: Optional[str] = Field(
        default=None,
        description="Current working hypothesis about what might cause poor performance",
    )
    next_steps: List[str] = Field(
        ..., description="Ordered list of 2-3 next steps to take (what and why)"
    )
    immediate_action: str = Field(
        ..., description="The immediate next action to take and specific reasoning"
    )
    success_criteria: str = Field(
        ..., description="What would indicate we're making progress"
    )


class ActionReflection(BaseModel):
    """Reflection on the results of an action."""

    action_taken: str = Field(..., description="What action was just completed")
    key_findings: List[str] = Field(
        ..., description="Key insights or data points learned from this action"
    )
    surprises: List[str] = Field(
        default_factory=list, description="Anything unexpected that we learned"
    )
    hypothesis_update: str = Field(
        ..., description="How this changes our working hypothesis (or confirms it)"
    )
    next_question: str = Field(
        ..., description="The most important question we should answer next"
    )
    confidence_assessment: str = Field(
        ...,
        description="How confident are we now in forming a valid hypothesis? Why?",
    )


# =============================================================================
# Algorithm Comparison
# =============================================================================

class AlgorithmComparison(BaseModel):
    """Structured comparison of two scheduling algorithms."""

    algorithm1_name: str
    algorithm2_name: str
    algorithm1_approach: str = Field(
        ...,
        description="How algorithm 1 works (key steps, prioritization, assignment)",
    )
    algorithm2_approach: str = Field(
        ...,
        description="How algorithm 2 works (key steps, prioritization, assignment)",
    )
    key_differences: List[str] = Field(
        ..., description="List of key differences between the algorithms"
    )
    algorithm1_strengths: List[str] = Field(
        ..., description="Scenarios where algorithm 1 excels"
    )
    algorithm1_weaknesses: List[str] = Field(
        ..., description="Scenarios where algorithm 1 struggles"
    )
    algorithm2_strengths: List[str] = Field(
        ..., description="Scenarios where algorithm 2 excels"
    )
    algorithm2_weaknesses: List[str] = Field(
        ..., description="Scenarios where algorithm 2 struggles"
    )
    predicted_adversarial_patterns: List[str] = Field(
        ...,
        description="Task graph patterns that might cause algorithm 1 to underperform vs algorithm 2",
    )
