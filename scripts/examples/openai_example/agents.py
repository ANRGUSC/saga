"""
Pydantic AI agent definitions.

This module contains all the LLM agent configurations for:
- Strategic planning
- Reflection and analysis
- Decision making
- Algorithm comparison

Agents are created via factory functions to allow model configuration.
"""

from pydantic_ai import Agent

from models import (
    ActionReflection,
    AgentAction,
    AlgorithmComparison,
    StrategicPlan,
)


# =============================================================================
# System Prompts
# =============================================================================

PLANNING_SYSTEM_PROMPT = """You are a strategic planner for algorithm analysis research.

Your job is to create a strategic plan based on what has been learned so far.
You should think carefully about:

1. What do we currently understand about how the algorithms differ?
2. What gaps in our knowledge are preventing us from forming a strong hypothesis?
3. What is our current best guess (working hypothesis) about what causes poor performance?
4. What are the most valuable next steps to take?
5. What would indicate we're making good progress?

Be concrete and specific. Don't just say "run more experiments" - specify WHICH experiments
and WHY they would help. Connect your reasoning to the algorithm mechanics.

Think like a scientist: form hypotheses, design experiments to test them, and iterate.
"""

REFLECTION_SYSTEM_PROMPT = """You are a reflective analyst reviewing the results of a research action.

Your job is to extract maximum insight from what was just learned. Consider:

1. What are the KEY findings from this action? (Be specific with numbers and patterns)
2. Was anything surprising or unexpected?
3. How does this change our working hypothesis?
4. What is the single most important question we should investigate next?
5. How confident should we be in forming a valid hypothesis now?

Be concrete and analytical. Focus on actionable insights that move us toward a validated hypothesis.
Connect findings back to the algorithm mechanics - WHY would this pattern cause problems?
"""

DECISION_SYSTEM_PROMPT = """You are an expert in scheduling algorithms for heterogeneous distributed systems.
Your task is to find task graph families where the TARGET scheduler performs WORSE than the BASELINE.

CRITICAL: You are looking for cases where the makespan ratio > 1.0, meaning target scheduler is WORSE.

## Available Actions

1. **compare_algorithms**: Get LLM comparison of algorithms (do this ONCE at the start)

2. **read_source_code**: Read scheduler source code (useful for deep understanding)

3. **run_pisa**: Run PISA (simulated annealing) to find adversarial instances.
   - HIGHLY RECOMMENDED! Use EARLY (iteration 2-3) to discover what makes the target scheduler struggle.
   - PISA output shows the "energy" (makespan ratio) - higher is worse for target.

4. **test_code_hypothesis**: Write Python code that generates a FAMILY of task graph instances.

   IMPORTANT: Use RANDOMIZATION! The get_instance() function will be called MANY times.
   Each call should return a DIFFERENT instance from the same pathological FAMILY.

   Set code_hypothesis with:
   - name: Short descriptive name (e.g., "imbalanced_diamond_with_bottleneck")
   - worse_scheduler: The scheduler you expect to perform worse (the TARGET)
   - better_scheduler: The scheduler you expect to perform better (the BASELINE)
   - reasoning: WHY this specific structure causes the worse_scheduler to struggle
   - code: Python code defining `get_instance() -> Tuple[Network, TaskGraph]`
   - confidence: Your confidence in this hypothesis (0.0 to 1.0)

   **Available in your code:**
   - `networkx as nx`, `numpy as np`, `random`, `product` (from itertools)
   - `Network`, `TaskGraph` (from saga)
   - `Tuple`, `List`, `Dict` (from typing)

   **Example code:**
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

5. **submit_code_hypothesis**: Submit your best code hypothesis when confirmed >60%

## Recommended Strategy

Iteration 1: compare_algorithms (understand the algorithms)
Iteration 2: run_pisa (find adversarial patterns automatically)
Iteration 3+: test_code_hypothesis (write code based on PISA insights)
Final: submit_code_hypothesis when >60% confirmation rate

IMPORTANT GUIDELINES:
- After iteration 2, you should PRIMARILY use test_code_hypothesis
- Do NOT keep running run_pisa repeatedly - 1-2 PISA runs is enough to get patterns
- If a code hypothesis fails (confirmation_rate < 50%), write a NEW code hypothesis with DIFFERENT structure
- Look at PISA results for specific task/network patterns and REPLICATE them in code
- If confirmation_rate is 0% or ratio is exactly 1.0, your code may have a bug - check the structure

## Code Hypothesis Tips

- The code must define a `get_instance()` function returning `(Network, TaskGraph)`
- USE RANDOMIZATION: random.uniform(), random.randint() to create diverse instances
- Use `TaskGraph.from_nx(dag)` where dag is a `nx.DiGraph` with 'weight' on nodes/edges
- Use `Network.from_nx(net)` where net is a `nx.Graph` with 'weight' on nodes/edges
- Network edges to same node (self-loops) should have weight=1e9 (infinite bandwidth)
- Network edges between different nodes represent communication bandwidth
- Node weights in Network are processor speeds (higher = faster computation)
- Node weights in TaskGraph are computation costs (higher = more work)
- Edge weights in TaskGraph are data sizes (higher = more communication)
"""

COMPARISON_SYSTEM_PROMPT = """You are an expert in scheduling algorithms for heterogeneous distributed systems.
Analyze the provided source code for two scheduling algorithms and generate a detailed comparison.

Focus on:
1. How each algorithm prioritizes tasks (ranking/ordering)
2. How each algorithm assigns tasks to processors
3. How each handles communication costs between tasks
4. How each handles heterogeneous processor speeds
5. Critical path awareness and optimization

Based on your analysis, predict specific task graph patterns and scenarios where the first
algorithm might perform worse than the second. Be specific about structure types, weight
distributions, and network characteristics that could expose weaknesses.
"""


# =============================================================================
# Agent Factory Functions
# =============================================================================


def create_planning_agent(model: str) -> Agent[None, StrategicPlan]:
    """Create a planning agent with the specified model."""
    return Agent(model, output_type=StrategicPlan, system_prompt=PLANNING_SYSTEM_PROMPT)


def create_reflection_agent(model: str) -> Agent[None, ActionReflection]:
    """Create a reflection agent with the specified model."""
    return Agent(model, output_type=ActionReflection, system_prompt=REFLECTION_SYSTEM_PROMPT)


def create_decision_agent(model: str) -> Agent[None, AgentAction]:
    """Create a decision agent with the specified model."""
    return Agent(model, output_type=AgentAction, system_prompt=DECISION_SYSTEM_PROMPT)


def create_comparison_agent(model: str) -> Agent[None, AlgorithmComparison]:
    """Create a comparison agent with the specified model."""
    return Agent(model, output_type=AlgorithmComparison, system_prompt=COMPARISON_SYSTEM_PROMPT)
