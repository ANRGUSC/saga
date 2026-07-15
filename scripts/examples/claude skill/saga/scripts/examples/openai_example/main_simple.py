"""
Simplified LLM-Powered Hypothesis Generation for Scheduling Algorithm Analysis

This is a streamlined version that uses a SINGLE agent with FULL context,
instead of 3 separate agents (planning/decision/reflection) that lose information.

Key improvements over the multi-agent approach:
1. Single agent sees full history - no context truncation
2. PISA results shown in full - not truncated to 300 chars
3. Explicit failure feedback - "ratio=1.0 means buggy code"
4. Clear action constraints - forced to write code after 1 PISA run
"""

import json
import logging
import pathlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_ai import Agent

from saga.pisa.simulated_annealing import SchedulerName

from hypothesis import execute_code_hypothesis, validate_code_hypothesis
from logger import ExperimentLogger
from models import AlgorithmComparison, CodeHypothesis, HypothesisValidationResult
from tools import get_scheduler_source_code, run_pisa_experiment, test_single_instance

# Suppress the task graph warnings
logging.getLogger().setLevel(logging.ERROR)

load_dotenv()

thisdir = pathlib.Path(__file__).parent.resolve()


# =============================================================================
# MODEL CONFIGURATION
# =============================================================================
MODEL = "openai:gpt-4o"  # Single model for the unified agent


# =============================================================================
# Simplified Output Model
# =============================================================================

class AgentResponse(BaseModel):
    """What the agent outputs each iteration."""

    thought: str = Field(
        ...,
        description="Your analysis of the current situation. What have you learned? What patterns do you see?"
    )
    action: str = Field(
        ...,
        description="One of: 'compare_algorithms', 'run_pisa', 'write_hypothesis', 'test_instance'"
    )
    hypothesis: Optional[CodeHypothesis] = Field(
        default=None,
        description="Required if action is 'write_hypothesis' or 'test_instance'"
    )


# =============================================================================
# Algorithm Comparison Prompt
# =============================================================================

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
distributions, and network characteristics that could expose weaknesses."""


# =============================================================================
# History Tracking
# =============================================================================

@dataclass
class AttemptRecord:
    """Record of a single attempt (comparison, PISA run, or hypothesis test)."""
    iteration: int
    action: str

    # For algorithm comparison
    comparison_result: Optional[str] = None

    # For PISA runs
    pisa_result: Optional[str] = None

    # For hypothesis tests
    hypothesis_name: Optional[str] = None
    hypothesis_code: Optional[str] = None
    hypothesis_pisa_pattern: Optional[str] = None
    hypothesis_reasoning: Optional[str] = None
    confirmation_rate: Optional[float] = None
    avg_ratio: Optional[float] = None
    error: Optional[str] = None

    # For test_instance (single instance test)
    test_instance_result: Optional[str] = None

    def to_history_string(self) -> str:
        """Format this attempt for the agent's context."""
        if self.action == "compare_algorithms":
            return f"""
## Iteration {self.iteration}: Algorithm Comparison
{self.comparison_result}
"""
        elif self.action == "run_pisa":
            return f"""
## Iteration {self.iteration}: PISA Experiment
{self.pisa_result}
"""
        elif self.action == "write_hypothesis":
            if self.error:
                return f"""
## Iteration {self.iteration}: Hypothesis Test - FAILED
Name: {self.hypothesis_name}
Error: {self.error}
Code:
```python
{self.hypothesis_code}
```
"""
            elif self.confirmation_rate == 0.0 and abs(self.avg_ratio - 1.0) < 0.001:
                return f"""
## Iteration {self.iteration}: Hypothesis Test - BUGGY CODE
Name: {self.hypothesis_name}
Result: confirmation_rate=0%, avg_ratio=1.0

BUG DETECTED: When avg_ratio is exactly 1.0, it means BOTH schedulers produce
THE SAME makespan. Your code is NOT creating adversarial instances - it's
creating instances where both algorithms perform identically.

Check your code for:
- Tasks/edges with all the same weights (no asymmetry)
- Network with all same-speed processors
- Dependencies that don't create interesting scheduling decisions

Code:
```python
{self.hypothesis_code}
```
Reasoning: {self.hypothesis_reasoning}
"""
            else:
                return f"""
## Iteration {self.iteration}: Hypothesis Test
Name: {self.hypothesis_name}
Result: avg_ratio={self.avg_ratio:.4f}

PISA Pattern: {self.hypothesis_pisa_pattern}
Reasoning: {self.hypothesis_reasoning}

Code:
```python
{self.hypothesis_code}
```
"""
        elif self.action == "test_instance":
            return f"""
## Iteration {self.iteration}: Single Instance Test
Name: {self.hypothesis_name}
{self.test_instance_result}
"""
        return f"## Iteration {self.iteration}: Unknown action {self.action}"


@dataclass
class SessionState:
    """Tracks the full session state."""

    target_scheduler: SchedulerName
    baseline_scheduler: SchedulerName
    max_iterations: int

    iteration: int = 0
    attempts: List[AttemptRecord] = field(default_factory=list)

    best_hypothesis: Optional[CodeHypothesis] = None
    best_validation: Optional[HypothesisValidationResult] = None

    # Algorithm info (loaded once at start)
    algorithm_comparison: Optional[str] = None

    @property
    def has_comparison(self) -> bool:
        return any(a.action == "compare_algorithms" for a in self.attempts)

    @property
    def pisa_count(self) -> int:
        return sum(1 for a in self.attempts if a.action == "run_pisa")

    @property
    def hypothesis_count(self) -> int:
        return sum(1 for a in self.attempts if a.action == "write_hypothesis")

    def get_full_history(self) -> str:
        """Build the full history string for the agent."""
        if not self.attempts:
            return "No experiments run yet."

        return "\n".join(a.to_history_string() for a in self.attempts)


# =============================================================================
# System Prompt
# =============================================================================

SYSTEM_PROMPT = """You are an expert in scheduling algorithms for heterogeneous distributed systems.

YOUR TASK: Find task graph families where {target} performs WORSE than {baseline}.
A "family" means code that generates many similar instances with the same pathological structure.

## What Success Looks Like
- avg_ratio > 1.0: The target scheduler has longer makespan on average
- Higher avg_ratio is better - aim for the HIGHEST possible average ratio
- We run your code 50 times and compute the average makespan ratio across all instances

## Available Actions

1. **compare_algorithms**: Analyze the source code of both algorithms to understand their differences.
   - Use this FIRST to understand how {target} and {baseline} work
   - Reveals prioritization strategies, assignment logic, and potential weaknesses
   - Only use ONCE at the start

2. **run_pisa**: Run PISA (simulated annealing) to automatically find ONE adversarial instance.
   - Use this ONCE to discover what makes {target} struggle
   - PISA output shows the exact task graph and network that cause problems
   - DO NOT run PISA more than once - use it to learn, then write code

3. **test_instance**: Test your hypothesis code on ONE instance and see detailed schedules.
   - Shows EXACTLY which task is assigned to which processor by each scheduler
   - Shows start/end times for every task
   - Shows which tasks were assigned DIFFERENTLY by the two schedulers
   - Use this to DEBUG your hypothesis - understand WHY one scheduler is worse
   - Requires a hypothesis with code (same as write_hypothesis)

4. **write_hypothesis**: Write Python code that generates a FAMILY of adversarial instances.
   - Your code defines `get_instance() -> Tuple[Network, TaskGraph]`
   - Use RANDOMIZATION so each call returns a different instance from the same family
   - We test your code 50 times and report the avg_ratio

## CRITICAL RULES

1. Start with compare_algorithms to understand the algorithms, then run_pisa to find patterns
2. After running PISA once, use write_hypothesis or test_instance for all remaining iterations
3. If your code has avg_ratio=1.0, it's BUGGY - both schedulers produce the same makespan
4. Use test_instance to understand WHY your hypothesis works (or doesn't work)
5. Your goal is to MAXIMIZE avg_ratio - keep iterating to find better hypotheses

## Code Template

```python
def get_instance() -> Tuple[Network, TaskGraph]:
    # Create task graph
    dag = nx.DiGraph()
    dag.add_node("task1", weight=random.uniform(0.1, 0.5))  # light task
    dag.add_node("task2", weight=random.uniform(5.0, 10.0))  # HEAVY task (10x difference!)
    dag.add_edge("task1", "task2", weight=random.uniform(0.1, 0.5))  # data dependency
    task_graph = TaskGraph.from_nx(dag)

    # Create network
    net = nx.Graph()
    net.add_node("p0", weight=random.uniform(5.0, 10.0))  # FAST processor
    net.add_node("p1", weight=random.uniform(0.5, 1.0))   # SLOW processor (10x difference!)
    net.add_edge("p0", "p0", weight=1e9)  # self-loop = infinite bandwidth
    net.add_edge("p1", "p1", weight=1e9)
    net.add_edge("p0", "p1", weight=random.uniform(0.5, 2.0))  # inter-processor bandwidth
    network = Network.from_nx(net)

    return network, task_graph
```

Available imports: networkx as nx, numpy as np, random, product (from itertools), Network, TaskGraph (from saga)
"""


def create_agent(target: str, baseline: str) -> Agent[None, AgentResponse]:
    """Create the unified agent."""
    prompt = SYSTEM_PROMPT.format(target=target, baseline=baseline)
    return Agent(MODEL, output_type=AgentResponse, system_prompt=prompt)


# =============================================================================
# Action Handlers
# =============================================================================

def handle_compare_algorithms(
    state: SessionState,
    logger: ExperimentLogger,
) -> AttemptRecord:
    """Compare the two algorithms using LLM analysis of source code."""
    print(f"\nComparing {state.target_scheduler} vs {state.baseline_scheduler}...")

    source1 = get_scheduler_source_code(state.target_scheduler)
    source2 = get_scheduler_source_code(state.baseline_scheduler)

    comparison_agent = Agent(
        MODEL,
        output_type=AlgorithmComparison,
        system_prompt=COMPARISON_SYSTEM_PROMPT,
    )

    prompt = f"""Analyze and compare these two scheduling algorithms:

## {state.target_scheduler} Source Code:
```python
{source1[:8000]}
```

## {state.baseline_scheduler} Source Code:
```python
{source2[:8000]}
```

Provide a detailed comparison focusing on their approaches, strengths, weaknesses,
and predict task graph patterns where {state.target_scheduler} might underperform compared to {state.baseline_scheduler}.
"""

    result = comparison_agent.run_sync(prompt)
    comparison = result.output
    logger.log_token_usage("comparison", result.usage(), MODEL.split(":")[-1])

    # Format the comparison result
    comparison_text = f"""
=== ALGORITHM COMPARISON: {state.target_scheduler} vs {state.baseline_scheduler} ===

{state.target_scheduler} APPROACH:
{comparison.algorithm1_approach}

{state.baseline_scheduler} APPROACH:
{comparison.algorithm2_approach}

KEY DIFFERENCES:
{chr(10).join(f'  - {d}' for d in comparison.key_differences)}

{state.target_scheduler} STRENGTHS:
{chr(10).join(f'  + {s}' for s in comparison.algorithm1_strengths)}

{state.target_scheduler} WEAKNESSES:
{chr(10).join(f'  - {w}' for w in comparison.algorithm1_weaknesses)}

{state.baseline_scheduler} STRENGTHS:
{chr(10).join(f'  + {s}' for s in comparison.algorithm2_strengths)}

{state.baseline_scheduler} WEAKNESSES:
{chr(10).join(f'  - {w}' for w in comparison.algorithm2_weaknesses)}

PREDICTED ADVERSARIAL PATTERNS (where {state.target_scheduler} underperforms):
{chr(10).join(f'  * {p}' for p in comparison.predicted_adversarial_patterns)}
"""

    print(comparison_text)

    return AttemptRecord(
        iteration=state.iteration,
        action="compare_algorithms",
        comparison_result=comparison_text,
    )


def handle_pisa(state: SessionState, results_dir: pathlib.Path) -> AttemptRecord:
    """Run PISA and record the result."""
    print("\nRunning PISA experiment...")
    pisa_result = run_pisa_experiment(
        results_dir,
        state.target_scheduler,
        state.baseline_scheduler,
    )
    print(pisa_result)

    record = AttemptRecord(
        iteration=state.iteration,
        action="run_pisa",
        pisa_result=pisa_result,
    )
    return record


def handle_test_instance(
    state: SessionState,
    hypothesis: CodeHypothesis,
) -> AttemptRecord:
    """Test a hypothesis on a single instance to see detailed schedules."""
    print(f"\nTesting single instance for: {hypothesis.name}")

    result = test_single_instance(
        hypothesis,
        state.target_scheduler,
        state.baseline_scheduler,
    )
    print(result)

    return AttemptRecord(
        iteration=state.iteration,
        action="test_instance",
        hypothesis_name=hypothesis.name,
        hypothesis_code=hypothesis.code,
        test_instance_result=result,
    )


def handle_hypothesis(
    state: SessionState,
    hypothesis: CodeHypothesis,
    num_test_instances: int = 50
) -> AttemptRecord:
    """Test a code hypothesis and record the result."""
    print(f"\nTesting hypothesis: {hypothesis.name}")
    print(f"PISA Pattern: {hypothesis.pisa_pattern[:200] if hypothesis.pisa_pattern else 'N/A'}...")
    print(f"Reasoning: {hypothesis.reasoning[:200]}...")

    # Try to execute the code
    network, task_graph, error = execute_code_hypothesis(hypothesis)
    if error:
        print(f"  ERROR: {error}")
        return AttemptRecord(
            iteration=state.iteration,
            action="write_hypothesis",
            hypothesis_name=hypothesis.name,
            hypothesis_code=hypothesis.code,
            hypothesis_pisa_pattern=hypothesis.pisa_pattern,
            hypothesis_reasoning=hypothesis.reasoning,
            error=error,
        )

    # Validate across many instances
    validation = validate_code_hypothesis(hypothesis, num_instances=num_test_instances)

    print(f"  Avg ratio: {validation.avg_makespan_ratio:.4f}")
    print(f"  Max ratio: {validation.max_makespan_ratio:.4f}")
    print(f"  Min ratio: {validation.min_makespan_ratio:.4f}")

    # Track best by avg_ratio (higher is better)
    if state.best_validation is None or validation.avg_makespan_ratio > state.best_validation.avg_makespan_ratio:
        state.best_hypothesis = hypothesis
        state.best_validation = validation
        print("  -> New best hypothesis!")

    return AttemptRecord(
        iteration=state.iteration,
        action="write_hypothesis",
        hypothesis_name=hypothesis.name,
        hypothesis_code=hypothesis.code,
        hypothesis_pisa_pattern=hypothesis.pisa_pattern,
        hypothesis_reasoning=hypothesis.reasoning,
        confirmation_rate=validation.confirmation_rate,
        avg_ratio=validation.avg_makespan_ratio,
    )


# =============================================================================
# Main Loop
# =============================================================================

def run_simple_loop(
    target_scheduler: SchedulerName,
    baseline_scheduler: SchedulerName,
    max_iterations: int = 10,
) -> Tuple[Optional[CodeHypothesis], Optional[HypothesisValidationResult]]:
    """
    Run the simplified single-agent loop.

    Runs all iterations and returns the hypothesis with the highest avg_ratio.
    """
    results_dir = thisdir / "results"
    results_dir.mkdir(exist_ok=True)

    logger = ExperimentLogger(results_dir)
    logger.log_run_start(
        target_scheduler=target_scheduler,
        baseline_scheduler=baseline_scheduler,
        max_iterations=max_iterations,
        min_confidence_threshold=0.0,  # Not used anymore
    )

    state = SessionState(
        target_scheduler=target_scheduler,
        baseline_scheduler=baseline_scheduler,
        max_iterations=max_iterations,
    )

    agent = create_agent(target_scheduler, baseline_scheduler)

    print(f"Starting simple loop: Find where {target_scheduler} underperforms vs {baseline_scheduler}")
    print(f"Max iterations: {max_iterations}")
    print(f"Model: {MODEL}")
    print(f"Logs: {logger.log_dir}")
    print("=" * 60)

    while state.iteration < max_iterations:
        state.iteration += 1
        print(f"\n{'='*60}")
        print(f"ITERATION {state.iteration}/{max_iterations}")
        print("=" * 60)

        # Build the prompt with full history
        history = state.get_full_history()

        # Add constraints based on state
        constraints = []
        if state.has_comparison:
            constraints.append(
                "You have already compared algorithms. DO NOT choose 'compare_algorithms' again."
            )
        if state.pisa_count >= 1:
            constraints.append(
                "You have already run PISA. You MUST now use 'write_hypothesis' to write code. "
                "DO NOT choose 'run_pisa' or 'compare_algorithms' again."
            )
        if state.hypothesis_count > 0 and state.best_validation:
            constraints.append(
                f"Your best hypothesis has avg_ratio={state.best_validation.avg_makespan_ratio:.4f}. "
                f"Try to beat it!"
            )

        constraint_text = "\n".join(f"CONSTRAINT: {c}" for c in constraints) if constraints else ""

        prompt = f"""Iteration {state.iteration}/{max_iterations}

## Your Goal
Find task graph families where {target_scheduler} performs worse than {baseline_scheduler}.

## History
{history}

{constraint_text}

## Best Result So Far
{f"Hypothesis '{state.best_hypothesis.name}' with avg_ratio={state.best_validation.avg_makespan_ratio:.4f}" if state.best_hypothesis and state.best_validation else "No successful hypothesis yet"}

What do you want to do next? Analyze the history, then choose an action.
"""

        print(f"\nPrompt length: {len(prompt)} chars")
        logger.log_iteration_start(state.iteration, prompt[:500] + "...")

        # Get agent response
        result = agent.run_sync(prompt)
        response = result.output
        logger.log_token_usage("agent", result.usage(), MODEL.split(":")[-1])

        print(f"\nTHOUGHT: {response.thought[:300]}...")
        print(f"ACTION: {response.action}")

        # Execute the action
        record: Optional[AttemptRecord] = None

        if response.action == "compare_algorithms":
            if state.has_comparison:
                print("  WARNING: Already compared algorithms. Skipping.")
                logger.log_agent_decision("compare_algorithms (blocked)", response.thought, {})
                logger.log_action_result("Blocked - already done", {"blocked": True})
                continue

            record = handle_compare_algorithms(state, logger)
            logger.log_agent_decision("compare_algorithms", response.thought, {})
            logger.log_action_result(record.comparison_result or "", {})

        elif response.action == "run_pisa":
            if state.pisa_count >= 1:
                print("  WARNING: Ignoring run_pisa - already run once. Forcing write_hypothesis.")
                # Log but don't execute
                logger.log_agent_decision("run_pisa (blocked)", response.thought, {})
                logger.log_action_result("Blocked - must write hypothesis", {"blocked": True})
                continue

            record = handle_pisa(state, results_dir)
            logger.log_agent_decision("run_pisa", response.thought, {})
            logger.log_action_result(record.pisa_result or "", {"pisa_count": state.pisa_count + 1})

        elif response.action == "write_hypothesis":
            if not response.hypothesis:
                print("  ERROR: No hypothesis provided")
                logger.log_agent_decision("write_hypothesis (no code)", response.thought, {})
                logger.log_action_result("Error: no hypothesis provided", {"error": True})
                continue

            record = handle_hypothesis(state, response.hypothesis)
            logger.log_agent_decision("write_hypothesis", response.thought, {
                "code_hypothesis": {
                    "name": response.hypothesis.name,
                    "hypothesis_id": response.hypothesis.hypothesis_id,
                    "pisa_pattern": response.hypothesis.pisa_pattern,
                    "code": response.hypothesis.code,
                    "reasoning": response.hypothesis.reasoning,
                    "confidence": response.hypothesis.confidence,
                    "worse_scheduler": response.hypothesis.worse_scheduler,
                    "better_scheduler": response.hypothesis.better_scheduler,
                }
            })
            logger.log_action_result(
                f"avg_ratio={record.avg_ratio:.4f}" if record.avg_ratio else f"Error: {record.error}",
                {
                    "name": record.hypothesis_name,
                    "avg_ratio": record.avg_ratio,
                    "error": record.error,
                }
            )

        elif response.action == "test_instance":
            if not response.hypothesis:
                print("  ERROR: No hypothesis provided for test_instance")
                logger.log_agent_decision("test_instance (no code)", response.thought, {})
                logger.log_action_result("Error: no hypothesis provided", {"error": True})
                continue

            record = handle_test_instance(state, response.hypothesis)
            logger.log_agent_decision("test_instance", response.thought, {
                "hypothesis_name": response.hypothesis.name,
            })
            logger.log_action_result(record.test_instance_result or "", {})

        if record:
            state.attempts.append(record)

        logger.finalize_iteration()

    # Finalize
    logger.log_run_complete(state.best_hypothesis, state.best_validation, state.iteration)
    logger.save_visualization_data()

    print("\n" + "=" * 60)
    print("LOOP COMPLETE")
    print("=" * 60)

    if state.best_hypothesis and state.best_validation:
        print(f"\nBest Hypothesis: {state.best_hypothesis.name}")
        print(f"Avg ratio: {state.best_validation.avg_makespan_ratio:.4f}")
        print(f"Max ratio: {state.best_validation.max_makespan_ratio:.4f}")
        print(f"Min ratio: {state.best_validation.min_makespan_ratio:.4f}")
        print(f"\nCode:\n{state.best_hypothesis.code}")

        # Save outputs
        output_path = results_dir / "best_hypothesis.json"
        output_path.write_text(state.best_hypothesis.model_dump_json(indent=2))
        print(f"\nSaved to: {output_path}")

        code_path = results_dir / "best_hypothesis_code.py"
        code_path.write_text(f'''"""
{state.best_hypothesis.name}

{state.best_hypothesis.reasoning}

Generated by the simple agentic system.
Confirmation rate: {state.best_validation.confirmation_rate:.1%}
Avg makespan ratio: {state.best_validation.avg_makespan_ratio:.4f}
"""
from typing import Tuple
import networkx as nx
import numpy as np
import random
from itertools import product

from saga import Network, TaskGraph

{state.best_hypothesis.code}


if __name__ == "__main__":
    network, task_graph = get_instance()
    print(f"Network nodes: {{list(network.graph.nodes())}}")
    print(f"Task graph nodes: {{list(task_graph.graph.nodes())}}")
''')
        print(f"Code saved to: {code_path}")
    else:
        print("\nNo hypothesis found.")

    # Token summary
    token_summary = logger.get_token_summary()
    print(f"\n{'='*60}")
    print("TOKEN USAGE & COST")
    print("=" * 60)
    print(f"  Total input tokens:  {token_summary['total_input_tokens']:,}")
    print(f"  Total output tokens: {token_summary['total_output_tokens']:,}")
    print(f"  Total tokens:        {token_summary['total_tokens']:,}")
    print(f"  Estimated cost:      ${token_summary['estimated_cost_usd']:.4f} USD")

    print(f"\nLogs saved to: {logger.log_dir}")

    return state.best_hypothesis, state.best_validation


def main():
    """Run the simplified hypothesis generation system."""
    hypothesis, validation = run_simple_loop(
        target_scheduler="CPoP",
        baseline_scheduler="HEFT",
        max_iterations=10,
    )
    return hypothesis


if __name__ == "__main__":
    main()
