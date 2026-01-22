"""
Hypothesis execution and validation.

This module provides functions for:
- Safely executing code-based hypotheses
- Validating hypotheses across multiple instances
"""

import random
from itertools import product
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np

from saga import Network, TaskGraph
from saga.pisa.simulated_annealing import SCHEDULERS

from models import CodeHypothesis, HypothesisValidationResult


def execute_code_hypothesis(
    hypothesis: CodeHypothesis,
) -> Tuple[Optional[Network], Optional[TaskGraph], Optional[str]]:
    """
    Safely execute a code hypothesis to generate an instance.

    Returns:
        (network, task_graph, error_message)
        If successful, error_message is None.
        If failed, network and task_graph are None and error_message contains the error.
    """
    exec_globals = {
        "nx": nx,
        "np": np,
        "random": random,
        "product": product,
        "Network": Network,
        "TaskGraph": TaskGraph,
        "Tuple": Tuple,
        "List": List,
        "Dict": Dict,
    }
    exec_locals: Dict[str, Any] = {}

    try:
        exec(hypothesis.code, exec_globals, exec_locals)

        if "get_instance" not in exec_locals:
            return None, None, "Code must define a 'get_instance()' function"

        get_instance = exec_locals["get_instance"]
        result = get_instance()

        if not isinstance(result, tuple) or len(result) != 2:
            return None, None, "get_instance() must return a tuple of (Network, TaskGraph)"

        network, task_graph = result

        if not isinstance(network, Network):
            return None, None, f"First element must be Network, got {type(network)}"
        if not isinstance(task_graph, TaskGraph):
            return None, None, f"Second element must be TaskGraph, got {type(task_graph)}"

        return network, task_graph, None

    except Exception as e:
        return None, None, f"Error executing code: {type(e).__name__}: {str(e)}"


def validate_code_hypothesis(
    hypothesis: CodeHypothesis, num_instances: int = 50
) -> HypothesisValidationResult:
    """
    Validate a code hypothesis by generating and testing instances.

    Args:
        hypothesis: The code hypothesis to validate
        num_instances: Number of instances to generate and test

    Returns:
        HypothesisValidationResult with statistics about the validation
    """
    worse_scheduler = SCHEDULERS[hypothesis.worse_scheduler]
    better_scheduler = SCHEDULERS[hypothesis.better_scheduler]

    ratios = []
    confirmations = 0
    errors = 0

    for _ in range(num_instances):
        network, task_graph, error = execute_code_hypothesis(hypothesis)

        if error:
            errors += 1
            if errors > 5:
                break
            continue

        try:
            if network is None or task_graph is None:
                raise ValueError("Generated instance is None")

            worse_schedule = worse_scheduler.schedule(network, task_graph)
            better_schedule = better_scheduler.schedule(network, task_graph)

            ratio = worse_schedule.makespan / better_schedule.makespan
            ratios.append(ratio)

            if ratio > 1.0:
                confirmations += 1
        except Exception:
            errors += 1
            continue

    if not ratios:
        return HypothesisValidationResult(
            hypothesis_id=hypothesis.hypothesis_id,
            num_instances_tested=0,
            num_instances_confirmed=0,
            avg_makespan_ratio=1.0,
            max_makespan_ratio=1.0,
            min_makespan_ratio=1.0,
            confirmation_rate=0.0,
            is_validated=False,
        )

    return HypothesisValidationResult(
        hypothesis_id=hypothesis.hypothesis_id,
        num_instances_tested=len(ratios),
        num_instances_confirmed=confirmations,
        avg_makespan_ratio=float(np.mean(ratios)),
        max_makespan_ratio=float(max(ratios)),
        min_makespan_ratio=float(min(ratios)),
        confirmation_rate=float(confirmations / len(ratios)),
        is_validated=confirmations / len(ratios) > 0.5,
    )
