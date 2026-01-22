"""
LLM-Powered Hypothesis Generation for Scheduling Algorithm Analysis

This module uses Pydantic AI to have an LLM iteratively:
1. Explore scheduling algorithm source code
2. Run PISA experiments to find adversarial instances
3. Generate structured hypotheses about task graph families where one algorithm underperforms
4. Validate hypotheses by generating random instances from the family
5. Refine hypotheses based on validation results

The agent runs in an agentic loop, deciding when to gather more information,
run more experiments, or finalize a hypothesis.
"""

import inspect
import json
import logging
import pathlib
import random
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from itertools import product
from typing import Annotated, Any, Dict, List, Literal, Optional, Tuple, Union

import networkx as nx
import numpy as np
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext

from saga import Network, TaskGraph, TaskGraphNode, TaskGraphEdge
from saga.pisa.simulated_annealing import (
    SimulatedAnnealing,
    SimulatedAnnealingConfig,
    SimulatedAnnealingIteration,
    SchedulerName,
    SCHEDULERS,
)
from saga.schedulers import HeftScheduler, CpopScheduler
from saga.utils.random_graphs import get_network, add_random_weights
from saga.utils.random_variable import UniformRandomVariable

# Suppress the task graph warnings
logging.getLogger().setLevel(logging.ERROR)

load_dotenv()

thisdir = pathlib.Path(__file__).parent.resolve()


# =============================================================================
# Experiment Logger - Tracks all agent actions for visualization
# =============================================================================

class ExperimentLogger:
    """Logs all agent actions, reasoning, and results for analysis and visualization."""

    def __init__(self, results_dir: pathlib.Path, run_id: Optional[str] = None):
        self.run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = results_dir
        self.log_dir = results_dir / "logs" / self.run_id
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.run_start_time = time.time()
        self.events: List[Dict[str, Any]] = []
        self.iteration_logs: List[Dict[str, Any]] = []
        self.current_iteration: Dict[str, Any] = {}

        # Token usage tracking
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cached_input_tokens = 0
        self.token_usage_by_agent: Dict[str, Dict[str, Any]] = {}

        # Pricing per 1M tokens (USD)
        # Based on user-provided pricing for GPT-5 series
        self.pricing = {
            # GPT-5.2 - best for coding and agentic tasks
            "gpt-5.2": {"input": 1.75, "cached_input": 0.175, "output": 14.00},
            "gpt-5": {"input": 1.75, "cached_input": 0.175, "output": 14.00},  # Alias
            # GPT-5.2 pro - smartest and most precise
            "gpt-5.2-pro": {"input": 21.00, "cached_input": 0.0, "output": 168.00},
            # GPT-5 mini - faster, cheaper for well-defined tasks
            "gpt-5-mini": {"input": 0.25, "cached_input": 0.025, "output": 2.00},
            # GPT-4o (current model being used) - estimate based on typical pricing
            "gpt-4o": {"input": 2.50, "cached_input": 1.25, "output": 10.00},
            # Fallback for other models (use gpt-5.2 pricing as default)
            "default": {"input": 1.75, "cached_input": 0.175, "output": 14.00},
        }

        # Initialize the run metadata
        self.run_metadata: Dict[str, Any] = {
            "run_id": self.run_id,
            "start_time": datetime.now().isoformat(),
            "status": "running",
        }

    def log_run_start(
        self,
        target_scheduler: str,
        baseline_scheduler: str,
        max_iterations: int,
        min_confidence_threshold: float,
    ):
        """Log the start of an experiment run."""
        self.run_metadata.update({
            "target_scheduler": target_scheduler,
            "baseline_scheduler": baseline_scheduler,
            "max_iterations": max_iterations,
            "min_confidence_threshold": min_confidence_threshold,
        })
        self._save_run_metadata()

        self._log_event("run_start", {
            "target_scheduler": target_scheduler,
            "baseline_scheduler": baseline_scheduler,
            "max_iterations": max_iterations,
        })

    def log_iteration_start(self, iteration: int, context_summary: str):
        """Log the start of an iteration."""
        self.current_iteration = {
            "iteration": iteration,
            "start_time": time.time(),
            "context_summary": context_summary,
            "action": None,
            "reasoning": None,
            "result": None,
            "duration_seconds": None,
        }
        self._log_event("iteration_start", {"iteration": iteration})

    def log_strategic_plan(self, plan: "StrategicPlan"):
        """Log the strategic plan for this iteration."""
        self.current_iteration["strategic_plan"] = {
            "current_understanding": plan.current_understanding,
            "key_unknowns": plan.key_unknowns,
            "working_hypothesis": plan.working_hypothesis,
            "next_steps": plan.next_steps,
            "immediate_action": plan.immediate_action,
            "success_criteria": plan.success_criteria,
        }

        self._log_event("strategic_plan", {
            "current_understanding": plan.current_understanding[:200],
            "key_unknowns": plan.key_unknowns,
            "working_hypothesis": plan.working_hypothesis,
            "next_steps": plan.next_steps,
        })

    def log_agent_decision(self, action: str, reasoning: str, action_params: Optional[Dict[str, Any]] = None):
        """Log what action the agent decided to take and why."""
        self.current_iteration["action"] = action
        self.current_iteration["reasoning"] = reasoning
        self.current_iteration["action_params"] = action_params or {}

        self._log_event("agent_decision", {
            "action": action,
            "reasoning": reasoning,
            "params": action_params,
        })

    def log_reflection(self, reflection: "ActionReflection"):
        """Log the reflection on an action's results."""
        self.current_iteration["reflection"] = {
            "action_taken": reflection.action_taken,
            "key_findings": reflection.key_findings,
            "surprises": reflection.surprises,
            "hypothesis_update": reflection.hypothesis_update,
            "next_question": reflection.next_question,
            "confidence_assessment": reflection.confidence_assessment,
        }

        self._log_event("action_reflection", {
            "key_findings": reflection.key_findings,
            "surprises": reflection.surprises,
            "hypothesis_update": reflection.hypothesis_update,
            "confidence_assessment": reflection.confidence_assessment,
        })

    def log_action_result(self, result: str, result_data: Optional[Dict[str, Any]] = None):
        """Log the result of an action."""
        self.current_iteration["result"] = result
        self.current_iteration["result_data"] = result_data or {}
        start_time = self.current_iteration.get("start_time", time.time())
        self.current_iteration["duration_seconds"] = time.time() - float(start_time)

        self._log_event("action_result", {
            "result_preview": result[:500] if result else None,
            "result_data": result_data,
        })
        # Note: iteration is saved after reflection in finalize_iteration()

    def finalize_iteration(self):
        """Finalize and save the current iteration after reflection is logged."""
        self.iteration_logs.append(self.current_iteration.copy())
        self._save_iteration_log(self.current_iteration)

    def log_token_usage(self, agent_name: str, usage: Any, model: str = "default"):
        """Log token usage from an agent call.

        Args:
            agent_name: Name of the agent (e.g., "planning", "decision", "reflection")
            usage: The usage object from pydantic_ai result (has input_tokens, output_tokens, etc.)
            model: The model name for pricing lookup
        """
        if usage is None:
            return

        # Extract token counts - use the newer attribute names (input_tokens/output_tokens)
        input_tokens = getattr(usage, 'input_tokens', 0) or 0
        output_tokens = getattr(usage, 'output_tokens', 0) or 0
        # Pydantic AI may report cached tokens differently
        cached_tokens = getattr(usage, 'cached_tokens', 0) or 0

        # Update totals
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_cached_input_tokens += cached_tokens

        # Update per-agent tracking
        if agent_name not in self.token_usage_by_agent:
            self.token_usage_by_agent[agent_name] = {
                "input_tokens": 0,
                "output_tokens": 0,
                "cached_tokens": 0,
                "calls": 0,
            }
        self.token_usage_by_agent[agent_name]["input_tokens"] += input_tokens
        self.token_usage_by_agent[agent_name]["output_tokens"] += output_tokens
        self.token_usage_by_agent[agent_name]["cached_tokens"] += cached_tokens
        self.token_usage_by_agent[agent_name]["calls"] += 1

        # Log the event
        self._log_event("token_usage", {
            "agent": agent_name,
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cached_tokens": cached_tokens,
        })

        # Track model for this agent (for per-model cost calculation)
        if "model" not in self.token_usage_by_agent[agent_name]:
            self.token_usage_by_agent[agent_name]["model"] = model

        # Track in current iteration
        if "token_usage" not in self.current_iteration:
            self.current_iteration["token_usage"] = []
        self.current_iteration["token_usage"].append({
            "agent": agent_name,
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cached_tokens": cached_tokens,
        })

    def get_total_cost(self) -> float:
        """Calculate total cost based on token usage and pricing per agent/model."""
        total_cost = 0.0

        for agent_name, usage in self.token_usage_by_agent.items():
            model = usage.get("model", "default")
            # Normalize model name for pricing lookup
            model_key = model.lower().replace("openai:", "").replace("_", "-")
            pricing = self.pricing.get(model_key, self.pricing["default"])

            # Calculate costs for this agent (pricing is per 1M tokens)
            input_cost = (usage["input_tokens"] / 1_000_000) * pricing["input"]
            cached_cost = (usage["cached_tokens"] / 1_000_000) * pricing["cached_input"]
            output_cost = (usage["output_tokens"] / 1_000_000) * pricing["output"]

            total_cost += input_cost + cached_cost + output_cost

        return total_cost

    def get_token_summary(self) -> Dict[str, Any]:
        """Get a summary of token usage."""
        return {
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cached_tokens": self.total_cached_input_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "by_agent": self.token_usage_by_agent,
            "estimated_cost_usd": self.get_total_cost(),
        }

    def log_hypothesis_tested(
        self,
        hypothesis: "CodeHypothesis",
        validation: "HypothesisValidationResult",
        is_best: bool = False,
    ):
        """Log when a hypothesis is tested."""
        hyp_data = {
            "hypothesis_id": hypothesis.hypothesis_id,
            "name": hypothesis.name,
            "reasoning": hypothesis.reasoning,
            "confidence": hypothesis.confidence,
            "code_preview": hypothesis.code[:500] + "..." if len(hypothesis.code) > 500 else hypothesis.code,
        }

        val_data = {
            "num_instances_tested": validation.num_instances_tested,
            "confirmation_rate": validation.confirmation_rate,
            "avg_makespan_ratio": validation.avg_makespan_ratio,
            "max_makespan_ratio": validation.max_makespan_ratio,
            "min_makespan_ratio": validation.min_makespan_ratio,
            "is_validated": validation.is_validated,
        }

        self._log_event("hypothesis_tested", {
            "hypothesis": hyp_data,
            "validation": val_data,
            "is_new_best": is_best,
        })

    def log_run_complete(
        self,
        best_hypothesis: Optional["CodeHypothesis"],
        best_validation: Optional["HypothesisValidationResult"],
        total_iterations: int,
    ):
        """Log the completion of a run."""
        self.run_metadata["status"] = "completed"
        self.run_metadata["end_time"] = datetime.now().isoformat()
        self.run_metadata["total_iterations"] = total_iterations
        self.run_metadata["total_duration_seconds"] = time.time() - self.run_start_time

        if best_hypothesis and best_validation:
            self.run_metadata["best_hypothesis"] = {
                "hypothesis_id": best_hypothesis.hypothesis_id,
                "name": best_hypothesis.name,
                "reasoning": best_hypothesis.reasoning[:200] + "..." if len(best_hypothesis.reasoning) > 200 else best_hypothesis.reasoning,
                "confirmation_rate": best_validation.confirmation_rate,
                "avg_makespan_ratio": best_validation.avg_makespan_ratio,
                "is_validated": best_validation.is_validated,
            }

        # Add token usage summary
        self.run_metadata["token_usage"] = self.get_token_summary()

        self._save_run_metadata()
        self._save_all_events()
        self._generate_summary_report()

    def _log_event(self, event_type: str, data: Dict[str, Any]):
        """Internal method to log an event."""
        event = {
            "timestamp": datetime.now().isoformat(),
            "elapsed_seconds": time.time() - self.run_start_time,
            "event_type": event_type,
            "data": data,
        }
        self.events.append(event)

    def _save_run_metadata(self):
        """Save run metadata to disk."""
        path = self.log_dir / "run_metadata.json"
        with open(path, "w") as f:
            json.dump(self.run_metadata, f, indent=2, default=str)

    def _save_iteration_log(self, iteration_data: Dict[str, Any]):
        """Save a single iteration log."""
        iteration_num = iteration_data["iteration"]
        path = self.log_dir / f"iteration_{iteration_num:03d}.json"
        with open(path, "w") as f:
            json.dump(iteration_data, f, indent=2, default=str)

    def _save_all_events(self):
        """Save all events to a single file."""
        path = self.log_dir / "all_events.json"
        with open(path, "w") as f:
            json.dump(self.events, f, indent=2, default=str)

    def _generate_summary_report(self):
        """Generate a human-readable summary report."""
        report_lines = [
            "=" * 70,
            f"EXPERIMENT RUN SUMMARY: {self.run_id}",
            "=" * 70,
            "",
            f"Target: {self.run_metadata.get('target_scheduler')} vs {self.run_metadata.get('baseline_scheduler')}",
            f"Status: {self.run_metadata.get('status')}",
            f"Total iterations: {self.run_metadata.get('total_iterations', len(self.iteration_logs))}",
            f"Total duration: {self.run_metadata.get('total_duration_seconds', 0):.1f} seconds",
            "",
            "-" * 70,
            "ITERATION TIMELINE",
            "-" * 70,
        ]

        for it in self.iteration_logs:
            report_lines.append(
                f"\n[Iteration {it['iteration']}] Action: {it['action']} ({it.get('duration_seconds', 0):.1f}s)"
            )

            # Strategic Plan
            plan = it.get('strategic_plan', {})
            if plan:
                report_lines.append(f"  📋 Plan: {plan.get('immediate_action', 'N/A')[:80]}...")
                if plan.get('working_hypothesis'):
                    report_lines.append(f"     Working hypothesis: {plan['working_hypothesis'][:80]}...")

            # Action reasoning
            report_lines.append(f"  🎯 Reasoning: {it['reasoning'][:100]}..." if it.get('reasoning') else "  Reasoning: N/A")

            # Add specific details based on action type
            if it['action'] in ('test_code_hypothesis', 'submit_code_hypothesis'):
                result_data = it.get('result_data', {})
                if result_data:
                    report_lines.append(
                        f"     Result: confirmation_rate={result_data.get('confirmation_rate', 'N/A')}, "
                        f"avg_ratio={result_data.get('avg_makespan_ratio', 'N/A')}"
                    )

            # Reflection
            reflection = it.get('reflection', {})
            if reflection:
                report_lines.append(f"  💭 Key findings: {'; '.join(reflection.get('key_findings', [])[:2])[:80]}...")
                report_lines.append(f"     Hypothesis update: {reflection.get('hypothesis_update', 'N/A')[:80]}...")

        # Best hypothesis summary
        if self.run_metadata.get('best_hypothesis'):
            bh = self.run_metadata['best_hypothesis']
            report_lines.extend([
                "",
                "-" * 70,
                "BEST HYPOTHESIS FOUND",
                "-" * 70,
                f"  Name: {bh.get('name')}",
                f"  Reasoning: {bh.get('reasoning', 'N/A')[:100]}...",
                f"  Confirmation rate: {bh.get('confirmation_rate', 0):.1%}",
                f"  Avg makespan ratio: {bh.get('avg_makespan_ratio', 0):.4f}",
                f"  Validated: {bh.get('is_validated')}",
            ])

        # Token usage summary
        token_summary = self.get_token_summary()
        report_lines.extend([
            "",
            "-" * 70,
            "TOKEN USAGE & COST",
            "-" * 70,
            f"  Total input tokens:  {token_summary['total_input_tokens']:,}",
            f"  Total output tokens: {token_summary['total_output_tokens']:,}",
            f"  Total cached tokens: {token_summary['total_cached_tokens']:,}",
            f"  Total tokens:        {token_summary['total_tokens']:,}",
            f"  Estimated cost:      ${token_summary['estimated_cost_usd']:.4f} USD",
            "",
            "  By Agent:",
        ])
        for agent_name, usage in token_summary['by_agent'].items():
            report_lines.append(
                f"    {agent_name}: {usage['calls']} calls, "
                f"{usage['input_tokens']:,} in, {usage['output_tokens']:,} out"
            )

        report_lines.extend(["", "=" * 70])

        # Save report
        report_path = self.log_dir / "summary_report.txt"
        with open(report_path, "w") as f:
            f.write("\n".join(report_lines))

        # Also print to console
        print("\n".join(report_lines))

    def get_visualization_data(self) -> Dict[str, Any]:
        """Get data formatted for visualization."""
        return {
            "run_id": self.run_id,
            "metadata": self.run_metadata,
            "iterations": [
                {
                    "iteration": it["iteration"],
                    "action": it["action"],
                    "reasoning": it["reasoning"],
                    "duration": it.get("duration_seconds", 0),
                    "result_preview": it.get("result", "")[:200] if it.get("result") else None,
                    "token_usage": it.get("token_usage", []),
                }
                for it in self.iteration_logs
            ],
            "events": self.events,
            "token_summary": self.get_token_summary(),
        }

    def save_visualization_data(self):
        """Save visualization-ready data."""
        path = self.log_dir / "visualization_data.json"
        with open(path, "w") as f:
            json.dump(self.get_visualization_data(), f, indent=2, default=str)

        # Also generate HTML visualization
        self._generate_html_visualization()

    def _generate_html_visualization(self):
        """Generate an interactive HTML visualization of the experiment."""
        html_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Experiment Run: {self.run_id}</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: #1a1a2e;
            color: #eee;
            padding: 20px;
            line-height: 1.6;
        }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        h1 {{ color: #00d9ff; margin-bottom: 20px; }}
        h2 {{ color: #ff6b6b; margin: 20px 0 10px; }}
        .metadata {{
            background: #16213e;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
        }}
        .metadata-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }}
        .metric {{
            background: #0f3460;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }}
        .metric-value {{
            font-size: 24px;
            font-weight: bold;
            color: #00d9ff;
        }}
        .metric-label {{ color: #aaa; font-size: 12px; }}
        .timeline {{
            position: relative;
            padding-left: 30px;
        }}
        .timeline::before {{
            content: '';
            position: absolute;
            left: 10px;
            top: 0;
            bottom: 0;
            width: 2px;
            background: #0f3460;
        }}
        .iteration {{
            background: #16213e;
            margin-bottom: 15px;
            padding: 20px;
            border-radius: 10px;
            position: relative;
        }}
        .iteration::before {{
            content: '';
            position: absolute;
            left: -24px;
            top: 25px;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background: #00d9ff;
        }}
        .iteration-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }}
        .iteration-number {{
            font-size: 18px;
            font-weight: bold;
            color: #00d9ff;
        }}
        .action-badge {{
            padding: 5px 12px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: bold;
        }}
        .action-compare_algorithms {{ background: #9b59b6; }}
        .action-read_source_code {{ background: #3498db; }}
        .action-run_pisa {{ background: #e74c3c; }}
        .action-run_benchmark {{ background: #f39c12; }}
        .action-refine_hypothesis {{ background: #1abc9c; }}
        .action-submit_hypothesis {{ background: #2ecc71; }}
        .reasoning {{
            background: #0f3460;
            padding: 15px;
            border-radius: 8px;
            margin-top: 10px;
            font-style: italic;
            color: #bbb;
        }}
        .result {{
            background: #0a0a1a;
            padding: 15px;
            border-radius: 8px;
            margin-top: 10px;
            font-family: monospace;
            font-size: 13px;
            white-space: pre-wrap;
            max-height: 300px;
            overflow-y: auto;
        }}
        .duration {{ color: #888; font-size: 12px; }}
        .hypothesis-card {{
            background: linear-gradient(135deg, #1abc9c 0%, #16a085 100%);
            padding: 20px;
            border-radius: 10px;
            margin-top: 20px;
        }}
        .hypothesis-card h3 {{ margin-bottom: 15px; }}
        .hypothesis-stats {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 10px;
        }}
        .stat {{
            background: rgba(0,0,0,0.2);
            padding: 10px;
            border-radius: 5px;
            text-align: center;
        }}
        .collapsible {{
            cursor: pointer;
            user-select: none;
        }}
        .collapsible::after {{
            content: ' ▼';
            font-size: 10px;
        }}
        .collapsed::after {{
            content: ' ▶';
        }}
        .content {{ display: block; }}
        .content.hidden {{ display: none; }}

        /* Planning and Reflection sections */
        .phase-section {{
            margin: 10px 0;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid;
        }}
        .plan-section {{
            background: rgba(52, 152, 219, 0.1);
            border-color: #3498db;
        }}
        .reflection-section {{
            background: rgba(155, 89, 182, 0.1);
            border-color: #9b59b6;
        }}
        .action-section {{
            background: rgba(46, 204, 113, 0.1);
            border-left: 4px solid #2ecc71;
            padding: 15px;
            border-radius: 8px;
            margin: 10px 0;
        }}
        .phase-header {{
            font-weight: bold;
            color: #fff;
            margin-bottom: 10px;
        }}
        .phase-content {{
            font-size: 13px;
            color: #bbb;
        }}
        .phase-content p {{
            margin: 5px 0;
        }}
        .phase-content strong {{
            color: #ddd;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔬 Experiment Run: {self.run_id}</h1>

        <div class="metadata">
            <div class="metadata-grid">
                <div class="metric">
                    <div class="metric-value">{self.run_metadata.get('target_scheduler', 'N/A')}</div>
                    <div class="metric-label">Target Scheduler</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{self.run_metadata.get('baseline_scheduler', 'N/A')}</div>
                    <div class="metric-label">Baseline Scheduler</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{self.run_metadata.get('total_iterations', len(self.iteration_logs))}</div>
                    <div class="metric-label">Total Iterations</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{self.run_metadata.get('total_duration_seconds', 0):.1f}s</div>
                    <div class="metric-label">Duration</div>
                </div>
            </div>
        </div>

        {"".join(self._generate_token_usage_html())}

        {"".join(self._generate_best_hypothesis_html())}

        <h2>📊 Iteration Timeline</h2>
        <div class="timeline">
            {"".join(self._generate_iteration_html(it) for it in self.iteration_logs)}
        </div>
    </div>

    <script>
        document.querySelectorAll('.collapsible').forEach(el => {{
            el.addEventListener('click', () => {{
                el.classList.toggle('collapsed');
                el.nextElementSibling.classList.toggle('hidden');
            }});
        }});
    </script>
</body>
</html>'''

        path = self.log_dir / "visualization.html"
        with open(path, "w") as f:
            f.write(html_content)

    def _generate_iteration_html(self, iteration: Dict[str, Any]) -> str:
        """Generate HTML for a single iteration."""
        action = iteration.get('action', 'unknown')
        reasoning = iteration.get('reasoning', '')
        result = iteration.get('result', '')
        duration = iteration.get('duration_seconds', 0)
        strategic_plan = iteration.get('strategic_plan', {})
        reflection = iteration.get('reflection', {})

        result_preview = result[:1000] if result else "No result"
        if len(result) > 1000:
            result_preview += "..."

        # Generate strategic plan HTML
        plan_html = ""
        if strategic_plan:
            plan_html = f'''
            <div class="phase-section plan-section">
                <div class="phase-header">📋 Strategic Plan</div>
                <div class="phase-content">
                    <p><strong>Understanding:</strong> {strategic_plan.get('current_understanding', 'N/A')[:300]}...</p>
                    <p><strong>Key Unknowns:</strong> {', '.join(strategic_plan.get('key_unknowns', [])[:3])}</p>
                    <p><strong>Working Hypothesis:</strong> {strategic_plan.get('working_hypothesis', 'None yet')}</p>
                    <p><strong>Next Steps:</strong> {'; '.join(strategic_plan.get('next_steps', [])[:2])}</p>
                </div>
            </div>
            '''

        # Generate reflection HTML
        reflection_html = ""
        if reflection:
            reflection_html = f'''
            <div class="phase-section reflection-section">
                <div class="phase-header">💭 Reflection</div>
                <div class="phase-content">
                    <p><strong>Key Findings:</strong> {'; '.join(reflection.get('key_findings', [])[:3])}</p>
                    {'<p><strong>Surprises:</strong> ' + '; '.join(reflection.get('surprises', [])) + '</p>' if reflection.get('surprises') else ''}
                    <p><strong>Hypothesis Update:</strong> {reflection.get('hypothesis_update', 'N/A')[:200]}...</p>
                    <p><strong>Next Question:</strong> {reflection.get('next_question', 'N/A')}</p>
                    <p><strong>Confidence:</strong> {reflection.get('confidence_assessment', 'N/A')[:150]}...</p>
                </div>
            </div>
            '''

        return f'''
        <div class="iteration">
            <div class="iteration-header">
                <span class="iteration-number">Iteration {iteration.get('iteration', '?')}</span>
                <span class="action-badge action-{action}">{action.replace('_', ' ').upper()}</span>
                <span class="duration">{duration:.1f}s</span>
            </div>
            {plan_html}
            <div class="action-section">
                <div class="phase-header collapsible">🎯 Action: {action.replace('_', ' ').title()}</div>
                <div class="content">
                    <p><strong>Reasoning:</strong> {reasoning}</p>
                </div>
                <div class="result collapsible">📋 Result</div>
                <div class="content result">{result_preview}</div>
            </div>
            {reflection_html}
        </div>
        '''

    def _generate_token_usage_html(self) -> List[str]:
        """Generate HTML for the token usage section."""
        token_summary = self.get_token_summary()

        # Generate per-agent breakdown
        agent_rows = ""
        for agent_name, usage in token_summary['by_agent'].items():
            agent_rows += f'''
            <tr>
                <td style="padding: 8px; border-bottom: 1px solid #0f3460;">{agent_name}</td>
                <td style="padding: 8px; border-bottom: 1px solid #0f3460; text-align: right;">{usage['calls']}</td>
                <td style="padding: 8px; border-bottom: 1px solid #0f3460; text-align: right;">{usage['input_tokens']:,}</td>
                <td style="padding: 8px; border-bottom: 1px solid #0f3460; text-align: right;">{usage['output_tokens']:,}</td>
            </tr>
            '''

        return [f'''
        <div class="metadata" style="margin-bottom: 20px;">
            <h2 style="color: #f39c12; margin-bottom: 15px;">💰 Token Usage & Cost</h2>
            <div class="metadata-grid">
                <div class="metric">
                    <div class="metric-value">{token_summary['total_input_tokens']:,}</div>
                    <div class="metric-label">Input Tokens</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{token_summary['total_output_tokens']:,}</div>
                    <div class="metric-label">Output Tokens</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{token_summary['total_tokens']:,}</div>
                    <div class="metric-label">Total Tokens</div>
                </div>
                <div class="metric" style="background: linear-gradient(135deg, #f39c12 0%, #e74c3c 100%);">
                    <div class="metric-value">${token_summary['estimated_cost_usd']:.4f}</div>
                    <div class="metric-label">Estimated Cost (USD)</div>
                </div>
            </div>
            <div style="margin-top: 15px;">
                <table style="width: 100%; border-collapse: collapse; font-size: 13px;">
                    <thead>
                        <tr style="background: #0f3460;">
                            <th style="padding: 10px; text-align: left;">Agent</th>
                            <th style="padding: 10px; text-align: right;">Calls</th>
                            <th style="padding: 10px; text-align: right;">Input</th>
                            <th style="padding: 10px; text-align: right;">Output</th>
                        </tr>
                    </thead>
                    <tbody>
                        {agent_rows}
                    </tbody>
                </table>
            </div>
        </div>
        ''']

    def _generate_best_hypothesis_html(self) -> List[str]:
        """Generate HTML for the best hypothesis section."""
        bh = self.run_metadata.get('best_hypothesis')
        if not bh:
            return []

        return [f'''
        <div class="hypothesis-card">
            <h3>🏆 Best Hypothesis Found: {bh.get('name', 'N/A')}</h3>
            <p style="color: #ddd; margin: 10px 0;">{bh.get('reasoning', 'N/A')}</p>
            <div class="hypothesis-stats">
                <div class="stat">
                    <div class="metric-value">{bh.get('confirmation_rate', 0):.1%}</div>
                    <div class="metric-label">Confirmation Rate</div>
                </div>
                <div class="stat">
                    <div class="metric-value">{bh.get('avg_makespan_ratio', 0):.4f}</div>
                    <div class="metric-label">Avg Makespan Ratio</div>
                </div>
                <div class="stat">
                    <div class="metric-value">{'✓' if bh.get('is_validated') else '✗'}</div>
                    <div class="metric-label">Validated</div>
                </div>
            </div>
        </div>
        ''']


# =============================================================================
# Pydantic Models for Hypothesis Validation
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
    is_validated: bool = Field(
        description="True if confirmation_rate > 0.5"
    )


# =============================================================================
# Code-Based Hypothesis - The hypothesis IS executable code
# =============================================================================

class CodeHypothesis(BaseModel):
    """
    A hypothesis expressed as executable Python code that generates task graph instances.

    The code should define a function `get_instance()` that returns a (Network, TaskGraph) tuple.
    This gives the LLM complete freedom to create any pathological case it can imagine.
    """
    hypothesis_id: str = Field(
        default_factory=lambda: str(uuid.uuid4())[:8],
        description="Unique identifier for this hypothesis"
    )

    name: str = Field(
        ..., description="Short descriptive name for this hypothesis (e.g., 'imbalanced_diamond')"
    )

    worse_scheduler: SchedulerName = Field(
        ..., description="The scheduler predicted to perform worse"
    )
    better_scheduler: SchedulerName = Field(
        ..., description="The scheduler predicted to perform better (baseline)"
    )

    reasoning: str = Field(
        ..., description="Explanation of WHY this code creates instances where worse_scheduler underperforms"
    )

    code: str = Field(
        ..., description="""Python code that defines a get_instance() function.

The function signature must be:
    def get_instance() -> Tuple[Network, TaskGraph]:

IMPORTANT: Use randomization! The function will be called MANY times to generate a FAMILY
of problem instances. Each call should return a DIFFERENT instance with the same
pathological structure but varied weights/sizes. This tests whether the hypothesis
holds across the entire family, not just one specific instance.

Available imports (already imported):
    - networkx as nx
    - numpy as np
    - random
    - from itertools import product
    - Network, TaskGraph (from saga)

Example code (WITH RANDOMIZATION):
```python
def get_instance() -> Tuple[Network, TaskGraph]:
    # Create a diamond DAG where one branch is much heavier than the other
    # Use randomization to create a FAMILY of instances with this property

    # Random number of middle branches (2-5)
    num_branches = random.randint(2, 5)

    dag = nx.DiGraph()
    dag.add_node("source", weight=random.uniform(0.1, 0.5))
    dag.add_node("sink", weight=random.uniform(0.1, 0.5))

    # Create branches - one is 10x heavier than the others
    heavy_branch = random.randint(0, num_branches - 1)
    for i in range(num_branches):
        branch_name = f"branch_{i}"
        if i == heavy_branch:
            # This branch is much heavier (5-10x)
            weight = random.uniform(3.0, 8.0)
            edge_weight = random.uniform(2.0, 5.0)
        else:
            # Light branches
            weight = random.uniform(0.1, 0.5)
            edge_weight = random.uniform(0.1, 0.5)

        dag.add_node(branch_name, weight=weight)
        dag.add_edge("source", branch_name, weight=random.uniform(0.1, 0.3))
        dag.add_edge(branch_name, "sink", weight=edge_weight)

    task_graph = TaskGraph.from_nx(dag)

    # Heterogeneous network with random speeds
    num_processors = random.randint(2, 4)
    net = nx.Graph()
    for i in range(num_processors):
        # One fast processor, others slower
        if i == 0:
            speed = random.uniform(5.0, 10.0)  # Fast
        else:
            speed = random.uniform(0.5, 2.0)   # Slower
        net.add_node(f"p{i}", weight=speed)

    # Fully connected network
    for i in range(num_processors):
        for j in range(num_processors):
            if i == j:
                net.add_edge(f"p{i}", f"p{j}", weight=1e9)  # Self-loop = infinite bandwidth
            else:
                net.add_edge(f"p{i}", f"p{j}", weight=random.uniform(0.5, 2.0))

    network = Network.from_nx(net)
    return network, task_graph
```
"""
    )

    confidence: float = Field(
        default=0.5,
        description="Confidence level in this hypothesis (0.0 to 1.0)",
        ge=0.0, le=1.0
    )


def execute_code_hypothesis(hypothesis: CodeHypothesis) -> Tuple[Optional[Network], Optional[TaskGraph], Optional[str]]:
    """
    Safely execute a code hypothesis to generate an instance.

    Returns:
        (network, task_graph, error_message)
        If successful, error_message is None.
        If failed, network and task_graph are None and error_message contains the error.
    """
    # Create a restricted execution environment
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
        # Execute the code to define the function
        exec(hypothesis.code, exec_globals, exec_locals)

        # Check that get_instance was defined
        if "get_instance" not in exec_locals:
            return None, None, "Code must define a 'get_instance()' function"

        get_instance = exec_locals["get_instance"]

        # Call the function
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
    hypothesis: CodeHypothesis,
    num_instances: int = 50
) -> HypothesisValidationResult:
    """Validate a code hypothesis by generating and testing instances."""
    worse_scheduler = SCHEDULERS[hypothesis.worse_scheduler]
    better_scheduler = SCHEDULERS[hypothesis.better_scheduler]

    ratios = []
    confirmations = 0
    errors = 0

    for _ in range(num_instances):
        network, task_graph, error = execute_code_hypothesis(hypothesis)

        if error:
            errors += 1
            if errors > 5:  # Too many errors, abort
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


# =============================================================================
# Agent Decision Model - What to do next
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

    # Optional parameters for different actions
    scheduler_to_read: Optional[SchedulerName] = Field(
        default=None, description="Which scheduler's source code to read"
    )

    code_hypothesis: Optional[CodeHypothesis] = Field(
        default=None,
        description="Code-based hypothesis with get_instance() function that generates (Network, TaskGraph) instances."
    )


class StrategicPlan(BaseModel):
    """A strategic plan for the current iteration."""
    current_understanding: str = Field(
        ..., description="Summary of what we currently understand about the algorithms"
    )
    key_unknowns: List[str] = Field(
        ..., description="What we still don't know that would help form a hypothesis"
    )
    working_hypothesis: Optional[str] = Field(
        default=None, description="Current working hypothesis about what might cause poor performance"
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
        default_factory=list,
        description="Anything unexpected that we learned"
    )
    hypothesis_update: str = Field(
        ..., description="How this changes our working hypothesis (or confirms it)"
    )
    next_question: str = Field(
        ..., description="The most important question we should answer next"
    )
    confidence_assessment: str = Field(
        ..., description="How confident are we now in forming a valid hypothesis? Why?"
    )


# =============================================================================
# Agent State - Tracks conversation history and findings
# =============================================================================

@dataclass
class AgentState:
    """Tracks the agent's state across iterations."""
    results_dir: pathlib.Path
    logger: ExperimentLogger
    target_scheduler: SchedulerName = "HEFT"
    baseline_scheduler: SchedulerName = "CPoP"

    iteration: int = 0
    max_iterations: int = 10

    # Track what the agent has learned
    algorithm_comparison: Optional[str] = None
    source_code_read: List[SchedulerName] = field(default_factory=list)
    pisa_results: List[str] = field(default_factory=list)
    code_hypotheses_tested: List[Tuple[CodeHypothesis, HypothesisValidationResult]] = field(default_factory=list)

    # Best hypothesis so far
    best_hypothesis: Optional[CodeHypothesis] = None
    best_validation: Optional[HypothesisValidationResult] = None

    def get_context_summary(self) -> str:
        """Generate a summary of what the agent has learned so far."""
        summary_parts = [
            f"=== ITERATION {self.iteration}/{self.max_iterations} ===",
            f"Target: Find task graph families where {self.target_scheduler} performs worse than {self.baseline_scheduler}",
            "",
        ]

        if self.algorithm_comparison:
            summary_parts.append("Algorithm comparison: COMPLETED")
            # Include key insights from comparison (truncated)
            summary_parts.append(f"  {self.algorithm_comparison[:500]}...")

        if self.source_code_read:
            summary_parts.append(f"\nSource code read: {', '.join(self.source_code_read)}")

        if self.pisa_results:
            summary_parts.append(f"\nPISA experiments run: {len(self.pisa_results)}")
            for i, result in enumerate(self.pisa_results[-3:], 1):  # Show last 3
                summary_parts.append(f"  Recent PISA {i}: {result[:200]}...")

        if self.code_hypotheses_tested:
            summary_parts.append(f"\nCode hypotheses tested: {len(self.code_hypotheses_tested)}")
            for hyp, val in self.code_hypotheses_tested[-3:]:
                summary_parts.append(
                    f"  - {hyp.name}: "
                    f"confirmation={val.confirmation_rate:.1%}, avg_ratio={val.avg_makespan_ratio:.3f}"
                )

        if self.best_hypothesis:
            summary_parts.append(f"\nBest hypothesis so far:")
            summary_parts.append(f"  Name: {self.best_hypothesis.name}")
            summary_parts.append(f"  Reasoning: {self.best_hypothesis.reasoning[:150]}...")
            if self.best_validation:
                summary_parts.append(f"  Confirmation rate: {self.best_validation.confirmation_rate:.1%}")
                summary_parts.append(f"  Avg makespan ratio: {self.best_validation.avg_makespan_ratio:.4f}")

        return "\n".join(summary_parts)


# =============================================================================
# Pydantic AI Agent - Decision Maker
# =============================================================================

# =============================================================================
# Planning Agent - Strategic planning before actions
# =============================================================================

planning_agent = Agent(
    "openai:gpt-5",
    output_type=StrategicPlan,
    system_prompt="""You are a strategic planner for algorithm analysis research.

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
""",
)


# =============================================================================
# Reflection Agent - Learning from action results
# =============================================================================

reflection_agent = Agent(
    "openai:gpt-5",
    output_type=ActionReflection,
    system_prompt="""You are a reflective analyst reviewing the results of a research action.

Your job is to extract maximum insight from what was just learned. Consider:

1. What are the KEY findings from this action? (Be specific with numbers and patterns)
2. Was anything surprising or unexpected?
3. How does this change our working hypothesis?
4. What is the single most important question we should investigate next?
5. How confident should we be in forming a valid hypothesis now?

Be concrete and analytical. Focus on actionable insights that move us toward a validated hypothesis.
Connect findings back to the algorithm mechanics - WHY would this pattern cause problems?
""",
)


# =============================================================================
# Decision Agent - Choosing next action
# =============================================================================

decision_agent = Agent(
    "openai:gpt-5",
    output_type=AgentAction,
    system_prompt="""You are an expert in scheduling algorithms for heterogeneous distributed systems.
Your task is to find task graph families where the TARGET scheduler performs WORSE than the BASELINE.

CRITICAL: You are looking for cases where the makespan ratio > 1.0, meaning target scheduler is WORSE.

## Available Actions

1. **compare_algorithms**: Get LLM comparison of algorithms (do this ONCE at the start)
   - Understand how the algorithms differ in their approach

2. **read_source_code**: Read scheduler source code (useful for deep understanding)
   - Understand exactly how the algorithm makes decisions

3. **run_pisa**: Run PISA (simulated annealing) to find adversarial instances.
   - HIGHLY RECOMMENDED! This automatically searches for problematic cases.
   - Use EARLY (iteration 2-3) to discover what makes the target scheduler struggle.
   - PISA output shows the "energy" (makespan ratio) it found - higher is worse for target.
   - Analyze PISA results to understand what structure it found!

4. **test_code_hypothesis**: Write Python code that generates a FAMILY of task graph instances.
   This gives you COMPLETE FREEDOM to create any pathological case you can imagine!

   IMPORTANT: Use RANDOMIZATION! The get_instance() function will be called MANY times.
   Each call should return a DIFFERENT instance from the same pathological FAMILY.
   This tests whether the hypothesis holds broadly, not just for one specific case.

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

   **Example code (WITH RANDOMIZATION):**
   ```python
   def get_instance() -> Tuple[Network, TaskGraph]:
       # Create a diamond family where one branch is MUCH heavier than others
       # Randomize sizes/weights to create diverse instances with same property

       num_branches = random.randint(2, 5)  # Random number of branches

       dag = nx.DiGraph()
       dag.add_node("source", weight=random.uniform(0.1, 0.5))
       dag.add_node("sink", weight=random.uniform(0.1, 0.5))

       # One branch is 5-10x heavier than the others
       heavy_branch = random.randint(0, num_branches - 1)
       for i in range(num_branches):
           branch_name = f"branch_{i}"
           if i == heavy_branch:
               # Heavy branch
               dag.add_node(branch_name, weight=random.uniform(3.0, 8.0))
               dag.add_edge(branch_name, "sink", weight=random.uniform(2.0, 5.0))
           else:
               # Light branches
               dag.add_node(branch_name, weight=random.uniform(0.1, 0.5))
               dag.add_edge(branch_name, "sink", weight=random.uniform(0.1, 0.3))
           dag.add_edge("source", branch_name, weight=random.uniform(0.1, 0.3))

       task_graph = TaskGraph.from_nx(dag)

       # Heterogeneous network with random speeds
       num_processors = random.randint(2, 4)
       net = nx.Graph()
       for i in range(num_processors):
           speed = random.uniform(5.0, 10.0) if i == 0 else random.uniform(0.5, 2.0)
           net.add_node(f"p{i}", weight=speed)

       # Fully connected with self-loops
       for i in range(num_processors):
           for j in range(num_processors):
               bw = 1e9 if i == j else random.uniform(0.5, 2.0)
               net.add_edge(f"p{i}", f"p{j}", weight=bw)

       network = Network.from_nx(net)
       return network, task_graph
   ```

   **Ideas for pathological families:**
   - Asymmetric diamonds/forks where one branch dominates (vary which branch, how much)
   - Critical paths that aren't obvious from local decisions
   - High fan-in nodes where communication order matters
   - Bottleneck edges that force processor choices
   - Networks where the "obvious" fast processor is actually suboptimal

5. **submit_code_hypothesis**: Submit your best code hypothesis when confirmed >60%

## Recommended Strategy

Iteration 1: compare_algorithms (understand the algorithms)
Iteration 2: run_pisa (find adversarial patterns automatically)
Iteration 3: Analyze PISA results + read_source_code if needed
Iteration 4+: test_code_hypothesis (write code to create specific pathological families)
   - Use randomization to test if the pattern holds broadly
   - If ratio < 1.0, try making asymmetry MORE extreme
   - If confirmation rate is low, the pathological case may be too specific
Final: submit_code_hypothesis when >60% confirmation rate

## Writing Effective Code Hypotheses

1. **Understand the algorithm weakness first** - What decision does it make that could be wrong?
2. **Design a structure that exploits it** - Create a case where that decision IS wrong
3. **Make it extreme** - If asymmetry might help, make it 10x or 100x asymmetric
4. **Use randomization** - Vary sizes, weights, structure to create a FAMILY of instances
5. **Test and iterate** - If confirmation rate is low, make the pathological case MORE extreme

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
""",
)


# =============================================================================
# LLM-Based Algorithm Comparison
# =============================================================================

class AlgorithmComparison(BaseModel):
    """Structured comparison of two scheduling algorithms."""
    algorithm1_name: str
    algorithm2_name: str

    algorithm1_approach: str = Field(
        ..., description="How algorithm 1 works (key steps, prioritization, assignment)"
    )
    algorithm2_approach: str = Field(
        ..., description="How algorithm 2 works (key steps, prioritization, assignment)"
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
        ..., description="Task graph patterns that might cause algorithm 1 to underperform vs algorithm 2"
    )


comparison_agent = Agent(
    "openai:gpt-5",
    output_type=AlgorithmComparison,
    system_prompt="""You are an expert in scheduling algorithms for heterogeneous distributed systems.
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
""",
)


def compare_algorithms_with_llm(
    scheduler1: SchedulerName,
    scheduler2: SchedulerName,
    logger: Optional[ExperimentLogger] = None,
) -> str:
    """Use an LLM to compare two scheduling algorithms based on their source code."""
    source1 = get_scheduler_source_code(scheduler1)
    source2 = get_scheduler_source_code(scheduler2)

    prompt = f"""Analyze and compare these two scheduling algorithms:

## {scheduler1} Source Code:
```python
{source1[:8000]}  # Truncate if too long
```

## {scheduler2} Source Code:
```python
{source2[:8000]}  # Truncate if too long
```

Provide a detailed comparison focusing on their approaches, strengths, weaknesses,
and predict task graph patterns where {scheduler1} might underperform compared to {scheduler2}.
"""

    result = comparison_agent.run_sync(prompt)
    comparison = result.output

    # Log token usage if logger provided
    if logger:
        logger.log_token_usage("comparison", result.usage(), "gpt-5")

    # Format as readable string
    output = f"""
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
    return output


# =============================================================================
# Tool Functions (not agent tools, called directly)
# =============================================================================

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
    state: AgentState,
    scheduler: SchedulerName,
    base_scheduler: SchedulerName,
    max_iterations: int = 1000,
    num_tasks: int = 6,
    num_nodes: int = 4,
) -> str:
    """Run PISA to find adversarial instances."""
    from saga.utils.random_graphs import get_chain_dag, get_network

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
        data_dir=state.results_dir,
    )

    sa.execute(progress=False)
    best = sa.best_iteration

    tg = best.current_task_graph
    net = best.current_network

    num_tasks_result = len(tg.tasks)
    num_deps = len(tg.dependencies)
    avg_task_cost = np.mean([t.cost for t in tg.tasks])
    avg_data_size = np.mean([d.size for d in tg.dependencies]) if tg.dependencies else 0

    in_degrees = [tg.in_degree(t.name) for t in tg.tasks]
    out_degrees = [tg.out_degree(t.name) for t in tg.tasks]
    max_in = max(in_degrees)
    max_out = max(out_degrees)

    if max_in <= 1 and max_out <= 1:
        structure = "chain-like"
    elif max_in > 2 and max_out <= 1:
        structure = "fan-in"
    elif max_in <= 1 and max_out > 2:
        structure = "fan-out"
    else:
        structure = "complex DAG"

    # Build detailed task graph representation
    task_details = []
    for task in sorted(tg.tasks, key=lambda t: t.name):
        predecessors = [d.source for d in tg.dependencies if d.target == task.name]
        successors = [d.target for d in tg.dependencies if d.source == task.name]
        task_details.append(
            f"    {task.name}: cost={task.cost:.3f}, "
            f"in={predecessors if predecessors else '[]'}, "
            f"out={successors if successors else '[]'}"
        )

    # Build edge details (data dependencies)
    edge_details = []
    for dep in tg.dependencies:
        edge_details.append(f"    {dep.source} -> {dep.target}: size={dep.size:.3f}")

    # Build network details
    node_details = []
    for node in sorted(net.nodes, key=lambda n: n.name):
        node_details.append(f"    {node.name}: speed={node.speed:.3f}")

    # Build network link details (bandwidth between processors)
    link_details = []
    seen_links = set()
    for edge in net.edges:
        # Avoid duplicate links (undirected graph)
        link_key = tuple(sorted([edge.source, edge.target]))
        if link_key not in seen_links and edge.source != edge.target:
            seen_links.add(link_key)
            link_details.append(f"    {edge.source} <-> {edge.target}: bandwidth={edge.speed:.3f}")

    summary = f"""PISA Results ({scheduler} vs {base_scheduler}):
- Energy (makespan ratio): {best.current_energy:.4f}
- {scheduler} makespan: {best.current_makespan:.4f}, {base_scheduler} makespan: {best.current_base_makespan:.4f}
- Tasks: {num_tasks_result}, Dependencies: {num_deps}, Structure: {structure}
- Avg task cost: {avg_task_cost:.3f}, Avg data size: {avg_data_size:.3f}
- Max in-degree: {max_in}, Max out-degree: {max_out}

ADVERSARIAL TASK GRAPH FOUND:
  Tasks (node: cost, predecessors, successors):
{chr(10).join(task_details)}

  Data Dependencies (edge: data size):
{chr(10).join(edge_details) if edge_details else '    (none)'}

NETWORK CONFIGURATION:
  Processors (node: speed):
{chr(10).join(node_details)}

  Links (bandwidth between processors):
{chr(10).join(link_details) if link_details else '    (fully connected, uniform bandwidth)'}

ANALYSIS HINT: Look at the task costs, data sizes, and processor speeds.
What makes this instance adversarial? Consider:
- Are there tasks with very different costs?
- Are there edges with high data transfer costs?
- Is there processor heterogeneity that one algorithm handles poorly?
- What is the critical path and does it align with the algorithm's priorities?"""

    return summary


# =============================================================================
# Main Agentic Loop
# =============================================================================

def run_agentic_loop(
    target_scheduler: SchedulerName = "HEFT",
    baseline_scheduler: SchedulerName = "CPoP",
    max_iterations: int = 10,
    min_confidence_threshold: float = 0.6,
) -> Tuple[Optional[CodeHypothesis], Optional[HypothesisValidationResult]]:
    """
    Run the agentic hypothesis generation loop.

    The agent iteratively:
    1. Reviews what it has learned
    2. Decides what action to take next
    3. Executes the action
    4. Updates its state
    5. Repeats until confident or max iterations reached
    """
    results_dir = thisdir / "results"
    results_dir.mkdir(exist_ok=True)

    # Initialize experiment logger
    logger = ExperimentLogger(results_dir)
    logger.log_run_start(
        target_scheduler=target_scheduler,
        baseline_scheduler=baseline_scheduler,
        max_iterations=max_iterations,
        min_confidence_threshold=min_confidence_threshold,
    )

    state = AgentState(
        results_dir=results_dir,
        logger=logger,
        target_scheduler=target_scheduler,
        baseline_scheduler=baseline_scheduler,
        max_iterations=max_iterations,
    )

    print(f"Starting agentic loop: Find where {target_scheduler} underperforms vs {baseline_scheduler}")
    print(f"Max iterations: {max_iterations}")
    print(f"Logs will be saved to: {logger.log_dir}")
    print("=" * 60)

    # Track the previous reflection for continuity
    previous_reflection: Optional[ActionReflection] = None

    while state.iteration < max_iterations:
        state.iteration += 1
        print(f"\n{'='*60}")
        print(f"ITERATION {state.iteration}/{max_iterations}")
        print("=" * 60)

        # Get context summary for the agent
        context = state.get_context_summary()
        print(f"\nContext:\n{context}\n")

        # Log iteration start
        logger.log_iteration_start(state.iteration, context)

        # =================================================================
        # PHASE 1: STRATEGIC PLANNING
        # =================================================================
        print(f"\n{'-'*40}")
        print("PHASE 1: STRATEGIC PLANNING")
        print(f"{'-'*40}")

        planning_prompt = f"""Create a strategic plan for this iteration.

CURRENT STATE:
{context}

{f'''PREVIOUS REFLECTION:
Key findings: {previous_reflection.key_findings}
Hypothesis update: {previous_reflection.hypothesis_update}
Next question: {previous_reflection.next_question}
Confidence: {previous_reflection.confidence_assessment}
''' if previous_reflection else 'This is the first iteration - no previous reflection.'}

Based on where we are, create a strategic plan for what we should do next.
"""

        plan_result = planning_agent.run_sync(planning_prompt)
        plan = plan_result.output
        logger.log_token_usage("planning", plan_result.usage(), "gpt-5")

        print(f"\n📋 STRATEGIC PLAN:")
        print(f"   Understanding: {plan.current_understanding[:150]}...")
        print(f"   Key unknowns: {', '.join(plan.key_unknowns[:3])}")
        if plan.working_hypothesis:
            print(f"   Working hypothesis: {plan.working_hypothesis[:150]}...")
        print(f"   Next steps: {'; '.join(plan.next_steps[:2])}")
        print(f"   Immediate action: {plan.immediate_action[:100]}...")

        logger.log_strategic_plan(plan)

        # =================================================================
        # PHASE 2: ACTION DECISION
        # =================================================================
        print(f"\n{'-'*40}")
        print("PHASE 2: ACTION DECISION")
        print(f"{'-'*40}")

        decision_prompt = f"""Based on the strategic plan, decide your next action.

CURRENT STATE:
{context}

STRATEGIC PLAN:
- Understanding: {plan.current_understanding}
- Key unknowns: {plan.key_unknowns}
- Working hypothesis: {plan.working_hypothesis}
- Next steps: {plan.next_steps}
- Immediate action: {plan.immediate_action}
- Success criteria: {plan.success_criteria}

{f'''PREVIOUS REFLECTION:
Key findings: {previous_reflection.key_findings}
Hypothesis update: {previous_reflection.hypothesis_update}
''' if previous_reflection else ''}

Choose the action that best aligns with the strategic plan.
When you have a validated hypothesis with >{min_confidence_threshold:.0%} confirmation, submit it.

IMPORTANT: When creating TaskGraphFamilyHypothesis with IntRange fields (like num_tasks, num_nodes),
use "min_val" and "max_val" as the field names, NOT "min" and "max".
"""

        result = decision_agent.run_sync(decision_prompt)
        action = result.output
        logger.log_token_usage("decision", result.usage(), "gpt-5")

        print(f"\n🎯 ACTION: {action.action}")
        print(f"   Reasoning: {action.reasoning}")

        # Log the agent's decision
        action_params = {}
        if action.scheduler_to_read:
            action_params["scheduler_to_read"] = action.scheduler_to_read
        if action.code_hypothesis:
            action_params["code_hypothesis"] = {
                "name": action.code_hypothesis.name,
                "hypothesis_id": action.code_hypothesis.hypothesis_id,
            }

        logger.log_agent_decision(action.action, action.reasoning, action_params)

        # Execute the action
        action_result = ""
        result_data: Dict[str, Any] = {}

        if action.action == "compare_algorithms":
            if state.algorithm_comparison is None:
                print(f"\nComparing {target_scheduler} vs {baseline_scheduler} using LLM...")
                comparison = compare_algorithms_with_llm(target_scheduler, baseline_scheduler, logger)
                state.algorithm_comparison = comparison
                action_result = comparison
                result_data = {"comparison_length": len(comparison)}
                print(comparison)
            else:
                action_result = "Algorithm comparison already done."
                print(action_result)

        elif action.action == "read_source_code":
            scheduler = action.scheduler_to_read or target_scheduler
            if scheduler not in state.source_code_read:
                print(f"\nReading source code for {scheduler}...")
                source = get_scheduler_source_code(scheduler)
                state.source_code_read.append(scheduler)
                action_result = f"Read {len(source)} characters of source code for {scheduler}"
                result_data = {"scheduler": scheduler, "source_length": len(source)}
                print(action_result)
            else:
                action_result = f"Already read source code for {scheduler}"
                print(action_result)

        elif action.action == "run_pisa":
            print(f"\nRunning PISA experiment...")
            pisa_result = run_pisa_experiment(
                state,
                target_scheduler,
                baseline_scheduler,
            )
            state.pisa_results.append(pisa_result)
            action_result = pisa_result
            # Extract key metrics from PISA result
            result_data = {"pisa_experiment_count": len(state.pisa_results)}
            print(pisa_result)

        elif action.action == "test_code_hypothesis":
            if action.code_hypothesis:
                print(f"\nTesting code-based hypothesis: {action.code_hypothesis.name}")
                print(f"Reasoning: {action.code_hypothesis.reasoning[:150]}...")
                print(f"Code preview: {action.code_hypothesis.code[:200]}...")

                # First, validate the code runs at all
                network, task_graph, error = execute_code_hypothesis(action.code_hypothesis)
                if error:
                    action_result = f"Code hypothesis FAILED to execute: {error}"
                    result_data = {
                        "hypothesis_id": action.code_hypothesis.hypothesis_id,
                        "name": action.code_hypothesis.name,
                        "error": error,
                        "success": False,
                    }
                    print(f"\n❌ Code execution error: {error}")
                else:
                    # Code works, now validate across multiple instances
                    validation = validate_code_hypothesis(action.code_hypothesis, num_instances=50)

                    action_result = f"Code hypothesis '{action.code_hypothesis.name}': confirmation_rate={validation.confirmation_rate:.1%}, avg_ratio={validation.avg_makespan_ratio:.4f}"
                    result_data = {
                        "hypothesis_id": action.code_hypothesis.hypothesis_id,
                        "name": action.code_hypothesis.name,
                        "confirmation_rate": validation.confirmation_rate,
                        "avg_makespan_ratio": validation.avg_makespan_ratio,
                        "max_makespan_ratio": validation.max_makespan_ratio,
                        "min_makespan_ratio": validation.min_makespan_ratio,
                        "is_validated": validation.is_validated,
                        "success": True,
                    }

                    print(f"\nValidation Results:")
                    print(f"  Confirmation rate: {validation.confirmation_rate:.1%}")
                    print(f"  Avg makespan ratio: {validation.avg_makespan_ratio:.4f}")
                    print(f"  Max makespan ratio: {validation.max_makespan_ratio:.4f}")
                    print(f"  Min makespan ratio: {validation.min_makespan_ratio:.4f}")
                    print(f"  Validated: {validation.is_validated}")

                    # Track tested hypothesis
                    state.code_hypotheses_tested.append((action.code_hypothesis, validation))

                    # Track as best if better
                    is_new_best = state.best_validation is None or validation.confirmation_rate > state.best_validation.confirmation_rate
                    if is_new_best:
                        state.best_hypothesis = action.code_hypothesis
                        state.best_validation = validation
                        print("  -> New best hypothesis!")
            else:
                action_result = "No code_hypothesis provided for test_code_hypothesis action"
                result_data = {"error": "missing code_hypothesis"}

        elif action.action == "submit_code_hypothesis":
            if action.code_hypothesis:
                print(f"\nSubmitting code-based hypothesis: {action.code_hypothesis.name}")

                # Validate with more instances
                validation = validate_code_hypothesis(action.code_hypothesis, num_instances=100)

                action_result = f"Submitted code hypothesis '{action.code_hypothesis.name}': confirmation_rate={validation.confirmation_rate:.1%}"
                result_data = {
                    "hypothesis_id": action.code_hypothesis.hypothesis_id,
                    "name": action.code_hypothesis.name,
                    "confirmation_rate": validation.confirmation_rate,
                    "avg_makespan_ratio": validation.avg_makespan_ratio,
                    "is_validated": validation.is_validated,
                    "accepted": validation.confirmation_rate >= min_confidence_threshold,
                }

                print(f"\nFinal Validation:")
                print(f"  Confirmation rate: {validation.confirmation_rate:.1%}")
                print(f"  Avg makespan ratio: {validation.avg_makespan_ratio:.4f}")

                if validation.confirmation_rate >= min_confidence_threshold:
                    print(f"\n✅ Code hypothesis ACCEPTED with {validation.confirmation_rate:.1%} confirmation!")
                    state.best_hypothesis = action.code_hypothesis
                    state.best_validation = validation
                    logger.log_action_result(action_result, result_data)
                    break
                else:
                    print(f"\n❌ Hypothesis not confident enough ({validation.confirmation_rate:.1%} < {min_confidence_threshold:.0%})")
                    state.code_hypotheses_tested.append((action.code_hypothesis, validation))
                    is_new_best = state.best_validation is None or validation.confirmation_rate > state.best_validation.confirmation_rate
                    if is_new_best:
                        state.best_hypothesis = action.code_hypothesis
                        state.best_validation = validation
            else:
                action_result = "No code_hypothesis provided for submit_code_hypothesis action"
                result_data = {"error": "missing code_hypothesis"}

        # Log the action result
        logger.log_action_result(action_result, result_data)

        # =================================================================
        # PHASE 3: REFLECTION
        # =================================================================
        print(f"\n{'-'*40}")
        print("PHASE 3: REFLECTION")
        print(f"{'-'*40}")

        reflection_prompt = f"""Reflect on the results of the action just taken.

ACTION TAKEN: {action.action}
REASONING: {action.reasoning}

RESULT:
{action_result[:2000]}

{f'''RESULT DATA:
{json.dumps(result_data, indent=2, default=str)}
''' if result_data else ''}

STRATEGIC PLAN WE WERE FOLLOWING:
- Working hypothesis: {plan.working_hypothesis}
- Success criteria: {plan.success_criteria}

What did we learn? How does this change our hypothesis? What should we investigate next?
"""

        reflection_result = reflection_agent.run_sync(reflection_prompt)
        reflection = reflection_result.output
        logger.log_token_usage("reflection", reflection_result.usage(), "gpt-5")
        previous_reflection = reflection

        print(f"\n💭 REFLECTION:")
        print(f"   Key findings: {'; '.join(reflection.key_findings[:3])}")
        if reflection.surprises:
            print(f"   Surprises: {'; '.join(reflection.surprises[:2])}")
        print(f"   Hypothesis update: {reflection.hypothesis_update[:150]}...")
        print(f"   Next question: {reflection.next_question}")
        print(f"   Confidence: {reflection.confidence_assessment[:100]}...")

        logger.log_reflection(reflection)
        logger.finalize_iteration()  # Save iteration after reflection is complete

    # Log run completion
    logger.log_run_complete(state.best_hypothesis, state.best_validation, state.iteration)
    logger.save_visualization_data()

    # Final output
    print("\n" + "=" * 60)
    print("AGENTIC LOOP COMPLETE")
    print("=" * 60)

    if state.best_hypothesis:
        print(f"\nBest Hypothesis Found:")
        print(f"  Name: {state.best_hypothesis.name}")
        print(f"  Reasoning: {state.best_hypothesis.reasoning}")
        print(f"\n  Code:\n{state.best_hypothesis.code}")

        # Save to file
        output_path = results_dir / "best_hypothesis.json"
        output_path.write_text(state.best_hypothesis.model_dump_json(indent=2))
        print(f"\nSaved to: {output_path}")

        # Also save just the code for easy reuse
        code_path = results_dir / "best_hypothesis_code.py"
        code_path.write_text(f'''"""
{state.best_hypothesis.name}

{state.best_hypothesis.reasoning}

Generated by the agentic hypothesis system.
"""
from typing import Tuple
import networkx as nx
import numpy as np
import random
from itertools import product

from saga import Network, TaskGraph

{state.best_hypothesis.code}


if __name__ == "__main__":
    # Test the hypothesis
    network, task_graph = get_instance()
    print(f"Network nodes: {{list(network.graph.nodes())}}")
    print(f"Task graph nodes: {{list(task_graph.graph.nodes())}}")
''')
        print(f"Code saved to: {code_path}")

        if state.best_validation:
            print(f"\nValidation:")
            print(f"  Confirmation rate: {state.best_validation.confirmation_rate:.1%}")
            print(f"  Avg ratio: {state.best_validation.avg_makespan_ratio:.4f}")
            print(f"  Validated: {state.best_validation.is_validated}")
    else:
        print("\nNo hypothesis found.")

    # Print token usage summary
    token_summary = logger.get_token_summary()
    print(f"\n{'='*60}")
    print("TOKEN USAGE & COST")
    print("=" * 60)
    print(f"  Total input tokens:  {token_summary['total_input_tokens']:,}")
    print(f"  Total output tokens: {token_summary['total_output_tokens']:,}")
    print(f"  Total tokens:        {token_summary['total_tokens']:,}")
    print(f"  Estimated cost:      ${token_summary['estimated_cost_usd']:.4f} USD")
    print("\n  By Agent:")
    for agent_name, usage in token_summary['by_agent'].items():
        print(f"    {agent_name}: {usage['calls']} calls, {usage['input_tokens']:,} in, {usage['output_tokens']:,} out")

    print(f"\nLogs saved to: {logger.log_dir}")

    return state.best_hypothesis, state.best_validation


def main():
    """Run the agentic hypothesis generation system."""
    hypothesis, validation = run_agentic_loop(
        target_scheduler="HEFT",
        baseline_scheduler="CPoP",
        max_iterations=10,
        min_confidence_threshold=0.6,
    )

    return hypothesis


if __name__ == "__main__":
    main()
