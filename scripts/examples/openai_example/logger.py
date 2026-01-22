"""
Experiment logging and visualization.

This module provides logging infrastructure for tracking agent actions,
token usage, and generating visualizations of experiment runs.
"""

import json
import pathlib
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from models import (
    ActionReflection,
    CodeHypothesis,
    HypothesisValidationResult,
    StrategicPlan,
)

# Default pricing per 1M tokens (USD)
DEFAULT_PRICING = {
    # GPT-4o series
    "gpt-4o": {"input": 2.50, "cached_input": 1.25, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "cached_input": 0.075, "output": 0.60},
    # GPT-5 series
    "gpt-5.2": {"input": 1.75, "cached_input": 0.175, "output": 14.00},
    "gpt-5": {"input": 1.75, "cached_input": 0.175, "output": 14.00},
    "gpt-5.2-pro": {"input": 21.00, "cached_input": 0.0, "output": 168.00},
    "gpt-5-mini": {"input": 0.25, "cached_input": 0.025, "output": 2.00},
    # Default fallback
    "default": {"input": 2.50, "cached_input": 1.25, "output": 10.00},
}


class ExperimentLogger:
    """Logs all agent actions, reasoning, and results for analysis and visualization."""

    def __init__(
        self,
        results_dir: pathlib.Path,
        run_id: Optional[str] = None,
        pricing: Optional[Dict[str, Dict[str, float]]] = None,
    ):
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

        self.pricing = pricing or DEFAULT_PRICING

        self.run_metadata: Dict[str, Any] = {
            "run_id": self.run_id,
            "start_time": datetime.now().isoformat(),
            "status": "running",
        }

    # =========================================================================
    # Run Lifecycle
    # =========================================================================

    def log_run_start(
        self,
        target_scheduler: str,
        baseline_scheduler: str,
        max_iterations: int,
        min_confidence_threshold: float,
    ):
        """Log the start of an experiment run."""
        self.run_metadata.update(
            {
                "target_scheduler": target_scheduler,
                "baseline_scheduler": baseline_scheduler,
                "max_iterations": max_iterations,
                "min_confidence_threshold": min_confidence_threshold,
            }
        )
        self._save_run_metadata()
        self._log_event(
            "run_start",
            {
                "target_scheduler": target_scheduler,
                "baseline_scheduler": baseline_scheduler,
                "max_iterations": max_iterations,
            },
        )

    def log_run_complete(
        self,
        best_hypothesis: Optional[CodeHypothesis],
        best_validation: Optional[HypothesisValidationResult],
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
                "reasoning": (
                    best_hypothesis.reasoning[:200] + "..."
                    if len(best_hypothesis.reasoning) > 200
                    else best_hypothesis.reasoning
                ),
                "confirmation_rate": best_validation.confirmation_rate,
                "avg_makespan_ratio": best_validation.avg_makespan_ratio,
                "is_validated": best_validation.is_validated,
            }

        self.run_metadata["token_usage"] = self.get_token_summary()
        self._save_run_metadata()
        self._save_all_events()
        self._generate_summary_report()

    # =========================================================================
    # Iteration Logging
    # =========================================================================

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

    def log_strategic_plan(self, plan: StrategicPlan):
        """Log the strategic plan for this iteration."""
        self.current_iteration["strategic_plan"] = {
            "current_understanding": plan.current_understanding,
            "key_unknowns": plan.key_unknowns,
            "working_hypothesis": plan.working_hypothesis,
            "next_steps": plan.next_steps,
            "immediate_action": plan.immediate_action,
            "success_criteria": plan.success_criteria,
        }
        self._log_event(
            "strategic_plan",
            {
                "current_understanding": plan.current_understanding[:200],
                "key_unknowns": plan.key_unknowns,
                "working_hypothesis": plan.working_hypothesis,
                "next_steps": plan.next_steps,
            },
        )

    def log_agent_decision(
        self, action: str, reasoning: str, action_params: Optional[Dict[str, Any]] = None
    ):
        """Log what action the agent decided to take and why."""
        self.current_iteration["action"] = action
        self.current_iteration["reasoning"] = reasoning
        self.current_iteration["action_params"] = action_params or {}
        self._log_event(
            "agent_decision",
            {"action": action, "reasoning": reasoning, "params": action_params},
        )

    def log_action_result(
        self, result: str, result_data: Optional[Dict[str, Any]] = None
    ):
        """Log the result of an action."""
        self.current_iteration["result"] = result
        self.current_iteration["result_data"] = result_data or {}
        start_time = self.current_iteration.get("start_time", time.time())
        self.current_iteration["duration_seconds"] = time.time() - float(start_time)
        self._log_event(
            "action_result",
            {"result_preview": result[:500] if result else None, "result_data": result_data},
        )

    def log_reflection(self, reflection: ActionReflection):
        """Log the reflection on an action's results."""
        self.current_iteration["reflection"] = {
            "action_taken": reflection.action_taken,
            "key_findings": reflection.key_findings,
            "surprises": reflection.surprises,
            "hypothesis_update": reflection.hypothesis_update,
            "next_question": reflection.next_question,
            "confidence_assessment": reflection.confidence_assessment,
        }
        self._log_event(
            "action_reflection",
            {
                "key_findings": reflection.key_findings,
                "surprises": reflection.surprises,
                "hypothesis_update": reflection.hypothesis_update,
                "confidence_assessment": reflection.confidence_assessment,
            },
        )

    def finalize_iteration(self):
        """Finalize and save the current iteration after reflection is logged."""
        self.iteration_logs.append(self.current_iteration.copy())
        self._save_iteration_log(self.current_iteration)

    # =========================================================================
    # Token Usage
    # =========================================================================

    def log_token_usage(self, agent_name: str, usage: Any, model: str = "default"):
        """Log token usage from an agent call."""
        if usage is None:
            return

        input_tokens = getattr(usage, "input_tokens", 0) or 0
        output_tokens = getattr(usage, "output_tokens", 0) or 0
        cached_tokens = getattr(usage, "cached_tokens", 0) or 0

        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_cached_input_tokens += cached_tokens

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

        if "model" not in self.token_usage_by_agent[agent_name]:
            self.token_usage_by_agent[agent_name]["model"] = model

        self._log_event(
            "token_usage",
            {
                "agent": agent_name,
                "model": model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cached_tokens": cached_tokens,
            },
        )

        if "token_usage" not in self.current_iteration:
            self.current_iteration["token_usage"] = []
        self.current_iteration["token_usage"].append(
            {
                "agent": agent_name,
                "model": model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cached_tokens": cached_tokens,
            }
        )

    def get_total_cost(self) -> float:
        """Calculate total cost based on token usage and pricing."""
        total_cost = 0.0
        for usage in self.token_usage_by_agent.values():
            model = usage.get("model", "default")
            model_key = model.lower().replace("openai:", "").replace("_", "-")
            pricing = self.pricing.get(model_key, self.pricing["default"])

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

    # =========================================================================
    # Internal Methods
    # =========================================================================

    def _log_event(self, event_type: str, data: Dict[str, Any]):
        """Internal method to log an event."""
        self.events.append(
            {
                "timestamp": datetime.now().isoformat(),
                "elapsed_seconds": time.time() - self.run_start_time,
                "event_type": event_type,
                "data": data,
            }
        )

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

    # =========================================================================
    # Report Generation
    # =========================================================================

    def _generate_summary_report(self):
        """Generate a human-readable summary report."""
        lines = [
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
            lines.append(
                f"\n[Iteration {it['iteration']}] Action: {it['action']} ({it.get('duration_seconds', 0):.1f}s)"
            )

            plan = it.get("strategic_plan", {})
            if plan:
                lines.append(f"  Plan: {plan.get('immediate_action', 'N/A')[:80]}...")
                if plan.get("working_hypothesis"):
                    lines.append(f"     Working hypothesis: {plan['working_hypothesis'][:80]}...")

            if it.get("reasoning"):
                lines.append(f"  Reasoning: {it['reasoning'][:100]}...")

            if it["action"] in ("test_code_hypothesis", "submit_code_hypothesis"):
                result_data = it.get("result_data", {})
                if result_data:
                    lines.append(
                        f"     Result: confirmation_rate={result_data.get('confirmation_rate', 'N/A')}, "
                        f"avg_ratio={result_data.get('avg_makespan_ratio', 'N/A')}"
                    )

            reflection = it.get("reflection", {})
            if reflection:
                findings = reflection.get("key_findings", [])[:2]
                lines.append(f"  Key findings: {'; '.join(findings)[:80]}...")
                lines.append(
                    f"     Hypothesis update: {reflection.get('hypothesis_update', 'N/A')[:80]}..."
                )

        if self.run_metadata.get("best_hypothesis"):
            bh = self.run_metadata["best_hypothesis"]
            lines.extend(
                [
                    "",
                    "-" * 70,
                    "BEST HYPOTHESIS FOUND",
                    "-" * 70,
                    f"  Name: {bh.get('name')}",
                    f"  Reasoning: {bh.get('reasoning', 'N/A')[:100]}...",
                    f"  Confirmation rate: {bh.get('confirmation_rate', 0):.1%}",
                    f"  Avg makespan ratio: {bh.get('avg_makespan_ratio', 0):.4f}",
                    f"  Validated: {bh.get('is_validated')}",
                ]
            )

        token_summary = self.get_token_summary()
        lines.extend(
            [
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
            ]
        )
        for agent_name, usage in token_summary["by_agent"].items():
            lines.append(
                f"    {agent_name}: {usage['calls']} calls, "
                f"{usage['input_tokens']:,} in, {usage['output_tokens']:,} out"
            )

        lines.extend(["", "=" * 70])

        report_path = self.log_dir / "summary_report.txt"
        with open(report_path, "w") as f:
            f.write("\n".join(lines))
        print("\n".join(lines))

    # =========================================================================
    # Visualization
    # =========================================================================

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
                    "result_preview": (
                        it.get("result", "")[:200] if it.get("result") else None
                    ),
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
        self._generate_html_visualization()

    def _generate_html_visualization(self):
        """Generate an interactive HTML visualization of the experiment."""
        token_summary = self.get_token_summary()
        bh = self.run_metadata.get("best_hypothesis")

        # Build iteration HTML
        iterations_html = "\n".join(
            self._render_iteration_html(it) for it in self.iteration_logs
        )

        # Build token table rows
        agent_rows = "\n".join(
            f"""<tr>
                <td style="padding: 8px; border-bottom: 1px solid #0f3460;">{name}</td>
                <td style="padding: 8px; border-bottom: 1px solid #0f3460; text-align: right;">{usage['calls']}</td>
                <td style="padding: 8px; border-bottom: 1px solid #0f3460; text-align: right;">{usage['input_tokens']:,}</td>
                <td style="padding: 8px; border-bottom: 1px solid #0f3460; text-align: right;">{usage['output_tokens']:,}</td>
            </tr>"""
            for name, usage in token_summary["by_agent"].items()
        )

        # Best hypothesis card
        hypothesis_html = ""
        if bh:
            hypothesis_html = f"""
            <div class="hypothesis-card">
                <h3>Best Hypothesis Found: {bh.get('name', 'N/A')}</h3>
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
                        <div class="metric-value">{'Y' if bh.get('is_validated') else 'N'}</div>
                        <div class="metric-label">Validated</div>
                    </div>
                </div>
            </div>"""

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Experiment Run: {self.run_id}</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                background: #1a1a2e; color: #eee; padding: 20px; line-height: 1.6; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        h1 {{ color: #00d9ff; margin-bottom: 20px; }}
        h2 {{ color: #ff6b6b; margin: 20px 0 10px; }}
        .metadata {{ background: #16213e; padding: 20px; border-radius: 10px; margin-bottom: 20px; }}
        .metadata-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; }}
        .metric {{ background: #0f3460; padding: 15px; border-radius: 8px; text-align: center; }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #00d9ff; }}
        .metric-label {{ color: #aaa; font-size: 12px; }}
        .timeline {{ position: relative; padding-left: 30px; }}
        .timeline::before {{ content: ''; position: absolute; left: 10px; top: 0; bottom: 0;
                            width: 2px; background: #0f3460; }}
        .iteration {{ background: #16213e; margin-bottom: 15px; padding: 20px; border-radius: 10px; position: relative; }}
        .iteration::before {{ content: ''; position: absolute; left: -24px; top: 25px;
                             width: 12px; height: 12px; border-radius: 50%; background: #00d9ff; }}
        .iteration-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; }}
        .iteration-number {{ font-size: 18px; font-weight: bold; color: #00d9ff; }}
        .action-badge {{ padding: 5px 12px; border-radius: 20px; font-size: 12px; font-weight: bold; background: #3498db; }}
        .duration {{ color: #888; font-size: 12px; }}
        .result {{ background: #0a0a1a; padding: 15px; border-radius: 8px; margin-top: 10px;
                  font-family: monospace; font-size: 13px; white-space: pre-wrap; max-height: 300px; overflow-y: auto; }}
        .hypothesis-card {{ background: linear-gradient(135deg, #1abc9c 0%, #16a085 100%);
                           padding: 20px; border-radius: 10px; margin-top: 20px; }}
        .hypothesis-stats {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; }}
        .stat {{ background: rgba(0,0,0,0.2); padding: 10px; border-radius: 5px; text-align: center; }}
        .phase-section {{ margin: 10px 0; padding: 15px; border-radius: 8px; border-left: 4px solid; }}
        .plan-section {{ background: rgba(52, 152, 219, 0.1); border-color: #3498db; }}
        .reflection-section {{ background: rgba(155, 89, 182, 0.1); border-color: #9b59b6; }}
        .phase-header {{ font-weight: bold; color: #fff; margin-bottom: 10px; }}
        .phase-content {{ font-size: 13px; color: #bbb; }}
        .phase-content p {{ margin: 5px 0; }}
        .collapsible {{ cursor: pointer; }}
        .collapsible::after {{ content: ' [+]'; font-size: 10px; }}
        .content {{ display: block; }}
        .content.hidden {{ display: none; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Experiment Run: {self.run_id}</h1>
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
        <div class="metadata" style="margin-bottom: 20px;">
            <h2 style="color: #f39c12; margin-bottom: 15px;">Token Usage & Cost</h2>
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
                    <tbody>{agent_rows}</tbody>
                </table>
            </div>
        </div>
        {hypothesis_html}
        <h2>Iteration Timeline</h2>
        <div class="timeline">{iterations_html}</div>
    </div>
    <script>
        document.querySelectorAll('.collapsible').forEach(el => {{
            el.addEventListener('click', () => {{
                el.nextElementSibling.classList.toggle('hidden');
            }});
        }});
    </script>
</body>
</html>"""

        path = self.log_dir / "visualization.html"
        with open(path, "w") as f:
            f.write(html)

    def _render_iteration_html(self, iteration: Dict[str, Any]) -> str:
        """Generate HTML for a single iteration."""
        action = iteration.get("action", "unknown")
        reasoning = iteration.get("reasoning", "")
        result = iteration.get("result", "") or ""
        duration = iteration.get("duration_seconds", 0)
        plan = iteration.get("strategic_plan", {})
        reflection = iteration.get("reflection", {})
        action_params = iteration.get("action_params", {})

        result_preview = result[:1000] + ("..." if len(result) > 1000 else "")

        plan_html = ""
        if plan:
            plan_html = f"""
            <div class="phase-section plan-section">
                <div class="phase-header">Strategic Plan</div>
                <div class="phase-content">
                    <p><strong>Understanding:</strong> {plan.get('current_understanding', 'N/A')[:300]}...</p>
                    <p><strong>Key Unknowns:</strong> {', '.join(plan.get('key_unknowns', [])[:3])}</p>
                    <p><strong>Working Hypothesis:</strong> {plan.get('working_hypothesis', 'None yet')}</p>
                    <p><strong>Next Steps:</strong> {'; '.join(plan.get('next_steps', [])[:2])}</p>
                </div>
            </div>"""

        reflection_html = ""
        if reflection:
            surprises = reflection.get("surprises", [])
            surprises_html = (
                f"<p><strong>Surprises:</strong> {'; '.join(surprises)}</p>"
                if surprises
                else ""
            )
            reflection_html = f"""
            <div class="phase-section reflection-section">
                <div class="phase-header">Reflection</div>
                <div class="phase-content">
                    <p><strong>Key Findings:</strong> {'; '.join(reflection.get('key_findings', [])[:3])}</p>
                    {surprises_html}
                    <p><strong>Hypothesis Update:</strong> {reflection.get('hypothesis_update', 'N/A')[:200]}...</p>
                    <p><strong>Next Question:</strong> {reflection.get('next_question', 'N/A')}</p>
                </div>
            </div>"""

        # Code hypothesis section
        code_html = ""
        if action_params.get("code_hypothesis"):
            hyp = action_params["code_hypothesis"]
            code = hyp.get("code", "No code captured")
            # Escape HTML in code
            code_escaped = (
                code.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
            )
            code_html = f"""
            <div class="phase-section" style="background: rgba(46, 204, 113, 0.1); border-color: #2ecc71;">
                <div class="phase-header">Code Hypothesis: {hyp.get('name', 'unnamed')}</div>
                <div class="phase-content">
                    <p><strong>Reasoning:</strong> {hyp.get('reasoning', 'N/A')}</p>
                    <p><strong>Worse Scheduler:</strong> {hyp.get('worse_scheduler', 'N/A')} vs <strong>Better:</strong> {hyp.get('better_scheduler', 'N/A')}</p>
                    <p><strong>Confidence:</strong> {hyp.get('confidence', 0):.0%}</p>
                    <div class="phase-header collapsible">View Code</div>
                    <pre class="content" style="background: #0a0a1a; padding: 10px; border-radius: 5px; overflow-x: auto; font-size: 12px;">{code_escaped}</pre>
                </div>
            </div>"""

        return f"""
        <div class="iteration">
            <div class="iteration-header">
                <span class="iteration-number">Iteration {iteration.get('iteration', '?')}</span>
                <span class="action-badge">{action.replace('_', ' ').upper()}</span>
                <span class="duration">{duration:.1f}s</span>
            </div>
            {plan_html}
            <div>
                <div class="phase-header collapsible">Action: {action.replace('_', ' ').title()}</div>
                <div class="content">
                    <p><strong>Reasoning:</strong> {reasoning}</p>
                </div>
                {code_html}
                <div class="phase-header collapsible">Result</div>
                <div class="content result">{result_preview}</div>
            </div>
            {reflection_html}
        </div>"""
