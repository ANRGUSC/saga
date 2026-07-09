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

import json
import logging
import pathlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

from saga.pisa.simulated_annealing import SchedulerName

from agents import (
    create_comparison_agent,
    create_decision_agent,
    create_planning_agent,
    create_reflection_agent,
)
from hypothesis import execute_code_hypothesis, validate_code_hypothesis
from logger import ExperimentLogger
from models import (
    ActionReflection,
    CodeHypothesis,
    HypothesisValidationResult,
)
from tools import (
    compare_algorithms_with_llm,
    get_scheduler_source_code,
    run_pisa_experiment,
    test_single_instance,
)

# Suppress the task graph warnings
logging.getLogger().setLevel(logging.ERROR)

load_dotenv()

thisdir = pathlib.Path(__file__).parent.resolve()


# =============================================================================
# MODEL CONFIGURATION
# =============================================================================
# Available models (examples):
#   OpenAI:    "openai:gpt-4o", "openai:gpt-4o-mini", "openai:gpt-5", "openai:o1", "openai:o3-mini"
#   Anthropic: "anthropic:claude-sonnet-4-20250514", "anthropic:claude-3-5-haiku-20241022"
#   Google:    "google-gla:gemini-2.0-flash", "google-gla:gemini-1.5-pro"
#   Groq:      "groq:llama-3.3-70b-versatile"
#
# You can use different models for different agents based on the task complexity.
# The decision agent benefits from a more capable model since it writes code.

MODEL_PLANNING = "openai:gpt-4o-mini"      # Strategic planning (moderate complexity)
MODEL_DECISION = "openai:gpt-4o-mini"      # Action decisions and code generation (high complexity)
MODEL_REFLECTION = "openai:gpt-4o-mini"    # Reflection and analysis (moderate complexity)
MODEL_COMPARISON = "openai:gpt-4o-mini"    # Algorithm comparison (moderate complexity)


# =============================================================================
# Agent State - Tracks conversation history and findings
# =============================================================================


@dataclass
class AgentState:
    """Tracks the agent's state across iterations."""

    results_dir: pathlib.Path
    logger: ExperimentLogger
    comparison_agent: Any  # The comparison agent instance
    target_scheduler: SchedulerName = "HEFT"
    baseline_scheduler: SchedulerName = "CPoP"

    iteration: int = 0
    max_iterations: int = 10

    # Track what the agent has learned
    algorithm_comparison: Optional[str] = None
    source_code: Dict[SchedulerName, str] = field(default_factory=dict)
    pisa_results: List[str] = field(default_factory=list)
    code_hypotheses_tested: List[Tuple[CodeHypothesis, HypothesisValidationResult]] = field(
        default_factory=list
    )

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
            summary_parts.append(f"  {self.algorithm_comparison}")

        if self.source_code:
            summary_parts.append(f"\nSource code read: {', '.join(self.source_code.keys())}")
            for scheduler, source in self.source_code.items():
                summary_parts.append(f"\n--- {scheduler} SOURCE CODE ---")
                summary_parts.append(source)

        if self.pisa_results:
            summary_parts.append(f"\nPISA experiments run: {len(self.pisa_results)}")
            if len(self.pisa_results) >= 2:
                summary_parts.append(
                    "  WARNING: You have run PISA multiple times. "
                    "STOP running PISA and START writing code hypotheses based on the patterns found!"
                )
            for i, result in enumerate(self.pisa_results, 1):
                summary_parts.append(f"  Recent PISA {i}: {result}")

        if self.code_hypotheses_tested:
            summary_parts.append(f"\nCode hypotheses tested: {len(self.code_hypotheses_tested)}")
            for hyp, val in self.code_hypotheses_tested:
                summary_parts.append(
                    f"  - {hyp.name}: "
                    f"confirmation={val.confirmation_rate:.1%} "
                    f"(95% CI: {val.confirmation_rate_ci_low:.1%}-{val.confirmation_rate_ci_high:.1%}), "
                    f"avg_ratio={val.avg_makespan_ratio:.3f}"
                )

        if self.best_hypothesis:
            summary_parts.append("\nBest hypothesis so far:")
            summary_parts.append(f"  Name: {self.best_hypothesis.name}")
            summary_parts.append(f"  Reasoning: {self.best_hypothesis.reasoning}")
            summary_parts.append(f"  Code:\n{self.best_hypothesis.code}")
            if self.best_validation:
                summary_parts.append(
                    f"  Confirmation rate: {self.best_validation.confirmation_rate:.1%} "
                    f"(95% CI: {self.best_validation.confirmation_rate_ci_low:.1%}-"
                    f"{self.best_validation.confirmation_rate_ci_high:.1%})"
                )
                summary_parts.append(
                    f"  Avg makespan ratio: {self.best_validation.avg_makespan_ratio:.4f}"
                )

        return "\n".join(summary_parts)


# =============================================================================
# Action Handlers
# =============================================================================


def handle_compare_algorithms(
    state: AgentState,
) -> Tuple[str, Dict[str, Any]]:
    """Handle the compare_algorithms action."""
    if state.algorithm_comparison is not None:
        return "Algorithm comparison already done.", {}

    print(f"\nComparing {state.target_scheduler} vs {state.baseline_scheduler} using LLM...")
    comparison = compare_algorithms_with_llm(
        state.target_scheduler,
        state.baseline_scheduler,
        state.comparison_agent,
        MODEL_COMPARISON.split(":")[-1],
        state.logger,
    )
    state.algorithm_comparison = comparison
    print(comparison)
    return comparison, {"comparison_length": len(comparison)}


def handle_read_source_code(
    state: AgentState, scheduler_to_read: Optional[SchedulerName]
) -> Tuple[str, Dict[str, Any]]:
    """Handle the read_source_code action."""
    scheduler = scheduler_to_read or state.target_scheduler
    if scheduler in state.source_code:
        result = f"Already read source code for {scheduler}"
        print(result)
        return result, {}

    print(f"\nReading source code for {scheduler}...")
    source = get_scheduler_source_code(scheduler)
    state.source_code[scheduler] = source
    result = f"Source code for {scheduler}:\n\n{source}"
    print(f"Read {len(source)} characters of source code for {scheduler}")
    return result, {"scheduler": scheduler, "source_length": len(source)}


def handle_run_pisa(state: AgentState) -> Tuple[str, Dict[str, Any]]:
    """Handle the run_pisa action."""
    print("\nRunning PISA experiment...")
    pisa_result = run_pisa_experiment(
        state.results_dir,
        state.target_scheduler,
        state.baseline_scheduler,
    )
    state.pisa_results.append(pisa_result)
    print(pisa_result)
    return pisa_result, {"pisa_experiment_count": len(state.pisa_results)}


def handle_test_code_hypothesis(
    state: AgentState, hypothesis: Optional[CodeHypothesis], min_confidence_threshold: float
) -> Tuple[str, Dict[str, Any]]:
    """Handle the test_code_hypothesis action."""
    if not hypothesis:
        return "No code_hypothesis provided for test_code_hypothesis action", {"error": "missing"}

    print(f"\nTesting code-based hypothesis: {hypothesis.name}")
    print(f"Reasoning: {hypothesis.reasoning}")
    print(f"Code preview: {hypothesis.code}")

    network, task_graph, error = execute_code_hypothesis(hypothesis)
    if error:
        print(f"\n Code execution error: {error}")
        return f"Code hypothesis FAILED to execute: {error}", {
            "hypothesis_id": hypothesis.hypothesis_id,
            "name": hypothesis.name,
            "error": error,
            "success": False,
        }

    # Run two independent validation batches (fresh random instances each) so a single lucky
    # draw can't get frozen into state.best_validation and drive the forced-submit guidance off
    # stale, unrepresentative data - the same lesson already applied to submit_code_hypothesis.
    validation_a = validate_code_hypothesis(
        hypothesis, num_instances=50, min_confidence_threshold=min_confidence_threshold
    )
    validation_b = validate_code_hypothesis(
        hypothesis, num_instances=50, min_confidence_threshold=min_confidence_threshold
    )
    # Report the more conservative (lower-confirmation) of the two trials.
    validation = min(validation_a, validation_b, key=lambda v: v.confirmation_rate)

    single_instance_detail = test_single_instance(
        hypothesis, state.target_scheduler, state.baseline_scheduler
    )

    result = (
        f"Code hypothesis '{hypothesis.name}': "
        f"trial_1_confirmation={validation_a.confirmation_rate:.1%}, "
        f"trial_2_confirmation={validation_b.confirmation_rate:.1%}, "
        f"avg_ratio={validation.avg_makespan_ratio:.4f}"
        f"\n\n{single_instance_detail}"
    )

    print(f"\nValidation Results (two independent 50-instance trials):")
    print(
        f"  Trial 1: confirmation_rate={validation_a.confirmation_rate:.1%} "
        f"(95% CI: {validation_a.confirmation_rate_ci_low:.1%}-{validation_a.confirmation_rate_ci_high:.1%})"
    )
    print(
        f"  Trial 2: confirmation_rate={validation_b.confirmation_rate:.1%} "
        f"(95% CI: {validation_b.confirmation_rate_ci_low:.1%}-{validation_b.confirmation_rate_ci_high:.1%})"
    )
    print(f"  Validated: {validation.is_validated}")
    print(f"\n{single_instance_detail}")

    state.code_hypotheses_tested.append((hypothesis, validation))

    is_new_best = (
        state.best_validation is None
        or validation.confirmation_rate > state.best_validation.confirmation_rate
    )
    if is_new_best:
        state.best_hypothesis = hypothesis
        state.best_validation = validation
        print("  -> New best hypothesis!")

    return result, {
        "hypothesis_id": hypothesis.hypothesis_id,
        "name": hypothesis.name,
        "trial_1_confirmation_rate": validation_a.confirmation_rate,
        "trial_2_confirmation_rate": validation_b.confirmation_rate,
        "confirmation_rate": validation.confirmation_rate,
        "confirmation_rate_ci_low": validation.confirmation_rate_ci_low,
        "confirmation_rate_ci_high": validation.confirmation_rate_ci_high,
        "avg_makespan_ratio": validation.avg_makespan_ratio,
        "max_makespan_ratio": validation.max_makespan_ratio,
        "min_makespan_ratio": validation.min_makespan_ratio,
        "is_validated": validation.is_validated,
        "success": True,
    }


def handle_refine_code_hypothesis(
    state: AgentState, hypothesis: Optional[CodeHypothesis], min_confidence_threshold: float
) -> Tuple[str, Dict[str, Any]]:
    """
    Handle the refine_code_hypothesis action: test a targeted variation of the CURRENT best
    hypothesis and report explicitly whether it actually improved things. Unlike
    test_code_hypothesis (which is meant for exploring a genuinely new idea), this is meant for
    taking the existing best_hypothesis's code and making one small, explained, diagnosed change.
    """
    if not hypothesis:
        return "No code_hypothesis provided for refine_code_hypothesis action", {"error": "missing"}

    if state.best_hypothesis is None or state.best_validation is None:
        return (
            "No best_hypothesis exists yet to refine - use test_code_hypothesis first to "
            "establish an initial hypothesis before refining it",
            {"error": "no_best_hypothesis"},
        )

    prior_name = state.best_hypothesis.name
    prior_confirmation = state.best_validation.confirmation_rate

    print(f"\nRefining '{prior_name}' -> '{hypothesis.name}'")
    print(f"Refinement reasoning: {hypothesis.reasoning}")

    network, task_graph, error = execute_code_hypothesis(hypothesis)
    if error:
        print(f"\n Code execution error: {error}")
        return f"Refined hypothesis FAILED to execute: {error}", {
            "hypothesis_id": hypothesis.hypothesis_id,
            "name": hypothesis.name,
            "error": error,
            "success": False,
        }

    validation_a = validate_code_hypothesis(
        hypothesis, num_instances=50, min_confidence_threshold=min_confidence_threshold
    )
    validation_b = validate_code_hypothesis(
        hypothesis, num_instances=50, min_confidence_threshold=min_confidence_threshold
    )
    validation = min(validation_a, validation_b, key=lambda v: v.confirmation_rate)

    delta = validation.confirmation_rate - prior_confirmation
    verdict = "IMPROVEMENT" if delta > 0.001 else "REGRESSION" if delta < -0.001 else "NO CHANGE"

    single_instance_detail = test_single_instance(
        hypothesis, state.target_scheduler, state.baseline_scheduler
    )

    result = (
        f"Refined '{prior_name}' -> '{hypothesis.name}': "
        f"confirmation {prior_confirmation:.1%} -> {validation.confirmation_rate:.1%} "
        f"({delta:+.1%}, {verdict})\n\n{single_instance_detail}"
    )

    print(
        f"  {prior_confirmation:.1%} -> {validation.confirmation_rate:.1%} "
        f"({delta:+.1%}, {verdict})"
    )
    print(f"\n{single_instance_detail}")

    state.code_hypotheses_tested.append((hypothesis, validation))

    if validation.confirmation_rate > prior_confirmation:
        state.best_hypothesis = hypothesis
        state.best_validation = validation
        print("  -> Refinement kept as new best hypothesis!")
    else:
        print("  -> Refinement did not improve on the prior best - keeping it unchanged.")

    return result, {
        "hypothesis_id": hypothesis.hypothesis_id,
        "name": hypothesis.name,
        "prior_hypothesis_name": prior_name,
        "prior_confirmation_rate": prior_confirmation,
        "trial_1_confirmation_rate": validation_a.confirmation_rate,
        "trial_2_confirmation_rate": validation_b.confirmation_rate,
        "confirmation_rate": validation.confirmation_rate,
        "confirmation_rate_ci_low": validation.confirmation_rate_ci_low,
        "confirmation_rate_ci_high": validation.confirmation_rate_ci_high,
        "delta": delta,
        "verdict": verdict,
        "avg_makespan_ratio": validation.avg_makespan_ratio,
        "is_validated": validation.is_validated,
        "success": True,
    }


def handle_submit_code_hypothesis(
    state: AgentState, min_confidence_threshold: float
) -> Tuple[str, Dict[str, Any], bool]:
    """
    Handle the submit_code_hypothesis action. Returns (result, data, should_break).

    Always re-validates state.best_hypothesis directly - it deliberately IGNORES any
    code_hypothesis the decision agent may have generated for this action. Submitting is
    supposed to mean "confirm the hypothesis that already earned best-so-far status," but the
    decision agent has been observed silently writing a brand-new, unrelated hypothesis and
    reusing the same name each time it "submits" - so trusting the model's own code_hypothesis
    field here would validate the wrong thing.
    """
    hypothesis = state.best_hypothesis
    if not hypothesis:
        return (
            "No best_hypothesis exists yet - use test_code_hypothesis first before submitting",
            {"error": "no_best_hypothesis"},
            False,
        )

    print(f"\nSubmitting code-based hypothesis: {hypothesis.name}")

    # Run two independent validation batches (fresh random instances each) so a single lucky
    # draw can't pass acceptance on its own - both batches must clear the threshold.
    validation_a = validate_code_hypothesis(
        hypothesis, num_instances=100, min_confidence_threshold=min_confidence_threshold
    )
    validation_b = validate_code_hypothesis(
        hypothesis, num_instances=100, min_confidence_threshold=min_confidence_threshold
    )
    # Report the more conservative (lower-confirmation) of the two trials.
    validation = min(validation_a, validation_b, key=lambda v: v.confirmation_rate)
    both_passed = (
        validation_a.confirmation_rate >= min_confidence_threshold
        and validation_b.confirmation_rate >= min_confidence_threshold
    )

    result = (
        f"Submitted code hypothesis '{hypothesis.name}': "
        f"trial_1_confirmation={validation_a.confirmation_rate:.1%} "
        f"(95% CI: {validation_a.confirmation_rate_ci_low:.1%}-{validation_a.confirmation_rate_ci_high:.1%}), "
        f"trial_2_confirmation={validation_b.confirmation_rate:.1%} "
        f"(95% CI: {validation_b.confirmation_rate_ci_low:.1%}-{validation_b.confirmation_rate_ci_high:.1%})"
    )
    result_data = {
        "hypothesis_id": hypothesis.hypothesis_id,
        "name": hypothesis.name,
        "trial_1_confirmation_rate": validation_a.confirmation_rate,
        "trial_2_confirmation_rate": validation_b.confirmation_rate,
        "confirmation_rate": validation.confirmation_rate,
        "confirmation_rate_ci_low": validation.confirmation_rate_ci_low,
        "confirmation_rate_ci_high": validation.confirmation_rate_ci_high,
        "avg_makespan_ratio": validation.avg_makespan_ratio,
        "is_validated": validation.is_validated,
        "accepted": both_passed,
    }

    print(f"\nFinal Validation (two independent 100-instance trials):")
    print(
        f"  Trial 1: confirmation_rate={validation_a.confirmation_rate:.1%} "
        f"(95% CI: {validation_a.confirmation_rate_ci_low:.1%}-{validation_a.confirmation_rate_ci_high:.1%}), "
        f"avg_ratio={validation_a.avg_makespan_ratio:.4f}"
    )
    print(
        f"  Trial 2: confirmation_rate={validation_b.confirmation_rate:.1%} "
        f"(95% CI: {validation_b.confirmation_rate_ci_low:.1%}-{validation_b.confirmation_rate_ci_high:.1%}), "
        f"avg_ratio={validation_b.avg_makespan_ratio:.4f}"
    )

    if both_passed:
        print(f"\n Code hypothesis ACCEPTED - both trials cleared {min_confidence_threshold:.0%} confirmation!")
        state.best_hypothesis = hypothesis
        state.best_validation = validation
        return result, result_data, True

    print(
        f"\n Hypothesis not confident enough "
        f"(trial 1={validation_a.confirmation_rate:.1%}, trial 2={validation_b.confirmation_rate:.1%}, "
        f"need BOTH >= {min_confidence_threshold:.0%})"
    )
    state.code_hypotheses_tested.append((hypothesis, validation))
    # This is a larger, more rigorous re-validation of the SAME hypothesis that was already
    # best_validation - always replace the old (smaller-sample) estimate with this one,
    # even though it's lower. Otherwise a stale, lucky small-sample number stays frozen in
    # state and keeps incorrectly triggering the forced-submit guidance every iteration.
    state.best_validation = validation

    return result, result_data, False


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
    results_dir = thisdir / "results-v2"
    results_dir.mkdir(exist_ok=True)

    # Create agents with configured models
    planning_agent = create_planning_agent(MODEL_PLANNING)
    decision_agent = create_decision_agent(MODEL_DECISION)
    reflection_agent = create_reflection_agent(MODEL_REFLECTION)
    comparison_agent = create_comparison_agent(MODEL_COMPARISON)

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
        comparison_agent=comparison_agent,
    )

    print(f"Starting agentic loop: Find where {target_scheduler} underperforms vs {baseline_scheduler}")
    print(f"Max iterations: {max_iterations}")
    print(f"Logs will be saved to: {logger.log_dir}")
    print("=" * 60)

    # Force source code reading for both schedulers before the loop
    print("\n--- PRE-LOOP: Reading source code ---")
    handle_read_source_code(state, target_scheduler)
    handle_read_source_code(state, baseline_scheduler)

    previous_reflection: Optional[ActionReflection] = None

    while state.iteration < max_iterations:
        state.iteration += 1
        print(f"\n{'='*60}")
        print(f"ITERATION {state.iteration}/{max_iterations}")
        print("=" * 60)

        context = state.get_context_summary()
        print(f"\nContext:\n{context}\n")
        logger.log_iteration_start(state.iteration, context)

        # PHASE 1: STRATEGIC PLANNING
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
        logger.log_token_usage("planning", plan_result.usage, MODEL_PLANNING.split(":")[-1])

        print(f"\n STRATEGIC PLAN:")
        print(f"   Understanding: {plan.current_understanding}")
        print(f"   Key unknowns: {', '.join(plan.key_unknowns)}")
        if plan.working_hypothesis:
            print(f"   Working hypothesis: {plan.working_hypothesis}")
        print(f"   Next steps: {'; '.join(plan.next_steps)}")
        print(f"   Immediate action: {plan.immediate_action}")

        logger.log_strategic_plan(plan)

        # PHASE 2: ACTION DECISION
        print(f"\n{'-'*40}")
        print("PHASE 2: ACTION DECISION")
        print(f"{'-'*40}")

        # Build action guidance based on state
        action_guidance = ""
        if (
            state.best_validation is not None
            and state.best_validation.confirmation_rate >= min_confidence_threshold
        ):
            action_guidance = f"""
IMPORTANT: Your best hypothesis so far ('{state.best_hypothesis.name}') already has
{state.best_validation.confirmation_rate:.0%} confirmation, which clears the required
{min_confidence_threshold:.0%} threshold. You MUST call submit_code_hypothesis now with
this hypothesis unless you have a specific, concrete reason to believe further exploration
will find something meaningfully better. Do not keep testing new hypotheses just because
iterations remain - a validated result in hand is better than continued unstructured search."""
        elif len(state.pisa_results) >= 2:
            action_guidance = """
IMPORTANT: You have already run PISA multiple times. DO NOT run PISA again.
You MUST now use test_code_hypothesis to write code that creates task graph families
based on the patterns PISA found. Look at the task costs, dependencies, and network
configurations from PISA results and write code to generate similar structures."""
        elif len(state.pisa_results) == 1 and not state.code_hypotheses_tested:
            action_guidance = """
You have PISA results showing adversarial patterns. Now use test_code_hypothesis
to write code that generates task graphs with similar characteristics."""
        elif state.code_hypotheses_tested:
            last_hyp, last_val = state.code_hypotheses_tested[-1]
            if last_val.confirmation_rate < 0.5:
                action_guidance = f"""
Your last hypothesis '{last_hyp.name}' had only {last_val.confirmation_rate:.0%} confirmation.
You MUST write a NEW code hypothesis with a DIFFERENT structure. Try:
- More extreme asymmetry in task weights
- Different dependency patterns (fan-in, fan-out, diamond)
- Different network configurations (heterogeneous speeds)"""

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
Next question: {previous_reflection.next_question}
Confidence: {previous_reflection.confidence_assessment}
''' if previous_reflection else ''}{action_guidance}

Choose the action that best aligns with the strategic plan.
When you have a validated hypothesis with >{min_confidence_threshold:.0%} confirmation, submit it.
"""

        result = decision_agent.run_sync(decision_prompt)
        action = result.output
        logger.log_token_usage("decision", result.usage, MODEL_DECISION.split(":")[-1])

        # Pre-flight dry run: actually execute a fresh code_hypothesis once (cheap - no
        # scheduler calls, no LLM calls) before committing to the action. execute_code_hypothesis
        # already catches both compile errors (SyntaxError) and runtime errors (KeyError on a
        # missing weight=, NetworkXUnfeasible on a cyclic graph, IndexError, etc.) and returns a
        # helpful, specific message for each. If it fails, give the decision agent one immediate,
        # targeted chance to fix it - rather than waiting a full external iteration to find out.
        if action.action in ("test_code_hypothesis", "refine_code_hypothesis") and action.code_hypothesis:
            _, _, preflight_error = execute_code_hypothesis(action.code_hypothesis)
            if preflight_error:
                print(f"\n Pre-flight check failed: {preflight_error}\nRequesting one immediate fix...")
                fix_prompt = f"""Your previously generated code_hypothesis fails when executed:

{preflight_error}

BROKEN CODE:
```python
{action.code_hypothesis.code}
```

Your response MUST set action='{action.action}' and MUST include a code_hypothesis field - do
not omit it. Reuse the SAME code_hypothesis name, reasoning, and schedulers, but with the `code`
field corrected so it runs successfully.
"""
                fix_result = decision_agent.run_sync(fix_prompt)
                logger.log_token_usage("decision", fix_result.usage, MODEL_DECISION.split(":")[-1])
                fixed_hypothesis = fix_result.output.code_hypothesis
                retry_error = "no code_hypothesis returned"
                if fixed_hypothesis:
                    _, _, retry_error = execute_code_hypothesis(fixed_hypothesis)
                if not retry_error:
                    action = fix_result.output
                    print(" Fix succeeded - code now runs cleanly.")
                else:
                    print(f" Fix attempt still fails ({retry_error}) - proceeding, will be reported as a failure.")

        print(f"\n ACTION: {action.action}")
        print(f"   Reasoning: {action.reasoning}")

        action_params: Dict[str, Any] = {}
        if action.scheduler_to_read:
            action_params["scheduler_to_read"] = action.scheduler_to_read
        if action.code_hypothesis:
            action_params["code_hypothesis"] = {
                "name": action.code_hypothesis.name,
                "hypothesis_id": action.code_hypothesis.hypothesis_id,
                "reasoning": action.code_hypothesis.reasoning,
                "code": action.code_hypothesis.code,
                "confidence": action.code_hypothesis.confidence,
                "worse_scheduler": action.code_hypothesis.worse_scheduler,
                "better_scheduler": action.code_hypothesis.better_scheduler,
            }

        logger.log_agent_decision(action.action, action.reasoning, action_params)

        # Execute the action
        action_result = ""
        result_data: Dict[str, Any] = {}
        should_break = False

        if action.action == "compare_algorithms":
            action_result, result_data = handle_compare_algorithms(state)

        elif action.action == "read_source_code":
            action_result, result_data = handle_read_source_code(state, action.scheduler_to_read)

        elif action.action == "run_pisa":
            action_result, result_data = handle_run_pisa(state)

        elif action.action == "test_code_hypothesis":
            action_result, result_data = handle_test_code_hypothesis(
                state, action.code_hypothesis, min_confidence_threshold
            )

        elif action.action == "refine_code_hypothesis":
            action_result, result_data = handle_refine_code_hypothesis(
                state, action.code_hypothesis, min_confidence_threshold
            )

        elif action.action == "submit_code_hypothesis":
            action_result, result_data, should_break = handle_submit_code_hypothesis(
                state, min_confidence_threshold
            )

        logger.log_action_result(action_result, result_data)

        if should_break:
            logger.finalize_iteration()
            break

        # PHASE 3: REFLECTION
        print(f"\n{'-'*40}")
        print("PHASE 3: REFLECTION")
        print(f"{'-'*40}")

        reflection_prompt = f"""Reflect on the results of the action just taken.

ACTION TAKEN: {action.action}
REASONING: {action.reasoning}

RESULT:
{action_result}

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
        logger.log_token_usage("reflection", reflection_result.usage, MODEL_REFLECTION.split(":")[-1])
        previous_reflection = reflection

        print(f"\n REFLECTION:")
        print(f"   Key findings: {'; '.join(reflection.key_findings)}")
        if reflection.surprises:
            print(f"   Surprises: {'; '.join(reflection.surprises)}")
        print(f"   Hypothesis update: {reflection.hypothesis_update}")
        print(f"   Next question: {reflection.next_question}")
        print(f"   Confidence: {reflection.confidence_assessment}")

        logger.log_reflection(reflection)
        logger.finalize_iteration()

    # Finalize
    logger.log_run_complete(state.best_hypothesis, state.best_validation, state.iteration)
    logger.save_visualization_data()

    print("\n" + "=" * 60)
    print("AGENTIC LOOP COMPLETE")
    print("=" * 60)

    if state.best_hypothesis:
        print(f"\nBest Hypothesis Found:")
        print(f"  Name: {state.best_hypothesis.name}")
        print(f"  Reasoning: {state.best_hypothesis.reasoning}")
        print(f"\n  Code:\n{state.best_hypothesis.code}")

        output_path = results_dir / "best_hypothesis.json"
        output_path.write_text(state.best_hypothesis.model_dump_json(indent=2))
        print(f"\nSaved to: {output_path}")

        code_path = results_dir / "best_hypothesis_code.py"
        code_path.write_text(
            f'''"""
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
    network, task_graph = get_instance()
    print(f"Network nodes: {{list(network.graph.nodes())}}")
    print(f"Task graph nodes: {{list(task_graph.graph.nodes())}}")
'''
        )
        print(f"Code saved to: {code_path}")

        if state.best_validation:
            print(f"\nValidation:")
            print(
                f"  Confirmation rate: {state.best_validation.confirmation_rate:.1%} "
                f"(95% CI: {state.best_validation.confirmation_rate_ci_low:.1%}-"
                f"{state.best_validation.confirmation_rate_ci_high:.1%})"
            )
            print(f"  Avg ratio: {state.best_validation.avg_makespan_ratio:.4f}")
            print(f"  Validated: {state.best_validation.is_validated}")
    else:
        print("\nNo hypothesis found.")

    token_summary = logger.get_token_summary()
    print(f"\n{'='*60}")
    print("TOKEN USAGE & COST")
    print("=" * 60)
    print(f"  Total input tokens:  {token_summary['total_input_tokens']:,}")
    print(f"  Total output tokens: {token_summary['total_output_tokens']:,}")
    print(f"  Total tokens:        {token_summary['total_tokens']:,}")
    print(f"  Estimated cost:      ${token_summary['estimated_cost_usd']:.4f} USD")
    print("\n  By Agent:")
    for agent_name, usage in token_summary["by_agent"].items():
        print(
            f"    {agent_name}: {usage['calls']} calls, "
            f"{usage['input_tokens']:,} in, {usage['output_tokens']:,} out"
        )

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