# LLM-Powered Hypothesis Generation for Scheduling Algorithm Analysis

This example demonstrates an agentic system that uses Pydantic AI to automatically discover task graph families where one scheduling algorithm underperforms compared to another.

## Overview

The system runs an iterative loop where an LLM agent:
1. Explores scheduling algorithm source code
2. Runs PISA (simulated annealing) experiments to find adversarial instances
3. Generates code-based hypotheses that create task graph families
4. Validates hypotheses by testing across multiple random instances
5. Refines hypotheses based on validation results

## Code Structure

```
openai_example/
├── main.py          # Orchestration layer and main agentic loop
├── models.py        # Pydantic data models
├── agents.py        # Pydantic AI agent definitions
├── hypothesis.py    # Hypothesis execution and validation
├── tools.py         # Utility functions (PISA, source code, LLM comparison)
├── logger.py        # Experiment logging and visualization
└── results/         # Output directory (created at runtime)
```

### Module Descriptions

#### `main.py`
The main entry point containing:
- `AgentState`: Dataclass tracking the agent's state across iterations
- Action handlers for each possible agent action
- `run_agentic_loop()`: The main loop orchestrating planning, decision, and reflection phases

#### `models.py`
Pydantic models for structured data:
- `CodeHypothesis`: A hypothesis expressed as executable Python code
- `HypothesisValidationResult`: Results from validating a hypothesis
- `AgentAction`: The agent's decision about what action to take
- `StrategicPlan`: A strategic plan for each iteration
- `ActionReflection`: Reflection on the results of an action
- `AlgorithmComparison`: Structured comparison of two algorithms

#### `agents.py`
Pydantic AI agent definitions:
- `planning_agent`: Creates strategic plans for each iteration
- `decision_agent`: Decides which action to take next
- `reflection_agent`: Analyzes results and updates hypotheses
- `comparison_agent`: Compares two scheduling algorithms

#### `hypothesis.py`
Functions for working with code-based hypotheses:
- `execute_code_hypothesis()`: Safely executes hypothesis code to generate instances
- `validate_code_hypothesis()`: Tests a hypothesis across multiple random instances

#### `tools.py`
Utility functions called by the agentic loop:
- `get_scheduler_source_code()`: Retrieves source code for a scheduler
- `run_pisa_experiment()`: Runs PISA to find adversarial instances
- `compare_algorithms_with_llm()`: Uses LLM to compare two algorithms

#### `logger.py`
Experiment logging and visualization:
- `ExperimentLogger`: Tracks all agent actions, token usage, and results
- Generates JSON logs, summary reports, and interactive HTML visualizations
- Calculates estimated costs based on token usage

## Usage

```python
from main import run_agentic_loop

hypothesis, validation = run_agentic_loop(
    target_scheduler="HEFT",      # Scheduler to find weaknesses in
    baseline_scheduler="CPoP",    # Scheduler to compare against
    max_iterations=10,            # Maximum iterations before stopping
    min_confidence_threshold=0.6, # Required confirmation rate to accept
)
```

Or run directly:
```bash
python main.py
```

## Model Configuration

Models are configured at the top of `main.py`. You can use different models for different agents:

```python
# In main.py - configure these variables
MODEL_PLANNING = "openai:gpt-4o"      # Strategic planning
MODEL_DECISION = "openai:gpt-4o"      # Action decisions and code generation
MODEL_REFLECTION = "openai:gpt-4o"    # Reflection and analysis
MODEL_COMPARISON = "openai:gpt-4o"    # Algorithm comparison
```

### Available Models

| Provider | Model Examples |
|----------|---------------|
| OpenAI | `openai:gpt-4o`, `openai:gpt-4o-mini`, `openai:gpt-5`, `openai:o1`, `openai:o3-mini` |
| Anthropic | `anthropic:claude-sonnet-4-20250514`, `anthropic:claude-3-5-haiku-20241022` |
| Google | `google-gla:gemini-2.0-flash`, `google-gla:gemini-1.5-pro` |
| Groq | `groq:llama-3.3-70b-versatile` |

The decision agent benefits from a more capable model since it writes code. You can use a cheaper/faster model for planning and reflection if desired.

## Requirements

- `pydantic-ai`: For LLM agent definitions
- `python-dotenv`: For loading environment variables
- `saga`: The scheduling library (install from parent project)
- API key for your chosen provider (e.g., `OPENAI_API_KEY` environment variable)

## Output

Results are saved to `results/`:
- `logs/<run_id>/`: Per-run logs including:
  - `run_metadata.json`: Run configuration and final results
  - `iteration_*.json`: Detailed logs for each iteration
  - `all_events.json`: Complete event timeline
  - `summary_report.txt`: Human-readable summary
  - `visualization.html`: Interactive HTML visualization
- `best_hypothesis.json`: The best hypothesis found (if any)
- `best_hypothesis_code.py`: Standalone Python file with the hypothesis code

## How It Works

Each iteration follows three phases:

1. **Strategic Planning**: The planning agent reviews the current state and creates a plan
2. **Action Decision**: The decision agent chooses one of:
   - `compare_algorithms`: Get LLM comparison of the two algorithms
   - `read_source_code`: Read a scheduler's source code
   - `run_pisa`: Run PISA to find adversarial instances automatically
   - `test_code_hypothesis`: Write and test Python code that generates instances
   - `submit_code_hypothesis`: Submit a hypothesis when confident enough
3. **Reflection**: The reflection agent analyzes results and updates the working hypothesis

The loop continues until a hypothesis achieves the required confirmation rate or max iterations is reached.
