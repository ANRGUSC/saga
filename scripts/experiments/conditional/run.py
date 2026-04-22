"""Conditional scheduling experiment.

Compares trace makespans from overlapping CTG scheduling vs scheduling each
trace as a standalone task graph.

Each execution creates a self-contained run folder under runs/<run_id>/ with
JSON instance dumps, CSVs, and plots so results are reproducible and PDFs can
be regenerated from the saved JSON files.

Workflow:
  for each heuristic
    for each network
      for each CTG
        - schedule CTG with overlapping conditional logic
        - extract per-trace schedules, recalculated to remove gaps
        - schedule each trace standalone as a plain TaskGraph
        - log makespans + metrics to CSV
        - save visualisations & JSON dumps
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from saga import Network, Schedule, ScheduledTask, Scheduler, TaskGraphNode
from saga.conditional import (
    ConditionalTaskGraph,
    ConditionalTaskGraphEdge,
    extract_traces_with_recalculation,
    schedule_trace_standalone,
)
from saga.schedulers import CpopScheduler, HeftScheduler
from saga.utils.draw import draw_gantt, draw_mutual_exclusion_graph, draw_task_graph
from saga.utils.random_graphs import get_random_conditional_branching_dag


# Configuration ---------------------------------------------------------------------------


HEURISTICS: Dict[str, Scheduler] = {
    "HEFT": HeftScheduler(),
    # "CPOP": CpopScheduler(),
}

NETWORKS: Dict[str, Network] = {
    "3node_hetero": Network.create(
        nodes=[("n0", 1.0), ("n1", 1.5), ("n2", 2.0)],
        edges=[("n0", "n1", 1.0), ("n0", "n2", 1.0), ("n1", "n2", 1.0)],
    ),
    # "2node_homo": Network.create(
    #     nodes=[("n0", 1.0), ("n1", 1.0)],
    #     edges=[("n0", "n1", 1.0)],
    # ),
}

CTG_CONFIGS: List[dict] = [
    {"levels": 3, "branching_factor": 2, "conditional_parent_probability": 0.50},
    # {"levels": 3, "branching_factor": 2, "conditional_parent_probability": 0.9},
]

SEED = 44
NUM_REPEATS = 1


# Helpers ---------------------------------------------------------------------------


thisdir = Path(__file__).resolve().parent


def _safe(name: str) -> str:
    return name.replace(" ", "_").replace(":", "").replace("/", "-")


def _trace_makespan(mapping: Dict[str, list]) -> float:
    return max(
        (t.end for tasks in mapping.values() for t in tasks),
        default=0.0,
    )


def _make_run_root() -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    run_id = f"{ts}_seed{SEED}"
    run_root = thisdir / "runs" / run_id
    run_root.mkdir(parents=True, exist_ok=False)
    return run_root


def generate_ctgs() -> List[ConditionalTaskGraph]:
    np.random.seed(SEED)
    ctgs: List[ConditionalTaskGraph] = []
    for _repeat in range(NUM_REPEATS):
        for graphs in CTG_CONFIGS:
            ctgs.append(get_random_conditional_branching_dag(**graphs))
    return ctgs


def _save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def _network_to_json(network: Network) -> dict:
    return network.model_dump(mode="json")


def _ctg_to_json(ctg: ConditionalTaskGraph) -> dict:
    """Serialise a ConditionalTaskGraph, preserving probability on each edge.

    Pydantic serialises ``dependencies`` using the base ``TaskGraphEdge``
    schema, which silently drops ``probability``. We rebuild the dict
    explicitly so probability is always present.
    """
    data = ctg.model_dump(mode="json")
    prob_by_edge = {
        (e.source, e.target): e.probability
        for e in ctg.dependencies
        if isinstance(e, ConditionalTaskGraphEdge)
    }
    for dep in data["dependencies"]:
        dep["probability"] = prob_by_edge.get((dep["source"], dep["target"]), 1.0)
    return data


def _ctg_from_json(data: dict) -> ConditionalTaskGraph:
    """Reconstruct a ConditionalTaskGraph from a model_dump JSON dict.

    Pydantic would deserialise dependencies as base TaskGraphEdge objects,
    losing probability, so we explicitly rebuild ConditionalTaskGraphEdge
    objects from the raw dict data.
    """
    tasks = [TaskGraphNode(**t) for t in data["tasks"]]
    deps = [
        ConditionalTaskGraphEdge(
            source=d["source"],
            target=d["target"],
            size=d["size"],
            probability=d.get("probability", 1.0),
        )
        for d in data["dependencies"]
    ]
    return ConditionalTaskGraph.create(tasks=tasks, dependencies=deps)


# Instance saving ---------------------------------------------------------------------------


def save_instance(
    run_root: Path,
    ctg_idx: int,
    ctg: ConditionalTaskGraph,
    network: Network,
    network_name: str,
    traces_detailed: List[dict],
) -> None:
    inst_dir = run_root / "instances" / f"ctg_{ctg_idx:03d}"
    inst_dir.mkdir(parents=True, exist_ok=True)
    _save_json(_network_to_json(network), inst_dir / f"network_{network_name}.json")
    _save_json(_ctg_to_json(ctg), inst_dir / "taskgraph_ctg.json")
    _save_json(traces_detailed, inst_dir / "traces.json")


def save_meta(run_root: Path) -> None:
    meta = {
        "run_id": run_root.name,
        "seed": SEED,
        "num_repeats": NUM_REPEATS,
        "heuristics": list(HEURISTICS.keys()),
        "networks": list(NETWORKS.keys()),
        "ctg_configs": CTG_CONFIGS,
    }
    _save_json(meta, run_root / "meta.json")


# Plot generation, factored out for replay -----------------------------------------------------


def generate_plots(
    run_root: Path,
    tag: str,
    ctg: ConditionalTaskGraph,
    schedule,
    meg,
    traces_detailed: List[dict],
    trace_schedules: Dict[str, Dict[str, list]],
    standalone_schedules: Dict[str, Schedule],
    ctg_makespan: float,
) -> None:
    full_dir = run_root / "plots" / "full" / tag
    full_dir.mkdir(parents=True, exist_ok=True)
    simp_dir = run_root / "plots" / "simple" / tag
    simp_dir.mkdir(parents=True, exist_ok=True)

    draw_task_graph(ctg.graph, use_latex=False).get_figure().savefig(
        full_dir / "conditional_task_graph.pdf"
    )
    draw_gantt(schedule.mapping, use_latex=False).get_figure().savefig(
        full_dir / "conditional_schedule.pdf"
    )
    draw_mutual_exclusion_graph(meg, use_latex=False).get_figure().savefig(
        full_dir / "mutual_exclusion_graph.pdf"
    )
    plt.close("all")

    for trace_info in traces_detailed:
        trace_name = trace_info["name"]
        trace_tasks = trace_info["tasks"]
        safe_trace = _safe(trace_name)

        trace_mapping = trace_schedules[trace_name]
        trace_ctg_ms = _trace_makespan(trace_mapping)
        standalone = standalone_schedules[trace_name]
        standalone_ms = standalone.makespan

        trace_subgraph = ctg.graph.subgraph(trace_tasks).copy()
        draw_task_graph(trace_subgraph, use_latex=False).get_figure().savefig(
            full_dir / f"trace_taskgraph_{safe_trace}.pdf"
        )
        draw_gantt(trace_mapping, use_latex=False).get_figure().savefig(
            full_dir / f"trace_ctg_schedule_{safe_trace}.pdf"
        )
        draw_gantt(standalone.mapping, use_latex=False).get_figure().savefig(
            full_dir / f"trace_standalone_schedule_{safe_trace}.pdf"
        )
        plt.close("all")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 4))
        xmax = max(ctg_makespan, standalone_ms) * 1.05
        draw_gantt(trace_mapping, use_latex=False, axis=ax1, xmax=xmax)
        ax1.set_title(f"From CTG ({trace_ctg_ms:.2f})")
        draw_gantt(standalone.mapping, use_latex=False, axis=ax2, xmax=xmax)
        ax2.set_title(f"Standalone ({standalone_ms:.2f})")
        fig.suptitle(trace_name, fontsize=14)
        fig.tight_layout()
        fig.savefig(simp_dir / f"{safe_trace}_comparison.pdf")
        plt.close("all")


# Replay: regenerate PDFs from saved JSON instances ---------------------------------------------


def _load_traces_json(ctg_dir: Path) -> List[dict]:
    """Load trace metadata, with fallback for old runs saved as branches.json."""
    traces_json_path = ctg_dir / "traces.json"
    if not traces_json_path.exists():
        traces_json_path = ctg_dir / "branches.json"

    with open(traces_json_path) as f:
        return json.load(f)


def _extract_schedules_from_traces(
    schedule: Schedule,
    traces_detailed: List[dict],
) -> Dict[str, Dict[str, List[ScheduledTask]]]:
    """Filter and recalculate trace schedules using pre-computed trace info.

    Equivalent to extract_traces_with_recalculation() but uses the trace task
    lists supplied directly instead of re-identifying them from the CTG. This
    also keeps replay safe for old JSON files that may be missing probability
    fields.
    """
    from saga.conditional import recalculate_trace_times

    result: Dict[str, Dict[str, List[ScheduledTask]]] = {}
    for trace_info in traces_detailed:
        task_set = set(trace_info["tasks"])
        trace_mapping: Dict[str, List[ScheduledTask]] = {
            node_name: [
                ScheduledTask(node=t.node, name=t.name, start=t.start, end=t.end)
                for t in tasks
                if t.name in task_set
            ]
            for node_name, tasks in schedule.mapping.items()
        }
        result[trace_info["name"]] = recalculate_trace_times(trace_mapping, schedule)
    return result


def replay_plots(run_dir: Path) -> None:
    """Regenerate all PDFs from JSON files saved in a previous run.

    Usage:
        replay_plots(Path("runs/2026-04-15T143022Z_seed42"))
    """
    meta_path = run_dir / "meta.json"
    with open(meta_path) as f:
        meta = json.load(f)

    heuristics = {name: HEURISTICS[name] for name in meta["heuristics"]}
    networks = {name: NETWORKS[name] for name in meta["networks"]}

    inst_base = run_dir / "instances"
    ctg_dirs = sorted(inst_base.iterdir())

    for ctg_dir in ctg_dirs:
        ctg_idx = int(ctg_dir.name.split("_")[1])
        ctg_json_path = ctg_dir / "taskgraph_ctg.json"

        with open(ctg_json_path) as f:
            ctg_data = json.load(f)
        ctg = _ctg_from_json(ctg_data)

        traces_detailed = _load_traces_json(ctg_dir)

        for heuristic_name, scheduler in heuristics.items():
            for network_name, network in networks.items():
                tag = f"{heuristic_name}_{network_name}_ctg{ctg_idx}"
                schedule = scheduler.schedule(network=network, task_graph=ctg)
                ctg_makespan = schedule.makespan
                meg = ctg.build_mutual_exclusion_graph()

                # Use traces from JSON as source of truth rather than
                # re-identifying from the CTG. This guards against old JSON
                # files where probability was not serialised.
                trace_schedules = _extract_schedules_from_traces(
                    schedule, traces_detailed
                )

                standalone_schedules = {}
                for trace_info in traces_detailed:
                    standalone_schedules[trace_info["name"]] = schedule_trace_standalone(
                        trace_info["tasks"], ctg, network, scheduler
                    )

                generate_plots(
                    run_dir,
                    tag,
                    ctg,
                    schedule,
                    meg,
                    traces_detailed,
                    trace_schedules,
                    standalone_schedules,
                    ctg_makespan,
                )
                print(f"[replay] {tag} plots regenerated")


# Main experiment loop ---------------------------------------------------------------------------


def run() -> None:
    run_root = _make_run_root()
    save_meta(run_root)
    # print(f"Run folder: {run_root}")

    rows: List[dict] = []
    ctgs = generate_ctgs()

    for heuristic_name, scheduler in HEURISTICS.items():
        for network_name, network in NETWORKS.items():
            for ctg_idx, ctg in enumerate(ctgs):
                tag = f"{heuristic_name}_{network_name}_ctg{ctg_idx}"
                print(f"[{tag}] scheduling CTG ({len(ctg.tasks)} tasks) ...")

                traces_detailed = ctg.identify_traces_detailed()

                save_instance(
                    run_root, ctg_idx, ctg, network, network_name, traces_detailed,
                )

                schedule = scheduler.schedule(network=network, task_graph=ctg)
                ctg_makespan = schedule.makespan
                meg = ctg.build_mutual_exclusion_graph()

                trace_schedules = extract_traces_with_recalculation(schedule)

                standalone_schedules: Dict[str, Schedule] = {}
                for trace_info in traces_detailed:
                    standalone_schedules[trace_info["name"]] = schedule_trace_standalone(
                        trace_info["tasks"], ctg, network, scheduler
                    )

                generate_plots(
                    run_root,
                    tag,
                    ctg,
                    schedule,
                    meg,
                    traces_detailed,
                    trace_schedules,
                    standalone_schedules,
                    ctg_makespan,
                )

                for trace_info in traces_detailed:
                    trace_name = trace_info["name"]
                    trace_tasks = trace_info["tasks"]
                    trace_mapping = trace_schedules[trace_name]
                    trace_ctg_ms = _trace_makespan(trace_mapping)
                    standalone_ms = standalone_schedules[trace_name].makespan

                    trace_gap = trace_ctg_ms - standalone_ms
                    trace_ratio = standalone_ms / max(trace_ctg_ms, 1e-9)

                    rows.append(
                        {
                            "heuristic": heuristic_name,
                            "network": network_name,
                            "ctg_id": ctg_idx,
                            "trace": trace_name,
                            "trace_probability": trace_info["probability"],
                            "num_tasks_ctg": len(ctg.tasks),
                            "num_tasks_trace": len(trace_tasks),
                            "ctg_overlapping_makespan": round(ctg_makespan, 4),
                            "trace_ctg_makespan": round(trace_ctg_ms, 4),
                            "trace_standalone_makespan": round(standalone_ms, 4),
                            "trace_gap": round(trace_gap, 4),
                            "trace_ratio_standalone_over_ctg": round(trace_ratio, 4),
                        }
                    )

                    print(
                        f"  {trace_name} (p={trace_info['probability']:.3f}): "
                        f"ctg={trace_ctg_ms:.3f}  "
                        f"standalone={standalone_ms:.3f}  "
                        f"gap={trace_gap:.3f}"
                    )

    df = pd.DataFrame(rows)
    df.to_csv(run_root / "results.csv", index=False)

    summary = (
        df.groupby(["heuristic", "network", "ctg_id"])
        .agg(
            num_traces=("trace", "count"),
            mean_gap=("trace_gap", "mean"),
            median_gap=("trace_gap", "median"),
            max_gap=("trace_gap", "max"),
            mean_ratio=("trace_ratio_standalone_over_ctg", "mean"),
            median_ratio=("trace_ratio_standalone_over_ctg", "median"),
        )
        .reset_index()
    )
    summary.to_csv(run_root / "summary_by_ctg.csv", index=False)

    print(f"\nResults saved to {run_root / 'results.csv'}")
    print(f"Summary saved to {run_root / 'summary_by_ctg.csv'}")
    print(df.to_string(index=False))


def main():
    #run()
    replay_plots(Path("runs/2026-04-22T054526Z_seed44"))


if __name__ == "__main__":
    main()
