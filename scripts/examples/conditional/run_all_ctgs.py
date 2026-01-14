"""Run all CTGs through the overlapping scheduler workflow and analyze results.

Similar structure to scripts/experiments/online/analyze_full.py for consistency.
"""
import sys
import pathlib
from typing import Dict, Hashable, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from ctg_library import get_all_ctgs, create_simple_network, create_heterogeneous_network
from overlapping_scheduler import OverlappingConditionalScheduler, OverlappingTask
from branch_extractor import (
    extract_all_branches_with_recalculation,
    generate_heft_comparison_schedules,
    identify_branches,
)
from draw_conditional_graph import draw_conditional_task_graph
from saga.utils.draw import draw_network

# ---------------- Config ----------------
THISDIR = pathlib.Path(__file__).resolve().parent
RESULTS_DIR = THISDIR / "results"
CSV_PATH = RESULTS_DIR / "results.csv"
OUTDIR = RESULTS_DIR / "plots"
RESULTS_DIR.mkdir(exist_ok=True)
OUTDIR.mkdir(exist_ok=True)


def get_makespan(schedule: Dict[Hashable, List]) -> float:
    """Calculate makespan (maximum end time) from a schedule."""
    max_end = 0.0
    for node, tasks in schedule.items():
        for task in tasks:
            max_end = max(max_end, task.end)
    return max_end


def collect_results() -> pd.DataFrame:
    """Run all CTGs through the workflow and collect results into DataFrame.
    
    For each CTG:
        - Run overlapping scheduler
        - Extract branches + recalculate times  
        - Run standard HEFT on each branch
        - Record makespans for comparison
    
    Returns:
        DataFrame with columns: ctg_name, ctg_type, num_tasks, num_branches,
                               branch_name, heft_makespan, overlapping_makespan,
                               makespan_ratio, overhead
    """
    results = []
    
    # Test with both simple and heterogeneous networks
    networks = {
        "simple_2node": create_simple_network(2),
        "simple_3node": create_simple_network(3),
        "heterogeneous": create_heterogeneous_network(),
    }
    
    all_ctgs = get_all_ctgs()
    
    for ctg, metadata in all_ctgs:
        ctg_name = metadata["name"]
        
        for network_name, network in networks.items():
            # Run overlapping scheduler
            scheduler = OverlappingConditionalScheduler()
            runtimes, commtimes = scheduler.get_runtimes(network, ctg)
            overlapping_schedule = scheduler.schedule(network, ctg)
            
            # Extract branches with recalculated times
            branch_schedules = extract_all_branches_with_recalculation(
                overlapping_schedule, ctg, runtimes, commtimes
            )
            
            # Run standard HEFT on each branch
            heft_schedules = generate_heft_comparison_schedules(ctg, network)
            
            # Collect results for each branch
            for branch_name in branch_schedules.keys():
                overlapping_makespan = get_makespan(branch_schedules[branch_name])
                heft_makespan = get_makespan(heft_schedules[branch_name])
                
                # Calculate metrics
                if heft_makespan > 0:
                    makespan_ratio = overlapping_makespan / heft_makespan
                    overhead = overlapping_makespan - heft_makespan
                else:
                    makespan_ratio = 1.0
                    overhead = 0.0
                
                results.append({
                    "ctg_name": ctg_name,
                    "ctg_type": metadata["ctg_type"],
                    "num_tasks": metadata["num_tasks"],
                    "num_branches": metadata["num_branches"],
                    "num_conditional_points": metadata["num_conditional_points"],
                    "network": network_name,
                    "branch_name": branch_name,
                    "heft_makespan": heft_makespan,
                    "overlapping_makespan": overlapping_makespan,
                    "makespan_ratio": makespan_ratio,
                    "overhead": overhead,
                })
    
    return pd.DataFrame(results)

#similar structure copied from scripts/experiments/online/analyze_full.py
def analyze_results(df: pd.DataFrame):
    """Generate analysis plots comparing HEFT vs Overlapping approach.
    
    Similar style to analyze_full.py for consistency.
    """
    sns.set(style="whitegrid", font_scale=1.2)

    # Plot 1: Scatter plot - Overlapping vs HEFT makespan

    plt.figure(figsize=(10, 8))
    
    # Add diagonal line (y = x)
    max_val = max(df["heft_makespan"].max(), df["overlapping_makespan"].max()) * 1.1
    plt.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='y = x (no overhead)')
    
    # Add jitter to separate overlapping points (commented out)
    # np.random.seed(42)  # For reproducibility
    # jitter_amount = 0.15
    # df_jittered = df.copy()
    # df_jittered["heft_makespan"] = df["heft_makespan"] + np.random.uniform(-jitter_amount, jitter_amount, len(df))
    # df_jittered["overlapping_makespan"] = df["overlapping_makespan"] + np.random.uniform(-jitter_amount, jitter_amount, len(df))
    
    # Scatter plot colored by CTG type
    ax = sns.scatterplot(
        data=df,
        x="heft_makespan",
        y="overlapping_makespan",
        hue="ctg_type",
        style="network",
        s=100,
        alpha=0.7,
    )
    
    ax.set_title("Overlapping vs Standard HEFT Makespan")
    ax.set_xlabel("HEFT Makespan (per-branch optimal)")
    ax.set_ylabel("Overlapping Approach Makespan")
    plt.legend(title="CTG Type / Network", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(OUTDIR / "scatter_heft_vs_overlapping.png", dpi=300, bbox_inches='tight')
    plt.savefig(OUTDIR / "scatter_heft_vs_overlapping.pdf", dpi=300, bbox_inches='tight')
    plt.close()
    

    # Plot 2: Box plot - Makespan ratio distribution by CTG type

    plt.figure(figsize=(12, 6))
    ax = sns.boxplot(
        data=df,
        x="ctg_type",
        y="makespan_ratio",
        hue="network",
    )
    ax.axhline(1.0, color='red', linestyle='--', linewidth=1, label='No overhead')
    ax.set_title("Makespan Ratio Distribution by CTG Type")
    ax.set_xlabel("CTG Type")
    ax.set_ylabel("Makespan Ratio (Overlapping / HEFT)")
    plt.xticks(rotation=45, ha='right')
    plt.legend(title="Network", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(OUTDIR / "boxplot_ratio_by_type.png", dpi=300, bbox_inches='tight')
    plt.savefig(OUTDIR / "boxplot_ratio_by_type.pdf", dpi=300, bbox_inches='tight')
    plt.close()
    

    # Plot 3: Heatmap - Average makespan ratio by CTG type vs network

    plt.figure(figsize=(10, 8))
    heatmap_data = df.pivot_table(
        index='ctg_type',
        columns='network',
        values='makespan_ratio',
        aggfunc='mean'
    )
    sns.heatmap(
        heatmap_data,
        annot=True,
        cmap='RdYlBu_r',
        center=1.0,
        fmt='.3f',
        vmin=0.9,
        vmax=1.5,
    )
    plt.title("Average Makespan Ratio: CTG Type vs Network")
    plt.tight_layout()
    plt.savefig(OUTDIR / "heatmap_type_vs_network.png", dpi=300, bbox_inches='tight')
    plt.savefig(OUTDIR / "heatmap_type_vs_network.pdf", dpi=300, bbox_inches='tight')
    plt.close()
    

    # Plot 4: Bar chart - Average overhead by CTG

    plt.figure(figsize=(12, 6))
    ctg_summary = df.groupby('ctg_name')['overhead'].mean().sort_values()
    ctg_summary.plot(kind='bar', color='steelblue', edgecolor='black')
    plt.axhline(0, color='red', linestyle='--', linewidth=1)
    plt.title("Average Overhead by CTG (Overlapping - HEFT)")
    plt.xlabel("CTG")
    plt.ylabel("Average Overhead (time units)")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(OUTDIR / "bar_overhead_by_ctg.png", dpi=300, bbox_inches='tight')
    plt.savefig(OUTDIR / "bar_overhead_by_ctg.pdf", dpi=300, bbox_inches='tight')
    plt.close()
    

    # Plot 5: Scatter plot - Makespan ratio vs num_conditional_points

    plt.figure(figsize=(10, 6))
    ax = sns.scatterplot(
        data=df,
        x="num_conditional_points",
        y="makespan_ratio",
        hue="ctg_name",
        style="network",
        s=100,
        alpha=0.7,
    )
    ax.axhline(1.0, color='red', linestyle='--', linewidth=1)
    ax.set_title("Makespan Ratio vs Number of Conditional Points")
    ax.set_xlabel("Number of Conditional Points")
    ax.set_ylabel("Makespan Ratio (Overlapping / HEFT)")
    plt.legend(title="CTG / Network", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(OUTDIR / "scatter_ratio_vs_conditional_points.png", dpi=300, bbox_inches='tight')
    plt.savefig(OUTDIR / "scatter_ratio_vs_conditional_points.pdf", dpi=300, bbox_inches='tight')
    plt.close()
    

    # Summary statistics

    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    print(f"\nTotal branches analyzed: {len(df)}")
    print(f"CTGs tested: {df['ctg_name'].nunique()}")
    print(f"Networks tested: {df['network'].nunique()}")
    
    print(f"\nMakespan Ratio (Overlapping / HEFT):")
    print(f"  Mean:   {df['makespan_ratio'].mean():.3f}")
    print(f"  Median: {df['makespan_ratio'].median():.3f}")
    print(f"  Min:    {df['makespan_ratio'].min():.3f}")
    print(f"  Max:    {df['makespan_ratio'].max():.3f}")
    print(f"  Std:    {df['makespan_ratio'].std():.3f}")
    
    print(f"\nOverhead (Overlapping - HEFT):")
    print(f"  Mean:   {df['overhead'].mean():.3f}")
    print(f"  Median: {df['overhead'].median():.3f}")
    
    # Breakdown by CTG type
    print("\n\nBy CTG Type:")
    type_summary = df.groupby('ctg_type').agg({
        'makespan_ratio': ['mean', 'std'],
        'overhead': 'mean',
    }).round(3)
    print(type_summary.to_string())
    
    # Cases where overlapping is worse
    worse_cases = df[df['makespan_ratio'] > 1.0]
    print(f"\n\nCases where Overlapping > HEFT: {len(worse_cases)} / {len(df)} ({100*len(worse_cases)/len(df):.1f}%)")
    
    # Cases where overlapping equals HEFT
    equal_cases = df[abs(df['makespan_ratio'] - 1.0) < 0.001]
    print(f"Cases where Overlapping ≈ HEFT: {len(equal_cases)} / {len(df)} ({100*len(equal_cases)/len(df):.1f}%)")


def generate_reference_pdf():
    """Generate a PDF showing all networks and CTGs used in the experiments.
    
    Creates a multi-page PDF with:
    - Page 1: All networks used (simple_2node, simple_3node, heterogeneous)
    - Pages 2+: All CTGs from the library
    """
    from matplotlib.backends.backend_pdf import PdfPages
    
    pdf_path = OUTDIR / "experiment_reference.pdf"
    
    with PdfPages(pdf_path) as pdf:
        # Page 1: Networks
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle("Networks Used in Experiments", fontsize=16, fontweight='bold')
        
        networks = {
            "simple_2node": create_simple_network(2),
            "simple_3node": create_simple_network(3),
            "heterogeneous": create_heterogeneous_network(),
        }
        
        for idx, (name, network) in enumerate(networks.items()):
            ax = axes[idx]
            
            # Draw network using networkx
            pos = nx.spring_layout(network, seed=42)
            
            # Get node weights for labels
            node_labels = {n: f"{n}\n(w={network.nodes[n].get('weight', 1.0)})" 
                          for n in network.nodes}
            
            # Draw nodes
            nx.draw_networkx_nodes(network, pos, ax=ax, node_size=1500, 
                                   node_color='lightblue', edgecolors='black')
            nx.draw_networkx_labels(network, pos, ax=ax, labels=node_labels, 
                                    font_size=10)
            
            # Draw edges (skip self-loops for clarity)
            edges_no_self = [(u, v) for u, v in network.edges if u != v]
            nx.draw_networkx_edges(network, pos, ax=ax, edgelist=edges_no_self,
                                   width=2, edge_color='gray')
            
            # Edge labels (bandwidth)
            edge_labels = {(u, v): f"bw={network.edges[u, v].get('weight', 1.0)}" 
                          for u, v in edges_no_self}
            nx.draw_networkx_edge_labels(network, pos, ax=ax, 
                                         edge_labels=edge_labels, font_size=8)
            
            ax.set_title(f"{name}\n({len(network.nodes)} nodes)", fontsize=12)
            ax.axis('off')
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        

        # Pages 2+: CTGs (2 per page)
        all_ctgs = get_all_ctgs()
        
        # Process CTGs in pairs (2 per page)
        for i in range(0, len(all_ctgs), 2):
            fig, axes = plt.subplots(1, 2, figsize=(16, 8))
            fig.suptitle("Conditional Task Graphs (CTGs)", fontsize=16, fontweight='bold')
            
            for j, ax in enumerate(axes):
                ctg_idx = i + j
                if ctg_idx >= len(all_ctgs):
                    ax.axis('off')
                    continue
                
                ctg, metadata = all_ctgs[ctg_idx]
                
                # Use graphviz layout for hierarchical display
                try:
                    pos = nx.nx_agraph.graphviz_layout(ctg, prog="dot")
                except:
                    pos = nx.spring_layout(ctg, seed=42)
                
                # Separate conditional and non-conditional edges
                conditional_edges = [(u, v) for u, v in ctg.edges 
                                    if ctg.edges[u, v].get('conditional', False)]
                non_conditional_edges = [(u, v) for u, v in ctg.edges 
                                        if not ctg.edges[u, v].get('conditional', False)]
                
                # Draw nodes with weights
                node_labels = {n: f"{n}\n(w={ctg.nodes[n].get('weight', 1)})" 
                              for n in ctg.nodes}
                nx.draw_networkx_nodes(ctg, pos, ax=ax, node_size=1200,
                                       node_color='lightyellow', edgecolors='black')
                nx.draw_networkx_labels(ctg, pos, ax=ax, labels=node_labels, 
                                        font_size=9)
                
                # Draw non-conditional edges (solid black)
                if non_conditional_edges:
                    nx.draw_networkx_edges(ctg, pos, ax=ax, 
                                          edgelist=non_conditional_edges,
                                          width=2, edge_color='black',
                                          style='solid', arrows=True,
                                          arrowsize=15, node_size=1200)
                
                # Draw conditional edges (dashed red)
                if conditional_edges:
                    nx.draw_networkx_edges(ctg, pos, ax=ax,
                                          edgelist=conditional_edges,
                                          width=2, edge_color='red',
                                          style='dashed', arrows=True,
                                          arrowsize=15, node_size=1200)
                
                # Title with metadata
                title = f"{metadata['name']}: {metadata['description']}\n"
                title += f"Type: {metadata['ctg_type']} | "
                title += f"Tasks: {metadata['num_tasks']} | "
                title += f"Branches: {metadata['num_branches']} | "
                title += f"Cond. Points: {metadata['num_conditional_points']}"
                ax.set_title(title, fontsize=10)
                ax.axis('off')
            
            # Add legend to first subplot
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], color='black', linewidth=2, linestyle='-', 
                       label='Unconditional'),
                Line2D([0], [0], color='red', linewidth=2, linestyle='--', 
                       label='Conditional'),
            ]
            axes[0].legend(handles=legend_elements, loc='lower left', fontsize=9)
            
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
    
    print(f"Reference PDF saved to: {pdf_path}")


def main():


    
    # Generate reference PDF with all networks and CTGs
    generate_reference_pdf()

    # Collect results
    df = collect_results()
    
    # Save to CSV
    df.to_csv(CSV_PATH, index=False)

    
    # Analyze and plot
    analyze_results(df)
    


if __name__ == "__main__":
    main()

