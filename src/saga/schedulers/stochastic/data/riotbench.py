"""Stochastic RIoTBench loaders: the generative source of truth for both regimes.

Unlike WfCommons, where the deterministic loader reads real traces and the stochastic
layer fits distributions on top, RIoTBench has no traces. Costs are modeled from the
paper's measured per-operator peak rates (Shukla, Chaturvedi, Simmhan, 2017;
arXiv:1701.08530). The random-variable form is therefore the fuller specification, and
the deterministic loader in saga.schedulers.data.riotbench collapses these by drawing
one sample per variable.

Noise is modeled, not trace-fitted: each mean is wrapped in a lognormal with a tunable
coefficient of variation, which keeps values positive and gives the right-skew real task
runtimes have (a task occasionally runs much slower, never in negative time).
"""
from functools import partial
from itertools import product
from typing import Callable, List

import numpy as np

from saga.stochastic import StochasticNetwork, StochasticTaskGraph
from saga.utils.random_variable import RandomVariable, DEFAULT_NUM_SAMPLES

# Per-operator peak sustained throughput on a single-core VM, in messages/second, from
# RIoTBench Section 7.3. Values marked "measured" are stated in the paper; the rest are
# estimates in the paper's stated tiers (most tasks sustain 3,000+ msg/sec; I/O tasks are
# slower). A lower rate is a heavier task. Costs are relative: the CCR sweep renormalizes
# compute against communication, so only the ratios between operators matter.
_PEAK_RATES = {
    # Cheap compute, measured at 68,000 msg/sec (ANN, BLF, RGF, ACC, DAC, KAL).
    "Annotate": 68000.0,        # ANN, measured
    "Annotation": 68000.0,      # ANN, measured (TRAIN's annotate)
    "RangeFilter": 68000.0,     # RGF, measured
    "BloomFilter": 68000.0,     # BLF, measured
    "KalmanFilter": 68000.0,    # KAL, measured
    "DistinctCount": 68000.0,   # DAC, measured
    "Join": 68000.0,            # accumulator-style join, ACC tier, estimate
    # Parse and moderate compute, paper's "most tasks 3,000+" tier, estimates.
    "SenMLParse": 3000.0,       # SML, faster than XML(310) per paper, estimate
    "CsvToSenML": 3000.0,       # C2S, estimate
    "Interpolation": 3000.0,    # INP, estimate
    "Average": 3000.0,          # AVG, higher CPU per paper but no rate given, estimate
    "SlidingLinearReg": 3000.0, # SLR, estimate
    "DecisionTree": 3000.0,     # DTC classify (WEKA), estimate
    "MultiVarLinearReg": 3000.0,# MLR (WEKA), estimate
    "ErrorEstimate": 10000.0,   # simple residual, estimate
    # Expensive compute, measured.
    "GroupViz": 25.0,           # ACC+PLT+ZIP meta-task, bottlenecked by PLT=25, measured
    "DecisionTreeTrain": 50.0,  # DTT (WEKA training), measured
    "MultiVarLinearRegTrain": 70.0,  # MLT (WEKA training), measured
    # I/O bound. TableRead is the measured 1 msg/min full-table scan (ATR); the rest are
    # Azure/broker I/O estimated slower than compute but far faster than the scan.
    "TableRead": 1.0 / 60.0,    # ATR = 1 msg/min, measured (TRAIN's dominant cost)
    "AzureTableInsert": 100.0,  # ATI, estimate
    "BlobUpload": 100.0,        # ABU, estimate
    "BlobWrite": 100.0,         # blob write, estimate
    "BlobRead": 100.0,          # ABD, estimate
    "MQTTPublish": 1000.0,      # MQP, estimate
    "MQTTSubscribe": 1000.0,    # MQS, estimate
    # Pure source and sink stubs contribute negligible compute.
    "Source": 1e9,
    "TimerSource": 1e9,
    "VirtualSource": 1e9,
    "Sink": 1e9,
}

DEFAULT_CV = 0.3  # coefficient of variation for the modeled lognormal noise


def gaussian(min_value: float, max_value: float) -> float:
    """Sample a clamped Gaussian, used to pick a per-instance base message size."""
    value = np.random.normal(
        loc=(min_value + max_value) / 2, scale=(max_value - min_value) / 3
    )
    return max(min_value, min(max_value, value))


def _const(value: float) -> RandomVariable:
    """A degenerate random variable at a single value."""
    return RandomVariable(samples=[value])


def lognormal_rv(mean: float, cv: float, num_samples: int) -> RandomVariable:
    """A lognormal random variable with the given mean and coefficient of variation.

    Solving for the underlying normal's parameters from (mean, cv) keeps the sampled
    mean at `mean` while giving the positive, right-skewed shape of real runtimes.
    """
    if mean <= 0.0 or cv <= 0.0:
        return _const(max(mean, 0.0))
    sigma = np.sqrt(np.log(1.0 + cv * cv))
    mu = np.log(mean) - 0.5 * sigma * sigma
    samples = np.random.lognormal(mean=mu, sigma=sigma, size=num_samples)
    return RandomVariable(samples=samples.tolist())


def _cost_rv(name: str, cv: float, num_samples: int) -> RandomVariable:
    """Per-message compute cost for an operator, drawn around its peak-rate mean."""
    rate = _PEAK_RATES.get(name, 3000.0)
    return lognormal_rv(1000.0 / rate, cv, num_samples)


def get_fog_networks(
    num: int,
    num_edges_nodes: int = 16,
    num_fog_nodes: int = 4,
    num_cloud_nodes: int = 1,
    edge_node_cpu: float = 1000.0,
    fog_node_cpu: float = 2800.0,
    cloud_node_cpu: float = 44800.0,
    edge_fog_bw: float = 100,
    fog_cloud_bw: float = 10,
    fog_fog_bw: float = 1000,
    cloud_cloud_bw: float = 10000,
    cv: float = DEFAULT_CV,
    num_samples: int = DEFAULT_NUM_SAMPLES,
) -> List[StochasticNetwork]:
    """Generate tiered edge/fog/cloud networks with stochastic speeds and bandwidths.

    Topology and per-tier compute follow iFogSim (Gupta, Dastjerdi, Ghosh, Buyya, 2017,
    "iFogSim: A Toolkit for Modeling and Simulation of Resource Management Techniques in
    the Internet of Things, Edge and Fog Computing Environments"). Defaults mirror its EEG
    Tractor Beam Game case study: a handful of edge devices under a few fog gateways under
    one cloud, with CPU capacities in MIPS (edge 1000, fog 2800, cloud 44800). iFogSim
    parameterizes links by latency; we map its LAN-fast, WAN-slow hierarchy to bandwidth,
    making the fog-to-cloud WAN hop the communication bottleneck while intra-edge and
    intra-fog LAN hops are fast. This deliberately differs from the RIoTBench paper, which
    ran on a homogeneous single-datacenter cluster: the dataflows are RIoTBench, the fog
    deployment they run on is the iFogSim scenario.

    Node CPU speeds and inter-node link bandwidths carry lognormal noise; only intra-node
    (self-loop) links stay constant.

    Args:
        num: Number of networks to generate.
        num_edges_nodes: Number of edge nodes per network.
        num_fog_nodes: Number of fog nodes per network.
        num_cloud_nodes: Number of cloud nodes per network.
        edge_node_cpu: Mean edge-node CPU capacity (MIPS).
        fog_node_cpu: Mean fog-node CPU capacity (MIPS).
        cloud_node_cpu: Mean cloud-node CPU capacity (MIPS).
        edge_fog_bw: Mean edge-fog bandwidth (Mbps, LAN).
        fog_cloud_bw: Mean fog-cloud bandwidth (Mbps, WAN, the bottleneck).
        fog_fog_bw: Mean fog-fog bandwidth (Mbps, LAN).
        cloud_cloud_bw: Mean cloud-cloud bandwidth (Mbps, intra-datacenter).
        cv: Coefficient of variation of the modeled noise.
        num_samples: Samples per random variable (1 collapses to a deterministic draw).

    Returns:
        A list of stochastic networks.
    """
    networks = []
    for _ in range(num):
        edge_nodes = {f"E{i}" for i in range(num_edges_nodes)}
        fog_nodes = {f"F{i}" for i in range(num_fog_nodes)}
        cloud_nodes = {f"C{i}" for i in range(num_cloud_nodes)}

        # Iterate the name sets in sorted order while drawing samples: set iteration
        # order is PYTHONHASHSEED-dependent, which would otherwise assign the same seeded
        # draws to different node names across runs.
        nodes = (
            [(name, lognormal_rv(edge_node_cpu, cv, num_samples)) for name in sorted(edge_nodes)]
            + [(name, lognormal_rv(fog_node_cpu, cv, num_samples)) for name in sorted(fog_nodes)]
            + [(name, lognormal_rv(cloud_node_cpu, cv, num_samples)) for name in sorted(cloud_nodes)]
        )
        all_node_names = edge_nodes | fog_nodes | cloud_nodes

        edge_edge_bw = edge_fog_bw
        edge_cloud_bw = min(edge_fog_bw, fog_cloud_bw)

        edges = []
        for src, dst in product(sorted(all_node_names), sorted(all_node_names)):
            if src == dst:
                edges.append((src, dst, _const(1e9)))
                continue

            has_edge = src in edge_nodes or dst in edge_nodes
            has_fog = src in fog_nodes or dst in fog_nodes
            has_cloud = src in cloud_nodes or dst in cloud_nodes

            if has_edge and not (has_fog or has_cloud):  # edge to edge
                bw = edge_edge_bw
            elif has_edge and has_fog:  # edge to fog
                bw = edge_fog_bw
            elif has_edge and has_cloud:  # edge to cloud
                bw = edge_cloud_bw
            elif has_fog and not (has_edge or has_cloud):  # fog to fog
                bw = fog_fog_bw
            elif has_fog and has_cloud:  # fog to cloud
                bw = fog_cloud_bw
            elif has_cloud and not (has_edge or has_fog):  # cloud to cloud
                bw = cloud_cloud_bw
            else:
                bw = 0.0

            # Scale by 125 so units are KiloBytes per second. Only intra-node self-loops
            # (handled above) are free; every inter-node tier is a finite, noised link.
            speed = bw * 125
            weight = _const(speed) if bw >= 1e9 else lognormal_rv(speed, cv, num_samples)
            edges.append((src, dst, weight))

        networks.append(StochasticNetwork.create(nodes=nodes, edges=edges))

    return networks


def get_etl_task_graphs(
    num: int,
    get_input_size: Callable[[], float] = partial(gaussian, 500, 1500),
    count_window: int = 10,
    batch_window: int = 10,
    cv: float = DEFAULT_CV,
    num_samples: int = DEFAULT_NUM_SAMPLES,
) -> List[StochasticTaskGraph]:
    """Generate ETL (Extract-Transform-Load) dataflows, RIoTBench Fig. 3a.

    SenMLParse fans one message out into per-observation-type streams (1:N), which the
    range, bloom and interpolation filters carry (1:1) until Join collapses them back to
    one message (N:1); we model the fan-out as a count_window volume factor. Annotate
    forks to a batched Azure table insert (M:1) and to a SenML/MQTT publish path, both
    feeding the logging sink.

    Args:
        num: Number of task graphs to generate.
        get_input_size: Function returning a per-instance base message size (bytes).
        count_window: Fan-out factor for the parse-filter-join stage (SenML 1:N).
        batch_window: Number of messages batched per Azure table insert (M:1).
        cv: Coefficient of variation of the modeled noise.
        num_samples: Samples per random variable (1 collapses to a deterministic draw).

    Returns:
        A list of stochastic task graphs.
    """
    task_graphs = []
    for _ in range(num):
        base = get_input_size()

        def cost(name: str) -> RandomVariable:
            return _cost_rv(name, cv, num_samples)

        def size(mean: float) -> RandomVariable:
            return lognormal_rv(mean, cv, num_samples)

        tasks = [
            ("Source", cost("Source")),
            ("SenMLParse", cost("SenMLParse")),
            ("RangeFilter", cost("RangeFilter")),
            ("BloomFilter", cost("BloomFilter")),
            ("Interpolation", cost("Interpolation")),
            ("Join", cost("Join")),
            ("Annotate", cost("Annotate")),
            ("CsvToSenML", cost("CsvToSenML")),
            ("AzureTableInsert", cost("AzureTableInsert")),
            ("MQTTPublish", cost("MQTTPublish")),
            ("Sink", cost("Sink")),
        ]

        dependencies = [
            ("Source", "SenMLParse", size(base)),
            ("SenMLParse", "RangeFilter", size(count_window * base)),
            ("RangeFilter", "BloomFilter", size(count_window * base)),
            ("BloomFilter", "Interpolation", size(count_window * base)),
            ("Interpolation", "Join", size(count_window * base)),
            ("Join", "Annotate", size(base)),
            ("Annotate", "CsvToSenML", size(base)),
            ("Annotate", "AzureTableInsert", size(batch_window * base)),
            ("CsvToSenML", "MQTTPublish", size(base)),
            ("AzureTableInsert", "Sink", size(base)),  # single ack after a batch insert
            ("MQTTPublish", "Sink", size(base)),  # single sink for logging
        ]

        task_graphs.append(StochasticTaskGraph.create(tasks=tasks, dependencies=dependencies))

    return task_graphs


def get_stats_task_graphs(
    num: int,
    get_input_size: Callable[[], float] = partial(gaussian, 500, 1500),
    count_window: int = 10,
    plot_window: int = 10,
    cv: float = DEFAULT_CV,
    num_samples: int = DEFAULT_NUM_SAMPLES,
) -> List[StochasticTaskGraph]:
    """Generate STATS (Statistical Summarization) dataflows, RIoTBench Fig. 3b.

    SenMLParse feeds three parallel analytics: an average over a count_window window
    (N:1), a Kalman filter into a sliding linear regression, and a distinct-approximate
    count. Their outputs are grouped, plotted and zipped by the GroupViz meta-task (the
    paper's ACC+PLT+ZIP collapsed into one node), which batches plot_window messages into
    a single file uploaded to blob storage.

    Args:
        num: Number of task graphs to generate.
        get_input_size: Function returning a per-instance base message size (bytes).
        count_window: Window size for the average (N:1 aggregation).
        plot_window: Messages batched per GroupViz plot/zip file.
        cv: Coefficient of variation of the modeled noise.
        num_samples: Samples per random variable (1 collapses to a deterministic draw).

    Returns:
        A list of stochastic task graphs.
    """
    task_graphs = []
    for _ in range(num):
        base = get_input_size()
        plot_mean = plot_window * base  # accumulated time-series zipped into one file

        def cost(name: str) -> RandomVariable:
            return _cost_rv(name, cv, num_samples)

        def size(mean: float) -> RandomVariable:
            return lognormal_rv(mean, cv, num_samples)

        tasks = [
            ("Source", cost("Source")),
            ("SenMLParse", cost("SenMLParse")),
            ("Average", cost("Average")),
            ("KalmanFilter", cost("KalmanFilter")),
            ("DistinctCount", cost("DistinctCount")),
            ("SlidingLinearReg", cost("SlidingLinearReg")),
            ("GroupViz", cost("GroupViz")),
            ("BlobUpload", cost("BlobUpload")),
            ("Sink", cost("Sink")),
        ]

        dependencies = [
            ("Source", "SenMLParse", size(base)),
            ("SenMLParse", "Average", size(count_window * base)),  # averages a window
            ("SenMLParse", "KalmanFilter", size(base)),
            ("SenMLParse", "DistinctCount", size(base)),
            ("KalmanFilter", "SlidingLinearReg", size(base)),
            ("Average", "GroupViz", size(base)),  # N:1, one aggregate out
            ("SlidingLinearReg", "GroupViz", size(base)),
            ("DistinctCount", "GroupViz", size(base)),
            ("GroupViz", "BlobUpload", size(plot_mean)),
            ("BlobUpload", "Sink", size(base)),  # single upload ack
        ]

        task_graphs.append(StochasticTaskGraph.create(tasks=tasks, dependencies=dependencies))

    return task_graphs


def get_train_task_graphs(
    num: int,
    get_input_size: Callable[[], float] = partial(gaussian, 500, 1500),
    train_window: int = 1000,
    cv: float = DEFAULT_CV,
    num_samples: int = DEFAULT_NUM_SAMPLES,
) -> List[StochasticTaskGraph]:
    """Generate TRAIN (Model Training) dataflows, RIoTBench Fig. 3c.

    A timer periodically triggers a training run. TableRead scans the table for the rows
    inserted since the last run (the paper's dominant cost: a full-table scan at 1 msg/min
    over a batch of ~1000 rows). That batch trains a multi-variate linear regression
    directly and, after annotation, a decision tree. Both model files are written to blob
    storage and their URLs published to MQTT.

    Args:
        num: Number of task graphs to generate.
        get_input_size: Function returning a per-instance base message size (bytes).
        train_window: Rows fetched and trained on per run (N:1 aggregation).
        cv: Coefficient of variation of the modeled noise.
        num_samples: Samples per random variable (1 collapses to a deterministic draw).

    Returns:
        A list of stochastic task graphs.
    """
    task_graphs = []
    for _ in range(num):
        base = get_input_size()
        batch_mean = train_window * base  # rows fetched per training run

        def cost(name: str) -> RandomVariable:
            return _cost_rv(name, cv, num_samples)

        def size(mean: float) -> RandomVariable:
            return lognormal_rv(mean, cv, num_samples)

        tasks = [
            ("TimerSource", cost("TimerSource")),
            ("TableRead", cost("TableRead")),
            ("Annotation", cost("Annotation")),
            ("MultiVarLinearRegTrain", cost("MultiVarLinearRegTrain")),
            ("DecisionTreeTrain", cost("DecisionTreeTrain")),
            ("BlobWrite", cost("BlobWrite")),
            ("MQTTPublish", cost("MQTTPublish")),
            ("Sink", cost("Sink")),
        ]

        dependencies = [
            ("TimerSource", "TableRead", size(base)),  # trigger
            ("TableRead", "Annotation", size(batch_mean)),
            ("TableRead", "MultiVarLinearRegTrain", size(batch_mean)),
            ("Annotation", "DecisionTreeTrain", size(batch_mean)),
            ("DecisionTreeTrain", "BlobWrite", size(base)),  # N:1, one model file
            ("MultiVarLinearRegTrain", "BlobWrite", size(base)),  # one model file
            ("BlobWrite", "MQTTPublish", size(base)),  # model URLs
            ("MQTTPublish", "Sink", size(base)),
        ]

        task_graphs.append(StochasticTaskGraph.create(tasks=tasks, dependencies=dependencies))

    return task_graphs


def get_predict_task_graphs(
    num: int,
    get_input_size: Callable[[], float] = partial(gaussian, 500, 1500),
    count_window: int = 10,
    cv: float = DEFAULT_CV,
    num_samples: int = DEFAULT_NUM_SAMPLES,
) -> List[StochasticTaskGraph]:
    """Generate PRED (Predictive Analytics) dataflows, RIoTBench Fig. 3d.

    Two sources: MQTTSubscribe receives model-update notifications and reads the new
    models from blob storage, while Source streams pre-processed messages. After parsing,
    each message forks to a decision-tree classifier and a multi-variate linear
    regression. The regression's prediction is compared against a moving average by
    ErrorEstimate. Classes, values and errors are published to MQTT. We add a virtual
    source so the dataflow has a single entry point.

    Args:
        num: Number of task graphs to generate.
        get_input_size: Function returning a per-instance base message size (bytes).
        count_window: Window size for the moving average (N:1 aggregation).
        cv: Coefficient of variation of the modeled noise.
        num_samples: Samples per random variable (1 collapses to a deterministic draw).

    Returns:
        A list of stochastic task graphs.
    """
    task_graphs = []
    for _ in range(num):
        base = get_input_size()

        def cost(name: str) -> RandomVariable:
            return _cost_rv(name, cv, num_samples)

        def size(mean: float) -> RandomVariable:
            return lognormal_rv(mean, cv, num_samples)

        tasks = [
            ("VirtualSource", _const(1e-9)),  # unify the two real sources into one entry
            ("Source", cost("Source")),
            ("MQTTSubscribe", cost("MQTTSubscribe")),
            ("BlobRead", cost("BlobRead")),
            ("SenMLParse", cost("SenMLParse")),
            ("DecisionTree", cost("DecisionTree")),
            ("MultiVarLinearReg", cost("MultiVarLinearReg")),
            ("Average", cost("Average")),
            ("ErrorEstimate", cost("ErrorEstimate")),
            ("MQTTPublish", cost("MQTTPublish")),
            ("Sink", cost("Sink")),
        ]

        dependencies = [
            ("VirtualSource", "Source", _const(1e-9)),
            ("VirtualSource", "MQTTSubscribe", _const(1e-9)),
            ("Source", "SenMLParse", size(base)),
            ("MQTTSubscribe", "BlobRead", size(base)),  # model-update notification
            ("BlobRead", "DecisionTree", size(base)),  # updated model
            ("BlobRead", "MultiVarLinearReg", size(base)),  # updated model
            ("SenMLParse", "DecisionTree", size(base)),
            ("SenMLParse", "MultiVarLinearReg", size(base)),
            ("SenMLParse", "Average", size(count_window * base)),  # moving-average window
            ("MultiVarLinearReg", "ErrorEstimate", size(base)),
            ("Average", "ErrorEstimate", size(base)),  # N:1, one moving average out
            ("ErrorEstimate", "MQTTPublish", size(base)),
            ("DecisionTree", "MQTTPublish", size(base)),
            ("MQTTPublish", "Sink", size(base)),
        ]

        task_graphs.append(StochasticTaskGraph.create(tasks=tasks, dependencies=dependencies))

    return task_graphs
