# Benchmarking Experiments

## Normal Benchmarking
To run benchmarking experiments performed in the paper [REMOVED FOR ANONYMITY], run the following command:
```bash
python exp_benchmarking.py
```
This will prepare the datasets, run the experiments, and generate the plots in the `./data`, `./results`, and `./plots` directories, respectively.

## Parametric Scheduler Benchmarking
To run benchmarking experiments like those from the paper [REMOVED FOR ANONYMITY], run the following command:
```bash
python exp_parametric.py run --datadir ./data/parametric/ --out ./results/parametric/batch0.csv --trim 10 --batch 0 --batches 1
python post_parametric_agg.py 
python post_parametric.py
```

The above commands runs a single batch of experiments, trimming each of the datasets to just 10 instances.
Because so many evaluations need to be performed for full benchmarks, the command supports batching.
Together with the slurm script ``exp_parametric.sl``, the experiments can be run on a cluster so that different batches can be run in parallel.

