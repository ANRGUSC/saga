# Parametric Insertion Bad Example
This is an example where the append-only strategy outperforms the insertion strategy described in the paper [Parameterized Task Graph Scheduling Algorithm for Comparing Algorithmic Components](https://arxiv.org/abs/2403.07112).
We include this example because it is counter-intuitive that the insertion strategy does not always outperform the append-only strategy.
The reason for this is that other peculiarities of the scheduling algorithm can lead to a situation where the append-only strategy is more efficient by coincidence.
