from itertools import combinations
from math import factorial
from pprint import pprint
import numpy as np
from typing import Callable, Dict, Iterable, List
from analyze import load_data


def all_subsets(iterable: Iterable[str]) -> Iterable[List[str]]:
    """Return all subsets of a given iterable.

    Args:
        iterable (Iterable[str]): iterable

    Returns:
        Iterable[List[str]]: all subsets of iterable
    """
    for i in range(len(iterable) + 1):
        for subset in combinations(iterable, i):
            yield list(subset)

def some_subsets(iterable: Iterable[str]) -> Iterable[List[str]]:
    """Return a random set of 100 subsets for each size of a given iterable.

    Args:
        iterable (Iterable[str]): iterable

    Returns:
        Iterable[List[str]]: random subsets of iterable
    """
    for i in range(1, len(iterable) + 1):
        # random sample of 100 subsets of size i
        for j in range(100):
            subset = np.random.choice(iterable, size=i, replace=False)
            yield list(subset)

def shapley_values(all_members: List[str],
                   value_function: Callable[[Iterable[str]], float]) -> Dict[str, float]:
    """Calculate the Shapley values for a given set of members.

    Args:
        all_members (List[str]): list of all members
        value_function (Callable[[Iterable[str]], float]): value function

    Returns:
        Dict[str, float]: Shapley values
    """
    # initialize dictionary of shapley values
    shapley_values = {member: 0.0 for member in all_members}
    for member in all_members:
        # iterate over all subsets of all_members that do not contain member
        num_subsets = factorial(len(all_members) - 1)
        for i, coalition in enumerate(some_subsets([m for m in all_members if m != member])):
            # compute value of subset
            subset_value = value_function(coalition)
            # compute value of subset with user
            subset_with_user_value = value_function([*coalition, member])
            # compute marginal contribution of user
            marginal_contribution = subset_with_user_value - subset_value
            # update user value
            weight = factorial(len(coalition)) * factorial(len(all_members) - len(coalition) - 1) / factorial(len(all_members))
            shapley_values[member] += weight * marginal_contribution
        print(f"{member}: {shapley_values[member]:0.4f}")
    return shapley_values

from typing import List, Callable, Dict
import random

def shapley_sampling(members: List[str], value_function: Callable[[List[str]], float], n_samples: int = 1000) -> Dict[str, float]:
    n = len(members)
    shapley_values = {member: 0.0 for member in members}

    # Loop through each sample
    for _ in range(n_samples):
        # Randomly sample a coalition size
        k = random.randint(1, n)

        # Randomly sample k members from the list to form a coalition
        coalition = random.sample(members, k)

        # Compute the value of the coalition
        coalition_value = value_function(coalition)

        # Loop through each member in the coalition to update their Shapley value
        for member in coalition:
            # Remove the member to form a sub-coalition
            sub_coalition = [m for m in coalition if m != member]
            if not sub_coalition:
                continue

            # Compute the value of the sub-coalition
            sub_coalition_value = value_function(sub_coalition)

            # Compute the marginal contribution of the member
            marginal_contribution = coalition_value - sub_coalition_value

            # Update the Shapley value of the member
            shapley_values[member] += marginal_contribution / n_samples

    return shapley_values


def main():
    df_results = load_data()
    # group by dataset/scheduler and number the rows (order matters because it represents the instance the scheduler was run on))
    df_results["trace"] = df_results.groupby(["dataset", "scheduler"]).cumcount()

    # group by trace and confirm there is at least one makespan_ratio == 1.0
    # for each trace
    trace_validity = df_results.groupby(["dataset", "trace"]).apply(lambda x: np.any(x["makespan_ratio"] == 1.0))
    print(f"All traces valid: {np.all(trace_validity)}")

    def value(coalition: Iterable[str]) -> float:
        """Shapley value fucntion.

        Args:
            coalition (Iterable[str]): set of schedulers

        Returns:
            float: average (over traces) of the minumum makespan ratio for the given coalition
        """
        # get the minimum makespan ratio for each trace
        # then average over all traces
        df_schedulers = df_results[df_results["scheduler"].isin(coalition)]
        df_schedulers = df_schedulers.drop(columns=["scheduler"])
        val = df_schedulers.groupby(["dataset", "trace"]).min().max()
        return 1 / float(val["makespan_ratio"])

    pprint({
        "DuplexScheduler": value(["DuplexScheduler"]),
        "MinMinScheduler": value(["MinMinScheduler"]),
        "MaxMinScheduler": value(["MaxMinScheduler"]),
        "Hybrid": value(["MinMinScheduler", "MaxMinScheduler", "DuplexScheduler"]),
    }, sort_dicts=False)

    # get the shapley values for each scheduler
    # shapley = shapley_values(list(df_results["scheduler"].unique()), value)
    shapley = shapley_sampling(list(df_results["scheduler"].unique()), value, n_samples=1000)
    for scheduler, val in sorted(shapley.items(), key=lambda x: x[1], reverse=True):
        print(f"{scheduler}: {val}")


if __name__ == "__main__":
    main()

