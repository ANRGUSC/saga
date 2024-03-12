from saga.utils.random_variable import RandomVariable
import numpy as np
import matplotlib.pyplot as plt

def main():
    ccr = 10 # Communication to Computation Ratio
    num_samples = 10000
    rv_comm = RandomVariable(samples=np.random.normal(ccr, ccr*1/3, num_samples))
    rv_comp = RandomVariable(samples=np.random.normal(1, 1/3, num_samples))
    rv_ccr = rv_comm / rv_comp

    rv_std = rv_ccr.std()
    rc_ccr_trimmed = RandomVariable(
        samples=rv_ccr.samples[(rv_ccr.samples > rv_ccr.mean() - 3 * rv_std) & (rv_ccr.samples < rv_ccr.mean() + 3 * rv_std)])

    print("CCR: ", ccr)
    print("Mean CCR: ", rv_ccr.mean())

    # Plot distributions
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].hist(rv_comm.samples, bins=20, density=True)
    ax[0].set_title("Communication Time")
    ax[1].hist(rv_comp.samples, bins=20, density=True)
    ax[1].set_title("Computation Time")
    ax[2].hist(rc_ccr_trimmed.samples, bins=20, density=True)
    ax[2].set_title("Communication to Computation Ratio")
    plt.show()

def other():
    ccr = 2
    ccr_std = 1/3
    num_samples = 10000
    rv_ccr = RandomVariable(samples=np.random.normal(ccr, ccr_std, num_samples))
    rv_comp = RandomVariable(samples=np.random.normal(1, 1/3, num_samples))
    rv_comm = rv_ccr * rv_comp

    print("CCR: ", ccr)
    print("Mean CCR: ", rv_ccr.mean())

    # Plot distributions
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].hist(rv_comm.samples, bins=20, density=True)
    ax[0].set_title("Communication Time")
    ax[1].hist(rv_comp.samples, bins=20, density=True)
    ax[1].set_title("Computation Time")
    ax[2].hist(rv_ccr.samples, bins=20, density=True)
    ax[2].set_title("Communication to Computation Ratio")
    plt.show()

    rv_ccr_2 = RandomVariable(samples=rv_ccr.sample(num_samples))
    rv_comm_2 = RandomVariable(samples=rv_comm.sample(num_samples))
    rv_ccr_2 = rv_comm_2 / rv_comp

    print("CCR: ", ccr)
    print("Mean CCR: ", rv_ccr_2.mean())

    # Plot distributions
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].hist(rv_comm_2.samples, bins=20, density=True)
    ax[0].set_title("Communication Time")
    ax[1].hist(rv_comp.samples, bins=20, density=True)
    ax[1].set_title("Computation Time")
    ax[2].hist(rv_ccr_2.samples, bins=20, density=True)
    ax[2].set_title("Communication to Computation Ratio")
    plt.show()
    


if __name__ == "__main__":
    main()
    # other()