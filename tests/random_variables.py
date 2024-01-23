import pandas as pd
from saga.utils.random_variable import RandomVariable
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm, uniform, beta
import pathlib 

thisdir = pathlib.Path(__file__).parent.absolute()
    
def test_sum(mode: str = 'norm'):
    x = np.linspace(-5, 5, 100000)
    if mode == 'norm':
        pdf1 = norm.pdf(x, loc=0, scale=1)
        pdf2 = norm.pdf(x, loc=0, scale=2)
    elif mode == 'uniform':
        pdf1 = uniform.pdf(x, loc=-1, scale=2)
        pdf2 = uniform.pdf(x, loc=-1, scale=2)
    elif mode == 'beta':
        pdf1 = beta.pdf(x, a=2, b=2)
        pdf2 = beta.pdf(x, a=2, b=2)
    dist1 = RandomVariable.from_pdf(x, pdf1)
    dist2 = RandomVariable.from_pdf(x, pdf2)
    dist3 = dist1 + dist2
    n_samples = 100000
    samples1 = dist1.sample(n_samples)
    samples2 = dist2.sample(n_samples)
    samples_add = samples1 + samples2
    samples_direct = dist3.sample(n_samples)

    # plot
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    ax[0, 0].hist(samples1, bins=100, density=True)
    ax[0, 0].plot(dist1.x, dist1.pdf, label='dist1')
    ax[0, 1].hist(samples2, bins=100, density=True)
    ax[0, 1].plot(dist2.x, dist2.pdf, label='dist2')
    ax[1, 0].hist(samples_add, bins=100, density=True)
    ax[1, 0].plot(dist3.x, dist3.pdf, label='sum')
    ax[1, 1].hist(samples_direct, bins=100, density=True)
    ax[1, 1].plot(dist3.x, dist3.pdf, label='sum')

    # axes labels
    ax[0, 0].set_title('dist1')
    ax[0, 1].set_title('dist2')
    ax[1, 0].set_title('sum by adding samples')
    ax[1, 1].set_title('sum by sampling from sum')

    # save to output/random_variables/test_sum.png
    path = thisdir / 'output' / 'random_variables' / 'test_sum' / f'{mode}.png'
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)

def test_max(mode: str = 'norm'):
    x = np.linspace(-5, 5, 100000)
    if mode == 'norm':
        pdf1 = norm.pdf(x, loc=0, scale=1)
        pdf2 = norm.pdf(x, loc=0, scale=2)
    elif mode == 'uniform':
        pdf1 = uniform.pdf(x, loc=-1, scale=2)
        pdf2 = uniform.pdf(x, loc=-1, scale=2)
    elif mode == 'beta':
        pdf1 = beta.pdf(x, a=2, b=2)
        pdf2 = beta.pdf(x, a=2, b=2)
    dist1 = RandomVariable.from_pdf(x, pdf1)
    dist2 = RandomVariable.from_pdf(x, pdf2)
    dist3 = RandomVariable.max(dist1, dist2)
    n_samples = 100000
    samples1 = dist1.sample(n_samples)
    samples2 = dist2.sample(n_samples)
    samples_max = np.maximum(samples1, samples2)
    samples_direct = dist3.sample(n_samples)

    # plot
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    ax[0, 0].hist(samples1, bins=100, density=True)
    ax[0, 0].plot(dist1.x, dist1.pdf, label='dist1')
    ax[0, 1].hist(samples2, bins=100, density=True)
    ax[0, 1].plot(dist2.x, dist2.pdf, label='dist2')
    ax[1, 0].hist(samples_max, bins=100, density=True)
    ax[1, 0].plot(dist3.x, dist3.pdf, label='max')
    ax[1, 1].hist(samples_direct, bins=100, density=True)
    ax[1, 1].plot(dist3.x, dist3.pdf, label='max')

    # axes labels
    ax[0, 0].set_title('dist1')
    ax[0, 1].set_title('dist2')
    ax[1, 0].set_title('max by taking max of samples')
    ax[1, 1].set_title('max by sampling from max')

    # save to output/random_variables/test_max.png
    path = thisdir / 'output' / 'random_variables' / 'test_max' / f'{mode}.png'
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path)
    plt.close()

def test_sub(mode: str = 'norm'):
    x = np.linspace(-5, 5, 100000)
    if mode == 'norm':
        pdf1 = norm.pdf(x, loc=0, scale=1)
        pdf2 = norm.pdf(x, loc=0, scale=2)
    elif mode == 'uniform':
        pdf1 = uniform.pdf(x, loc=-1, scale=2)
        pdf2 = uniform.pdf(x, loc=-1, scale=2)
    elif mode == 'beta':
        pdf1 = beta.pdf(x, a=2, b=2)
        pdf2 = beta.pdf(x, a=2, b=2)
    dist1 = RandomVariable.from_pdf(x, pdf1)
    dist2 = RandomVariable.from_pdf(x, pdf2)
    dist3 = dist1 - dist2
    n_samples = 100000
    samples1 = dist1.sample(n_samples)
    samples2 = dist2.sample(n_samples)
    samples_sub = samples1 - samples2
    samples_direct = dist3.sample(n_samples)

    # plot
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    ax[0, 0].hist(samples1, bins=100, density=True)
    ax[0, 0].plot(dist1.x, dist1.pdf, label='dist1')
    ax[0, 1].hist(samples2, bins=100, density=True)
    ax[0, 1].plot(dist2.x, dist2.pdf, label='dist2')
    ax[1, 0].hist(samples_sub, bins=100, density=True)
    ax[1, 0].plot(dist3.x, dist3.pdf, label='sub')
    ax[1, 1].hist(samples_direct, bins=100, density=True)
    ax[1, 1].plot(dist3.x, dist3.pdf, label='sub')

    # axes labels
    ax[0, 0].set_title('dist1')
    ax[0, 1].set_title('dist2')
    ax[1, 0].set_title('sub by subtracting samples')
    ax[1, 1].set_title('sub by sampling from sub')

    # save to output/random_variables/test_sub.png
    path = thisdir / 'output' / 'random_variables' / 'test_sub' / f'{mode}.png'
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path)
    plt.close()

def test_mul(mode: str = 'norm'):
    x = np.linspace(-10, 10, 100000)
    if mode == 'norm':
        pdf1 = norm.pdf(x, loc=2, scale=1)
        pdf2 = norm.pdf(x, loc=2, scale=2)
    elif mode == 'uniform':
        pdf1 = uniform.pdf(x, loc=-1, scale=2)
        pdf2 = uniform.pdf(x, loc=-1, scale=2)
    elif mode == 'beta':
        pdf1 = beta.pdf(x, a=2, b=2)
        pdf2 = beta.pdf(x, a=2, b=2)
    dist1 = RandomVariable.from_pdf(x, pdf1)
    dist2 = RandomVariable.from_pdf(x, pdf2)
    dist3 = dist1 * dist2
    n_samples = 100000
    samples1 = dist1.sample(n_samples)
    samples2 = dist2.sample(n_samples)
    samples_mul = samples1 * samples2
    samples_direct = dist3.sample(n_samples)

    # plot
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    ax[0, 0].hist(samples1, bins=100, density=True)
    ax[0, 0].plot(dist1.x, dist1.pdf, label='dist1')
    ax[0, 1].hist(samples2, bins=100, density=True)
    ax[0, 1].plot(dist2.x, dist2.pdf, label='dist2')
    ax[1, 0].hist(samples_mul, bins=100, density=True)
    ax[1, 0].plot(dist3.x, dist3.pdf, label='mul')
    ax[1, 1].hist(samples_direct, bins=100, density=True)
    ax[1, 1].plot(dist3.x, dist3.pdf, label='mul')

    # axes labels
    ax[0, 0].set_title('dist1')
    ax[0, 1].set_title('dist2')
    ax[1, 0].set_title('mul by multiplying samples')
    ax[1, 1].set_title('mul by sampling from mul')

    # save to output/random_variables/test_mul.png
    path = thisdir / 'output' / 'random_variables' / 'test_mul' / f'{mode}.png'
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path)
    plt.close()

def test_div(mode: str = 'norm'):
    x = np.linspace(-10, 10, 100000)
    if mode == 'norm':
        pdf1 = norm.pdf(x, loc=2, scale=1)
        pdf2 = norm.pdf(x, loc=5, scale=2)
    elif mode == 'uniform':
        pdf1 = uniform.pdf(x, loc=2, scale=2)
        pdf2 = uniform.pdf(x, loc=1, scale=2)
    elif mode == 'beta':
        pdf1 = beta.pdf(x, a=2, b=2)
        pdf2 = beta.pdf(x, a=3, b=5)
    dist1 = RandomVariable.from_pdf(x, pdf1)
    dist2 = RandomVariable.from_pdf(x, pdf2)
    dist3 = dist1 / dist2
    n_samples = 100000
    samples1 = dist1.sample(n_samples)
    samples2 = dist2.sample(n_samples)
    samples_div = samples1 / samples2
    samples_direct = dist3.sample(n_samples)

    df_dist3 = pd.DataFrame({'x': dist3.x, 'pdf': dist3.pdf})

    # plot
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    ax[0, 0].hist(samples1, bins=100, density=True)
    ax[0, 0].plot(dist1.x, dist1.pdf, label='dist1')
    ax[0, 1].hist(samples2, bins=100, density=True)
    ax[0, 1].plot(dist2.x, dist2.pdf, label='dist2')
    ax[1, 0].hist(samples_div, bins=100, density=True)
    ax[1, 0].plot(dist3.x, dist3.pdf, label='div')
    ax[1, 1].hist(samples_direct, bins=100, density=True)
    ax[1, 1].plot(dist3.x, dist3.pdf, label='div')

    # axes labels
    ax[0, 0].set_title('dist1')
    ax[0, 1].set_title('dist2')
    ax[1, 0].set_title('div by dividing samples')
    ax[1, 1].set_title('div by sampling from div')

    # save to output/random_variables/test_div.png
    path = thisdir / 'output' / 'random_variables' / 'test_div' / f'{mode}.png'
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path)
    plt.close()

def main():
    test_sum('norm')
    test_sum('uniform')
    test_sum('beta')

    test_sub('norm')
    test_sub('uniform')
    test_sub('beta')

    test_max('norm')
    test_max('uniform')
    test_max('beta')

    test_mul('norm')
    test_mul('uniform')
    test_mul('beta')

    test_div('norm')
    test_div('uniform')
    test_div('beta')

if __name__ == "__main__":
    main()


