import numpy as np
from scipy.stats import norm, uniform, beta
from saga.utils.random_variable import RandomVariable
import pytest
import random

# set seed for reproducibility
random.seed(0)
np.random.seed(0)
n_samples = 1000000

# Helper function to compare distributions via sample means and variances
def compare_distributions(dist1_samples, dist2_samples, atol=0.01, rtol=0.1):
    """Compares two sets of samples by checking their means and variances."""
    mean1, mean2 = np.mean(dist1_samples), np.mean(dist2_samples)
    var1, var2 = np.var(dist1_samples), np.var(dist2_samples)
    assert np.isclose(mean1, mean2, atol=atol, rtol=rtol), f"Mean mismatch: {mean1} vs {mean2}"
    assert np.isclose(var1, var2, atol=atol, rtol=rtol), f"Variance mismatch: {var1} vs {var2}"

@pytest.mark.parametrize("mode", ['norm', 'uniform', 'beta'])
def test_sum(mode: str):
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
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    
    dist1 = RandomVariable.from_pdf(x, pdf1)
    dist2 = RandomVariable.from_pdf(x, pdf2)
    dist3 = dist1 + dist2

    samples1 = dist1.sample(n_samples)
    samples2 = dist2.sample(n_samples)
    samples_add = samples1 + samples2
    samples_direct = dist3.sample(n_samples)

    # Compare the distributions
    compare_distributions(samples_add, samples_direct)

@pytest.mark.parametrize("mode", ['norm', 'uniform', 'beta'])
def test_max(mode: str):
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
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    
    dist1 = RandomVariable.from_pdf(x, pdf1)
    dist2 = RandomVariable.from_pdf(x, pdf2)
    dist3 = RandomVariable.max(dist1, dist2)

    samples1 = dist1.sample(n_samples)
    samples2 = dist2.sample(n_samples)
    samples_max = np.maximum(samples1, samples2)
    samples_direct = dist3.sample(n_samples)

    # Compare the distributions
    compare_distributions(samples_max, samples_direct)

@pytest.mark.parametrize("mode", ['norm', 'uniform', 'beta'])
def test_sub(mode: str):
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
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    dist1 = RandomVariable.from_pdf(x, pdf1)
    dist2 = RandomVariable.from_pdf(x, pdf2)
    dist3 = dist1 - dist2

    samples1 = dist1.sample(n_samples)
    samples2 = dist2.sample(n_samples)
    samples_sub = samples1 - samples2
    samples_direct = dist3.sample(n_samples)

    # Compare the distributions
    compare_distributions(samples_sub, samples_direct)

@pytest.mark.parametrize("mode", ['norm', 'uniform', 'beta'])
def test_mul(mode: str):
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
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    dist1 = RandomVariable.from_pdf(x, pdf1)
    dist2 = RandomVariable.from_pdf(x, pdf2)
    dist3 = dist1 * dist2

    samples1 = dist1.sample(n_samples)
    samples2 = dist2.sample(n_samples)
    samples_mul = samples1 * samples2
    samples_direct = dist3.sample(n_samples)

    # Compare the distributions
    compare_distributions(samples_mul, samples_direct)

@pytest.mark.parametrize("mode", ['norm', 'uniform', 'beta'])
def test_div(mode: str):
    x = np.linspace(-10, 10, 100000)
    if mode == 'norm':
        pdf1 = norm.pdf(x, loc=2, scale=1)
        pdf2 = norm.pdf(x, loc=10, scale=2)
    elif mode == 'uniform':
        pdf1 = uniform.pdf(x, loc=2, scale=2)
        pdf2 = uniform.pdf(x, loc=1, scale=2)
    elif mode == 'beta':
        pdf1 = beta.pdf(x, a=2, b=2)
        pdf2 = beta.pdf(x, a=10, b=5)
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    dist1 = RandomVariable.from_pdf(x, pdf1)
    dist2 = RandomVariable.from_pdf(x, pdf2)
    dist3 = dist1 / dist2

    samples1 = dist1.sample(n_samples)
    samples2 = dist2.sample(n_samples)
    samples_div = samples1 / samples2
    samples_direct = dist3.sample(n_samples)

    # Compare the distributions
    compare_distributions(samples_div, samples_direct)
