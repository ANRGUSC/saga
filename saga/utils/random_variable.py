import numpy as np
from scipy import integrate

class RandomVariable:
    def __init__(self, x: np.ndarray, pdf: np.ndarray):
        self.x = x
        self.pdf = pdf

        # sort x (and pdf by x)
        sort_idx = np.argsort(self.x)
        self.x = self.x[sort_idx]
        self.pdf = self.pdf[sort_idx]

    def sample(self, n: int) -> np.ndarray:
        cdf = np.cumsum(self.pdf)
        cdf /= cdf[-1]
        return np.interp(np.random.rand(n), cdf, self.x)

    @property
    def x_min(self):
        return self.x[0]
    
    @property
    def x_max(self):
        return self.x[-1]

    def __add__(self, other: "RandomVariable") -> "RandomVariable":
        x_min = min(self.x_min, other.x_min)
        x_max = max(self.x_max, other.x_max)
        x_range = np.linspace(x_min, x_max, 100000)
        # interpolate pdfs
        pdf1 = np.interp(x_range, self.x, self.pdf)
        pdf2 = np.interp(x_range, other.x, other.pdf)

        # convolve pdfs
        dx = x_range[1] - x_range[0]
        pdf = np.convolve(pdf1, pdf2, mode='same') * dx

        return RandomVariable(x_range, pdf)
    
    def __sub__(self, other: "RandomVariable") -> "RandomVariable":
        return self.__add__(-other)
    
    def __neg__(self):
        return RandomVariable(-self.x[::-1], self.pdf[::-1])
    
    def __mul__(self, other: "RandomVariable") -> "RandomVariable":
        # TODO: I would rather not use the histogram method, but my attempts at 
        # integrating the pdfs have failed so far.
        num_samples = 1000000
        samples1 = self.sample(num_samples)
        samples2 = other.sample(num_samples)
        product_samples = samples1 * samples2

        # Compute the histogram of the product samples to approximate the pdf
        pdf, edges = np.histogram(product_samples, bins='auto', density=True)

        # Calculate the x values corresponding to the pdf
        x_range = (edges[:-1] + edges[1:]) / 2

        return RandomVariable(x_range, pdf)

    def __truediv__(self, other: "RandomVariable") -> "RandomVariable":
        num_samples = 1000000
        samples1 = self.sample(num_samples)
        samples2 = other.sample(num_samples)
        product_samples = samples1 / samples2

        # Compute the histogram of the product samples to approximate the pdf
        pdf, edges = np.histogram(product_samples, bins='auto', density=True)

        # Calculate the x values corresponding to the pdf
        x_range = (edges[:-1] + edges[1:]) / 2

        return RandomVariable(x_range, pdf)
    
    
    @staticmethod
    def max(*rvs: "RandomVariable") -> "RandomVariable":
        if not all(isinstance(rv, RandomVariable) for rv in rvs):
            raise TypeError("All arguments must be RandomVariable instances")
        x_min = min(rv.x_min for rv in rvs)
        x_max = max(rv.x_max for rv in rvs)
        x_range = np.linspace(x_min, x_max, 100000)
        dx = x_range[1] - x_range[0]
        # cdf1 = np.cumsum(rv.pdf) * dx
        pdfs = [np.interp(x_range, rv.x, rv.pdf) for rv in rvs]
        cdfs = [np.cumsum(pdf) * dx for pdf in pdfs]
        pdf_max = np.zeros_like(x_range)
        for pdf, cdf in zip(pdfs, cdfs):
            pdf_max += pdf * cdf
        return RandomVariable(x_range, pdf_max)
    
    def expectation(self):
        return np.sum(self.x * self.pdf)

    def mean(self):
        return self.expectation()

    def variance(self):
        return np.sum((self.x - self.mean())**2 * self.pdf)

    def var(self):
        return self.variance()
    
    def std(self):
        return np.sqrt(self.variance())