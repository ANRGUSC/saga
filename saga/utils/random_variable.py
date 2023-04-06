from functools import lru_cache
from typing import Callable, Tuple, Union
import numpy as np
from scipy import integrate
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV

class RandomVariable:
    DEFAULT_NUM_SAMPLES = 100000
    def __init__(self, samples: np.ndarray, num_samples: int = DEFAULT_NUM_SAMPLES) -> None:
        self.samples = samples
        self.num_samples = num_samples

    @staticmethod
    def from_pdf(x: np.ndarray, pdf: np.ndarray, num_samples: int = DEFAULT_NUM_SAMPLES) -> "RandomVariable":
        """Create a random variable from a probability density function.
        
        Args:
            x (np.ndarray): The x values of the pdf.
            pdf (np.ndarray): The probability density function.

        Returns:
            RandomVariable: The random variable.
        """

        cdf = np.cumsum(pdf)
        cdf /= cdf[-1]
        samples = np.interp(np.random.rand(num_samples), cdf, x)
        return RandomVariable(samples)
    
    @lru_cache(maxsize=1)
    def get_histogram(self, bins: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """Get the histogram of the samples.
        
        Args:
            bins (int, optional): The number of bins. Defaults to 100.

        Returns:
            np.ndarray: The histogram.
        """
        return np.histogram(self.samples, bins=bins)
    
    @property
    def pdf(self) -> np.ndarray:
        """The probability density function evaluated at across the range of samples.
        
        Returns:
            np.ndarray: The pdf.
        """
        hist, bin_edges = self.get_histogram()
        _pdf = hist / hist.sum()
        # rescale the pdf so that it integrates to 1
        _pdf /= integrate.trapz(_pdf, bin_edges[:-1])
        return _pdf
    
    @property
    def x(self, bins: int = 100) -> np.ndarray:
        """The x values of the pdf.
        
        Returns:
            np.ndarray: The x values.
        """
        hist, bin_edges = self.get_histogram(bins)
        return bin_edges[:-1]
    
    def sample(self, num_samples: int = 1) -> Union[float, np.ndarray]:
        """Sample from the random variable.
        
        Args:
            num_samples (int, optional): The number of samples. Defaults to 1.

        Returns:
            Union[float, np.ndarray]: The samples.
        """
        hist, bin_edges = self.get_histogram()
        # Select bins based on the histogram's probabilities
        selected_bins = np.random.choice(bin_edges[:-1], num_samples, p=hist / hist.sum())

        # Sample a value uniformly within each selected bin
        bin_width = bin_edges[1] - bin_edges[0]
        new_points = np.random.rand(num_samples) * bin_width + selected_bins
        return new_points
        

    @property
    def x_min(self):
        return min(self.samples)
    
    @property
    def x_max(self):
        return max(self.samples)

    def __add__(self, other: Union["RandomVariable", float, int]) -> "RandomVariable":
        if isinstance(other, (float, int)):
            return RandomVariable(self.samples + other, self.num_samples)
        
        # add the samples of the two random variables
        if self.num_samples >= other.num_samples:
            self_samples = self.samples
            other_samples = np.random.choice(other.samples, self.num_samples, replace=True)
        else:
            self_samples = np.random.choice(self.samples, other.num_samples, replace=True)
            other_samples = other.samples
        samples = self_samples + other_samples
        return RandomVariable(samples, self.num_samples)
    
    def __radd__(self, other: Union["RandomVariable", float, int]) -> "RandomVariable":
        return self.__add__(other)
    
    def __sub__(self, other: Union["RandomVariable", float, int]) -> "RandomVariable":
        return self.__add__(-other)
    
    def __rsub__(self, other: Union["RandomVariable", float, int]) -> "RandomVariable":
        return -self.__add__(-other)
    
    def __neg__(self):
        return RandomVariable(-self.samples, self.num_samples)
    
    def __mul__(self, other: Union["RandomVariable", float, int]) -> "RandomVariable":
        if isinstance(other, (float, int)):
            return RandomVariable(self.samples * other, self.num_samples)
        
        # multiply the samples of the two random variables
        if self.num_samples >= other.num_samples:
            self_samples = self.samples
            other_samples = np.concatenate([other.samples, np.random.choice(other.samples, self.num_samples - other.num_samples, replace=True)])
        else:
            self_samples = np.concatenate([other.samples, np.random.choice(other.samples, self.num_samples - other.num_samples, replace=True)])
            other_samples = other.samples
        samples = self_samples * other_samples
        return RandomVariable(samples, self.num_samples)
    
    def __rmul__(self, other: Union["RandomVariable", float, int]) -> "RandomVariable":
        return self.__mul__(other)

    def __truediv__(self, other: Union["RandomVariable", int, float]) -> "RandomVariable":
        if isinstance(other, (float, int)):
            return RandomVariable(self.samples / other, self.num_samples)
        
        # divide the samples of the two random variables
        if self.num_samples >= other.num_samples:
            self_samples = self.samples
            other_samples = np.concatenate([other.samples, np.random.choice(other.samples, self.num_samples - other.num_samples, replace=True)])
        else:
            self_samples = np.concatenate([other.samples, np.random.choice(other.samples, self.num_samples - other.num_samples, replace=True)])
            other_samples = other.samples
        samples = self_samples / other_samples
        return RandomVariable(samples, self.num_samples)
    
    def __rtruediv__(self, other: Union["RandomVariable", int, float]) -> "RandomVariable":
        if isinstance(other, (float, int)):
            return RandomVariable(other / self.samples, self.num_samples)
        
        # divide the samples of the two random variables
        if self.num_samples >= other.num_samples:
            self_samples = self.samples
            other_samples = np.concatenate([other.samples, np.random.choice(other.samples, self.num_samples - other.num_samples, replace=True)])
        else:
            self_samples = np.concatenate([other.samples, np.random.choice(other.samples, self.num_samples - other.num_samples, replace=True)])
            other_samples = other.samples
        samples = other_samples / self_samples
        return RandomVariable(samples, self.num_samples)
    
    
    @staticmethod
    def max(*rvs: Union["RandomVariable", float, int]) -> "RandomVariable":
        """Get the maximum of the random variables.
        
        Args:
            rvs (Union[RandomVariable, float, int]): The random variables.

        Returns:
            RandomVariable: The maximum of the random variables.
        """
        max_num_samples = max(rv.num_samples for rv in rvs)
        all_samples = [np.concatenate([rv.samples, np.random.choice(rv.samples, max_num_samples - rv.num_samples, replace=True)]) for rv in rvs]
        samples = np.max(all_samples, axis=0)
        return RandomVariable(samples, max_num_samples)
    
    def expectation(self):
        return np.mean(self.samples)

    def mean(self):
        return self.expectation()

    def variance(self):
        return np.var(self.samples)

    def var(self):
        return self.variance()
    
    def std(self):
        return np.sqrt(self.variance())