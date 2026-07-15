from functools import cached_property, lru_cache
from typing import List, Literal, Tuple, Union
import numpy as np
from pydantic import BaseModel, Field
from scipy import integrate

DEFAULT_NUM_SAMPLES = 10000


class RandomVariable(BaseModel):
    """A random variable. This class is used to represent a random variable and
    provides methods for sampling from the random variable and computing its
    probability density function.
    """

    samples: List[float] = Field(..., description="The samples of the random variable.")

    @cached_property
    def samples_arr(self) -> np.ndarray:
        """The samples as a numpy array."""
        return np.array(self.samples)

    def __format__(self, format_spec: str) -> str:
        """Format the random variable.

        Args:
            format_spec (str): The format specifier.

        Returns:
            str: The formatted random variable.
        """
        return f"{self.mean():{format_spec}} ± {self.std():{format_spec}}"

    def __round__(self, n: int) -> "RandomVariable":
        """Round the random variable to the given number of decimal places.

        Args:
            n (int): The number of decimal places.

        Returns:
            RandomVariable: The rounded random variable.
        """
        return self

    def __str__(self) -> str:
        """Get the string representation of the random variable."""
        return self.__format__("0.3f")

    def __repr__(self) -> str:
        """Get the string representation of the random variable."""
        return str(self)

    @staticmethod
    def from_pdf(
        x_vals: np.ndarray, pdf: np.ndarray, num_samples: int = DEFAULT_NUM_SAMPLES
    ) -> "RandomVariable":
        """Create a random variable from a probability density function.

        Args:
            x_vals (np.ndarray): The x values of the pdf.
            pdf (np.ndarray): The probability density function.

        Returns:
            RandomVariable: The random variable.
        """
        cdf = np.cumsum(pdf)
        cdf /= cdf[-1]
        samples = np.interp(np.random.rand(num_samples), cdf, x_vals)
        return RandomVariable(samples=samples.tolist())

    @lru_cache(maxsize=1)
    def get_histogram(self, bins: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """Get the histogram of the samples.

        Args:
            bins (int, optional): The number of bins. Defaults to 100.

        Returns:
            np.ndarray: The histogram.
        """
        return np.histogram(self.samples_arr, bins=bins)

    @property
    def pdf(self) -> np.ndarray:
        """The probability density function evaluated at across the range of samples.

        Returns:
            np.ndarray: The pdf.
        """
        hist, bin_edges = self.get_histogram()
        _pdf = hist / hist.sum()
        # rescale the pdf so that it integrates to 1
        _pdf /= integrate.trapezoid(_pdf, bin_edges[:-1])
        return _pdf

    @property
    def x(self, bins: int = 100) -> np.ndarray:  # pylint: disable=invalid-name
        """The x values of the pdf.

        Returns:
            np.ndarray: The x values.
        """
        _, bin_edges = self.get_histogram(bins)
        return bin_edges[:-1]

    def sample(self, num_samples: int = 1) -> np.ndarray:
        """Sample from the random variable.

        Args:
            num_samples (int, optional): The number of samples. Defaults to 1.

        Returns:
            Union[float, np.ndarray]: The samples.
        """
        return np.random.choice(self.samples_arr, size=num_samples, replace=True)

    @property
    def x_min(self):
        """The minimum value of the samples."""
        return min(self.samples_arr)

    @property
    def x_max(self):
        """The maximum value of the samples."""
        return max(self.samples_arr)

    def __add__(self, other: Union["RandomVariable", float, int]) -> "RandomVariable":
        """Add two random variables.

        Args:
            other (Union[RandomVariable, float, int]): The other random variable.

        Returns:
            RandomVariable: The sum of the two random variables.
        """
        if isinstance(other, (float, int)):
            return RandomVariable(samples=(self.samples_arr + other).tolist())

        self_num_samples = len(self.samples_arr)
        other_num_samples = len(other.samples_arr)

        # add the samples of the two random variables
        if self_num_samples >= other_num_samples:
            self_samples = self.samples_arr
            missing_samples = self_num_samples - other_num_samples
            other_samples = np.concatenate(
                [
                    other.samples_arr,
                    np.random.choice(other.samples_arr, missing_samples),
                ]
            )
        else:
            missing_samples = other_num_samples - self_num_samples
            self_samples = np.concatenate(
                [self.samples_arr, np.random.choice(self.samples_arr, missing_samples)]
            )
            other_samples = other.samples_arr
        samples = self_samples + other_samples
        return RandomVariable(samples=samples.tolist())

    def __radd__(self, other: Union["RandomVariable", float, int]) -> "RandomVariable":
        """Add two random variables."""
        return self.__add__(other)

    def __sub__(self, other: Union["RandomVariable", float, int]) -> "RandomVariable":
        """Subtract two random variables."""
        return self.__add__(-other)

    def __rsub__(self, other: Union["RandomVariable", float, int]) -> "RandomVariable":
        """Subtract two random variables."""
        return -self.__add__(-other)

    def __neg__(self):
        """Negate the random variable."""
        return RandomVariable(samples=(-self.samples_arr).tolist())

    def __mul__(self, other: Union["RandomVariable", float, int]) -> "RandomVariable":
        """Multiply two random variables."""
        if isinstance(other, (float, int)):
            return RandomVariable(samples=(self.samples_arr * other).tolist())

        if np.isclose(other.std(), 0):
            if np.isclose(self.std(), 0):
                return RandomVariable(samples=[self.mean() * other.mean()])
            else:
                return RandomVariable(
                    samples=(self.samples_arr * other.mean()).tolist()
                )

        self_num_samples = len(self.samples_arr)
        other_num_samples = len(other.samples_arr)

        # multiply the samples of the two random variables
        if self_num_samples >= other_num_samples:
            self_samples = self.samples_arr
            missing_samples = self_num_samples - other_num_samples
            other_samples = np.concatenate(
                [
                    other.samples_arr,
                    np.random.choice(other.samples_arr, missing_samples),
                ]
            )
        else:
            missing_samples = other_num_samples - self_num_samples
            self_samples = np.concatenate(
                [self.samples_arr, np.random.choice(self.samples_arr, missing_samples)]
            )
            other_samples = other.samples_arr
        samples = self_samples * other_samples
        return RandomVariable(samples=samples.tolist())

    def __rmul__(self, other: Union["RandomVariable", float, int]) -> "RandomVariable":
        """Multiply two random variables."""
        return self.__mul__(other)

    def __truediv__(
        self, other: Union["RandomVariable", int, float]
    ) -> "RandomVariable":
        """Divide two random variables."""
        if isinstance(other, (float, int)):
            return RandomVariable(samples=(self.samples_arr / other).tolist())

        if np.isclose(other.std(), 0):
            if np.isclose(self.std(), 0):
                return RandomVariable(samples=[self.mean() / other.mean()])
            else:
                return RandomVariable(
                    samples=(self.samples_arr / other.mean()).tolist()
                )

        self_num_samples = len(self.samples_arr)
        other_num_samples = len(other.samples_arr)

        # divide the samples of the two random variables
        if self_num_samples >= other_num_samples:
            self_samples = self.samples_arr
            missing_samples = self_num_samples - other_num_samples
            other_samples = np.concatenate(
                [
                    other.samples_arr,
                    np.random.choice(other.samples_arr, missing_samples),
                ]
            )
        else:
            missing_samples = other_num_samples - self_num_samples
            self_samples = np.concatenate(
                [self.samples_arr, np.random.choice(self.samples_arr, missing_samples)]
            )
            other_samples = other.samples_arr
        samples = self_samples / other_samples
        return RandomVariable(samples=samples.tolist())

    def __rtruediv__(
        self, other: Union["RandomVariable", int, float]
    ) -> "RandomVariable":
        """Divide two random variables."""
        other = (
            other
            if isinstance(other, RandomVariable)
            else RandomVariable(samples=[other])
        )
        if np.isclose(self.std(), 0.0):
            if np.isclose(other.std(), 0.0):
                return RandomVariable(samples=[other.mean() / self.mean()])
            else:
                return RandomVariable(
                    samples=(other.samples_arr / self.mean()).tolist()
                )

        if isinstance(other, (float, int)):
            return RandomVariable(samples=(other / np.array(self.samples_arr)).tolist())

        self_num_samples = len(self.samples_arr)
        other_num_samples = len(other.samples_arr)

        # divide the samples of the two random variables
        if self_num_samples >= other_num_samples:
            self_samples = self.samples_arr
            missing_samples = self_num_samples - other_num_samples
            other_samples = np.concatenate(
                [
                    other.samples_arr,
                    np.random.choice(other.samples_arr, missing_samples),
                ]
            )
        else:
            missing_samples = other_num_samples - self_num_samples
            self_samples = np.concatenate(
                [self.samples_arr, np.random.choice(self.samples_arr, missing_samples)]
            )
            other_samples = other.samples_arr
        samples = other_samples / self_samples
        return RandomVariable(samples=samples.tolist())

    @staticmethod
    def max(*rvs: Union["RandomVariable", float, int]) -> "RandomVariable":
        """Get the maximum of the random variables.

        Args:
            rvs (Union[RandomVariable, float, int]): The random variables.

        Returns:
            RandomVariable: The maximum of the random variables.
        """
        _rvs = [
            rv if isinstance(rv, RandomVariable) else RandomVariable(samples=[rv])
            for rv in rvs
        ]
        max_num_samples = max(len(rv.samples_arr) for rv in _rvs)
        all_samples = [
            np.concatenate(
                [
                    rv.samples_arr,
                    np.random.choice(
                        rv.samples_arr,
                        max_num_samples - len(rv.samples_arr),
                        replace=True,
                    ),
                ]
            )
            for rv in _rvs
        ]
        samples = np.max(all_samples, axis=0)
        return RandomVariable(samples=samples.tolist())

    def __lt__(self, other: Union["RandomVariable", float, int]) -> "RandomVariable":
        """Less than comparison."""
        if isinstance(other, (float, int)):
            return RandomVariable(samples=(self.samples_arr < other).tolist())
        return RandomVariable(samples=(self.samples_arr < other.samples_arr).tolist())

    def __le__(self, other: Union["RandomVariable", float, int]) -> "RandomVariable":
        """Less than or equal to comparison."""
        if isinstance(other, (float, int)):
            return RandomVariable(samples=(self.samples_arr <= other).tolist())
        return RandomVariable(samples=(self.samples_arr <= other.samples_arr).tolist())

    def __gt__(self, other: Union["RandomVariable", float, int]) -> "RandomVariable":
        """Greater than comparison."""
        if isinstance(other, (float, int)):
            return RandomVariable(samples=(self.samples_arr > other).tolist())
        return RandomVariable(samples=(self.samples_arr > other.samples_arr).tolist())

    def __ge__(self, other: Union["RandomVariable", float, int]) -> "RandomVariable":
        """Greater than or equal to comparison."""
        if isinstance(other, (float, int)):
            return RandomVariable(samples=(self.samples_arr >= other).tolist())
        return RandomVariable(samples=(self.samples_arr >= other.samples_arr).tolist())

    def expectation(self) -> float:
        """The expectation of the random variable."""
        return float(np.mean(self.samples_arr))

    def mean(self) -> float:
        """The mean of the random variable."""
        return self.expectation()

    def variance(self) -> float:
        """The variance of the random variable."""
        return float(np.var(self.samples_arr))

    def var(self) -> float:
        """The variance of the random variable."""
        return self.variance()

    def std(self) -> float:
        """The standard deviation of the random variable."""
        return np.sqrt(self.variance())


class UniformRandomVariable(RandomVariable):
    """A uniform random variable."""

    name: Literal["UniformRandomVariable"] = "UniformRandomVariable"

    def __init__(
        self, low: float, high: float, num_samples: int = DEFAULT_NUM_SAMPLES
    ) -> None:
        """Initialize a uniform random variable.

        Args:
            low (float): The lower bound.
            high (float): The upper bound.
            num_samples (int, optional): The number of samples. Defaults to DEFAULT_NUM_SAMPLES.
        """
        samples = np.random.uniform(low, high, num_samples)
        super().__init__(samples=samples.tolist())


class NormalRandomVariable(RandomVariable):
    """A normal random variable."""

    name: Literal["NormalRandomVariable"] = "NormalRandomVariable"

    def __init__(
        self, mean: float, std: float, num_samples: int = DEFAULT_NUM_SAMPLES
    ) -> None:
        """Initialize a normal random variable.

        Args:
            mean (float): The mean.
            std (float): The standard deviation.
            num_samples (int, optional): The number of samples. Defaults to DEFAULT_NUM_SAMPLES.
        """
        samples = np.random.normal(mean, std, num_samples)
        super().__init__(samples=samples.tolist())


class ConstantRandomVariable(RandomVariable):
    """A constant random variable (always returns the same value)."""

    name: Literal["ConstantRandomVariable"] = "ConstantRandomVariable"

    def __init__(self, value: float) -> None:
        """Initialize a constant random variable.

        Args:
            value (float): The constant value.
        """
        super().__init__(samples=[value])

    def sample(self, num_samples: int = 1) -> np.ndarray:
        """Sample from the random variable (always returns the constant value)."""
        return np.full(num_samples, self.samples_arr[0])
