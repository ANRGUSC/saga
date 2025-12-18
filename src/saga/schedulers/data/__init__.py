import pathlib
import os
from typing import Generator, List
from pydantic import BaseModel
from pydantic import Field

from saga import TaskGraph, Network


def get_data_dir() -> pathlib.Path:
    """Get the SAGA data directory.

    Returns:
        pathlib.Path: The SAGA data directory.
    """
    data_dir = pathlib.Path(
        os.getenv("SAGA_DATA_DIR", pathlib.Path.home() / ".saga" / "data")
    )
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


class ProblemInstance(BaseModel):
    """Base class for problem instances."""

    name: str = Field(..., description="The name of the problem instance.")
    task_graph: TaskGraph = Field(
        ..., description="The task graph of the problem instance."
    )
    network: Network = Field(..., description="The network of the problem instance.")

    def __init__(self, name: str, task_graph: TaskGraph, network: Network):
        super().__init__(name=name, task_graph=task_graph, network=network)


class Dataset(BaseModel):
    """Base class for datasets."""

    name: str = Field(..., description="The name of the dataset.")
    data_dir: pathlib.Path = Field(
        default_factory=get_data_dir,
        description="The directory where the dataset is stored.",
    )

    def __init__(self, name: str, data_dir: pathlib.Path | None = None):
        super().__init__(name=name, data_dir=data_dir or get_data_dir())
        self._save_dir = self.data_dir / self.name
        self._save_dir.mkdir(parents=True, exist_ok=True)

    @property
    def instances(self) -> List[str]:
        """List of problem instance names in the dataset."""
        return [p.stem for p in self._save_dir.glob("*.json")]

    @property
    def size(self) -> int:
        """Number of problem instances in the dataset."""
        return len(self.instances)

    def get_instance(self, instance_name: str) -> ProblemInstance:
        """Load a problem instance from the dataset.

        Args:
            instance_name (str): The name of the problem instance.
        Returns:
            ProblemInstance: The loaded problem instance.
        """
        instance_path = self._save_dir / f"{instance_name}.json"
        return ProblemInstance.model_validate_json(instance_path.read_text())

    def save_instance(self, instance: ProblemInstance) -> None:
        """Save a problem instance to the dataset.

        Args:
            instance (ProblemInstance): The problem instance to save.
        """
        instance_path = self._save_dir / f"{instance.name}.json"
        instance_path.write_text(instance.model_dump_json(indent=4))

    def iter_instances(self) -> Generator[ProblemInstance, None, None]:
        """Iterate over all problem instances in the dataset.

        Yields:
            ProblemInstance: The next problem instance in the dataset.
        """
        for instance_name in self.instances:
            yield self.get_instance(instance_name)
