import os
import pathlib
from abc import ABC, abstractmethod

home = pathlib.Path.home()

# path where datasets are stored
datadir = pathlib.Path(os.getenv('SAGA_DATADIR', home / '.saga' / 'datasets')).expanduser().resolve()
os.environ['SAGA_DATADIR'] = str(datadir)

# path to save experiment results
resultsdir = pathlib.Path(os.getenv('SAGA_RESULTSDIR', home / '.saga' / 'results')).expanduser().resolve()
os.environ['SAGA_RESULTSDIR'] = str(resultsdir)

# path to save figures
outputdir = pathlib.Path(os.getenv('SAGA_OUTPUTDIR', home / '.saga' / 'output')).expanduser().resolve()
os.environ['SAGA_OUTPUTDIR'] = str(outputdir)


class Experiment(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def prepare(self) -> None:
        raise NotImplementedError
    
    @abstractmethod
    def run(self) -> None:
        raise NotImplementedError
    
    @abstractmethod
    def analyze(self) -> None:
        raise NotImplementedError
    