from abc import ABC, abstractmethod

class RankingHeuristic(ABC):
    def __init__(self):
        pass
    
    @abstractmethod
    def __call__(self, network, task_graph):
        raise NotImplementedError
    