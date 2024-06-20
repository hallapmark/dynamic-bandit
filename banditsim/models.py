## RESULTS
from enum import Enum, auto
from typing import NamedTuple, Optional

class ResultType(Enum):
    FALSE_CONSENSUS = auto()
    TRUE_CONSENSUS = auto()
    INDETERMINATE = auto() # Hardcoded max rounds were reached before false or true consensus formed

class GraphShape(Enum):
    COMPLETE = auto()
    CYCLE = auto()

class SimResults(NamedTuple):
    graph_shape: GraphShape
    agents: int
    epochs: int
    av_utility: float # per agent per round
    result: ResultType
    trials: int
    epsilon: float
    mistrust: Optional[float]
