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

class SimParams(NamedTuple):
    n_simulations: int
    graph: GraphShape
    a: int
    n: int
    e: float
    m: Optional[float]
    max_epochs: int

class SimResults(NamedTuple):
    ## Results for a given sim
    graph_shape: GraphShape
    agents: int
    epochs: int
    av_utility: float # per agent per round
    result: ResultType
    trials: int
    epsilon: float
    mistrust: Optional[float]

class AnalyzedResults(NamedTuple):
    ## Analyzed results – averages over e.g. 5000 runs of sims
    graph_shape: GraphShape
    agents: int
    av_utility: float
    trials: int
    epsilon: float
    mistrust: Optional[float]

    prop_true_cons: float
    prop_false_cons: float
    prop_indeterminate: float
