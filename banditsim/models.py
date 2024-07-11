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

class DynamicEpsilonConfig(NamedTuple):
    change_after_n_rounds: int
    epsilon_d: float

class SimResults(NamedTuple):
    ## Results for a given sim
    graph_shape: GraphShape
    agents: int
    max_epochs: int
    epochs: int
    av_utility: float # per agent per round
    result: ResultType
    trials: int
    epsilon: float
    mistrust: Optional[float]
    burn_in: int
    e_change_n_rounds: Optional[bool]
    epsilon_d: Optional[bool]

class AnalyzedResults(NamedTuple):
    ## Analyzed results – includes averages over e.g. 5000 runs of sims of a given configuration
    ## Config
    graph_shape: GraphShape
    agents: int
    trials: int
    epsilon: float
    max_epochs: int
    mistrust: Optional[float]
    burn_in: int
    e_change_n_rounds: Optional[bool]
    epsilon_d: Optional[bool]

    ## Results over all sims
    av_utility: float
    prop_true_cons: float
    prop_false_cons: float
    prop_indeterminate: float
