## RESULTS
from enum import Enum, auto
from typing import NamedTuple, Optional

class GraphShape(Enum):
    COMPLETE = auto()
    CYCLE = auto()

class DynamicEpsilonConfig(NamedTuple):
    change_after_n_rounds: int
    epsilon_d: float

class SimResults(NamedTuple):
    ## Results for a given sim
    # Config
    graph_shape: GraphShape
    agents: int
    max_epochs: int
    trials: int
    epsilon: float
    burn_in: int
    e_change_n_rounds: Optional[bool]
    epsilon_d: Optional[bool]

    # Outcome
    epochs: int
    av_utility: float # per agent per round

class AnalyzedResults(NamedTuple):
    ## Analyzed results – includes averages over e.g. 5000 runs of sims of a given configuration
    # Config
    graph_shape: GraphShape
    agents: int
    max_epochs: int
    trials: int
    epsilon: float
    burn_in: int
    e_change_n_rounds: Optional[bool]
    epsilon_d: Optional[bool]

    # Results over all sims
    av_utility: float
