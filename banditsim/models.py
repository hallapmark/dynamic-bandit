## RESULTS
from enum import Enum, auto
from typing import NamedTuple, Optional

class GraphShape(Enum):
    COMPLETE = auto()
    CYCLE = auto()

class SimResults(NamedTuple):
    ## Results for a given sim
    # Config
    graph_shape: GraphShape
    agents: int
    max_epochs: int
    trials: int
    sine_amp: float
    sine_period: int
    burn_in: int
    epsilon: float # Epsilon here is the epsilon of the e-greedy strategy
    window_s: Optional[int]
    lifecycle: bool # Not yet implemented
    
    # Outcome
    epochs: int
    av_utility: float # per agent per round per trial
    
class AnalyzedResults(NamedTuple):
    ## Analyzed results – includes averages over e.g. 5000 runs of sims of a given configuration
    # Config
    graph_shape: GraphShape
    agents: int
    max_epochs: int
    trials: int
    sine_amp: float
    sine_period: int
    burn_in: int
    epsilon: float
    window_s: Optional[int]
    lifecycle: bool

    # Results over all sims
    av_utility: float
