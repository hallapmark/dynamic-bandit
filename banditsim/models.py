## RESULTS
from enum import Enum, auto
from typing import NamedTuple, Optional

class GraphShape(Enum):
    COMPLETE = auto()
    CYCLE = auto()

class AdmitteeType(Enum):
    NONCONFORMIST = auto() # Enter with random[??] belief/expectation for B
    CONFORMIST = auto() # Conform to existing network's average expectation
    
class SimResults(NamedTuple):
    ## Results for a given sim
    # Config
    graph_shape: GraphShape
    agents: int
    max_epochs: int
    trials: int
    max_epsilon: float
    sine_period: int
    burn_in: int
    window_s: Optional[int]
    lifecycle: bool
    admitteetype: AdmitteeType
    
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
    max_epsilon: float
    sine_period: int
    burn_in: int
    window_s: Optional[int]
    lifecycle: bool
    admitteetype: AdmitteeType

    # Results over all sims
    av_utility: float
