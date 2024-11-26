# dynamic-bandit
WIP. 
Python code for modelling a community of k/n updaters working on a two-armed bandit problem. 
One arm's chance of payoff is fixed at .5 and the other arm's chance of payoff follows a sinusoid function centered on .5.

This project requires Python 3.12.

# Installation
(1) clone this repository

(2) 
```
cd dynamic-bandit
conda env create -f environment.yml
conda activate dynamic-bandits
```

# Running the code
To run the model, do
```
conda activate dynamic-bandits
python dynamic_bandit.py
```

You can modify the parameters in `dynamic_bandit.py` to conduct experiments. By default, the agents update myopically. 

`epsilon` > 0 enables the epsilon-greedy strategy

`window_s` enables windowed updating (agents discard old data from rounds that fall outside the window size).
