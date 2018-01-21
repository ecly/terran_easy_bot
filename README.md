# TerranEasyBot
A simple Terran Bot with limited build/training options, using Q-learning.  
Originially based on: https://github.com/skjb/pysc2-tutorial

## Setup
1. Install game (tested on version 3.16.1) from https://github.com/Blizzard/s2client-proto
2. Install maps from same repo. We've used Simple64 from the Melee pack for all testing
3. Install python3 librarias panda and pysc2 with pip
4. Run agent with `python -m pysc2.bin.agent --map Simple64 --agent sparse_agent.SparseAgent --agent_race T --max_agent_steps 0`
