"""
Experiment_Rollout.py - PPO rollout-length (buffer size) ablation, reproduced in
one command. TSP analogue of the CartPole fork's dedicated Experiment_*.py
sweeps. (A very large value effectively uses whole episodes as one rollout.)

Run from the project root with:  python "Ablation Scripts/Experiment_Rollout.py"
"""

import os
import sys

# Allow "from Experiment import Test_TSP" to resolve from this subdirectory.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Experiment import Test_TSP

if __name__ == "__main__":
    Test_TSP(overrides={
        "global_config": {
            "checkpoints": {
                "use_saved_disk_networks_checkpoints": True,
                "skip_selection_hyperparameter_match": True,
            },
        },
        "ppo_config": {"rollout_steps": [256, 512, 1024]},
    })
