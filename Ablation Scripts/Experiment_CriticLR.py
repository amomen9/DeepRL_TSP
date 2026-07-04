"""
Experiment_CriticLR.py - PPO critic learning-rate ablation, reproduced in one
command. TSP analogue of the CartPole fork's dedicated Experiment_*.py sweeps.

Run from the project root with:  python "Ablation Scripts/Experiment_CriticLR.py"
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
        "ppo_config": {"critic_lr": [4e-3, 1e-2, 3e-2]},
    })
