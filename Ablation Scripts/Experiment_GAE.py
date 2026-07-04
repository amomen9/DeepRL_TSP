"""
Experiment_GAE.py - PPO GAE-lambda ablation, reproduced in one command. TSP
analogue of the CartPole fork's dedicated Experiment_*.py sweeps. (lambda = 1.0
falls back to plain Monte-Carlo advantages.)

Run from the project root with:  python "Ablation Scripts/Experiment_GAE.py"
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
        "ppo_config": {"gae_lambda": [0.9, 0.96, 1.0]},
    })
