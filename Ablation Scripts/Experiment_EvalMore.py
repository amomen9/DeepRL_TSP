"""
Experiment_EvalMore.py - denser evaluation schedule (more frequent evaluations
and more episodes per evaluation), reproduced in one command. TSP analogue of
the CartPole fork's Experiment_EvalMore.py.

Run from the project root with:  python "Ablation Scripts/Experiment_EvalMore.py"
"""

import os
import sys

# Allow "from Experiment import Test_TSP" to resolve from this subdirectory.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Experiment import Test_TSP

if __name__ == "__main__":
    Test_TSP(overrides={
        "global_config": {
            "eval_interval": 100,
            "n_eval_episodes": 10,
            "checkpoints": {
                "use_saved_disk_networks_checkpoints": True,
                "skip_selection_hyperparameter_match": True,
            },
        },
    })
