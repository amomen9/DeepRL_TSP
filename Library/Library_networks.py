"""
Library_networks.py - Policy- and value-network building blocks.

Contents
--------
Value_NN          - State- or action-value critic (MLP).
Policy_NN         - Actor / policy network (MLP).
trained_nn_policy - Greedy action selector from a trained actor.
"""
import numpy as np
import torch.nn as nn


################[ Value_NN (Critic) ]################
class Value_NN(nn.Module):      # This is neural network for the critic (Values NN), which can be used as either a state-value critic V_phi(s) or an action-value critic Q_phi(s) depending on the output size and how it's trained.
    """State-value critic V_phi(s), or Q-value critic Q_phi(s) when used per-action."""
    def __init__(self, nn_hidden_layer_widths=np.array([64, 64]), output_size=1):
        super().__init__()
        hidden_widths = np.asarray(nn_hidden_layer_widths, dtype=np.int32).tolist()
        if len(hidden_widths) == 0:
            raise ValueError("nn_hidden_layer_widths must contain at least one hidden-layer width")

        layers = []
        input_size = 4  # TSP state dimension
        for width in hidden_widths:
            layers.append(nn.Linear(input_size, int(width)))
            layers.append(nn.ReLU())
            input_size = int(width)
        layers.append(nn.Linear(input_size, output_size))   # Output layer: 1 for V_phi(s), 2 for Q_phi(s) with two actions
        self.net = nn.Sequential(*layers)

    def forward(self, state):
        return self.net(state)
#########################################################


################[ Policy_NN Class              ]################
class Policy_NN(nn.Module):
    def __init__(self, nn_hidden_layer_widths=np.array([5]), output_size=1):     # output_size: 1 for binary (Bernoulli) actions, n_actions for categorical TSP next-city selection.
        super().__init__()
        hidden_widths = np.asarray(nn_hidden_layer_widths, dtype=np.int32).tolist()
        if len(hidden_widths) == 0:
            raise ValueError("nn_hidden_layer_widths must contain at least one hidden-layer width")

        layers = []
        input_layer_size = 4  # input layer size (state dimension). Is always 4 for TSP, but we keep it here for generality and readability.
        for width in hidden_widths:
            layers.append(nn.Linear(input_layer_size, int(width)))
            layers.append(nn.ReLU())
            input_layer_size = int(width)
        layers.append(nn.Linear(input_layer_size, output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, state):
        return self.net(state)
####################################################################
