"""
SAC_Agent.py - Soft Actor-Critic (SAC) agent for TSP.

Discrete-action SAC: two Q-critics (clipped double-Q), entropy-regularized
categorical actor, and target Q networks. Follows the same template as
A2C_Agent: episode-based on-policy trajectory updates over the
DP_Table-backed environment.
"""

import copy

import numpy as np
import torch

from Policy_Based_Agent import TSP_Policy_Based_Agent
from Library.Library_networks import Value_NN


class TSP_SAC_Agent(TSP_Policy_Based_Agent):
    """Discrete Soft Actor-Critic with twin Q-critics for TSP."""

    def __init__(self, env,
                 actor_hidden_nn=np.array([16, 16]),
                 critic_hidden_nn=np.array([64, 64]),
                 actor_lr=0.001, critic_lr=0.001,
                 gamma=0.99, TN_step=10,
                 alpha=0.2, alpha_lr=0.001,
                 auto_tune_alpha=True,
                 target_entropy_ratio=0.98,
                 tau=0.005):
        super().__init__(
            env,
            actor_hidden_nn=actor_hidden_nn,
            critic_hidden_nn=critic_hidden_nn,
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            gamma=gamma,
            use_critic=True,
            critic_type='Q',
        )
        self.TN_step = int(TN_step)
        self.tau = float(tau)
        self._update_count = 0

        # Second Q-network for clipped double-Q learning.
        self.critic2 = Value_NN(
            nn_hidden_layer_widths=critic_hidden_nn,
            output_size=self.n_actions,
        )
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=critic_lr)

        # Target Q-networks (frozen).
        self.target_critic = copy.deepcopy(self.critic)
        self.target_critic2 = copy.deepcopy(self.critic2)
        for p in self.target_critic.parameters():
            p.requires_grad = False
        for p in self.target_critic2.parameters():
            p.requires_grad = False

        # Entropy temperature.
        self.auto_tune_alpha = bool(auto_tune_alpha)
        if self.auto_tune_alpha:
            # For categorical actions, max entropy is log(n_actions).
            # If target_entropy_ratio > 1, the target is impossible and can
            # drive log_alpha -> inf (leading to NaNs in training).
            target_entropy_ratio_f = float(target_entropy_ratio)
            target_entropy_ratio_f = max(0.0, min(target_entropy_ratio_f, 1.0))
            self.target_entropy = target_entropy_ratio_f * float(np.log(max(self.n_actions, 1)))

            # Keep alpha numerically sane during auto-tuning.
            self._min_log_alpha = -20.0
            self._max_log_alpha = 2.0

            self.log_alpha = torch.tensor(float(np.log(max(float(alpha), 1e-8))), requires_grad=True)
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)
            self.alpha = self.log_alpha.exp().detach()
        else:
            self.log_alpha = None
            self.alpha_optimizer = None
            self.alpha = torch.tensor(float(alpha))

    def _hard_sync_targets(self):
        for tp, sp in zip(self.target_critic.parameters(), self.critic.parameters()):
            tp.data.copy_(sp.data)
        for tp, sp in zip(self.target_critic2.parameters(), self.critic2.parameters()):
            tp.data.copy_(sp.data)

    def _soft_update_targets(self):
        for tp, sp in zip(self.target_critic.parameters(), self.critic.parameters()):
            tp.data.mul_(1.0 - self.tau).add_(self.tau * sp.data)
        for tp, sp in zip(self.target_critic2.parameters(), self.critic2.parameters()):
            tp.data.mul_(1.0 - self.tau).add_(self.tau * sp.data)

    def update(self, **kwargs):
        """Discrete-SAC update on one trajectory."""
        states = kwargs["states"]
        actions = kwargs["actions"]
        rewards = kwargs["rewards"]
        next_states = kwargs["next_states"]
        dones = kwargs["dones"]

        states_t = torch.as_tensor(np.array(states), dtype=torch.float32)
        actions_t = torch.as_tensor(np.array(actions), dtype=torch.long)
        rewards_t = torch.as_tensor(np.array(rewards), dtype=torch.float32)
        next_states_t = torch.as_tensor(np.array(next_states), dtype=torch.float32)
        dones_t = torch.as_tensor(np.array(dones), dtype=torch.float32)

        alpha = self.alpha if isinstance(self.alpha, torch.Tensor) else torch.tensor(float(self.alpha))

        # ── Critic update: soft Bellman target with twin target Q-networks ──
        assert self.critic is not None
        assert self.critic_optimizer is not None
        with torch.no_grad():
            next_logits = self.actor(next_states_t)
            next_probs = torch.softmax(next_logits, dim=-1)
            next_log_probs = torch.log_softmax(next_logits, dim=-1)
            q1_t_next = self.target_critic(next_states_t)
            q2_t_next = self.target_critic2(next_states_t)
            q_t_next_min = torch.min(q1_t_next, q2_t_next)
            v_next = (next_probs * (q_t_next_min - alpha * next_log_probs)).sum(dim=-1)
            td_target = rewards_t + self.gamma * (1.0 - dones_t) * v_next

        q1_all = self.critic(states_t)
        q2_all = self.critic2(states_t)
        q1 = q1_all.gather(-1, actions_t.unsqueeze(-1)).squeeze(-1)
        q2 = q2_all.gather(-1, actions_t.unsqueeze(-1)).squeeze(-1)
        critic1_loss = ((q1 - td_target) ** 2).sum()
        critic2_loss = ((q2 - td_target) ** 2).sum()

        self.critic_optimizer.zero_grad()
        critic1_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=10.0)
        self.critic_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), max_norm=10.0)
        self.critic2_optimizer.step()

        # ── Actor update: minimize KL between policy and soft-Q distribution ──
        logits = self.actor(states_t)
        probs = torch.softmax(logits, dim=-1)
        log_probs = torch.log_softmax(logits, dim=-1)
        with torch.no_grad():
            q_min_eval = torch.min(self.critic(states_t), self.critic2(states_t))
        actor_loss = (probs * (alpha * log_probs - q_min_eval)).sum(dim=-1).sum()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ── Optional temperature auto-tuning ──
        if self.auto_tune_alpha and self.alpha_optimizer is not None and self.log_alpha is not None:
            with torch.no_grad():
                entropy = -(probs * log_probs).sum(dim=-1)
            alpha_loss = -(self.log_alpha * (self.target_entropy - entropy).detach()).sum()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            # Prevent log_alpha from exploding to +/-inf (can lead to NaNs).
            with torch.no_grad():
                self.log_alpha.clamp_(self._min_log_alpha, self._max_log_alpha)
            self.alpha = self.log_alpha.exp().detach()

        # ── Target network update ──
        # TN_step <= 1 ⇒ Polyak (soft) update every trajectory; otherwise hard
        # sync every TN_step trajectories (mirrors A2C's TN_step convention).
        self._update_count += 1
        if self.TN_step <= 1:
            self._soft_update_targets()
        elif self._update_count % self.TN_step == 0:
            self._hard_sync_targets()
