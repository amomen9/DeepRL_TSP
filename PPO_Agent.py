"""
PPO_Agent.py - Proximal Policy Optimization (PPO) agent for TSP.

On-policy actor-critic with a clipped surrogate objective, GAE advantages,
and multiple epochs of mini-batch updates per collected trajectory. Follows
the same episode-based template as A2C_Agent / TSP_SAC_Agent: each call
to ``update()`` consumes a single trajectory walked over the DP_Table-backed
environment.
"""

import numpy as np
import torch

from Policy_Based_Agent import TSP_Policy_Based_Agent


class TSP_PPO_Agent(TSP_Policy_Based_Agent):
    """PPO with clipped surrogate objective and GAE advantages for TSP."""

    def __init__(self, env,
                 actor_hidden_nn=np.array([16, 16]),
                 critic_hidden_nn=np.array([64, 64]),
                 actor_lr=0.001, critic_lr=0.001,
                 gamma=0.99,
                 gae_lambda=0.95,
                 clip_epsilon=0.2,
                 n_epochs=4,
                 entropy_coef=0.01,
                 value_coef=0.5,
                 max_grad_norm=0.5):
        super().__init__(
            env,
            actor_hidden_nn=actor_hidden_nn,
            critic_hidden_nn=critic_hidden_nn,
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            gamma=gamma,
            use_critic=True,
            critic_type='V',
        )
        self.gae_lambda = float(gae_lambda)
        self.clip_epsilon = float(clip_epsilon)
        self.n_epochs = int(n_epochs)
        self.entropy_coef = float(entropy_coef)
        self.value_coef = float(value_coef)
        self.max_grad_norm = float(max_grad_norm)

    def compute_gae(self, rewards, dones, values, next_values):
        """Generalized Advantage Estimation (Schulman et al., 2016).

        A_t = delta_t + (gamma * lambda) * (1 - done_t) * A_{t+1}
        where delta_t = r_t + gamma * (1 - done_t) * V(s_{t+1}) - V(s_t)
        """
        T = len(rewards)
        # Pull the critic tensors into plain Python lists once, instead of
        # calling .item() on each element inside the loop (every .item() forces
        # a tensor->Python sync; this is one bulk transfer instead of 2*T).
        values_list = values.tolist()
        next_values_list = next_values.tolist()
        adv = [0.0] * T
        gae = 0.0
        for t in reversed(range(T)):
            non_terminal = 1.0 - float(dones[t])
            delta = (
                float(rewards[t])
                + self.gamma * next_values_list[t] * non_terminal
                - values_list[t]
            )
            gae = delta + self.gamma * self.gae_lambda * non_terminal * gae
            adv[t] = gae
        advantages = torch.tensor(adv, dtype=torch.float32)
        returns = advantages + values.detach()
        return advantages, returns

    def update(self, **kwargs):
        """PPO update on one trajectory.

        Computes old log-probs and advantages once (frozen baseline), then
        performs ``n_epochs`` of full-batch updates with the clipped surrogate
        objective and a value-function regression target.
        """
        states = kwargs["states"]
        actions = kwargs["actions"]
        rewards = kwargs["rewards"]
        next_states = kwargs["next_states"]
        dones = kwargs["dones"]
        masks = kwargs.get("masks")

        states_t = torch.as_tensor(np.array(states), dtype=torch.float32)
        actions_t = torch.as_tensor(np.array(actions), dtype=torch.long)
        next_states_t = torch.as_tensor(np.array(next_states), dtype=torch.float32)

        # Invalid-action mask captured at rollout time (per state). Applying it
        # to the policy logits here keeps the update distribution identical to
        # the behaviour distribution in select_action; otherwise the actor can
        # pile probability on never-trained invalid actions, driving the taken
        # action's log-prob to -inf and the PPO ratio exp(new-old) to inf -> NaN.
        masks_t = None
        if masks is not None:
            masks_t = torch.stack([torch.as_tensor(m, dtype=torch.bool) for m in masks])

        def _masked_dist(logits):
            if masks_t is not None:
                logits = logits.masked_fill(masks_t, -1e9)
            return torch.distributions.Categorical(logits=logits)

        assert self.critic is not None
        assert self.critic_optimizer is not None

        with torch.no_grad():
            values = self.critic(states_t).squeeze(-1)
            next_values = self.critic(next_states_t).squeeze(-1)
            old_dist = _masked_dist(self.actor(states_t))
            old_log_probs = old_dist.log_prob(actions_t)

        advantages, returns = self.compute_gae(rewards, dones, values, next_values)

        # Advantage normalization improves PPO stability when a trajectory has
        # enough samples; for tiny trajectories (T<2) we skip it.
        if advantages.numel() > 1:
            adv_std = advantages.std(unbiased=False)
            if float(adv_std.item()) > 1e-8:
                advantages = (advantages - advantages.mean()) / (adv_std + 1e-8)

        for _ in range(self.n_epochs):
            dist = _masked_dist(self.actor(states_t))
            new_log_probs = dist.log_prob(actions_t)
            entropy = dist.entropy().mean()

            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            if self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.actor_optimizer.step()

            v_pred = self.critic(states_t).squeeze(-1)
            critic_loss = self.value_coef * ((v_pred - returns) ** 2).mean()

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            if self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.critic_optimizer.step()
