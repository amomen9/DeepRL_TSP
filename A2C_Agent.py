"""
A2C_Agent.py - Advantage Actor-Critic (A2C) agent for TSP.

Uses an n-step bootstrapped target for the critic / advantage, mirroring the
CartPole_PolicyBased A2C agent.
"""

import numpy as np
import torch

from Policy_Based_Agent import TSP_Policy_Based_Agent


class A2C_Agent(TSP_Policy_Based_Agent):
    """A2C with an n-step bootstrap target for TSP."""

    def __init__(self, env,
                 actor_hidden_nn=np.array([16, 16]),
                 critic_hidden_nn=np.array([64, 64]),
                 actor_lr=0.001, critic_lr=0.001,
                 gamma=0.99, TN_step=10):
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
        self.TN_step = TN_step

    def compute_returns(self, rewards, next_states, dones):
        """n-step return estimate q_hat_t = sum_{k=0}^{n-1} gamma^k r_{t+k}
        + gamma^n * V_phi(s_{t+n}) for each timestep t (no bootstrap if the
        n-step window crosses a terminal transition).
        """
        episode_length = len(rewards)
        next_states_t = torch.as_tensor(np.array(next_states), dtype=torch.float32)
        assert self.critic is not None
        with torch.no_grad():
            v_s_next = self.critic(next_states_t).squeeze(-1)

        q_hat = torch.zeros(episode_length)
        for t in range(episode_length):
            ret = 0.0
            for k in range(self.TN_step):
                if t + k >= episode_length:
                    break
                ret += (self.gamma ** k) * rewards[t + k]
                if dones[t + k]:
                    break
            else:
                # n-step bootstrap should use V(s_{t+TN_step}), which corresponds to
                # v_s_next[t+TN_step-1] because next_states[t] == s_{t+1}.
                boot_idx = min(t + self.TN_step - 1, episode_length - 1)
                ret += (self.gamma ** self.TN_step) * float(v_s_next[boot_idx].item())
            q_hat[t] = ret
        return q_hat

    def update(self, **kwargs):
        """Update policy and value networks from one trajectory."""
        states = kwargs["states"]
        actions = kwargs["actions"]
        rewards = kwargs["rewards"]
        next_states = kwargs["next_states"]
        dones = kwargs["dones"]

        states_t = torch.as_tensor(np.array(states), dtype=torch.float32)
        actions_t = torch.as_tensor(np.array(actions), dtype=torch.long)

        q_hat = self.compute_returns(rewards, next_states, dones)

        assert self.critic is not None
        assert self.critic_optimizer is not None
        v_s = self.critic(states_t).squeeze(-1)
        advantages = q_hat - v_s

        critic_loss = (advantages ** 2).sum()
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        logits = self.actor(states_t)
        dist = torch.distributions.Categorical(logits=logits)
        log_probs = dist.log_prob(actions_t)
        actor_loss = -(advantages.detach() * log_probs).sum()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
