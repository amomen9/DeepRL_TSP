"""Policy-based RL agents for stochastic TSP.

REINFORCE, AC, and A2C use one clean on-policy update on the freshly collected
complete tours.  PPO is the only method that reuses a rollout with clipped
objective, GAE, mini-batches, and several internal epochs.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from env import StochasticTSPEnv

# PPO-only internal defaults.  Kept here deliberately so no extra controller is
# needed for this project.
PPO_EPOCHS = 4
PPO_CLIP = 0.20
PPO_GAE_LAMBDA = 0.95
PPO_VALUE_COEF = 0.50
PPO_MAX_GRAD_NORM = 0.50


def mlp(input_dim: int, hidden: tuple[int, ...], output_dim: int) -> nn.Sequential:
    layers, d = [], input_dim
    for h in hidden:
        layers += [nn.Linear(d, int(h)), nn.ReLU()]
        d = int(h)
    layers.append(nn.Linear(d, output_dim))
    return nn.Sequential(*layers)


@dataclass
class AgentConfig:
    method: str = "A2C"          # REINFORCE, AC, A2C, PPO
    actor_hidden: tuple[int, ...] = (32, 32)
    critic_hidden: tuple[int, ...] = (64, 64)
    actor_lr: float = 1e-3
    critic_lr: float = 3e-3
    gamma: float = 1.0
    entropy_coef: float = 0.01
    n_step: int = 5              # A2C only


class PolicyBasedAgent:
    def __init__(self, env: StochasticTSPEnv, config: AgentConfig | None = None):
        self.env = env
        self.cfg = config or AgentConfig()
        self.method = self.cfg.method.upper()
        if self.method not in {"REINFORCE", "AC", "A2C", "PPO"}:
            raise ValueError(f"unknown RL method: {self.cfg.method}")

        self.actor = mlp(4, self.cfg.actor_hidden, env.n_actions)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.cfg.actor_lr)
        self.critic = None if self.method == "REINFORCE" else mlp(4, self.cfg.critic_hidden, 1)
        self.critic_opt = None if self.critic is None else torch.optim.Adam(self.critic.parameters(), lr=self.cfg.critic_lr)

    @staticmethod
    def _tensor(x, dtype=torch.float32) -> torch.Tensor:
        return x if torch.is_tensor(x) else torch.as_tensor(x, dtype=dtype)

    @staticmethod
    def _normalize(x: torch.Tensor) -> torch.Tensor:
        return x if x.numel() <= 1 else (x - x.mean()) / (x.std(unbiased=False) + 1e-8)

    def _masked_logits(self, obs, masks=None) -> torch.Tensor:
        x = self._tensor(obs, torch.float32)
        logits = self.actor(x)
        if masks is None:
            masks = self.env.valid_action_mask()
        m = self._tensor(masks, torch.bool)
        return logits.masked_fill(~m, -1e9)

    def _dist(self, obs, masks=None) -> torch.distributions.Categorical:
        return torch.distributions.Categorical(logits=self._masked_logits(obs, masks))

    def act(self, obs: np.ndarray, greedy: bool = False, mask: np.ndarray | None = None):
        logits = self._masked_logits(obs, mask)
        dist = torch.distributions.Categorical(logits=logits)
        action = torch.argmax(logits).item() if greedy else dist.sample().item()
        action_t = torch.as_tensor(int(action), dtype=torch.long)
        return int(action), dist.log_prob(action_t), dist.entropy()

    def act_batch(
        self,
        obs: np.ndarray,
        masks: np.ndarray,
        greedy: bool = False,
        action_cities_batch: list[tuple[int, ...]] | None = None,
    ) -> np.ndarray:
        actions = []
        logits = self.actor(self._tensor(obs, torch.float32))
        for i in range(len(masks)):
            mask_t = torch.as_tensor(masks[i], dtype=torch.bool)
            logits_i = logits[i].masked_fill(~mask_t, -1e9)
            dist = torch.distributions.Categorical(logits=logits_i)
            action = torch.argmax(logits_i).item() if greedy else dist.sample().item()
            actions.append(int(action))
        return np.asarray(actions, dtype=int)

    def run_episode(self, greedy: bool = False, seed: int | None = None) -> dict:
        obs, _ = self.env.reset(seed)
        states: list[np.ndarray] = []
        actions: list[int] = []
        rewards: list[float] = []
        masks: list[np.ndarray] = []
        next_states: list[np.ndarray] = []
        dones: list[bool] = []
        route, total_return, done = [self.env.depot_city], 0.0, False
        while not done:
            mask = self.env.valid_action_mask().copy()
            action, _, _ = self.act(obs, greedy=greedy, mask=mask)
            next_obs, reward, done, _, info = self.env.step(action)
            states.append(obs.copy())
            actions.append(int(action))
            rewards.append(float(reward))
            masks.append(mask.copy())
            next_states.append(next_obs.copy())
            dones.append(bool(done))
            obs = next_obs
            total_return += float(reward)
            route.append(int(info.get("next_city", self.env.action_to_city(action))))
        route.append(self.env.depot_city)
        return {
            "states": states,
            "actions": actions,
            "rewards": rewards,
            "masks": masks,
            "next_states": next_states,
            "dones": dones,
            "route": route,
            "return": total_return,
            "cost": -total_return,
            "done": done,
        }

    def _discounted_returns(self, rewards: list[float]) -> torch.Tensor:
        out, g = [], 0.0
        for r in reversed(rewards):
            g = float(r) + self.cfg.gamma * g
            out.append(g)
        return torch.as_tensor(list(reversed(out)), dtype=torch.float32)

    def _n_step_returns(self, tr: dict) -> torch.Tensor:
        rewards, next_states, dones = tr["rewards"], tr["next_states"], tr["dones"]
        T = len(rewards)
        out = torch.zeros(T, dtype=torch.float32)
        critic = self.critic
        assert critic is not None
        with torch.no_grad():
            next_v = critic(self._tensor(np.asarray(next_states, dtype=np.float32))).squeeze(-1)
        for t in range(T):
            g = 0.0
            for k in range(self.cfg.n_step):
                if t + k >= T:
                    break
                g += (self.cfg.gamma ** k) * float(rewards[t + k])
                if dones[t + k]:
                    break
            else:
                idx = min(t + self.cfg.n_step - 1, T - 1)
                g += (self.cfg.gamma ** self.cfg.n_step) * float(next_v[idx])
            out[t] = g
        return out

    def _flat_batch(self, trajectories: list[dict]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        states, actions, masks, targets = [], [], [], []
        for tr in trajectories:
            states.extend(tr["states"])
            actions.extend(tr["actions"])
            masks.extend(tr["masks"])
            if self.method == "REINFORCE":
                targets.append(self._discounted_returns(tr["rewards"]))
            elif self.method == "AC":
                targets.append(self._discounted_returns(tr["rewards"]))
            else:  # A2C
                targets.append(self._n_step_returns(tr))
        return (
            self._tensor(np.asarray(states, dtype=np.float32)),
            torch.as_tensor(actions, dtype=torch.long),
            torch.as_tensor(np.asarray(masks, dtype=bool), dtype=torch.bool),
            torch.cat(targets).detach(),
        )

    def update(self, traj: dict, minibatch_size: int | None = None) -> dict:
        return self.update_batch([traj], minibatch_size=minibatch_size)

    def update_batch(self, trajectories: list[dict], minibatch_size: int | None = None) -> dict:
        if self.method == "PPO":
            return self._update_ppo(trajectories, minibatch_size=minibatch_size)
        return self._update_vanilla(trajectories)

    def _update_vanilla(self, trajectories: list[dict]) -> dict:
        """Single fresh-batch update for REINFORCE / AC / A2C.

        No mini-batch and no repeated epochs here: these methods are plain
        on-policy gradients.  Parallel envs only enlarge the one fresh batch.
        """
        states, actions, masks, targets = self._flat_batch(trajectories)
        dist = self._dist(states, masks)
        logp = dist.log_prob(actions)
        entropy = dist.entropy().mean()

        if self.method == "REINFORCE":
            weight = self._normalize(targets)
            actor_loss = -(logp * weight).mean() - self.cfg.entropy_coef * entropy
            self.actor_opt.zero_grad(); actor_loss.backward(); self.actor_opt.step()
            critic_loss = torch.tensor(0.0)
        else:
            critic = self.critic
            critic_opt = self.critic_opt
            assert critic is not None and critic_opt is not None
            values = critic(states).squeeze(-1)
            critic_loss = F.mse_loss(values, targets)
            critic_opt.zero_grad(); critic_loss.backward(); critic_opt.step()

            if self.method == "AC":
                weight = targets
            else:  # A2C advantage target
                with torch.no_grad():
                    weight = targets - critic(states).squeeze(-1)
            weight = self._normalize(weight.detach())
            dist = self._dist(states, masks)
            actor_loss = -(dist.log_prob(actions) * weight).mean() - self.cfg.entropy_coef * dist.entropy().mean()
            self.actor_opt.zero_grad(); actor_loss.backward(); self.actor_opt.step()

        return {"actor_loss": float(actor_loss.detach()), "critic_loss": float(critic_loss.detach())}

    def _ppo_advantages(self, trajectories: list[dict]):
        states, actions, masks, old_logps, advs, rets = [], [], [], [], [], []
        assert self.critic is not None and self.critic_opt is not None
        for tr in trajectories:
            s = self._tensor(np.asarray(tr["states"], dtype=np.float32))
            ns = self._tensor(np.asarray(tr["next_states"], dtype=np.float32))
            a = torch.as_tensor(tr["actions"], dtype=torch.long)
            m = torch.as_tensor(np.asarray(tr["masks"], dtype=bool), dtype=torch.bool)
            r = self._tensor(np.asarray(tr["rewards"], dtype=np.float32))
            d = self._tensor(np.asarray(tr["dones"], dtype=np.float32))
            with torch.no_grad():
                v = self.critic(s).squeeze(-1)
                nv = self.critic(ns).squeeze(-1)
                lp = self._dist(s, m).log_prob(a)
            gae = torch.tensor(0.0)
            adv = torch.zeros_like(r)
            for t in range(len(r) - 1, -1, -1):
                delta = r[t] + self.cfg.gamma * nv[t] * (1.0 - d[t]) - v[t]
                gae = delta + self.cfg.gamma * PPO_GAE_LAMBDA * (1.0 - d[t]) * gae
                adv[t] = gae
            states.append(s); actions.append(a); masks.append(m); old_logps.append(lp)
            advs.append(adv); rets.append(adv + v)
        return (torch.cat(states), torch.cat(actions), torch.cat(masks),
                torch.cat(old_logps).detach(), torch.cat(advs).detach(), torch.cat(rets).detach())

    def _update_ppo(self, trajectories: list[dict], minibatch_size: int | None = None) -> dict:
        states, actions, masks, old_logp, adv, ret = self._ppo_advantages(trajectories)
        adv = self._normalize(adv)
        n = int(states.shape[0])
        mb = max(1, min(int(minibatch_size or n), n))
        actor_losses, critic_losses = [], []
        critic = self.critic
        critic_opt = self.critic_opt
        assert critic is not None and critic_opt is not None

        for _ in range(PPO_EPOCHS):
            for idx in torch.randperm(n).split(mb):
                dist = self._dist(states[idx], masks[idx])
                new_logp = dist.log_prob(actions[idx])
                ratio = torch.exp(new_logp - old_logp[idx])
                unclipped = ratio * adv[idx]
                clipped = torch.clamp(ratio, 1.0 - PPO_CLIP, 1.0 + PPO_CLIP) * adv[idx]
                actor_loss = -torch.min(unclipped, clipped).mean() - self.cfg.entropy_coef * dist.entropy().mean()
                value_loss = F.mse_loss(critic(states[idx]).squeeze(-1), ret[idx])
                loss = actor_loss + PPO_VALUE_COEF * value_loss

                self.actor_opt.zero_grad(); critic_opt.zero_grad(); loss.backward()
                if PPO_MAX_GRAD_NORM is not None:
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), PPO_MAX_GRAD_NORM)
                    torch.nn.utils.clip_grad_norm_(critic.parameters(), PPO_MAX_GRAD_NORM)
                self.actor_opt.step(); critic_opt.step()
                actor_losses.append(float(actor_loss.detach()))
                critic_losses.append(float(value_loss.detach()))

        return {"actor_loss": float(np.mean(actor_losses)), "critic_loss": float(np.mean(critic_losses))}

    def save(self, path: str) -> None:
        torch.save({
            "config": asdict(self.cfg),
            "actor": self.actor.state_dict(),
            "critic": None if self.critic is None else self.critic.state_dict(),
        }, path)

    @classmethod
    def load(cls, path: str, env: StochasticTSPEnv) -> "PolicyBasedAgent":
        data = torch.load(path, map_location="cpu")
        agent = cls(env, AgentConfig(**data["config"]))
        agent.actor.load_state_dict(data["actor"])
        if agent.critic is not None and data.get("critic") is not None:
            agent.critic.load_state_dict(data["critic"])
        return agent
