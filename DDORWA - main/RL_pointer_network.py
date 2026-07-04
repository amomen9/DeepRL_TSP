from __future__ import annotations

from dataclasses import dataclass, asdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from env import StochasticTSPEnv
from RL_policy_based import AgentConfig

# Gradient clipping — same value as PPO uses.
POINTER_MAX_GRAD_NORM = 0.50


@dataclass
class PointerNetConfig(AgentConfig):
    """Configuration for the Pointer Network agent."""

    method: str = "A2C"          # keep A2C as the RL update rule
    embed_dim: int = 64          # 128 in Bello et al.; 64 is fine for small n
    n_glimpses: int = 1          # 1 glimpse as recommended by Bello et al.
    tanh_clip: float = 10.0      # logit clipping constant C
    actor_lr: float = 1e-3
    critic_lr: float = 1e-3
    entropy_coef: float = 0.01
    gamma: float = 1.0
    n_step: int = 5


class CityEncoder(nn.Module):
    """Encode every city as a d-dimensional vector using its distance row."""

    def __init__(self, input_dim: int, embed_dim: int):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, city_features: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.proj(city_features))


class AttentionPointer(nn.Module):
    """Single-head attention pointing mechanism."""

    def __init__(self, embed_dim: int, tanh_clip: float = 10.0):
        super().__init__()
        self.W_ref = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v = nn.Linear(embed_dim, 1, bias=False)
        self.C = tanh_clip

    def forward(self, query: torch.Tensor, ref: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Return logits shape (n,) or (B, n)."""
        ref_proj = self.W_ref(ref)
        query_proj = self.W_q(query)
        u = self.v(torch.tanh(ref_proj + query_proj.unsqueeze(-2))).squeeze(-1)
        u = self.C * torch.tanh(u)
        return u.masked_fill(~mask, -1e9)


class GlimpseAttention(nn.Module):
    """Glimpse mechanism."""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.W_ref = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v = nn.Linear(embed_dim, 1, bias=False)

    def forward(self, query: torch.Tensor, ref: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        ref_proj = self.W_ref(ref)
        query_proj = self.W_q(query)
        e = self.v(torch.tanh(ref_proj + query_proj.unsqueeze(-2))).squeeze(-1)
        e = e.masked_fill(~mask, -1e9)
        attn = torch.softmax(e, dim=-1)
        return (attn.unsqueeze(-1) * ref).sum(dim=-2)


class PointerNetworkActor(nn.Module):
    """Full pointer network policy."""

    def __init__(self, n_cities: int, embed_dim: int, n_glimpses: int, tanh_clip: float):
        super().__init__()
        self.encoder = CityEncoder(input_dim=n_cities, embed_dim=embed_dim)
        self.decoder_rnn = nn.LSTMCell(embed_dim, embed_dim)
        self.glimpses = nn.ModuleList([GlimpseAttention(embed_dim) for _ in range(n_glimpses)])
        self.pointer = AttentionPointer(embed_dim, tanh_clip)
        self.embed_dim = embed_dim
        self.n_cities = n_cities
        self.start_embed = nn.Parameter(torch.randn(embed_dim) * 0.01)

    def encode(self, dist_matrix: torch.Tensor) -> torch.Tensor:
        return self.encoder(dist_matrix)

    def step(
        self,
        enc: torch.Tensor,
        hx: tuple,
        last_city: int,
        mask: torch.Tensor,
        action_cities: tuple[int, ...] | None = None,
        first_step: bool = False,
    ) -> tuple[torch.Tensor, tuple]:
        """One decoder step over the env's depot-relative action order."""
        if first_step:
            inp = self.start_embed
        else:
            inp = enc[last_city]

        hx = self.decoder_rnn(inp.unsqueeze(0), (hx[0], hx[1]))
        query = hx[0].squeeze(0)

        if action_cities is None:
            action_cities = tuple(range(1, self.n_cities))
        action_idx = torch.as_tensor(action_cities, dtype=torch.long)

        full_mask = torch.zeros(self.n_cities, dtype=torch.bool)
        for city in action_cities:
            full_mask[int(city)] = True
        for glimpse in self.glimpses:
            query = glimpse(query, enc, full_mask)

        logits = self.pointer(query, enc[action_idx], mask)
        return logits, hx


class PointerNetworkCritic(nn.Module):
    """Simple critic: encodes cities, mean-pools, then MLP → scalar value."""

    def __init__(self, n_cities: int, embed_dim: int):
        super().__init__()
        self.encoder = CityEncoder(input_dim=n_cities, embed_dim=embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim + 4, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
        )

    def forward(self, dist_matrix: torch.Tensor, state_obs: torch.Tensor) -> torch.Tensor:
        enc = self.encoder(dist_matrix)
        pool = enc.mean(dim=0)
        x = torch.cat([pool, state_obs], dim=-1)
        return self.mlp(x).squeeze(-1)


class PointerNetAgent:
    """Pointer Network RL agent, drop-in replacement for PolicyBasedAgent."""

    def __init__(self, env: StochasticTSPEnv, config: PointerNetConfig | None = None):
        self.env = env
        self.cfg = config or PointerNetConfig()
        self.method = "A2C"

        n = env.n
        d = self.cfg.embed_dim

        self.actor = PointerNetworkActor(
            n_cities=n,
            embed_dim=d,
            n_glimpses=self.cfg.n_glimpses,
            tanh_clip=self.cfg.tanh_clip,
        )
        self.critic = PointerNetworkCritic(n_cities=n, embed_dim=d)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.cfg.actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=self.cfg.critic_lr)

        self._enc_cache: torch.Tensor | None = None
        self._hx: tuple | None = None

    def _get_dist_tensor(self) -> torch.Tensor:
        return torch.as_tensor(self.env.base, dtype=torch.float32)

    def _init_decoder(self) -> tuple:
        d = self.cfg.embed_dim
        return (torch.zeros(1, d), torch.zeros(1, d))

    def _encode_cities(self) -> torch.Tensor:
        dist = self._get_dist_tensor()
        with torch.no_grad():
            enc = self.actor.encode(dist)
        return enc

    def act(
        self,
        obs: np.ndarray,
        greedy: bool = False,
        mask: np.ndarray | None = None,
        action_cities: tuple[int, ...] | None = None,
        first_step: bool = False,
        last_city: int = 0,
    ):
        """Select an action."""
        if self._enc_cache is None:
            self._enc_cache = self._encode_cities()
        if self._hx is None or first_step:
            self._hx = self._init_decoder()

        if mask is None:
            mask = self.env.valid_action_mask()
        if action_cities is None:
            action_cities = self.env.action_cities
        mask_t = torch.as_tensor(mask, dtype=torch.bool)

        logits, self._hx = self.actor.step(
            self._enc_cache,
            self._hx,
            last_city,
            mask_t,
            action_cities=action_cities,
            first_step=first_step,
        )
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
        enc = self._encode_cities()
        for i in range(len(masks)):
            mask = masks[i]
            action_cities = action_cities_batch[i] if action_cities_batch is not None else self.env.action_cities
            logits, _ = self.actor.step(
                enc,
                self._init_decoder(),
                0,
                torch.as_tensor(mask, dtype=torch.bool),
                action_cities=action_cities,
                first_step=True,
            )
            dist = torch.distributions.Categorical(logits=logits)
            a = torch.argmax(logits).item() if greedy else dist.sample().item()
            actions.append(int(a))
        return np.array(actions, dtype=int)

    def run_episode(self, greedy: bool = False, seed: int | None = None) -> dict:
        obs, _ = self.env.reset(seed)
        self._enc_cache = self._encode_cities()
        self._hx = self._init_decoder()

        action_cities = tuple(self.env.action_cities)
        traj = {
            "states": [],
            "actions": [],
            "rewards": [],
            "masks": [],
            "next_states": [],
            "dones": [],
            "last_cities": [],
            "action_cities": action_cities,
        }

        route = [self.env.depot_city]
        total_return = 0.0
        done = False
        last_city = self.env.depot_city
        first_step = True

        while not done:
            mask = self.env.valid_action_mask().copy()
            action, _, _ = self.act(
                obs,
                greedy=greedy,
                mask=mask,
                action_cities=action_cities,
                first_step=first_step,
                last_city=last_city,
            )
            next_obs, reward, done, _, info = self.env.step(action)

            traj["states"].append(obs.copy())
            traj["actions"].append(action)
            traj["rewards"].append(float(reward))
            traj["masks"].append(mask.copy())
            traj["next_states"].append(next_obs.copy())
            traj["dones"].append(bool(done))
            traj["last_cities"].append(last_city)

            last_city = int(info.get("next_city", self.env.action_to_city(action)))
            obs = next_obs
            total_return += float(reward)
            first_step = False
            route.append(last_city)

        route.append(self.env.depot_city)
        traj["route"] = route
        traj["return"] = total_return
        traj["cost"] = -total_return
        traj["done"] = done
        return traj

    def update(self, traj: dict, minibatch_size: int | None = None) -> dict:
        return self.update_batch([traj], minibatch_size=minibatch_size)

    def update_batch(self, trajectories: list[dict], minibatch_size: int | None = None) -> dict:
        """A2C update using n-step returns as targets."""
        dist_t = self._get_dist_tensor()
        gamma = self.cfg.gamma
        n_step = self.cfg.n_step
        ent_coef = self.cfg.entropy_coef

        all_actor_loss = []
        all_critic_loss = []

        for tr in trajectories:
            rewards = tr["rewards"]
            states = np.asarray(tr["states"], dtype=np.float32)
            next_states = np.asarray(tr["next_states"], dtype=np.float32)
            masks_np = np.asarray(tr["masks"], dtype=bool)
            actions = tr["actions"]
            last_cities = tr["last_cities"]
            action_cities = tuple(tr.get("action_cities", self.env.action_cities))
            T = len(rewards)

            state_ts = torch.as_tensor(states, dtype=torch.float32)
            next_state_ts = torch.as_tensor(next_states, dtype=torch.float32)

            with torch.no_grad():
                values = torch.stack([self.critic(dist_t, state_ts[t]) for t in range(T)])
                next_values = torch.stack([self.critic(dist_t, next_state_ts[t]) for t in range(T)])

            targets = torch.zeros(T, dtype=torch.float32)
            for t in range(T):
                g = 0.0
                for k in range(n_step):
                    if t + k >= T:
                        break
                    g += (gamma ** k) * rewards[t + k]
                    if tr["dones"][t + k]:
                        break
                else:
                    idx = min(t + n_step - 1, T - 1)
                    g += (gamma ** n_step) * float(next_values[idx])
                targets[t] = g

            pred_values = torch.stack([self.critic(dist_t, state_ts[t]) for t in range(T)])
            critic_loss = F.mse_loss(pred_values, targets.detach())
            self.critic_opt.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), POINTER_MAX_GRAD_NORM)
            self.critic_opt.step()

            enc = self.actor.encode(dist_t)
            hx = self._init_decoder()

            advantages = (targets - values).detach()
            if advantages.numel() > 1:
                advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

            actor_loss = torch.tensor(0.0)
            for t in range(T):
                mask_t = torch.as_tensor(masks_np[t], dtype=torch.bool)
                logits, hx = self.actor.step(
                    enc,
                    hx,
                    last_city=last_cities[t],
                    mask=mask_t,
                    action_cities=action_cities,
                    first_step=(t == 0),
                )
                dist_obj = torch.distributions.Categorical(logits=logits)
                logp = dist_obj.log_prob(torch.as_tensor(actions[t], dtype=torch.long))
                entropy = dist_obj.entropy()
                actor_loss = actor_loss - logp * advantages[t] - ent_coef * entropy

            actor_loss = actor_loss / T
            self.actor_opt.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), POINTER_MAX_GRAD_NORM)
            self.actor_opt.step()

            all_actor_loss.append(float(actor_loss.detach()))
            all_critic_loss.append(float(critic_loss.detach()))

        return {
            "actor_loss": float(np.mean(all_actor_loss)),
            "critic_loss": float(np.mean(all_critic_loss)),
        }

    def save(self, path: str) -> None:
        torch.save(
            {
                "config": asdict(self.cfg),
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
            },
            path,
        )

    @classmethod
    def load(cls, path: str, env: StochasticTSPEnv) -> "PointerNetAgent":
        data = torch.load(path, map_location="cpu")
        cfg = PointerNetConfig(**{k: v for k, v in data["config"].items() if k in PointerNetConfig.__dataclass_fields__})
        agent = cls(env, cfg)
        agent.actor.load_state_dict(data["actor"])
        agent.critic.load_state_dict(data["critic"])
        return agent
