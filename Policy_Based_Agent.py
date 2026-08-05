"""
Policy_Based_Agent.py - Base policy-gradient TSP agent.

Mirrors the CartPole_PolicyBased BaseAgent in structure: builds an actor
(categorical policy over the n-1 next-city actions) and an optional V or Q
critic, and provides shared utilities (action sampling, discounted returns,
greedy evaluation, episode roll-out). Subclasses implement only `update()`;
the generic episode loop in `solve()` walks the DP_Table-backed environment
until all cities are visited.
"""

import numpy as np
import torch

from Base_Agent import Base_Agent
from Library.Library_networks import build_actor, build_critic


class TSP_Policy_Based_Agent(Base_Agent):
    """Base agent for policy-gradient methods on TSP.

    Subclasses must override `update()`.

    The actor produces logits over n_actions = n-1 next-city actions -- either a
    Policy_NN MLP or, when ``actor_arch='pointer'``, an attention network that
    points at a city using the enriched observation's per-city features. The
    critic mirrors that choice:
        - V_phi(s) for value-based methods
        - Q_phi(s) when an explicit Q critic is desired

    The observation width is read from the environment
    (``env.observation_dim``), so the same agent works with the legacy 4-D state
    and with the enriched one; see ``Environment.observation_mode``.
    """

    n_observations = 4  # legacy default; the instance value comes from the env

    def __init__(self, env,
                 actor_hidden_nn=np.array([16, 16]),
                 critic_hidden_nn=np.array([64, 64]),
                 actor_lr=0.001, critic_lr=0.001,
                 gamma=0.99, use_critic=False, critic_type='V',
                 actor_arch='mlp',
                 pointer_embed=128, pointer_n_heads=8, pointer_n_layers=2,
                 pointer_ff_hidden=256, pointer_clip_logits=10.0):
        super().__init__(env)
        self.n_actions = env.n_actions

        # Observation geometry comes from the env, not a hard-coded 4.
        self.n_observations = int(getattr(env, "observation_dim", 4))
        self.city_feature_dim = int(getattr(env, "city_feature_dim", 0))
        self.observation_mode = str(getattr(env, "observation_mode", "legacy"))
        self.actor_arch = str(actor_arch).lower()

        net_kwargs = dict(
            actor_arch=self.actor_arch,
            n_actions=self.n_actions,
            input_size=self.n_observations,
            city_feature_dim=self.city_feature_dim,
            embed_dim=pointer_embed,
            n_heads=pointer_n_heads,
            n_layers=pointer_n_layers,
            ff_hidden=pointer_ff_hidden,
        )

        # Actor (policy network pi_theta) -- categorical over n_actions next-city choices
        self.actor = build_actor(
            hidden_nn=actor_hidden_nn, clip_logits=pointer_clip_logits, **net_kwargs
        )
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

        # Critic setup mirrors CartPole BaseAgent: V for advantage methods, Q for explicit Q critic
        self.use_critic = use_critic
        self.critic_type = critic_type

        if use_critic:
            output_size = 1 if critic_type == 'V' else self.n_actions
            self.critic = build_critic(
                hidden_nn=critic_hidden_nn, output_size=output_size, **net_kwargs
            )
            self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        else:
            self.critic = None
            self.critic_optimizer = None

        self.gamma = gamma

    def _get_invalid_action_mask(self):
        """Return a boolean mask for actions that would revisit a city."""
        mask = torch.zeros(self.n_actions, dtype=torch.bool)
        visited_cities = getattr(self.env, "current_visited_cities", None)
        current_location = getattr(self.env, "current_location", None)

        if not visited_cities:
            return mask

        # Build the action->city map once (O(n)). Calling env._action_to_city
        # per action rebuilds the available-cities list every time, making this
        # O(n^2) per step; available_cities[action] is exactly that mapping and
        # honours the active (possibly non-zero, cycling) depot.
        available_cities = self.env._available_cities()
        for action, next_city in enumerate(available_cities):
            if next_city in visited_cities or next_city == current_location:
                mask[action] = True

        return mask

    def select_action(self, obs, mask=None):
        """Sample action a ~ pi_theta(a|s). Returns (action, log_prob).

        ``mask`` may be a precomputed invalid-action mask for ``obs`` (the
        on-policy loops capture it before stepping the env); when omitted it is
        computed here. Passing it avoids recomputing the mask twice per step.
        """
        state = torch.as_tensor(obs, dtype=torch.float32)
        logits = self.actor(state)
        if mask is None:
            mask = self._get_invalid_action_mask()

        if bool(torch.all(mask)):
            action = torch.argmax(logits)
            log_prob = torch.log_softmax(logits, dim=-1)[action]
        else:
            masked_logits = logits.masked_fill(mask, -1e9)
            dist = torch.distributions.Categorical(logits=masked_logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)

        return int(action.item()), log_prob.squeeze()

    def get_value(self, obs):
        """Return V_phi(s) (scalar) or Q_phi(s) (vector) from the critic."""
        if self.critic is None:
            return torch.tensor(0.0)
        state = torch.as_tensor(obs, dtype=torch.float32)
        return self.critic(state).squeeze()

    def update(self):
        raise NotImplementedError("Subclasses must implement their specific update method")

    def compute_discounted_returns(self, rewards):
        """Compute G_t = sum_{k=0}^{T-t-1} gamma^k * r_{t+k} for each timestep."""
        T = len(rewards)
        returns = torch.zeros(T)
        G = 0.0
        for t in reversed(range(T)):
            G = float(rewards[t]) + self.gamma * G
            returns[t] = G
        return returns

    def greedy_action(self, obs):
        """argmax over the masked policy logits for ``obs`` (no gradient)."""
        with torch.no_grad():
            logits = self.actor(torch.as_tensor(obs, dtype=torch.float32))
            mask = self._get_invalid_action_mask()
            if bool(torch.all(mask)):
                return int(torch.argmax(logits).item())
            return int(torch.argmax(logits.masked_fill(mask, -1e9)).item())

    def _greedy_rollout(self, obs, first_action=None):
        """Greedy rollout from the env's current state; returns the episode return."""
        ep_return = 0.0
        done, truncated = False, False
        if first_action is not None:
            obs, reward, done, truncated, _ = self.env.step(int(first_action))
            ep_return += reward
        while not (done or truncated):
            obs, reward, done, truncated, _ = self.env.step(self.greedy_action(obs))
            ep_return += reward
        return ep_return

    def evaluate(self, n_eval_episodes=10, n_starts=1):
        """Evaluate the current policy greedily (argmax over masked logits).

        ``n_starts > 1`` is the multi-start decode of POMO (Kwon et al., 2020)
        adapted to this env: within one evaluation episode the depot is pinned
        and the greedy rollout is repeated with the first move *forced* to each
        of ``n_starts`` evenly spaced action slots, keeping the best tour. A
        single greedy trajectory is one sample of a deterministic policy; the
        best of several diverse starts is a far better estimate of what the
        policy can actually construct. Cost scales linearly in ``n_starts``
        (``n_starts * n`` env steps per evaluation episode), so it is off by
        default. ``n_starts <= 1`` reproduces the original single rollout
        exactly, including the per-episode depot cycling.
        """
        n_starts = max(1, int(n_starts))
        total_returns = []
        for _ in range(n_eval_episodes):
            if n_starts == 1:
                obs, _ = self.env.reset()
                total_returns.append(self._greedy_rollout(obs))
                continue

            # Pin the depot across the start sweep so every start solves the same
            # episode, then leave it where a single reset would have left it.
            pinned_depot = int(self.env.depot_city)
            starts = np.unique(
                np.linspace(0, self.n_actions - 1, num=min(n_starts, self.n_actions))
                .round().astype(int)
            )
            best_return = -np.inf
            for first_action in starts:
                self.env.depot_city = pinned_depot
                obs, _ = self.env.reset()
                mask = self._get_invalid_action_mask()
                if bool(mask[int(first_action)]):
                    continue  # slot is the depot / already used under this depot
                best_return = max(best_return, self._greedy_rollout(obs, first_action))
            if not np.isfinite(best_return):
                self.env.depot_city = pinned_depot
                obs, _ = self.env.reset()
                best_return = self._greedy_rollout(obs)
            self.env.depot_city = (pinned_depot + 1) % self.env.n
            total_returns.append(best_return)
        return float(np.mean(total_returns))

    def _rollout_episode(self):
        """Walk the env once until all cities are visited (done=True) or truncated.

        Returns a dict of trajectory tensors/lists used by the algorithm-specific
        update methods.
        """
        obs, _ = self.env.reset()
        states, actions, rewards = [], [], []
        log_probs, next_states, dones = [], [], []
        tour = [self.env.current_location]

        done, truncated = False, False
        while not (done or truncated):
            action, log_prob = self.select_action(obs)
            next_obs, reward, done, truncated, _ = self.env.step(action)

            states.append(obs)
            actions.append(action)
            rewards.append(float(reward))
            log_probs.append(log_prob)
            next_states.append(next_obs)
            dones.append(bool(done or truncated))

            obs = next_obs
            tour.append(self.env.current_location)

        return {
            "states": states,
            "actions": actions,
            "rewards": rewards,
            "log_probs": log_probs,
            "next_states": next_states,
            "dones": dones,
            "tour": tour,
            "terminated": bool(done) and not bool(truncated),
        }

    def solve(self, n_episodes=500, eval_interval=50):
        """Generic policy-gradient training loop.

        Each episode walks the DP_Table-backed environment until the agent has
        visited every city (env terminates when len(visited) == n). After each
        episode, the algorithm-specific `update()` is called.

        Returns
        -------
        (eval_returns, n_eval_episodes) : tuple of lists
            Greedy-evaluation returns and the episode indices at which they
            were recorded. The best valid tour discovered during training is
            stored in ``self.optimal_tours`` / ``self.optimal_cost``.
        """
        eval_returns, n_eval_episodes = [], []
        best_cost = float("inf")
        best_tour = None

        for ep in range(n_episodes):
            traj = self._rollout_episode()
            self.update(**{k: v for k, v in traj.items() if k != "tour" and k != "terminated"})

            if traj["terminated"] and self.env.validate_tour(traj["tour"]):
                cost = float(self.env.tour_cost(traj["tour"]))
                if cost < best_cost:
                    best_cost = cost
                    best_tour = list(traj["tour"])

            if (ep + 1) % eval_interval == 0:
                eval_returns.append(self.evaluate(n_eval_episodes=5))
                n_eval_episodes.append(ep + 1)

        if best_tour is not None:
            self.optimal_cost = best_cost
            self.optimal_tours = [best_tour]

        return eval_returns, n_eval_episodes
