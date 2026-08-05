"""
Library_networks.py - Policy- and value-network building blocks.

Contents
--------
Value_NN            - State- or action-value critic (MLP).
Policy_NN           - Actor / policy network (MLP).
Pointer_Policy_NN   - Attention actor that points at a city (needs the enriched
                      observation; see ``Environment.observation_mode``).
Pointer_Value_NN    - Attention critic over the same per-city features.
build_actor         - Factory returning the actor class named by ``actor_arch``.
build_critic        - Matching factory for the critic.

Why the pointer variants exist
------------------------------
The MLP actor emits one logit per *action index*, and an action index is a fixed
slot in the non-depot city list. With the legacy 4-D observation the network has
no feature describing the city behind a slot, so "logit 37" is grounded in
nothing. The pointer actor instead reads the enriched observation's per-action
feature block, encodes it with permutation-invariant self-attention, and scores
each slot by attention compatibility with a context query - the action is chosen
*because of* the city's own features, which is the mechanism the Attention Model
(Kool et al. 2019) / POMO (Kwon et al. 2020) use.
"""
import math

import numpy as np
import torch
import torch.nn as nn


ACTOR_ARCHITECTURES = ("mlp", "pointer")


################[ Value_NN (Critic) ]################
class Value_NN(nn.Module):      # This is neural network for the critic (Values NN), which can be used as either a state-value critic V_phi(s) or an action-value critic Q_phi(s) depending on the output size and how it's trained.
    """State-value critic V_phi(s), or Q-value critic Q_phi(s) when used per-action."""
    def __init__(self, nn_hidden_layer_widths=np.array([64, 64]), output_size=1, input_size=4):
        super().__init__()
        hidden_widths = np.asarray(nn_hidden_layer_widths, dtype=np.int32).tolist()
        if len(hidden_widths) == 0:
            raise ValueError("nn_hidden_layer_widths must contain at least one hidden-layer width")

        layers = []
        input_size = int(input_size)  # TSP state dimension (4 in legacy observation mode)
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
    def __init__(self, nn_hidden_layer_widths=np.array([5]), output_size=1, input_size=4):     # output_size: 1 for binary (Bernoulli) actions, n_actions for categorical TSP next-city selection.
        super().__init__()
        hidden_widths = np.asarray(nn_hidden_layer_widths, dtype=np.int32).tolist()
        if len(hidden_widths) == 0:
            raise ValueError("nn_hidden_layer_widths must contain at least one hidden-layer width")

        layers = []
        input_layer_size = int(input_size)  # state dimension: 4 for the legacy TSP observation, n_actions*F+4 when enriched.
        for width in hidden_widths:
            layers.append(nn.Linear(input_layer_size, int(width)))
            layers.append(nn.ReLU())
            input_layer_size = int(width)
        layers.append(nn.Linear(input_layer_size, output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, state):
        return self.net(state)
####################################################################


################[ Attention (pointer) networks ]################
class _NodeNormalization(nn.Module):
    """Instance normalization over the node dimension (Kool 2019 / POMO).

    Skipped when there is a single node: ``InstanceNorm1d`` rejects a length-1
    spatial dimension in training mode, and normalizing one element is a no-op
    anyway.
    """

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.norm = nn.InstanceNorm1d(embed_dim, affine=True)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        if h.size(1) < 2:
            return h
        # h: (batch, n_nodes, embed) -> InstanceNorm1d wants (batch, embed, n_nodes)
        return self.norm(h.transpose(1, 2)).transpose(1, 2)


class _EncoderLayer(nn.Module):
    """One post-norm transformer block: MHA + residual, FF + residual."""

    def __init__(self, embed_dim: int, n_heads: int, ff_hidden: int) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)
        self.norm_attn = _NodeNormalization(embed_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, ff_hidden),
            nn.ReLU(),
            nn.Linear(ff_hidden, embed_dim),
        )
        self.norm_ff = _NodeNormalization(embed_dim)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attention(h, h, h, need_weights=False)
        h = self.norm_attn(h + attn_out)
        return self.norm_ff(h + self.feed_forward(h))


class _CityEncoder(nn.Module):
    """Linear per-city embedding followed by ``n_layers`` self-attention blocks.

    No positional encoding: the action slots are an unordered *set* of candidate
    cities, so the encoding is permutation-invariant by construction.
    """

    def __init__(self, city_feature_dim: int, embed_dim: int, n_heads: int,
                 n_layers: int, ff_hidden: int) -> None:
        super().__init__()
        self.init_embed = nn.Linear(city_feature_dim, embed_dim)
        self.layers = nn.ModuleList(
            _EncoderLayer(embed_dim, n_heads, ff_hidden) for _ in range(n_layers)
        )

    def forward(self, city_features: torch.Tensor) -> torch.Tensor:
        h = self.init_embed(city_features)
        for layer in self.layers:
            h = layer(h)
        return h


class _EnrichedObservationSplitter(nn.Module):
    """Shared plumbing: split the flat enriched observation into the per-action
    feature block ``(batch, n_actions, city_feature_dim)`` and the trailing
    global context scalars ``(batch, global_feature_dim)``.
    """

    def __init__(self, n_actions: int, city_feature_dim: int, global_feature_dim: int) -> None:
        super().__init__()
        if city_feature_dim <= 0:
            raise ValueError(
                "The pointer networks require the enriched observation "
                "(observation_mode='enriched'); the legacy 4-D observation has no "
                "per-city feature block to point at."
            )
        self.n_actions = int(n_actions)
        self.city_feature_dim = int(city_feature_dim)
        self.global_feature_dim = int(global_feature_dim)
        self.city_block_size = self.n_actions * self.city_feature_dim

    def split(self, state: torch.Tensor):
        """Returns ``(city_features, context, was_unbatched)``."""
        was_unbatched = state.dim() == 1
        if was_unbatched:
            state = state.unsqueeze(0)
        expected = self.city_block_size + self.global_feature_dim
        if state.size(-1) != expected:
            raise ValueError(
                f"Expected an enriched observation of width {expected} "
                f"({self.n_actions} actions x {self.city_feature_dim} features + "
                f"{self.global_feature_dim} global), got {state.size(-1)}."
            )
        city_features = state[:, : self.city_block_size].reshape(
            state.size(0), self.n_actions, self.city_feature_dim
        )
        context = state[:, self.city_block_size:]
        return city_features, context, was_unbatched


class Pointer_Policy_NN(_EnrichedObservationSplitter):
    """Attention actor: encode the candidate cities, then point at one.

    Emits one logit per *action slot*, so it is a drop-in replacement for
    ``Policy_NN`` everywhere the project builds an actor - the invalid-action
    mask and ``Categorical(logits=...)`` sampling are unchanged.

    Decoder follows Kool et al. (2019): a multi-head glimpse over the city
    embeddings, then a single-head pointer whose logits are ``C * tanh(.)``
    clipped (Bello's exploration trick). The visited flag that the enriched
    observation already carries per city is reused to mask the glimpse, so
    attention never spends mass on cities the agent cannot pick.
    """

    VISITED_FEATURE_INDEX = 2  # feature 2 of the enriched per-action block

    def __init__(self, n_actions, city_feature_dim, global_feature_dim=4,
                 embed_dim=128, n_heads=8, n_layers=2, ff_hidden=256,
                 clip_logits=10.0):
        super().__init__(n_actions, city_feature_dim, global_feature_dim)
        embed_dim = int(embed_dim)
        n_heads = int(n_heads)
        if embed_dim % n_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by n_heads ({n_heads})."
            )
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.clip_logits = float(clip_logits)

        self.encoder = _CityEncoder(
            self.city_feature_dim, embed_dim, n_heads, int(n_layers), int(ff_hidden)
        )
        # Context query = [graph embedding ; projected global scalars].
        self.context_proj = nn.Linear(self.global_feature_dim, embed_dim)
        self.W_query = nn.Linear(2 * embed_dim, embed_dim, bias=False)
        self.W_key_glimpse = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_val_glimpse = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_out = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_key_logit = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        city_features, context, was_unbatched = self.split(state)
        batch = city_features.size(0)

        h = self.encoder(city_features)                              # (B, A, d)
        graph_embed = h.mean(dim=1)                                  # (B, d)
        query_context = torch.cat((graph_embed, self.context_proj(context)), dim=-1)

        # Multi-head glimpse, masked by the visited flag the observation carries.
        visited = city_features[..., self.VISITED_FEATURE_INDEX] > 0.5   # (B, A)
        # A fully-visited row would make every score -inf and the softmax NaN;
        # leave such rows unmasked (the caller's argmax fallback handles them).
        visited = visited & ~visited.all(dim=1, keepdim=True)

        q = self.W_query(query_context).view(batch, 1, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.W_key_glimpse(h).view(batch, -1, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.W_val_glimpse(h).view(batch, -1, self.n_heads, self.head_dim).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)     # (B, H, 1, A)
        scores = scores.masked_fill(visited[:, None, None, :], float("-inf"))
        glimpse = torch.softmax(scores, dim=-1) @ v                       # (B, H, 1, hd)
        glimpse = glimpse.transpose(1, 2).reshape(batch, 1, self.embed_dim)
        glimpse = self.W_out(glimpse)

        # Single-head pointer with tanh clipping.
        logits = (glimpse @ self.W_key_logit(h).transpose(1, 2)).squeeze(1)
        logits = self.clip_logits * torch.tanh(logits / math.sqrt(self.embed_dim))
        return logits.squeeze(0) if was_unbatched else logits


class Pointer_Value_NN(_EnrichedObservationSplitter):
    """Attention critic V_phi(s) over the same per-city feature block.

    Permutation-invariant by construction (mean-pool over the encoded cities),
    which the MLP critic is not: it would have to learn that the ordering of the
    per-action block is arbitrary.
    """

    def __init__(self, n_actions, city_feature_dim, global_feature_dim=4,
                 embed_dim=128, n_heads=8, n_layers=2, ff_hidden=256,
                 output_size=1):
        super().__init__(n_actions, city_feature_dim, global_feature_dim)
        embed_dim = int(embed_dim)
        self.encoder = _CityEncoder(
            self.city_feature_dim, embed_dim, int(n_heads), int(n_layers), int(ff_hidden)
        )
        self.context_proj = nn.Linear(self.global_feature_dim, embed_dim)
        self.head = nn.Sequential(
            nn.Linear(2 * embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, int(output_size)),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        city_features, context, was_unbatched = self.split(state)
        h = self.encoder(city_features).mean(dim=1)
        value = self.head(torch.cat((h, self.context_proj(context)), dim=-1))
        return value.squeeze(0) if was_unbatched else value


def build_actor(*, actor_arch, n_actions, input_size, hidden_nn,
                city_feature_dim=0, global_feature_dim=4,
                embed_dim=128, n_heads=8, n_layers=2, ff_hidden=256,
                clip_logits=10.0):
    """Construct the actor named by ``actor_arch`` ('mlp' or 'pointer')."""
    actor_arch = str(actor_arch).lower()
    if actor_arch == "mlp":
        return Policy_NN(
            nn_hidden_layer_widths=hidden_nn,
            output_size=int(n_actions),
            input_size=int(input_size),
        )
    if actor_arch == "pointer":
        return Pointer_Policy_NN(
            n_actions=int(n_actions),
            city_feature_dim=int(city_feature_dim),
            global_feature_dim=int(global_feature_dim),
            embed_dim=embed_dim,
            n_heads=n_heads,
            n_layers=n_layers,
            ff_hidden=ff_hidden,
            clip_logits=clip_logits,
        )
    raise ValueError(
        f"Unknown actor_arch {actor_arch!r}; valid values are {ACTOR_ARCHITECTURES}."
    )


def build_critic(*, actor_arch, n_actions, input_size, hidden_nn, output_size=1,
                 city_feature_dim=0, global_feature_dim=4,
                 embed_dim=128, n_heads=8, n_layers=2, ff_hidden=256):
    """Construct the critic matching ``actor_arch`` ('mlp' or 'pointer')."""
    actor_arch = str(actor_arch).lower()
    if actor_arch == "mlp":
        return Value_NN(
            nn_hidden_layer_widths=hidden_nn,
            output_size=int(output_size),
            input_size=int(input_size),
        )
    if actor_arch == "pointer":
        return Pointer_Value_NN(
            n_actions=int(n_actions),
            city_feature_dim=int(city_feature_dim),
            global_feature_dim=int(global_feature_dim),
            embed_dim=embed_dim,
            n_heads=n_heads,
            n_layers=n_layers,
            ff_hidden=ff_hidden,
            output_size=int(output_size),
        )
    raise ValueError(
        f"Unknown actor_arch {actor_arch!r}; valid values are {ACTOR_ARCHITECTURES}."
    )


def actor_from_payload(payload, *, n_actions, fallback_hidden_nn=None):
    """Rebuild a saved actor from a checkpoint payload.

    Checkpoints written before the representation work carry none of the
    architecture keys; those default to the legacy MLP on the 4-D observation,
    so old files keep loading unchanged. The returned actor is tagged with
    ``tsp_observation_mode`` so the caller can put its evaluation environment
    into the representation the network was trained on.
    """
    hidden_nn = payload.get("actor_hidden_nn", fallback_hidden_nn)
    if hidden_nn is None:
        hidden_nn = [64, 64]
    observation_mode = str(payload.get("observation_mode", "legacy"))
    actor = build_actor(
        actor_arch=payload.get("actor_arch", "mlp"),
        n_actions=int(n_actions),
        input_size=int(payload.get("input_dim", 4)),
        hidden_nn=np.asarray(hidden_nn, dtype=np.int32),
        city_feature_dim=int(payload.get("city_feature_dim", 0)),
        embed_dim=int(payload.get("pointer_embed", 128)),
        n_heads=int(payload.get("pointer_n_heads", 8)),
        n_layers=int(payload.get("pointer_n_layers", 2)),
        ff_hidden=int(payload.get("pointer_ff_hidden", 256)),
    )
    actor.load_state_dict(payload["state_dict"])
    actor.eval()
    actor.tsp_observation_mode = observation_mode
    return actor, observation_mode
####################################################################
