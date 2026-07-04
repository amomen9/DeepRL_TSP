"""
Library_network_conversion.py - resize a trained MLP's weights to a different
hidden-layer architecture so an existing checkpoint can warm-start a network of
another size (e.g. continue a critic trained at ``[512, 512]`` into a freshly
built ``[128, 128]`` critic, or the reverse).

Both :class:`Library_networks.Value_NN` and :class:`Library_networks.Policy_NN`
are plain ``nn.Sequential(Linear, ReLU, ..., Linear)`` MLPs, which is exactly the
case the *Net2Net* transforms of Chen et al. (2016) handle:

  * **Widening** a hidden layer (e.g. 128 -> 512) is *function preserving*: each
    new unit copies an existing one, and the next layer's incoming weights for
    the replicated units are divided by their replication count, so the network
    computes the same function it did before - a clean warm-start.
  * **Narrowing** a hidden layer (e.g. 512 -> 128) cannot preserve the function:
    there is no exact inverse of widening, so we keep the first ``target`` units
    and drop the rest. This is a lossy warm-start, not an equivalence.

Only the hidden-layer *widths* may change. The input dimension, the output
dimension and the number of layers (depth) must match between source and target;
otherwise :class:`NetworkConversionError` is raised and the caller should train
from scratch instead.
"""

import numpy as np
import torch


class NetworkConversionError(ValueError):
    """Raised when a source ``state_dict`` cannot be resized onto the target
    architecture (different depth, input dim, output dim or module layout)."""


def _linear_chain(state_dict) -> list[tuple[str, str, str]]:
    """Return the ordered ``(base, weight_key, bias_key)`` triples of the Linear
    layers in a sequential MLP ``state_dict``.

    Relies on ``state_dict`` preserving module-registration order (it does for
    ``nn.Sequential``), so the returned list is in forward order."""
    chain = []
    for key in state_dict.keys():
        if not key.endswith(".weight"):
            continue
        base = key[: -len(".weight")]
        bias_key = base + ".bias"
        weight = state_dict[key]
        if hasattr(weight, "dim") and weight.dim() == 2 and bias_key in state_dict:
            chain.append((base, key, bias_key))
    return chain


def _chain_dims(chain, state_dict) -> tuple[int, list[int], int]:
    """Return ``(input_dim, hidden_widths, output_dim)`` for a linear chain."""
    input_dim = int(state_dict[chain[0][1]].shape[1])
    output_dim = int(state_dict[chain[-1][1]].shape[0])
    hidden = [int(state_dict[wk].shape[0]) for (_b, wk, _bk) in chain[:-1]]
    return input_dim, hidden, output_dim


def hidden_widths_of_state_dict(state_dict) -> list[int]:
    """Convenience accessor: the hidden-layer widths encoded in ``state_dict``."""
    chain = _linear_chain(state_dict)
    if len(chain) < 2:
        return []
    _in, hidden, _out = _chain_dims(chain, state_dict)
    return hidden


def _build_unit_mapping(src_width: int, tgt_width: int, rng) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the per-unit index mapping for resizing one hidden layer.

    Returns ``(index, counts)`` where ``index`` has length ``tgt_width`` with
    values in ``[0, src_width)`` (which source unit each target unit comes from),
    and ``counts`` has length ``src_width`` giving how many target units each
    source unit was replicated into (used for the function-preserving rescale).

    Widening keeps every original unit then samples extra units to replicate;
    narrowing keeps the first ``tgt_width`` units (lossy)."""
    if tgt_width >= src_width:
        extra = [int(rng.integers(0, src_width)) for _ in range(tgt_width - src_width)]
        index_list = list(range(src_width)) + extra
    else:
        index_list = list(range(tgt_width))  # narrow: keep the first units, drop the rest
    index = torch.tensor(index_list, dtype=torch.long)
    counts = torch.bincount(index, minlength=src_width).to(torch.float32).clamp(min=1.0)
    return index, counts


def convert_state_dict(
    source_state_dict,
    target_state_dict,
    *,
    seed: int = 0,
    symmetry_break_noise: float = 0.0,
) -> dict:
    """Resize ``source_state_dict`` onto the architecture of
    ``target_state_dict`` and return a new state dict that loads cleanly into the
    target model (``strict=True``).

    ``seed`` makes the widening replication reproducible. ``symmetry_break_noise``
    (>0) adds small Gaussian noise to the *duplicated* widened units to break the
    gradient symmetry between a unit and its copies; this trades exact function
    preservation for faster subsequent training and is off by default."""
    src_chain = _linear_chain(source_state_dict)
    tgt_chain = _linear_chain(target_state_dict)
    if len(src_chain) != len(tgt_chain) or len(src_chain) < 2:
        raise NetworkConversionError(
            f"Incompatible depth: source has {len(src_chain)} linear layers, "
            f"target has {len(tgt_chain)}."
        )
    if [b for b, _w, _bk in src_chain] != [b for b, _w, _bk in tgt_chain]:
        raise NetworkConversionError("Source and target module layouts differ.")

    s_in, s_hidden, s_out = _chain_dims(src_chain, source_state_dict)
    t_in, t_hidden, t_out = _chain_dims(tgt_chain, target_state_dict)
    if s_in != t_in or s_out != t_out or len(s_hidden) != len(t_hidden):
        raise NetworkConversionError(
            f"Incompatible shapes: source in/out/depth=({s_in},{s_out},{len(s_hidden)}) "
            f"vs target ({t_in},{t_out},{len(t_hidden)})."
        )

    rng = np.random.default_rng(seed)
    new = {k: v.detach().clone().float() for k, v in source_state_dict.items()}

    for i, (src_w, tgt_w) in enumerate(zip(s_hidden, t_hidden)):
        if src_w == tgt_w:
            continue
        _pb, prod_w, prod_b = src_chain[i]       # layer producing hidden unit i
        _cb, cons_w, _cons_b = src_chain[i + 1]  # next layer consuming it
        index, counts = _build_unit_mapping(src_w, tgt_w, rng)

        # Replicate / select the producing layer's output rows and bias.
        new[prod_w] = new[prod_w][index, :].clone()
        new[prod_b] = new[prod_b][index].clone()

        # Resize the consuming layer's input columns, dividing replicated columns
        # by their replication count so the layer's pre-activation is unchanged
        # (function preserving when widening; a no-op rescale when narrowing).
        col_scale = (1.0 / counts[index]).view(1, -1)
        new[cons_w] = (new[cons_w][:, index] * col_scale).clone()

        if symmetry_break_noise > 0 and tgt_w > src_w:
            duplicated = torch.arange(tgt_w) >= src_w
            if duplicated.any():
                noise = torch.randn_like(new[prod_w]) * symmetry_break_noise
                noise[~duplicated] = 0.0
                new[prod_w] = new[prod_w] + noise

    converted = {}
    for key, target_tensor in target_state_dict.items():
        if key not in new:
            raise NetworkConversionError(f"Source checkpoint is missing parameter '{key}'.")
        resized = new[key]
        if tuple(resized.shape) != tuple(target_tensor.shape):
            raise NetworkConversionError(
                f"Post-conversion shape mismatch for '{key}': "
                f"{tuple(resized.shape)} vs target {tuple(target_tensor.shape)}."
            )
        converted[key] = resized.to(dtype=target_tensor.dtype)
    return converted


def convert_state_dict_into_model(
    *,
    source_state_dict,
    model: torch.nn.Module,
    seed: int = 0,
    symmetry_break_noise: float = 0.0,
) -> dict:
    """Resize ``source_state_dict`` onto ``model``'s architecture and load it in
    place (``strict=True``). Returns the converted state dict that was loaded."""
    target_state = model.state_dict()
    converted = convert_state_dict(
        source_state_dict,
        target_state,
        seed=seed,
        symmetry_break_noise=symmetry_break_noise,
    )
    for key in converted:
        converted[key] = converted[key].to(device=target_state[key].device)
    model.load_state_dict(converted, strict=True)
    return converted


# ── Down conversion (narrowing only) ─────────────────────────────────────────
#
# The checkpoint loader (:mod:`Library_checkpointing`) reuses a saved network of a
# different size *only* when the saved (disk) network is at least as wide as the
# configured architecture, in which case it is narrowed down to fit. A saved
# network that is smaller than the configured one is never widened up - that trial
# trains from scratch instead. The helpers below enforce that one-directional
# policy on top of the general (widen-or-narrow) ``convert_state_dict`` above.

def _hidden_widths_geq(source_widths, target_widths) -> bool:
    """True when ``source_widths`` and ``target_widths`` have the same depth and
    every source hidden layer is at least as wide as the matching target layer."""
    return (
        len(source_widths) == len(target_widths)
        and all(int(s) >= int(t) for s, t in zip(source_widths, target_widths))
    )


def down_convert_state_dict(source_state_dict, target_state_dict) -> dict:
    """Narrow a *wider-or-equal* source MLP onto ``target_state_dict``'s smaller
    architecture and return a state dict that loads cleanly into the target
    (``strict=True``).

    This is the lossy narrowing branch of :func:`convert_state_dict` made explicit
    and one-directional: every hidden layer of the source must be at least as wide
    as the target's (with matching depth, input dim and output dim). If any target
    layer is *wider* than the source - which would require function-preserving
    widening - :class:`NetworkConversionError` is raised, because the project's
    checkpoint policy is to train such a trial from scratch rather than widen a
    smaller saved network up to the configured size."""
    src_chain = _linear_chain(source_state_dict)
    tgt_chain = _linear_chain(target_state_dict)
    if len(src_chain) < 2 or len(tgt_chain) < 2:
        raise NetworkConversionError(
            f"Both networks must have at least two linear layers "
            f"(source has {len(src_chain)}, target has {len(tgt_chain)})."
        )
    _s_in, s_hidden, _s_out = _chain_dims(src_chain, source_state_dict)
    _t_in, t_hidden, _t_out = _chain_dims(tgt_chain, target_state_dict)
    if not _hidden_widths_geq(s_hidden, t_hidden):
        raise NetworkConversionError(
            f"Not a down conversion: source hidden widths {s_hidden} are not all "
            f">= target widths {t_hidden}; widening is disallowed (train from scratch)."
        )
    # Source is wider-or-equal everywhere, so convert_state_dict only ever narrows
    # or no-ops per layer here; seed / symmetry-break noise are irrelevant (no
    # widening replication happens).
    return convert_state_dict(source_state_dict, target_state_dict)


def down_convert_state_dict_into_model(*, source_state_dict, model: torch.nn.Module) -> dict:
    """Down-convert (narrow) ``source_state_dict`` onto ``model``'s smaller-or-equal
    architecture and load it in place (``strict=True``). Raises
    :class:`NetworkConversionError` if the source is narrower than ``model`` in any
    hidden layer. Returns the converted state dict that was loaded."""
    target_state = model.state_dict()
    converted = down_convert_state_dict(source_state_dict, target_state)
    for key in converted:
        converted[key] = converted[key].to(device=target_state[key].device)
    model.load_state_dict(converted, strict=True)
    return converted
