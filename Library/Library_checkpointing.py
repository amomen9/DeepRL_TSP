"""
Library_checkpointing.py - checkpoint persistence, reuse and continuation for the
TSP project.

Adapted port of the CartPole fork's ``Checkpointing.py``
(https://github.com/amomen9/CartPole-v1-PolicyBased-pytorch). The CartPole
version names checkpoints by network-architecture signature and saves a bare
``state_dict``. This TSP version keeps the project's existing layout and payload
shape so the legacy evaluator (:mod:`Use_Trained_Model`) keeps working:

  * Files live at ``Checkpoints/TSP/<ALGO>/actor_rep{N}[_S{id}].pt`` (and the
    matching ``critic_rep{N}[_S{id}].pt``), exactly as before.
  * Each ``.pt`` stores the project's dict payload ``{"state_dict", "n_actions",
    "actor_hidden_nn", ...}`` rather than a bare ``state_dict``.
  * A ``.txt`` JSON metadata *sidecar* is written next to every ``.pt`` recording
    the hyperparameters and the cumulative ``n_timesteps``.

On top of that it adds the CartPole reuse/continuation machinery:

  * exact-match and loose-match (``skip_selection_hyperparameter_match``)
    resolution of an existing checkpoint from its sidecar;
  * atomic, non-overwriting saves;
  * ``load_*_for_continuation`` / ``save_continuation_or_new`` to *resume*
    training from a saved checkpoint and accumulate the timestep counter.

Mandatory TSP adaptation: the strict-field gate matches on ``n_actions`` (which
encodes the TSP instance size) instead of the CartPole training-truncation
field, because an actor trained for a different number of cities is
weight-incompatible with the current environment.
"""

import json
import os
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch


@dataclass(frozen=True)
class CheckpointPaths:
    dir_path: str
    file_path: str

    def ensure_dir(self) -> None:
        os.makedirs(self.dir_path, exist_ok=True)


def _as_int_vector(x: Sequence[int] | np.ndarray) -> np.ndarray:
    arr = np.asarray(x)
    if arr.size == 0:
        raise ValueError("Hidden-layer widths must be non-empty.")
    arr = np.asarray(arr, dtype=np.int64).reshape(-1)
    return arr


def architecture_signature(hidden_layer_widths: Sequence[int] | np.ndarray) -> str:
    """Stable string signature based on hidden-layer widths only.
    Example: [64, 64] -> '64-64'."""
    arr = _as_int_vector(hidden_layer_widths)
    return "-".join(str(int(v)) for v in arr)


# ── TSP checkpoint path helpers ────────────────────────────────────────────
#
# Unlike the CartPole fork, the TSP project keys checkpoints by repetition index
# and sweep-setting suffix (``actor_rep{N}[_S{id}]``), so the path helpers build
# those stems instead of architecture-signature stems.

def _tsp_algo_dir(algo_type: str) -> str:
    return os.path.join("Checkpoints", "TSP", str(algo_type).upper())


def _rep_stem(component: str, rep_index: int, checkpoint_suffix: str | None) -> str:
    suffix = f"_{checkpoint_suffix}" if checkpoint_suffix else ""
    return f"{component}_rep{int(rep_index)}{suffix}"


def tsp_actor_checkpoint_path(
    *, algo_type: str, rep_index: int, checkpoint_suffix: str | None = None
) -> CheckpointPaths:
    dir_path = _tsp_algo_dir(algo_type)
    file_path = os.path.join(dir_path, f"{_rep_stem('actor', rep_index, checkpoint_suffix)}.pt")
    return CheckpointPaths(dir_path=dir_path, file_path=file_path)


def tsp_critic_checkpoint_path(
    *, algo_type: str, rep_index: int, checkpoint_suffix: str | None = None
) -> CheckpointPaths:
    dir_path = _tsp_algo_dir(algo_type)
    file_path = os.path.join(dir_path, f"{_rep_stem('critic', rep_index, checkpoint_suffix)}.pt")
    return CheckpointPaths(dir_path=dir_path, file_path=file_path)


# ── Metadata sidecars ───────────────────────────────────────────────────────

def _metadata_path_for_checkpoint(checkpoint_path: str) -> str:
    root, _ext = os.path.splitext(checkpoint_path)
    return f"{root}.txt"


def _normalize_metadata_value(value):
    if isinstance(value, np.ndarray):
        return [_normalize_metadata_value(item) for item in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_normalize_metadata_value(item) for item in value]
    if isinstance(value, set):
        return sorted(_normalize_metadata_value(item) for item in value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if torch.is_tensor(value):
        if value.ndim == 0:
            return _normalize_metadata_value(value.item())
        return _normalize_metadata_value(value.detach().cpu().tolist())
    return value


def _normalize_metadata(metadata):
    if metadata is None:
        return None
    if not isinstance(metadata, dict):
        raise TypeError("metadata must be a dict or None")
    return {str(key): _normalize_metadata_value(value) for key, value in metadata.items()}


def _metadata_to_text(metadata) -> str:
    normalized = _normalize_metadata(metadata)
    if normalized is None:
        return ""
    return json.dumps(normalized, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def _read_metadata_file(metadata_path: str):
    if not os.path.isfile(metadata_path):
        return None
    with open(metadata_path, "r", encoding="utf-8") as f:
        raw_text = f.read().strip()
    if not raw_text:
        return {}
    parsed = json.loads(raw_text)
    if not isinstance(parsed, dict):
        raise ValueError(f"Checkpoint metadata file '{metadata_path}' does not contain a JSON object.")
    return _normalize_metadata(parsed)


def _write_metadata_file(metadata_path: str, metadata) -> None:
    tmp_path = metadata_path + f".tmp_{os.getpid()}"
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(_metadata_to_text(metadata))
    os.replace(tmp_path, metadata_path)


# ── Candidate enumeration ───────────────────────────────────────────────────

def _checkpoint_stem(checkpoint_path: str) -> str:
    return os.path.splitext(os.path.basename(checkpoint_path))[0]


def _checkpoint_index_for_candidate(candidate_path: str, base_stem: str) -> int:
    candidate_stem = _checkpoint_stem(candidate_path)
    if candidate_stem == base_stem:
        return 0
    prefix = f"{base_stem}_"
    if candidate_stem.startswith(prefix):
        suffix = candidate_stem[len(prefix):]
        if suffix.isdigit():
            return int(suffix)
    return -1


def _iter_candidate_checkpoint_paths(checkpoint_path: str):
    base_dir = os.path.dirname(checkpoint_path)
    base_stem = _checkpoint_stem(checkpoint_path)
    seen: set[str] = set()

    if os.path.isfile(checkpoint_path):
        seen.add(os.path.abspath(checkpoint_path))
        yield checkpoint_path

    if not os.path.isdir(base_dir):
        return

    prefix = f"{base_stem}_"
    for name in sorted(os.listdir(base_dir)):
        if not name.endswith(".pt"):
            continue
        candidate_path = os.path.join(base_dir, name)
        abs_candidate = os.path.abspath(candidate_path)
        if abs_candidate in seen:
            continue
        candidate_stem = _checkpoint_stem(candidate_path)
        if candidate_stem == base_stem or candidate_stem.startswith(prefix):
            seen.add(abs_candidate)
            yield candidate_path


def _resolve_non_overwriting_path(checkpoint_path: str) -> str:
    if not os.path.exists(checkpoint_path):
        return checkpoint_path

    root, ext = os.path.splitext(checkpoint_path)
    index = 1
    while True:
        candidate = f"{root}_{index}{ext}"
        if not os.path.exists(candidate):
            return candidate
        index += 1


_LOOSE_TIMESTEPS_KEYS = ("n_timesteps", "n_env_steps")

# Instance-identity fields. When ``match_training_matrices`` is enabled the
# checkpoint metadata carries the full instance matrices as text (provenance)
# plus a hash that is the value actually compared. A checkpoint trained on a
# different TSP instance must never be reused, so the hash is enforced even under
# loose matching (it joins the strict-field gate below). The full text is
# excluded from exact-match equality since the hash already represents it.
_INSTANCE_HASH_KEY = "instance_matrix_hash"
_INSTANCE_TEXT_KEY = "instance_matrices_text"

# Keys recorded in the sidecar for provenance / loose-match preference but
# irrelevant to weight compatibility, so excluded from exact-match equality.
_EXACT_MATCH_EXCLUDED_KEYS = (
    "use_saved_disk_networks_checkpoints",
    _INSTANCE_TEXT_KEY,
)


def _strip_exact_match_excluded(metadata: dict | None) -> dict | None:
    if not isinstance(metadata, dict):
        return metadata
    return {k: v for k, v in metadata.items() if k not in _EXACT_MATCH_EXCLUDED_KEYS}


def _extract_loose_timesteps(candidate_metadata: dict | None) -> int | None:
    if not isinstance(candidate_metadata, dict):
        return None
    for key in _LOOSE_TIMESTEPS_KEYS:
        if key in candidate_metadata:
            try:
                return int(candidate_metadata[key])
            except (TypeError, ValueError):
                continue
    return None


def _get_n_actions(metadata: dict | None):
    """TSP strict-match field: the size of the action space, which encodes the
    TSP instance size. A checkpoint trained for a different ``n_actions`` is
    weight-incompatible and must never be loaded into the current actor."""
    if not isinstance(metadata, dict):
        return None
    return metadata.get("n_actions")


# Network-architecture fields that determine weight tensor shapes. A checkpoint
# whose hidden-layer sizes differ from the current model is weight-incompatible
# (``load_state_dict`` would raise a size mismatch), so these are gated even
# under loose matching, exactly like ``n_actions``.
_ARCHITECTURE_STRICT_KEYS = ("actor_hidden_nn", "critic_hidden_nn")


def _n_actions_match(candidate_metadata, target_metadata) -> bool:
    """The unconditional weight-compatibility gate: a checkpoint trained for a
    different action-space size can never be loaded into the current model, with
    or without architecture resizing."""
    if not isinstance(candidate_metadata, dict) or not isinstance(target_metadata, dict):
        return False
    return _get_n_actions(candidate_metadata) == _get_n_actions(target_metadata)


def _instance_hash_match(candidate_metadata, target_metadata) -> bool:
    """Instance-identity gate. When the target records an instance-matrix hash
    (i.e. ``match_training_matrices`` was on for this run), a candidate must carry
    the identical hash: a checkpoint trained on a different TSP instance must
    never be reused, even under loose matching. When the target has no hash
    (matrix matching disabled), this gate is a no-op."""
    if not isinstance(target_metadata, dict):
        return False
    target_hash = target_metadata.get(_INSTANCE_HASH_KEY)
    if target_hash is None:
        return True
    if not isinstance(candidate_metadata, dict):
        return False
    return candidate_metadata.get(_INSTANCE_HASH_KEY) == target_hash


def _candidate_passes_strict_fields(candidate_metadata, target_metadata) -> bool:
    """Strict-match gate enforced under any circumstances (exact or loose):
    the candidate's ``n_actions`` and hidden-layer architecture must equal the
    target's (both determine weight-shape compatibility), and -- when the run
    enabled matrix matching -- its instance-matrix hash must match too."""
    if not _n_actions_match(candidate_metadata, target_metadata):
        return False
    if not _instance_hash_match(candidate_metadata, target_metadata):
        return False
    for key in _ARCHITECTURE_STRICT_KEYS:
        if key in target_metadata and candidate_metadata.get(key) != target_metadata.get(key):
            return False
    return True


def _component_hidden_widths(metadata) -> tuple[int, ...] | None:
    """The hidden-layer widths of the network this sidecar describes: the critic
    widths for a Critic checkpoint, otherwise the actor widths. Returned as a
    tuple of ints so two architectures can be compared for depth/width."""
    if not isinstance(metadata, dict):
        return None
    component = str(metadata.get("component", "")).lower()
    key = "critic_hidden_nn" if component == "critic" else "actor_hidden_nn"
    value = metadata.get(key)
    if value is None:
        value = metadata.get("actor_hidden_nn", metadata.get("critic_hidden_nn"))
    if not isinstance(value, (list, tuple)):
        return None
    try:
        return tuple(int(x) for x in value)
    except (TypeError, ValueError):
        return None


def _is_down_convertible(
    candidate_widths: tuple[int, ...] | None,
    target_widths: tuple[int, ...] | None,
) -> bool:
    """True when a *wider-or-equal* candidate can be narrowed (down-converted) onto
    the target: same depth, and every candidate hidden layer at least as wide as
    the matching target layer. A candidate that is narrower in any layer would
    require widening the saved network up to the configured size, which the project
    forbids - such a trial trains from scratch instead."""
    if not candidate_widths or not target_widths:
        return False
    if len(candidate_widths) != len(target_widths):
        return False
    return all(c >= t for c, t in zip(candidate_widths, target_widths))


def _candidate_is_convertible(candidate_metadata, target_metadata) -> bool:
    """Down-conversion gate: the candidate can be *narrowed* onto the target
    architecture. Requires the same ``n_actions``, the same instance-matrix hash
    (when matrix matching is on), the same number of hidden layers (depth), and
    every candidate hidden layer at least as wide as the target's. A candidate
    smaller than the target in any layer is rejected here (the trial then trains
    from scratch) - only down conversions are allowed, never widening."""
    if not _n_actions_match(candidate_metadata, target_metadata):
        return False
    if not _instance_hash_match(candidate_metadata, target_metadata):
        return False
    candidate_widths = _component_hidden_widths(candidate_metadata)
    target_widths = _component_hidden_widths(target_metadata)
    return _is_down_convertible(candidate_widths, target_widths)


def has_strict_field_candidate(*, checkpoint_path: str, target_metadata: dict) -> bool:
    """True if at least one candidate sidecar matches the strict field
    (``n_actions``) of ``target_metadata``. Used by the up-front orchestrator
    report to decide whether to announce a load."""
    if not isinstance(target_metadata, dict):
        return False
    target = _normalize_metadata(target_metadata)
    base_stem = _checkpoint_stem(checkpoint_path)
    for candidate_path in _iter_candidate_checkpoint_paths(checkpoint_path):
        index = _checkpoint_index_for_candidate(candidate_path, base_stem)
        if index < 0:
            continue
        metadata_path = _metadata_path_for_checkpoint(candidate_path)
        try:
            candidate_metadata = _read_metadata_file(metadata_path)
        except Exception:
            continue
        if _candidate_passes_strict_fields(candidate_metadata, target):
            return True
    return False


def _resolve_loose_checkpoint_path(checkpoint_path: str, target_metadata: dict | None) -> str | None:
    """Two-tier loose lookup among candidates sharing the rep/suffix filename.
    The strict-field gate (``n_actions``) is enforced first.

      1. Prefer candidates whose sidecar has
         ``use_saved_disk_networks_checkpoints == True`` (produced by a
         continuation run): pick the one with the largest ``n_timesteps``.
      2. Otherwise pick the strict-passing candidate with the largest
         ``n_timesteps``.
    Ties broken by max index then max mtime."""
    if not isinstance(target_metadata, dict):
        return None
    target = _normalize_metadata(target_metadata)
    base_stem = _checkpoint_stem(checkpoint_path)
    preferred: list[tuple[int, int, float, str]] = []
    fallback: list[tuple[int, int, float, str]] = []
    for candidate_path in _iter_candidate_checkpoint_paths(checkpoint_path):
        index = _checkpoint_index_for_candidate(candidate_path, base_stem)
        if index < 0:
            continue
        metadata_path = _metadata_path_for_checkpoint(candidate_path)
        try:
            candidate_metadata = _read_metadata_file(metadata_path)
        except Exception:
            continue
        if not _candidate_passes_strict_fields(candidate_metadata, target):
            continue
        timesteps = _extract_loose_timesteps(candidate_metadata)
        if timesteps is None:
            continue
        try:
            mtime = os.path.getmtime(candidate_path)
        except OSError:
            mtime = 0.0
        entry = (timesteps, index, mtime, candidate_path)
        if (
            isinstance(candidate_metadata, dict)
            and bool(candidate_metadata.get("use_saved_disk_networks_checkpoints", False))
        ):
            preferred.append(entry)
        else:
            fallback.append(entry)

    pool = preferred if preferred else fallback
    if not pool:
        return None
    pool.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    return pool[0][3]


def resolve_matching_checkpoint_path(
    *,
    checkpoint_path: str,
    metadata: dict | None = None,
    skip_selection_hyperparameter_match: bool = False,
) -> str | None:
    """Resolve the checkpoint file that matches ``metadata``.

    If ``metadata`` is None, behaves like the legacy path-based lookup.
    If ``skip_selection_hyperparameter_match`` is True, ignore the exact metadata
    match and pick the strict-passing candidate with the largest ``n_timesteps``.
    Otherwise require a sidecar whose JSON matches ``metadata`` exactly after
    normalization (excluding provenance-only keys)."""
    if skip_selection_hyperparameter_match:
        return _resolve_loose_checkpoint_path(checkpoint_path, metadata)

    if metadata is None:
        return checkpoint_path if os.path.isfile(checkpoint_path) else None

    target_metadata = _strip_exact_match_excluded(_normalize_metadata(metadata))
    if target_metadata is None:
        return checkpoint_path if os.path.isfile(checkpoint_path) else None

    candidates = []
    base_stem = _checkpoint_stem(checkpoint_path)
    for candidate_path in _iter_candidate_checkpoint_paths(checkpoint_path):
        index = _checkpoint_index_for_candidate(candidate_path, base_stem)
        if index < 0:
            continue
        metadata_path = _metadata_path_for_checkpoint(candidate_path)
        try:
            candidate_metadata = _read_metadata_file(metadata_path)
        except Exception:
            continue
        if _strip_exact_match_excluded(candidate_metadata) == target_metadata:
            try:
                mtime = os.path.getmtime(candidate_path)
            except OSError:
                mtime = 0.0
            candidates.append((index, mtime, candidate_path))

    if not candidates:
        return None

    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return candidates[0][2]


def resolve_convertible_checkpoint_path(
    checkpoint_path: str, target_metadata: dict | None
) -> tuple[str | None, tuple[int, ...] | None]:
    """Find the best checkpoint to *down-convert* (narrow) from when no same-size
    checkpoint exists.

    Candidates must share the target's ``n_actions`` and hidden-layer *depth* and
    be at least as wide as the target in every hidden layer; a saved network that
    is strictly smaller in any layer is not reused (its trial trains from scratch).
    Preference, highest first:
      1. exact-architecture candidates (a no-op conversion / direct load),
      2. continuation-produced candidates (``use_saved_disk_networks_checkpoints``),
      3. the most-trained candidate (largest ``n_timesteps``),
    ties broken by candidate index then mtime. Returns ``(path, source_widths)``
    or ``(None, None)``."""
    if not isinstance(target_metadata, dict):
        return None, None
    target = _normalize_metadata(target_metadata)
    target_widths = _component_hidden_widths(target)
    base_stem = _checkpoint_stem(checkpoint_path)

    best_key = None
    best_path = None
    best_widths = None
    for candidate_path in _iter_candidate_checkpoint_paths(checkpoint_path):
        index = _checkpoint_index_for_candidate(candidate_path, base_stem)
        if index < 0:
            continue
        try:
            candidate_metadata = _read_metadata_file(_metadata_path_for_checkpoint(candidate_path))
        except Exception:
            continue
        if not _candidate_is_convertible(candidate_metadata, target):
            continue
        candidate_widths = _component_hidden_widths(candidate_metadata)
        same_arch = candidate_widths == target_widths
        is_continuation = bool(candidate_metadata.get("use_saved_disk_networks_checkpoints", False))
        timesteps = _extract_loose_timesteps(candidate_metadata) or 0
        try:
            mtime = os.path.getmtime(candidate_path)
        except OSError:
            mtime = 0.0
        key = (same_arch, is_continuation, timesteps, index, mtime)
        if best_key is None or key > best_key:
            best_key, best_path, best_widths = key, candidate_path, candidate_widths

    if best_path is None:
        return None, None
    return best_path, best_widths


@dataclass(frozen=True)
class ContinuationSource:
    """How to resume training for one network: which checkpoint to load and
    whether its weights must be resized (Net2Net) to fit the current model."""
    path: str
    needs_conversion: bool
    source_hidden: tuple[int, ...] | None = None
    target_hidden: tuple[int, ...] | None = None


def resolve_continuation_source(
    *,
    checkpoint_path: str,
    metadata: dict | None = None,
    skip_selection_hyperparameter_match: bool = False,
) -> ContinuationSource | None:
    """Decide how the current model should be warm-started from disk.

    Tier 1 - a same-size checkpoint (exact or loose match, both architecture
    gated): load it directly. Tier 2 - otherwise the best *wider-or-equal*
    checkpoint: load it and down-convert (narrow) its weights onto the current
    architecture. A saved network smaller than the configured architecture is
    never widened; when no same-size or wider checkpoint exists this returns
    ``None`` and the trial trains from scratch."""
    same_size = resolve_matching_checkpoint_path(
        checkpoint_path=checkpoint_path,
        metadata=metadata,
        skip_selection_hyperparameter_match=skip_selection_hyperparameter_match,
    )
    if same_size is not None and os.path.isfile(same_size):
        return ContinuationSource(path=same_size, needs_conversion=False)

    source_path, source_widths = resolve_convertible_checkpoint_path(checkpoint_path, metadata)
    if source_path is None or not os.path.isfile(source_path):
        return None
    target_widths = _component_hidden_widths(_normalize_metadata(metadata))
    needs_conversion = not (
        source_widths is not None and target_widths is not None and source_widths == target_widths
    )
    return ContinuationSource(
        path=source_path,
        needs_conversion=needs_conversion,
        source_hidden=source_widths,
        target_hidden=target_widths,
    )


# ── Payload save / load (TSP dict payload, not a bare state_dict) ────────────

def read_checkpoint_timesteps(checkpoint_path: str) -> int | None:
    """Read the recorded training timesteps from a checkpoint's sidecar."""
    try:
        metadata = _read_metadata_file(_metadata_path_for_checkpoint(checkpoint_path))
    except Exception:
        return None
    return _extract_loose_timesteps(metadata)


def _timesteps_key_for_metadata(metadata: dict | None) -> str:
    if isinstance(metadata, dict):
        for key in _LOOSE_TIMESTEPS_KEYS:
            if key in metadata:
                return key
    return _LOOSE_TIMESTEPS_KEYS[0]


def _atomic_save_payload(payload: dict, output_path: str, metadata: dict | None) -> None:
    """Atomically write the dict ``payload`` to ``output_path`` (overwriting any
    existing file there) and, if provided, its metadata sidecar."""
    dir_name = os.path.dirname(output_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    tmp_path = output_path + f".tmp_{os.getpid()}"
    torch.save(payload, tmp_path)
    os.replace(tmp_path, output_path)

    if metadata is not None:
        _write_metadata_file(_metadata_path_for_checkpoint(output_path), metadata)


def save_payload_in_place(*, payload: dict, checkpoint_path: str, metadata: dict | None = None) -> str:
    """Overwrite ``checkpoint_path`` in place with ``payload`` + sidecar.

    This is the non-continuation path (reuse flag off): it reproduces the
    project's historical behaviour of overwriting ``actor_rep{N}.pt`` every run,
    now also emitting the metadata sidecar. Returns the written path."""
    _atomic_save_payload(payload, checkpoint_path, metadata)
    return checkpoint_path


def load_payload_for_continuation(
    *,
    model: torch.nn.Module,
    checkpoint_path: str,
    metadata: dict | None = None,
    skip_selection_hyperparameter_match: bool = False,
) -> tuple[str | None, int | None]:
    """Resolve and load a checkpoint's ``state_dict`` into ``model`` to resume
    training.

    A *same-size* checkpoint is loaded directly and its path is returned so the
    caller overwrites it in place (continuing its timestep count). If only a
    *wider* checkpoint exists, its weights are down-converted (narrowed) onto
    ``model``'s architecture and ``(None, None)`` is returned, so the caller leaves
    that source file untouched and saves the trained result as a new same-size
    checkpoint beside it. A saved network smaller than ``model`` is never widened;
    when nothing same-size or wider matches, ``(None, None)`` is returned and the
    trial trains from scratch.

    Returns ``(resolved_path, loaded_timesteps)`` for a direct load, or
    ``(None, None)`` when the model was down-converted from a wider source or
    nothing compatible was found."""
    source = resolve_continuation_source(
        checkpoint_path=checkpoint_path,
        metadata=metadata,
        skip_selection_hyperparameter_match=skip_selection_hyperparameter_match,
    )
    if source is None:
        return None, None
    payload = torch.load(source.path, map_location="cpu", weights_only=False)
    state = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
    if not source.needs_conversion:
        model.load_state_dict(state, strict=True)
        return source.path, read_checkpoint_timesteps(source.path)

    from .Library_network_conversion import down_convert_state_dict_into_model

    down_convert_state_dict_into_model(source_state_dict=state, model=model)
    print(
        f"  [checkpoint] down-converted {os.path.relpath(source.path)} "
        f"{list(source.source_hidden or [])} -> {list(source.target_hidden or [])} "
        f"(narrowed warm-start; source kept, trained result saved separately)"
    )
    return None, None


def save_continuation_or_new(
    *,
    payload: dict,
    checkpoint_path: str,
    metadata: dict | None = None,
    loaded_path: str | None = None,
    loaded_timesteps: int | None = None,
    n_timesteps: int | None = None,
) -> str:
    """Persist ``payload`` after a training repetition.

    If ``loaded_path`` is given (training continued from a loaded checkpoint),
    that exact file is overwritten in place and its recorded timesteps are set to
    the cumulative total (``loaded_timesteps + n_timesteps``).

    Otherwise a fresh, non-destructive ``<stem>_N.pt`` is written. Returns the
    path that was written."""
    if loaded_path is not None:
        out_metadata = dict(metadata) if isinstance(metadata, dict) else metadata
        if isinstance(out_metadata, dict) and n_timesteps is not None:
            base = loaded_timesteps if loaded_timesteps is not None else 0
            out_metadata[_timesteps_key_for_metadata(out_metadata)] = int(base) + int(n_timesteps)
        _atomic_save_payload(payload, loaded_path, out_metadata)
        return loaded_path

    output_path = _resolve_non_overwriting_path(checkpoint_path)
    _atomic_save_payload(payload, output_path, metadata)
    return output_path
