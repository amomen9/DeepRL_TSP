"""
Helper_progress.py - tqdm progress bars and file-path helpers.

Contents
--------
_create_step_progress_bar  - tqdm bar with the project's shared formatting.
get_unique_filepath        - Return a non-overwriting filepath, enumerating if needed.
"""
import os

from tqdm import tqdm


def _create_step_progress_bar(total, desc, position=None, leave=False):
    """Create a tqdm progress bar with the shared project formatting."""

    tqdm_kwargs = {
        "total": total,
        "desc": desc,
        "unit": "step",
        "bar_format": "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
        "dynamic_ncols": True,
        "leave": leave,
    }
    if position is not None:
        tqdm_kwargs["position"] = int(position)
    return tqdm(**tqdm_kwargs)


def get_unique_filepath(path: str) -> str:
    """
    Return a non-overwriting filepath by enumerating if `path` already exists.

    Example:
      plots/foo.png  -> plots/foo.png
      plots/foo.png  (exists) -> plots/foo_1.png
      plots/foo.png  (exists) -> plots/foo_2.png
    """
    if not isinstance(path, str) or not path:
        raise ValueError("path must be a non-empty string")

    if not os.path.exists(path):
        return path

    directory = os.path.dirname(path)
    filename = os.path.basename(path)
    stem, ext = os.path.splitext(filename)

    for i in range(1, 100000):
        candidate = os.path.join(directory, f"{stem}_{i}{ext}")
        if not os.path.exists(candidate):
            return candidate

    raise RuntimeError(f"Could not find a non-overwriting filename for: {path}")
