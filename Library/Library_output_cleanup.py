"""
Library_output_cleanup.py - Run-id stamping and end-of-run pruning of the
``plots`` / ``Trial Continuation Analysis`` output directories.

Every output file written during a run gets the run's *execution-start*
timestamp appended to its name (format ``yy.mm.dd-HHMMSS`` -> ``26.11.22-161811``).
That stamp doubles as a **run id**: all files that share it belong to the same
run, so the pruner can group depth-0 files by run without any external state.

At the end of a run (after the blocking plots have been closed) ``cleanup_output_dirs``
removes, at depth 0 only (never recursing into subdirectories):

  * Age retention (OR semantics): a run's files are deleted if the run is
    *beyond the newest 3 runs* (current + 2 previous) **OR** the run is *older
    than one week*. The current run is always kept. Legacy files with no run id
    are pruned only by the 1-week rule, keyed on their modification time.

  * Smoothing-window thinning (surviving, non-current runs only): keep just one
    smoothed window - the first that exists in the preference order
    151, 201, 101, 251, 301, 351, ... - and delete everything else for that run
    (the ``w1`` not-smoothed plots, the other smoothed windows, and the
    ``Twin_*_combined`` plots). The current run keeps all of its windows.
"""
import os
import re
from datetime import datetime, timedelta

# Execution-start timestamp used as both the filename suffix and the run id.
# Fixed-width and big-endian, so lexical order == chronological order.
RUN_ID_FORMAT = "%y.%m.%d-%H%M%S"
_RUN_ID_RE = re.compile(r"(\d{2}\.\d{2}\.\d{2}-\d{6})")

# Filename smoothing-window tokens, e.g. ``_w1-not-smoothed`` / ``_w201-smoothed``
# and the twin composite ``_w101-w201-combined``.
_WINDOW_RE = re.compile(r"_w(\d+)-(not-smoothed|smoothed)")
_TWIN_RE = re.compile(r"_w\d+-w\d+-combined")

# Preference order for the single smoothed window kept on non-current runs:
# 151 first, then 201, then 101, then the remaining windows in ascending order.
_WINDOW_PRIORITY = {151: 0, 201: 1, 101: 2}

_RUN_ID = None


# ── Run-id management ────────────────────────────────────────────────────────
def configure_run_id(run_id=None):
    """Fix this process's run id (call once at execution start). Returns it."""
    global _RUN_ID
    _RUN_ID = run_id if run_id is not None else datetime.now().strftime(RUN_ID_FORMAT)
    return _RUN_ID


def get_run_id():
    """Return the current run id, lazily initialising it on first use."""
    if _RUN_ID is None:
        return configure_run_id()
    return _RUN_ID


def parse_run_id(name):
    """Return the run id embedded in *name*, or ``None`` if it carries none."""
    m = _RUN_ID_RE.search(name)
    return m.group(1) if m else None


def run_id_to_datetime(run_id):
    """Parse a run id back into a ``datetime`` (``None`` if malformed)."""
    try:
        return datetime.strptime(run_id, RUN_ID_FORMAT)
    except (ValueError, TypeError):
        return None


def stamp_run_id(path, run_id=None):
    """Insert ``_<run_id>`` before *path*'s extension (idempotent).

    A path that already carries a run id is returned unchanged, so re-stamping
    or stamping an already-tagged file is safe.
    """
    if run_id is None:
        run_id = get_run_id()
    directory, filename = os.path.split(path)
    if parse_run_id(filename) is not None:
        return path
    stem, ext = os.path.splitext(filename)
    return os.path.join(directory, f"{stem}_{run_id}{ext}")


# ── Smoothing-window classification ──────────────────────────────────────────
def classify_window(name):
    """Classify a filename's smoothing artefact.

    Returns ``(kind, window)`` where kind is one of ``"smoothed"`` (window>1),
    ``"notsmoothed"`` (the ``w1`` plots), ``"combined"`` (twin composite), or
    ``"none"`` (route plots, summaries - not window-bearing). ``window`` is the
    integer window for smoothed plots, else ``None``.
    """
    if _TWIN_RE.search(name):
        return ("combined", None)
    m = _WINDOW_RE.search(name)
    if m is None:
        return ("none", None)
    window, token = int(m.group(1)), m.group(2)
    if token == "smoothed":
        return ("smoothed", window)
    return ("notsmoothed", window)


def preferred_window(windows):
    """Pick the smoothed window to keep: 151 > 201 > 101 > (ascending rest)."""
    return min(windows, key=lambda w: (_WINDOW_PRIORITY.get(w, 3), w))


# ── End-of-run pruning ───────────────────────────────────────────────────────
def cleanup_output_dirs(
    dirs,
    run_id=None,
    *,
    keep_runs=3,
    max_age_days=7,
    verbose=True,
    dry_run=False,
):
    """Prune depth-0 files in *dirs* per the age + smoothing-window rules.

    Parameters
    ----------
    dirs : iterable[str]
        Output directories to prune (``plots`` / ``Trial Continuation Analysis``).
    run_id : str or None
        The current run id; defaults to :func:`get_run_id`.
    keep_runs : int
        Number of most-recent runs to retain (current + ``keep_runs - 1`` prior).
    max_age_days : int
        Runs/files older than this are pruned regardless of the run count.
    dry_run : bool
        When True, report what would be removed without deleting anything.

    Returns the number of files removed (or that would be removed).
    """
    if run_id is None:
        run_id = get_run_id()
    age_cutoff = datetime.now() - timedelta(days=max_age_days)
    removed_total = 0

    for directory in dirs:
        if not os.path.isdir(directory):
            continue

        names = [
            f for f in os.listdir(directory)
            if os.path.isfile(os.path.join(directory, f))
        ]

        tagged: dict[str, list[str]] = {}
        untagged: list[str] = []
        for name in names:
            rid = parse_run_id(name)
            if rid is None:
                untagged.append(name)
            else:
                tagged.setdefault(rid, []).append(name)

        # Newest-first run ids (fixed-width format sorts chronologically).
        rids_newest_first = sorted(tagged, reverse=True)
        kept_recent = set(rids_newest_first[:keep_runs])
        kept_recent.add(run_id)  # never prune the current run

        to_delete: list[str] = []

        # Pass 1 - age retention (delete if beyond newest N runs OR older than a week).
        surviving_rids: list[str] = []
        for rid, files in tagged.items():
            if rid == run_id:
                surviving_rids.append(rid)
                continue
            rid_dt = run_id_to_datetime(rid)
            older_than_max_age = rid_dt is not None and rid_dt < age_cutoff
            beyond_recent = rid not in kept_recent
            if beyond_recent or older_than_max_age:
                to_delete.extend(files)
            else:
                surviving_rids.append(rid)

        # Legacy (un-tagged) files: only the age rule, keyed on modification time.
        for name in untagged:
            try:
                mtime = datetime.fromtimestamp(os.path.getmtime(os.path.join(directory, name)))
            except OSError:
                continue
            if mtime < age_cutoff:
                to_delete.append(name)

        # Pass 2 - smoothing-window thinning of surviving, non-current runs.
        for rid in surviving_rids:
            if rid == run_id:
                continue  # current run keeps every window
            group = tagged[rid]
            smoothed_present = [
                w for f in group
                for (kind, w) in [classify_window(f)]
                if kind == "smoothed"
            ]
            if not smoothed_present:
                continue  # nothing to thin down to
            keep_w = preferred_window(smoothed_present)
            for f in group:
                kind, w = classify_window(f)
                if kind == "smoothed" and w == keep_w:
                    continue  # the single window we keep
                if kind in ("smoothed", "notsmoothed", "combined"):
                    to_delete.append(f)  # drop w1, other windows, twin composites
                # kind == "none" (route plots, summaries) is left untouched here.

        # De-duplicate while preserving order.
        to_delete = list(dict.fromkeys(to_delete))

        removed_here = 0
        for name in to_delete:
            path = os.path.join(directory, name)
            if dry_run:
                removed_here += 1
                continue
            try:
                os.remove(path)
                removed_here += 1
            except OSError as exc:
                if verbose:
                    print(f"[cleanup] Could not remove {path}: {exc}")
        removed_total += removed_here

        if verbose and removed_here:
            verb = "Would remove" if dry_run else "Removed"
            print(f"[cleanup] {verb} {removed_here} stale file(s) from {directory}/")

    return removed_total
