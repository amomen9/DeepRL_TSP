"""
Helper_legend.py - Legend label/value formatting.

Contents
--------
LegendEntry             - Type alias: (display_label, show_flag).
_fmt                    - Format numbers safely for filenames.
_clean_float            - Snap a float to float32 precision (drop promotion noise).
_format_number_normal   - Render a number in normal (non-scientific) form.
_format_number_sci      - Render a number in scientific form (or None).
_fmt_legend             - Format numbers for legend display (shortest of normal/sci).
_format_legend_value    - Format legend values, unwrapping single-item containers.
_format_legend_label    - Pass-through label formatter (kept as an extension point).
_normalize_legend_entry - Normalize a legend entry to a (label, show_flag) pair.
_resolve_legend_flags   - Resolve which parameters appear in the legend.
_build_legend_parts     - Build "label=value" strings for legend-enabled parameters.
"""
import ast
from typing import Any, Tuple

import numpy as np


LegendEntry = Tuple[str, bool]


def _fmt(v):
    """Format numbers safely for filenames."""
    if isinstance(v, float):
        return f"{v:.3g}".replace(".", "p")
    return str(v)


def _clean_float(v: Any) -> Any:
    """Remove float64-promotion noise from a float before formatting.

    Config/hyperparameter (and TSP cost) values in this project are
    float32-scale. A value such as Beta=0.01 stored as ``float32`` and then
    promoted to ``float64`` reads back as ``0.009999999776482582``; snapping
    it to ``float32`` precision restores the intended ``0.01``. Values that
    overflow ``float32`` are returned unchanged.
    """
    x32 = np.float32(v)
    return x32 if np.isfinite(x32) else v


def _coerce_legend_value(v: Any) -> Any:
    """Convert stored text values back to numbers / containers when possible."""
    if isinstance(v, str):
        text = v.strip()
        if not text:
            return v
        try:
            return ast.literal_eval(text)
        except Exception:
            return v
    return v


def _format_number_normal(v: Any) -> str:
    """Render a number in normal (non-scientific) form.

    Uses the shortest round-trip representation, so values are faithful
    (no precision loss): integers keep all their digits and floats keep
    just enough digits to be reproduced exactly.
    """
    if isinstance(v, (bool, np.bool_)):
        return "✓" if v else "✗"

    if isinstance(v, (int, np.integer)):
        return str(int(v))

    if isinstance(v, (float, np.floating)):
        if not np.isfinite(v):
            return str(v)
        return np.format_float_positional(_clean_float(v), trim="-")

    return str(v)


def _format_number_sci(v: Any) -> str | None:
    """Render a number in scientific form, or None if not a finite number.

    Mantissa/exponent are the shortest round-trip form (faithful), and the
    exponent is compacted (e.g. 'e-04' -> 'e-4', 'e+06' -> 'e6').
    """
    if isinstance(v, (bool, np.bool_)):
        return None

    if isinstance(v, (int, np.integer)):
        n = int(v)
        if n == 0:
            return "0"
        sign = "-" if n < 0 else ""
        digits = str(abs(n))
        exponent = len(digits) - 1
        mantissa_digits = digits.rstrip("0") or "0"
        if len(mantissa_digits) == 1:
            mantissa = mantissa_digits
        else:
            mantissa = mantissa_digits[0] + "." + mantissa_digits[1:]
        return f"{sign}{mantissa}e{exponent}"

    if isinstance(v, (float, np.floating)):
        if not np.isfinite(v):
            return None
        mantissa, exponent = np.format_float_scientific(_clean_float(v), trim="-").split("e")
        return f"{mantissa}e{int(exponent)}"

    return None


def _fmt_legend(v: Any) -> str:
    """Format one scalar for legend display.

    Every numeric value is first normalized to remove float math noise, then
    rendered in scientific notation only when that is strictly shorter than the
    normal form. Ties favor normal form.
    """
    v = _coerce_legend_value(v)

    if isinstance(v, (bool, np.bool_)):
        return "✓" if v else "✗"

    normal = _format_number_normal(v)
    sci = _format_number_sci(v)
    if sci is not None and len(sci) < len(normal):
        return sci
    return normal


def _format_legend_value(val: Any) -> str:
    """Format legend values, recursively formatting nested numeric content."""
    val = _coerce_legend_value(val)

    if isinstance(val, np.ndarray):
        val = np.atleast_1d(val).tolist()

    if isinstance(val, dict):
        if len(val) == 0:
            return "{}"
        return "{" + ", ".join(
            f"{_format_legend_value(k)}: {_format_legend_value(v)}" for k, v in val.items()
        ) + "}"

    if isinstance(val, list):
        if len(val) == 0:
            return "[]"
        return "[" + ",".join(_format_legend_value(item) for item in val) + "]"

    if isinstance(val, tuple):
        if len(val) == 0:
            return "()"
        inner = ",".join(_format_legend_value(item) for item in val)
        if len(val) == 1:
            inner += ","
        return "(" + inner + ")"

    if isinstance(val, set):
        if len(val) == 0:
            return "set()"
        return "{" + ",".join(sorted(_format_legend_value(item) for item in val)) + "}"

    if hasattr(val, "__iter__") and not isinstance(val, (str, bytes)):
        try:
            values = list(val)
        except TypeError:
            return _fmt_legend(val)
        return _format_legend_value(values)

    return _fmt_legend(val)


def _format_legend_label(label: str) -> str:
    """Return legend labels exactly as configured in Experiment.py."""
    return str(label)


def _normalize_legend_entry(param_name: str, entry: Any) -> LegendEntry:
    """Normalize a legend entry to a (label, show_flag) pair."""
    if isinstance(entry, (list, tuple)) and len(entry) == 2:
        label, show = entry
    else:
        label, show = param_name, entry
    return str(label), bool(show)


def _resolve_legend_flags(cfg: dict[str, Any], *, warn_on_suppression: bool = True) -> dict[str, LegendEntry]:
    """Resolve which parameters should appear in the legend.

    Uses ``legend_parameters`` dict if present. Falls back to (and is
    overridden by) the legacy ``nn_include_hp_in_legend`` /
    ``nn_include_lr_in_legend`` flags for backward compatibility.

    When epsilon decay is disabled (``epsilon_decay_interval == 0``),
    suppress the dependent epsilon-decay fields from the legend. If
    ``warn_on_suppression`` is True, print one warning per suppressed field.

    Returns a dict {param_name: (display_label, bool)}.
    """
    legend: dict[str, LegendEntry] = {}
    raw_legend_parameters = cfg.get("legend_parameters", {})
    if isinstance(raw_legend_parameters, dict):
        for param_name, entry in raw_legend_parameters.items():
            legend[param_name] = _normalize_legend_entry(param_name, entry)

    # Legacy overrides
    if "nn_include_lr_in_legend" in cfg:
        lr_key = "learning_rate" if "learning_rate" in cfg else "actor_lr"
        current_lr_entry = legend.get(lr_key)
        if current_lr_entry is not None:
            current_lr_label = current_lr_entry[0]
        else:
            current_lr_label = lr_key
        legend[lr_key] = (current_lr_label, bool(cfg["nn_include_lr_in_legend"]))
    if "nn_include_hp_in_legend" in cfg:
        hp_flag = bool(cfg["nn_include_hp_in_legend"])
        nn_key = "nn_hidden_layer_widths" if "nn_hidden_layer_widths" in cfg else "actor_hidden_nn"
        current_nn_entry = legend.get(nn_key)
        if current_nn_entry is not None:
            current_nn_label = current_nn_entry[0]
        else:
            current_nn_label = nn_key
        legend[nn_key] = (current_nn_label, hp_flag)
        current_gamma_entry = legend.get("gamma")
        if current_gamma_entry is not None:
            current_gamma_label = current_gamma_entry[0]
        else:
            current_gamma_label = "gamma"
        legend["gamma"] = (current_gamma_label, hp_flag)

    if cfg.get("exploration_method", "egreedy") == "egreedy":
        try:
            epsilon_decay_interval = int(cfg.get("epsilon_decay_interval", 1))
        except (TypeError, ValueError):
            epsilon_decay_interval = 1
        if epsilon_decay_interval == 0:
            for key in ("epsilon_start", "epsilon_end", "epsilon_decay"):
                entry = legend.get(key)
                if entry is None or not entry[1]:
                    continue
                if warn_on_suppression:
                    display_label = entry[0] if str(entry[0]).strip() else key
                    print(
                        f"[legend] Warning (plot): '{display_label}' legend entry suppressed because "
                        "epsilon_decay_interval=0 disables epsilon decay trials."
                    )
                legend[key] = (entry[0], False)

    return legend


def _build_legend_parts(legend: dict[str, LegendEntry], cfg: dict[str, Any]) -> list[str]:
    """Build a list of 'label=value' strings for every legend-enabled parameter."""
    parts: list[str] = []
    for key, (label, show) in legend.items():
        if not show:
            continue
        val = cfg.get(key)
        if val is None:
            continue
        label = _format_legend_label(label)
        if isinstance(val, (bool, np.bool_)):
            parts.append(f"{label}{'✓' if val else '✗'}")
        else:
            parts.append(f"{label}{_format_legend_value(val)}")
    return parts
