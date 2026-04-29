"""Pipeline visualization — text tree and minimal slide-friendly diagram.

Two entry points, both also surfaced as ``Pipeline.show_text()`` and
``Pipeline.show()`` methods:

- :func:`show_pipeline_tree` — ASCII indented tree (notebook-friendly).
- :func:`show_pipeline_diagram` — colored hierarchical-text matplotlib
  figure (slide-friendly; saves to PNG/PDF via ``save_path=``).

Categories control color coding in the diagram. Resolution order:

1. ``category_map={step_name: 'category'}`` passed to the call (highest priority)
2. ``feature_cls.category`` class attribute (preferred; declared by the feature)
3. Built-in registry :data:`_DEFAULT_CLASS_TO_CATEGORY` — safety net for any
   feature that doesn't declare ``category`` directly
4. ``'other'`` fallback

Built-in category names: ``per-frame``, ``summary``, ``tag``, ``global``,
``viz``, ``callback``, ``other``. New categories may be introduced ad-hoc by
passing ``category_colors={'my_cat': '#hex'}``.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

from .pipeline import CallbackStep, FeatureStep

if TYPE_CHECKING:
    from .pipeline import Pipeline


# ---------------------------------------------------------------------------
# Category palette and registry
# ---------------------------------------------------------------------------

_CAT_COLORS = {
    "per-frame":  "#0d6efd",   # blue   — time-resolved features
    "summary":    "#6f42c1",   # purple — per-id collapses
    "tag":        "#fd7e14",   # orange — id label columns
    "global":     "#20c997",   # teal   — fit-then-apply / templates / models
    "viz":        "#e83e8c",   # pink   — overlays / timelines / crops
    "callback":   "#adb5bd",   # light  — orchestration
    "other":      "#495057",
}

# Fallback registry for built-in feature classes that don't declare a
# `category` attribute. The class attribute is the source of truth; this
# table is a safety net for classes that haven't opted in yet.
_DEFAULT_CLASS_TO_CATEGORY = {
    "TrajectorySmooth":         "per-frame",
    "FFGroups":                 "per-frame",
    "NearestNeighbor":          "per-frame",
    "SpeedAngvel":              "per-frame",
    "NearestNeighborDelta":     "per-frame",
    "PairEgocentricFeatures":   "per-frame",
    "PairWavelet":              "per-frame",
    "ApproachAvoidance":        "per-frame",
    "TemporalStackingFeature":  "per-frame",
    "FFGroupsMetrics":          "summary",
    "NearestNeighborDeltaBins": "summary",
    "IdTagColumns":             "tag",
    "ExtractTemplates":         "global",
    "ExtractLabeledTemplates":  "global",
    "GlobalScaler":             "global",
    "GlobalTSNE":               "global",
    "GlobalKMeansClustering":   "global",
    "GlobalWardClustering":     "global",
    "ArHmmFeature":             "global",
    "KpmsFeature":              "global",
    "XgboostFeature":           "global",
    "GlobalIdentityModel":      "global",
    "LightningActionFeature":   "global",
}


def _resolve_category(step, category_map: dict | None) -> str:
    """Resolve a step's category via override → class attr → registry → 'other'."""
    if isinstance(step, CallbackStep):
        return "callback"
    if category_map and step.name in category_map:
        return category_map[step.name]
    cls = step.feature_cls
    declared = getattr(cls, "category", None)
    if isinstance(declared, str) and declared:
        return declared
    return _DEFAULT_CLASS_TO_CATEGORY.get(cls.__name__, "other")


# ---------------------------------------------------------------------------
# Shared traversal
# ---------------------------------------------------------------------------


def _pipeline_rows(pipe: "Pipeline", *, stop_at: str | None = None,
                   include_stop: bool = True,
                   category_map: dict | None = None,
                   root_label: str = "tracks") -> list[dict]:
    """Walk pipe → ordered row dicts (depth, level, category, parents, ...).

    Each row carries everything both renderers need; pure data so it's
    cheap to call and easy to test.
    """
    all_names = [s.name for s in pipe.steps]
    keep = set(all_names)
    if stop_at is not None:
        if stop_at not in keep:
            msg = (f"stop_at={stop_at!r} not in pipeline. "
                   f"Available: {all_names}")
            raise ValueError(msg)
        keep = pipe._upstream_of(stop_at)  # noqa: SLF001
        if not include_stop:
            keep -= {stop_at}

    steps_by_name = {s.name: s for s in pipe.steps}

    def _parents(s):
        refs = (s.input_names if isinstance(s, FeatureStep)
                else s.depends_on)
        return [p for p in refs if p in keep]

    primary, extras, parents_all = {}, {}, {}
    for s in pipe.steps:
        if s.name not in keep:
            continue
        ps = _parents(s)
        primary[s.name] = ps[0] if ps else root_label
        extras[s.name]  = ps[1:]
        parents_all[s.name] = ps

    children = defaultdict(list)
    for s in pipe.steps:
        if s.name in keep:
            children[primary[s.name]].append(s.name)

    levels: dict[str, int] = {}

    def _lv(n, stk=None):
        if n in levels:
            return levels[n]
        stk = stk or set()
        if n in stk:
            return 0
        stk.add(n)
        ps = parents_all.get(n, [])
        levels[n] = max((_lv(p, stk) for p in ps), default=-1) + 1
        return levels[n]

    for n in [s.name for s in pipe.steps if s.name in keep]:
        _lv(n)

    rows: list[dict] = []

    def _walk(parent, depth):
        for kid in children.get(parent, []):
            s = steps_by_name[kid]
            rows.append({
                "name": kid, "depth": depth, "level": levels[kid],
                "parents": parents_all.get(kid, []),
                "extras":  extras.get(kid, []),
                "is_callback": isinstance(s, CallbackStep),
                "category": _resolve_category(s, category_map),
                "feature_class": (s.feature_cls.__name__
                                  if isinstance(s, FeatureStep) else None),
            })
            _walk(kid, depth + 1)

    _walk(root_label, 0)
    return rows


# ---------------------------------------------------------------------------
# Text tree (ASCII)
# ---------------------------------------------------------------------------


def show_pipeline_tree(
    pipe: "Pipeline",
    *,
    highlight: str | list[str] | None = None,
    stop_at: str | None = None,
    include_stop: bool = True,
    show_extras: bool = True,
    show_feature_class: bool = False,
    root_label: str = "tracks",
    return_string: bool = False,
) -> str | None:
    """Print a Pipeline as an ASCII tree.

    The DAG is flattened by giving each node its first parent as the
    tree parent; any additional parents are noted inline with
    "← name1, name2". Callback steps are shown in [brackets].

    Parameters
    ----------
    pipe : Pipeline
    highlight : str | list[str], optional
        Step name(s) to mark with ★.
    stop_at : str, optional
        Show only this focal step and its transitive ancestors. Sibling
        branches and downstream steps are hidden.
    include_stop : bool
        If False, also drop the focal step itself (show ancestors only).
    show_extras : bool
        Show additional parents inline as "← p1, p2".
    show_feature_class : bool
        Append "(FeatureClass)" after the step name.
    root_label : str
        Label for the implicit root above all top-level steps.
    return_string : bool
        Return the rendered text instead of printing it.
    """
    highlights = set(
        [highlight] if isinstance(highlight, str)
        else list(highlight) if highlight else []
    )

    all_names = [s.name for s in pipe.steps]
    keep = set(all_names)
    if stop_at is not None:
        if stop_at not in keep:
            msg = (f"stop_at={stop_at!r} not in pipeline. "
                   f"Available: {all_names}")
            raise ValueError(msg)
        keep = pipe._upstream_of(stop_at)  # noqa: SLF001
        if not include_stop:
            keep -= {stop_at}

    steps_by_name = {s.name: s for s in pipe.steps}

    def _parents(step):
        refs = (step.input_names if isinstance(step, FeatureStep)
                else step.depends_on)
        return [p for p in refs if p in keep]

    primary, extras = {}, {}
    for step in pipe.steps:
        if step.name not in keep:
            continue
        ps = _parents(step)
        primary[step.name] = ps[0] if ps else root_label
        extras[step.name]  = ps[1:]

    children = defaultdict(list)
    for step in pipe.steps:
        if step.name in keep:
            children[primary[step.name]].append(step.name)

    def _format(name):
        step = steps_by_name[name]
        text = f"[{name}]" if isinstance(step, CallbackStep) else name
        if show_feature_class and isinstance(step, FeatureStep):
            text += f" ({step.feature_cls.__name__})"
        if name in highlights:
            text = f"★ {text} ★"
        if show_extras and extras.get(name):
            text += f"  ← {', '.join(extras[name])}"
        return text

    lines = [root_label]

    def _walk(parent, prefix):
        kids = children.get(parent, [])
        for i, kid in enumerate(kids):
            last = (i == len(kids) - 1)
            lines.append(
                f"{prefix}{'└─ ' if last else '├─ '}{_format(kid)}"
            )
            _walk(kid, prefix + ("   " if last else "│  "))

    _walk(root_label, "")

    text = "\n".join(lines)
    if return_string:
        return text
    print(text)
    return None


# ---------------------------------------------------------------------------
# Image diagram (Option A — colored hierarchical text)
# ---------------------------------------------------------------------------


def show_pipeline_diagram(
    pipe: "Pipeline",
    *,
    highlight: str | list[str] | None = None,
    stop_at: str | None = None,
    include_stop: bool = True,
    category_map: dict | None = None,
    category_colors: dict | None = None,
    title: str | None = None,
    save_path: str | None = None,
    figsize: tuple | None = None,
    row_height: float = 0.32,
    indent_unit: float = 0.45,
    width: float = 6,
    show_feature_class: bool = False,
    show_inputs: str = "off",
    return_fig: bool = False,
):
    """Render a Pipeline as a slide-friendly colored hierarchical-text figure.

    Each step is one row; step names are colored by category, indented by
    tree depth. The result saves cleanly to vector PDF for embedding in
    Keynote / PowerPoint.

    Parameters
    ----------
    pipe : Pipeline
    highlight : str | list[str], optional
        Step name(s) to emphasize (bold + soft yellow background).
    stop_at : str, optional
        Show only this focal step and its transitive ancestors. Sibling
        branches and downstream steps are hidden.
    include_stop : bool
        If False, also drop the focal step itself (show ancestors only).
    category_map : dict, optional
        ``{step_name: category}`` overrides class-based detection.
    category_colors : dict, optional
        ``{category: '#hex'}`` extends or overrides the default palette.
    title : str, optional
        Figure title (left-aligned).
    save_path : str | Path, optional
        Save the figure to this path. ``.pdf`` for vector graphics
        suitable for slides; ``.png`` also works.
    figsize : tuple, optional
        ``(width, height)`` in inches. Auto-sized if omitted.
    row_height, indent_unit, width : float
        Layout knobs. Bump ``width`` if step names + class + inputs overflow.
    show_feature_class : bool
        Append ``(FeatureClass)`` in muted small italic after each name.
    show_inputs : {"off", "extras", "all"}
        Append ``← p1, p2`` in muted small text after each name.
        - ``"off"``    : no inputs shown (default)
        - ``"extras"`` : parents beyond the first (primary parent is
          implicit from indentation; matches :func:`show_pipeline_tree`)
        - ``"all"``    : every parent
    return_fig : bool
        Return ``(fig, ax)`` instead of calling ``plt.show()``.
    """
    import matplotlib.pyplot as plt

    if show_inputs not in ("off", "extras", "all"):
        msg = (f"show_inputs must be 'off', 'extras', or 'all'; "
               f"got {show_inputs!r}")
        raise ValueError(msg)

    cat_colors = {**_CAT_COLORS, **(category_colors or {})}
    rows = _pipeline_rows(pipe, stop_at=stop_at, include_stop=include_stop,
                          category_map=category_map)
    hi = set(
        [highlight] if isinstance(highlight, str)
        else list(highlight) if highlight else []
    )

    n = len(rows)
    if figsize is None:
        figsize = (width, max(2.5, row_height * (n + 3)))

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, width)
    ax.set_ylim(-n - 2, 1.2)
    ax.axis("off")

    ax.text(0.1, 0.5, "tracks", fontsize=10, color="#888",
            style="italic", va="center")

    # 1 data unit ≈ 1 inch (xlim=(0,width); figsize width = width)
    def _tw(s, fs):
        return len(s) * 0.55 * fs / 72

    for i, r in enumerate(rows):
        y = -(i + 0.5)
        color = cat_colors.get(r["category"], cat_colors["other"])
        x = 0.3 + r["depth"] * indent_unit
        text = f"[{r['name']}]" if r["is_callback"] else r["name"]
        is_h = r["name"] in hi
        name_fs = 12 if is_h else 11
        ax.text(x, y, text,
                fontsize=name_fs,
                fontweight="bold" if is_h else "normal",
                color=color, va="center",
                bbox=(dict(boxstyle="round,pad=0.2",
                           facecolor="#fff3cd", edgecolor="none")
                      if is_h else None))
        cursor = x + _tw(text, name_fs) + 0.18

        if show_feature_class and r["feature_class"]:
            cls_str = f"({r['feature_class']})"
            ax.text(cursor, y, cls_str, fontsize=9, color="#888",
                    style="italic", va="center")
            cursor += _tw(cls_str, 9) + 0.22

        if show_inputs != "off":
            ps = r["parents"] if show_inputs == "all" else r["extras"]
            if ps:
                in_str = "← " + ", ".join(ps)
                ax.text(cursor, y, in_str, fontsize=8, color="#aaa",
                        va="center")

    # Inline category legend at bottom
    used = list(dict.fromkeys(r["category"] for r in rows))
    legend_y = -(n + 1.2)
    x_cursor = 0.3
    for c in used:
        col = cat_colors.get(c, cat_colors["other"])
        ax.text(x_cursor, legend_y, "■", fontsize=10, color=col,
                va="center")
        ax.text(x_cursor + 0.18, legend_y, c, fontsize=8,
                color="#444", va="center")
        x_cursor += len(c) * 0.11 + 0.5

    if title:
        ax.set_title(title, fontsize=12, loc="left", pad=8)

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight", dpi=200)

    if return_fig:
        return fig, ax
    plt.show()
    return None
