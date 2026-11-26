"""Figures for the full study (light mode, reference data-viz palette)."""
import matplotlib.pyplot as plt

from plots import (AXIS, CI_CAP, DPI, INK_MUTED, INK_PRIMARY, INK_SECONDARY, SERIES_1, SERIES_2,
                   SURFACE, _style)

SERIES_COLORS = [SERIES_1, SERIES_2]
TARGET_COVERAGE = 0.90


def _dots(ax, categories, series, metric, horizontal=True):
    """Dot + 95% CI per category for up to two named series (fixed color order)."""
    offsets = {name: (i - (len(series) - 1) / 2) * 0.22 for i, name in enumerate(series)}
    for color, (name, by_cat) in zip(SERIES_COLORS, series.items()):
        vals = [by_cat[c][metric]["value"] for c in categories]
        lo = [v - by_cat[c][metric]["ci"][0] for v, c in zip(vals, categories)]
        hi = [by_cat[c][metric]["ci"][1] - v for v, c in zip(vals, categories)]
        pos = [i + offsets[name] for i in range(len(categories))]
        style = dict(fmt="o", ms=7, color=color, ecolor=color, lw=1.5, capsize=CI_CAP, label=name,
                     markeredgecolor=SURFACE, markeredgewidth=1)
        if horizontal:
            ax.errorbar(vals, pos, xerr=[lo, hi], **style)
        else:
            ax.errorbar(pos, vals, yerr=[lo, hi], **style)
    ticks = list(range(len(categories)))
    if horizontal:
        ax.set_yticks(ticks, categories)
        ax.grid(axis="y", visible=False)
    else:
        ax.set_xticks(ticks, categories)
        ax.grid(axis="x", visible=False)


def _legend_below(fig, ax):
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8.5, loc="lower left",
               bbox_to_anchor=(0.01, -0.08), ncol=len(labels))


def plot_two_metric_panels(series, categories, title, path):
    """R² and unhealthy-day recall side by side, categories on the y axis."""
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
    for ax, metric, label in [(axes[0], "r2", "R² (higher is better)"),
                              (axes[1], "unhealthy_recall", "Unhealthy days caught (PM2.5 > 35 µg/m³)")]:
        _dots(ax, categories, series, metric)
        ax.set_title(label, loc="left")
        ax.set_xlim(0, 1.0)
    axes[0].invert_yaxis()  # shared y axis: invert once so the first category is on top
    fig.suptitle(title, x=0.01, ha="left", fontsize=12.5, weight="bold")
    _legend_below(fig, axes[0])
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_distance_curve(points, path):
    """Skill vs. median distance from a test monitor to its nearest training monitor."""
    _style()
    points = sorted(points, key=lambda p: p["median_dist_km"])
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, metric, label in [(axes[0], "r2", "R²"), (axes[1], "unhealthy_recall", "Unhealthy days caught")]:
        x = [p["median_dist_km"] for p in points]
        y = [p["summary"][metric]["value"] for p in points]
        lo = [p["summary"][metric]["ci"][0] for p in points]
        hi = [p["summary"][metric]["ci"][1] for p in points]
        ax.fill_between(x, lo, hi, color=SERIES_1, alpha=0.15, lw=0)
        ax.plot(x, y, color=SERIES_1, lw=2, marker="o", ms=7, markeredgecolor=SURFACE)
        if metric == "r2":  # label points once, on the left panel, to avoid collisions
            for xi, yi, p in zip(x, y, points):
                ax.annotate(p["label"], (xi, yi), textcoords="offset points", xytext=(4, 7),
                            fontsize=7.5, color=INK_SECONDARY)
        ax.set_xscale("log")
        ax.set_xlabel("Median distance to nearest training monitor (km, log scale)")
        ax.set_title(label, loc="left")
        ax.set_ylim(0, 1.0)
    fig.suptitle("Skill decays with distance from the training network", x=0.01, ha="left",
                 fontsize=12.5, weight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_coverage(by_scheme, by_isolation, path):
    """90% interval coverage: by validation scheme (all vs unhealthy days) and by isolation."""
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
    schemes = list(by_scheme)
    _dots(axes[0], schemes, {
        "All days": {s: {"m": by_scheme[s]["coverage90"]} for s in schemes},
        "Unhealthy days": {s: {"m": by_scheme[s]["coverage90_unhealthy"]} for s in schemes},
    }, "m", horizontal=False)
    axes[0].set_title("By validation scheme", loc="left")
    bins = list(by_isolation)
    _dots(axes[1], bins, {"All days": {b: {"m": by_isolation[b]["coverage90"]} for b in bins}},
          "m", horizontal=False)
    axes[1].set_title("By distance to nearest monitor (leave regions out)", loc="left")
    for ax in axes:
        ax.axhline(TARGET_COVERAGE, color=INK_MUTED, lw=1, ls="--")
        ax.text(ax.get_xlim()[1], TARGET_COVERAGE, " target 90%", va="center", fontsize=7.5, color=INK_MUTED)
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("Share of days inside the 90% interval")
        ax.tick_params(axis="x", labelsize=7.5)
    fig.suptitle("Are the uncertainty intervals honest at new locations?", x=0.01, ha="left",
                 fontsize=12.5, weight="bold")
    _legend_below(fig, axes[0])
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_importance(importance, path):
    """Permutation importance (drop in R² when a feature is shuffled), single series."""
    _style()
    ordered = importance.sort_values("mean")
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(ordered.index, ordered["mean"], xerr=ordered["std"], height=0.55, color=SERIES_1,
            error_kw={"ecolor": INK_SECONDARY, "lw": 1, "capsize": 3})
    ax.axvline(0, color=AXIS, lw=1)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Drop in R² when the feature is shuffled")
    ax.set_title("What the model relies on in an unseen region", loc="left", fontsize=12,
                 weight="bold", color=INK_PRIMARY)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
