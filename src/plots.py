"""Static figures (light mode) using the reference data-viz palette."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES_1 = "#2a78d6"
SERIES_2 = "#eb6834"
SEQUENTIAL_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DPI = 200
CI_CAP = 4


def _style():
    plt.rcParams.update({
        "font.family": ["Segoe UI", "DejaVu Sans"],
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "text.color": INK_PRIMARY, "axes.labelcolor": INK_SECONDARY, "axes.edgecolor": AXIS,
        "xtick.color": INK_MUTED, "ytick.color": INK_MUTED, "axes.titlecolor": INK_PRIMARY,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
        "axes.titlesize": 11, "axes.labelsize": 9.5, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    })


def _ci_bars(ax, labels, metric):
    values = [m["value"] for m in metric]
    low = [m["value"] - m["ci"][0] for m in metric]
    high = [m["ci"][1] - m["value"] for m in metric]
    ax.barh(labels, values, height=0.45, color=SERIES_1)
    ax.errorbar(values, labels, xerr=[low, high], fmt="none", ecolor=INK_SECONDARY, capsize=CI_CAP, lw=1)
    for label, metric_item in zip(labels, metric):
        ax.text(metric_item["ci"][1] + 0.015, label, f"{metric_item['value']:.2f}",
                va="center", fontsize=9, color=INK_PRIMARY)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)


def plot_cv_gap(schemes, path):
    """Headline: the same model scored three ways."""
    _style()
    labels = list(schemes)
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.2), sharey=True)
    _ci_bars(axes[0], labels, [schemes[s]["overall"]["r2"] for s in labels])
    axes[0].set_title("R² (higher is better)", loc="left")
    axes[0].set_xlim(0, 1.05)
    _ci_bars(axes[1], labels, [schemes[s]["overall"]["unhealthy_recall"] for s in labels])
    axes[1].set_title("Unhealthy days caught (PM2.5 > 35 µg/m³)", loc="left")
    axes[1].set_xlim(0, 1.05)
    fig.suptitle("Same model, three ways to validate it", x=0.01, ha="left", fontsize=12.5, weight="bold")
    fig.text(0.01, -0.04, "Bars: pooled estimate. Whiskers: 95% CI from resampling monitoring sites.",
             fontsize=8, color=INK_MUTED)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def _two_series_dots(ax, levels, series):
    offsets = {name: (i - 0.5) * 0.18 for i, name in enumerate(series)}
    colors = dict(zip(series, [SERIES_1, SERIES_2]))
    x = range(len(levels))
    for name, by_level in series.items():
        vals = [by_level[lv]["nmae"]["value"] for lv in levels]
        lo = [v - by_level[lv]["nmae"]["ci"][0] for v, lv in zip(vals, levels)]
        hi = [by_level[lv]["nmae"]["ci"][1] - v for v, lv in zip(vals, levels)]
        xs = [i + offsets[name] for i in x]
        ax.errorbar(xs, vals, yerr=[lo, hi], fmt="o", ms=7, color=colors[name], ecolor=colors[name],
                    lw=1.5, capsize=CI_CAP, label=name, markeredgecolor=SURFACE, markeredgewidth=1)
    ax.set_xticks(list(x), levels)
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left", bbox_to_anchor=(0, -0.2), ncol=2)


def plot_slice(series, levels, title, xlabel, path):
    """Normalized MAE by slice for two validation schemes."""
    _style()
    fig, ax = plt.subplots(figsize=(7, 3.6))
    _two_series_dots(ax, levels, series)
    ax.set_title(title, loc="left", fontsize=12, weight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Normalized MAE (error ÷ mean PM2.5)")
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_site_map(site_errors, value_column, path):
    """Each monitor colored by its error at a held-out (new) location."""
    _style()
    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQUENTIAL_BLUE)
    vmax = site_errors[value_column].quantile(0.95)
    fig, ax = plt.subplots(figsize=(9, 5))
    points = ax.scatter(site_errors["lon"], site_errors["lat"], c=site_errors[value_column],
                        cmap=cmap, vmin=0, vmax=vmax, s=22, edgecolors=SURFACE, linewidths=0.6)
    bar = fig.colorbar(points, ax=ax, shrink=0.8, pad=0.01)
    bar.set_label("Normalized MAE at a new location", color=INK_SECONDARY)
    bar.outline.set_visible(False)
    ax.set_aspect(1.25)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Where predictions fail when a monitor is left out", loc="left", fontsize=12, weight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
