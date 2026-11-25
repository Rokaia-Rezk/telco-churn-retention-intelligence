"""Matplotlib charts in the project's palette (navy / orange / cream)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

NAVY, ORANGE, CREAM, GREY, LINE, BLUE = "#14213D", "#E07B1F", "#F6F5F0", "#5B6472", "#DAD7CB", "#3B6EA8"


def _fig(w=6.4, h=3.4):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(CREAM)
    ax.set_facecolor(CREAM)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(LINE)
    ax.tick_params(colors=NAVY, labelsize=9, length=0)
    return fig, ax


def churn_bars(series, title, overall=None, color=ORANGE, rotate=0):
    fig, ax = _fig()
    labels = [str(i) for i in series.index]
    bars = ax.bar(labels, series.values, color=color, width=0.62)
    for b, v in zip(bars, series.values):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.8, f"{v:.1f}%", ha="center", va="bottom", fontsize=9, color=NAVY, fontweight="bold")
    if overall is not None:
        ax.axhline(overall, color=NAVY, linestyle="--", linewidth=1)
        ax.text(len(labels) - 0.5, overall + 0.8, f"overall {overall:.1f}%", ha="right", fontsize=8, color=NAVY)
    ax.set_ylim(0, max(series.values) * 1.22)
    ax.set_yticks([])
    ax.set_title(title, loc="left", fontsize=11, color=NAVY, fontweight="bold")
    plt.setp(ax.get_xticklabels(), rotation=rotate, ha="right" if rotate else "center")
    fig.tight_layout()
    return fig


def importance_bars(values: dict, title, fmt="{:.3f}", color=ORANGE, n=10):
    items = list(values.items())[:n][::-1]
    fig, ax = _fig(5.6, 3.8)
    ax.barh([k for k, _ in items], [v for _, v in items], color=color, height=0.62)
    top = max(v for _, v in items)
    for i, (_, v) in enumerate(items):
        ax.text(v + top * 0.015, i, fmt.format(v), va="center", fontsize=9, color=NAVY, fontweight="bold")
    ax.set_xlim(0, top * 1.2)
    ax.set_xticks([])
    ax.set_title(title, loc="left", fontsize=11, color=NAVY, fontweight="bold")
    fig.tight_layout()
    return fig


def contribution_chart(contrib_df, n=8):
    d = contrib_df.head(n).iloc[::-1]
    fig, ax = _fig(6.4, 3.6)
    colors = [ORANGE if v > 0 else NAVY for v in d["contribution"]]
    ax.barh([f"{f}: {v}" for f, v in zip(d["feature"], d["value"])], d["contribution"] * 100, color=colors, height=0.62)
    ax.axvline(0, color=GREY, linewidth=0.8)
    for i, v in enumerate(d["contribution"] * 100):
        ax.text(v + (0.6 if v >= 0 else -0.6), i, f"{v:+.1f}", va="center", ha="left" if v >= 0 else "right", fontsize=8.5, color=NAVY)
    lim = max(abs(d["contribution"] * 100)) * 1.3
    ax.set_xlim(-lim, lim)
    ax.set_xticks([])
    ax.set_title("What moves this customer's risk (percentage points)", loc="left", fontsize=11, color=NAVY, fontweight="bold")
    fig.tight_layout()
    return fig


def risk_bar(prob, threshold):
    fig, ax = plt.subplots(figsize=(6.4, 0.9))
    fig.patch.set_facecolor(CREAM)
    ax.set_facecolor(CREAM)
    ax.barh([0], [1], color="#E4E1D6", height=0.5)
    ax.barh([0], [prob], color=ORANGE if prob >= threshold else BLUE, height=0.5)
    ax.axvline(threshold, color=NAVY, linewidth=1.5)
    ax.text(threshold, 0.42, f"threshold {threshold:.2f}", ha="center", fontsize=8, color=NAVY)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.4, 0.7)
    ax.axis("off")
    fig.tight_layout(pad=0.2)
    return fig


def confusion_fig(tn, fp, fn, tp):
    fig, ax = plt.subplots(figsize=(4.4, 3.4))
    fig.patch.set_facecolor(CREAM)
    cells = [[tn, fp], [fn, tp]]
    fills = [["#E9E7DF", "#F6D9BC"], ["#F6D9BC", ORANGE]]
    for i in range(2):
        for j in range(2):
            ax.add_patch(plt.Rectangle((j, 1 - i), 1, 1, facecolor=fills[i][j], edgecolor=CREAM, linewidth=3))
            ax.text(j + 0.5, 1.5 - i, f"{cells[i][j]:,}", ha="center", va="center", fontsize=18, fontweight="bold",
                    color="white" if (i == 1 and j == 1) else NAVY)
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.set_xticks([0.5, 1.5])
    ax.set_xticklabels(["Predicted: stay", "Predicted: churn"], fontsize=9, color=NAVY)
    ax.set_yticks([0.5, 1.5])
    ax.set_yticklabels(["Actual: churned", "Actual: stayed"], fontsize=9, color=NAVY)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout()
    return fig


def share_vs_churners(profile):
    fig, ax = _fig(6.4, 3.4)
    names = [n.split(":")[0] for n in profile["name"]]
    x = range(len(names))
    w = 0.36
    cust = profile["share"] * 100
    lost = profile["lost_customers"] / profile["lost_customers"].sum() * 100
    b1 = ax.bar([i - w / 2 for i in x], cust, w, color=NAVY, label="share of customers")
    b2 = ax.bar([i + w / 2 for i in x], lost, w, color=ORANGE, label="share of all churners")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1, f"{b.get_height():.0f}%", ha="center", fontsize=9, color=NAVY, fontweight="bold")
    ax.set_xticks(list(x))
    ax.set_xticklabels(names)
    ax.set_yticks([])
    ax.set_ylim(0, max(max(cust), max(lost)) * 1.2)
    ax.legend(frameon=False, fontsize=8, labelcolor=NAVY)
    ax.set_title("Where the churners are", loc="left", fontsize=11, color=NAVY, fontweight="bold")
    fig.tight_layout()
    return fig
