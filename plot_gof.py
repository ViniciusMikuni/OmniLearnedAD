#!/usr/bin/env python3
import argparse
import numpy as np
import uproot
import matplotlib.pyplot as plt


def SetStyle():
    from matplotlib import rc
    import matplotlib as mpl

    rc('font', family='serif')
    rc('font', size=22)
    rc('xtick', labelsize=15)
    rc('ytick', labelsize=15)
    rc('legend', fontsize=15)
    mpl.rcParams.update({'font.size': 19})
    mpl.rcParams['text.usetex'] = False
    mpl.rcParams['mathtext.fontset'] = 'cm'
    mpl.rcParams.update({'xtick.labelsize': 18})
    mpl.rcParams.update({'ytick.labelsize': 18})
    mpl.rcParams.update({'axes.labelsize': 18})
    mpl.rcParams.update({'legend.frameon': False})
    mpl.rcParams.update({'lines.linewidth': 2})
    mpl.rcParams['figure.figsize'] = (9, 9)


def load_scan(root_file):
    with uproot.open(root_file) as f:
        tree = f["limit"]
        r = np.array(tree["r"].array(library="np"), dtype=float)
        dnll = np.array(tree["deltaNLL"].array(library="np"), dtype=float)

    y = 2.0 * dnll
    mask = np.isfinite(r) & np.isfinite(y) & (y >= -1e-8)
    r = r[mask]
    y = y[mask]

    order = np.argsort(r)
    r = r[order]
    y = y[order]

    uniq = {}
    for ri, yi in zip(r, y):
        if (ri not in uniq) or (yi < uniq[ri]):
            uniq[ri] = yi

    r = np.array(sorted(uniq.keys()), dtype=float)
    y = np.array([uniq[ri] for ri in r], dtype=float)

    return r, y


def find_crossing(x, y, level, side):
    imin = np.argmin(y)

    if side == "left":
        for i in range(imin, 0, -1):
            x1, x2 = x[i - 1], x[i]
            y1, y2 = y[i - 1], y[i]
            if (y1 - level) * (y2 - level) <= 0 and y1 != y2:
                return x1 + (level - y1) * (x2 - x1) / (y2 - y1)
    elif side == "right":
        for i in range(imin, len(x) - 1):
            x1, x2 = x[i], x[i + 1]
            y1, y2 = y[i], y[i + 1]
            if (y1 - level) * (y2 - level) <= 0 and y1 != y2:
                return x1 + (level - y1) * (x2 - x1) / (y2 - y1)

    return np.nan


def get_interval(x, y, level=1.0):
    imin = np.argmin(y)
    xbest = float(x[imin])
    xlo = find_crossing(x, y, level, "left")
    xhi = find_crossing(x, y, level, "right")
    return xbest, xlo, xhi


def interval_label(name, x, y):
    xbest, xlo, xhi = get_interval(x, y, 1.0)
    if np.isfinite(xlo) and np.isfinite(xhi):
        return (
            rf"{name}: "
            rf"$\hat{{r}}={xbest:.2f}"
            rf"^{{+{xhi - xbest:.2f}}}"
            rf"_{{-{xbest - xlo:.2f}}}$"
        )
    return rf"{name}: $\hat{{r}}={xbest:.2f}$"


def make_plot(r_full, y_full, r_stat, y_stat, output, poi="r", title_right=""):
    xmin = min(np.min(r_full), np.min(r_stat))
    xmax = max(np.max(r_full), np.max(r_stat))
    xpad = 0.05 * (xmax - xmin) if xmax > xmin else 1.0
    xmin -= xpad
    xmax += xpad

    ymax = max(np.max(y_full), np.max(y_stat), 4.0)
    ymax = max(4.5, 1.15 * ymax)

    fig, ax = plt.subplots(figsize=(6, 7))

    ax.plot(r_full, y_full, color="#1f77b4", linestyle="-", linewidth=2.5, label="Total")
    ax.plot(r_stat, y_stat, color="#d62728", linestyle="--", linewidth=2.5, label="Stat-only")

    ax.axhline(1.0, linestyle=":", linewidth=1.5, color="black", alpha=0.6)
    ax.axhline(3.84, linestyle=":", linewidth=1.5, color="black", alpha=0.4)

    ax.plot(
        r_full[np.argmin(y_full)], np.min(y_full),
        marker="o", color="#1f77b4", ms=6
    )
    ax.plot(
        r_stat[np.argmin(y_stat)], np.min(y_stat),
        marker="o", color="#d62728", ms=6
    )

    ax.set_xlabel(poi, fontsize=14)
    ax.set_ylabel(r"$-2\Delta\ln L$", fontsize=14)

    if title_right:
        ax.text(0.98, 0.88, title_right, transform=ax.transAxes, ha="right", va="top")

    txt1 = interval_label("Total", r_full, y_full)
    txt2 = interval_label("Stat-only", r_stat, y_stat)

    ax.text(0.05, 0.95, txt1, transform=ax.transAxes, ha="left", va="top", fontsize=13)
    ax.text(0.05, 0.88, txt2, transform=ax.transAxes, ha="left", va="top", fontsize=13)

    ax.text(0.98, 1.0 / ymax + 0.01, "68% CL", transform=ax.transAxes, ha="right", va="bottom", fontsize=12)
    ax.text(0.98, 3.84 / ymax + 0.01, "95% CL", transform=ax.transAxes, ha="right", va="bottom", fontsize=12)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(0.0, ymax)
    ax.tick_params(axis="both", labelsize=14)
    ax.legend(frameon=False, fontsize=14)

    fig.tight_layout()
    fig.savefig(f"{output}.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", required=True)
    parser.add_argument("--stat", required=True)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--poi", default="r")
    parser.add_argument("--title-right", default="")
    args = parser.parse_args()

    SetStyle()
    r_full, y_full = load_scan(args.full)
    r_stat, y_stat = load_scan(args.stat)
    make_plot(r_full, y_full, r_stat, y_stat, args.output, args.poi, args.title_right)


if __name__ == "__main__":
    main()
