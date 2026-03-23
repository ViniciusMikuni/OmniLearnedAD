#!/usr/bin/env python3
import json
import argparse
import numpy as np
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
    mpl.rcParams.update({'xtick.labelsize': 18})
    mpl.rcParams.update({'ytick.labelsize': 18})
    mpl.rcParams.update({'axes.labelsize': 18})
    mpl.rcParams.update({'legend.frameon': False})
    mpl.rcParams.update({'lines.linewidth': 2})
    mpl.rcParams['figure.figsize'] = (9, 9)


def load_gof(json_file, mass):
    with open(json_file, "r") as f:
        data = json.load(f)

    entry = data[mass]
    obs = float(entry["obs"][0])
    toys = np.array(entry["toy"], dtype=float)

    # recompute p-value
    pval = np.mean(toys >= obs)

    return obs, toys, pval


def make_plot(obs, toys, pval, statistic, mass, output, title_right=""):
    # range
    xmin = min(np.min(toys), obs)
    xmax = max(np.max(toys), obs)
    xpad = 0.05 * (xmax - xmin) if xmax > xmin else 1.0
    xmin -= xpad
    xmax += xpad

    nbins = max(20, min(60, int(np.sqrt(len(toys)) * 2)))
    bins = np.linspace(xmin, xmax, nbins + 1)

    # wide figure
    fig, ax = plt.subplots(figsize=(6, 7))

    # gray histogram
    ax.hist(
        toys,
        bins=bins,
        histtype="stepfilled",
        color="0.7",
        edgecolor="black",
        linewidth=1.0
    )

    # observed vertical line
    ax.axvline(
        obs,
        linestyle="--",
        linewidth=2.0,
        color="black",
        label=f"Observed (p = {pval:.4f})"
    )

    # labels
    ax.set_xlabel(f"{statistic} Goodness of Fit Test", fontsize=16)
    ax.set_ylabel("Toys / bin", fontsize=16)

    # right title only

    # limits
    ax.set_xlim(xmin, xmax)

    # cleaner look
    ax.tick_params(axis="both", labelsize=16)

    # legend
    ax.legend(frameon=False, fontsize=14)

    fig.tight_layout()
    fig.savefig(f"{output}.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("json_file")
    parser.add_argument("--statistic", default="Saturated")
    parser.add_argument("--mass", default="125.0")
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--title-right", default="")
    args = parser.parse_args()
    SetStyle()
    obs, toys, pval = load_gof(args.json_file, args.mass)
    make_plot(obs, toys, pval, args.statistic, args.mass, args.output, args.title_right)


if __name__ == "__main__":
    main()
