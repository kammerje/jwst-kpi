import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np


def plot_emp_uncertainties(emcor, emcor_sample, kpdat, wmdat, wmsig=None, emsig=None):
    plt.ioff()
    nkp = kpdat.shape[2]
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    gs = gridspec.GridSpec(2, 2)
    f = plt.figure(figsize=(2.00 * 6.4, 2.00 * 4.8))
    ax = plt.subplot(gs[0, 0])
    p00 = ax.imshow(emcor[0, 0], origin="lower", cmap="RdBu", vmin=-1.0, vmax=1.0)
    c00 = plt.colorbar(p00, ax=ax)
    c00.set_label("Kernel phase correlation", rotation=270, labelpad=20)
    ax.set_title(
        "Theoretical estimate",
        y=1.0,
        pad=-20,
        bbox=dict(facecolor="white", edgecolor="lightgrey", boxstyle="round"),
    )
    ax = plt.subplot(gs[0, 1])
    p01 = ax.imshow(
        emcor_sample[0, 0], origin="lower", cmap="RdBu", vmin=-1.0, vmax=1.0
    )
    c01 = plt.colorbar(p01, ax=ax)
    c01.set_label("Kernel phase correlation", rotation=270, labelpad=20)
    ax.set_title(
        "Empirical estimate",
        y=1.0,
        pad=-20,
        bbox=dict(facecolor="white", edgecolor="lightgrey", boxstyle="round"),
    )
    ax = plt.subplot(gs[1, :])
    ax.fill_between(
        np.arange(nkp),
        np.min(kpdat[:, 0], axis=0),
        np.max(kpdat[:, 0], axis=0),
        edgecolor="None",
        facecolor=colors[0],
        alpha=1.0 / 3.0,
        label="Min/max range",
    )
    if emsig is not None:
        ax.errorbar(
            np.arange(nkp),
            wmdat[0, 0],
            yerr=emsig[0, 0],
            color=colors[0],
            label="Weighted mean",
        )
    elif wmsig is not None:
        ax.errorbar(
            np.arange(nkp),
            wmdat[0, 0],
            yerr=wmsig[0, 0],
            color=colors[0],
            label="Weighted mean",
        )
    else:
        raise TypeError("One of wmsig or emsig is expected to be provided.")
    ax.axhline(0.0, ls="--", color="black")
    ylim = ax.get_ylim()
    ax.set_ylim([-np.max(np.abs(ylim)), np.max(np.abs(ylim))])
    ax.grid(axis="y")
    ax.set_xlabel("Index")
    ax.set_ylabel("Kernel phase [rad]")
    ax.legend(loc="upper right")
    plt.suptitle("Empirical uncertainties step", size=18)
    plt.tight_layout()

    return f
