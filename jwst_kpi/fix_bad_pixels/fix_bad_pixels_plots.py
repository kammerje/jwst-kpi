import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import numpy as np


def plot_badpix(data, data_bpfixed, bad_bits, mask, good_frames, method="medfilt"):
    plt.ioff()
    f, ax = plt.subplots(1, 3, figsize=(2.25 * 6.4, 0.75 * 4.8))
    if good_frames is None:
        p0 = ax[0].imshow(mask[0], origin="lower")
    else:
        p0 = ax[0].imshow(mask[good_frames[0]], origin="lower")
    plt.colorbar(p0, ax=ax[0])
    t0 = ax[0].text(
        0.01,
        0.01,
        bad_bits,
        color="white",
        ha="left",
        va="bottom",
        transform=ax[0].transAxes,
        size=12,
    )
    t0.set_path_effects(
        [PathEffects.withStroke(linewidth=3, foreground="black")]
    )
    ax[0].set_title(
        "Bad pixel map",
        y=1.0,
        pad=-20,
        bbox=dict(facecolor="white", edgecolor="lightgrey", boxstyle="round"),
    )
    if good_frames is None:
        p1 = ax[1].imshow(np.log10(np.abs(data[0])), origin="lower")
    else:
        p1 = ax[1].imshow(
            np.log10(np.abs(data[good_frames[0]])), origin="lower"
        )
    plt.colorbar(p1, ax=ax[1])
    ax[1].set_title(
        "Frame (log)",
        y=1.0,
        pad=-20,
        bbox=dict(facecolor="white", edgecolor="lightgrey", boxstyle="round"),
    )
    if good_frames is None:
        p2 = ax[2].imshow(np.log10(np.abs(data_bpfixed[0])), origin="lower")
    else:
        p2 = ax[2].imshow(
            np.log10(np.abs(data_bpfixed[good_frames[0]])), origin="lower"
        )
    plt.colorbar(p2, ax=ax[2])
    t2 = ax[2].text(
        0.01,
        0.01,
        "method = " + method,
        color="white",
        ha="left",
        va="bottom",
        transform=ax[2].transAxes,
        size=12,
    )
    t2.set_path_effects(
        [PathEffects.withStroke(linewidth=3, foreground="black")]
    )
    ax[2].set_title(
        "Fixed frame (log)",
        y=1.0,
        pad=-20,
        bbox=dict(facecolor="white", edgecolor="lightgrey", boxstyle="round"),
    )
    plt.suptitle("Fix bad pixels step", size=18)
    plt.tight_layout()

    return f
