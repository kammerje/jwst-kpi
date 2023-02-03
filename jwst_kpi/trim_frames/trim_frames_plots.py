import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patheffects as PathEffects


def plot_trim(data, data_trimmed, pxdq, pxdq_trimmed, ww_max, trim_halfsize, good_frames=None):
    plt.ioff()
    f, ax = plt.subplots(2, 2, figsize=(1.50 * 6.4, 1.50 * 4.8))
    if good_frames is None:
        p00 = ax[0, 0].imshow(np.log10(np.abs(data[0])), origin="lower")
    else:
        p00 = ax[0, 0].imshow(
            np.log10(np.abs(data[good_frames[0]])), origin="lower"
        )
    plt.colorbar(p00, ax=ax[0, 0])
    r00 = Rectangle(
        (
            ww_max[1] - trim_halfsize - 0.5,
            ww_max[0] - trim_halfsize - 0.5,
        ),
        2 * trim_halfsize,
        2 * trim_halfsize,
        facecolor="none",
        edgecolor="red",
    )
    ax[0, 0].add_patch(r00)
    t00 = ax[0, 0].text(
        0.01,
        0.01,
        "center = %.0f, %.0f" % (ww_max[1], ww_max[0]),
        color="white",
        ha="left",
        va="bottom",
        transform=ax[0, 0].transAxes,
        size=12,
    )
    t00.set_path_effects(
        [PathEffects.withStroke(linewidth=3, foreground="black")]
    )
    ax[0, 0].set_title(
        "Full frame (log)",
        y=1.0,
        pad=-20,
        bbox=dict(facecolor="white", edgecolor="lightgrey", boxstyle="round"),
    )
    if good_frames is None:
        p01 = ax[0, 1].imshow(pxdq[0] & 1 == 1, origin="lower")
    else:
        p01 = ax[0, 1].imshow(pxdq[good_frames[0]] & 1 == 1, origin="lower")
    plt.colorbar(p01, ax=ax[0, 1])
    r01 = Rectangle(
        (
            ww_max[1] - trim_halfsize - 0.5,
            ww_max[0] - trim_halfsize - 0.5,
        ),
        2 * trim_halfsize,
        2 * trim_halfsize,
        facecolor="none",
        edgecolor="red",
    )
    ax[0, 1].add_patch(r01)
    t01 = ax[0, 1].text(
        0.01,
        0.01,
        "DO_NOT_USE",
        color="white",
        ha="left",
        va="bottom",
        transform=ax[0, 1].transAxes,
        size=12,
    )
    t01.set_path_effects(
        [PathEffects.withStroke(linewidth=3, foreground="black")]
    )
    ax[0, 1].set_title(
        "Bad pixel map",
        y=1.0,
        pad=-20,
        bbox=dict(facecolor="white", edgecolor="lightgrey", boxstyle="round"),
    )
    if good_frames is None:
        p10 = ax[1, 0].imshow(np.log10(np.abs(data_trimmed[0])), origin="lower")
    else:
        p10 = ax[1, 0].imshow(
            np.log10(np.abs(data_trimmed[good_frames[0]])), origin="lower"
        )
    plt.colorbar(p10, ax=ax[1, 0])
    t10 = ax[1, 0].text(
        0.01,
        0.01,
        "center = %.0f, %.0f" % (ww_max[1], ww_max[0]),
        color="white",
        ha="left",
        va="bottom",
        transform=ax[1, 0].transAxes,
        size=12,
    )
    t10.set_path_effects(
        [PathEffects.withStroke(linewidth=3, foreground="black")]
    )
    ax[1, 0].set_title(
        "Trimmed frame (log)",
        y=1.0,
        pad=-20,
        bbox=dict(facecolor="white", edgecolor="lightgrey", boxstyle="round"),
    )
    if good_frames is None:
        p11 = ax[1, 1].imshow(pxdq_trimmed[0] & 1 == 1, origin="lower")
    else:
        p11 = ax[1, 1].imshow(
            pxdq_trimmed[good_frames[0]] & 1 == 1, origin="lower"
        )
    plt.colorbar(p11, ax=ax[1, 1])
    t11 = ax[1, 1].text(
        0.01,
        0.01,
        "DO_NOT_USE",
        color="white",
        ha="left",
        va="bottom",
        transform=ax[1, 1].transAxes,
        size=12,
    )
    t11.set_path_effects(
        [PathEffects.withStroke(linewidth=3, foreground="black")]
    )
    ax[1, 1].set_title(
        "Trimmed bad pixel map",
        y=1.0,
        pad=-20,
        bbox=dict(facecolor="white", edgecolor="lightgrey", boxstyle="round"),
    )
    plt.suptitle("Trim frames step", size=18)
    plt.tight_layout()

    return f
