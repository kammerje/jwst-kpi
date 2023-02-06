import matplotlib.patheffects as PathEffects
import matplotlib.pyplot as plt
import numpy as np


def plot_window(data_windowed, wrad, good_frames=None):
    """
    Plot windowed data

    Parameters
    ----------
    data_windowed : np.ndarray
        Windowed data
    wrad : int
        Windowing radius for supergaussian
    good_frames : List[int]
        List of good frames, bad frames will be skipped.
    """
    plt.ioff()
    _, sy, sx = data_windowed.shape
    f, ax = plt.subplots(1, 2, figsize=(1.50 * 6.4, 0.75 * 4.8))
    if good_frames is None:
        p0 = ax[0].imshow(data_windowed[0], origin="lower")
    else:
        p0 = ax[0].imshow(data_windowed[good_frames[0]], origin="lower")
    plt.colorbar(p0, ax=ax[0])
    c0 = plt.Circle(
        (sx // 2, sy // 2),
        wrad,
        color="red",
        ls="--",
        fill=False,
    )
    ax[0].add_patch(c0)
    t0 = ax[0].text(
        0.01,
        0.01,
        "wrad = %.0f pix" % wrad,
        color="white",
        ha="left",
        va="bottom",
        transform=ax[0].transAxes,
        size=12,
    )
    t0.set_path_effects([PathEffects.withStroke(linewidth=3, foreground="black")])
    ax[0].set_title(
        "Windowed frame",
        y=1.0,
        pad=-20,
        bbox=dict(facecolor="white", edgecolor="lightgrey", boxstyle="round"),
    )
    if good_frames is None:
        p1 = ax[1].imshow(np.log10(np.abs(data_windowed[0])), origin="lower")
    else:
        p1 = ax[1].imshow(
            np.log10(np.abs(data_windowed[good_frames[0]])), origin="lower"
        )
    plt.colorbar(p1, ax=ax[1])
    c1 = plt.Circle(
        (sx // 2, sy // 2),
        wrad,
        color="red",
        ls="--",
        fill=False,
    )
    ax[1].add_patch(c1)
    t1 = ax[1].text(
        0.01,
        0.01,
        "wrad = %.0f pix" % wrad,
        color="white",
        ha="left",
        va="bottom",
        transform=ax[1].transAxes,
        size=12,
    )
    t1.set_path_effects([PathEffects.withStroke(linewidth=3, foreground="black")])
    ax[1].set_title(
        "Windowed frame (log)",
        y=1.0,
        pad=-20,
        bbox=dict(facecolor="white", edgecolor="lightgrey", boxstyle="round"),
    )
    plt.suptitle("Window frames step", size=18)
    plt.tight_layout()

    return f
