import matplotlib.patheffects as PathEffects
import matplotlib.pyplot as plt


def plot_recenter(data, data_recentered, dx, dy, good_frames=None):
    """
    Plot recentering.

    Parameters
    ----------
    data : np.ndarray
        Original data
    data_recentered : np.ndarray
        Recentered data
    dx : np.ndarray
        X offsets (per frame)
    dy : np.ndarray
        Y offsets (per frame)
    good_frames : List[int]
        List of good frames, bad frames will be skipped.
    """
    plt.ioff()
    _, sy, sx = data.shape
    f, ax = plt.subplots(1, 2, figsize=(1.50 * 6.4, 0.75 * 4.8))
    if good_frames is None:
        p0 = ax[0].imshow(data[0], origin="lower")
    else:
        p0 = ax[0].imshow(data[good_frames[0]], origin="lower")
    plt.colorbar(p0, ax=ax[0])
    if good_frames is None:
        ax[0].axvline(sx // 2 + dx[0], color="red")
        ax[0].axhline(sy // 2 + dy[0], color="red")
        t0 = ax[0].text(
            0.01,
            0.01,
            "center = %.2f, %.2f" % (sx // 2 + dx[0], sy // 2 + dy[0]),
            color="white",
            ha="left",
            va="bottom",
            transform=ax[0].transAxes,
            size=12,
        )
    else:
        ax[0].axvline(sx // 2 + dx[good_frames[0]], color="red")
        ax[0].axhline(sy // 2 + dy[good_frames[0]], color="red")
        t0 = ax[0].text(
            0.01,
            0.01,
            "center = %.2f, %.2f"
            % (sx // 2 + dx[good_frames[0]], sy // 2 + dy[good_frames[0]]),
            color="white",
            ha="left",
            va="bottom",
            transform=ax[0].transAxes,
            size=12,
        )
    t0.set_path_effects([PathEffects.withStroke(linewidth=3, foreground="black")])
    ax[0].set_title(
        "Frame",
        y=1.0,
        pad=-20,
        bbox=dict(facecolor="white", edgecolor="lightgrey", boxstyle="round"),
    )
    if good_frames is None:
        p1 = ax[1].imshow(data_recentered[0], origin="lower")
    else:
        p1 = ax[1].imshow(data_recentered[good_frames[0]], origin="lower")
    plt.colorbar(p1, ax=ax[1])
    ax[1].axvline(sx // 2, color="red")
    ax[1].axhline(sy // 2, color="red")
    t1 = ax[1].text(
        0.01,
        0.01,
        "center = %.2f, %.2f" % (sx // 2, sy // 2),
        color="white",
        ha="left",
        va="bottom",
        transform=ax[1].transAxes,
        size=12,
    )
    t1.set_path_effects([PathEffects.withStroke(linewidth=3, foreground="black")])
    ax[1].set_title(
        "Recentered frame",
        y=1.0,
        pad=-20,
        bbox=dict(facecolor="white", edgecolor="lightgrey", boxstyle="round"),
    )
    plt.suptitle("Recenter frames step", size=18)
    plt.tight_layout()

    return f
