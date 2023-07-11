import numpy as np
import matplotlib.pyplot as plt


def plot_kerphase(data, KPO, m2pix, kpcor, kpsig, good_frames=None):
    """
    Plot kernel phases

    Parameters
    ----------
    data : np.ndarray
        Input image data
    KPO : xara.kpo.KPO
        Xara KPO object
    m2pix : float
        meter to pixel conversion factor
    kpcor : np.ndarray
        Kernel phase correlations
    kpsig : np.ndarray
        Kernel phase errors
    good_frames : List[int]
        List of good frame indices to keep
    """
    plt.ioff()
    _, sy, sx = data.shape
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    f, ax = plt.subplots(2, 2, figsize=(1.50 * 6.4, 1.50 * 4.8))
    if good_frames is None:
        d00 = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(data[0])))
    else:
        d00 = np.fft.fftshift(
            np.fft.fft2(np.fft.fftshift(data[good_frames[0]]))
        )
    p00 = ax[0, 0].imshow(
        np.angle(d00), origin="lower", vmin=-np.pi, vmax=np.pi
    )
    c00 = plt.colorbar(p00, ax=ax[0, 0])
    c00.set_label("Fourier phase [rad]", rotation=270, labelpad=20)
    xx = KPO.kpi.UVC[:, 0] * m2pix + sx // 2
    yy = KPO.kpi.UVC[:, 1] * m2pix + sy // 2
    ax[0, 0].scatter(xx, yy, s=0.2, c="red")
    xx = -KPO.kpi.UVC[:, 0] * m2pix + sx // 2
    yy = -KPO.kpi.UVC[:, 1] * m2pix + sy // 2
    ax[0, 0].scatter(xx, yy, s=0.2, c="red")
    ax[0, 0].set_title(
        "Fourier phase",
        y=1.0,
        pad=-20,
        bbox=dict(facecolor="white", edgecolor="lightgrey", boxstyle="round"),
    )
    p01 = ax[0, 1].imshow(
        kpcor[0], origin="lower", cmap="RdBu", vmin=-1.0, vmax=1.0
    )
    c01 = plt.colorbar(p01, ax=ax[0, 1])
    c01.set_label("Correlation", rotation=270, labelpad=20)
    ax[0, 1].set_title(
        "Kernel phase correlation",
        y=1.0,
        pad=-20,
        bbox=dict(facecolor="white", edgecolor="lightgrey", boxstyle="round"),
    )
    ww = np.argsort(KPO.kpi.BLEN)
    # ax[1, 0].plot(np.angle(KPO.CVIS[0][0, ww]))
    temp = np.angle(np.array(KPO.CVIS))
    medfp = np.median(temp, axis=(0, 1))
    minfp = np.min(temp, axis=(0, 1))
    maxfp = np.max(temp, axis=(0, 1))
    ax[1, 0].plot(medfp[ww])
    ax[1, 0].fill_between(np.arange(len(medfp)), minfp[ww], maxfp[ww], alpha=1. / 3.)
    ax[1, 0].axhline(0.0, ls="--", color="black")
    ax[1, 0].set_ylim([-np.pi, np.pi])
    ax[1, 0].grid(axis="y")
    ax[1, 0].set_xlabel("Index sorted by baseline length")
    ax[1, 0].set_ylabel("Fourier phase [rad]")
    ax[1, 0].set_title(
        "Fourier phase",
        y=1.0,
        pad=-20,
        bbox=dict(facecolor="white", edgecolor="lightgrey", boxstyle="round"),
    )
    ax[1, 1].errorbar(
    #     np.arange(KPO.KPDT[0][0, :].shape[0]),
    #     KPO.KPDT[0][0, :],
    #     yerr=kpsig[0],
    #     color=colors[0],
    #     alpha=1.0 / 3.0,
    # )
    # ax[1, 1].plot(
    #     np.arange(KPO.KPDT[0][0, :].shape[0]),
    #     KPO.KPDT[0][0, :],
    #     color=colors[0],
    # )
    temp = np.array(KPO.KPDT)
    medkp = np.median(temp, axis=(0, 1))
    minkp = np.min(temp, axis=(0, 1))
    maxkp = np.max(temp, axis=(0, 1))
    ax[1, 1].plot(medkp)
    ax[1, 1].fill_between(np.arange(len(medkp)), minkp, maxkp, alpha=1. / 3.)
    ax[1, 1].axhline(0.0, ls="--", color="black")
    ylim = ax[1, 1].get_ylim()
    ax[1, 1].set_ylim([-np.max(np.abs(ylim)), np.max(np.abs(ylim))])
    ax[1, 1].grid(axis="y")
    ax[1, 1].set_xlabel("Index")
    ax[1, 1].set_ylabel("Kernel phase [rad]")
    ax[1, 1].set_title(
        "Kernel phase",
        y=1.0,
        pad=-20,
        bbox=dict(facecolor="white", edgecolor="lightgrey", boxstyle="round"),
    )
    plt.suptitle("Extract kerphase step", size=18)
    plt.tight_layout()

    return f
