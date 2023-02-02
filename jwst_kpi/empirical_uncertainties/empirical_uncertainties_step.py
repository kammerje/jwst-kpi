import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

from . utils import ut


class empirical_uncertainties:
    """
    Compute empirical uncertainties for the kernel phase.
    """

    def __init__(self):
        """
        Initialize the pipeline step.
        """

        # Initialize the step parameters.
        self.skip = False
        self.plot = True
        self.get_emp_err = True
        self.get_emp_cor = False

    def step(
        self,
        file,
        suffix,
        output_dir=None,
        show_plots=False,
        good_frames=None,
    ):
        """
        Run the pipeline step.

        Parameters
        ----------
        file: str
            Path to stage 2-calibrated pipeline product.
        suffix: str
            Suffix for the file path to find the product from the previous
            step.
        output_dir: str
            Output directory, if None uses same directory as input file.
        show_plots: bool
            Show plots?
        good_frames: list of int
            List of good frames, bad frames will be skipped.
        """

        print("--> Running empirical uncertainties step...")

        # Open file.
        if suffix == "":
            hdul = ut.open_fits(file, suffix=suffix, file_dir=None)
        else:
            hdul = ut.open_fits(file, suffix=suffix, file_dir=output_dir)
        kpdat = hdul["KP-DATA"].data
        kpsig = hdul["KP-SIGM"].data
        kpcov = hdul["KP-COV"].data
        if kpdat.ndim != 3 or kpsig.ndim != 3 or kpcov.ndim != 4:
            raise UserWarning("Input file is not a valid KPFITS file")
        if kpdat.shape[0] < 3:
            print(
                "--> Not enough frames to estimate uncertainties empirically, skipping step!"
            )
            return suffix

        # Suffix for the file path from the current step.
        suffix_out = "_emp_kpfits"

        #
        nf = kpdat.shape[0]

        print("--> Estimating empirical uncertainties from %.0f frames" % nf)

        nwave = kpdat.shape[1]
        nkp = kpdat.shape[2]
        wmdat = np.zeros((1, nwave, nkp))
        wmsig = np.zeros((1, nwave, nkp))
        wmcov = np.zeros((1, nwave, nkp, nkp))
        emsig = np.zeros((1, nwave, nkp))  # empirical uncertainties
        emcov = np.zeros((1, nwave, nkp, nkp))  # empirical uncertainties
        emcor = np.zeros((1, nwave, nkp, nkp))  # empirical uncertainties
        emsig_sample = np.zeros((1, nwave, nkp))  # empirical uncertainties
        emcov_sample = np.zeros((1, nwave, nkp, nkp))  # empirical uncertainties
        emcor_sample = np.zeros((1, nwave, nkp, nkp))  # empirical uncertainties
        for i in range(nwave):
            invcov = []
            invcovdat = []
            for j in range(nf):
                invcov += [np.linalg.inv(kpcov[j, i])]
                invcovdat += [invcov[-1].dot(kpdat[j, i])]
            wmcov[0, i] = np.linalg.inv(np.sum(np.array(invcov), axis=0))
            wmsig[0, i] = np.sqrt(np.diag(wmcov[0, i]))
            wmdat[0, i] = wmcov[0, i].dot(np.sum(np.array(invcovdat), axis=0))
            emsig[0, i] = np.std(kpdat[:, i], axis=0) / np.sqrt(nf)
            wmcor = np.true_divide(
                wmcov[0, i], wmsig[0, i][:, None] * wmsig[0, i][None, :]
            )
            emcov[0, i] = np.multiply(
                wmcor, emsig[0, i][:, None] * emsig[0, i][None, :]
            )
            emcor[0, i] = wmcor.copy()
            emcov_sample[0, i] = np.cov(kpdat[:, i].T)
            emsig_sample[0, i] = np.sqrt(np.diag(emcov_sample[0, i]))
            emcor_sample[0, i] = np.true_divide(
                emcov_sample[0, i],
                emsig_sample[0, i][:, None] * emsig_sample[0, i][None, :],
            )

        # Get output file path.
        path = ut.get_output_base(file, output_dir=output_dir)

        # Plot.
        if self.plot:
            colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
            plt.ioff()
            gs = gridspec.GridSpec(2, 2)
            f = plt.figure(figsize=(2.00 * 6.4, 2.00 * 4.8))
            ax = plt.subplot(gs[0, 0])
            p00 = ax.imshow(
                emcor[0, 0], origin="lower", cmap="RdBu", vmin=-1.0, vmax=1.0
            )
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
            if self.get_emp_err:
                ax.errorbar(
                    np.arange(nkp),
                    wmdat[0, 0],
                    yerr=emsig[0, 0],
                    color=colors[0],
                    label="Weighted mean",
                )
            else:
                ax.errorbar(
                    np.arange(nkp),
                    wmdat[0, 0],
                    yerr=wmsig[0, 0],
                    color=colors[0],
                    label="Weighted mean",
                )
            ax.axhline(0.0, ls="--", color="black")
            ylim = ax.get_ylim()
            ax.set_ylim([-np.max(np.abs(ylim)), np.max(np.abs(ylim))])
            ax.grid(axis="y")
            ax.set_xlabel("Index")
            ax.set_ylabel("Kernel phase [rad]")
            ax.legend(loc="upper right")
            plt.suptitle("Empirical uncertainties step", size=18)
            plt.tight_layout()
            plt.savefig(path + suffix_out + ".pdf")
            if show_plots:
                plt.show()
            plt.close()

        # Save file.
        hdul["KP-DATA"].data = wmdat
        if self.get_emp_err:
            if self.get_emp_cor:
                hdul["KP-SIGM"].data = emsig_sample
                hdul["KP-COV"].data = emcov_sample
            else:
                hdul["KP-SIGM"].data = emsig
                hdul["KP-COV"].data = emcov
        else:
            hdul["KP-SIGM"].data = wmsig
            hdul["KP-COV"].data = wmcov
        hdul["DETPA"].data = np.array([np.mean(hdul["DETPA"].data)])
        hdul.writeto(path + suffix_out + ".fits", output_verify="fix", overwrite=True)
        hdul.close()

        print("--> Empirical uncertainties step done")

        return suffix_out
