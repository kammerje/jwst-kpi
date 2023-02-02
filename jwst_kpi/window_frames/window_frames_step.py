import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import matplotlib.patheffects as PathEffects
from xara import core

from . import utils as ut


class window_frames:
    """
    Window frames.
    """

    def __init__(self):
        """
        Initialize the pipeline step.
        """

        # Initialize the step parameters.
        self.skip = False
        self.plot = True
        self.wrad = 24  # pix

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

        print("--> Running window frames step...")

        # Open file.
        if suffix == "":
            hdul = ut.open_fits(file, suffix=suffix, file_dir=None)
        else:
            hdul = ut.open_fits(file, suffix=suffix, file_dir=output_dir)
        data = hdul["SCI"].data
        erro = hdul["ERR"].data
        if data.ndim not in [2, 3]:
            raise UserWarning("Only implemented for 2D image/3D data cube")
        if data.ndim == 2:
            is2d = True
            data = data[np.newaxis]
            erro = erro[np.newaxis]
        else:
            is2d = False
        nf, sy, sx = data.shape
        if good_frames is not None:
            if len(good_frames) < 1:
                raise UserWarning(
                    "List of good frames needs to contain at least one element"
                )
            elif not all(isinstance(item, int) for item in good_frames):
                raise UserWarning(
                    "List of good frames may only contain integer elements"
                )
            elif np.max(good_frames) >= nf or np.min(good_frames) < nf * (-1):
                raise UserWarning(
                    "Some of the provided good frames are outside the data range"
                )

        # Suffix for the file path from the current step.
        suffix_out = "_windowed"

        # Window.
        if self.wrad is None:
            self.wrad = 24

        print(
            "--> Windowing frames with super-Gaussian mask with radius %.0f pixels"
            % self.wrad
        )

        data_windowed = data.copy()
        erro_windowed = erro.copy()
        sgmask = core.super_gauss(sy, sx, self.wrad, between_pix=False)
        if good_frames is None:
            data_windowed *= sgmask
            erro_windowed *= sgmask
        else:
            for i in range(nf):
                if i in good_frames or (nf - i) * (-1) in good_frames:
                    data_windowed[i] *= sgmask
                    erro_windowed[i] *= sgmask

        # Get output file path.
        path = ut.get_output_base(file, output_dir=output_dir)

        # Plot.
        if self.plot:
            plt.ioff()
            f, ax = plt.subplots(1, 2, figsize=(1.50 * 6.4, 0.75 * 4.8))
            if good_frames is None:
                p0 = ax[0].imshow(data_windowed[0], origin="lower")
            else:
                p0 = ax[0].imshow(data_windowed[good_frames[0]], origin="lower")
            plt.colorbar(p0, ax=ax[0])
            c0 = plt.Circle(
                (sx // 2, sy // 2),
                self.wrad,
                color="red",
                ls="--",
                fill=False,
            )
            ax[0].add_patch(c0)
            t0 = ax[0].text(
                0.01,
                0.01,
                "wrad = %.0f pix" % self.wrad,
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
                self.wrad,
                color="red",
                ls="--",
                fill=False,
            )
            ax[1].add_patch(c1)
            t1 = ax[1].text(
                0.01,
                0.01,
                "wrad = %.0f pix" % self.wrad,
                color="white",
                ha="left",
                va="bottom",
                transform=ax[1].transAxes,
                size=12,
            )
            t1.set_path_effects(
                [PathEffects.withStroke(linewidth=3, foreground="black")]
            )
            ax[1].set_title(
                "Windowed frame (log)",
                y=1.0,
                pad=-20,
                bbox=dict(facecolor="white", edgecolor="lightgrey", boxstyle="round"),
            )
            plt.suptitle("Window frames step", size=18)
            plt.tight_layout()
            plt.savefig(path + suffix_out + ".pdf")
            if show_plots:
                plt.show()
            plt.close()

        # Save file.
        if is2d:
            data = data[0]
            erro = erro[0]
            data_windowed = data_windowed[0]
            erro_windowed = erro_windowed[0]
        try:
            hdul.index_of("SCI-ORG")
        except Exception:
            hdu_sci_org = fits.ImageHDU(data)
            hdu_sci_org.header["EXTNAME"] = "SCI-ORG"
            hdu_err_org = fits.ImageHDU(erro)
            hdu_err_org.header["EXTNAME"] = "ERR-ORG"
            hdul += [hdu_sci_org, hdu_err_org]
        hdul["SCI"].data = data_windowed
        hdul["SCI"].header["WRAD"] = self.wrad
        hdul["ERR"].data = erro_windowed
        hdul["ERR"].header["WRAD"] = self.wrad
        hdu_win = fits.ImageHDU(sgmask)
        hdu_win.header["EXTNAME"] = "WINMASK"
        hdu_win.header["WRAD"] = self.wrad
        hdul += [hdu_win]
        hdul.writeto(path + suffix_out + ".fits", output_verify="fix", overwrite=True)
        hdul.close()

        print("--> Window frames step done")

        return suffix_out
