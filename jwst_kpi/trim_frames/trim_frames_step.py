import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from scipy.ndimage import median_filter
import matplotlib.patheffects as PathEffects
from jwst.stpipe import Step

from .. import utils as ut


class TrimFramesStep(Step):
    """
    Trim frames.
    """

    class_alias = "trim_frames"

    spec = """
    """

    def __init__(self):
        """
        Initialize the pipeline step.
        """

        # Initialize the step parameters.
        self.skip = False
        self.plot = True
        self.trim_cent = None
        self.trim_halfsize = 32  # pix

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

        print("--> Running trim frames step...")

        # Open file.
        if suffix == "":
            hdul = ut.open_fits(file, suffix=suffix, file_dir=None)
        else:
            hdul = ut.open_fits(file, suffix=suffix, file_dir=output_dir)
        data = hdul["SCI"].data
        erro = hdul["ERR"].data
        pxdq = hdul["DQ"].data
        if data.ndim not in [2, 3]:
            raise UserWarning("Only implemented for 2D image/3D data cube")
        if data.ndim == 2:
            is2d = True
            data = data[np.newaxis]
            erro = erro[np.newaxis]
            pxdq = pxdq[np.newaxis]
        else:
            is2d = False
        nf, sy, sx = data.shape
        if good_frames is not None:
            # NOTE: This will raise error if good_frames is int. Supporting only list seems fine
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
        suffix_out = "_trimmed"

        # Trim frames. Need to trim all frames (also bad ones) to match their
        # shapes.
        if self.trim_cent is None:
            ww_max = []
            for i in range(nf):
                ww_max += [
                    np.unravel_index(
                        np.nanargmax(median_filter(data[i], size=3)), data[i].shape
                    )
                ]
            ww_max = np.array(ww_max)
        else:
            ww_max = np.array([[self.trim_cent[0], self.trim_cent[1]]] * nf)
        if good_frames is None:
            if np.max(np.abs(ww_max - np.mean(ww_max, axis=0))) > 3.0:
                raise UserWarning(
                    "Brightest source jitters by more than 3 pixels, use trim_frames.trim_cent to specify source position"
                )
            ww_max = [
                int(round(np.median(ww_max[:, 0]))),
                int(round(np.median(ww_max[:, 1]))),
            ]
        else:
            if (
                np.max(
                    np.abs(ww_max[good_frames] - np.mean(ww_max[good_frames], axis=0))
                )
                > 3.0
            ):
                raise UserWarning(
                    "Brightest source jitters by more than 3 pixels, use trim_frames.trim_cent to specify source position"
                )
            ww_max = [
                int(round(np.median(ww_max[good_frames, 0]))),
                int(round(np.median(ww_max[good_frames, 1]))),
            ]
        xh_max = min(sx - ww_max[1], ww_max[1])
        yh_max = min(sy - ww_max[0], ww_max[0])
        sh_max = min(xh_max, yh_max)
        self.trim_halfsize = min(sh_max, self.trim_halfsize)

        print(
            "--> Trimming frames around (x, y) = (%.0f, %.0f) to a size of %.0fx%.0f pixels"
            % (ww_max[1], ww_max[0], 2 * self.trim_halfsize, 2 * self.trim_halfsize)
        )

        for i in range(nf):
            data_trimmed = data[
                :,
                ww_max[0] - self.trim_halfsize : ww_max[0] + self.trim_halfsize,
                ww_max[1] - self.trim_halfsize : ww_max[1] + self.trim_halfsize,
            ].copy()
            erro_trimmed = erro[
                :,
                ww_max[0] - self.trim_halfsize : ww_max[0] + self.trim_halfsize,
                ww_max[1] - self.trim_halfsize : ww_max[1] + self.trim_halfsize,
            ].copy()
            pxdq_trimmed = pxdq[
                :,
                ww_max[0] - self.trim_halfsize : ww_max[0] + self.trim_halfsize,
                ww_max[1] - self.trim_halfsize : ww_max[1] + self.trim_halfsize,
            ].copy()

        # Get output file path.
        path = ut.get_output_base(file, output_dir=output_dir)

        # Plot.
        if self.plot:
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
                    ww_max[1] - self.trim_halfsize - 0.5,
                    ww_max[0] - self.trim_halfsize - 0.5,
                ),
                2 * self.trim_halfsize,
                2 * self.trim_halfsize,
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
                    ww_max[1] - self.trim_halfsize - 0.5,
                    ww_max[0] - self.trim_halfsize - 0.5,
                ),
                2 * self.trim_halfsize,
                2 * self.trim_halfsize,
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
            plt.savefig(path + suffix_out + ".pdf")
            if show_plots:
                plt.show()
            plt.close()

        # Save file.
        if is2d:
            data_trimmed = data_trimmed[0]
            erro_trimmed = erro_trimmed[0]
            pxdq_trimmed = pxdq_trimmed[0]
        hdul["SCI"].data = data_trimmed
        hdul["SCI"].header["CENT_X"] = ww_max[1]
        hdul["SCI"].header["CENT_Y"] = ww_max[0]
        hdul["ERR"].data = erro_trimmed
        hdul["ERR"].header["CENT_X"] = ww_max[1]
        hdul["ERR"].header["CENT_Y"] = ww_max[0]
        hdul["DQ"].data = pxdq_trimmed
        hdul["DQ"].header["CENT_X"] = ww_max[1]
        hdul["DQ"].header["CENT_Y"] = ww_max[0]
        hdul.writeto(path + suffix_out + ".fits", output_verify="fix", overwrite=True)
        hdul.close()

        print("--> Trim frames step done")

        return suffix_out
