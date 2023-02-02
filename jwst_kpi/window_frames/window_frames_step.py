import os

import matplotlib.patheffects as PathEffects
import matplotlib.pyplot as plt
import numpy as np
# from astropy.io import fits
from jwst import datamodels
from jwst.stpipe import Step
from xara import core

# from . import utils as ut
from .window_frames_plots import plot_window


WRAD_DEFAULT = 24


class WindowFramesStep(Step):
    """
    Window frames.
    """

    class_alias = "window_frames"

    # TODO: wrad is int or float?
    spec = f"""
        plot = boolean(default=True)
        previous_suffix = string(default=None)
        wrad = int(default={WRAD_DEFAULT})
        show_plots = boolean(default=False)
        good_frames = int_list(default=None)
    """

    def process(
        self,
        input_data,
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

        self.log.info("--> Running window frames step...")

        good_frames = self.good_frames

        # Open file.
        # TODO: Mention this in PR
        if self.previous_suffix is None:
            input_models = datamodels.open(input_data)
        else:
            raise ValueError("Unexpected previous_suffix attribute")
        # if suffix == "":
        #     hdul = ut.open_fits(file, suffix=suffix, file_dir=None)
        # else:
        #     hdul = ut.open_fits(file, suffix=suffix, file_dir=output_dir)
        data = input_models.data
        erro = input_models.err
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
        suffix_out = f"_{self.suffix or self.default_suffix()}"

        # Window.
        if self.wrad is None:
            self.log.warning(f"wrad was set to None. Using default value of {WRAD_DEFAULT}")
            self.wrad = WRAD_DEFAULT

        self.log.info(
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
        # TODO: Test and cleanup
        # path = ut.get_output_base(file, output_dir=output_dir)
        mk_path = self.make_output_path()
        path = os.path.splitext(mk_path)[0]

        # Plot.
        if self.plot:
            plot_window(data_windowed, self.wrad, good_frames=good_frames)
            plt.savefig(path + suffix_out + ".pdf")
            if self.show_plots:
                plt.show()
            plt.close()

        # Save file.
        # TODO: Mark step completed
        # TODO: How add keywords in pipeline?
        # TODO: Might want to just save this direclty here instead of using default saving mech
        if is2d:
            data = data[0]
            erro = erro[0]
            data_windowed = data_windowed[0]
            erro_windowed = erro_windowed[0]
        output_models = input_models.copy()
        output_models.data = data_windowed
        output_models.err = erro_windowed
        # try:
        #     hdul.index_of("SCI-ORG")
        # except Exception:
        #     hdu_sci_org = fits.ImageHDU(data)
        #     hdu_sci_org.header["EXTNAME"] = "SCI-ORG"
        #     hdu_err_org = fits.ImageHDU(erro)
        #     hdu_err_org.header["EXTNAME"] = "ERR-ORG"
        #     hdul += [hdu_sci_org, hdu_err_org]
        # hdul["SCI"].data = data_windowed
        # hdul["SCI"].header["WRAD"] = self.wrad
        # hdul["ERR"].data = erro_windowed
        # hdul["ERR"].header["WRAD"] = self.wrad
        # hdu_win = fits.ImageHDU(sgmask)
        # hdu_win.header["EXTNAME"] = "WINMASK"
        # hdu_win.header["WRAD"] = self.wrad
        # hdul += [hdu_win]
        # hdul.writeto(path + suffix_out + ".fits", output_verify="fix", overwrite=True)
        # hdul.close()

        self.log.info("--> Window frames step done")

        return output_models
