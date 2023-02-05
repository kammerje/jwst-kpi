import os

import matplotlib.pyplot as plt
import numpy as np
from jwst.stpipe import Step

from .. import utils as ut
from ..datamodels import KPFitsModel
from .empirical_uncertainties_plots import plot_emp_uncertainties


class EmpiricalUncertaintiesStep(Step):
    """
    Compute empirical uncertainties for the kernel phase.
    """

    class_alias = "empirical_uncertainties"

    spec = """
        plot = boolean(default=True)
        previous_suffix = string(default=None)
        show_plots = boolean(default=False)
        get_emp_err = boolean(default=True)
        get_emp_cor = boolean(default=False)
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
        """

        self.log.info("--> Running empirical uncertainties step...")

        # Open file.
        # TODO: Could use "open" if it can figure out datamodels
        if self.previous_suffix is None:
            input_models = KPFitsModel(input_data)
        else:
            raise ValueError("Unexpected previous_suffix attribute")
        kpdat = input_models.kp_data
        kpsig = input_models.kp_sigm
        kpcov = input_models.kp_cov
        if kpdat.ndim != 3 or kpsig.ndim != 3 or kpcov.ndim != 4:
            raise UserWarning("Input file is not a valid KPFITS file")
        if kpdat.shape[0] < 3:
            self.log.warning(
                "--> Not enough frames to estimate uncertainties empirically, skipping step!"
            )
            return input_models

        #
        nf = kpdat.shape[0]

        self.log.info("--> Estimating empirical uncertainties from %.0f frames" % nf)

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

        mk_path = self.make_output_path()
        stem = os.path.splitext(mk_path)[0]

        # Plot.
        if self.plot:
            emsig_plot_arg = emsig if self.get_emp_err else None
            plot_emp_uncertainties(
                emcor,
                emcor_sample,
                kpdat,
                wmdat,
                wmsig=wmsig,
                emsig=emsig_plot_arg,
            )
            plt.savefig(stem + ".pdf")
            if self.show_plots:
                plt.show()
            plt.close()

        # Save file.
        output_models = input_models.copy()
        output_models.kp_data = wmdat
        if self.get_emp_err:
            if self.get_emp_cor:
                output_models.kp_sigm = emsig_sample
                output_models.kp_cov = emcov_sample
            else:
                output_models.kp_sigm = emsig
                output_models.kp_cov = emcov
        else:
            output_models.kp_sigm = wmsig
            output_models.kp_cov = wmcov
        output_models.detpa = np.array([np.mean(input_models.detpa)])

        self.log.info("--> Empirical uncertainties step done")

        return output_models

    def remove_suffix(self, name):
        new_name, separator = super(EmpiricalUncertaintiesStep, self).remove_suffix(name)
        if new_name == name:
            new_name, separator = ut.remove_suffix_kpi(new_name)
        return new_name, separator
