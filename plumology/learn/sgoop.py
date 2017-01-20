'''SGOOP - Spectral Gap Optimization Of Order Parameters'''

from typing import Tuple

import numpy as np
from pyemma import msm
from scipy.optimize import basinhopping


class SGOOP:
    '''
    SGOOP - Spectral Gap Optimization Of Order Parameters [EXPERIMENTAL]

    Given a trajectory along a trial CV, calculates
    the ideal linear combination maximizing the spectral gap
    of the associated transition matrix omega.

    Parameters
    ----------
    data : Dateframe containing CV data and weights over time.
    weight_name : Name of the column containing the weights.
    rho : Lagrange multiplier to use for omega calculation.
    size : Discretization interval for the trial CV.
    lag_time : Lag time to use for MSM estimation.
    n : Use the nth and n+1th eigenvalue for calculating the spectral gap.

    Attributes
    ----------
    coeffs : The coefficients found through optimization.

    References
    ----------
    Tiwary, P., & Berne, B. J. (2016). "Spectral gap optimization
    of order parameters for sampling complex molecular systems"
    PNAS, 113(11), 2839–2844. http://doi.org/10.1073/pnas.1600917113

    '''
    def __init__(
        self,
        data: np.ndarray,
        weights: np.ndarray=None,
        rho: float=0.0,
        size: int=16,
        lag_time: int=12,
        n: int=3
    ) -> None:

        self._data = data
        self._rho = rho
        self._size = size
        self._lag_time = lag_time
        self._weights = weights
        self._n = n

    def fit(self, niter: int=1000) -> Tuple[float, np.ndarray]:
        '''
        Optimize the spectral gap using basin hopping.

        Parameters
        ----------
        niter : Number of iterations to use.

        Returns
        -------
        coeffs : The found ideal coefficients.

        '''
        dim = self._data.shape[1]
        rs = np.ones((dim))
        rs *= 1 / np.sqrt((rs ** 2).sum())
        result = basinhopping(
            func=self._score,
            x0=rs,
            niter=niter,
            minimizer_kwargs=dict(
                method='L-BFGS-B',
                bounds=[(0.0001, 1.) for _ in range(dim)]
            ),
            stepsize=0.1,
            T=2.5
        )
        self.coeffs = result['x']
        return -result['fun'], self.coeffs

    def _find_omega_msm(self) -> np.ndarray:
        _, bins = np.histogram(self._newcv, self._size)
        newcv_ind = np.digitize(self._newcv, bins)
        return msm.estimate_markov_model(
            newcv_ind, self._lag_time, reversible=False
        ).eigenvalues()

    def _score(self, coeffs: np.ndarray) -> float:
        '''
        Calculates the negative of the spectral gap given a set of coefficients

        '''
        coeffs *= 1 / np.sqrt((coeffs ** 2).sum())
        self._newcv = (self._data * coeffs).sum(axis=1)
        self.eigen = self._find_omega_msm()
        maxgap = abs(self.eigen[self._n] - self.eigen[self._n + 1])
        return -maxgap
