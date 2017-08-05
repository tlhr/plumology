import pytest

import numpy as np
import pandas as pd
from plumology import util

arr = [[1.0, 10.5, 0.2], [-0.3, 0.5, 0.4], [-0.2, 0.5, 0.4]]
data = pd.DataFrame(data=arr, columns=['a', 'b', 'c'])
expdata = data.rename(columns={'c': 'exp_a'}).drop('b', axis=1)


class Test_last_nonzero:
    def test_1(self):
        assert (util.last_nonzero(data.copy()).values ==
                np.array([-0.2, 0.5, 0.4])).all()

    def test_2(self):
        ndata = data.copy()
        ndata.iloc[2, 1] = 0
        assert (util.last_nonzero(ndata).values ==
                np.array([-0.2, 0.5, 0.4])).all()

    def test_3(self):
        ndata = data.copy()
        ndata.iloc[2, 1] = 0
        ndata['c'] = 0.0
        assert (util.last_nonzero(ndata).values ==
                np.array([-0.2, 0.5, 0.0])).all()


class Test_dict_to_dataframe:
    def test_1(self):
        ndata = {'alpha': data.copy(), 'beta': data.copy().assign(a=[0, 1, 2])}
        arr = np.array([[1., 10.5, 0.2],
                        [0., 10.5, 0.2],
                        [-0.3, 0.5, 0.4],
                        [1., 0.5, 0.4],
                        [-0.2, 0.5, 0.4],
                        [2., 0.5, 0.4]])
        res = util.dict_to_dataframe(ndata).values
        assert (res.round(decimals=2) == arr.round(decimals=2)).all()
