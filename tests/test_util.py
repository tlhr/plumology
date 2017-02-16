import pytest

import numpy as np
import pandas as pd
from plumology import util

arr = [[1.0, 10.5, 0.2], [-0.3, 0.5, 0.4], [-0.2, 0.5, 0.4]]
data = pd.DataFrame(data=arr, columns=['a', 'b', 'c'])


class Test_population:
    def test_1(self):
        with pytest.raises(KeyError):
            util.population(data.copy(), [(0.0, 0.5)])

    def test_2(self):
        res = util.population(data.copy(), [(0.0, 0.5)], cv_names=('a', 'b'))
        assert res == {(0.0, 0.5): 2}

    def test_3(self):
        res = util.population(
            data.copy(), [(0.0, 0.5)], cv_names=('a', 'b'), weight_name='c'
        )
        assert res == {(0.0, 0.5): 0.80000000000000004}

    def test_4(self):
        res = util.population(
            data.copy(), [(0.0, 0.5)], radius=100,
            cv_names=('a', 'b'), weight_name='c'
        )
        assert res == {(0.0, 0.5): 1.0}


class Test_clip:
    def test_1(self):
        res = util.clip(data.copy(),
                        {'a': (-5., 5.), 'b': (-5., 5.)},
                        ignore=['c'])
        assert (res == data.loc[1:]).all().all()

    def test_2(self):
        arr = [[-0.3, 0.5, 0.5], [-0.2, 0.5, 0.5]]
        res = util.clip(data.copy(),
                        {'a': (-5., 5.), 'b': (-5., 5.)},
                        weight_name='c')
        assert (arr == res.values).all()

    def test_3(self):
        arr = np.array([[1.0, 10.5, 0.2],
                        [-0.3, 0.5, 0.4],
                        [-0.2, 0.5, 0.4]])
        res = util.clip(data.copy(),
                        {'a': (-5., 5.)},
                        ignore=['b'],
                        weight_name='c')
        assert (res.values == arr).all()


class Test_stats:
    def test_1(self):
        arr = np.ones((32, 3))
        fields = ['1', '2', '3']
        res = [
            ('1                min =     1.0000 :: max =     1.0000 ::'
             ' mean =     1.0000 :: stddev =     0.0000 :: var =     0.0000'),
            ('2                min =     1.0000 :: max =     1.0000 ::'
             ' mean =     1.0000 :: stddev =     0.0000 :: var =     0.0000'),
            ('3                min =     1.0000 :: max =     1.0000 ::'
             ' mean =     1.0000 :: stddev =     0.0000 :: var =     0.0000')
        ]
        assert util.stats(fields, arr) == res

    def test_2(self):
        arr = np.ones((32, 4))
        fields = ['1', '2', '3']
        with pytest.raises(ValueError):
            util.stats(fields, arr)


class Test_chunk_range:
    def test_1(self):
        assert util.chunk_range(0, 10000, 3, 1000) == [1000, 5500.0, 10000.0]

    def test_2(self):
        assert util.chunk_range(5000, 9000, 3, 1000) == [6000, 7500.0, 9000.0]

    def test_3(self):
        assert util.chunk_range(-1, 9, 3, 2) == [1, 5.0, 9.0]

    def test_4(self):
        with pytest.raises(ValueError):
            util.chunk_range(-1, -10, -3, 2.5)

    def test_5(self):
        assert util.chunk_range(-1, 8, 3) == [2.0, 5.0, 8.0]

    def test_6(self):
        with pytest.raises(TypeError):
            util.chunk_range(-1, 10, 3.14159, 2.5)


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


class Test_calc_bse:
    def test_1(self):
        ndata = data.copy()
        ndata['time'] = [0, 1, 2]
        arr = np.array([[0.3410224, 2.7216552, 0.0544331]]).round(decimals=3)
        assert (util.calc_bse(ndata).values.round(decimals=3) == arr).all()

    def test_2(self):
        ndata = data.copy()
        ndata['time'] = [0, 1, 2]
        arr = np.array([[0.3410224, 2.7216552]])
        res = util.calc_bse(ndata, weight_name='c').values
        assert (res.round(decimals=3) == arr.round(decimals=3)).all()

    def test_3(self):
        ndata = data.copy()
        ndata['time'] = [0, 1, 2]
        arr = np.array([[0.3410224]])
        res = util.calc_bse(ndata, ignore=['b'], weight_name='c').values
        assert (res.round(decimals=3) == arr.round(decimals=3)).all()


class Test_calc_wham:
    def test_1(self):
        data = np.array([75, 73, 78, 73, 72, 71])
        arr = np.array([0.17445063, 0.07813428, 0.58199452,
                        0.07813428, 0.05229091, 0.03499538])
        res = util.calc_wham(data, kbt=2.49)
        assert (res.round(decimals=3) == arr.round(decimals=3)).all()

    def test_2(self):
        data = np.array([-75, 73, 78, -73, 72, 71])
        arr = np.array([0.25681114, 0.12252817, 0.11950294,
                        0.25425582, 0.12314234, 0.1237596 ])
        res = util.calc_wham(data, kbt=-200)
        assert (res.round(decimals=3) == arr.round(decimals=3)).all()
