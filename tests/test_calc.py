import pytest

import numpy as np
import pandas as pd
from plumology import calc, util

arr = [[1.0, 10.5, 0.2], [-0.3, 0.5, 0.4], [-0.2, 0.5, 0.4]]
data = pd.DataFrame(data=arr, columns=['a', 'b', 'c'])
expdata = data.rename(columns={'c': 'exp_a'}).drop('b', axis=1)


class Test_population:
    def test_1(self):
        with pytest.raises(KeyError):
            calc.population(data.copy(), [(0.0, 0.5)])

    def test_2(self):
        res = calc.population(data.copy(), [(0.0, 0.5)], cv_names=('a', 'b'))
        assert res == {(0.0, 0.5): 2}

    def test_3(self):
        res = calc.population(
            data.copy(), [(0.0, 0.5)], cv_names=('a', 'b'), weight_name='c'
        )
        assert res == {(0.0, 0.5): 0.80000000000000004}

    def test_4(self):
        res = calc.population(
            data.copy(), [(0.0, 0.5)], radius=100,
            cv_names=('a', 'b'), weight_name='c'
        )
        assert res == {(0.0, 0.5): 1.0}


class Test_clip:
    def test_1(self):
        res = calc.clip(data.copy(),
                        {'a': (-5., 5.), 'b': (-5., 5.)},
                        ignore=['c'])
        assert (res == data.loc[1:]).all().all()

    def test_2(self):
        arr = [[-0.3, 0.5, 0.5], [-0.2, 0.5, 0.5]]
        res = calc.clip(data.copy(),
                        {'a': (-5., 5.), 'b': (-5., 5.)},
                        weight_name='c')
        assert (arr == res.values).all()

    def test_3(self):
        arr = np.array([[1.0, 10.5, 0.2],
                        [-0.3, 0.5, 0.4],
                        [-0.2, 0.5, 0.4]])
        res = calc.clip(data.copy(),
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
        assert calc.stats(fields, arr) == res

    def test_2(self):
        arr = np.ones((32, 4))
        fields = ['1', '2', '3']
        with pytest.raises(ValueError):
            calc.stats(fields, arr)


class Test_chunk_range:
    def test_1(self):
        assert calc.chunk_range(0, 10000, 3, 1000) == [1000, 5500.0, 10000.0]

    def test_2(self):
        assert calc.chunk_range(5000, 9000, 3, 1000) == [6000, 7500.0, 9000.0]

    def test_3(self):
        assert calc.chunk_range(-1, 9, 3, 2) == [1, 5.0, 9.0]

    def test_4(self):
        with pytest.raises(ValueError):
            calc.chunk_range(-1, -10, -3, 2.5)

    def test_5(self):
        assert calc.chunk_range(-1, 8, 3) == [2.0, 5.0, 8.0]

    def test_6(self):
        with pytest.raises(TypeError):
            calc.chunk_range(-1, 10, 3.14159, 2.5)




class Test_bse:
    def test_1(self):
        ndata = data.copy()
        ndata['time'] = [0, 1, 2]
        arr = np.array([[0.3410224, 2.7216552, 0.0544331]]).round(decimals=3)
        assert (calc.bse(ndata).values.round(decimals=3) == arr).all()

    def test_2(self):
        ndata = data.copy()
        ndata['time'] = [0, 1, 2]
        arr = np.array([[0.3410224, 2.7216552]])
        res = calc.bse(ndata, weight_name='c').values
        assert (res.round(decimals=3) == arr.round(decimals=3)).all()

    def test_3(self):
        ndata = data.copy()
        ndata['time'] = [0, 1, 2]
        arr = np.array([[0.3410224]])
        res = calc.bse(ndata, ignore=['b'], weight_name='c').values
        assert (res.round(decimals=3) == arr.round(decimals=3)).all()


class Test_wham:
    def test_1(self):
        data = np.array([75, 73, 78, 73, 72, 71])
        arr = np.array([0.17445063, 0.07813428, 0.58199452,
                        0.07813428, 0.05229091, 0.03499538])
        res = calc.wham(data, kbt=2.49)
        assert (res.round(decimals=3) == arr.round(decimals=3)).all()

    def test_2(self):
        data = np.array([-75, 73, 78, -73, 72, 71])
        arr = np.array([0.25681114, 0.12252817, 0.11950294,
                        0.25425582, 0.12314234, 0.1237596])
        res = calc.wham(data, kbt=-200)
        assert (res.round(decimals=3) == arr.round(decimals=3)).all()


class Test_dist1d:
    def test_1(self):
        d = np.array([[2, 2, 1],
                      [0, 0, 0],
                      [0, 0, 0],
                      [0, 0, 0],
                      [1, 1, 2]])
        res = calc.dist1d(data, nbins=5)
        assert (res[0] == d).all().all()

    def test_2(self):
        r = np.array([[ -0.3  ,   0.5  ,   0.2  ],
                      [  0.025,   3.   ,   0.25 ],
                      [  0.35 ,   5.5  ,   0.3  ],
                      [  0.675,   8.   ,   0.35 ],
                      [  1.   ,  10.5  ,   0.4  ]])
        res = calc.dist1d(data, nbins=5)
        assert (res[1].round(decimals=3) == r).all().all()

    def test_3(self):
        d = np.array([[2, 1],
                      [0, 0],
                      [0, 0],
                      [0, 0],
                      [1, 2]])
        res = calc.dist1d(data, nbins=5, ignore='a')
        assert (res[0].round(decimals=3) == d).all().all()

    def test_4(self):
        r = np.array([[  0.5 ,   0.2 ],
                      [  3.  ,   0.25],
                      [  5.5 ,   0.3 ],
                      [  8.  ,   0.35],
                      [ 10.5 ,   0.4 ]])
        res = calc.dist1d(data, nbins=5, ignore='a')
        assert (res[1].round(decimals=3) == r).all().all()

    def test_5(self):
        d = np.array([[ 0.8],
                      [ 0. ],
                      [ 0. ],
                      [ 0. ],
                      [ 0.2]])
        res = calc.dist1d(data, nbins=5, ignore='a', weight_name='c')
        assert (res[0].round(decimals=3) == d).all().all()

    def test_6(self):
        r = np.array([[  0.5],
                      [  3. ],
                      [  5.5],
                      [  8. ],
                      [ 10.5]])
        res = calc.dist1d(data, nbins=5, ignore='a', weight_name='c')
        assert (res[1].round(decimals=3) == r).all().all()

    def test_7(self):
        d = np.array([[ 0.4],
                      [ 0. ],
                      [ 0. ],
                      [ 0. ],
                      [ 0.1]])
        res = calc.dist1d(data, ret='dists', nbins=5, ignore='a',
                          weight_name='c', normed=True)
        assert (res.round(decimals=3) == d).all().all()


class Test_dist2d:
    def test_1(self):
        d = np.array([[ 2.,  0.,  0.],
                      [ 0.,  0.,  0.],
                      [ 0.,  2.,  2.],
                      [ 0.,  0.,  0.],
                      [ 0.,  0.,  0.],
                      [ 0.,  0.,  0.],
                      [ 0.,  1.,  1.],
                      [ 0.,  0.,  0.],
                      [ 1.,  0.,  0.]])
        res = calc.dist2d(data, nbins=3)
        assert (res.round(decimals=3) == d).all().all()

    def test_2(self):
        d = np.array([[ 2.],
                      [ 0.],
                      [ 0.],
                      [ 0.],
                      [ 0.],
                      [ 0.],
                      [ 0.],
                      [ 0.],
                      [ 1.]])
        res = calc.dist2d(data, nbins=3, cvs=[('a', 'b')])
        assert (res.round(decimals=3) == d).all().all()

    def test_3(self):
        d = np.array([[ 0.8],
                      [ 0. ],
                      [ 0. ],
                      [ 0. ],
                      [ 0. ],
                      [ 0. ],
                      [ 0. ],
                      [ 0. ],
                      [ 0.2]])
        res = calc.dist2d(data, nbins=3, cvs=[('a', 'b')], weight_name='c')
        assert (res.round(decimals=3) == d).all().all()

    def test_4(self):
        d = np.array([[ 0.55384615],
                      [ 0.        ],
                      [ 0.        ],
                      [ 0.        ],
                      [ 0.        ],
                      [ 0.        ],
                      [ 0.        ],
                      [ 0.        ],
                      [ 0.13846154]])
        res = calc.dist2d(data, nbins=3, weight_name='c', normed=True)
        assert (res.round(decimals=3) == d.round(decimals=3)).all().all()

    def test_5(self):
        d = np.array([[ 0.46153846],
                      [ 0.        ],
                      [ 0.        ],
                      [ 0.        ],
                      [ 0.        ],
                      [ 0.        ],
                      [ 0.        ],
                      [ 0.        ],
                      [ 0.23076923]])
        res = calc.dist2d(data, nbins=3, ignore=['c'], normed=True)
        assert (res.round(decimals=3) == d.round(decimals=3)).all().all()


class Test_free_energy:
    def test_1(self):
        d = np.array([[-0.   , -5.855,  4.008],
                      [ 2.998,  1.726,  2.282],
                      [ 4.008,  1.726,  2.282]])
        res = calc.free_energy(data.abs(), 2.49)
        assert (res.round(decimals=3) == d.round(decimals=3)).all().all()


class Test_sqdev:
    def test_1(self):
        d = np.array([[ 0.64],
                      [ 0.49],
                      [ 0.36]])
        res = calc.sqdev(data.rename(columns={'c': 'exp_a'}).drop('b', axis=1))
        assert (res.round(decimals=3) == d.round(decimals=3)).all().all()


class Test_rmsd:
    def test_1(self):
        d = np.array([[ 1.26491106],
                      [ 1.10679718],
                      [ 0.9486833 ]])
        edata = (util.dict_to_dataframe({1: expdata, 2: expdata * 2},
                                        grouper='res_nr')
                 .reset_index().set_index(keys=['level_0', 'res_nr']))
        res = calc.rmsd(edata, grouper='level_0')
        assert (res.round(decimals=3) == d.round(decimals=3)).all().all()