"""util - Utility functions"""

from collections import OrderedDict
import itertools
import functools
from inspect import signature
import os
from typing import (Any, Dict, Callable, Tuple)

import numpy as np
import pandas as pd


__all__ = ['last_nonzero', 'dict_to_dataframe', 'preserve_cwd', 'typecheck',
           'hypercube', 'make_blobs', 'py2c', 'aligned', 'array_to_pointer',
           'extend_array']


def preserve_cwd(func: Callable[[Any], Any]) -> Callable[[Any], Any]:
    """
    This decorator preserves the current working
    directory through the function call.

    """
    @functools.wraps(func)
    def decorator(*args, **kwargs):
        cwd = os.getcwd()
        try:
            return func(*args, **kwargs)
        finally:
            os.chdir(cwd)

    return decorator


def typecheck(func: Callable[[Any], Any]) -> Callable[[Any], Any]:
    """
    Checks type annotated function arguments.

    """
    @functools.wraps(func)
    def decorator(*args, **kwargs):

        if hasattr(func, '__annotations__'):
            hints = func.__annotations__
            sig = signature(func)
            bound_values = sig.bind(*args, **kwargs)
            for name, value in bound_values.arguments.items():
                if name in hints and not isinstance(value, hints[name]):
                    raise TypeError(
                        'Type mismatch: {0} != {1}'.format(name, hints[name])
                    )

        return func(*args, **kwargs)

    return decorator


def last_nonzero(data: pd.DataFrame) -> pd.Series:
    """
    Get the last non-zero elements from a dataframe.

    Parameters
    ----------
    data : Dataframe containing data.

    Returns
    -------
    nonzero : Series of last non-zero datapoints.

    """
    nonzero = OrderedDict()
    for col in data.columns:
        vals = data[col][(data[col] > 0.000) | (data[col] < 0.000)]
        nonzero[col] = vals.iloc[-1] if vals.size > 0 else 0.0
    return pd.Series(nonzero)


def dict_to_dataframe(
        data: Dict[str, pd.DataFrame],
        grouper: str='ff'
) -> pd.DataFrame:
    """
    Convert a dictionary of dataframes to a multiindexed dataframe.

    Parameters
    ----------
    data : Dict of dataframes.
    grouper : Name of the new multiindex.

    Returns
    -------
    data : Multiindexed dataframe.

    """
    dfs = []
    for key, df in data.items():
        df[grouper] = key
        dfs.append(df)

    return pd.concat(dfs).set_index(grouper, append=True).sort_index()


def extend_array(arr: np.ndarray, modulo: int) -> np.ndarray:
    """
    Extend a 2D array with zeros along axis 1, so that it's width is divisible
    by modulo.

    Parameters
    ----------
    arr : Numpy array to be extended.
    modulo : Divisor, so that arr.shape[1] % modulo == 0

    Returns
    -------
    data : extended numpy array

    """
    if arr.shape[1] % modulo == 0:
        return arr
    else:
        return np.column_stack(
            (arr, np.zeros((arr.shape[0], modulo - arr.shape[1] % modulo),
                           dtype=arr.dtype))
        )


def hypercube(n: int) -> np.ndarray:
    """
    Create hypercube coordinates.

    Parameters
    ----------
    n : Dimensionality

    Returns
    -------
    arr : 2D numpy array with all cube vertices.

    """
    return np.asarray(list(itertools.product((0, 1), repeat=n)))


def make_blobs(dim: int, npoints: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create blobs of points in higher dimensions on corners of a hypercube.
    Good for testing clustering algorithms.

    Parameters
    ----------
    dim : Dimensionality of the dataset
    npoints : Number of points to generate

    Returns
    -------
    highd : 2D array containing the positions of all points
    clusters : 1D array containing the cluster identity

    """
    hc = hypercube(dim)
    highd = np.empty((npoints, dim))
    clusters = np.empty((npoints,), dtype=int)
    for i in range(npoints):
        index = np.random.randint(0, hc.shape[0])
        highd[i] = hc[index] + np.random.randn(dim) / 10
        clusters[i] = index
    return highd, clusters


def py2c(*args, dtype=np.float64) -> Tuple[np.ndarray, ...]:
    """
    Make ndarrays C contiguous and ensure a certain datatype.

    Parameters
    ----------
    args : Any number of numpy ndarrays of any shape and datatype
    dtype : Numpy-compatible datatype of the array

    Returns
    -------
    arr : C-contiguous array

    """
    return tuple([np.ascontiguousarray(arg, dtype=dtype) for arg in args])


def array_to_pointer(arr: np.ndarray) -> np.ndarray:
    """
    Convert 2D numpy arrays to array of pointers.

    Parameters
    ----------
    arr : Array to be converted, should be C-contiguous

    Returns
    -------
    pointer : Array of pointers

    """
    return (arr.__array_interface__['data'][0] +
            np.arange(arr.shape[0]) * arr.strides[0]).astype(np.uintp)


def aligned(arr: np.ndarray, alignment: int=32) -> np.ndarray:
    """
    Align numpy array to a specific boundary.

    Parameters
    ----------
    arr : Numpy array
    alignment : Alignment in bytes

    Returns
    -------
    aligned : Aligned array

    """

    # Already aligned
    if (arr.ctypes.data % alignment) == 0:
        return arr

    extra = alignment / arr.itemsize
    buffer = np.empty(arr.size + extra, dtype=arr.dtype)
    offset = (-buffer.ctypes.data % alignment) / arr.itemsize
    newarr = buffer[offset:offset + arr.size].reshape(arr.shape)
    np.copyto(newarr, arr)
    assert (newarr.ctypes.data % alignment) == 0
    return newarr
