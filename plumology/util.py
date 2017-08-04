"""util - Utility functions"""

import functools
from inspect import signature
import os
from typing import (Any, Dict, Callable)

import pandas as pd


__all__ = ['last_nonzero', 'dict_to_dataframe', 'preserve_cwd', 'typecheck']


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
    nonzero = {}
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
