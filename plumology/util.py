'''util - Utilities and calculation functions'''

import functools
import glob
from inspect import signature
import itertools
import os
from os.path import join
import subprocess
import tempfile
from typing import (Any, Sequence, List, Tuple, Dict,
                    Callable, Union, Optional, Mapping)

import h5py
import numpy as np
import pandas as pd
from scipy.stats import entropy

from .io import read_nmr, read_rdc

__all__ = ['calc_entropy', 'calc_nmr', 'calc_rdc', 'calc_rmsd', 'stats',
           'calc_wham', 'chunk_range', 'dist1D', 'dist2D', 'free_energy',
           'clip', 'population']


def _preserve_cwd(func: Callable[[Any], Any]) -> Callable[[Any], Any]:
    '''
    This decorator preserves the current working
    directory through the function call.

    '''
    @functools.wraps(func)
    def decorator(*args, **kwargs):
        cwd = os.getcwd()
        try:
            return func(*args, **kwargs)
        finally:
            os.chdir(cwd)

    return decorator


def _typecheck(func: Callable[[Any], Any]) -> Callable[[Any], Any]:
    '''
    Checks type annotated function arguments.

    '''
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


def population(
        data: pd.DataFrame,
        minima: Sequence[Tuple[float, float]],
        radius: float=2.0,
        weight_name: Optional[str]=None,
        cv_names: Tuple[str, str]=('cv1', 'cv2')
) -> Dict[Tuple[float, float], float]:
    '''
    Calculate the population on a 2D free energy surface by summing up weights.

    Parameters
    ----------
    data : Dataframe including weights and two collective variables.
    minima : List of centers of minima.
    radius : Aggregation radius.
    weight_name : Name of the weight column.

    Returns
    -------
    population : Dictionary containing percentages mapped to minima.

    '''
    if any(cv not in data.columns for cv in cv_names):
        raise KeyError('cv_names not found in dataframe!')

    # Just count up if weight is not given
    if weight_name is None:
        weight_name = 'ww'
        data[weight_name] = 1

    pops = {}
    for coords in minima:
        p = sum(data[(data[cv_names[0]] > (coords[0] - radius)) &
                     (data[cv_names[0]] < (coords[0] + radius)) &
                     (data[cv_names[1]] > (coords[1] - radius)) &
                     (data[cv_names[1]] < (coords[1] + radius))][weight_name])
        pops[coords] = p
    return pops


def clip(
        data: pd.DataFrame,
        ranges: Mapping[str, Tuple[float, float]],
        ignore: Sequence[str]=None,
        weight_name: Optional[str]=None,
        renormalize: bool=True
) -> pd.DataFrame:
    '''
    Clip a dataset to a fixed range, discarding other datapoints.

    Parameters
    ----------
    data : Dataset to clip.
    ranges : Ranges to clip in.
    ignore : Columns to ignore.
    renormalize : Recalculate the weights if needed.

    Returns
    -------
    data : Clipped data

    '''
    if ignore is None:
        ignore = [weight_name]
    else:
        ignore.append(weight_name)

    for col in data.columns:
        if col in ignore:
            continue
        data = (data[(data[col] > ranges[col][0]) &
                     (data[col] < ranges[col][1])])

    if renormalize and weight_name is not None:
        data[weight_name] /= data[weight_name].sum()

    return data


def stats(fields: Sequence[str], data: np.ndarray) -> List[str]:
    '''
    Calculate statistical properties of dataset and format them nicely.

    Parameters
    ----------
    fields : List of field names.
    data : 2D numpy ndarray with the data contents.

    Returns
    -------
    stats : List of formatted strings with statistical information.

    '''
    if len(fields) != np.size(data, axis=1):
        raise ValueError('Fields and data must have identical sizes!')
    return [
        ('{:16} min = {:>10.4f} :: max = {:>10.4f} :: mean = {:>10.4f} '
         ':: stddev = {:>10.4f} :: var = {:>10.4f}').format(
             fi,
             min(data[:, i]),
             max(data[:, i]),
             np.mean(data[:, i]),
             np.std(data[:, i]),
             np.var(data[:, i])
             ) for i, fi in enumerate(fields)
    ]


def chunk_range(chunk_min: float,
                chunk_max: float,
                nchunks: int,
                first_chunk_size: Optional[float]=None) -> List[float]:
    '''
    Create chunk indices based on continuous data.

    Parameters
    ----------
    chunk_min : Minimum value of region.
    chunk_max : Maximum value of region.
    nchunks : Number of chunks to create.
    first_chunk_size : Size of the first chunk. If not given,
        will divide the data equally.

    Returns
    -------
    chunks : List of chunk areas.

    '''
    if chunk_max <= chunk_min:
        raise ValueError('chunk_max must be larger than chunk_min!')
    elif nchunks <= 0:
        raise ValueError('nchunks can not be smaller than 0!')

    if first_chunk_size is not None:
        chunk = first_chunk_size + chunk_min
        chunks = [chunk]
        nchunks -= 1
    else:
        chunks = []
        chunk = chunk_min

    chunkrange = (chunk_max - chunk) / nchunks

    for _ in range(nchunks):
        chunk += chunkrange
        chunks.append(chunk)

    return chunks


def last_nonzero(data: pd.DataFrame) -> pd.Series:
    '''
    Get the last non-zero elements from a dataframe.

    Parameters
    ----------
    data : Dataframe containing data.

    Returns
    -------
    nonzero : Series of last non-zero datapoints.

    '''
    nonzero = {}
    for col in data.columns:
        nonzero[col] = data[col][(data[col] > 0.000) |
                                 (data[col] < 0.000)].iloc[-1]
    return pd.Series(nonzero)


def dict_to_dataframe(
        data: Dict[str, pd.DataFrame],
        grouper: str='ff'
) -> pd.DataFrame:
    '''
    Convert a dictionary of dataframes to a multiindexed dataframe.

    Parameters
    ----------
    data : Dict of dataframes.
    grouper : Name of the new multiindex.

    Returns
    -------
    data : Multiindexed dataframe.

    '''
    dfs = []
    for key, df in data.items():
        df[grouper] = key
        dfs.append(df)

    return pd.concat(dfs).set_index(grouper, append=True).sort_index()


def calc_bse(
        data: pd.DataFrame,
        weight_name: Optional[str]=None,
        ignore: List[str]=None
) -> pd.DataFrame:
    '''
    Calculate the Block Standard Error (BSE).

    Parameters
    ----------
    data : Dataframe with CV data over time and weights.
    weight_name : Name of the weight column.
    ignore : List of column names to ignore.

    Returns
    -------
    bse : Dataframe containing BSEs over all iterations.

    References
    ----------
    Flyvbjerg, H., Petersen, H. G. Error estimates on averages of correlated
    data. The Journal of Chemical Physics, 91(1), 461 (1989)

    '''
    if ignore is None:
        ignore = []
    if 'time' not in ignore:
        ignore.append('time')

    # Prepare input, first element
    if weight_name is not None:
        weights = data[weight_name].values
        ignore.append(weight_name)

    length = data.shape[0]
    width = data.shape[1]
    index = data.T.index
    data = data.values
    blist = [data.std(axis=0) / np.sqrt(length)]
    length = length // 2

    # Iteratively increase block size
    while length > 2:
        halved = np.empty((length, width))

        # Each iteration, we halve the dataset
        for i in range(0, length):
            if weight_name is not None:
                halved[i] = (1 / (weights[2 * i - 1] + weights[2 * i]) *
                             (data[2 * i - 1] * weights[2 * i - 1] +
                              data[2 * i] * weights[2 * i]))
            else:
                halved[i] = 0.5 * (data[2 * i - 1] + data[2 * i])

        # Calculate the BSE
        bse = halved.std(axis=0) / np.sqrt(length)
        blist.append(bse)
        length = length // 2

    # Reconstruct Dataframe
    return pd.DataFrame(np.asarray(blist), columns=index).drop(ignore, axis=1)


def calc_wham(bias: Union[str, np.ndarray],
              kbt: float=2.49339) -> np.ndarray:
    '''
    Perform the weighted histogram analysis method on an array of biases.

    Parameters
    ----------
    bias : Filename or array containing the bias in the last column.
    kbt : k_b * T as calculated by plumed.

    Returns
    -------
    weights : The weights corresponding to timesteps.

    '''
    # Check and load input
    if isinstance(bias, str):
        data = np.loadtxt(bias, usecols=(-1,))
    else:
        data = bias

    # I guess these values are fine for most uses
    nwham, thres, z = 10000, 1e-30, 1.0
    weights = np.ones_like(data)
    expv = np.exp((data.min() - data) / kbt)

    # Iterate until convergence
    for _ in range(nwham):
        z_prev = z
        weights = z / expv
        norm = sum(weights)
        weights /= norm
        z = sum(weights * expv)
        eps = np.log(z / z_prev) ** 2
        if eps < thres:
            break
    return weights


def calc_entropy(data: pd.DataFrame,
                 keys: Sequence[str],
                 kind: str='kl') -> pd.DataFrame:
    '''
    Compute the divergence of two probability distributions.

    Parameters
    ----------
    data : Probability distributions.
    keys : Two keys indicating the distributions to be compared.
    kind : Type of metric to use.

    Returns
    -------
    kld : Dataframe of length 1 with the calculated values.

    '''
    # Check input
    valid_kinds = ['kl', 'kls', 'shannon', 'js', 'hellinger']
    if kind not in valid_kinds:
        e = ('\'{0}\' is not a valid entropy kind!'
             'Valid types: {1}').format(kind, valid_kinds)
        raise ValueError(e)

    if isinstance(keys, list):
        keylist = keys
    else:
        keylist = [keys]

    kls = []
    for k1, k2 in keylist:
        data_a = data.xs(k1)
        data_b = data.xs(k2)

        kl = []
        for col in data_a.columns:
            if kind == 'kls':
                S = (entropy(data_a[col], data_b[col]) +
                     entropy(data_b[col], data_a[col]))
            elif kind == 'kl':
                S = entropy(data_a[col], data_b[col])
            elif kind == 'shannon':
                M = 0.5 * (data_a[col] + data_b[col])
                S = 0.5 * (entropy(data_a[col], M) + entropy(data_b[col], M))
            elif kind == 'hellinger':
                S = (1 / np.sqrt(2) *
                     np.sqrt(((np.sqrt(data_a[col]) -
                               np.sqrt(data_b[col])) ** 2).sum()))
            kl.append(pd.Series(
                {'{0}-{1}'.format(k1, k2): S},
                name=col
            ))
        kls.append(pd.concat(kl, axis=1))
    return pd.concat(kls)


def dist1D(
        data: Union[pd.DataFrame, h5py.Group],
        ret: str='both',
        nbins: int=50,
        weight_name: Optional[str]=None,
        ignore: List[str]=None,
        normed: bool=False
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    '''
    Create a 1D weighted probability distribution.

    Parameters
    ----------
    data : Dataframe with CV data over time and weights.
    nbins : Number of bins to use for the histogram.
    weight_name : Name of the weight column.
    ignore : List of column names to ignore.
    normed : Normalize the histograms.

    Returns
    -------
    dist1D : Probability distribution as Dataframe.

    '''
    if ignore is None:
        ignore = ['time']

    # Check return option
    if not any(ret.startswith(s) for s in ['b', 'd', 'r', 'e']):
        raise ValueError(
            'Invalid return type! Valid types: both, dist, (edges/ranges)'
        )

    # Get possible keys (HDF or dataframe)
    if isinstance(data, h5py.Group):
        cols = data.keys()
    else:
        cols = data.columns

    edges, dists = [], []

    # Iterate over CVs
    for col in cols:

        # We copy into new arrays because numpy
        # operations on h5py datasets are very slow
        data_col = data[col][::]
        if weight_name is not None:
            data_ww = data[weight_name][::]

        # Skip non-CV columns
        if col in ignore or col == weight_name:
            continue

        if weight_name is not None:
            dist, _ = np.histogram(data_col, weights=data_ww,
                                   bins=nbins, density=normed)
        else:
            dist, _ = np.histogram(data_col, bins=nbins, density=normed)

        # edges created by scipy are not usable
        dist_range = np.linspace(data_col.min(), data_col.max(), nbins)
        dists.append(pd.Series(dist, name=col))
        edges.append(pd.Series(dist_range, name=col))

    if ret.startswith('b'):
        return pd.concat(dists, axis=1), pd.concat(edges, axis=1)
    elif ret.startswith('d'):
        return pd.concat(dists, axis=1)
    else:
        return pd.concat(edges, axis=1)


def dist2D(data: Union[pd.DataFrame, h5py.Group],
           cvs: Union[Tuple[str, str], List[Tuple[str, str]], None]=None,
           nbins: int=50,
           weight_name: Optional[str]=None,
           ignore: List[str]=None,
           normed: bool=False) -> pd.DataFrame:
    '''
    Create a 2D weighted probability distribution.

    Parameters
    ----------
    data : Dataframe with CV data over time and weights.
    cvs : If given, will only plot these CVs.
    nbins : Number of bins to use for the histogram.
    weight_name : Name of the weight column.
    ignore : List of column names to ignore.

    Returns
    -------
    dist2D : Probability distribution as Dataframe.

    '''

    if ignore is None:
        ignore = ['time']

    # Get possible keys (HDF or dataframe)
    if isinstance(data, h5py.Group):
        cols = data.keys()
    else:
        cols = data.columns

    # Prepare the possible combinations of CVs
    if not cvs:
        combs = list(itertools.combinations(
            (c for c in cols if c not in ignore + [weight_name]), 2
        ))
    elif isinstance(cvs, list):
        combs = cvs
    else:
        combs = [cvs]

    # Create 2D histograms and stack them to form a panel
    dists = []
    for cv1, cv2 in combs:
        if weight_name is not None:
            H, _, _ = np.histogram2d(
                data[cv1][::],
                data[cv2][::],
                bins=nbins,
                normed=normed,
                weights=data[weight_name][::]
            )
        else:
            H, _, _ = np.histogram2d(
                data[cv1][::],
                data[cv2][::],
                bins=nbins,
                normed=normed
            )

        dists.append(pd.DataFrame(H).stack())

    return (pd.concat(dists, axis=1)
            .rename(columns={
                i: '{0}.{1}'.format(cv1, cv2)
                for i, (cv1, cv2) in enumerate(combs)
            }))


def free_energy(dist: pd.DataFrame, kbt: float) -> pd.DataFrame:
    '''
    Compute the free energy from a probability distribution.

    Parameters
    ----------
    dist : Probability distribution.
    kbt : k_b * T as calculated by plumed.

    Returns
    -------
    free_energy : Free energy of probability distribution as a Dataframe.

    '''
    return dist.applymap(lambda p: -kbt * np.log(p)
                         if p != 0 else float('inf'))


def calc_sqdev(data: pd.DataFrame) -> pd.DataFrame:
    '''
    Calculate the Squared-Deviation per residue.

    Parameters
    ----------
    data : Multiindexed dataframe with force field and residue number

    Returns
    -------
    sqdev : Force field indexed dataframe

    '''
    cols = [c for c in data.columns if 'exp' not in c]

    # Separate dataframe into simulation data and experimental data
    ef = data.drop(cols, axis=1).rename(columns={'exp_' + c: c for c in cols})
    af = data.drop([c for c in data.columns if 'exp' in c], axis=1)

    # return sqdev dataframe
    return (af - ef) ** 2


def calc_rmsd(data: pd.DataFrame,
              grouper: Optional[str]='ff') -> pd.DataFrame:
    '''
    Calculate the Root-Mean-Squared-Deviation.

    Parameters
    ----------
    data : Multiindexed dataframe with force field and residue number

    Returns
    -------
    rmsd : Force field indexed dataframe

    '''
    return (calc_sqdev(data).groupby(level=[grouper])
            .agg(np.mean)
            .apply(np.sqrt))


def calc_rdc(executable: str,
             weight_file: str,
             trajectory_file: str,
             pdb_file: str,
             exp_files: List[str]) -> Dict[str, float]:
    '''
    Calculate RDCs using Tensor alignment.

    Parameters
    ----------
    executable : Path to RDC program.
    weight_file : Path to weight file.
    trajectory_file : Path to full trajectory.
    pdb_file : Path to .pdb file for the system.
    exp_files : Paths to files containing exp data.

    Returns
    -------
    rdcs : Dictionary containing the RDC name
        as keys and the calculated value.

    '''

    with tempfile.TemporaryDirectory() as tmp:
        output = [join(tmp, 'fit_{0}'.format(i))
                  for i, _ in enumerate(exp_files)]
        for i, exp_file in enumerate(exp_files):
            command = '{0} -w {1} -d {2} -o {3} -s {4} {5}'.format(
                executable,
                weight_file,
                exp_file,
                output[i],
                pdb_file,
                trajectory_file
            ).split()
            subprocess.check_output(command)
        rdc = read_rdc(output)
    return rdc


@_preserve_cwd
def calc_nmr(executable: str,
             weights: Union[str, np.ndarray],
             trajectory_file: str,
             runtime_file: str,
             nres: int,
             skip: int=50) -> Dict[str, float]:
    '''
    Calculate chemical shifts using an external program.

    Parameters
    ----------
    executable : Path to CS predictor program.
    weights : Path to weight file or array of weights.
    trajectory_file : Path to full trajectory.
    runtime_file : Path to GROMACS .tpr runtime file.
    skip : Number of frames to skip when extracting PDB frames.
    nres : Number of residues of the system.

    Returns
    -------
    cs : Dictionary containing the chemical shift names
        as keys and the calculated values.

    '''

    # Check and read input
    if isinstance(weights, str):
        weight = pd.read_csv(
            weights, sep=r'\s+', header=None, comment='#',
            names=['ww'], dtype=np.float64, skiprows=1
        ).values[::skip]
    else:
        weight = weights[::skip]

    # Renormalize weights
    wn = (weight / sum(weight)).flatten()

    # Use a temp directory to store pdb and prediction files
    with tempfile.TemporaryDirectory() as tmp:

        # Create pdb snapshots
        output = join(tmp, 'traj.pdb')
        trj_cmd = 'gmx_mpi trjconv -f {0} -s {1} -o {2} -skip {3} -sep'.format(
            trajectory_file,
            runtime_file,
            output,
            skip
        ).split()
        subprocess.run(trj_cmd, input=b'1\n', check=True)

        # Use sparta to predict CS
        os.chdir(tmp)
        calc_command = '{0} -in {1}'.format(
            executable, join(tmp, 'traj*')
        ).split()
        subprocess.run(calc_command, check=True)

        # Parse data
        nfiles = len(glob.glob1(tmp, '*pred.tab'))
        nmr = read_nmr(wn, tmp, nfiles, nres)

    return nmr
