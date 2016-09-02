'''rw - IO tools for PLUMED files and related data'''

from collections import OrderedDict
import glob
import itertools
import re
from typing import (Any, Sequence, List, Tuple, Iterator, Mapping,
                    Dict, Union, Optional)

import sys
import numpy as np
import pandas as pd

__all__ = ['is_plumed', 'is_same_shape', 'read_plumed_fields',
           'read_rdc', 'read_nmr', 'read_plumed', 'read_multi',
           'read_all_hills', 'plumed_iterator', 'file_length',
           'field_glob', 'fields_to_columns']


def is_plumed(file: str) -> bool:
    '''
    Checks if the file is of plumed format.

    Parameters
    ----------
    file : Path to plumed file.

    Returns
    -------
    is_plumed : Returns true if file is valid plumed format,
        raises ValueError otherwise.

    '''
    with open(file, 'r') as f:
        head = f.readlines(0)[0]
        if head.startswith('#!'):
            return True
        else:
            raise ValueError('Not a valid plumed file')


def is_same_shape(data: List[np.ndarray]) -> bool:
    '''
    Checks if a list of ndarrays all have the same shape.

    Parameters
    ----------
    data : List of arrays

    Returns
    -------
    is_same_shape : True if same shape, False if not

    '''
    return len(set(d.shape for d in data)) == 1


def read_plumed_fields(file: str) -> List[str]:
    '''
    Reads the fields specified in the plumed file.

    Parameters
    ----------
    file : Path to plumed file.

    Returns
    -------
    fields : List of field names.

    '''
    is_plumed(file)
    with open(file, 'br') as f:
        head = f.readlines(0)[0].split()[2:]
        fields = [x.decode('utf-8') for x in head]
    return fields


def plumed_iterator(file: str) -> Iterator[List[float]]:
    '''
    Creates an iterator over a plumed file.

    Parameters
    ----------
    file : Path to plumed file.

    Yields
    ------
    iter : List of floats for each line read.

    '''
    is_plumed(file)
    with open(file, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            yield [float(n) for n in line.split()]


def file_length(file: str, skip_comments: bool=False) -> int:
    '''
    Counts number of lines in file.

    Parameters
    ----------
    file : Path to file.
    skip_comments : Skipping comments is slightly slower,
        because we have to check each line.

    Returns
    -------
    length : Length of the file.

    '''
    with open(file, 'r') as f:
        i = -1
        if skip_comments:
            for line in f:
                if line.startswith('#'):
                    continue
                i += 1
        else:
            for i, _ in enumerate(f):
                pass
    return i + 1


def read_plumed(
    file: str,
    columns: Union[Sequence[int], Sequence[str], str, None]=None,
    step: int=1,
    start: int=0,
    stop: int=sys.maxsize,
    high_mem: bool=True,
    dataframe: bool=True,
    raise_error: bool=False,
    drop_nan: bool=True
) -> Union[pd.DataFrame, Tuple[List[str], np.ndarray]]:
    '''
    Read a plumed file and return its contents as a 2D ndarray.

    Parameters
    ----------
    file : Path to plumed file.
    columns : Column numbers or field names to read from file.
    step : Stepsize to use. Will skip every [step] rows.
    start : Starting point in lines from beginning of file,
        including commented lines.
    stop : Stopping point in lines from beginning of file,
        including commented lines.
    high_mem : Use high memory version, which might be faster.
        Reads in the whole array and then slices it.
    dataframe : Return as pandas dataframe.
    raise_error : Raise error in case of length mismatch.
    drop_nan : Drop missing values.

    Returns
    -------
    fields : List of field names.
    data : 2D numpy ndarray with the contents from file.

    '''
    is_plumed(file)
    length = file_length(file)
    if stop != sys.maxsize and length < stop and raise_error:
        raise ValueError('Value for [stop] is larger than number of lines')

    full_fields = read_plumed_fields(file)
    if columns is not None and not isinstance(columns[0], int):
        columns = field_glob(columns, full_fields)
    fields, columns = fields_to_columns(columns, full_fields)

    full_array = (step == 1 and start == 0 and
                  stop == sys.maxsize and columns is None)

    if full_array or high_mem:
        nrows = stop - start if stop != sys.maxsize else None
        df = pd.read_csv(
            file,
            sep='\s+',
            header=None,
            comment='#',
            names=full_fields,
            dtype=np.float64,
            skiprows=start,
            nrows=nrows,
            usecols=columns
        )
        data = df[::step]
        if drop_nan:
            data.dropna(axis=0)

        if not dataframe:
            data = data.values

    else:
        with open(file, 'br') as f:
            data = np.genfromtxt(itertools.islice(f, start, stop, step),
                                 skip_header=1, invalid_raise=False,
                                 usecols=columns)
        if dataframe:
            data = pd.DataFrame(OrderedDict(zip(fields, data.T)))

    if not dataframe:
        return fields, data
    else:
        return data


def read_multi(
    files: Union[Sequence[str], str],
    **kwargs: Optional[Mapping[str, Any]]
) -> pd.DataFrame:
    '''
    Read multiple Plumed files and return as concatenated dataframe.

    Parameters
    ----------
    files : Sequence of (globbed) files to be read.
    kwargs : Arguments passed to read_plumed().

    Returns
    -------
    df : Dataframe with concatenated data.

    '''

    if isinstance(files, str):
        files = [files]

    # Sanity check
    kwargs.pop('dataframe', {'': ''})

    filelist = []  # type: List[str]
    for file in files:
        if any(char in file for char in '*?[]'):
            filelist.extend(glob.iglob(file))
        else:
            filelist.append(file)

    dflist = []  # type: List[str]
    for i, file in enumerate(filelist):
        df = read_plumed(file, **kwargs)
        dflist.append(df.rename(columns={
            k: '{0}_{1}'.format(i, k) for k in df.columns if 'time' not in k
        }))

    return pd.concat(dflist, axis=1)


def field_glob(
    fields: Union[str, Sequence[str]],
    full_fields: Sequence[str]
) -> List[str]:
    '''
    Gets a list of matching fields from valid regular expressions.

    Parameters
    ----------
    fields : Regular expression(s) to be used to find matches.
    full_fields : Full list of fields to match from.

    Returns
    -------
    matches : List of matching fields.

    '''

    if isinstance(fields, str):
        fields = [fields]

    globbed = set()
    for field in fields:
        if field in full_fields:
            globbed.add(field)

        for f_target in full_fields:
            if re.search(field, f_target):
                globbed.add(f_target)

    return list(globbed)


def fields_to_columns(
    fields: Union[Sequence[str], Sequence[int]],
    full_fields: Sequence[str]
) -> Tuple[Tuple[str, ...], Tuple[int, ...]]:
    '''
    Transforms a sequence of field names to their respective column indices.

    Parameters
    ----------
    fields : Sequence of field names or columns numbers
    full_fields : The full list of field names as
        created by read_plumed_fields().

    Returns
    -------
    columns : Column indices, None if fields is none.

    '''
    if fields is None:
        return tuple(full_fields), None

    elif type(fields[0]) == int:
        return tuple(full_fields[i] for i in fields), tuple(fields)

    else:
        return tuple(fields), tuple(full_fields.index(s)
                                    for s in fields if s in full_fields)


def read_all_hills(files: Sequence[str],
                   step: int=1) -> pd.DataFrame:
    '''
    Read CV information from HILLS files.

    Parameters
    ----------
    files : List of filenames to read from.
    step : Stepsize to use while reading.
    Returns
    -------
    timedata : Dataframe with time as first column and CV data for the rest.

    '''
    length = file_length(files[0]) - 1
    timedata = read_plumed(files[0], step=step, stop=length,
                           columns=(0,), dataframe=True)
    for file in files:
        data = read_plumed(file, step=step, stop=length,
                           columns=(1,), dataframe=True)
        timedata = pd.concat([timedata, data], axis=1)
    return timedata.dropna(axis=0)


def read_rdc(files: Sequence[str]) -> Dict[str, float]:
    '''
    Reads files generated by an RDC program.

    Parameters
    ----------
    files : Paths to files to be read.

    Returns
    -------
    rdcs : Dictionary containing the RDC name
        as keys and the calculated value.

    '''
    rdcs = {}

    # Open dat files and parse
    for file in files:
        with open(file, 'r') as f:
            for line in f.readlines():

                if line.startswith('#'):
                    continue
                data = line.split()
                res_nr, rdc_type, val = int(data[0]), data[3], float(data[4])

                # Translate to our key style
                if rdc_type == 'HN':
                    rdc_type = 'nh'
                elif rdc_type == 'C':
                    rdc_type = 'cac'
                elif rdc_type == 'HA':
                    rdc_type = 'caha'
                else:
                    raise ValueError('Invalid RDC type!')

                rdcs['{0}.rdc{1}'.format(rdc_type, res_nr)] = val
    return rdcs


def read_nmr(
        weights: np.ndarray,
        directory: str,
        nfiles: int,
        skip: int,
        nres: int
) -> Dict[str, float]:
    '''
    Read chemical shifts from an external program.

    Parameters
    ----------
    weights : Array of weights.
    directory : Directory conatining the pred.tab files.
    nfiles : Number of pred.tab files.
    skip : Number of frames to skip when extracting PDB frames.
    nres : Number of residues of the system.

    Returns
    -------
    cs : Dictionary containing the chemical shift names
        as keys and the calculated values.

    '''

    # Prepare input files
    spfiles = [directory + '/traj{0}_pred.tab'.format(i)
               for i in range(nfiles)]
    posatoms = ['nh', 'hn', 'ha', 'ca', 'cb', 'co']

    # Check input
    if len(spfiles) != len(weights):
        raise ValueError(
                'Number of SPARTA+ files must be the same '
                'as the length of the weight array!'
        )

    # Iterate over files
    shifts = {}  # type: Dict[str, float]
    for i, file in enumerate(spfiles):
        with open(file, 'r') as sf:

            # Only use lines starting with digits
            for line in sf.readlines():

                if re.match(r'^\s*\d+', line):
                    tmpline = re.split(r'\s+', line)
                    residue = int(tmpline[1])
                    atoms = tmpline[3].lower()
                    shift = float(tmpline[5])

                    # Skip first and last residues
                    if residue == nres or residue == 1:
                        continue

                    # Use proper Camshift naming convention
                    if atoms == 'n':
                        atoms = 'nh'
                    elif atoms == 'c':
                        atoms = 'co'
                    elif atoms not in posatoms:
                        continue

                    # Add weighted shift to dict
                    key = 'cs.{0}_{1}'.format(atoms, residue - 1)
                    if key in shifts:
                        shifts[key] += shift * weights[i]
                    else:
                        shifts[key] = shift * weights[i]
    return shifts
