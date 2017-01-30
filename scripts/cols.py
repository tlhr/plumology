#!/usr/bin/env python3

import argparse
from typing import List


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Show PLUMED file columns.')
    parser.add_argument('filename', type=str, help='PLUMED file')
    args = parser.parse_args()
    fields = read_plumed_fields(args.filename)
    for i, field in enumerate(fields):
        print('{0}: {1}'.format(i + 1, field))
