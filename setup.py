#!/usr/bin/env python3

try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup


DESCRIPTION = 'Plumology - Biased molecular dynamics analysis'
LONG_DESCRIPTION = '''
Plumology allows the analysis of biased molecular dynamics data,
in particular output from PLUMED. It features:
    - Reading plumed output files
    - Creating weighted probability distributions, WHAM
    - An implementation of a self-organising-map
    - Efficient storage and retrieval of data via HDF5
    - Wrappers for common backcalculation of experimental values
    - Convergence analysis for Metadynamics
    - Comparing data from different simulations
    - Calculating experimental data RMSDs
    - A diverse set of flexible plotting routines
'''
DISTNAME = 'plumology'
MAINTAINER = 'Thomas Löhr'
MAINTAINER_EMAIL = 'thomas.loehr@tum.de'
DOWNLOAD_URL = 'https://github.com/tlhr/plumology'
URL = 'https://github.com/tlhr/plumology'
LICENSE = 'MIT'
VERSION = '0.0.1'


setup(
    name=DISTNAME,
    author=MAINTAINER,
    author_email=MAINTAINER_EMAIL,
    maintainer=MAINTAINER,
    maintainer_email=MAINTAINER_EMAIL,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    license=LICENSE,
    url=URL,
    version=VERSION,
    download_url=DOWNLOAD_URL,
    install_requires=['numpy', 'pandas', 'matplotlib', 'scipy', 'sklearn',
                      'h5py', 'pyemma', 'bokeh'],
    packages=find_packages(),
    classifiers=[
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3.5',
        'Operating System :: POSIX',
        'Operating System :: Unix',
        'Operating System :: MacOS'
    ]
)
