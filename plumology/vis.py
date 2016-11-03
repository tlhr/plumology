'''vis - Visualisation and plotting tools'''

from typing import Union, Sequence, Optional, List, Tuple
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import RegularPolyCollection
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

from .util import stats, chunk_range, free_energy
from .util import dist1D as calc_dist1D
from .io import read_multi, read_plumed

__all__ = ['fast', 'dist1D', 'dist2D', 'hexplot', 'histogram', 'dihedral',
           'history', 'interactive', 'metai', 'rmsd', 'convergence']


def fast(filename: str,
         step: int=1,
         columns: Union[Sequence[int], Sequence[str], str, None]=None,
         start: int=0,
         stop: int=sys.maxsize,
         stat: bool=True,
         plot: bool=True) -> None:
    '''
    Plot first column with every other column and show statistical information.

    Parameters
    ----------
    filename : Plumed file to read.
    step : Reads every step-th line instead of the whole file.
    columns : Column numbers or field names to read from file.
    start : Starting point in lines from beginning of file,
        including commented lines.
    stop : Stopping point in lines from beginning of file,
        including commented lines.
    stats : Show statistical information.
    plot : Plot Information.

    '''
    if columns is not None:
        if isinstance(columns, str):
            columns = [columns]
        if 'time' not in columns:
            columns.insert(0, 'time')

    data = read_multi(
        filename,
        columns=columns,
        step=step,
        start=start,
        stop=stop,
        dataframe=True
    )

    if len(data['time'].values.shape) > 1:
        time = data['time'].values[:, 0]
        data = data.drop(['time'], axis=1)
        data['time'] = time

    if stat:
        stat_strings = stats(data.columns, data.values)
        for s in stat_strings:
            print(s)

    if plot:
        fig = plt.figure(figsize=(16, 3 * len(data.columns)))

        i = 0
        for col in data.columns:
            if col == 'time':
                continue
            i += 1
            ax = fig.add_subplot(len(data.columns) // 2 + 1, 2, i)
            ax.plot(data['time'], data[col])
            ax.set_xlabel('time')
            ax.set_ylabel(col)


def hexplot(
        ax: plt.Axes,
        grid: np.ndarray,
        data: np.ndarray,
        hex_size: float=11.5,
        cmap: str='viridis'
) -> plt.Axes:
    '''
    Plot grid and data on a hexagon grid. Useful for SOMs.

    Parameters
    ----------
    ax : Axes to plot on.
    grid : Array of (x, y) tuples.
    data : Array of len(grid) with datapoint.
    hex_size : Radius in points determining the hexagon size.
    cmap : Colormap to use for colouring.

    Returns
    -------
    ax : Axes with hexagon plot.

    '''

    # Create hexagons
    collection = RegularPolyCollection(
        numsides=6,
        sizes=(2 * np.pi * hex_size ** 2,),
        edgecolors=(0, 0, 0, 0),
        transOffset=ax.transData,
        offsets=grid,
        array=data,
        cmap=plt.get_cmap(cmap)
    )

    # Scale the plot properly
    ax.add_collection(collection, autolim=True)
    ax.set_xlim(grid[:, 0].min() - 0.75, grid[:, 0].max() + 0.75)
    ax.set_ylim(grid[:, 1].min() - 0.75, grid[:, 1].max() + 0.75)
    ax.axis('off')

    return ax


def dihedral(cvdata: pd.DataFrame,
             cmap: Optional[ListedColormap]=None) -> plt.Figure:
    '''
    Plot dihedral angle data as an eventplot.

    Parameters
    ----------
    cvdata : Dataframe with angle data only.
    cmap : Colormap to use.

    Returns
    -------
    fig : Figure with drawn events.

    '''
    # Custom periodic colormap
    if cmap is None:

        # Define some colors
        blue = '#2971B1'
        red = '#B92732'
        white = '#F7F6F6'
        black = '#3B3B3B'

        # Define the colormap
        periodic = LinearSegmentedColormap.from_list(
            'periodic',
            [black, red, red, white, blue, blue, black],
            N=2560,
            gamma=1
        )
        cmap = periodic

    # Setup plot
    fig, axes = plt.subplots(figsize=(16, 32), nrows=cvdata.shape[1])
    fig.subplots_adjust(top=0.95, bottom=0.01, left=0.2, right=0.99, hspace=0)

    for ax, col in zip(axes, cvdata.columns):

        # No interpolation because we want discrete lines
        ax.imshow(np.atleast_2d(cvdata[col]), aspect='auto',
                  cmap=periodic, interpolation='none')

        # Create labels
        pos = list(ax.get_position().bounds)
        x_text = pos[0] - 0.01
        y_text = pos[1] + pos[3]/2.
        fig.text(x_text, y_text, col, va='center', ha='right', fontsize=10)

        # Remove clutter
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    return fig


def history(cvdata: pd.DataFrame) -> None:
    '''
    Plot CV history as a 2D scatter plot.

    Parameters
    ----------
    cvdata : Dataframe with time as first column and CV values for the rest.

    '''
    fig = plt.figure(figsize=(16, 3 * len(cvdata.columns) // 3))
    for i, col in enumerate(cvdata.columns):
        if col == 'time':
            continue
        ax = fig.add_subplot(len(cvdata.columns) // 3 + 1, 3, i)
        ax.plot(cvdata['time'], cvdata[col], 'o')
        ax.set_xlabel('time')
        ax.set_ylabel(col)
    plt.tight_layout()


def histogram(cvdata: pd.DataFrame,
              cv_min: Optional[Sequence[float]]=None,
              cv_max: Optional[Sequence[float]]=None,
              time: Optional[float]=None,
              nchunks: int=3,
              nbins: int=50) -> None:
    '''
    Plots histograms of CVs in predefined chunks.

    Parameters
    ----------
    cvdata : Dataframe with time as first column and CV values for the rest.
    cv_min : Minimum possible values for CVs.
    cv_max : Maximum possible values for CVs.
    time : If given, timespan of the first histogram,
        equal to other chunks otherwise.
    nchunks : Number of histograms to plot.
    nbins : Number of bins to use for the histogram.

    '''
    if cv_min is None or cv_max is None:
        cv_min = [cvdata[cv].values.min() for cv in cvdata.columns]
        cv_max = [cvdata[cv].values.max() for cv in cvdata.columns]

    plot_chunks = nchunks
    chunks = chunk_range(cvdata['time'].values[0],
                         cvdata['time'].values[-2],
                         nchunks, time)

    fig = plt.figure(figsize=(16, 3 * len(cvdata.columns)))
    for i, col in enumerate(cvdata.columns):
        if col == 'time':
            continue
        for j, time in enumerate(chunks):
            ax = fig.add_subplot(len(cvdata.columns),
                                 plot_chunks,
                                 nchunks * i + j + 1)
            hist, bins = np.histogram(cvdata[cvdata['time'] < time][col],
                                      range=(cv_min[i - 1], cv_max[i - 1]),
                                      bins=nbins, normed=False)
            center = (bins[:-1] + bins[1:]) / 2
            width = (abs(cv_min[i - 1]) + abs(cv_max[i - 1])) / nbins
            ax.bar(center, hist, width=width, align='center')
            ax.set_xlabel(col)
            ax.set_xlim(cv_min[i - 1], cv_max[i - 1])
    plt.tight_layout()


def convergence(
        hills: pd.DataFrame,
        summed_hills: pd.DataFrame,
        time: int,
        kbt: float,
        factor: float=1.0,
        constant: float=0.0
) -> plt.Figure:
    '''
    Estimate convergence by comparing CV histograms and sum_hills output.

    Parameters
    ----------
    hills : Hills files to be passed to PLUMED, globbing allowed.
    summed_hills : Output from io.sum_hills().
    time : The minimum time to use from the histogram.
    kbt : k_B * T for the simulation as output by PLUMED.
    factor : Factor to rescale the FES.
    constant : Constant to translate the FES.

    Returns
    -------
    fig : Matplotlib figure.

    '''

    dist, ranges = calc_dist1D(hills[hills['time'] > time])
    fes = factor * free_energy(dist, kbt) + constant

    # consistent naming
    summed_hills.columns = hills.columns.drop('time')

    # sum_hills binning can be inconsistent
    if summed_hills.shape[0] > fes.shape[0]:
        summed_hills = summed_hills[:fes.shape[0]]

    ncols = len(fes.columns)
    fig = plt.figure(figsize=(16, 4 * (ncols // 2 + 1)))
    for i, col in enumerate(fes.columns):
        ax = fig.add_subplot(ncols // 2 + 1, 2, i + 1)
        ax.plot(ranges[col], fes[col], label='histogram')
        ax.plot(ranges[col], summed_hills[col], label='sum_hills')
        ax.set_xlabel(col)
    ax.legend()

    return fig


def dist1D(dist: pd.DataFrame,
           ranges: pd.DataFrame,
           grouper: Optional[str]=None,
           nx: Optional[int]=2,
           size: Optional[Tuple[int, int]]=(8, 6)) -> plt.Figure:
    '''
    Plot 1D probability distributions.

    Parameters
    ----------
    dist : Multiindexed dataframe with force field as primary
        index and distributions as created by dist1D().
    ranges : Multiindexed dataframe with force field as primary
        index and edges as created by dist1D().
    grouper : Primary index to use for plotting multiple lines.
    nx : Number of plots per row.
    size : Relative size of each plot.

    Returns
    -------
    fig : matplotlib figure.

    '''

    # Setup plotting parameters
    if grouper is not None:
        for k, df in dist.groupby(level=[grouper]):
            cols = df.columns
            break
    else:
        cols = dist.columns

    nplots = len(cols)
    xsize, ysize = nx, nplots // nx + 1
    fig = plt.figure(figsize=(xsize * size[0], ysize * size[1]))

    # Iterate over CVs
    for j, col in enumerate(cols):

        ax = fig.add_subplot(ysize, xsize, j + 1)
        ax.set_xlabel(col)

        if grouper is not None:
            # Iterate over dataframes in groupby object, plot them together
            for (k, df), (_, rf) in zip(dist.groupby(level=[grouper]),
                                        ranges.groupby(level=[grouper])):
                ax.plot(rf[col], df[col], label=k, linewidth=2)
        else:
            ax.plot(ranges[col], dist[col], linewidth=2)

    if grouper is not None:
        ax.legend(loc=2, framealpha=0.75)

    return fig


def dist2D(dist: pd.DataFrame,
           ranges: pd.DataFrame,
           nlevels: int=16,
           nx: int=2,
           size: int=6,
           colorbar: bool=True,
           name: str='dist') -> plt.Figure:
    '''
    Plot 2D probability distributions.

    Parameters
    ----------
    dist : Multiindexed dataframe with force field as primary
        index and distributions as created by dist2D().
    ranges : Multiindexed dataframe with force field as primary
        index and edges as created by dist1D().
    nlevels : Number of contour levels to use.
    nx : Number of plots per row.
    size : Relative size of each plot.
    colorbar : If true, will plot a colorbar.
    name : Name of the distribution.

    Returns
    -------
    fig : matplotlib figure.

    '''

    # Setup plotting parameters
    nplots = dist.shape[1]
    xsize, ysize = nx, (nplots // nx) + 1
    cmap = plt.get_cmap('viridis')
    fig = plt.figure(figsize=(xsize * size, ysize * size))

    for i, k in enumerate(dist.keys()):

        # Get keys for both CVs
        kx, ky = k.split('.')

        # Prepare plotting grid (np.meshgrid doesn't work)
        X = np.broadcast_to(ranges[kx], dist[k].unstack().shape)
        Y = np.broadcast_to(ranges[ky], dist[k].unstack().shape).T
        Z = dist[k].unstack().values.T

        # Contour levels taking inf into account
        levels = np.linspace(np.amin(Z[~np.isinf(Z)]),
                             np.amax(Z[~np.isinf(Z)]), nlevels)
        ax = fig.add_subplot(ysize, xsize, i + 1)
        cm = ax.contourf(X, Y, Z, cmap=cmap, levels=levels)
        ax.set_xlabel(kx)
        ax.set_ylabel(ky)
        ax.set_title(name)

    if colorbar:
        fig.colorbar(cm)

    return fig


def rmsd(rmsd: pd.DataFrame,
         aspect: Tuple[int, int]=(4, 6),
         nx: int=5) -> plt.Figure:
    '''
    Plot RMSDs.

    Parameters
    ----------
    rmsd : Dataframe with force field as index and CVs as columns.
    aspect : Aspect ratio of individual plots.
    nx : Number of plots before wrapping to next row.

    Returns
    -------
    rmsd : Figure with RMSDs.

    '''

    ny = len(rmsd.columns) // nx + 1
    fig = plt.figure(figsize=(nx * aspect[0], ny * aspect[1]))

    for i, col in enumerate(rmsd.columns):
        nbars = rmsd[col].shape[0]
        ax = fig.add_subplot(ny, nx, i + 1)
        ax.bar(np.arange(nbars), rmsd[col].values, linewidth=0)
        ax.set_title(col)
        ax.set_xticks(np.linspace(0.5, 0.5 + nbars - 1, nbars))
        ax.set_xticklabels(list(rmsd[col].keys()))
        ax.set_ylabel('RMSD [{0}]'.format('ppm' if 'CS' in col else 'Hz'))
        plt.setp(
            plt.gca().get_xticklabels(),
            rotation=45,
            horizontalalignment='right'
        )
    plt.tight_layout()

    return fig


def metai(file: str,
          step: int=1,
          start: int=0,
          stop: int=sys.maxsize) -> None:
    '''
    Plot metainference information.

    Parameters
    ----------
    file : Plumed file to read.
    step : Plot every step-th value.
    start : Start plotting from here.
    stop : Stop plotting here.

    '''
    data = read_plumed(
        file,
        step=step,
        start=start,
        stop=stop
    )

    ny, nx = len(data.columns) // 2 + 1, 2
    fig = plt.figure(figsize=(nx * 8, ny * 5))

    ax_sigmaMean = fig.add_subplot(ny, nx, 1)
    ax_sigma = fig.add_subplot(ny, nx, 2)
    ax_kappa = fig.add_subplot(ny, nx, 3)

    sc = 0
    i = 4
    for col in data.columns:
        if 'sigmaMean' in col:
            ax_sigmaMean.plot(data['time'], data[col], label=col.split('_')[1])
            sc += 1
        elif 'sigma' in col:
            ax_sigma.plot(data['time'], data[col], label=col.split('_')[1])
        elif 'time' not in col:
            ax = fig.add_subplot(ny, nx, i)
            i += 1
            ax.plot(data['time'], data[col])
            ax.set_xlabel('time')
            ax.set_ylabel(col)

    name = data.columns[1].split('.')[0]
    for j in range(sc):
        kappa = 1 / ((data[name + '.sigmaMean_' + str(j)] *
                      data[name + '.rewSigmaMean']) ** 2 +
                     data[name + '.sigma_' + str(j)] ** 2)
        ax_kappa.plot(data['time'], kappa, label=j)

    ax_sigmaMean.set_xlabel('time')
    ax_sigmaMean.set_ylabel('sigma_mean')
    ax_sigmaMean.legend(loc=3, framealpha=0.75)
    ax_sigma.set_xlabel('time')
    ax_sigma.set_ylabel('sigma')
    ax_sigma.legend(loc=1, framealpha=0.75)
    ax_kappa.set_xlabel('time')
    ax_kappa.set_ylabel('kappa')
    ax_kappa.legend(loc=1, framealpha=0.75)


def interactive(file: str,
                x: Union[str, int]=0,
                y: Union[str, int, List[str], List[int]]=1,
                step: int=1,
                start: int=0,
                stop: int=sys.maxsize) -> None:
    '''
    Plot values interactively.

    Parameters
    ----------
    file : Plumed file to read.
    x : x-axis to use for plotting, can be specified
        either as column index or field.
    y : y-axis to use for plotting, can be specified
        either as column index or field.
    step : Plot every step-th value.
    start : Start plotting from here.
    stop : Stop plotting here.

    '''
    try:
        from bokeh.plotting import show, figure
        import bokeh.palettes as pal
    except ImportError:
        raise ImportError(
            'Interactive plotting requires Bokeh to be installed'
        )

    cols = [x] + y if isinstance(y, list) else [x] + [y]
    fields, data = read_plumed(
        file,
        step=step,
        start=start,
        stop=stop,
        columns=cols,
        dataframe=False
    )
    TOOLS = 'pan,wheel_zoom,box_zoom,resize,hover,reset,save'
    p = figure(tools=TOOLS, plot_width=960, plot_height=600)
    for i in range(1, len(fields)):
        p.line(
            data[:, 0],
            data[:, i],
            legend=fields[i],
            line_color=pal.Spectral10[i],
            line_width=1.5
        )
    p.xaxis.axis_label = fields[0]
    p.yaxis.axis_label = fields[1]
    show(p)
