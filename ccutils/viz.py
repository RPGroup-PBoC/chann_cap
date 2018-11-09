# Import plotting utilities
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl

# Seaborn, useful for graphics
import seaborn as sns

import numpy as np

"""
Title:
    viz.py
Last update:
    2018-10-22
Author(s):
    Manuel Razo-Mejia
Purpose:
    This file compiles all of the relevant functions for plotting style
    related to the channel capacity project.
"""

# Default RP plotting style
def set_plotting_style():
    """
    Formats plotting enviroment to that used in Physical Biology of the Cell,
    2nd edition. To format all plots within a script, simply execute
    `mwc_induction_utils.set_plotting_style() in the preamble.
    """
    rc = {'lines.linewidth': 1.25,
          'axes.labelsize': 8,
          'axes.titlesize': 9,
          'axes.facecolor': '#E3DCD0',
          'xtick.labelsize': 7,
          'ytick.labelsize': 7,
          'font.family': 'Lucida Sans Unicode',
          'grid.linestyle': '-',
          'grid.linewidth': 0.5,
          'grid.color': '#ffffff',
          'legend.fontsize': 8}
    plt.rc('text.latex', preamble=r'\usepackage{sfmath}')
    plt.rc('xtick.major', pad=-1)
    plt.rc('ytick.major', pad=-1)
    plt.rc('mathtext', fontset='stixsans', sf='sansserif')
    plt.rc('figure', figsize=[3.5, 2.5])
    plt.rc('svg', fonttype='none')
    plt.rc('legend', title_fontsize='8', frameon=True, 
           facecolor='#E3DCD0', framealpha=1)
    sns.set_style('darkgrid', rc=rc)
    sns.set_palette("colorblind", color_codes=True)
    sns.set_context('notebook', rc=rc)
    
# Plotting the standard PMF CDF plot
def pmf_cdf_plot(x, px, legend_var, color_palette='Blues',
                 mean_mark=True, marker_height=None,
                 marker_size=200,
                 pmf_edgecolor='k', pmf_alpha=0.8,
                 color_bar=True, cbar_label='', binstep=1,
                 figsize=(3.5, 3.5), title='', xlabel='', xlim=None, ylim=None,
                 labelsize=8, cbar_fontsize=8):
    '''
    Custom plot of the PMF and the CDF of multiple distributions
    with a side legend.
    Parameters
    ----------
    x : array-like. 1 x N.
        X values at which the probability P(X) is being plotted
    px : array-like. M x N
        Probability of each of the values of x for different conditions
        such as varying repressor copy number, inducer concentration or
        binding energy.
    legend_var : array-like. 1 X M.
        Value of the changing variable between different distributions
        being plotted
    colors : str or list.
        Color palete from the seaborn options to use for the different
        distributions.
        The user can feed the name of a seaborn color palette or a list of
        RGB colors that would like to use as color palette.
    mean_mark : bool.
        Boolean indicating if a marker should be placed to point at
        the mean of each distribution. Default=True
    marker_height : float.
        Height that all of the markers that point at the mean should
        have.
    marker_size : float. Default = 200.
        Size of the markers that indicate the mean of the PMF.
    pmf_edgecolor : string or RGB colors. Default : 'k'
        Color for the edges of the histograms in the PMF plot.
        If a single entry is listed, this color is used for all PMF edges.
    pmf_alpha : float. [0, 1]
        Alpha value for the histogram colors.
    color_bar : bool.
        Boolean indicating if a color bar should be added on the side
        to indicate the different variable between distributions.
        Default=True
    cbar_label : str.
        Side label for color bar.
    binstep : int.
        If not all the bins need to be plot it can plot every binstep
        bins. Especially useful when plotting a lot of bins.
    figsize : array-like. 1 x 2.
        Size of the figure
    title : str.
        Title for the plot.
    xlabel : str.
        Label for the x plot
    xlim : array-like. 1 x 2.
        Limits on the x-axis.
    ylim : array-like. 1 x 2.
        Limits on the y-axis for the PMF. The CDF goes from 0 to 1 by
        definition.
    labelsize : float. Default = 8
        Font size for the plot labels, meaning the (A) and (B) on the corner of
        the plot.
    cbar_fontsize : float. Default = 8
        Font size for the labels on the colorbar
    '''
    # Determine if user gave the name of a color palette or a list of colors
    if type(color_palette) == str:
        colors = sns.color_palette(color_palette, n_colors=len(legend_var))
    else:
        colors = list(color_palette)

    # Determine if a single or multiple colors were listed for pmf_edgecolor
    if len(pmf_edgecolor) == 1:
        pmf_edgecolor = [pmf_edgecolor] * len(legend_var)
    # Initialize figure
    fig, ax = plt.subplots(2, 1, figsize=figsize, sharex=True)
    ax[0].yaxis.set_major_formatter(mpl.ticker.ScalarFormatter(
                                    useMathText=True,
                                    useOffset=False))

    # Loop through inducer concentrations
    for i, c in enumerate(legend_var):
        # PMF plot
        ax[0].plot(x[0::binstep], px[i, 0::binstep],
                   label=str(c), drawstyle='steps',
                   color=pmf_edgecolor[i])
        # Fill between each histogram
        ax[0].fill_between(x[0::binstep], px[i, 0::binstep],
                           color=colors[i], alpha=pmf_alpha, step='pre')
        # CDF plot
        ax[1].plot(x[0::binstep], np.cumsum(px[i, :])[0::binstep],
                   drawstyle='steps',
                   color=colors[i], linewidth=2)

    # Label axis
    ax[0].set_title(title)
    ax[0].set_ylabel('probability')
    ax[0].margins(0.02)
    # Set scientific notation
    ax[0].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    ax[0].set_xlim(xlim)
    ax[0].set_ylim(ylim)

    ax[1].legend(loc=0)
    ax[1].set_xlabel(xlabel)
    ax[1].set_ylabel('CDF')
    ax[1].margins(0.02)

    # Declare color map for legend
    # cmap = plt.cm.get_cmap(color_palette, len(legend_var))
    cmap = mpl.colors.LinearSegmentedColormap.from_list('custom', colors,
                                                        N=len(legend_var))
    bounds = np.linspace(0, len(legend_var), len(legend_var) + 1)

    # Compute mean mRAN copy number from distribution
    mean_dist = [np.sum(x * prob) for prob in px]
    # Plot a little triangle indicating the mean of each distribution
    if marker_height is None:
        height = np.max(px) * 1.1
    else:
        height = marker_height
    mean_plot = ax[0].scatter(mean_dist, [height] * len(mean_dist),
                              marker='v', s=marker_size,
                              c=np.arange(len(mean_dist)), cmap=cmap,
                              edgecolor='k', linewidth=1.5)

    # Generate a colorbar with the concentrations
    cbar_ax = fig.add_axes([0.95, 0.25, 0.03, 0.5])
    cbar = fig.colorbar(mean_plot, cax=cbar_ax)
    cbar.ax.get_yaxis().set_ticks([])
    for j, c in enumerate(legend_var):
        cbar.ax.text(1, j / len(legend_var) + 1 / (2 * len(legend_var)),
                     c, ha='left', va='center',
                     transform=cbar_ax.transAxes, fontsize=cbar_fontsize)
    cbar.ax.get_yaxis().labelpad = 35
    cbar.set_label(r'{:s}'.format(cbar_label))

    plt.figtext(-0.02, .9, '(A)', fontsize=labelsize)
    plt.figtext(-0.02, .46, '(B)', fontsize=labelsize)

    plt.subplots_adjust(hspace=0.06)


# Plot a joint 2D distribution with marginals on the sides
def joint_marginal_plot(x, y, Pxy,
                        xlabel='', ylabel='', title='',
                        size=5.5, ratio=5, space=0.1,
                        marginal_color='black',
                        marginal_fill=sns.color_palette('colorblind',
                                                        n_colors=1),
                        marginal_alpha=0.8,
                        joint_cmap='Blues', include_cbar=True,
                        cbar_label='probability', vmin=None, vmax=None):
    '''
    Plots the joint and marginal distributions like the seaborn jointplot.

    Parameters
    ----------
    x, y : array-like.
        Arrays that contain the values of the x and y axis. Used to set the
        ticks on the axis.
    Pxy : 2d array. len(x) x len(y)
        2D array containing the value of the joint distributions to be plot
    xlabel : str.
        X-label for the joint plot.
    ylabel : str.
        Y-label for the joint plot.
    title : str.
        Title for the entire plot.
    size : float.
        Figure size.
    ratio : float.
        Plot size ratio between the joint 2D hist and the marginals.
    space : float.
        Space beteween marginal and joint plot.
    marginal_color: str or RGB number. Default 'black'
        Color used for the line of the marginal distribution
    marginal_fill: str or RGB number. Default seaborn colorblind default
        Color used for the filling of the marginal distribution
    marginal_alpha : float. [0, 1]. Default = 0.8
        Value of alpha for the fill_between used in the marginal plot.
    joint_cmap : string. Default = 'Blues'
        Name of the color map to be used in the joint distribution.
    include_cbar : bool. Default = True
        Boolean indicating if a color bar should be included for the joint
        distribution values.
    cbar_label : str. Default = 'probability'
        Label for the color bar
    vmin, vmax : scalar, optional, default: None
        From the plt.imshow documentation:
        `vmin` and `vmax` are used in conjunction with norm to normalize
        luminance data.  Note if you pass a `norm` instance, your
        settings for `vmin` and `vmax` will be ignored.
    '''
    # Define the extent of axis and aspect ratio of heatmap
    extent = [x.min(), x.max(), y.min(), y.max()]
    aspect = (x.max() - x.min()) / (y.max() - y.min())

    # Initialize figure
    f = plt.figure(figsize=(size, size))

    # Specify gridspec
    gs = plt.GridSpec(ratio + 1, ratio + 1)

    # Generate axis
    # Joint
    ax_joint = f.add_subplot(gs[1:, :-1])

    # Marginals
    ax_marg_x = f.add_subplot(gs[0, :-1], sharex=ax_joint)
    ax_marg_y = f.add_subplot(gs[1:, -1], sharey=ax_joint)

    # Turn off tick visibility for the measure axis on the marginal plots
    plt.setp(ax_marg_x.get_xticklabels(), visible=False)
    plt.setp(ax_marg_y.get_yticklabels(), visible=False)

    # Turn off the ticks on the density axis for the marginal plots
    plt.setp(ax_marg_x.yaxis.get_majorticklines(), visible=False)
    plt.setp(ax_marg_x.yaxis.get_minorticklines(), visible=False)
    plt.setp(ax_marg_y.xaxis.get_majorticklines(), visible=False)
    plt.setp(ax_marg_y.xaxis.get_minorticklines(), visible=False)
    plt.setp(ax_marg_x.get_yticklabels(), visible=False)
    plt.setp(ax_marg_y.get_xticklabels(), visible=False)
    ax_marg_x.yaxis.grid(False)
    ax_marg_y.xaxis.grid(False)

    # Set spacing between plots
    f.subplots_adjust(hspace=space, wspace=space)

    # Plot marginals
    ax_marg_x.plot(x, Pxy.sum(axis=0), drawstyle='steps', color=marginal_color)
    ax_marg_x.fill_between(x, Pxy.sum(axis=0), alpha=marginal_alpha, step='pre',
                           color=marginal_fill)
    ax_marg_y.plot(Pxy.sum(axis=1), y, drawstyle='steps', color=marginal_color)
    ax_marg_y.fill_between(Pxy.sum(axis=1), y, alpha=marginal_alpha, step='pre',
                           color=marginal_fill)

    # Set title above the ax_arg_x plot
    ax_marg_x.set_title(title)

    # Plot joint distribution
    cax = ax_joint.matshow(Pxy, cmap=joint_cmap, origin='lower',
                           extent=extent, aspect=aspect, vmin=vmin, vmax=vmax)
    # Move ticks to the bottom of the plot
    ax_joint.xaxis.tick_bottom()
    ax_joint.grid(False)

    # Label axis
    ax_joint.set_xlabel(xlabel)
    ax_joint.set_ylabel(ylabel)

    if include_cbar:
        # Generate a colorbar with the concentrations
        cbar_ax = f.add_axes([1.0, 0.25, 0.03, 0.5])

        # Add colorbar, make sure to specify tick locations to match desired ticklabels
        cbar = f.colorbar(cax, cax=cbar_ax, format='%.0E')

        # Label colorbar
        cbar.set_label(cbar_label)
