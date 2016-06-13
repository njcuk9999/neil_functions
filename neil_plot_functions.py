"""
Description of program
"""
import numpy as np
import matplotlib.pyplot as plt
from numexpr import evaluate as ne
import matplotlib.mlab as ml
import matplotlib.patches as mpatch
import matplotlib.colors as colors
import matplotlib.cm as cmx
from tqdm import tqdm


# ==============================================================================
# Define variables
# ==============================================================================


# ==============================================================================
# Define Actual Plotting functions
# ==============================================================================
# plots a range of contour types
def plot_contours(frame, x, y, z=None, nx=128, ny=128, res=None, **kwargs):
    """
    Plots various contours

    if x, y creates number density contours
    if x, y, z uses the z data to create contours

    nx, ny are resolution of surface in x and y

    :param frame: matplotllib axis, i.e. plt.subplot(111)
    :param x: array of floats, the horizontal axis values
    :param y: array of floats, the vertical axis values
    :param z: None or array of floats, the contour/colour axis
    :param nx: number of bins to use in horizontal axis
    :param ny: number of bins to use in the vertical axis
    :param res: number of levels of contours (i.e contour resolution)
    :param kwargs: see below

    kwargs are:

        - log       if not None then logNorm function       default is None
        - fill      if True fills contours                  default is False
        - lines     if True plots line contours             default is True
        - zorder    sets zorder of contours                 default is 1
        - cmap      if using griddata colourmap is cmap
                        default is plt.get_cmap('gist_heat')
        - linecolour    colour of contour lines             default is 'k'
        - cb        if True plots colourbar                 default is False
        - cbo       colourbar orientation               default is 'vertical'
        - cbtitle   String colourbar label              default is 'Z'
        - filla     alpha on contour fill               default is 1
        - fillc     colormap to use on filled contours  default is None (grey)
        - linea     alpha on contour line               default is 1
    """
    # -------------------------------------------------------------------------
    # defaults
    d = dict()
    d['log'] = kwargs.get('log', None)
    d['fill'] = kwargs.get('fill', False)
    d['lines'] = kwargs.get('lines', True)
    d['zorder'] = kwargs.get('zorder', 1)
    d['cmap'] = kwargs.get('cmap', plt.get_cmap('gist_heat'))
    d['linecolour'] = kwargs.get('linecolour', 'k')
    d['cb'] = kwargs.get('cb', False)
    d['cbo'] = kwargs.get('cbo', 'vertical')
    d['cbtitle'] = kwargs.get('cbtitle', 'Z')
    d['filla'] = kwargs.get('filla', 1)
    d['fillc'] = kwargs.get('fillc', None)
    d['linea'] = kwargs.get('linea', 1)
    d['label'] = kwargs.get('label', None)
    d['levels'] = kwargs.get('levels', None)
    d['outliers'] = kwargs.get('outliers', False)
    # -------------------------------------------------------------------------
    # if no frame assume no plot and return object(s)
    plot = True
    if frame is None:
        plot = False
        framex = plt.gca()
    else:
        framex = frame
    # -------------------------------------------------------------------------
    # sort out the surface xi, yi, zi
    xmin, xmax = np.min(x), np.max(x)
    ymin, ymax = np.min(y), np.max(y)
    xi = np.linspace(xmin, xmax, nx)
    yi = np.linspace(ymin, ymax, ny)
    if z is None:
        pc = plot_cont(x, y, nx, ny, res=res, levels=d['levels'],
                       fillc=d['fillc'])
        zi, d['extent'], d['levels'], d['colours'] = pc
        xi, yi = x, y
        N = len(d['levels'])
    else:
        zi = ml.griddata(np.array(x), np.array(y), np.array(z), xi, yi)
        if res is None:
            N = 15
        else:
            N = len(np.arange(np.min(z), np.max(z), res))
    # -------------------------------------------------------------------------
    # if d['fillc'] is not None:
    #    d['colours'] = 'r'
    # -------------------------------------------------------------------------
    # plot the contours if there is some data
    if len(zi[zi != 0]) > 0:
        if d['fill']:
            if z is None:
                im = framex.contourf(zi, extent=d['extent'],
                                     levels=d['levels'],
                                     interpolation='nearest', alpha=d['filla'],
                                     colors=d['colours'], zorder=d['zorder'])
            elif d['log'] is None:
                im = framex.pcolormesh(xi, yi, zi, edgecolor='None',
                                       cmap=d['cmap'], alpha=d['filla'])
            else:
                im = framex.pcolormesh(xi, yi, zi, norm=d['log'],
                                       edgecolor='None', cmap=d['cmap'],
                                       zorder=d['zorder'], alpha=d['filla'])
            try:
                im.set_rasterized(True)
            except AttributeError:
                print "\n\tWARNING: Could not set figure to rasterized"
            if d['cb'] and d['cbo'] == 'vertical':
                cb = plt.colorbar(im, ax=framex, orientation='vertical')
                cb.set_label(d['cbtitle'], rotation=270)
                cb.solids.set_edgecolor("face")
                cb.ax.tick_params(labelsize=16)
            elif d['cbo'] == 'horizontal':
                cb = plt.colorbar(im, ax=framex, orientation='horizontal')
                cb.set_label(d['cbtitle'])
                cb.solids.set_edgecolor("face")
                cb.ax.tick_params(labelsize=16)
        else:
            im = None
        if d['lines'] or not d['fill']:
            # add contour lines
            if z is None:
                cs = framex.contour(zi, extent=d['extent'], levels=d['levels'],
                                    interpolation='nearest', alpha=d['linea'],
                                    zorder=d['zorder'], colors=d['linecolour'],
                                    label=d['label'])
            else:
                cs = framex.contour(xi, yi, np.array(zi), N,
                                    colors=d['linecolour'], label=d['label'],
                                    interpolation='nearest',
                                    alpha=d['linea'], zorder=d['zorder'])
        else:
            cs = None

        # deal with plotting outliers
        if d['outliers'] and z is None:
            outliers = get_outliers(im, x, y)
            if d['fill']:
                return framex, im, outliers
            else:
                return framex, cs, outliers
        # finally return
        if not plot:
            plt.close()
            if d['fill']:
                return im
            else:
                return cs
        else:
            if d['fill']:
                return framex, im
            else:
                return framex, cs


def cut_plot(xx, yy, exx, eyy, xycuts, xylogic, name, cs, xlabel, ylabel,
             savename, limits, lvls, use_limits=True, return_frame=False,
             plot_outliers=True, plot_rejected=False, plot_accepted=False,
             reject_from_contour=None, lw=1.5, compare=None, no_conts=False,
             custom_fig_size=None, no_legend=False, no_colourbar=False):
    """

    :param xx: array of float, x points

    :param yy: array of floats, y points

    :param exx: array of float, uncertainty in x points

    :param eyy: array of float, uncertainty in y points

    :param xycuts: list of cuts in the form:

                   xycuts = [cuts1, cuts2, ..., cutsn]

                   where cuts1 = [cut1, cut2, ..., cutn]

                   where cut1 = string logic statement

            if one wants a cut of x > 5 cut1 would be as follows:

                   cut1 = 'x > 5'

                   one may also use y and x as variables i.e.:

                   cut1 = 'y > 5*x'

            The logic is as follows:

                   if xycuts = [[cut1a, cut1b, cut1c], [cut2a, cut2b]]

                   the cuts produced are as follows:

                   cut1 = cut1a & cut1b & cut1c
                   cut2 = cut2a & cut2b

                   currently and is the only combination available, thus or
                   statements should be treated as separate cuts

            Thus an example xycuts could be:

                   [['x > 5', 'y > 5*x'], ['y > 2', 'y < 4']

    :param xylogic: currently not used (will be a list of '&' and '|' statements
                    in a similar configuration to xycuts such that one can
                    switch between "and" and "or" logic

                    default value should be [['&', '&'], ['&']]
                    i.e. must be the same shape as xycuts

    :param name: list of strings, name of each cut (for legend and stats)
                 must be equal to the length of xycuts

    :param cs: list of strings, line colour for each cut (for plotting)
               must be equal to the length of xycuts

    :param xlabel: string, label for the x axis
                   if xlabel == 'V-J' I assume this is a proxy for spectral
                   type and thus twin the axis to show spectral type

    :param ylabel: string, label for the y axis

    :param savename: string, location and filename (without extension) for the
                     created plot

    :param limits: limits of the plot in x and y in the form:
                        [ [x lower limit, x upper limit],
                          [y lower limit, y upper limit]]

    :param lvls: list of numbers, contour levels, this defines the levels
                 and number of levels
                 i.e. lvls = [2, 5, 10, 20, 50]

    :param use_limits: bool, if True uses "limits" parameter to add an extra
                       mask to the plot i.e. do not plot any points outside
                       the area that will be plot finally (saves on computation
                       in the case of a large number of points)

    :param return_frame: bool, if True does not save or show the graph,
                         simply returns the matplotlib.axes for further use

    :param plot_outliers: bool, if True plots those points which do not fall
                          into the contours (i.e. below the density required
                          by "lvls" these points will be plotted in
                          black. This saves having to plot all the points
                          under the contours (smaller plot save size, and easier
                          to load)

    :param plot_rejected: bool, if True plots any points rejected (i.e. False)
                          by the cuts defined by xycuts

    :param plot_accepted: bool, if True plots any points accepted (i.e. True)
                          by the cuts defined by xycuts

    :param reject_from_contour: array of booleans, an additional mask of
                                parameters to not use in the contour plot
                                (nans and infinites already removed)

    :param lw: float, standard matplotlib linewidth keyword argument for the
               widths of the cut lines plotted

    :param compare: list of variables or None, if None then do not plot
                    if a list then list must be in the form
                    compare = [xs, ys, ss, ls, bs]

                    where xs, ys are arrays of floats

                          ss, ls are array of strings, spt and luminosity class
                          respectively

                          bs is an array of booleans, to flag binary

                    (see plot_comparison function for more information)

    :param no_conts: bool, if True the outliers, rejected and accepted points
                     will NOT be plotted as contours i.e. all points will be
                     plotted separately

    :param custom_fig_size: tuple or None, if tuple should be the size in
                            inches of the image in (x, y)
                            i.e. for use in fig.set_size_inches()

    :param no_legend: bool, if True do not plot the legend

    :param no_colourbar: bool, if True do not plot a colorbar

    :return:
    """

    # placeholder, xylogic should be able to handle "or" as well as "and"
    # currently all logic is "and"s
    if xylogic is None:
        pass
    # --------------------------------------------------------------------------
    # set up plot
    plt.close()
    fig, frame = plt.subplots(ncols=1, nrows=1)
    if custom_fig_size is not None:
        fig.set_size_inches(*custom_fig_size)
    elif compare is not None:
        fig.set_size_inches(10, 10)
    else:
        fig.set_size_inches(16, 10)
    xlim, ylim = limits
    # remove nans as they mess with plot_contours
    no_use = np.isnan(xx) | np.isnan(yy) | np.isinf(xx) | np.isinf(yy)
    # if we have a coutour mask apply to no_use
    if reject_from_contour is not None:
        no_use |= reject_from_contour
    # --------------------------------------------------------------------------
    # plot colour distribution as contours
    pcs = dict(linecolour='0.5', alpha=1, zorder=1, fill=True, filla=1.0,
               levels=lvls, outliers=True)
    frame, tmp, omask = plot_contours(frame, xx[-no_use], yy[-no_use], **pcs)
    # add color bar
    if not no_colourbar:
        cb = fig.colorbar(tmp, ticks=lvls, orientation='horizontal')

        binarg = (xlim[1] - xlim[0]) / 128.0, (ylim[1] - ylim[0]) / 128.0
        cbtitle = r' per bin ({0:.1e}$\times${1:.1e})'.format(*binarg)
        cb.ax.set_title('Number of objects' + cbtitle, loc='center')
        cb.ax.set_xticklabels(lvls)
    # plot add artist to legend
    legendrec = mpatch.Rectangle((0, 0), 1, 1, facecolor='0.5', zorder=4,
                                 edgecolor='0.75', linewidth=1)
    legendname = 'Total number = {0:,}'.format(len(xx))
    # --------------------------------------------------------------------------
    pointkwargs = dict(marker='x', alpha=0.25)
    # plot outlier points
    if plot_outliers:
        frame.scatter((xx[-no_use])[omask], (yy[-no_use])[omask], color='k',
                      zorder=0, s=5, **pointkwargs)
    # --------------------------------------------------------------------------
    # plot rejected points
    if plot_rejected:
        frame = plot_accepted_rejected(frame, 'r', xx, yy, xycuts, name,
                                       cs, pointkwargs, lvls, no_use,
                                       no_conts=no_conts)
    # --------------------------------------------------------------------------
    # plot accepted points
    if plot_accepted:
        frame = plot_accepted_rejected(frame, 'a', xx, yy, xycuts, name,
                                       cs, pointkwargs, lvls, no_use)
    # --------------------------------------------------------------------------
    # plot selection lines
    frame = plot_selection_lines(frame, xx, yy, xycuts, name, use_limits,
                                 cs, xlim, ylim, linewidth=lw, zorder=4)
    # --------------------------------------------------------------------------
    # plot average error indication
    frame = plot_error_indication(frame, 0.8, 0.9, exx, eyy, xlim, ylim,
                                  zorder=5)
    # --------------------------------------------------------------------------
    # plot labels
    frame.set_xlabel(xlabel)
    frame.set_ylabel(ylabel)
    # set limits
    frame.set_xlim(*xlim)
    frame.set_ylim(*ylim)
    # --------------------------------------------------------------------------
    # deal with V-J being proxy for spectral type
    if 'V-J' in xlabel:
        frame2 = frame.twiny()
        frame2.set_xlim(*xlim)
        frame2.set_ylim(*ylim)
        frame2.set_xlabel('Spectral Type')
        plt.xticks([2.90, 3.50, 4.60, 6.17, 6.86],
                   ['M0.0', 'M2.0', 'M4.0', 'M6.0', 'M8.0'])
    # --------------------------------------------------------------------------
    h, l = frame.get_legend_handles_labels()
    # --------------------------------------------------------------------------
    # deal with adding comparison stars
    if compare is not None:
        print '\n Adding comparison stars...'
        frame = plot_comparison(frame, compare, h, l, zorder=2)
    # --------------------------------------------------------------------------
    # plot legend
    h, l = [legendrec] + h, [legendname] + l
    if compare is not None and not no_legend:
        plt.legend(h, l, loc=6, numpoints=1, scatterpoints=1,
                   bbox_to_anchor=(1.05, 0.4))
    elif not no_legend:
        plt.legend(h, l, loc=2, numpoints=1, scatterpoints=1)
    # --------------------------------------------------------------------------
    if return_frame:
        return frame
    elif savename is None:
        plt.show()
        plt.close()
    else:
        # save and close
        plt.savefig(savename + '.png', bbox_inches='tight')
        plt.savefig(savename + '.ps')
        plt.savefig(savename + '.pdf', bbox_inches='tight')
        plt.close()


def plot_accepted_rejected(f, kind, xx, yy, xycuts, name, pointkwargs,
                           lvls, no_use, no_conts=False):
    """

    Plots the points that conform to xycuts

    :param f: matplotlib axes, i.e. plt.subplot(111)

    :param kind: string, if 'a' or 'accepted' plot the points that were True
                 in xycuts, else assume we want to plot the 'rejected' points
                 and plot those points that were False in xycuts

    :param xx: array of float, x points

    :param yy: array of floats, y points

    :param xycuts: list of cuts in the form:

                   xycuts = [cuts1, cuts2, ..., cutsn]

                   where cuts1 = [cut1, cut2, ..., cutn]

                   where cut1 = string logic statement

            if one wants a cut of x > 5 cut1 would be as follows:

                   cut1 = 'x > 5'

                   one may also use y and x as variables i.e.:

                   cut1 = 'y > 5*x'

            The logic is as follows:

                   if xycuts = [[cut1a, cut1b, cut1c], [cut2a, cut2b]]

                   the cuts produced are as follows:

                   cut1 = cut1a & cut1b & cut1c
                   cut2 = cut2a & cut2b

                   currently and is the only combination available, thus or
                   statements should be treated as separate cuts

            Thus an example xycuts could be:

                   [['x > 5', 'y > 5*x'], ['y > 2', 'y < 4']

    :param name: list of strings, name of each cut (for legend and stats)
                 must be equal to the length of xycuts

    :param pointkwargs: kwargs passed to scatter plot

    :param lvls: list of numbers, contour levels, this defines the levels
                 and number of levels
                 i.e. lvls = [2, 5, 10, 20, 50]

    :param no_use: array of bool, an additional mask of parameters to
                   not use in the contour plot (e.g. remove nans and infinites)

    :param no_conts: bool, if True the outliers, rejected and accepted points
                     will NOT be plotted as contours i.e. all points will be
                     plotted separately

    :return:
    """
    # plot colour distribution as contours
    pcs = dict(linecolour='0.1', linea=0.25, alpha=0.5, zorder=2, fill=True,
               filla=0.5, levels=lvls, outliers=True, fillc='Reds')

    for row in range(len(xycuts)):
        amask = mask_cut(xx, yy, xycuts[row])
        if kind in ['a', 'accepted']:
            umask = np.array(amask)
        else:
            umask = np.array(-amask)
        # label for points
        llabel = '{0} = {1:,}'.format(name[row], len(xx[umask]))
        # don't plot contours if we have no objects
        if len(xx[umask]) == 0:
            f.scatter(xx[umask], yy[umask], color='r', label=llabel,
                      zorder=4, s=25, **pointkwargs)
            continue
        # want to plot the points that are in umask but not in no_use
        no_use = np.array(no_use, dtype=bool)
        nxx, nyy = np.array(xx[umask & ~no_use]), np.array(yy[umask & ~no_use])
        # plot as distribution
        if no_conts:
            omask = np.ones(len(nxx), dtype=bool)
        else:
            f, tmp, omask = plot_contours(f, nxx, nyy, **pcs)
        # plot outliers using omask from plot_contours
        f.scatter(nxx[omask], nyy[omask], color='r', label=llabel,
                  zorder=4, s=25, **pointkwargs)
    return f


def plot_selection_lines(f, xx, yy, xycuts, name, use_limits, cs, xlim, ylim,
                         **kwargs):
    """

    Plots the selection lines (wrapper for plot_cut_lines) and uses limits
    if use_limits is True

    :param f: matplotlib.axes, i.e. plt.subplot(111)

    :param xx: array of float, x points

    :param yy: array of floats, y points

    :param xycuts: list of cuts in the form:

                   xycuts = [cuts1, cuts2, ..., cutsn]

                   where cuts1 = [cut1, cut2, ..., cutn]

                   where cut1 = string logic statement

            if one wants a cut of x > 5 cut1 would be as follows:

                   cut1 = 'x > 5'

                   one may also use y and x as variables i.e.:

                   cut1 = 'y > 5*x'

            The logic is as follows:

                   if xycuts = [[cut1a, cut1b, cut1c], [cut2a, cut2b]]

                   the cuts produced are as follows:

                   cut1 = cut1a & cut1b & cut1c
                   cut2 = cut2a & cut2b

                   currently and is the only combination available, thus or
                   statements should be treated as separate cuts

            Thus an example xycuts could be:

                   [['x > 5', 'y > 5*x'], ['y > 2', 'y < 4']

    :param name: list of strings, name of each cut (for legend and stats)
                 must be equal to the length of xycuts

    :param use_limits: bool, if True uses "limits" parameter to add an extra
                       mask to the plot i.e. do not plot any points outside
                       the area that will be plot finally (saves on computation
                       in the case of a large number of points)

    :param cs: list of strings, line colour for each cut (for plotting)
               must be equal to the length of xycuts

    :param xlim: tuple or list of floats, [x lower limit, x upper limit]

    :param ylim: tuple or list of floats, [y lower limits, y upper limit]

    :param kwargs: all other keyword arguments passed to plt.plot()

    :return f: matplotlib.axes, i.e. plt.subplot(111)
    """
    # make sure specific keyword args have
    kwargs['zorder'] = kwargs.get('zorder', 1)
    kwargs['linewidth'] = kwargs.get('linewidth', 1)

    # plot selection lines
    # iterate around separate cuts
    for row in range(len(xycuts)):
        # work out mask
        mask = mask_cut(xx, yy, xycuts[row])
        # plot cut lines
        llabel = '{0} = {1:,}'.format(name[row], len(xx[mask]))
        if use_limits:
            plot_cut_line(f, xx, yy, xycuts[row], llabel=llabel, crs=cs[row],
                          xlim=xlim, ylim=ylim, **kwargs)
        else:
            plot_cut_line(f, xx, yy, xycuts[row], llabel=llabel, crs=cs[row],
                          **kwargs)
    return f


def plot_error_indication(f, px, py, ex, ey, xlim, ylim, **kwargs):
    """

    Instead of plotting uncertainties for every point plots the median
    uncertainty at axes (graph) position (px, py) i.e. between 0 and 1 in
    x and y

    :param f: matplotlib.axes, i.e. plt.subplot(111)

    :param px: float, x position on plot (in graph units i.e. between 0 and 1)

    :param py: float, x position on plot (in graph units i.e. between 0 and 1)

    :param ex: array of floats, uncertainties in x

    :param ey: array of floats, uncertainties in y

    :param xlim: tuple or list of floats, [x lower limit, x upper limit]

    :param ylim: tuple or list of floats, [y lower limits, y upper limit]

    :param kwargs: all other keyword arguments passed to plt.errorbar()

    :return f: matplotlib.axes, i.e. plt.subplot(111)
    """
    # make sure specific keyword args have
    kwargs['zorder'] = kwargs.get('zorder', 0)
    kwargs['color'] = kwargs.get('color', 'b')
    kwargs['ms'] = kwargs.get('ms', '5')
    kwargs['label'] = kwargs.get('label', 'median error')

    # if we have no errors don't plot
    if ex is None or ey is None:
        return f
    # mask for nans and infs
    nouse = np.isnan(ex) | np.isnan(ey) | np.isinf(ex) | np.isinf(ey)
    # work out the mean error in x and y
    mex, mey = np.median(ex[-nouse]), np.median(ey[-nouse])
    # convert mean errors in to graph units (0 --> 1)
    mx = (xlim[1] - xlim[0]) * px + xlim[0]
    my = (ylim[1] - ylim[0]) * py + ylim[0]
    # plot the error bar
    f.errorbar([mx], [my], yerr=[mey], xerr=[mex], **kwargs)
    return f


def plot_cut_line(f, xx, yy, cuts, llabel, xlim=None, ylim=None, crs=None,
                  **kwargs):
    """

    :param f: matplotlib.axes, i.e. plt.subplot(111)

    :param xx: array of float, x points

    :param yy: array of floats, y points

    :param cuts: list of cuts in the form:

                   cuts = [cut1, cut2, ..., cutn]

                   where cut1 = string logic statement

            if one wants a cut of x > 5 cut1 would be as follows:

                   cut1 = 'x > 5'

                   one may also use y and x as variables i.e.:

                   cut1 = 'y > 5*x'

            The logic is as follows:

                   if cuts = [cut1a, cut1b, cut1c]

                   the cuts produced are as follows:

                   cut1 = cut1a & cut1b & cut1c

                   currently and is the only combination available, thus or
                   statements should be treated as separate cuts

            Thus an example cuts could be:

                   ['x > 5', 'y > 5*x']

    :param llabel: string, name of this cut

    :param xlim: tuple or list of floats, [x lower limit, x upper limit]

    :param ylim: tuple or list of floats, [y lower limits, y upper limit]

    :param crs: line colour for this cut (for plotting)

    :param kwargs: all other keyword arguments passed to plt.plot()

    :return f: matplotlib.axes, i.e. plt.subplot(111)
    """
    # make sure specific keyword args have
    kwargs['zorder'] = kwargs.get('zorder', 1)
    kwargs['linewidth'] = kwargs.get('linewidth', 1)
    kwargs['color'] = kwargs.get('color', crs)

    # plot cut lines
    for c in range(len(cuts)):
        # replace < and > with equals (we want to plot lines
        string = cuts[c].replace('<', '=').replace('>', '=')
        # calculate line across x y space (use limits if not None)
        if xlim is None:
            x = np.linspace(np.min(xx), np.max(xx), 10000)
        else:
            x = np.linspace(xlim[0], xlim[1], 10000)
        if ylim is None:
            y = np.linspace(np.min(yy), np.max(yy), 10000)
        else:
            y = np.linspace(ylim[0], ylim[1])
        # if we are working out x using y
        if 'x = ' in string:
            string = string.replace('x = ', '')
            newy = y
            if 'y' in string:
                newx = ne(string)
            else:
                newx = np.array(np.repeat(string, len(newy)), dtype=float)
        # else we are working out y using x
        elif 'y = ' in string:
            string = string.replace('y = ', '')
            newx = x
            if 'x' in string:
                newy = ne(string)
            else:
                newy = np.array(np.repeat(string, len(newx)), dtype=float)
        # if neither in string then skip (shouldn't happen)
        else:
            continue
        # now mask line by full ccut
        mask = mask_cut(newx, newy, cuts, include_border=True)
        # apply mask and plot
        if c == 0:
            label = llabel
        else:
            label = None
        f.plot(newx[mask], newy[mask], label=label, **kwargs)
    return f


def plot_comparison(f, comp, handles=None, labels=None, **kwargs):
    """

    This is a very specific plot. If one has x and y for a set of objects
    and has spectral type information and luminosity class information (and
    possibly binary information) then one can plot a representation of all
    objects based on their spectral type, luminosity class and whether they are
    a binary or not (see "comp" parameter)

    :param f: matplotlib.axes, i.e. plt.subplot(111)

    :param comp: list of variables or None, if None then do not plot
                    if a list then list must be in the form
                    compare = [xs, ys, ss, ls, bs]

                    where xs, ys are arrays of floats

                          ss, ls are array of strings, spt and luminosity class
                          respectively

                          bs is an array of booleans, to flag binary

                    (see plot_comparison function for more information)

            ss[i] = numerical spectral type
                        i.e. 0.0 = M0.0

            ls[i] = numerical luminosity class
                        i.e. -100 = sub dwarf
                                0 = main sequence/dwarf
                              100 = giant/subgiant

            bs[i] = boolean  if True then binary if False then single star

    :param handles: handles from ax.get_legend_handles_labels()
                    i.e. a list of matplotlib legend objects

    :param labels: labels from ax.get_legend_handles_labels()
                   i.e. a list of strings (label text)

    :param kwargs: all other keyword arguments passed to plt.plot()


    WARNING: this functions main use is with SET kawrgs
             overriding them will override the main functino of the plot

    plotting properties are as follows:


    for luminosity class determines size
        lumorder = [-100, 0, 100]
        lumfmt = {100: 10, 0: 5, -100: 2}
        lumstr = {100: 'Giants', 0: 'Main Sequence', -100: 'Subdwarfs'}

    spectral subclass determines marker
        subtypeorder = [0, 2, 4, 6, 8]
        sptsubfmt = {0: 'o', 1: 'o', 2: 's', 3: 's', 4: 'd', 5: 'd',
                     6: '^', 7: '^', 8: 'v', 9: 'v'}
        sptsubstr = {0: r'$X0-X2$', 2: r'$X2-X4$', 4: r'$X4-X6$', 6: r'$X6-X8$',
                     8: r'$X8+$'}
        sptsubalp = {0: 0.40, 1: 0.40, 2: 0.55, 3: 0.55, 4: 0.70, 5: 0.70,
                     6: 0.85, 7: 0.85, 8: 1.00, 9: 1.00}

    spectral class determines colour
        spts_inorder = ['O', 'B', 'A', 'F', 'G', 'K', 'M', 'L', 'T', 'Y']
        sptclassfmt = {'O': 'red', 'B': 'crimson', 'A': 'darkorange',
                       'F': 'darkgreen', 'G': 'lawngreen',
                       'K': 'palegreen', 'M': 'blue', 'L': 'saddlebrown',
                       'T': 'silver', 'Y': 'gray'}

    :return:
    """
    # make sure certain kwargs are set
    kwargs['zorder'] = kwargs.get('zorder', 1)

    # plotting properties
    # luminosity class determines size
    lumorder = [-100, 0, 100]
    lumfmt = {100: 10, 0: 5, -100: 2}
    lumstr = {100: 'Giants', 0: 'Main Sequence', -100: 'Subdwarfs'}
    # spectral subclass determines marker
    subtypeorder = [0, 2, 4, 6, 8]
    sptsubfmt = {0: 'o', 1: 'o', 2: 's', 3: 's', 4: 'd', 5: 'd',
                 6: '^', 7: '^', 8: 'v', 9: 'v'}
    sptsubstr = {0: r'$X0-X2$', 2: r'$X2-X4$', 4: r'$X4-X6$', 6: r'$X6-X8$',
                 8: r'$X8+$'}
    # sptsubalp = {0: 0.40, 1: 0.40, 2: 0.55, 3: 0.55, 4: 0.70, 5: 0.70,
    #              6: 0.85, 7: 0.85, 8: 1.00, 9: 1.00}
    # spectral class determines colour
    spts_inorder = ['O', 'B', 'A', 'F', 'G', 'K', 'M', 'L', 'T', 'Y']
    sptclassfmt = {'O': 'red', 'B': 'crimson', 'A': 'darkorange',
                   'F': 'darkgreen', 'G': 'lawngreen',
                   'K': 'palegreen', 'M': 'blue', 'L': 'saddlebrown',
                   'T': 'silver', 'Y': 'gray'}

    xs, ys, ss, ls, bs = comp
    # --------------------------------------------------------------------------
    # remove all nans
    m1 = np.isfinite(xs) & np.isfinite(ys) & np.isfinite(ss) & np.isfinite(ls)
    xs, ys, ss, ls, bs = xs[m1], ys[m1], ss[m1], ls[m1], bs[m1]
    # --------------------------------------------------------------------------
    # loop round each point (will have to plot all points separately to set
    # correct format
    for i in tqdm(range(len(xs))):
        # ----------------------------------------------------------------------
        # get iterated values
        x, y, s, l, b = xs[i], ys[i], ss[i], ls[i], bs[i]
        # based on spt work out spt class and spt subclass
        sptclass, sptsubclass = get_spt_class_subclass(s)
        # ----------------------------------------------------------------------
        # set format depending on what we have
        # set marker based on spt subclass (integer spts only, round X.5 down)
        marker = sptsubfmt[np.floor(sptsubclass)]
        # alpha = sptsubalp[np.floor(sptsubclass)]
        # set colour based on spt class
        colour = sptclassfmt[sptclass]
        # set size from luminosity class
        size = lumfmt[l]
        # ----------------------------------------------------------------------
        # set kwargs for plot based on internal definitions
        kwargset = dict()
        kwargset['marker'] = kwargs.get('marker', marker)
        kwargset['mec'] = kwargs.get('mec', colour)
        kwargset['ms'] = kwargs.get('ms', size)
        kwargset['markeredgewidth'] = kwargs.get('markeredgewidth', 1.5)
        kwargset['linestyle'] = kwargs.get('linestyle', '')
        kwargset['zorder'] = kwargs.get('zorder', 1)
        kwargset['alpha'] = kwargs.get('alpha', 0.5)

        if b:
            kwargset['mfc'] = colour
        else:
            kwargset['mfc'] = 'none'
        # ----------------------------------------------------------------------
        # plot
        f.plot(x, y, **kwargset)
    # --------------------------------------------------------------------------
    # make a completely custom axis
    # --------------------------------------------------------------------------
    kwargsets = []
    # --------------------------------------------------------------------------
    # deal with lum class variable (size)
    for lkey in lumorder:
        kwargset = dict()
        kwargset['color'] = kwargs.get('color', 'k')
        kwargset['marker'] = kwargs.get('marker', 'o')
        kwargset['mec'] = kwargs.get('mec', 'k')
        kwargset['mfc'] = kwargs.get('mfc', 'none')
        kwargset['ms'] = kwargs.get('ms', lumfmt[lkey])
        kwargset['linestyle'] = kwargs.get('linestyle', '')
        kwargset['markeredgewidth'] = kwargs.get('markeredgewidth', 1.5)
        kwargsets.append(kwargset)
        labels.append(lumstr[lkey])
    # --------------------------------------------------------------------------
    # deal with spt sub class variable (marker)
    for skey in subtypeorder:
        kwargset = dict()
        kwargset['color'] = kwargs.get('color', 'k')
        kwargset['marker'] = kwargs.get('marker', sptsubfmt[skey])
        kwargset['mec'] = kwargs.get('mec', 'k')
        kwargset['mfc'] = kwargs.get('mfc', 'none')
        kwargset['ms'] = kwargs.get('ms', 5)
        kwargset['linestyle'] = kwargs.get('linestyle', '')
        kwargset['markeredgewidth'] = kwargs.get('markeredgewidth', 2)
        kwargsets.append(kwargset)
        labels.append(sptsubstr[skey])
    # --------------------------------------------------------------------------
    # deal with spt class variable (colour)
    for skey in spts_inorder:
        kwargs = dict(color=sptclassfmt[skey], mec=sptclassfmt[skey], ms=5,
                      marker='o', mfc='none', linestyle='', markeredgewidth=1.5)
        kwargsets.append(kwargs)
        labels.append('Spectral type = {0}'.format(skey))
    # --------------------------------------------------------------------------
    # deal with binaries
    kwargset = dict()
    kwargset['color'] = kwargs.get('color', 'k')
    kwargset['marker'] = kwargs.get('marker', 'o')
    kwargset['linestyle'] = kwargs.get('linestyle', '')
    kwargset['markeredgewidth'] = kwargs.get('markeredgewidth', 1.5)

    kwargsets.append(kwargset)
    labels.append('Filled shapes are binaries')
    # --------------------------------------------------------------------------
    for kwargset in kwargsets:
        h = plt.Line2D((0, 1), (0, 0), **kwargset)
        handles.append(h)
    return f, handles, labels


# ==============================================================================
# Define internally used functions
# ==============================================================================

# plot a single contour plot (for use in plot_contours)
def plot_cont(x, y, binx=None, biny=None, res=None, levels=None, fillc=None):
    """
    ===========================================================================
    2D Contour plot
    ===========================================================================

    returns a 2d histogram for a contour plot based on x and y values (must be
    the same length) with the additional options to set bin widths in both
    x (binx) and y (biny) directions - default is 128

    returns H the 2D histogram function, the extent, levels and colours of each

    :param x: array of floats, the x data
    :param y: array of floats, the y data
    :param binx: the number of bins to use in the x direction
    :param biny: the number of bins to use in the y direction
    :param res: number of levels of contours (i.e contour resolution)
    :param levels: [level0, level1, ..., leveln]

                    A list of floating point numbers indicating the level
                    curves to draw; eg to draw just the zero contour pass
                    ``levels=[0]``
    :param fillc: the colourmap to use for the coloring of the levels if None
                  a grey colour map is used

    """
    if binx is None:
        binx, biny = 128, 128
    elif biny is None:
        biny = binx
    H, xedges, yedges = np.histogram2d(y, x, bins=(binx, biny))
    # H.shape, xedges.shape, yedges.shape
    extent = [yedges[0], yedges[-1], xedges[0], xedges[-1]]
    # sort out levels-----------------------------------------------
    # nan_mask = (H == H) & (H != float('inf'))
    minimum, maximum = 1, H.max()
    low = int(round(np.log10(minimum)))
    high = int(round(np.log10(maximum)))
    # --------------------------------------------------------------
    # define levels
    if levels is not None:
        levels = list(levels)
    elif res is not None:
        levels = np.arange(0, maximum, res)
    else:
        levels = []
        for i in range(low, high):
            levels.append(pow(10, low + i))
            levels.append(5 * pow(10, low + i))
        if len(levels) <= 4:
            levels = []
            for i in range(low, high):
                levels.append(pow(10, low + i))
                levels.append(2 * pow(10, low + i))
                levels.append(5 * pow(10, low + i))
    # --------------------------------------------------------------
    # add colours based on number of levels
    if fillc is None:
        cmap = 'Greys'
    else:
        cmap = fillc
    colours = []
    cNorm = colors.Normalize(vmin=-2, vmax=len(levels))
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)
    for j in range(len(levels)):
        colorVal = scalarMap.to_rgba(j)
        colours.append(colorVal)
    # --------------------------------------------------------------
    return H, extent, levels, colours


def get_outliers(im, x, y):
    """
    From a matplotlib.contour.QuadContourSet (im) find those points in x and y
    that lay outside the contours
    :param im: matplotlib.contour.QuadContourSet from e.g. plt.contourf()
    :param x: array of float, horizontal axis values
    :param y: array of float, vertical axis values
    :return:
    """
    p, pin = make_polygons(im)
    inmask = mask_from_polygons(x, y, p, pin)
    return -inmask


def make_polygons(im):
    """
    Makes a set of polygons from matplotlib.contour.QuadContourSet (im)
    and returns a list of polygon objects, a polygon collection
    :param im: matplotlib.contour.QuadContourSet from e.g. plt.contourf()
    :return polygons: the list or array of polygons, each polygon should be
                      a list of arrays that contain a set of vertices each
                      with a list of (x, y) coordinates:

                      polygons = [polygon1, polygon2, ... , polygonN]

                      where:

                      polygon1 = [vertexlist1, vertexlist2, ..., vertexlistN]

                      vertexlist1 = [(x0, y0), (x1, y1), ...., (xN, yN)]

                      i.e. a single polygon (a square) could be:
                          [[[(0, 0), (1, 0), (1, 1), (0, 1)]]]

    :return polygons_in: Array of bools Same shape as polygons
                         (except no vertices). This controls whether a polygon
                         is "inside" another polygon if include_holes is False
                         we assume any polygon with polygon_in = True is a hole
                         and thus the count should take these points as NOT
                         being in the polygons.

                         polygons_in = None

                         polygons_in = [polygon_in1, polygon_in2, ...,
                                       polygon_inN]

                         where:

                         polygon_in = [True, False, ..., True]
    """
    import matplotlib.path as mplPath
    # get outer contour and plot outliers
    polygons = np.zeros(len(im.collections), dtype=object)
    # define threshold in x and y directions
    # loop around contour object collections
    for i in range(len(im.collections)):
        cs = im.collections[i]
        polygon = []
        # loop around contour object collection paths
        for j in range(len(cs.get_paths())):
            cspath = cs.get_paths()[j]
            polygon.append(cspath.vertices)
        polygons[i] = polygon
    # check if any of the polygons are contained within another polygon
    polygons_in = np.zeros(len(im.collections), dtype=object)
    for k in range(len(polygons)):
        polygon, polygon_in = polygons[k], []
        for i in range(len(polygon)):
            poly1 = mplPath.Path(polygon[i])
            contained = False
            for j in range(len(polygon)):
                if i != j:
                    poly2 = mplPath.Path(polygon[j])
                    contained |= bool(poly2.contains_path(poly1))
            polygon_in.append(contained)
        polygons_in[k] = polygon_in
    return polygons, polygons_in


def mask_from_polygons(xarr, yarr, polygons, polygons_in):
    """
    Takes a polygons set, and a polygon inside set with some x points and
    y points and returns a mask of of any inside the polygons set
    :param xarr: array of floats, x data points
    :param yarr: array of floats, y data poitns
    :param polygons: list of polygon objects, a polygon collection: this is
                     the list or array of polygons, each polygon should be
                     a list of arrays that contain a set of vertices each
                     with a list of (x, y) coordinates:

                     polygons = [polygon1, polygon2, ... , polygonN]

                     where:

                     polygon1 = [vertexlist1, vertexlist2, ..., vertexlistN]

                     vertexlist1 = [(x0, y0), (x1, y1), ...., (xN, yN)]

                     i.e. a single polygon (a square) could be:
                         [[[(0, 0), (1, 0), (1, 1), (0, 1)]]]
    :param polygons_in: Array of bools Same shape as polygons
                        (except no vertices). This controls whether a polygon
                        is "inside" another polygon if include_holes is False
                        we assume any polygon with polygon_in = True is a hole
                        and thus the count should take these points as NOT
                        being in the polygons.

                        polygons_in = None

                        polygons_in = [polygon_in1, polygon_in2, ...,
                                       polygon_inN]

                        where:

                        polygon_in = [True, False, ..., True]
    :return: mask of poitns inside polygons (len of xarr and yarr)
    """
    import matplotlib.path as mplPath
    xyarr = np.array(zip(xarr, yarr))
    falsearray = np.array([False] * len(xarr), dtype=bool)
    insideany = falsearray.copy()
    for k in range(len(polygons)):
        polygon = polygons[k]
        # +++++++++++++++++++++++++++++++++++++++++++++
        # deal with annoying contained polygons
        if polygons_in is None:
            polygon_in = None
        else:
            polygon_in = polygons_in[k]
        # +++++++++++++++++++++++++++++++++++++++++++++
        for j in range(len(polygon)):
            poly = polygon[j]
            # mask out the points outside the poly clip box
            pmax_x, pmax_y = np.max(poly, axis=0)
            pmin_x, pmin_y = np.min(poly, axis=0)
            mask1 = (xarr > pmin_x) & (xarr < pmax_x)
            mask2 = (yarr > pmin_y) & (yarr < pmax_y)
            mask = mask1 & mask2
            # if no points inside poly clip box then don't bother counting
            if len(mask[mask]) == 0:
                continue
            # -----------------------------------------------------------------
            # deal with annoying contained polygons
            if polygon_in is None:
                poly_in = False
            else:
                poly_in = polygon_in[j]
            # -----------------------------------------------------------------
            # Creates a mask for points (xs, ys) based on whether they are
            # inside a polygon poly from http://stackoverflow.com/a/23453678
            if not poly_in:
                bbPath = mplPath.Path(poly)
                inside = falsearray.copy()
                inside[mask] = bbPath.contains_points(xyarr[mask])
                # --------------------------------------------------------------
                # if polygon is inside another polygon (as defined by poly_in)
                # do not count it as inside
                insideany |= inside
            # -----------------------------------------------------------------
    return insideany


def mask_cut(xx, yy, cuts, include_border=False):
    """

    :param xx: array of float, x points

    :param yy: array of floats, y points

    :param cuts: list of cuts in the form:

                   cuts = [cut1, cut2, ..., cutn]

                   where cut1 = string logic statement

            if one wants a cut of x > 5 cut1 would be as follows:

                   cut1 = 'x > 5'

                   one may also use y and x as variables i.e.:

                   cut1 = 'y > 5*x'

            The logic is as follows:

                   if cuts = [cut1a, cut1b, cut1c]

                   the cuts produced are as follows:

                   cut1 = cut1a & cut1b & cut1c

                   currently and is the only combination available, thus or
                   statements should be treated as separate cuts

            Thus an example cuts could be:

                   ['x > 5', 'y > 5*x']

    :param include_border: bool, if True replaces '>' with '>=' and
                           '<' with '<=' i.e. includes the values 'on the edge'

    :return mask: array of booleans, where cuts are True (length equal to xx)
    """
    emsg = 'Error: {0} does is not a list or array'
    if not hasattr(xx, '__len__'):
        raise Exception(emsg.format('xx'))
    if not hasattr(yy, '__len__'):
        raise Exception(emsg.format('yy'))
    mask = np.ones(len(xx), dtype=bool)
    for c in range(len(cuts)):
        string = cuts[c].replace('x', 'xx').replace('y', 'yy')
        if include_border:
            string = string.replace('>', '>=').replace('<', '<=')
        mask &= ne(string)
    return mask


def get_spcs():
    """
    Defines the SIMBAD Spectral classes and then sorts them largest to
    smallest

    from http://simbad.u-strasbg.fr/simbad/sim-display?data=sptypes
    :return:
    """
    # define string spectral type convertion i.e. M0.0 --> 0.0
    spckey = ['O', 'B', 'A', 'F', 'G', 'K', 'M', 'L', 'T', 'Y']
    spcval = [-60, -50, -40, -30, -20, -10, 0, 10, 20, 30]
    # define uncertain spts as the lower spectral type
    spckey1 = ['o-b', 'o/b', 'b-a', 'b/a', 'a-f', 'a/f', 'f-g', 'f/g',
               'g-k', 'g/k', 'k-m', 'k/m', 'm-l', 'm/l', 'l-t', 'l/t', 't-y',
               't/y']
    spcval1 = [-50, -50, -40, -40, -30, -30, -20, -20, -10, -10, 0, 0, 10, 10,
               20, 20, 30, 30]
    spckey, spcval = np.array(spckey), np.array(spcval)
    spckey1, spcval1 = np.array(spckey1), np.array(spcval1)
    mask = sort_str_array_by_length(spckey)
    mask1 = sort_str_array_by_length(spckey1)
    return spckey[mask], spcval[mask], spckey1[mask1], spcval1[mask1]


def sort_str_array_by_length(array, first='longest'):
    """
    Takes a list or array of strings (array) and returns a mask of the sorting
    order (sorted by the length of each string) if first=shortest then shortest
    first else longest first
    :param array: list or array of string objects (or any object with length)
    :param first: if shortest returned mask has shortest first, else longest
    :return: sorting mask (longest to shortest or shortest to longest string)
    """
    lengths = []
    for row in array:
        lengths = np.append(lengths, [len(row)])
    if first == 'shortest':
        return np.argsort(lengths)
    else:
        return np.argsort(lengths)[::-1]


def get_spt_class_subclass(x):
    """
    Turn a numerical spectral type into a class and subclass

    i.e. 0.0 --> M0.0    -22.5 --> F7.5       22.5 --> T2.5

    :param x: float, numerical spectral type
    :return:
    """
    # get spectral type conversion information
    spckey, spcval, spckey1, spcval1 = get_spcs()
    # convert to dict
    spc = dict(zip(spcval, spckey))
    # find the nearest spectral type class to x
    nclass = np.floor(x / 10.0) * 10
    # use the spc dictionary to select this spectral class string
    sclass = spc[nclass]
    # spectral subclass is just the remainder
    ssubclass = x - nclass
    # return spectral class and subclass
    return sclass, ssubclass


# ==============================================================================
# Start of code
# ==============================================================================
if __name__ == '__main__':
    # -------------------------------------------------------------------------
    # Example use of cut plot
    # -------------------------------------------------------------------------
    vj_names = ['$V-J > 4$ and $V > 15$ cut', r'$V-J>2.7$ cut']
    vj_cuts = [['x > 4', 'y > 15'], ['x > 2.7']]
    vj_logic = [['&']]
    vj_colours = ['r', 'g']
    vj_limits = [[0, 8], [25, 5]]
    vj_labels = ['$V-J$', '$V$']
    vj_compare = True
    # -------------------------------------------------------------------------
    # make some fake data
    # -------------------------------------------------------------------------
    v = np.random.normal(15, 2, size=10000)
    vj = np.random.normal(4.0, 1.0, size=10000)
    ev, evj = 0.1*np.sqrt(v), 0.1*np.sqrt(vj)
    clevels = [2, 5, 10, 50]
    vc = np.random.normal(15, 2, size=100)
    vjc = np.random.normal(4.00, 1.00, size=100)
    numspt = np.random.choice(range(-60, 30) + range(0, 10)*5, size=100)
    lumspt = np.random.choice([-100, 0, 0, 0, 0, 0, 0, 100], size=100)
    binaryflag = np.random.choice([True, False], size=100)
    # -------------------------------------------------------------------------
    # plot V-J vs V band
    # -------------------------------------------------------------------------
    print ('\n Plotting V-J vs V band...')
    psave = None
    cut_plot(vj, v, evj, ev, vj_cuts, vj_logic, vj_names,
             vj_colours, vj_labels[0], vj_labels[1], psave, vj_limits, clevels,
             compare=[vjc, vc, numspt, lumspt, binaryflag])

# ------------------------------------------------------------------------------

# ==============================================================================
# End of code
# ==============================================================================
