# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

from tempfile import TemporaryFile
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib import rcParams, cm
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.axes_grid1 import ImageGrid
import scipy.stats.mstats as mstats
import numpy as np
import numpy.ma as ma

import ocw.utils as utils

# Set the default colormap to coolwarm
mpl.rc('image', cmap='coolwarm')


def set_cmap(name):
    '''
    Sets the default colormap (eg when setting cmap=None in a function)
    See: http://matplotlib.org/examples/pylab_examples/show_colormaps.html
    for a list of possible colormaps.
    Appending '_r' to a matplotlib colormap name will give you a reversed
    version of it.
    :param name: The name of the colormap.
    :type name: :mod:`string`
    '''
    # The first line is redundant but it prevents the user from setting
    # the cmap rc value improperly
    cmap = plt.get_cmap(name)
    mpl.rc('image', cmap=cmap.name)


def _nice_intervals(data, nlevs):
    '''
    Purpose::
        Calculates nice intervals between each color level for colorbars
        and contour plots. The target minimum and maximum color levels are
        calculated by taking the minimum and maximum of the distribution
        after cutting off the tails to remove outliers.
    Input::
        data - an array of data to be plotted
        nlevs - an int giving the target number of intervals
    Output::
        clevs - A list of floats for the resultant colorbar levels
    '''
    # Find the min and max levels by cutting off the tails of the distribution
    # This mitigates the influence of outliers
    data = data.ravel()
    mn = mstats.scoreatpercentile(data, 5)
    mx = mstats.scoreatpercentile(data, 95)
    # if min less than 0 and or max more than 0 put 0 in center of color bar
    if mn < 0 and mx > 0:
        level = max(abs(mn), abs(mx))
        mnlvl = -1 * level
        mxlvl = level
    # if min is larger than 0 then have color bar between min and max
    else:
        mnlvl = mn
        mxlvl = mx

    # hack to make generated intervals from mpl the same for all versions
    autolimit_mode = mpl.rcParams.get('axes.autolimit_mode')
    if autolimit_mode:
        mpl.rc('axes', autolimit_mode='round_numbers')

    locator = mpl.ticker.MaxNLocator(nlevs)
    clevs = locator.tick_values(mnlvl, mxlvl)
    if autolimit_mode:
        mpl.rc('axes', autolimit_mode=autolimit_mode)

    # Make sure the bounds of clevs are reasonable since sometimes
    # MaxNLocator gives values outside the domain of the input data
    clevs = clevs[(clevs >= mnlvl) & (clevs <= mxlvl)]
    return clevs


def _best_grid_shape(nplots, oldshape):
    '''
    Purpose::
        Calculate a better grid shape in case the user enters more columns
        and rows than needed to fit a given number of subplots.
    Input::
        nplots - an int giving the number of plots that will be made
        oldshape - a tuple denoting the desired grid shape (nrows, ncols) for arranging
                    the subplots originally requested by the user.
    Output::
        newshape - the smallest possible subplot grid shape needed to fit nplots
    '''
    nrows, ncols = oldshape
    size = nrows * ncols
    diff = size - nplots
    if diff < 0:
        raise ValueError(
            'gridshape=(%d, %d): Cannot fit enough subplots for data' % (nrows, ncols))
    else:
        # If the user enters an excessively large number of
        # rows and columns for gridshape, automatically
        # correct it so that it fits only as many plots
        # as needed
        while diff >= nrows:
            ncols -= 1
            size = nrows * ncols
            diff = size - nplots

        # Don't forget to remove unnecessary columns too
        if ncols == 1:
            nrows = nplots

        newshape = nrows, ncols
        return newshape


def _fig_size(gridshape, aspect=None):
    '''
    Purpose::
        Calculates the figure dimensions from a subplot gridshape
    Input::
        gridshape - Tuple denoting the subplot gridshape
        aspect - Float denoting approximate aspect ratio of each subplot
                 (width / height). Default is 8.5 / 5.5
    Output::
        width - float for width of the figure in inches
        height - float for height of the figure in inches
    '''
    if aspect is None:
        aspect = 8.5 / 5.5

    # Default figure size is 8.5" x 5.5" if nrows == ncols
    # and then adjusted by given aspect ratio
    nrows, ncols = gridshape
    if nrows >= ncols:
        # If more rows keep width constant
        width, height = (aspect * 5.5), 5.5 * (nrows // ncols)
    else:
        # If more columns keep height constant
        width, height = (aspect * 5.5) * (ncols // nrows), 5.5

    return width, height

import yaml
import sys
config_file = str(sys.argv[1])
config = yaml.safe_load(open(config_file))

# reg=config['region']
# if config['spatial_season?']:
    # sn= 'spatial_'+config['spatial_season']['season_name']
# if config['temporal_annual?']:
    # sn= 'temporal_'+config['temporal_annual']['season_name']
# if config['temporal_annual_cycle?']:
    # sn= 'temporal_'+config['temporal_annual_cycle']['season_name']
# if not config['spatial_season?'] and  not config['temporal_annual?']\
    # and  not config['temporal_annual_cycle?']: 
    # sn ='cekFileName'  
# print('sn_plotter:',sn)

#tipe= config['Metric_type']
#title = sn+'_'+reg+'_'+'_'+tipe
def draw_taylor_diagram(results, names, refname, fname, fmt='png',
                        gridshape=(1, 1), ptitle='', subtitles=None,
                        pos='upper right', frameon=True, radmax=1.5, 
                        legend_size=13):
    ''' Draw a Taylor diagram.
    :param results: An Nx2 array containing normalized standard deviations,
       correlation coefficients, and names of evaluation results.
    :type results: :class:`numpy.ndarray`
    :param names: A list of names for each evaluated dataset
    :type names: :class:`list` of :mod:`string`
    :param refname: The name of the reference dataset.
    :type refname: :mod:`string`
    :param fname: The filename of the plot.
    :type fname: :mod:`string`
    :param fmt: (Optional) filetype for the output plot.
    :type fmt: :mod:`string`
    :param gridshape: (Optional) Tuple denoting the desired grid shape
        (num_rows, num_cols) for arranging the subplots.
    :type gridshape: A :class:`tuple` of the form (num_rows, num_cols)
    :param ptitle: (Optional) plot title.
    :type ptitle: :mod:`string`
    :param subtitles: (Optional) list of strings specifying the title for each
        subplot.
    :type subtitles: :class:`list` of :mod:`string`
    :param pos: (Optional) string or tuple of floats used to set the position
        of the legend. Check the `Matplotlib docs <http://matplotlib.org/api/legend_api.html#matplotlib.legend.Legend>`_
        for additional information.
    :type pos: :mod:`string` or :func:`tuple` of :class:`float`
    :param frameon: (Optional) boolean specifying whether to draw a frame
        around the legend box.
    :type frameon: :class:`bool`
    :param radmax: (Optional) float to adjust the extent of the axes in terms of
        standard deviation.
    :type radmax: :class:`float`
    :param legend_size: (Optional) float to control the font size of the legend        
    :type legend_size: :class:`float`
    '''
    import string, math
    string_list = list(string.ascii_uppercase) 
    #max_std=math.ceil(np.max(results))
    max_std=np.max(results)
    print('max_std=',max_std)
    #untuk SEA ini radmax =rd terlalu kecil maka
    radmax=max_std
    #radmax=2
    
    # Handle the single plot case.
    if results.ndim == 2:
        results = results.reshape(1, *results.shape)

    # Make sure gridshape is compatible with input data
    nplots = results.shape[0]
    gridshape = _best_grid_shape(nplots, gridshape)

    # Set up the figure
    fig = plt.figure()
    fig.subplots_adjust(left=.3)
    #fig.set_size_inches((8.5, 11))
    fig.dpi = 300
    for i, data in enumerate(results):
        rect = gridshape + (i + 1,)
        # Convert rect to string form as expected by TaylorDiagram constructor
        rect = int(str(rect[0]) + str(rect[1]) + str(rect[2]))

        # Create Taylor Diagram object
        dia = TaylorDiagram(1, fig=fig, rect=rect,
                            label=refname, radmax=radmax)
        for i, (stddev, corrcoef) in enumerate(data):
            #ini untuk 2 obs dan mmew=1
            #print(i)
            #print(names)
            #---angka
            #dia.add_sample(stddev, corrcoef, marker='$%d$' % (i + 1), ms=12,
            #---huruf 
            #dia.add_sample(stddev, corrcoef, marker="$"+string_list[i]+"$", 
            #mfc='black', mec='black',
            #ms=8, label=names[i])
            #if subtitles is not None:
            #    dia._ax.set_title(subtitles[i])
            #--5obs angka dan model huruf
            
            #ini untuk 1 obs pembanding dan mmew>1
            fontsize=7
            if i<1:
                dia.add_sample(stddev, corrcoef,
                marker='$%d$' % (i+1), ms=fontsize, ls='',
                #marker = "$"+string_list[i]+"$", ms=fontsize, ls='',    
                #marker='p', ms=fontsize, ls='',                 
                mfc='black', mec='black',
                label=names[i])
                
            elif i==10:
                fontsize=5
                dia.add_sample(stddev, corrcoef,
                #marker='$%d$' % (i+1-5), ms=fontsize, ls='',
                marker = "$"+string_list[i-1]+"$", ms=fontsize, ls='',    
                #marker='p', ms=fontsize, ls='',                 
                mfc='b', mec='b',
                label=names[i]) 
                
            elif i>10:
                fontsize=5
                dia.add_sample(stddev, corrcoef,
                #marker='$%d$' % (i+1-5), ms=fontsize, ls='',
                marker = "$"+string_list[i-1]+"$", ms=fontsize, ls='',    
                #marker='p', ms=fontsize, ls='',                 
                mfc='r', mec='r',
                label=names[i]) 
            
            else:
                dia.add_sample(stddev, corrcoef,
                #marker='$%d$' % (i+1-5), ms=fontsize, ls='',
                marker = "$"+string_list[i-1]+"$", ms=fontsize, ls='',    
                #marker='p', ms=fontsize, ls='',                 
                mfc='black', mec='black',
                label=names[i])
                
    #tambahan
    # Add RMS contours, and label them
    dia.add_rms_contours()
    # Add grid
    dia.add_grid()
    
    # Add legend
    
    legend = fig.legend(dia.samplePoints,
                        [p.get_label() for p in dia.samplePoints],
                        handlelength=0., prop=dict(size='small'), numpoints=1,
                        loc='right', bbox_to_anchor=(0.6, 0.2, 0.5, 0.5))
    legend.draw_frame(frameon)
    plt.subplots_adjust(wspace=0)

    # Add title and save the figure
    #try: sn+'_'+reg+'_'=sn1
    #except: sn+'_'+reg+'_'=sn2    
    #fig.suptitle(sn+'_'+reg+'_'+tipe)
    fig.suptitle(fname)
   
    plt.tight_layout()
    #plt.show()
    plt.show()
    print('plotter_2')
    print('saving taylor diagram...')
    #fig.savefig('%s.%s' % (fname, fmt), bbox_inches='tight', dpi=fig.dpi)
    fig.savefig(fname+'.png', bbox_inches='tight', dpi=fig.dpi)
    fig.clf()

    
    
def draw_taylor_diagram2(results, names, refname, fname, fmt='png',
                        gridshape=(1, 1), ptitle='', subtitles=None,
                        pos='upper right', frameon=True, radmax=1.5, 
                        legend_size=13):
  
    import string, math
    string_list = list(string.ascii_uppercase) 
    max_std=math.ceil(np.max(results))
    print('max_std=',max_std)
    #untuk SEA ini radmax =rd terlalu kecil maka
    radmax=max_std
    
    # Handle the single plot case.
    if results.ndim == 2:
        results = results.reshape(1, *results.shape)

    # Make sure gridshape is compatible with input data
    nplots = results.shape[0]
    gridshape = _best_grid_shape(nplots, gridshape)

    # Set up the figure
    fig = plt.figure()
    fig.subplots_adjust(left=.3)
    #fig.set_size_inches((8.5, 11))
    fig.dpi = 300
    for i, data in enumerate(results):
        rect = gridshape + (i + 1,)
        # Convert rect to string form as expected by TaylorDiagram constructor
        rect = int(str(rect[0]) + str(rect[1]) + str(rect[2]))

        # Create Taylor Diagram object
        dia = TaylorDiagram(1, fig=fig, rect=rect,
                            label=refname, radmax=radmax)
        for i, (stddev, corrcoef) in enumerate(data):
            #print(i)
            #print(names)
            #---angka
            #dia.add_sample(stddev, corrcoef, marker='$%d$' % (i + 1), ms=12,
            #---huruf 
            dia.add_sample(stddev, corrcoef, marker="$"+string_list[i]+"$", 
            mfc='black', mec='black',
            ms=8, label=names[i])
            #if subtitles is not None:
            #    dia._ax.set_title(subtitles[i])
            
            '''
            #Untuk 4obs! angka dan model huruf
            #1 as ref so 3obs+moe+9model=13 and i[0-12] => mme i=13 
            fontsize=7
            #Beri MME warna hitam for non models
            if i<4:
                dia.add_sample(stddev, corrcoef,
                marker='$%d$' % (i+1), ms=fontsize, ls='',
                #marker = "$"+string_list[i]+"$", ms=fontsize, ls='',    
                #marker='p', ms=fontsize, ls='',                 
                mfc='black', mec='black',
                label=names[i])
            #Beri MME warna biru 
            elif i==13:
                fontsize=5
                dia.add_sample(stddev, corrcoef,
                #marker='$%d$' % (i+1-4), ms=fontsize, ls='',
                marker = "$"+string_list[i-4]+"$", ms=fontsize, ls='',    
                #marker='p', ms=fontsize, ls='',                 
                mfc='b', mec='b',
                label=names[i]) 
            #Beri MMEW warna merah    
            elif i>13:
                fontsize=5
                dia.add_sample(stddev, corrcoef,
                #marker='$%d$' % (i+1-4), ms=fontsize, ls='',
                marker = "$"+string_list[i-4]+"$", ms=fontsize, ls='',    
                #marker='p', ms=fontsize, ls='',                 
                mfc='r', mec='r',
                label=names[i]) 
            
            else:
                dia.add_sample(stddev, corrcoef,
                #marker='$%d$' % (i+1-4), ms=fontsize, ls='',
                marker = "$"+string_list[i-4]+"$", ms=fontsize, ls='',    
                #marker='p', ms=fontsize, ls='',                 
                mfc='black', mec='black',
                label=names[i])
            '''   
    #tambahan
    # Add RMS contours, and label them
    dia.add_rms_contours()
    # Add grid
    dia.add_grid()
    
    # Add legend
    
    legend = fig.legend(dia.samplePoints,
                        [p.get_label() for p in dia.samplePoints],
                        handlelength=0., prop=dict(size='small'), numpoints=1,
                        loc='right', bbox_to_anchor=(0.6, 0.2, 0.5, 0.5))
    legend.draw_frame(frameon)
    plt.subplots_adjust(wspace=0)

    # Add title and save the figure
    #try: sn+'_'+reg+'_'=sn1
    #except: sn+'_'+reg+'_'=sn2    
    #fig.suptitle(sn+'_'+reg+'_'+tipe)
    fig.suptitle(fname)
   
    plt.tight_layout()
    #plt.show()
    plt.show()
    print('plotter_2')
    print('saving taylor diagram2...')
    #fig.savefig('%s.%s' % (fname, fmt), bbox_inches='tight', dpi=fig.dpi)
    fig.savefig(fname+'.png', bbox_inches='tight', dpi=fig.dpi)
    fig.clf()



class TaylorDiagram2(object):
    """ Taylor diagram helper class
    Plot model standard deviation and correlation to reference (data)
    sample in a single-quadrant polar plot, with r=stddev and
    theta=arccos(correlation).
    This class was released as public domain by the original author
    Yannick Copin. You can find the original Gist where it was
    released at: https://gist.github.com/ycopin/3342888
    """

    def __init__(self, refstd, radmax=1.5, fig=None, rect=111, label='_'):
        """Set up Taylor diagram axes, i.e. single quadrant polar
        plot, using mpl_toolkits.axisartist.floating_axes. refstd is
        the reference standard deviation to be compared to.
        """

        from matplotlib.projections import PolarAxes
        import mpl_toolkits.axisartist.floating_axes as FA
        import mpl_toolkits.axisartist.grid_finder as GF

        self.refstd = refstd            # Reference standard deviati'on=',tr

        tr = PolarAxes.PolarTransform()
        #print('tr=',tr)
        
        # Correlation labels
        rlocs = np.concatenate((np.arange(10) / 10., [0.95, 0.99]))
        tlocs = np.arccos(rlocs)        # Conversion to polar angles
        gl1 = GF.FixedLocator(tlocs)    # Positions
        tf1 = GF.DictFormatter(dict(zip(tlocs, map(str, rlocs))))
        #print('rlocs=',rlocs)
        #print('tlocs=',tlocs)
        #print('gl1=',gl1)
        #print('tf1=',tf1)
        

        # Standard deviation axis extent
        self.smin = 0
        self.smax = radmax * self.refstd

        ghelper = FA.GridHelperCurveLinear(tr,
                                           extremes=(0, np.pi / 2,  # 1st quadrant
                                                     self.smin, self.smax),
                                           grid_locator1=gl1,
                                           tick_formatter1=tf1,
                                           )
        #print('ghelper=',ghelper)
        if fig is None:
            fig = plt.figure()

        ax = FA.FloatingSubplot(fig, rect, grid_helper=ghelper)
        fig.add_subplot(ax)   
        #plt.show()

        # Adjust axes
        ax.axis["top"].set_axis_direction("bottom")  # "Angle axis"
        ax.axis["top"].toggle(ticklabels=True, label=True)
        ax.axis["top"].major_ticklabels.set_axis_direction("top")
        ax.axis["top"].label.set_axis_direction("top")
        ax.axis["top"].label.set_text("Correlation")

        ax.axis["left"].set_axis_direction("bottom")  # "X axis"
        #ax.axis["left"].label.set_text("Standard deviation")

        ax.axis["right"].set_axis_direction("top")   # "Y axis"
        ax.axis["right"].toggle(ticklabels=True, label=True)
        #jika label=True label di Y muncul
        ax.axis["right"].major_ticklabels.set_axis_direction("left")
        #tambah
        ax.axis["right"].label.set_text("Standard deviation ratio")

        ax.axis["bottom"].set_visible(False)         # Useless
        ax.axis[:].major_ticks.set_tick_out(True)

        # Contours along standard deviations
        ax.grid(False)

        self._ax = ax                   # Graphical axes
        self.ax = ax.get_aux_axes(tr)   # Polar coordinates

        # Add reference point and stddev contour
        # print "Reference std:", self.refstd
        l, = self.ax.plot([0], self.refstd, 'k*',
                          ls='', ms=10, label=label)
        t = np.linspace(0, np.pi / 2)
        r = np.zeros_like(t) + self.refstd
        self.ax.plot(t, r, 'k--', label='_')

        # Collect sample points for latter use (e.g. legend)
        self.samplePoints = [l]

    def add_sample(self, stddev, corrcoef, *args, **kwargs):
        """Add sample (stddev,corrcoeff) to the Taylor diagram. args
        and kwargs are directly propagated to the Figure.plot
        command."""

        l, = self.ax.plot(np.arccos(corrcoef), stddev,*args, **kwargs)  # (theta,radius)
        self.samplePoints.append(l)

        return l

    def add_rms_contours(self, levels=5, **kwargs):
        """Add constant centered RMS difference contours."""

        rs, ts = np.meshgrid(np.linspace(self.smin, self.smax),
                             np.linspace(0, np.pi / 2))
        # Compute centered RMS difference
        rms = np.sqrt(self.refstd**2 + rs**2 - 2 *
                      self.refstd * rs * np.cos(ts))
        #print(rms)
        #contours = self.ax.contour(ts, rs, rms, levels, **kwargs)
        contours = self.ax.contour(ts, rs, rms, levels,colors='0.5', **kwargs)
        #plt.clabel(contours, contours.levels, inline=True, fmt='%.2f', fontsize=15)
        plt.clabel(contours, inline=1, fmt='%.2f', fontsize=10, colors='0.5')

    def add_stddev_contours(self, std, corr1, corr2, **kwargs):
        """Add a curved line with a radius of std between two points
        [std, corr1] and [std, corr2]"""

        t = np.linspace(np.arccos(corr1), np.arccos(corr2))
        r = np.zeros_like(t) + std
        return self.ax.plot(t, r, 'red', linewidth=2)

    def add_contours(self, std1, corr1, std2, corr2, **kwargs):
        """Add a line between two points
        [std1, corr1] and [std2, corr2]"""

        t = np.linspace(np.arccos(corr1), np.arccos(corr2))
        r = np.linspace(std1, std2)

        return self.ax.plot(t, r, 'red', linewidth=2)
    
    def add_grid(self, *args, **kwargs):
        """Add a grid."""

        self._ax.grid(*args, **kwargs)

#####
def draw_Sum_taylor_diagram(workdir):

    #temporal Annual_cycle_Sum__cycle
    import string
    string_list = list(string.ascii_uppercase) 

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    path='out2'

    sp=pd.read_excel('D:/Cordex/RCMES/'+path+'/Taylor_diagram_temporal_Annual_cycle_Sum__T.xlsx')
    sp4=np.array(sp)

    sp=pd.read_excel('D:/Cordex/RCMES/'+path+'/Taylor_diagram_temporal_Annual_cycle_Sum__Taylor.xlsx')
    sp1=np.array(sp)

    sp=pd.read_excel('D:/Cordex/RCMES/'+path+'/Taylor_diagram_temporal_Annual_cycle_Sum__CPI.xlsx')
    sp2=np.array(sp)

    sp=pd.read_excel('D:/Cordex/RCMES/'+path+'/Taylor_diagram_temporal_Annual_cycle_Sum__PI.xlsx')
    sp3=np.array(sp)

    sp=pd.read_excel('D:/Cordex/RCMES/'+path+'/Taylor_diagram_temporal_Annual_cycle_Sum__D.xlsx')
    sp5=np.array(sp)

    obsN=pd.read_excel('D:/Cordex/RCMES/'+path+'/Taylor_diagram_temporal_ref.xlsx')
    obsN=np.array(obsN)

    #set
    radmax=3.2
    adjust_fig=0

    fig = plt.figure()
    fig.subplots_adjust(left=adjust_fig)
    fig.dpi = 300
    dia = TaylorDiagram2(refstd=1, fig=fig, radmax=radmax, label=obsN[0,1])
    dia.samplePoints[0].set_color('r')  # Mark reference point as a red star
    #import string
    #from matplotlib.lines import Line2D
    #markers=Line2D.filled_markers
    # Add models to Taylor diagram
    for i, (name, stddev, corrcoef) in enumerate(sp4):                   
        if i>10:
            dia.add_sample(stddev, corrcoef,
            marker="$"+string_list[i]+"$", ms=10, ls='',
            mfc='black', mec='black',
            label=name)
        else:       
            dia.add_sample(stddev, corrcoef,
            marker="$"+string_list[i]+"$", ms=10, ls='',
            mfc='k', mec='k',
            label=name)

    for i, (name, stddev, corrcoef) in enumerate(sp1):

        if i>10:
            dia.add_sample(stddev, corrcoef,
            marker="$"+string_list[i]+"$",ms=10, ls='',
            mfc='green', mec='green',
            label=name)
        else:       
            dia.add_sample(stddev, corrcoef,
            marker="$"+string_list[i]+"$", ms=10, ls='',
            mfc='k', mec='k',
            label=name)
    for i, (name, stddev, corrcoef) in enumerate(sp2):
        if i>10:
            dia.add_sample(stddev, corrcoef,
            marker="$"+string_list[i]+"$", ms=10, ls='',
            mfc='blue', mec='blue',
            label=name)
        else:       
            dia.add_sample(stddev, corrcoef,
            marker="$"+string_list[i]+"$", ms=10, ls='',
            mfc='k', mec='k',
            label=name)

    for i, (name, stddev, corrcoef) in enumerate(sp3):                   
        if i>10:
            dia.add_sample(stddev, corrcoef,
            marker="$"+string_list[i]+"$", ms=10, ls='',
            mfc='red', mec='red',
            label=name)
        else:       
            dia.add_sample(stddev, corrcoef,
            marker="$"+string_list[i]+"$",ms=10, ls='',
            mfc='k', mec='k',
            label=name)

    for i, (name, stddev, corrcoef) in enumerate(sp5):                   
        if i>10:
            dia.add_sample(stddev, corrcoef,
            marker="$"+string_list[i]+"$", ms=10, ls='',
            mfc='purple', mec='purple',
            label=name)
        else:       
            dia.add_sample(stddev, corrcoef,
            marker="$"+string_list[i]+"$", ms=10, ls='',
            mfc='k', mec='k',
            label=name)
        # Add RMS contours, and label them
    contours = dia.add_rms_contours(levels=5)  # 5 levels in grey
    #plt.clabel(contours, inline=1, fontsize=10, fmt='%.1f')

    dia.add_grid()                                  # Add grid
    dia._ax.axis[:].major_ticks.set_tick_out(True)  # Put ticks outward

        #print('sp=',dia.samplePoints)
        # Add a figure legend and title
    fig.legend(dia.samplePoints,
    [ p.get_label() for p in dia.samplePoints[0:13] ],numpoints=1, 
    prop=dict(size='small'), loc='right', bbox_to_anchor=(0.52, 0.2, 0.45, 0.45))
    #fig.suptitle("Skills 9 rainfall model: Annual(black), JJA(red), DJF(blue)", size='x-large')  # Figure title
    fig.suptitle("Skills temporal of 9 model: Taylor(green), CPI(blue), PI(red), T(black), D(purple)", size='small')  # Figure title
    #plt.tight_layout(.05, .05)

    #plt.autoscale(enable=True) 
    #plt.axis('scaled')
    #fig.add_subplot(aspect='auto')
    #fig.tight_layout()
    fig.savefig(workdir+'Sum_taylor_diagram') #, bbox_inches='tight')
    #plt.show()

def draw_subregions(subregions, lats, lons, fname, fmt='png', ptitle='',
                    parallels=None, meridians=None, subregion_masks=None):
    ''' Draw subregion domain(s) on a map.
    :param subregions: The subregion objects to plot on the map.
    :type subregions: :class:`list` of subregion objects (Bounds objects)
    :param lats: Array of latitudes values.
    :type lats: :class:`numpy.ndarray`
    :param lons: Array of longitudes values.
    :type lons: :class:`numpy.ndarray`
    :param fname: The filename of the plot.
    :type fname: :mod:`string`
    :param fmt: (Optional) filetype for the output.
    :type fmt: :mod:`string`
    :param ptitle: (Optional) plot title.
    :type ptitle: :mod:`string`
    :param parallels: (Optional) :class:`list` of :class:`int` or :class:`float` for the parallels to
        be drawn. See the `Basemap documentation <http://matplotlib.org/basemap/users/graticule.html>`_
        for additional information.
    :type parallels: :class:`list` of :class:`int` or :class:`float`
    :param meridians: (Optional) :class:`list` of :class:`int` or :class:`float` for the meridians to
        be drawn. See the `Basemap documentation <http://matplotlib.org/basemap/users/graticule.html>`_
        for additional information.
    :type meridians: :class:`list` of :class:`int` or :class:`float`
    :param subregion_masks: (Optional) :class:`dict` of :class:`bool` arrays for each
        subregion for giving finer control of the domain to be drawn, by default
        the entire domain is drawn.
    :type subregion_masks: :class:`dict` of :class:`bool` arrays
    '''
    # Set up the figure
    fig = plt.figure()
    fig.set_size_inches((8.5, 11.))
    fig.dpi = 300
    ax = fig.add_subplot(111)

    # Determine the map boundaries and construct a Basemap object
    lonmin = lons.min()
    lonmax = lons.max()
    latmin = lats.min()
    latmax = lats.max()
    m = Basemap(projection='cyl', llcrnrlat=latmin, urcrnrlat=latmax,
                llcrnrlon=lonmin, urcrnrlon=lonmax, resolution='l', ax=ax)

    # Draw the borders for coastlines and countries
    m.drawcoastlines(linewidth=1)
    m.drawcountries(linewidth=.75)
    m.drawstates()

    # Create default meridians and parallels. The interval between
    # them should be 1, 5, 10, 20, 30, or 40 depending on the size
    # of the domain
    length = max((latmax - latmin), (lonmax - lonmin)) / 5
    if length <= 1:
        dlatlon = 1
    elif length <= 5:
        dlatlon = 5
    else:
        dlatlon = np.round(length, decimals=-1)

    if meridians is None:
        meridians = np.r_[
            np.arange(0, -180, -dlatlon)[::-1], np.arange(0, 180, dlatlon)]
    if parallels is None:
        parallels = np.r_[np.arange(0, -90, -dlatlon)
                          [::-1], np.arange(0, 90, dlatlon)]

    # Draw parallels / meridians
    m.drawmeridians(meridians, labels=[0, 0, 0, 1], linewidth=.75, fontsize=10)
    m.drawparallels(parallels, labels=[1, 0, 0, 1], linewidth=.75, fontsize=10)

    # Set up the color scaling
    cmap = plt.cm.rainbow
    norm = mpl.colors.BoundaryNorm(np.arange(1, len(subregions) + 3), cmap.N)

    # Process the subregions
    for i, reg in enumerate(subregions):
        if subregion_masks is not None and reg.name in subregion_masks.keys():
            domain = (i + 1) * subregion_masks[reg.name]
        else:
            domain = (i + 1) * np.ones((2, 2))

        nlats, nlons = domain.shape
        domain = ma.masked_equal(domain, 0)
        reglats = np.linspace(reg.lat_min, reg.lat_max, nlats)
        reglons = np.linspace(reg.lon_min, reg.lon_max, nlons)
        reglons, reglats = np.meshgrid(reglons, reglats)

        # Convert to to projection coordinates. Not really necessary
        # for cylindrical projections but keeping it here in case we need
        # support for other projections.
        x, y = m(reglons, reglats)

        # Draw the subregion domain
        m.pcolormesh(x, y, domain, cmap=cmap, norm=norm, alpha=.5)

        # Label the subregion
        xm, ym = x.mean(), y.mean()
        m.plot(xm, ym, marker='$%s$' %
               ("R" + str(i + 1)), markersize=12, color='k')

    # Add the title
    ax.set_title(ptitle)

    # Save the figure
    fig.savefig('%s.%s' % (fname, fmt), bbox_inches='tight', dpi=fig.dpi)
    fig.clf()


def _get_colors(num_colors):
    """
    matplotlib will recycle colors after a certain number.  This can make
    line type charts confusing as colors will be reused.  This function
    provides a distribution of colors across the default color map
    to better approximate uniqueness.
    :param num_colors: The number of unique colors to generate.
    :return: A color map with num_colors.
    """
    cmap = plt.get_cmap()
    return [cmap(1. * i / num_colors) for i in range(num_colors)]


def draw_time_series(results, times, labels, fname, fmt='png', gridshape=(1, 1),
                     xlabel='', ylabel='', ptitle='', subtitles=None,
                     label_month=False, yscale='linear', aspect=None,
                     cycle_colors=True, cmap=None):
    ''' Draw a time series plot.
    :param results: 3D array of time series data.
    :type results: :class:`numpy.ndarray`
    :param times: List of Python datetime objects used by Matplotlib to handle
        axis formatting.
    :type times: :class:`list` of :class:`datetime.datetime`
    :param labels: List of names for each data being plotted.
    :type labels: :class:`list` of :mod:`string`
    :param fname: Filename of the plot.
    :type fname: :mod:`string`
    :param fmt: (Optional) filetype for the output.
    :type fmt: :mod:`string`
    :param gridshape: (Optional) tuple denoting the desired grid shape
        (num_rows, num_cols) for arranging the subplots.
    :type gridshape: :func:`tuple` of the form (num_rows, num_cols)
    :param xlabel: (Optional) x-axis title.
    :type xlabel: :mod:`string`
    :param ylabel: (Optional) y-axis title.
    :type ylabel: :mod:`string`
    :param ptitle: (Optional) plot title.
    :type ptitle: :mod:`string`
    :param subtitles: (Optional) list of titles for each subplot.
    :type subtitles: :class:`list` of :mod:`string`
    :param label_month: (Optional) flag to toggle drawing month labels on the
        x-axis.
    :type label_month: :class:`bool`
    :param yscale: (Optional) y-axis scale value, 'linear' for linear and 'log'
        for log base 10.
    :type yscale: :mod:`string`
    :param aspect: (Optional) approximate aspect ratio of each subplot
        (width / height). Default is 8.5 / 5.5
    :type aspect: :class:`float`
    :param cycle_colors: (Optional) flag to toggle whether to allow matlibplot
        to re-use colors when plotting or force an evenly distributed range.
    :type cycle_colors: :class:`bool`
    :param cmap: (Optional) string or :class:`matplotlib.colors.LinearSegmentedColormap`
        instance denoting the colormap. This must be able to be recognized by
        `Matplotlib's get_cmap function <http://matplotlib.org/api/cm_api.html#matplotlib.cm.get_cmap>`_.
        Maps like rainbow and spectral with wide spectrum of colors are nice choices when used with
        the cycle_colors option. tab20, tab20b, and tab20c are good if the plot has less than 20 datasets.
    :type cmap: :mod:`string` or :class:`matplotlib.colors.LinearSegmentedColormap`
    '''
    if cmap is not None:
        set_cmap(cmap)

    # Handle the single plot case.
    if results.ndim == 2:
        results = results.reshape(1, *results.shape)

    # Make sure gridshape is compatible with input data
    nplots = results.shape[0]
    gridshape = _best_grid_shape(nplots, gridshape)

    # Set up the figure
    width, height = _fig_size(gridshape)
    fig = plt.figure()
    fig.set_size_inches((width, height))
    fig.dpi = 300

    # Make the subplot grid
    grid = ImageGrid(fig, 111,
                     nrows_ncols=gridshape,
                     axes_pad=0.3,
                     share_all=True,
                     add_all=True,
                     ngrids=nplots,
                     label_mode='L',
                     aspect=False,
                     cbar_mode='single',
                     cbar_location='bottom',
                     cbar_size=.05,
                     cbar_pad=.20
                     )

    # Make the plots
    for i, ax in enumerate(grid):
        data = results[i]

        if not cycle_colors:
            ax.set_prop_cycle('color', _get_colors(data.shape[0]))

        if label_month:
            xfmt = mpl.dates.DateFormatter('%b')
            xloc = mpl.dates.MonthLocator()
            ax.xaxis.set_major_formatter(xfmt)
            ax.xaxis.set_major_locator(xloc)

        # Set the y-axis scale
        ax.set_yscale(yscale)

        # Set up list of lines for legend
        lines = []
        ymin, ymax = 0, 0

        # Plot each line
        for tSeries in data:
            line = ax.plot_date(times, tSeries, '')
            lines.extend(line)
            cmin, cmax = tSeries.min(), tSeries.max()
            ymin = min(ymin, cmin)
            ymax = max(ymax, cmax)

        # Add a bit of padding so lines don't touch bottom and top of the plot
        ymin = ymin - ((ymax - ymin) * 0.1)
        ymax = ymax + ((ymax - ymin) * 0.1)
        ax.set_ylim((ymin, ymax))

        # Set the subplot title if desired
        if subtitles is not None:
            ax.set_title(subtitles[i], fontsize='small')

    # Create a master axes rectangle for figure wide labels
    fax = fig.add_subplot(111, frameon=False)
    fax.tick_params(labelcolor='none', top='off',
                    bottom='off', left='off', right='off')
    fax.set_ylabel(ylabel)
    fax.set_title(ptitle, fontsize=16)
    fax.title.set_y(1.04)

    # Create the legend using a 'fake' colorbar axes. This lets us have a nice
    # legend that is in sync with the subplot grid
    cax = ax.cax
    cax.set_frame_on(False)
    cax.set_xticks([])
    cax.set_yticks([])
    cax.legend((lines), labels, loc='upper center', ncol=10, fontsize='small',
               mode='expand', frameon=False)

    # Note that due to weird behavior by axes_grid, it is more convenient to
    # place the x-axis label relative to the colorbar axes instead of the
    # master axes rectangle.
    cax.set_title(xlabel, fontsize=12)
    cax.title.set_y(-1.5)

    # Rotate the x-axis tick labels
    for ax in grid:
        for xtick in ax.get_xticklabels():
            xtick.set_ha('right')
            xtick.set_rotation(30)

    # Save the figure
    fig.savefig('%s.%s' % (fname, fmt), bbox_inches='tight', dpi=fig.dpi)
    fig.clf()


def draw_barchart(results, yvalues, fname, ptitle='', fmt='png',
                  xlabel='', ylabel=''):
    ''' Draw a barchart.
    :param results: 1D array of  data.
    :type results: :class:`numpy.ndarray`
    :param yvalues: List of y-axis labels
    :type times: :class:`list`
    :param fname: Filename of the plot.
    :type fname: :mod:`string`
    :param ptitle: (Optional) plot title.
    :type ptitle: :mod:`string`
    :param fmt: (Optional) filetype for the output.
    :type fmt: :mod:`string`
    :param xlabel: (Optional) x-axis title.
    :type xlabel: :mod:`string`
    :param ylabel: (Optional) y-axis title.
    :type ylabel: :mod:`string`
    '''

    y_pos = list(range(len(yvalues)))
    fig = plt.figure()
    fig.set_size_inches((11., 8.5))
    fig.dpi = 300
    ax = plt.subplot(111)
    plt.barh(y_pos, results, align="center", height=0.8, linewidth=0)
    plt.yticks(y_pos, yvalues)
    plt.tick_params(axis="both", which="both", bottom="on", top="off",
                    labelbottom="on", left="off", right="off", labelleft="on")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ymin = min(y_pos)
    ymax = max(y_pos)
    ymin = min((ymin - ((ymax - ymin) * 0.1) / 2), 0.5)
    ymax = ymax + ((ymax - ymin) * 0.1)
    ax.set_ylim((ymin, ymax))
    plt.xlabel(xlabel)
    plt.tight_layout()

    # Save the figure
    fig.savefig('%s.%s' % (fname, fmt), bbox_inches='tight', dpi=fig.dpi)
    fig.clf()


def draw_marker_on_map(lat, lon, fname, fmt='png', location_name=' ', gridshape=(1, 1)):
    '''Draw a marker on a map.
    :param lat: Latitude for plotting a marker.
    :type lat: :class:`float`
    :param lon: Longitude for plotting a marker.
    :type lon: :class:`float`
    :param fname: The filename of the plot.
    :type fname: :class:`string`
    :param fmt: (Optional) Filetype for the output.
    :type fmt: :class:`string`
    :param location_name: (Optional) A label for the map marker.
    :type location_name: :class:`string`
    '''
    fig = plt.figure()
    fig.dpi = 300
    ax = fig.add_subplot(111)

    m = Basemap(projection='cyl', resolution='c', llcrnrlat=lat -
                30, urcrnrlat=lat + 30, llcrnrlon=lon - 60, urcrnrlon=lon + 60)
    m.drawcoastlines(linewidth=1)
    m.drawcountries(linewidth=1)
    m.drawmapboundary(fill_color='aqua')
    m.fillcontinents(color='coral', lake_color='aqua')
    m.ax = ax

    xpt, ypt = m(lon, lat)
    m.plot(xpt, ypt, 'bo')  # plot a blue dot there
    # put some text next to the dot, offset a little bit
    # (the offset is in map projection coordinates)
    plt.text(xpt + 0.5, ypt + 1.5, location_name +
             '\n(lon: %5.1f, lat: %3.1f)' % (lon, lat))

    fig.savefig('%s.%s' % (fname, fmt), bbox_inches='tight', dpi=fig.dpi)
    fig.clf()


def draw_contour_map(dataset, lats, lons, fname, fmt='png', gridshape=(1, 1),
                     clabel='', ptitle='', subtitles=None, cmap=None,
                     clevs=None, nlevs=10, parallels=None, meridians=None,
                     extend='neither', aspect=8.5 / 2.5):
    ''' Draw a multiple panel contour map plot.
    :param dataset: 3D array of data to be plotted with shape (nT, nLat, nLon).
    :type dataset: :class:`numpy.ndarray`
    :param lats: Array of latitudes values.
    :type lats: :class:`numpy.ndarray`
    :param lons: Array of longitudes
    :type lons: :class:`numpy.ndarray`
    :param fname: The filename of the plot.
    :type fname: :mod:`string`
    :param fmt: (Optional) filetype for the output.
    :type fmt: :mod:`string`
    :param gridshape: (Optional) tuple denoting the desired grid shape
        (num_rows, num_cols) for arranging the subplots.
    :type gridshape: :func:`tuple` of the form (num_rows, num_cols)
    :param clabel: (Optional) colorbar title.
    :type clabel: :mod:`string`
    :param ptitle: (Optional) plot title.
    :type ptitle: :mod:`string`
    :param subtitles: (Optional) list of titles for each subplot.
    :type subtitles: :class:`list` of :mod:`string`
    :param cmap: (Optional) string or :class:`matplotlib.colors.LinearSegmentedColormap`
        instance denoting the colormap. This must be able to be recognized by
        `Matplotlib's get_cmap function <http://matplotlib.org/api/cm_api.html#matplotlib.cm.get_cmap>`_.
    :type cmap: :mod:`string` or :class:`matplotlib.colors.LinearSegmentedColormap`
    :param clevs: (Optional) contour levels values.
    :type clevs: :class:`list` of :class:`int` or :class:`float`
    :param nlevs: (Optional) target number of contour levels if clevs is None.
    :type nlevs: :class:`int`
    :param parallels: (Optional) list of ints or floats for the parallels to
        be drawn. See the `Basemap documentation <http://matplotlib.org/basemap/users/graticule.html>`_
        for additional information.
    :type parallels: :class:`list` of :class:`int` or :class:`float`
    :param meridians: (Optional) list of ints or floats for the meridians to
        be drawn. See the `Basemap documentation <http://matplotlib.org/basemap/users/graticule.html>`_
        for additional information.
    :type meridians: :class:`list` of :class:`int` or :class:`float`
    :param extend: (Optional) flag to toggle whether to place arrows at the colorbar
         boundaries. Default is 'neither', but can also be 'min', 'max', or
         'both'. Will be automatically set to 'both' if clevs is None.
    :type extend: :mod:`string`
    '''
    # Handle the single plot case. Meridians and Parallels are not labeled for
    # multiple plots to save space.
    if dataset.ndim == 2 or (dataset.ndim == 3 and dataset.shape[0] == 1):
        if dataset.ndim == 2:
            dataset = dataset.reshape(1, *dataset.shape)
        mlabels = [0, 0, 0, 1]
        plabels = [1, 0, 0, 1]
    else:
        mlabels = [0, 0, 0, 0]
        plabels = [0, 0, 0, 0]

    # Make sure gridshape is compatible with input data
    nplots = dataset.shape[0]
    gridshape = _best_grid_shape(nplots, gridshape)

    # Set up the figure
    fig = plt.figure()
    fig.set_size_inches((8.5, 11.))
    fig.dpi = 300

    # Make the subplot grid
    grid = ImageGrid(fig, 111,
                     nrows_ncols=gridshape,
                     axes_pad=0.3,
                     share_all=True,
                     add_all=True,
                     ngrids=nplots,
                     label_mode='L',
                     cbar_mode='single',
                     cbar_location='bottom',
                     cbar_size=.15,
                     cbar_pad='0%'
                     )

    # Determine the map boundaries and construct a Basemap object
    lonmin = lons.min()
    lonmax = lons.max()
    latmin = lats.min()
    latmax = lats.max()
    m = Basemap(projection='cyl', llcrnrlat=latmin, urcrnrlat=latmax,
                llcrnrlon=lonmin, urcrnrlon=lonmax, resolution='l')

    # Convert lats and lons to projection coordinates
    if lats.ndim == 1 and lons.ndim == 1:
        lons, lats = np.meshgrid(lons, lats)

    # Calculate contour levels if not given
    if clevs is None:
        # Cut off the tails of the distribution
        # for more representative contour levels
        clevs = _nice_intervals(dataset, nlevs)
        extend = 'both'

    cmap = plt.get_cmap(cmap)

    # Create default meridians and parallels. The interval between
    # them should be 1, 5, 10, 20, 30, or 40 depending on the size
    # of the domain
    length = max((latmax - latmin), (lonmax - lonmin)) / 5
    if length <= 1:
        dlatlon = 1
    elif length <= 5:
        dlatlon = 5
    else:
        dlatlon = np.round(length, decimals=-1)
    if meridians is None:
        meridians = np.r_[
            np.arange(0, -180, -dlatlon)[::-1], np.arange(0, 180, dlatlon)]
    if parallels is None:
        parallels = np.r_[np.arange(0, -90, -dlatlon)
                          [::-1], np.arange(0, 90, dlatlon)]

    x, y = m(lons, lats)
    for i, ax in enumerate(grid):
        # Load the data to be plotted
        data = dataset[i]
        m.ax = ax

        # Draw the borders for coastlines and countries
        m.drawcoastlines(linewidth=1)
        m.drawcountries(linewidth=.75)

        # Draw parallels / meridians
        m.drawmeridians(meridians, labels=mlabels, linewidth=.75, fontsize=10)
        m.drawparallels(parallels, labels=plabels, linewidth=.75, fontsize=10)

        # Draw filled contours
        cs = m.contourf(x, y, data, cmap=cmap, levels=clevs, extend=extend)

        # Add title
        if subtitles is not None:
            ax.set_title(subtitles[i], fontsize='small')

    # Add colorbar
    cbar = fig.colorbar(cs, cax=ax.cax, drawedges=True,
                        orientation='horizontal', extendfrac='auto')
    cbar.set_label(clabel)
    cbar.set_ticks(clevs)
    cbar.ax.tick_params(labelsize=6)
    cbar.ax.xaxis.set_ticks_position('none')
    cbar.ax.yaxis.set_ticks_position('none')

    # This is an ugly hack to make the title show up at the correct height.
    # Basically save the figure once to achieve tight layout and calculate
    # the adjusted heights of the axes, then draw the title slightly above
    # that height and save the figure again
    fig.savefig(TemporaryFile(), bbox_inches='tight', dpi=fig.dpi)
    ymax = 0
    for ax in grid:
        bbox = ax.get_position()
        ymax = max(ymax, bbox.ymax)

    # Add figure title
    fig.suptitle(ptitle, y=ymax + .06, fontsize=16)
    fig.savefig('%s.%s' % (fname, fmt), bbox_inches='tight', dpi=fig.dpi)
    fig.clf()


def draw_portrait_diagram(results, rowlabels, collabels, fname, 
                          fmt='png',
                          gridshape=(1, 1), 
                          xlabel='', ylabel='', clabel='',
                          ptitle='', subtitles=None, cmap=None, 
                          clevs=None,
                          nlevs=10, extend='neither', aspect=None):
    ''' Draw a portrait diagram plot.
    :param results: 3D array of the fields to be plotted. The second dimension
              should correspond to the number of rows in the diagram and the
              third should correspond to the number of columns.
    :type results: :class:`numpy.ndarray`
    :param rowlabels: Labels for each row.
    :type rowlabels: :class:`list` of :mod:`string`
    :param collabels: Labels for each row.
    :type collabels: :class:`list` of :mod:`string`
    :param fname: Filename of the plot.
    :type fname: :mod:`string`
    :param fmt: (Optional) filetype for the output.
    :type fmt: :mod:`string`
    :param gridshape: (Optional) tuple denoting the desired grid shape
        (num_rows, num_cols) for arranging the subplots.
    :type gridshape: :func:`tuple` of the form (num_rows, num_cols)
    :param xlabel: (Optional) x-axis title.
    :type xlabel: :mod:`string`
    :param ylabel: (Optional) y-ayis title.
    :type ylabel: :mod:`string`
    :param clabel: (Optional) colorbar title.
    :type clabel: :mod:`string`
    :param ptitle: (Optional) plot title.
    :type ptitle: :mod:`string`
    :param subtitles: (Optional) list of titles for each subplot.
    :type subtitles: :class:`list` of :mod:`string`
    :param cmap: (Optional) string or :class:`matplotlib.colors.LinearSegmentedColormap`
        instance denoting the colormap. This must be able to be recognized by
        `Matplotlib's get_cmap function <http://matplotlib.org/api/cm_api.html#matplotlib.cm.get_cmap>`_.
    :type cmap: :mod:`string` or :class:`matplotlib.colors.LinearSegmentedColormap`
    :param clevs: (Optional) contour levels values.
    :type clevs: :class:`list` of :class:`int` or :class:`float`
    :param nlevs: Optional target number of contour levels if clevs is None.
    :type nlevs: :class:`int`
    :param extend: (Optional) flag to toggle whether to place arrows at the colorbar
         boundaries. Default is 'neither', but can also be 'min', 'max', or
         'both'. Will be automatically set to 'both' if clevs is None.
    :type extend: :mod:`string`
    :param aspect: (Optional) approximate aspect ratio of each subplot
        (width / height). Default is 8.5 / 5.5
    :type aspect: :class:`float`
    '''
    # Handle the single plot case.
    if results.ndim == 2:
        results = results.reshape(1, *results.shape)

    nplots = results.shape[0]

    # Make sure gridshape is compatible with input data
    gridshape = _best_grid_shape(nplots, gridshape)

    # Row and Column labels must be consistent with the shape of
    # the input data too
    prows, pcols = results.shape[1:]
    if len(rowlabels) != prows or len(collabels) != pcols:
        raise ValueError(
            'rowlabels and collabels must have %d and %d elements respectively' % (prows, pcols))

    # Set up the figure
    width, height = _fig_size(gridshape)
    fig = plt.figure()
    fig.set_size_inches((width, height))
    fig.dpi = 300

    # Make the subplot grid
    grid = ImageGrid(fig, 111,
                     nrows_ncols=gridshape,
                     axes_pad=0.4,
                     share_all=True,
                     aspect=False,
                     #add_all=True,
                     ngrids=nplots,
                     label_mode='all',
                     cbar_mode='single',
                     cbar_location='bottom',
                     cbar_size=.15,
                     cbar_pad='3%'
                     )

    # Calculate colorbar levels if not given
    if clevs is None:
        # Cut off the tails of the distribution
        # for more representative colorbar levels
        clevs = _nice_intervals(results, nlevs)
        extend = 'both'

    cmap = plt.get_cmap(cmap)
    norm = mpl.colors.BoundaryNorm(clevs, cmap.N)

    # Do the plotting
    for i, ax in enumerate(grid):
        #print(i)
        data = results[i]

        cs = ax.matshow(data, cmap=cmap, aspect='auto',
                        origin='lower', norm=norm)

        # Add grid lines
        ax.xaxis.set_ticks(np.arange(data.shape[1] + 0))
        #print(np.arange(data.shape[1] + 1))
        ax.yaxis.set_ticks(np.arange(data.shape[0] + 0))
        #Error awalnya 1
        #print(np.arange(data.shape[0] + 1))
        x = (ax.xaxis.get_majorticklocs() - .5)
        y = (ax.yaxis.get_majorticklocs() - .5)
        ax.vlines(x, y.min(), y.max())
        ax.hlines(y, x.min(), x.max())

        # Configure ticks
        ax.xaxis.tick_bottom()
        ax.xaxis.set_ticks_position('none')
        ax.yaxis.set_ticks_position('none')
        ax.set_xticklabels(collabels, fontsize='xx-small')
        #ValueError: The number of FixedLocator locations (13), 
        #usually from a call to set_ticks, 
        #does not match the number of ticklabels (12).
        
        ax.set_yticklabels(rowlabels, fontsize='xx-small')

        # Add axes title
        if subtitles is not None:
            ax.text(0.5, 1.04, subtitles[i], va='center', ha='center',
                    transform=ax.transAxes, fontsize='small')

    # Create a master axes rectangle for figure wide labels
    fax = fig.add_subplot(111, frameon=False)
    fax.tick_params(labelcolor='none', top='off',
                    bottom='off', left='off', right='off')
    fax.set_ylabel(ylabel)
    fax.set_title(ptitle, fontsize=16)
    fax.title.set_y(1.04)

    # Add colorbar
    cax = ax.cax
    cbar = fig.colorbar(cs, cax=cax, norm=norm, boundaries=clevs, drawedges=True,
                        extend=extend, orientation='horizontal', extendfrac='auto')
    cbar.set_label(clabel)
    cbar.set_ticks(clevs)
    cbar.ax.tick_params(labelsize=6)
    cbar.ax.xaxis.set_ticks_position('none')
    cbar.ax.yaxis.set_ticks_position('none')

    # Note that due to weird behavior by axes_grid, it is more convenient to
    # place the x-axis label relative to the colorbar axes instead of the
    # master axes rectangle.
    cax.set_title(xlabel, fontsize=12)
    cax.title.set_y(1.5)

    # Save the figure
    print('saving taylor diagram...')
    fig.savefig('%s.%s' % (fname, fmt), bbox_inches='tight', dpi=fig.dpi)
    fig.clf()

def draw_portrait_diagram2(results, rowlabels, collabels, fname, 
                          fmt='png',
                          gridshape=(1, 1), 
                          xlabel='', ylabel='', clabel='',
                          ptitle='', subtitles=None, cmap=None, 
                          clevs=None,
                          nlevs=10, extend='neither', aspect=None):
    
    #error diatasi sedikit disini
    
    # Handle the single plot case.
    if results.ndim == 2:
        results = results.reshape(1, *results.shape)

    nplots = results.shape[0]

    # Make sure gridshape is compatible with input data
    gridshape = _best_grid_shape(nplots, gridshape)

    # Row and Column labels must be consistent with the shape of
    # the input data too
    prows, pcols = results.shape[1:]
    if len(rowlabels) != prows or len(collabels) != pcols:
        raise ValueError(
            'rowlabels and collabels must have %d and %d elements respectively' % (prows, pcols))

    # Set up the figure
    width, height = _fig_size(gridshape)
    fig = plt.figure()
    fig.set_size_inches((width, height))
    fig.dpi = 300

    # Make the subplot grid
    grid = ImageGrid(fig, 111,
                     nrows_ncols=gridshape,
                     axes_pad=0.4,
                     share_all=True,
                     aspect=False,
                     #add_all=True,
                     ngrids=nplots,
                     label_mode='all',
                     cbar_mode='single',
                     cbar_location='bottom',
                     cbar_size=.15,
                     cbar_pad='3%'
                     )

    # Calculate colorbar levels if not given
    if clevs is None:
        # Cut off the tails of the distribution
        # for more representative colorbar levels
        clevs = _nice_intervals(results, nlevs)
        extend = 'both'

    cmap = plt.get_cmap(cmap)
    norm = mpl.colors.BoundaryNorm(clevs, cmap.N)

    # Do the plotting
    for i, ax in enumerate(grid):
        #print(i)
        data = results[i]

        cs = ax.matshow(data, cmap=cmap, aspect='auto',
                        origin='lower', norm=norm)

        # Add grid lines
        ax.xaxis.set_ticks(np.arange(data.shape[1] + 1))
        #print(np.arange(data.shape[1] + 1))
        ax.yaxis.set_ticks(np.arange(data.shape[0] + 1))
        #Error awalnya 1
        #print(np.arange(data.shape[0] + 1))
        x = (ax.xaxis.get_majorticklocs() - .5)
        y = (ax.yaxis.get_majorticklocs() - .5)
        ax.vlines(x, y.min(), y.max(),color='black')
        ax.hlines(y, x.min(), x.max(),color='black')
        
        #tambahan
        ax.xaxis.set_ticks(np.arange(data.shape[1] + 0))
        #print(np.arange(data.shape[1] + 1))
        ax.yaxis.set_ticks(np.arange(data.shape[0] + 0))

        # Configure ticks
        ax.xaxis.tick_bottom()
        ax.xaxis.set_ticks_position('none')
        ax.yaxis.set_ticks_position('none')
        ax.set_xticklabels(collabels, fontsize='xx-small')
        #ValueError: The number of FixedLocator locations (13), 
        #usually from a call to set_ticks, 
        #does not match the number of ticklabels (12).
        
        ax.set_yticklabels(rowlabels, fontsize='xx-small')

        # Add axes title
        if subtitles is not None:
            ax.text(0.5, 1.04, subtitles[i], va='center', ha='center',
                    transform=ax.transAxes, fontsize='small')

    # Create a master axes rectangle for figure wide labels
    fax = fig.add_subplot(111, frameon=False)
    fax.tick_params(labelcolor='none', top='off',
                    bottom='off', left='off', right='off')
    fax.set_ylabel(ylabel)
    fax.set_title(ptitle, fontsize=16)
    fax.title.set_y(1.04)

    # Add colorbar
    cax = ax.cax
    cbar = fig.colorbar(cs, cax=cax, norm=norm, boundaries=clevs, drawedges=True,
                        extend=extend, orientation='horizontal', extendfrac='auto')
    cbar.set_label(clabel)
    cbar.set_ticks(clevs)
    cbar.ax.tick_params(labelsize=6)
    cbar.ax.xaxis.set_ticks_position('none')
    cbar.ax.yaxis.set_ticks_position('none')

    # Note that due to weird behavior by axes_grid, it is more convenient to
    # place the x-axis label relative to the colorbar axes instead of the
    # master axes rectangle.
    cax.set_title(xlabel, fontsize=12)
    cax.title.set_y(1.5)

    # Save the figure
    fig.savefig('%s.%s' % (fname+'2', fmt), bbox_inches='tight', dpi=fig.dpi)
    fig.clf()
    
class TaylorDiagram(object):
    """ Taylor diagram helper class
    Plot model standard deviation and correlation to reference (data)
    sample in a single-quadrant polar plot, with r=stddev and
    theta=arccos(correlation).
    This class was released as public domain by the original author
    Yannick Copin. You can find the original Gist where it was
    released at: https://gist.github.com/ycopin/3342888
    """

    def __init__(self, refstd, radmax=1.5, fig=None, rect=111, label='_'):
        """Set up Taylor diagram axes, i.e. single quadrant polar
        plot, using mpl_toolkits.axisartist.floating_axes. refstd is
        the reference standard deviation to be compared to.
        """

        from matplotlib.projections import PolarAxes
        import mpl_toolkits.axisartist.floating_axes as FA
        import mpl_toolkits.axisartist.grid_finder as GF

        self.refstd = refstd            # Reference standard deviation

        tr = PolarAxes.PolarTransform()

        # Correlation labels
        rlocs = np.concatenate((np.arange(10) / 10., [0.95, 0.99]))
        tlocs = np.arccos(rlocs)        # Conversion to polar angles
        gl1 = GF.FixedLocator(tlocs)    # Positions
        tf1 = GF.DictFormatter(dict(zip(tlocs, map(str, rlocs))))

        # Standard deviation axis extent
        print('Taylor diagram rad_max=',radmax)
        self.smin = 0
        self.smax = radmax * self.refstd

        ghelper = FA.GridHelperCurveLinear(tr,
                                           extremes=(0, np.pi / 2,  # 1st quadrant
                                                     self.smin, self.smax),
                                           grid_locator1=gl1,
                                           tick_formatter1=tf1,
                                           )

        if fig is None:
            fig = plt.figure()
            

        ax = FA.FloatingSubplot(fig, rect, grid_helper=ghelper)
        fig.add_subplot(ax)
        
        #adjust_fig=0.2
        #fig.subplots_adjust(bottom=adjust_fig)

        # Adjust axes
        ax.axis["top"].set_axis_direction("bottom")  # "Angle axis"
        ax.axis["top"].toggle(ticklabels=True, label=True)
        ax.axis["top"].major_ticklabels.set_axis_direction("top")
        ax.axis["top"].label.set_axis_direction("top")
        ax.axis["top"].label.set_text("Correlation")

        ax.axis["left"].set_axis_direction("bottom")  # "X axis"
        #ax.axis["left"].label.set_text("Standard deviation")

        ax.axis["right"].set_axis_direction("top")   # "Y axis"
        ax.axis["right"].toggle(ticklabels=True, label=True)
        #jika label=True label di Y muncul
        ax.axis["right"].major_ticklabels.set_axis_direction("left")
        #tambah
        ax.axis["right"].label.set_text("Standard deviation ratio")

        ax.axis["bottom"].set_visible(False)         # Useless

        # Contours along standard deviations
        ax.grid(False)

        self._ax = ax                   # Graphical axes
        self.ax = ax.get_aux_axes(tr)   # Polar coordinates

        # Add reference point and stddev contour
        # print "Reference std:", self.refstd
        l, = self.ax.plot([0], self.refstd, 'k*',
                          ls='', ms=10, label=label)
        t = np.linspace(0, np.pi / 2)
        r = np.zeros_like(t) + self.refstd
        self.ax.plot(t, r, 'k--', label='_')

        # Collect sample points for latter use (e.g. legend)
        self.samplePoints = [l]

    def add_sample(self, stddev, corrcoef, *args, **kwargs):
        """Add sample (stddev,corrcoeff) to the Taylor diagram. args
        and kwargs are directly propagated to the Figure.plot
        command."""

        l, = self.ax.plot(np.arccos(corrcoef), stddev,
                          *args, **kwargs)  # (theta,radius)
        self.samplePoints.append(l)

        return l

    def add_rms_contours(self, levels=5, **kwargs):
        """Add constant centered RMS difference contours."""

        rs, ts = np.meshgrid(np.linspace(self.smin, self.smax),
                             np.linspace(0, np.pi / 2))
        # Compute centered RMS difference
        rms = np.sqrt(self.refstd**2 + rs**2 - 2 *
                      self.refstd * rs * np.cos(ts))

        #contours = self.ax.contour(ts, rs, rms, levels, **kwargs)
        contours = self.ax.contour(ts, rs, rms, levels,colors='0.5', **kwargs)
        #plt.clabel(contours, contours.levels, inline=True, fmt='%.2f', fontsize=15)
        plt.clabel(contours, inline=1, fmt='%.2f', fontsize=8, colors='0.5')

    def add_stddev_contours(self, std, corr1, corr2, **kwargs):
        """Add a curved line with a radius of std between two points
        [std, corr1] and [std, corr2]"""

        t = np.linspace(np.arccos(corr1), np.arccos(corr2))
        r = np.zeros_like(t) + std
        return self.ax.plot(t, r, 'red', linewidth=2)

    def add_contours(self, std1, corr1, std2, corr2, **kwargs):
        """Add a line between two points
        [std1, corr1] and [std2, corr2]"""

        t = np.linspace(np.arccos(corr1), np.arccos(corr2))
        r = np.linspace(std1, std2)

        return self.ax.plot(t, r, 'red', linewidth=2)
    
    def add_grid(self, *args, **kwargs):
        """Add a grid."""

        self._ax.grid(*args, **kwargs)


def draw_histogram(dataset_array, data_names, fname, fmt='png', nbins=10):
    '''
    Purpose:: Draw a histogram for the input dataset.
    :param dataset_array: A list of data values [data1, data2, ....].
    :type dataset_array: :class:`list` of :class:`float`
    :param data_names: A list of data names  ['name1','name2',....].
    :type data_names: :class:`list` of :class:`string`
    :param fname: The filename of the plot.
    :type fname: :class:`string`
    :param fmt: (Optional) Filetype for the output.
    :type fmt: :class:`string`
    :param bins: (Optional) Number of bins.
    :type bins: :class:`integer`
    '''
    fig = plt.figure()
    fig.dpi = 300
    ndata = len(dataset_array)

    data_min = 500.
    data_max = 0.

    for data in dataset_array:
        data_min = np.min([data_min, data.min()])
        data_max = np.max([data_max, data.max()])

    bins = np.linspace(np.round(data_min), np.round(data_max + 1), nbins)
    for idata, data in enumerate(dataset_array):
        ax = fig.add_subplot(ndata, 1, idata + 1)
        ax.hist(data, bins, alpha=0.5, label=data_names[idata], normed=True)
        leg = ax.legend()
        leg.get_frame().set_alpha(0.5)
        ax.set_xlim([data_min - (data_max - data_min) * 0.15,
                     data_max + (data_max - data_min) * 0.15])

    fig.savefig('%s.%s' % (fname, fmt), bbox_inches='tight', dpi=fig.dpi)

def fill_US_states_with_color(regions, fname, fmt='png', ptitle='',
                              colors=False, values=None, region_names=None):

    ''' Fill the States over the contiguous US with colors
    :param regions: The list of subregions(lists of US States) to be filled
                    with different colors.
    :type regions: :class:`list`
    :param fname: The filename of the plot.
    :type fname: :mod:`string`
    :param fmt: (Optional) filetype for the output.
    :type fmt: :mod:`string`
    :param ptitle: (Optional) plot title.
    :type ptitle: :mod:`string`
    :param colors: (Optional) : If True, each region will be filled
                                with different colors without using values
    :type colors: :class:`bool`
    :param values: (Optional) : If colors==False, color for each region scales
                                an associated element in values
    :type values: :class:`numpy.ndarray`
    '''

    nregion = len(regions)
    if colors:
        cmap = plt.cm.rainbow
    if not (values is None):
        cmap = plt.cm.seismic
        max_abs = np.abs(values).max()

    # Set up the figure
    fig = plt.figure()
    fig.set_size_inches((8.5, 11.))
    fig.dpi = 300
    ax = fig.add_subplot(111)

    # create the map
    m = Basemap(llcrnrlon=-127,llcrnrlat=22,urcrnrlon=-65,urcrnrlat=52,
                ax=ax)

    for iregion, region in enumerate(regions):
        shapes = utils.shapefile_boundary('us_states', region)
        patches=[]
        lats=np.empty(0)
        lons=np.empty(0)
        for shape in shapes:
            patches.append(Polygon(np.array(shape), True))

            lons = np.append(lons, shape[:,0])
            lats = np.append(lats, shape[:,1])
        if colors:
            color_to_fill=cmap((iregion+0.5)/nregion)
        if not (values is None):
            value = values[iregion]
            color_to_fill = cmap(0.5+np.sign(value)*abs(value)/max_abs*0.45)
        ax.add_collection(PatchCollection(patches, facecolor=color_to_fill))
        if region_names:
            ax.text(lons.mean(), lats.mean(), region_names[iregion],
                    ha='center', va='center', fontsize=10)
    m.drawcountries(linewidth=0.)

    # Add the title
    ax.set_title(ptitle)
    # Save the figure
    fig.savefig('%s.%s' % (fname, fmt), bbox_inches='tight', dpi=fig.dpi)
    fig.clf()

def draw_plot_to_compare_trends(obs_data, ens_data, model_data,
                          fname, fmt='png', ptitle='', data_labels=None,
                          xlabel='', ylabel=''):

    ''' Fill the States over the contiguous US with colors
    :param obs_data: An array of observed trend and standard errors for regions
    :type obs_data: :class:'numpy.ndarray'
    :param ens_data: An array of trend and standard errors from a multi-model ensemble for regions
    :type ens_data: : class:'numpy.ndarray'
    :param model_data: An array of trends from models for regions
    :type model_data: : class:'numpy.ndarray'
    :param fname: The filename of the plot.
    :type fname: :mod:`string`
    :param fmt: (Optional) filetype for the output.
    :type fmt: :mod:`string`
    :param ptitle: (Optional) plot title.
    :type ptitle: :mod:`string`
    :param data_labels: (Optional) names of the regions
    :type data_labels: :mod:`list`
    :param xlabel: (Optional) a label for x-axis
    :type xlabel: :mod:`string`
    :param ylabel: (Optional) a label for y-axis
    :type ylabel: :mod:`string`
    '''
    nregions = obs_data.shape[1]

    # Set up the figure
    fig = plt.figure()
    fig.set_size_inches((8.5, 11.))
    fig.dpi = 300
    ax = fig.add_subplot(111)

    b_plot = ax.boxplot(model_data, widths=np.repeat(0.2, nregions), positions=np.arange(nregions)+1.3)
    plt.setp(b_plot['medians'], color='black')
    plt.setp(b_plot['whiskers'], color='black')
    plt.setp(b_plot['boxes'], color='black')
    plt.setp(b_plot['fliers'], color='black')
    ax.errorbar(np.arange(nregions)+0.8, obs_data[0,:], yerr=obs_data[1,:],
                fmt='o', color='r', ecolor='r')
    ax.errorbar(np.arange(nregions)+1., ens_data[0,:], yerr=ens_data[1,:],
                fmt='o', color='b', ecolor='b')
    ax.set_xticks(np.arange(nregions)+1)
    ax.set_xlim([0, nregions+1])

    if data_labels:
        ax.set_xticklabels(data_labels)
    fig.savefig('%s.%s' % (fname, fmt), bbox_inches='tight')

def draw_precipitation_JPDF (plot_data, plot_level, x_ticks, x_names,y_ticks,y_names,
               output_file, title=None, diff_plot=False, cmap = cm.BrBG,
               cbar_ticks=[0.01, 0.10, 0.5, 2, 5, 25],
               cbar_label=['0.01', '0.10', '0.5', '2', '5', '25']):
    '''
    :param plot_data: a numpy array of data to plot (dimY, dimX)
    :type plot_data: :class:'numpy.ndarray'
    :param plot_level: levels to plot
    :type plot_level: :class:'numpy.ndarray'
    :param x_ticks: x values where tick makrs are located
    :type x_ticks: :class:'numpy.ndarray'
    :param x_names: labels for the ticks on x-axis (dimX)
    :type x_names: :class:'list'
    :param y_ticks: y values where tick makrs are located
    :type y_ticks: :class:'numpy.ndarray'
    :param y_names: labels for the ticks on y-axis (dimY)
    :type y_names: :class:'list'
    :param output_file: name of output png file
    :type output_file: :mod:'string'
    :param title: (Optional) title of the plot
    :type title: :mod:'string'
    :param diff_plot: (Optional) if true, a difference plot will be generated
    :type diff_plot: :mod:'bool'
    :param cbar_ticks: (Optional) tick marks for the color bar
    :type cbar_ticks: :class:'list'
    :param cbar_label: (Optional) lables for the tick marks of the color bar
    :type cbar_label: :class:'list'
    '''

    if diff_plot:
        cmap = cm.RdBu_r

    fig = plt.figure()
    sb = fig.add_subplot(111)
    dimY, dimX = plot_data.shape
    plot_data2 = np.zeros([dimY,dimX]) # sectioned array for plotting

    # fontsize
    rcParams['axes.labelsize'] = 8
    rcParams['xtick.labelsize'] = 8
    rcParams['ytick.labelsize'] = 8
    # assign the values in plot_level to plot_data
    for iy in range(dimY):
         for ix in range(dimX):
             if plot_data[iy,ix] <= np.min(plot_level):
                 plot_data2[iy,ix] = -1.
             else:
                 plot_data2[iy,ix] = plot_level[np.where(plot_level <= plot_data[iy,ix])].max()
    sb.set_xticks(x_ticks)
    sb.set_xticklabels(x_names)
    sb.set_yticks(y_ticks)
    sb.set_yticklabels(y_names)

    norm = BoundaryNorm(plot_level[1:], cmap.N)
    a=sb.pcolor(plot_data2, edgecolors = 'k', linewidths=0.5, cmap = cmap, norm = norm)
    a.cmap.set_under('w')
    sb.set_xlabel('Spell duration [hrs]')
    sb.set_ylabel('Peak rainfall [mm/hr]')
    cax = fig.add_axes([0.2, -0.06, 0.6, 0.02])
    cbar = plt.colorbar(a, cax=cax, orientation = 'horizontal', extend='both')
    cbar.set_ticks(cbar_ticks)
    cbar.set_ticklabels(cbar_label)
    if title:
        fig.suptitle(title)
    fig.savefig(output_file, dpi=600,bbox_inches='tight')
