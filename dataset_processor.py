#
#  Licensed to the Apache Software Foundation (ASF) under one or more
#  contributor license agreements.  See the NOTICE file distributed with
#  this work for additional information regarding copyright ownership.
#  The ASF licenses this file to You under the Apache License, Version 2.0
#  (the "License"); you may not use this file except in compliance with
#  the License.  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
from cftime import datetime as utime
import datetime
import logging

import netCDF4
import numpy as np
import numpy.ma as ma
import scipy.ndimage
from matplotlib.path import Path
from scipy.interpolate import griddata
from scipy.ndimage import map_coordinates

import ocw.utils as utils
from ocw import dataset as ds

logger = logging.getLogger(__name__)

import yaml
import sys

import ocw.metrics as metrics
config_file = str(sys.argv[1])
config = yaml.safe_load(open(config_file))


def mmew(reference_dataset, target_datasets, target_names, workdir):
    import matplotlib.pyplot as plt
    from sklearn import preprocessing
    import pandas as pd
    
    fig,ax=plt.subplots(figsize=[8,6])
    plt.subplots_adjust(bottom=.2)
    
    #MMEW khusus nobs=2
    n=config['nobs']-1  #-1 ?? => era5 tidak ikut 
    reg=config['region']
    tipe= config['Metric_type']
    
    print('')
    print('Generating multi-model weighted ensemble ...')
    print('#target_names[n:]=',target_names[n:])
    #print('weighted temporal subsetting =>', reference_dataset.values.shape)


    lastElementIndex = len(target_datasets)-1
    # print(lastElementIndex)
    # # Removing the last element using slicing 
    # target_datasets = target_datasets[:lastElementIndex]
    # print(target_datasets)

    # lastElementIndex = len(target_names)-1
    # print(lastElementIndex)
    # # Removing the last element using slicing 
    # target_names = target_names[:lastElementIndex]
    # print(target_names)
    # exit()

    # ERA5 dan MME tidak ikut mmew
    w_datasets=target_datasets[n:lastElementIndex]
    w_datasets_names=target_names[n:lastElementIndex]
    print('#w_target_names[obs2:MME]=',w_datasets_names)
    w_simpan=[]
    name_simpan=[]
    if len(target_datasets) >= 2.:
        
        ww=[]
        #pmetrics=['Taylor','CPI','SS','Tian','D']
        #pmetrics=['MME'] #'T','D']
        pmetrics=['Rand']
        #pmetrics=['Taylor','SS','Tian', 'Rand']
        
        n=0
        mm=['s','>','*','o']
        #pmetrics=config['metrics_taylor_diagram']
        for metric in pmetrics:
            '''
            if metric=='PI':
                print('metric=','PI')
                import xarray as xr
                ds = xr.open_dataset("d:/Cordex/ClimWIP-Brunner_etal_2020_ESD/data/xxtes_DJF.nc")
                #weights1=np.append(ds.weights_q.data,0)
                #weights1=ds.weights_q.data #P only
                weights1=ds.weights.data  # PI
                ww=weights1
            '''
            #if metric=='rand':
            if metric=='Rand':
                
                #DJF
                #sdr 1.21 to 1.02 # gain 0.2an ini yg dipake dijurnal agromet
                a='[0.49426241 0.07584506 0.69924236 0.07423632 0.14060344 0.14365076 0.22516981 0.43338157 0.29333322]'
                #rmse 2.4 to 1.95 #implisit di Taylor diagram
                #a='[0.94016854 0.36861419 0.93699925 0.75798127 0.01845237 0.48284197 0.54228094 0.07454796 0.34062455]'
                #r 0.89 to 0.90  #kenaikan kecil sekali 0.01
                #a='[0.70057932 0.64883376 0.23671563 0.67707414 0.12771409 0.01684781 0.59530162 0.0709224 0.97037914]'
                a = eval( a.replace(' ', ', '))
                ww=a
                #ww=np.array(np.repeat(1,9))
                
            else:
                print('metric=',metric)
                for i in np.arange(len(w_datasets)):
                    import pandas as pd
                    
                    if config['Weighted based spatial or temporal?']== 1:
                        print('spatial based weighted')
                        pmetric = metrics.spatial_weights(reference_dataset, w_datasets[i], M_type=metric)
                        #print(pmetric)
                        ww.append(pmetric)
                    
                    #semua EW dari temporal performance
                    if config['Weighted based spatial or temporal?']== 2:
                        print('temporal based weighted')
                       
                        #print(j,i)
                        #print(w_datasets[i])
                        pmetric = metrics.temporal_weights(reference_dataset, w_datasets[i], M_type=metric)
                        #print(pmetric)
                        ww.append(pmetric)
                    #print('ww',ww)
            
            #if metric== 'MME':
            # no ipsl maka di nolkan xxx
            #ww=[1,1,0,1,1,1,1,1] xxx 8 bobot? 9 model kan
            #ww=[1,1,1,1,1,1,1]
            weights = ww
            print('#weights = ',weights) 
            
            #normalized
            result = pd.DataFrame(ww)#,target_names,columns=['SS'])
            ww = preprocessing.normalize(result, axis=0)
            
            ###-----------------
            ###plot weigths
            ax.plot(w_datasets_names,ww, label = metric, marker=mm[n],color='black')
            plt.ylabel('Weight factors')
            '''
            target_datasets.append(ensemble_weighted(w_datasets,weights))
            target_names.append('WE_'+metric) 
            #calc_ensemble_average
         
            print('#target_names_pasca_w=',target_names)
            
            #---file output terpisah, var=pr, nama file output=input
            #---for rainfall only 
            w_simpan.append(ensemble_weighted(w_datasets,weights))
            name_simpan.append('WE_'+metric+'_'+reg+'_1981-2005')
            
            write_netcdf_multiple_datasets_with_subregions4(
               w_simpan, 
               name_simpan, 
               path=workdir)
            '''
            ##Batas plot weights kasih '''
            w_simpan=[]
            name_simpan=[]
            ww=[]
            
            n=n+1
            
        plt.legend(loc='best', prop={'size':8.5}, frameon=False) 
        plt.xticks(rotation=45)
        plt.show()
        
def mmew2(reference_dataset, target_datasets, target_names, workdir):
    import matplotlib.pyplot as plt
    from sklearn import preprocessing
    import pandas as pd
    
    fig,ax=plt.subplots(figsize=[8,6])
    plt.subplots_adjust(bottom=.2)
    
    reg=config['region']
    tipe= config['Metric_type']
    
    print('')
    print('Generating multi-model weighted ensemble ...')
    print('#target_names=',target_names)
    w_simpan=[]
    name_simpan=[]
    if len(target_datasets) >= 2:
        
        ww=[]
        pmetrics=['Taylor','SS','Tian','Rand']
        #pmetrics=['MME'] #'T','D']
        #pmetrics=['Rand']
        #pmetrics=['MME','Taylor','SS','Tian'] #, 'Rand']
        
        n=0
        mm=['s','>','*','o']
        #pmetrics=config['metrics_taylor_diagram']
        for metric in pmetrics:
            '''
            if metric=='PI':
                print('metric=','PI')
                import xarray as xr
                ds = xr.open_dataset("d:/Cordex/ClimWIP-Brunner_etal_2020_ESD/data/xxtes_DJF.nc")
                #weights1=np.append(ds.weights_q.data,0)
                #weights1=ds.weights_q.data #P only
                weights1=ds.weights.data  # PI
                ww=weights1
            '''
            #if metric=='rand':
            if metric=='Rand':
                #9 model 9 bobot
                #DJF
                #sdr 1.21 to 1.02 # gain 0.2an ini yg dipake dijurnal agromet
                a='[0.49426241 0.07584506 0.69924236 0.07423632 0.14060344 0.14365076 0.22516981 0.43338157 0.29333322]'
                #rmse 2.4 to 1.95 #implisit di Taylor diagram
                #a='[0.94016854 0.36861419 0.93699925 0.75798127 0.01845237 0.48284197 0.54228094 0.07454796 0.34062455]'
                #r 0.89 to 0.90  #kenaikan kecil sekali 0.01
                #a='[0.70057932 0.64883376 0.23671563 0.67707414 0.12771409 0.01684781 0.59530162 0.0709224 0.97037914]'
                a = eval( a.replace(' ', ', '))
                ww=a
                #ww=np.array(np.repeat(1,9))
            #kalo metric=Rand matikan ini
            #if metric== 'MME':
                # no ipsl maka 9-1=8
            #    ww=[1,1,1,1,1,1,1,1]     
            else:
                print('metric=',metric)
                for i in np.arange(len(target_datasets)):
                    import pandas as pd
                    
                    if config['Weighted based spatial or temporal?']== 1:
                        print('spatial based weighted')
                        pmetric = metrics.spatial_weights(reference_dataset, target_datasets[i], M_type=metric)
                        #print(pmetric)
                        ww.append(pmetric)
                    
                    #semua EW dari temporal performance
                    if config['Weighted based spatial or temporal?']== 2:
                        print('temporal based weighted')
                       
                        #print(j,i)
                        #print(w_datasets[i])
                        pmetric = metrics.temporal_weights(reference_dataset, target_datasets[i], M_type=metric)
                        #print(pmetric)
                        ww.append(pmetric)
                    #print('ww',ww)
            
            
            weights = ww
            print('#weights = ',weights) 
            
            #normalized
            result = pd.DataFrame(ww)#,target_names,columns=['SS'])
            ww = preprocessing.normalize(result, axis=0)
            
            ###-----------------
            ###plot weigths
            ax.plot(target_names,ww, label = metric, marker=mm[n],color='black')
            plt.ylabel('Weight factors')
            '''
            #---file output terpisah, var=pr, nama file output=input
            #---for rainfall only 
            w_simpan.append(ensemble_weighted(target_datasets,weights))
            name_simpan.append('3WE_'+metric+'_'+reg+'_1981-2005')
            
            write_netcdf_multiple_datasets_with_subregions4(
               w_simpan, 
               name_simpan, 
               path=workdir)
            '''
            ##Batas plot weights kasih '''
            w_simpan=[]
            name_simpan=[]
            ww=[]
            
            n=n+1
            
        plt.legend(loc='best', prop={'size':8.5}, frameon=False) 
        plt.xticks(rotation=45)
        # Menghilangkan garis sumbu
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)

        plt.show()
      
        
#for downscaling
def temporal_subset(target_dataset, month_start, month_end,
                    average_each_year=True):
    """ Temporally subset data given month_index.

    :param month_start: An integer for beginning month (Jan=1)
    :type month_start: :class:`int`

    :param month_end: An integer for ending month (Jan=1)
    :type month_end: :class:`int`

    :param target_dataset: Dataset object that needs temporal subsetting
    :type target_dataset: Open Climate Workbench Dataset Object

    :param average_each_year: If True, output dataset is averaged for each year
    :type average_each_year: :class:'boolean'

    :returns: A temporal subset OCW Dataset
    :rtype: Open Climate Workbench Dataset Object
    """

    if month_start > month_end:
        month_index = list(range(month_start, 13))
        month_index.extend(list(range(1, month_end + 1)))
    else:
        month_index = range(month_start, month_end + 1)

    dates = target_dataset.times
    #print('dates=',dates)
    months = np.array([d.month for d in dates])
    #print('months=',months)
    
    time_index = []
    for m_value in month_index:
        time_index.extend(list(np.where(months == m_value)[0]))
    time_index.sort()
    #print('time_index=',time_index)
    
    
    new_dataset = ds.Dataset(target_dataset.lats,
                             target_dataset.lons,
                             target_dataset.times[time_index],
                             target_dataset.values[time_index, :],
                             variable=target_dataset.variable,
                             units=target_dataset.units,
                             name=target_dataset.name)
    #print(new_dataset)  
    if average_each_year:
        new_times = new_dataset.times
        nmonth = len(month_index)
        ntime = new_times.size
        nyear = ntime // nmonth
        if ntime % nmonth != 0:
            logger.warning("Number of times in dataset ({}) does not "
                           "divide evenly into {} year(s). Trimming data..."
                           .format(ntime, nyear))
            slc = utils.trim_dataset(new_dataset)
            new_dataset.values = new_dataset.values[slc]
            new_times = new_times[slc]
            nyear = new_times.size // nmonth

        averaged_time = []
        ny, nx = target_dataset.values.shape[1:]
        averaged_values = ma.zeros([nyear, ny, nx])
        for iyear in np.arange(nyear):
            # centered time index of the season between month_start and
            # month_end in each year
            center_index = int(nmonth / 2 + iyear * nmonth)
            if nmonth == 1:
                center_index = iyear
            averaged_time.append(new_times[center_index])
            averaged_values[iyear, :] = ma.average(new_dataset.values[
                nmonth * iyear: nmonth * iyear + nmonth, :], axis=0)
        new_dataset = ds.Dataset(target_dataset.lats,
                                 target_dataset.lons,
                                 np.array(averaged_time),
                                 averaged_values,
                                 variable=target_dataset.variable,
                                 units=target_dataset.units,
                                 name=target_dataset.name)
                                 
        #print(new_dataset)                         
    return new_dataset


def annual_subset(target_dataset):
    import xarray as xr
     
    dsi = xr.DataArray(target_dataset.values,
    coords={'times': target_dataset.times,
            'lats': target_dataset.lats, 
            'lons': target_dataset.lons},
            #'mask':target_dataset.values.mask},
    dims=["times", "lats", "lons"])
 
    d = ds.groupby('times.year').mean()
    #print('d=',d)
    
    #d = dsi.groupby('times.year').sum()
    #print('d2=',d2)
    #jika sum() error => sudah ok hasil R sama dgn mean()
    #nan at correlation
    
    d=d.fillna(-100)
   
    mask_array = np.zeros(d.values.shape)
    index=np.where(d.values== -100)
    mask_array[index]=1
    dd=ma.array(d.values, mask= mask_array)
    
   
    new_dataset = ds.Dataset(target_dataset.lats,
                             target_dataset.lons,
                             d.year,
                             #d.times,
                             dd,
                             variable=target_dataset.variable,
                             units=target_dataset.units,
                             name=target_dataset.name)
        
    return new_dataset

def annual_cycle_subset(target_dataset):
    import xarray as xr
    #pakai dataset masih error cek
    
    #print(target_dataset.values.mask)
    
    d = xr.DataArray(target_dataset.values,
    coords={'times': target_dataset.times,
            'lats': target_dataset.lats, 
            'lons': target_dataset.lons},
            #'mask':target_dataset.values.mask},
    dims=["times", "lats", "lons"])
    #print(d.values)
    #d=d.where(d.values < 9999.)  
    d = d.groupby('times.month').mean()
    #ubah nan di xr to '--'
    d=d.fillna(-100)
    #d=d.fillna(--) gak mau tapi '--'  
    #print(target_dataset.values)
    #print(d.values)
    mask_array = np.zeros(d.values.shape)
    index=np.where(d.values== -100)
    mask_array[index]=1
    #mask_array
    
    #mask_array=d.to_masked_array(copy=False)
    
    #2 ini sudah ok asal tidak groupby....
    #mask_array=target_dataset.values.mask
    #dd=np.ma.masked_array(d.values, mask= mask_array)
    
    #mask_array=d.where(d.values=='nan')
    #mask_array=d.where(d.values==-100)
    #d.coords['mask'] = (('times','lats', 'lons'), mask_array)
    dd=ma.array(d.values, mask= mask_array)
    
    #print(dd)
    
    
    #print(d)
    #d=d.dropna(dim=("lats"), how="any")
    #d=d.dropna(dim=("lons"), how="any")
    new_dataset = ds.Dataset(target_dataset.lats,
                             target_dataset.lons,
                             d.month,
                             #d.times,
                             dd,
                             variable=target_dataset.variable,
                             units=target_dataset.units,
                             name=target_dataset.name)
    #new_dataset = mask_missing_data([new_dataset])
    #print('new_dataset=',new_dataset[0].values)
    
    return new_dataset
    
def season_subset(target_dataset):
    import xarray as xr
   
    d = xr.DataArray(target_dataset.values,
    coords={'times': target_dataset.times,
            'lats': target_dataset.lats, 
            'lons': target_dataset.lons},       
    dims=["times", "lats", "lons"])
   
    d = d.groupby('times.season').mean()
   # print(d.season)
    season1=config['temporal_season']['season_name']
    #d1=d['season'].sel(season=season1)
   # print(d1.values)
    #print('')
    #d2=d.season.sel(season=season1)
   # print(d2)
   
    #pilih angka untuk fill na
    d=d.fillna(-100)
  
    mask_array = np.zeros(d.values.shape)
    index=np.where(d.values== -100)
    mask_array[index]=1
   
    dd=ma.array(d.values, mask= mask_array)
    
    #print(dd)
  
    #print('season',season, dd)
    #data=
    
    new_dataset = ds.Dataset(target_dataset.lats,
                             target_dataset.lons,
                             d.season,
                             #d.times,
                             dd,
                             variable=target_dataset.variable,
                             units=target_dataset.units,
                             name=target_dataset.name)
    return new_dataset

def temporal_rebin(target_dataset, temporal_resolution):
    """ Rebin a Dataset to a new temporal resolution

    :param target_dataset: Dataset object that needs temporal rebinned
    :type target_dataset: :class:`dataset.Dataset`

    :param temporal_resolution: The new temporal resolution
    :type temporal_resolution: :mod:`string`

    :returns: A new temporally rebinned Dataset
    :rtype: :class:`dataset.Dataset`
    """
    # Decode the temporal resolution into a string format that
    # _rcmes_calc_average_on_new_time_unit_K() can understand

    binned_values, binned_dates = _rcmes_calc_average_on_new_time_unit(
        target_dataset.values, target_dataset.times, temporal_resolution)
    binned_dates = np.array(binned_dates)
    new_dataset = ds.Dataset(target_dataset.lats,
                             target_dataset.lons,
                             binned_dates,
                             binned_values,
                             variable=target_dataset.variable,
                             units=target_dataset.units,
                             name=target_dataset.name,
                             origin=target_dataset.origin)
    return new_dataset


def temporal_rebin_with_time_index(target_dataset, nt_average):
    """ Rebin a Dataset to a new temporal resolution

    :param target_dataset: Dataset object that needs temporal rebinned
    :type target_dataset: :class:`dataset.Dataset`

    :param nt_average: Time resolution for the output datasets.
     It is the same as the number of time indicies to be averaged.
     length of time dimension in the rebinned dataset) =
     (original time dimension length/nt_average)
    :type nt_average: integer

    :returns: A new temporally rebinned Dataset
    :rtype: :class:`dataset.Dataset`
    """
    nt = target_dataset.times.size
    if nt % nt_average != 0:
        logger.warning('Length of time dimension must '
                       'be a multiple of nt_average')
    # nt2 is the length of time dimension in the rebinned dataset
    nt2 = nt // nt_average
    binned_dates = target_dataset.times[np.arange(nt2) * nt_average]
    binned_values = ma.zeros(
        np.insert(target_dataset.values.shape[1:], 0, nt2))
    for it in np.arange(nt2):
        binned_values[it, :] = ma.average(
            target_dataset.values[nt_average * it:
                                  nt_average * it + nt_average,
                                  :],
            axis=0)
    new_dataset = ds.Dataset(target_dataset.lats,
                             target_dataset.lons,
                             binned_dates,
                             binned_values,
                             variable=target_dataset.variable,
                             units=target_dataset.units,
                             name=target_dataset.name,
                             origin=target_dataset.origin)
    return new_dataset


def spatial_regrid(target_dataset, new_latitudes, new_longitudes,
                   boundary_check=True):
    """ Regrid a Dataset using the new latitudes and longitudes

    :param target_dataset: Dataset object that needs spatially regridded
    :type target_dataset: :class:`dataset.Dataset`

    :param new_latitudes: Array of latitudes
    :type new_latitudes: :class:`numpy.ndarray`

    :param new_longitudes: Array of longitudes
    :type new_longitudes: :class:`numpy.ndarray`

    :param boundary_check:  Check if the regriding domain's boundaries
                            are outside target_dataset's domain
    :type boundary_check: :class:'bool'

    :returns: A new spatially regridded Dataset
    :rtype: :class:`dataset.Dataset`
    """
    #print(target_dataset)
    #print(np.mean(target_dataset.values))
    # Create grids of the given lats and lons for the underlying API
    # NOTE: np.meshgrid() requires inputs (x, y) and returns data
    #       of shape(y|lat|rows, x|lon|columns).  So we pass in lons, lats
    #       and get back data.shape(lats, lons)
    if target_dataset.lons.ndim == 1 and target_dataset.lats.ndim == 1:
        regular_grid = True
        lons, lats = np.meshgrid(target_dataset.lons, target_dataset.lats)
    else:
        regular_grid = False
        lons = target_dataset.lons
        lats = target_dataset.lats
    if new_longitudes.ndim == 1 and new_latitudes.ndim == 1:
        new_lons, new_lats = np.meshgrid(new_longitudes, new_latitudes)
    else:
        new_lons = new_longitudes
        new_lats = new_latitudes

    ny_old, nx_old = lats.shape
    ny_new, nx_new = new_lats.shape

    for iy in np.arange(ny_old):
        if not all(x < y for x, y in zip(lons[iy, :], lons[iy, 1:])):
            lons[iy, :][lons[iy, :] < 0] = lons[iy, :][lons[iy, :] < 0] + 360.

    # Make masked array of shape (times, new_latitudes,new_longitudes)
    new_values = ma.zeros([len(target_dataset.times),
                           ny_new, nx_new])+1.e+20

    # Boundary vertices of target_dataset
    vertices = []

    if regular_grid:
        vertices.append([lons[0, 0], lats[0, 0]])
        vertices.append([lons[-1, 0], lats[-1, 0]])
        vertices.append([lons[-1, -1], lats[-1, -1]])
        vertices.append([lons[0, -1], lats[0, -1]])
    else:
        # from south to north along the west boundary
        for iy in np.arange(ny_old):
            vertices.append([lons[iy, 0], lats[iy, 0]])
        # from west to east along the north boundary
        for ix in np.arange(nx_old):
            vertices.append([lons[-1, ix], lats[-1, ix]])
        # from north to south along the east boundary
        for iy in np.arange(ny_old)[::-1]:
            vertices.append([lons[iy, -1], lats[iy, -1]])
        # from east to west along the south boundary
        for ix in np.arange(nx_old)[::-1]:
            vertices.append([lons[0, ix], lats[0, ix]])
    path = Path(vertices)
    new_xy_mask = np.ones(new_lats.shape)

    for iy in np.arange(ny_new):
        print(iy, end="")
        for ix in np.arange(nx_new):
            if path.contains_point([new_lons[iy, ix],
                                    new_lats[iy, ix]]) or not boundary_check:
               new_xy_mask[iy, ix] = 0.

    new_index = np.where(new_xy_mask == 0.)
    # Regrid the data on each time slice
    for i in range(len(target_dataset.times)):
        if len(target_dataset.times) == 1 and target_dataset.values.ndim == 2:
            values_original = ma.array(target_dataset.values)
        else:
            values_original = ma.array(target_dataset.values[i])
        new_mask = np.copy(values_original.mask)
        #print(new_mask)
        for shift in (-1, 1):
            for axis in (0, 1):
                q_shifted = np.roll(values_original, shift=shift, axis=axis)
                if (np.where((values_original.mask == True) & (q_shifted.mask == False)))[0].size !=0:
                    index1 =np.where((values_original.mask == True) & (q_shifted.mask == False))
                    n_indices = len(index1[0])
                    values_original.data[index1] = q_shifted[index1]
                    new_mask[index1] = np.repeat(False, n_indices)
                    mask_index = np.where(~new_mask)
        if new_mask.size != 1:
            mask_index = np.where(~new_mask)
        else:
            mask_index = np.where(~np.isnan(values_original))
        #print(mask_index)
        
        #mask_index=mask_index-100
        #print(mask_index)
        #print(new_lons[mask_index])
        new_values_temp = griddata((lons[mask_index], 
                                    lats[mask_index]), 
                           values_original[mask_index],
                              (new_lons[new_index],
                               new_lats[new_index]),
                               method='linear')
        
        #print(np.mean(values_original))
        #print(np.mean(new_values_temp))
        #scipy.interpolate.griddata(points, values, xi, 
           #method='linear', fill_value=nan, rescale=False)

        # Make a masking map using nearest neighbour interpolation -use this to
        # determine locations with MDI and mask these
        qmdi = np.zeros_like(values_original)
        values_true_indices = np.where(values_original.mask == True)
        values_false_indices = np.where(values_original.mask == False)
        qmdi[values_true_indices] = 1.
        qmdi[values_false_indices] = 0.
        qmdi_r = griddata((lons.flatten(), lats.flatten()), qmdi.flatten(),
                             (new_lons[new_index],
                              new_lats[new_index]),
                              method='nearest')
        new_values_temp = ma.masked_where(qmdi_r != 0.0, new_values_temp)
        # Combine missing data mask, with outside domain mask define above.
        new_values[i, new_index[0], new_index[1]] = new_values_temp[:]
        new_values[i,:] = ma.masked_equal(new_values[i,:], 1.e+20)
    
    
    # TODO:
    # This will call down to the _congrid() function and the lat and lon
    # axis will be adjusted with the time axis being held constant

    # Create a new Dataset Object to return using new data
    regridded_dataset = ds.Dataset(new_latitudes,
                                   new_longitudes,
                                   target_dataset.times,
                                   new_values,
                                   variable=target_dataset.variable,
                                   units=target_dataset.units,
                                   name=target_dataset.name,
                                   origin=target_dataset.origin)
    return regridded_dataset


def ensemble(datasets):
    """
    Generate a single dataset which is the mean of the input datasets

    An ensemble datasets combines input datasets assuming the all have
    similar shape, dimensions, and units.

    :param datasets: Datasets to be used to compose the ensemble dataset from.
        All Datasets must be the same shape.
    :type datasets: :class:`list` of :class:`dataset.Dataset`

     :returns: New Dataset with a name of 'Dataset Ensemble'
    :rtype: :class:`dataset.Dataset`
    """
    _check_dataset_shapes(datasets)
    dataset_values = [dataset.values for dataset in datasets]
    ensemble_values = ma.mean(dataset_values, axis=0)

    # Build new dataset object from the input datasets and
    # the ensemble values and return it
    ensemble_dataset = ds.Dataset(datasets[0].lats,
                                  datasets[0].lons,
                                  datasets[0].times,
                                  ensemble_values,
                                  units=datasets[0].units,
                                  name="Dataset Ensemble")

    return ensemble_dataset
    
#tambahan weight ensemble
def ensemble_weighted(datasets,weights):
    """
    Generate a single dataset which is the mean of the input datasets

    An ensemble datasets combines input datasets assuming the all have
    similar shape, dimensions, and units.

    :param datasets: Datasets to be used to compose the ensemble dataset from.
        All Datasets must be the same shape.
    :type datasets: :class:`list` of :class:`dataset.Dataset`

     :returns: New Dataset with a name of 'Dataset Ensemble'
    :rtype: :class:`dataset.Dataset`
    """
    _check_dataset_shapes(datasets)
    dataset_values = [dataset.values for dataset in datasets]
    #print(dataset_values)
    
    #weights=ma.array([1,1,1,1,1,1])
    #ensemble_values = ma.mean(dataset_values, axis=0)
    #ensemble_values = ma.average(ma.dot(dataset_values[1],weights), axis=0)#, weights=weights)
    
    ensemble_values = ma.average(dataset_values, axis=0, weights=weights)
    
    # Build new dataset object from the input datasets and
    # the ensemble values and return it
    ensemble_dataset = ds.Dataset(datasets[0].lats,
                                  datasets[0].lons,
                                  datasets[0].times,
                                  ensemble_values,
                                  units=datasets[0].units,
                                  name="Dataset Ensemble")

    return ensemble_dataset

def subset(target_dataset, subregion, subregion_name=None, extract=True, user_mask_values=[1]):
    '''Subset given dataset(s) with subregion information

    :param subregion: The Bounds with which to subset the target Dataset.
    :type subregion: :class:`dataset.Bounds`

    :param target_dataset: The Dataset object to subset.
    :type target_dataset: :class:`dataset.Dataset`

    :param subregion_name: The subset-ed Dataset name
    :type subregion_name: :mod:`string`

    :param extract: If False, the dataset inside regions will be masked.
    :type extract: :mod:`boolean`

    :param user_mask_value: grid points where mask_variable == user_mask_value will be extracted or masked .
    :type user_mask_value: :mod:`int`

    :returns: The subset-ed Dataset object
    :rtype: :class:`dataset.Dataset`

    :raises: ValueError
    '''

    if not subregion.start:
        time_start = target_dataset.times[0]
        time_end = target_dataset.times[-1]
    else:
        time_start = subregion.start
        time_end = subregion.end

    if not subregion_name:
        subregion_name = target_dataset.name

    if hasattr(subregion, 'lat_min'):
        _are_bounds_contained_by_dataset(target_dataset, subregion)

        if target_dataset.lats.ndim == 2 and target_dataset.lons.ndim == 2:
            temporal_subset = temporal_slice(
                target_dataset, time_start, time_end)
            nt = temporal_subset.values.shape[0]
            y_index, x_index = np.where(
                (target_dataset.lats >= subregion.lat_max) | (
                    target_dataset.lats <= subregion.lat_min) |
                (target_dataset.lons >= subregion.lon_max) | (
                    target_dataset.lons <= subregion.lon_min))
            for it in np.arange(nt):
                temporal_subset.values[it, y_index, x_index] = 1.e+20
            new_values = ma.masked_equal(
                temporal_subset.values, 1.e+20)
            return ds.Dataset(
                target_dataset.lats,
                target_dataset.lons,
                temporal_subset.times,
                new_values,
                variable=target_dataset.variable,
                units=target_dataset.units,
                name=subregion_name,
                origin=target_dataset.origin)

        elif target_dataset.lats.ndim == 1 and target_dataset.lons.ndim == 1:
            # Get subregion indices into subregion data
            dataset_slices = _get_subregion_slice_indices(target_dataset, subregion)
            
            '''
            dataset_slices=
            from
            return {
                    "lat_start": latStart,
                    "lat_end": latEnd,
                    "lon_start": lonStart,
                    "lon_end": lonEnd,
                    "time_start": timeStart,
                    "time_end": timeEnd
            }
            '''
            
            # Slice the values array with our calculated slice indices
            if target_dataset.values.ndim == 2:
                subset_values = ma.zeros([len(target_dataset.values[
                    dataset_slices["lat_start"]:dataset_slices["lat_end"]]),
                    len(target_dataset.values[
                        dataset_slices["lon_start"]:dataset_slices["lon_end"]])])

                subset_values = target_dataset.values[
                    dataset_slices["lat_start"]:dataset_slices["lat_end"] + 1,
                    dataset_slices["lon_start"]:dataset_slices["lon_end"] + 1]

            elif target_dataset.values.ndim == 3:
                subset_values = ma.zeros([len(target_dataset.values[
                    dataset_slices["time_start"]:dataset_slices["time_end"]]),
                    len(target_dataset.values[
                        dataset_slices["lat_start"]:dataset_slices["lat_end"]]),
                    len(target_dataset.values[
                        dataset_slices["lon_start"]:dataset_slices["lon_end"]])])

                subset_values = target_dataset.values[
                    dataset_slices["time_start"]:dataset_slices["time_end"] + 1,
                    dataset_slices["lat_start"]:dataset_slices["lat_end"] + 1,
                    dataset_slices["lon_start"]:dataset_slices["lon_end"] + 1]
            
            # Build new dataset with subset information
            return ds.Dataset(
                # Slice the lats array with our calculated slice indices
                target_dataset.lats[dataset_slices["lat_start"]:
                                    dataset_slices["lat_end"] + 1],
                # Slice the lons array with our calculated slice indices
                target_dataset.lons[dataset_slices["lon_start"]:
                                    dataset_slices["lon_end"] + 1],
                # Slice the times array with our calculated slice indices
                target_dataset.times[dataset_slices["time_start"]:
                                     dataset_slices["time_end"] + 1],
                # Slice the values array with our calculated slice indices
                subset_values,
                variable=target_dataset.variable,
                units=target_dataset.units,
                name=subregion_name,
                origin=target_dataset.origin
            )

    if subregion.boundary_type == 'us_states' or subregion.boundary_type == 'countries':
        temporal_subset = temporal_slice(
            target_dataset, time_start, time_end)
        spatial_mask = utils.mask_using_shapefile_info(target_dataset.lons, target_dataset.lats,
                                                       subregion.masked_regions, extract=extract)
        subset_values = utils.propagate_spatial_mask_over_time(
            temporal_subset.values, mask=spatial_mask)
        return ds.Dataset(
            target_dataset.lats,
            target_dataset.lons,
            temporal_subset.times,
            subset_values,
            variable=target_dataset.variable,
            units=target_dataset.units,
            name=subregion_name,
            origin=target_dataset.origin)

    if subregion.boundary_type == 'user':
        temporal_subset = temporal_slice(
            target_dataset, time_start, time_end)
        spatial_mask = utils.regrid_spatial_mask(target_dataset.lons, target_dataset.lats,
                                                 subregion.mask_longitude, subregion.mask_latitude, subregion.mask_variable,
                                                 user_mask_values, extract=extract)
        subset_values = utils.propagate_spatial_mask_over_time(
            temporal_subset.values, mask=spatial_mask)
        return ds.Dataset(
            target_dataset.lats,
            target_dataset.lons,
            temporal_subset.times,
            subset_values,
            variable=target_dataset.variable,
            units=target_dataset.units,
            name=subregion_name,
            origin=target_dataset.origin)

def spatial_slice(target_dataset, lat_min, lat_max,lon_min, lon_max):
    #tambahan berhasil                                 
    
    latStart = min(np.nonzero(target_dataset.lats >= lat_min)[0])
    latEnd = max(np.nonzero(target_dataset.lats <= lat_max)[0])

    lonStart = min(np.nonzero(target_dataset.lons >= lon_min)[0])
    lonEnd = max(np.nonzero(target_dataset.lons <= lon_max)[0])
    
    subset_values = ma.zeros([len(target_dataset.values[:]),
                              len(target_dataset.values[latStart:latEnd]),
                              len(target_dataset.values[lonStart:lonEnd])])
    
    subset_values = target_dataset.values[:, latStart:latEnd + 1, 
                                             lonStart: lonEnd + 1]
   
    # Build new dataset with subset information
    return ds.Dataset(
                # Slice the lats array with our calculated slice indices
                target_dataset.lats[latStart:latEnd + 1],
                                    
                # Slice the lons array with our calculated slice indices
                target_dataset.lons[lonStart:lonEnd+ 1],
                                    
                # Slice the times
                target_dataset.times[:],
               
                # Fill with sliced values
                subset_values,
                variable=target_dataset.variable,
                units=target_dataset.units,
                
                origin=target_dataset.origin
            )

    
   

def temporal_slice(target_dataset, start_time, end_time):
    '''Temporally slice given dataset(s) with subregion information. This does not
    spatially subset the target_Dataset

    :param start_time: start time
    :type start_time: :class:'int'

    :param end_time: end time
    :type end_time: :class:'datetime.datetime'

    :param target_dataset: The Dataset object to subset.
    :type target_dataset: :class:`dataset.Dataset`

    :returns: The subset-ed Dataset object
    :rtype: :class:`dataset.Dataset`

    :raises: ValueError
    '''

    # https://issues.apache.org/jira/browse/CLIMATE-938
    # netCDF datetimes allow for a variety of calendars while Python has
    # only one.  This would throw an error about a calendar mismatch when
    # comparing a Python datetime object to a netcdf datetime object.
    # Cast the date as best we can so the comparison will compare like
    # data types  This will still throw an excdeption if the start / end date are
    # not valid in given calendar.  February 29th in a DatetimeNoLeap calendar for example.
    slice_start_time = start_time
    slice_end_time = end_time

    if isinstance(target_dataset.times.item(0), utime):
        slice_start_time =\
            type(target_dataset.times.item(0))(start_time.year, start_time.month, start_time.day,
                                            start_time.hour, start_time.minute, start_time.second)

        slice_end_time =\
            type(target_dataset.times.item(0))(end_time.year, end_time.month, end_time.day,
                                            end_time.hour, end_time.minute, end_time.second)

    start_time_index = np.where(
        target_dataset.times >= slice_start_time)[0][0]

    end_time_index = np.where(
        target_dataset.times <= slice_end_time)[0][-1]

    new_times = target_dataset.times[start_time_index:end_time_index + 1]
    new_values = target_dataset.values[start_time_index:end_time_index + 1, :]

    return ds.Dataset(
        target_dataset.lats,
        target_dataset.lons,
        new_times,
        new_values,
        variable=target_dataset.variable,
        units=target_dataset.units,
        origin=target_dataset.origin)

def safe_subset(target_dataset, subregion, subregion_name=None):
    '''Safely subset given dataset with subregion information
    A standard subset requires that the provided subregion be entirely
    contained within the datasets bounds. `safe_subset` returns the
    overlap of the subregion and dataset without returning an error.

    :param subregion: The Bounds with which to subset the target Dataset.
    :type subregion: :class:`dataset.Bounds`

    :param target_dataset: The Dataset object to subset.
    :type target_dataset: :class:`dataset.Dataset`

    :param subregion_name: The subset-ed Dataset name
    :type subregion_name: :mod:`string`

    :returns: The subset-ed Dataset object
    :rtype: :class:`dataset.Dataset`
    '''

    lat_min, lat_max, lon_min, lon_max = target_dataset.spatial_boundaries()
    start, end = target_dataset.temporal_boundaries()
    if subregion.lat_min:
        if subregion.lat_min < lat_min:
            subregion.lat_min = lat_min

        if subregion.lat_max > lat_max:
            subregion.lat_max = lat_max

        if subregion.lon_min < lon_min:
            subregion.lon_min = lon_min

        if subregion.lon_max > lon_max:
            subregion.lon_max = lon_max

    if subregion.start:
        if subregion.start < start:
            subregion.start = start

    if subregion.end:
        if subregion.end > end:
            subregion.end = end

    return subset(target_dataset, subregion, subregion_name)


def normalize_dataset_datetimes(dataset, timestep):
    ''' Normalize Dataset datetime values.

    Force daily to an hour time value of 00:00:00.
    Force monthly data to the first of the month at midnight.

    :param dataset: The Dataset which will have its time value normalized.
    :type dataset: :class:`dataset.Dataset`

    :param timestep: The timestep of the Dataset's values. Either 'daily' or
        'monthly'.
    :type timestep: :mod:`string`

    :returns: A new Dataset with normalized datetime values.
    :rtype: :class:`dataset.Dataset`
    '''
    new_times = _rcmes_normalize_datetimes(dataset.times, timestep)
    return ds.Dataset(
        dataset.lats,
        dataset.lons,
        np.array(new_times),
        dataset.values,
        variable=dataset.variable,
        units=dataset.units,
        name=dataset.name,
        origin=dataset.origin
    )


def write_netcdf(dataset, path, compress=True):
    ''' Write a dataset to a NetCDF file.

    :param dataset: The dataset to write.
    :type dataset: :class:`dataset.Dataset`

    :param path: The output file path.
    :type path: :mod:`string`
    '''
    out_file = netCDF4.Dataset(path, 'w', format='NETCDF4')

    # Set attribute lengths
    if dataset.lats.ndim == 2:
        lat_len = dataset.lats.shape[0]
        lon_len = dataset.lons.shape[1]
        lat_dim_info = ('lat', 'lon')
        lon_dim_info = ('lat', 'lon')

    else:
        lat_len = len(dataset.lats)
        lon_len = len(dataset.lons)
        lat_dim_info = ('lat',)
        lon_dim_info = ('lon',)

    time_len = len(dataset.times)

    # Create attribute dimensions
    out_file.createDimension('lat', lat_len)
    out_file.createDimension('lon', lon_len)
    out_file.createDimension('time', time_len)

    # Create variables
    lats = out_file.createVariable('lat', 'f8', lat_dim_info, zlib=compress)
    lons = out_file.createVariable('lon', 'f8', lon_dim_info, zlib=compress)
    times = out_file.createVariable('time', 'f8', ('time',), zlib=compress)

    var_name = dataset.variable if dataset.variable else 'var'
    values = out_file.createVariable(var_name,
                                     'f8',
                                     ('time', 'lat', 'lon'),
                                     zlib=compress)

    # Set the time variable units
    # We don't deal with hourly/minutely/anything-less-than-a-day data so
    # we can safely stick with a 'days since' offset here. Note that the
    # NetCDF4 helper date2num doesn't support 'months' or 'years' instead
    # of days.
    times.units = "days since %s" % dataset.times[0]

    # Store the dataset's values
    lats[:] = dataset.lats
    lons[:] = dataset.lons
    times[:] = netCDF4.date2num(dataset.times, times.units)
    values[:] = dataset.values
    values.units = dataset.units

    out_file.close()


def write_netcdf_multiple_datasets_with_subregions(ref_dataset, ref_name,
                                                   model_dataset_array,
                                                   model_names,
                                                   path,
                                                   subregions=None,
                                                   subregion_array=None,
                                                   ref_subregion_mean=None,
                                                   ref_subregion_std=None,
                                                   model_subregion_mean=None,
                                                   model_subregion_std=None):
    # Write multiple reference and model datasets and their subregional means
    # and standard deivations in a NetCDF file.

    # :To be updated
    #
    out_file = netCDF4.Dataset(path, 'w', format='NETCDF4')
    #out_file.close()

    dataset = ref_dataset
    # Set attribute lenghts
    nobs = 1
    nmodel = len(model_dataset_array)
    lat_len, lon_len = dataset.values.shape[1:]
    lat_ndim = dataset.lats.ndim
    lon_ndim = dataset.lons.ndim
    time_len = len(dataset.times)

    if subregions is not None:
        nsubregion = len(subregions)

    # Create attribute dimensions
    out_file.createDimension('y', lat_len)
    out_file.createDimension('x', lon_len)
    out_file.createDimension('time', time_len)

    # Create variables and store the values
    if lat_ndim == 2:
        lats = out_file.createVariable('lat', 'f8', ('y', 'x'))
    else:
        lats = out_file.createVariable('lat', 'f8', ('y'))
    lats[:] = dataset.lats
    if lon_ndim == 2:
        lons = out_file.createVariable('lon', 'f8', ('y', 'x'))
    else:
        lons = out_file.createVariable('lon', 'f8', ('x'))
    lons[:] = dataset.lons
    times = out_file.createVariable('time', 'f8', ('time',))
    times.units = "days since %s" % dataset.times[0]
    times[:] = netCDF4.date2num(dataset.times, times.units)
    
    #print('dataset.times=',dataset.times)
    #print('times[:]=',times[:])
    # mask_array = np.zeros([time_len, lat_len, lon_len])
    # for iobs in np.arange(nobs):
    #    index = np.where(ref_dataset_array[iobs].values.mask[:] == True)
    #    mask_array[index] = 1
    out_file.createVariable(ref_name, 'f8', ('time', 'y', 'x'))
    out_file.variables[ref_name][:] = ref_dataset.values
    out_file.variables[ref_name].units = ref_dataset.units
    print('mean_ref_dataset=',ref_dataset.values.mean())
    
    for imodel in np.arange(nmodel):
        out_file.createVariable(model_names[imodel], 'f8', ('time', 'y', 'x'))
        # out_file.variables[model_names[imodel]][:] = ma.array(
        #    model_dataset_array[imodel].values, mask = mask_array)
        out_file.variables[model_names[imodel]][:] = model_dataset_array[
            imodel].values
        out_file.variables[model_names[imodel]].units = model_dataset_array[
            imodel].units

    if subregions is not None:
        out_file.createVariable('subregion_array', 'i4', ('y', 'x'))
        out_file.variables['subregion_array'][:] = subregion_array[:]
        nsubregion = len(subregions)
        out_file.createDimension('nsubregion', nsubregion)
        out_file.createDimension('nobs', nobs)
        out_file.createDimension('nmodel', nmodel)
        out_file.createVariable('obs_subregion_mean', 'f8', ('nobs',
                                                             'time',
                                                             'nsubregion'))
        out_file.variables['obs_subregion_mean'][:] = ref_subregion_mean[:]
        out_file.createVariable('obs_subregion_std', 'f8', ('nobs',
                                                            'time',
                                                            'nsubregion'))
        out_file.variables['obs_subregion_std'][:] = ref_subregion_std[:]
        out_file.createVariable('model_subregion_mean', 'f8', ('nmodel',
                                                               'time',
                                                               'nsubregion'))
        out_file.variables['model_subregion_mean'][:] = model_subregion_mean[:]
        out_file.createVariable('model_subregion_std', 'f8', ('nmodel',
                                                              'time',
                                                              'nsubregion'))
        out_file.variables['model_subregion_std'][:] = model_subregion_std[:]

    out_file.close()

def write_netcdf_multiple_datasets_with_subregions2(ref_dataset, ref_name,
                                                   model_dataset_array,
                                                   model_names,
                                                   path,
                                                   subregions=None,
                                                   subregion_array=None,
                                                   ref_subregion_mean=None,
                                                   ref_subregion_std=None,
                                                   model_subregion_mean=None,
                                                   model_subregion_std=None):
    # Write multiple reference and model datasets and their subregional means
    # and standard deivations in a NetCDF file.

    # :To be updated
    #
    nmodel = len(model_dataset_array)
    for imodel in np.arange(nmodel):
        out_file = netCDF4.Dataset(path+model_names[imodel]+'.nc', 'w', format='NETCDF4')
        #out_file.close()

        dataset = ref_dataset
        # Set attribute lenghts
        nobs = 1
        
        lat_len, lon_len = dataset.values.shape[1:]
        lat_ndim = dataset.lats.ndim
        lon_ndim = dataset.lons.ndim
        time_len = len(dataset.times)

        if subregions is not None:
            nsubregion = len(subregions)

        # Create attribute dimensions
        out_file.createDimension('y', lat_len)
        out_file.createDimension('x', lon_len)
        out_file.createDimension('time', time_len)

        # Create variables and store the values
        if lat_ndim == 2:
            lats = out_file.createVariable('lat', 'f8', ('y', 'x'))
        else:
            lats = out_file.createVariable('lat', 'f8', ('y'))
        lats[:] = dataset.lats
        if lon_ndim == 2:
            lons = out_file.createVariable('lon', 'f8', ('y', 'x'))
        else:
            lons = out_file.createVariable('lon', 'f8', ('x'))
        lons[:] = dataset.lons
        times = out_file.createVariable('time', 'f8', ('time',))
        times.units = "days since %s" % dataset.times[0]
        times[:] = netCDF4.date2num(dataset.times, times.units)
        
        #print('dataset.times=',dataset.times)
        #print('times[:]=',times[:])
        # mask_array = np.zeros([time_len, lat_len, lon_len])
        # for iobs in np.arange(nobs):
        #    index = np.where(ref_dataset_array[iobs].values.mask[:] == True)
        #    mask_array[index] = 1
        #out_file.createVariable(ref_name, 'f8', ('time', 'y', 'x'))
        #out_file.variables[ref_name][:] = ref_dataset.values
        #out_file.variables[ref_name].units = ref_dataset.units
        #for imodel in np.arange(nmodel):
        out_file.createVariable('pr', 'f8', ('time', 'y', 'x'))
        # out_file.variables[model_names[imodel]][:] = ma.array(
        #    model_dataset_array[imodel].values, mask = mask_array)
        out_file.variables['pr'][:] = model_dataset_array[
            imodel].values
        out_file.variables['pr'].units = model_dataset_array[
            imodel].units

    if subregions is not None:
        out_file.createVariable('subregion_array', 'i4', ('y', 'x'))
        out_file.variables['subregion_array'][:] = subregion_array[:]
        nsubregion = len(subregions)
        out_file.createDimension('nsubregion', nsubregion)
        out_file.createDimension('nobs', nobs)
        out_file.createDimension('nmodel', nmodel)
        out_file.createVariable('obs_subregion_mean', 'f8', ('nobs',
                                                             'time',
                                                             'nsubregion'))
        out_file.variables['obs_subregion_mean'][:] = ref_subregion_mean[:]
        out_file.createVariable('obs_subregion_std', 'f8', ('nobs',
                                                            'time',
                                                            'nsubregion'))
        out_file.variables['obs_subregion_std'][:] = ref_subregion_std[:]
        out_file.createVariable('model_subregion_mean', 'f8', ('nmodel',
                                                               'time',
                                                               'nsubregion'))
        out_file.variables['model_subregion_mean'][:] = model_subregion_mean[:]
        out_file.createVariable('model_subregion_std', 'f8', ('nmodel',
                                                              'time',
                                                              'nsubregion'))
        out_file.variables['model_subregion_std'][:] = model_subregion_std[:]

    out_file.close()
    
def write_netcdf_multiple_datasets_with_subregions3(ref_dataset, ref_name,
                                                   model_dataset_array,
                                                   model_names,
                                                   path,
                                                   subregions=None,
                                                   subregion_array=None,
                                                   ref_subregion_mean=None,
                                                   ref_subregion_std=None,
                                                   model_subregion_mean=None,
                                                   model_subregion_std=None):
    # Write multiple reference and model datasets and their subregional means
    # and standard deivations in a NetCDF file.

    # :To be updated
    #
    
    out_file = netCDF4.Dataset(path+ref_name+'.nc', 'w', format='NETCDF4')
    #out_file.close()

    dataset = ref_dataset
    # Set attribute lenghts
    nobs = 1
    nmodel = len(model_dataset_array)
    lat_len, lon_len = dataset.values.shape[1:]
    lat_ndim = dataset.lats.ndim
    lon_ndim = dataset.lons.ndim
    time_len = len(dataset.times)

    if subregions is not None:
        nsubregion = len(subregions)

    # Create attribute dimensions
    out_file.createDimension('lat', lat_len)
    out_file.createDimension('lon', lon_len)
    out_file.createDimension('time', time_len)

    # Create variables and store the values
    if lat_ndim == 2:
        lats = out_file.createVariable('lat', 'f8', ('lat', 'lon'))
    else:
        lats = out_file.createVariable('lat', 'f8', ('lat'))
    lats[:] = dataset.lats
    if lon_ndim == 2:
        lons = out_file.createVariable('lon', 'f8', ('lat', 'lon'))
    else:
        lons = out_file.createVariable('lon', 'f8', ('lon'))
    lons[:] = dataset.lons
    times = out_file.createVariable('time', 'f8', ('time',))
    times.units = "days since %s" % dataset.times[0]
    times[:] = netCDF4.date2num(dataset.times, times.units)
    
    #print('dataset.times=',dataset.times)
    #print('times[:]=',times[:])
    # mask_array = np.zeros([time_len, lat_len, lon_len])
    # for iobs in np.arange(nobs):
    #    index = np.where(ref_dataset_array[iobs].values.mask[:] == True)
    #    mask_array[index] = 1
    out_file.createVariable('pr', 'f8', ('time', 'lat', 'lon'))
    out_file.variables['pr'][:] = ref_dataset.values
    out_file.variables['pr'].units = ref_dataset.units
    out_file.close()
    for imodel in np.arange(nmodel):
        out_file = netCDF4.Dataset(path+model_names[imodel]+'.nc', 'w', format='NETCDF4')
        #out_file.close()
        # Create attribute dimensions
        out_file.createDimension('lat', lat_len)
        out_file.createDimension('lon', lon_len)
        out_file.createDimension('time', time_len)

        # Create variables and store the values
        if lat_ndim == 2:
            lats = out_file.createVariable('lat', 'f8', ('lat', 'lon'))
        else:
            lats = out_file.createVariable('lat', 'f8', ('lat'))
        lats[:] = dataset.lats
        if lon_ndim == 2:
            lons = out_file.createVariable('lon', 'f8', ('lat', 'lon'))
        else:
            lons = out_file.createVariable('lon', 'f8', ('lon'))
        lons[:] = dataset.lons
        times = out_file.createVariable('time', 'f8', ('time',))
        times.units = "days since %s" % dataset.times[0]
        times[:] = netCDF4.date2num(dataset.times, times.units)
        out_file.createVariable('pr', 'f8', ('time', 'lat', 'lon'))
        # out_file.variables[model_names[imodel]][:] = ma.array(
        #    model_dataset_array[imodel].values, mask = mask_array)
        out_file.variables['pr'][:] = model_dataset_array[
            imodel].values
        out_file.variables['pr'].units = model_dataset_array[
            imodel].units
        out_file.close()

def write_netcdf_multiple_datasets_with_subregions4(
                                                   datasets,
                                                   model_names,
                                                   path,
                                                   subregions=None,
                                                   subregion_array=None,
                                                   ref_subregion_mean=None,
                                                   ref_subregion_std=None,
                                                   model_subregion_mean=None,
                                                   model_subregion_std=None):
    # Set attribute lenghts

    nmodel = len(datasets)
    lat_len, lon_len = datasets[0].values.shape[1:]
    lat_ndim = datasets[0].lats.ndim
    lon_ndim = datasets[0].lons.ndim
    time_len = len(datasets[0].times)

    if subregions is not None:
        nsubregion = len(subregions)

   
    for imodel in np.arange(nmodel):
        out_file = netCDF4.Dataset(path+model_names[imodel]+'.nc', 'w', format='NETCDF4')
        #out_file.close()
        # Create attribute dimensions
        out_file.createDimension('lat', lat_len)
        out_file.createDimension('lon', lon_len)
        out_file.createDimension('time', time_len)

        # Create variables and store the values
        if lat_ndim == 2:
            lats = out_file.createVariable('lat', 'f8', ('lat', 'lon'))
        else:
            lats = out_file.createVariable('lat', 'f8', ('lat'))
        lats[:] = datasets[0].lats
        
        if lon_ndim == 2:
            lons = out_file.createVariable('lon', 'f8', ('lat', 'lon'))
        else:
            lons = out_file.createVariable('lon', 'f8', ('lon'))
        lons[:] = datasets[0].lons
        
        times = out_file.createVariable('time', 'f8', ('time',))
        times.units = "days since %s" % datasets[0].times[0]
        times[:] = netCDF4.date2num(datasets[0].times, times.units)
        
        out_file.createVariable('pr', 'f8', ('time', 'lat', 'lon'))
        # out_file.variables[model_names[imodel]][:] = ma.array(
        #    model_dataset_array[imodel].values, mask = mask_array)
        out_file.variables['pr'][:] = datasets[imodel].values
        out_file.variables['pr'].units = datasets[imodel].units
        out_file.variables['pr'].standard_name = "precipitation_flux"
        out_file.variables['pr'].cell_methods = "time: mean"
        out_file.close() 
    
def water_flux_unit_conversion(dataset):
    ''' Convert water flux variables units as necessary

    Convert full SI units water flux units to more common units.

    :param dataset: The dataset to convert.
    :type dataset: :class:`dataset.Dataset`

    :returns: A Dataset with values converted to new units.
    :rtype: :class:`dataset.Dataset`
    '''
    water_flux_variables = ['pr', 'prec', 'evspsbl', 'mrro', 'swe']
    variable = dataset.variable.lower()

    if any(sub_string in variable for sub_string in water_flux_variables):
        dataset_units = dataset.units.lower()
        if variable in 'swe':
            if any(unit in dataset_units for unit in ['m', 'meter']):
                dataset.values = 1.e3 * dataset.values
                dataset.units = 'km'
        else:
            if any(unit in dataset_units
                    for unit in ['kg m-2 s-1', 'mm s-1', 'mm/sec']):
                dataset.values = 86400. * dataset.values
                dataset.units = 'mm/day'

    return dataset


def temperature_unit_conversion(dataset):
    ''' Convert temperature units as necessary \
    Automatically convert Celcius to Kelvin in the given dataset.

    :param dataset: The dataset for which units should be updated. \
    :type dataset; :class:`dataset.Dataset`

    :returns: The dataset with (potentially) updated units. \
    :rtype: :class:`dataset.Dataset`

    '''

    temperature_variables = ['temp', 'tas', 'tasmax', 'taxmin', 'T', 'tg', 'tos','sst']
    variable = dataset.variable.lower()
    #print('dsp 1361 dataset.variable n units',dataset.variable, dataset.units)
    #K to C
    if any(sub_string in variable for sub_string in temperature_variables):
        dataset_units = dataset.units.lower()
        if dataset_units == 'k' or dataset_units == 'kelvin':
            print('konversi dari K ke C')
            dataset.values = -273.15 + dataset.values
            dataset.units = 'C'
    else:
        print('temperature_variable belum ada dalam list konversi')

    return dataset
    
def precipitation_unit_conversion(dataset):
    ''' Convert temperature units as necessary \
    Automatically convert Celcius to Kelvin in the given dataset.

    :param dataset: The dataset for which units should be updated. \
    :type dataset; :class:`dataset.Dataset`

    :returns: The dataset with (potentially) updated units. \
    :rtype: :class:`dataset.Dataset`

    '''

    pr_variables = ['prec', 'pre','pr', 'rr', 'precip', 'p', 'precipitation','tp']
    variable = dataset.variable.lower()
    
    if any(sub_string in variable for sub_string in pr_variables):
        dataset_units = dataset.units.lower()
        if  dataset_units == 'kg m-2 s-1' :
            dataset.values =  dataset.values*86400
            dataset.units = 'mm/day'
        elif dataset_units == 'mm/month' : #khusus mon_sum to mon_mean sementara=cru/30 
             dataset.values =  dataset.values/30
             dataset.units = 'mm'
        elif dataset_units == 'm' : #khusus mon_sum to mon_mean sementara=cru/30 
             dataset.values =  dataset.values*1000
             dataset.units = 'mm/day'
        

    return dataset


def variable_unit_conversion(dataset):
    ''' Convert water flux or temperature variables units as necessary

    For water flux variables, convert full SI units water flux units
    to more common units.
    For temperature, convert Celcius to Kelvin.

    :param dataset: The dataset to convert.
    :type dataset: :class:`dataset.Dataset`

    :returns: A Dataset with values converted to new units.
    :rtype: :class:`dataset.Dataset`
    '''

    #dataset = water_flux_unit_conversion(dataset)
    #dataset = temperature_unit_conversion(dataset)
    dataset = precipitation_unit_conversion(dataset)

    return dataset


def _rcmes_normalize_datetimes(datetimes, timestep):
    """ Normalize Dataset datetime values.

    Force daily to an hour time value of 00:00:00.
    Force monthly data to the first of the month at midnight.

    :param datetimes: The datetimes to normalize.
    :type datetimes: List of `datetime` values.

    :param timestep: The flag for how to normalize the datetimes.
    :type timestep: String
    """

    normalDatetimes = []
    if timestep.lower() == 'monthly':
        for inputDatetime in datetimes:
            if inputDatetime.day != 1:
                # Clean the inputDatetime
                inputDatetimeString = inputDatetime.strftime('%Y%m%d')
                normalInputDatetimeString = inputDatetimeString[:6] + '01'
                inputDatetime = datetime.datetime.strptime(
                    normalInputDatetimeString, '%Y%m%d')

            normalDatetimes.append(inputDatetime)

    elif timestep.lower() == 'daily':
        for inputDatetime in datetimes:
            if (inputDatetime.hour != 0 or inputDatetime.minute != 0 or
                    inputDatetime.second != 0):
                datetimeString = inputDatetime.strftime('%Y%m%d%H%M%S')
                normalDatetimeString = datetimeString[:8] + '000000'
                #aslinya tanpa try:
                try:
                    inputDatetime = datetime.datetime.strptime(
                        normalDatetimeString, '%Y%m%d%H%M%S')
                except ValueError:
                    continue

            normalDatetimes.append(inputDatetime)

    return normalDatetimes


def mask_missing_data(dataset_array):
    ''' Check missing values in observation and model datasets.
    If any of dataset in dataset_array has missing values at a grid point,
    the values at the grid point in all other datasets are masked.
    :param dataset_array: an array of OCW datasets
    '''

    mask_array = np.zeros(dataset_array[0].values.shape)
    for dataset in dataset_array:
        # CLIMATE-797 - Not every array passed in will be a masked array.
        # For those that are, action based on the mask passed in.
        # For those that are not, take no action (else AttributeError).
        if hasattr(dataset.values, 'mask'):
            index = np.where(dataset.values.mask == True)
            if index[0].size > 0:
                mask_array[index] = 1
    masked_array = []
    for dataset in dataset_array:
        dataset.values = ma.array(dataset.values, mask=mask_array)
        masked_array.append(dataset)
    return [masked_dataset for masked_dataset in masked_array]


def deseasonalize_dataset(dataset):
    '''Calculate daily climatology and subtract the climatology from
    the input dataset

    :param dataset: The dataset to convert.
    :type dataset: :class:`dataset.Dataset`

    :returns: A Dataset with values converted to new units.
    :rtype: :class:`dataset.Dataset`
    '''

    days = [d.month * 100. + d.day for d in dataset.times]
    days_sorted = np.unique(days)
    ndays = days_sorted.size
    nt, ny, nx = dataset.values.shape
    values_clim = ma.zeros([ndays, ny, nx])
    for iday, day in enumerate(days_sorted):
        t_index = np.where(days == day)[0]
        values_clim[iday, :] = ma.mean(dataset.values[t_index, :], axis=0)
    for iday, day in enumerate(days_sorted):
        t_index = np.where(days == day)[0]
        dataset.values[t_index, :] = dataset.values[
            t_index, :] - values_clim[iday, :]
    return dataset


def _rcmes_spatial_regrid(spatial_values, lat, lon, lat2, lon2, order=1):
    '''
    Spatial regrid from one set of lat,lon values onto a new set (lat2,lon2)

    :param spatial_values: Values in a spatial grid that need to be regridded
    :type spatial_values: 2d masked numpy array.  shape (latitude, longitude)
    :param lat: Grid of latitude values which map to the spatial values
    :type lat: 2d numpy array. shape(latitudes, longitudes)
    :param lon: Grid of longitude values which map to the spatial values
    :type lon: 2d numpy array. shape(latitudes, longitudes)
    :param lat2: Grid of NEW latitude values to regrid the spatial_values onto
    :type lat2: 2d numpy array. shape(latitudes, longitudes)
    :param lon2: Grid of NEW longitude values to regrid the spatial_values onto
    :type lon2: 2d numpy array. shape(latitudes, longitudes)
    :param order: Interpolation order flag.  1=bi-linear, 3=cubic spline
    :type order: [optional] Integer

    :returns: 2d masked numpy array with shape(len(lat2), len(lon2))
    :rtype: (float, float)
    '''

    nlat = spatial_values.shape[0]
    nlon = spatial_values.shape[1]

    nlat2 = lat2.shape[0]
    nlon2 = lon2.shape[1]

    # To make our lives easier down the road, let's
    # turn these into arrays of x & y coords
    loni = lon2.ravel()
    lati = lat2.ravel()

    loni = loni.copy()  # NB. it won't run unless you do this...
    lati = lati.copy()

    # Now, we'll set points outside the boundaries to lie along an edge
    loni_max_indices = np.where(loni > lon.max())
    loni_min_indices = np.where(loni < lon.min())
    loni[loni_max_indices] = lon.max()
    loni[loni_min_indices] = lon.min()

    # To deal with the "hard" break, we'll have to treat y differently,
    # so we're just setting the min here...
    lati_max_indices = np.where(lati > lat.max())
    lati_min_indices = np.where(lati < lat.min())
    lati[lati_max_indices] = lat.max()
    lati[lati_min_indices] = lat.min()

    # We need to convert these to (float) indicies
    #   (xi should range from 0 to (nx - 1), etc)
    loni = (nlon - 1) * (loni - lon.min()) / (lon.max() - lon.min())

    # Deal with the "hard" break in the y-direction
    lati = (nlat - 1) * (lati - lat.min()) / (lat.max() - lat.min())

    """
    TODO: Review this docstring and see if it still holds true.
    NOTE:  This function doesn't use MDI currently.  These are legacy comments

    Notes on dealing with MDI when regridding data.
     Method adopted here:
       Use bilinear interpolation of data by default
       (but user can specify other order using order=... in call)
       Perform bilinear interpolation of data, and of mask.
       To be conservative, new grid point which contained some missing data on
       the old grid is set to missing data.
               -this is achieved by looking for any non-zero interpolated
               mask values.
       To avoid issues with bilinear interpolation producing strong gradients
       leading into the MDI, set values at MDI points to mean data value so
       little gradient visible = not ideal, but acceptable for now.

    Set values in MDI so that similar to surroundings so don't produce large
    gradients when interpolating Preserve MDI mask, by only changing data part
    of masked array object.
    """
    for shift in (-1, 1):
        for axis in (0, 1):
            q_shifted = np.roll(spatial_values, shift=shift, axis=axis)
            idx = ~q_shifted.mask * spatial_values.mask
            indices = np.where(idx)
            spatial_values.data[indices] = q_shifted[indices]

    # Now we actually interpolate
    # map_coordinates does cubic interpolation by default,
    # use "order=1" to preform bilinear interpolation instead...

    regridded_values = map_coordinates(spatial_values,
                                       [lati, loni],
                                       order=order)
    regridded_values = regridded_values.reshape([nlat2, nlon2])

    # Set values to missing data outside of original domain
    regridded_values = ma.masked_array(regridded_values,
                                       mask=np.logical_or(
                                           np.logical_or(lat2 >= lat.max(),
                                                         lat2 <= lat.min()),
                                           np.logical_or(lon2 <= lon.min(),
                                                         lon2 >= lon.max())))

    # Make second map using nearest neighbour interpolation -use this to
    # determine locations with MDI and mask these
    qmdi = np.zeros_like(spatial_values)
    spatial_indices_true = np.where(spatial_values.mask)
    spatial_indices_false = np.where(not spatial_values.mask)
    qmdi[spatial_indices_true] = 1.
    qmdi[spatial_indices_false] = 0.
    qmdi_r = map_coordinates(qmdi, [lati, loni], order=order)
    qmdi_r = qmdi_r.reshape([nlat2, nlon2])
    mdimask = (qmdi_r != 0.0)

    # Combine missing data mask, with outside domain mask define above.
    regridded_values.mask = np.logical_or(mdimask, regridded_values.mask)

    return regridded_values


def _rcmes_create_mask_using_threshold(masked_array, threshold=0.5):
    '''Mask an array if percent of values missing data is above a threshold.

    For each value along an axis, if the proportion of steps that are missing
    data is above ``threshold`` then the value is marked as missing data.

    ..note:: The 0th axis is currently always used.

    :param masked_array: Masked array of data
    :type masked_array: Numpy Masked Array
    :param threshold: (optional) Threshold proportion above which a value is
        marked as missing data.
    :type threshold: Float

    :returns: A Numpy array describing the mask for masked_array.
    '''

    # try, except used as some model files don't have a full mask, but a single
    # bool the except catches this situation and deals with it appropriately.
    try:
        nT = masked_array.mask.shape[0]

        # For each pixel, count how many times are masked.
        nMasked = masked_array.mask[:, :, :].sum(axis=0)

        # Define new mask as when a pixel has over a defined threshold ratio of
        # masked data
        #   e.g. if the threshold is 75%, and there are 10 times,
        #        then a pixel will be masked if more than 5 times are masked.
        mymask = nMasked > (nT * threshold)

    except:
        mymask = np.zeros_like(masked_array.data[0, :, :])

    return mymask


def _rcmes_calc_average_on_new_time_unit(data, dates, unit):
    """ Rebin 3d array and list of dates using the provided unit parameter

    :param data: Input data that needs to be averaged
    :type data: 3D masked numpy array of shape (times, lats, lons)
    :param dates: List of dates that correspond to the given data values
    :type dates: Python datetime objects
    :param unit: Time unit to average the data into
    :type unit: String matching one of these values :
        full | annual | monthly | daily

    :returns: meanstorem, newTimesList
    :rtype: 3D numpy masked array the same shape as the input array,
            list of python datetime objects
    :raises: ValueError
    """

    # Check if the user-selected temporal grid is valid. If not, EXIT
    acceptable = ((unit == 'full') | (unit == 'annual') |
                  (unit == 'monthly') | (unit == 'daily'))
    if not acceptable:
        raise ValueError('Error: unknown unit type selected '
                         'for time averaging: EXIT')

    nt, ny, nx = data.shape
    if unit == 'full':
        new_data = ma.mean(data, axis=0)
        new_date = [dates[dates.size // 2]]
    if unit == 'annual':
        years = [d.year for d in dates]
        years_sorted = np.unique(years)
        new_data = ma.zeros([years_sorted.size, ny, nx])
        it = 0
        new_date = []
        for year in years_sorted:
            index = np.where(years == year)[0]
            new_data[it, :] = ma.mean(data[index, :], axis=0)
            new_date.append(datetime.datetime(year=year, month=7, day=2))
            it = it + 1
    if unit == 'monthly':
        years = [d.year for d in dates]
        years_sorted = np.unique(years)
        months = [d.month for d in dates]
        months_sorted = np.unique(months)

        new_data = ma.zeros([years_sorted.size * months_sorted.size, ny, nx])
        it = 0
        new_date = []
        for year in years_sorted:
            for month in months_sorted:
                index = np.where((years == year) & (months == month))[0]
                new_data[it, :] = ma.mean(data[index, :], axis=0)
                new_date.append(datetime.datetime(year=year,
                                                  month=month,
                                                  day=15))
                it = it + 1
    if unit == 'daily':
        days = [d.year * 10000. + d.month * 100. + d.day for d in dates]
        days_sorted = np.unique(days)

        new_data = ma.zeros([days_sorted.size, ny, nx])
        it = 0
        new_date = []
        for day in days_sorted:
            index = np.where(days == day)[0]
            new_data[it, :] = ma.mean(data[index, :], axis=0)
            y = int(day // 10000)
            m = int(day % 10000) // 100
            d = int(day % 100)
            new_date.append(datetime.datetime(year=y, month=m, day=d))
            it = it + 1

    return new_data, np.array(new_date)


def _rcmes_calc_average_on_new_time_unit_K(data, dates, unit):
    """ Rebin 3d array and list of dates using the provided unit parameter

    :param data: Input data that needs to be averaged
    :type data: 3D masked numpy array of shape (times, lats, lons)
    :param dates: List of dates that correspond to the given data values
    :type dates: Python datetime objects
    :param unit: Time unit to average the data into
    :type unit: String matching one of these values :
                full | annual | monthly | daily

    :returns: meanstorem, newTimesList
    :rtype: 3D numpy masked array the same shape as the input array,
            list of python datetime objects
    :raises: ValueError
    """

    # Check if the user-selected temporal grid is valid. If not, EXIT
    acceptable = ((unit == 'full') | (unit == 'annual') |
                  (unit == 'monthly') | (unit == 'daily'))
    if not acceptable:
        raise ValueError('Error: unknown unit type selected '
                         'for time averaging: EXIT')

    # Calculate arrays of: annual timeseries: year (2007,2007),
    #                      monthly time series: year-month (200701,200702),
    #                      daily timeseries: year-month-day (20070101,20070102)
    #  depending on user-selected averaging period.

    # Year list
    if unit == 'annual':
        timeunits = np.array([int(d.strftime("%Y")) for d in dates])
        unique_times = np.unique(timeunits)

    # YearMonth format list
    if unit == 'monthly':
        timeunits = np.array([int(d.strftime("%Y%m")) for d in dates])
        unique_times = np.unique(timeunits)

    # YearMonthDay format list
    if unit == 'daily':
        timeunits = np.array([int(d.strftime("%Y%m%d")) for d in dates])
        unique_times = np.unique(timeunits)

    # TODO: add pentad setting using Julian days?

    # Full list: a special case
    if unit == 'full':
        # Calculating means data over the entire time range:
        # i.e., annual-mean climatology
        timeunits = []
        for i in np.arange(len(dates)):
            # i.e. we just want the same value for all times.
            timeunits.append(999)
        timeunits = np.array(timeunits, dtype=int)
        unique_times = np.unique(timeunits)

    # empty list to store new times
    newTimesList = []

    # Decide whether or not you need to do any time averaging.
    #   i.e. if data are already on required time unit then just
    #        pass data through and calculate and return representative
    #        datetimes.
    processing_required = True
    if len(timeunits) == (len(unique_times)):
        processing_required = False

    # 1D data arrays, i.e. time series
    if data.ndim == 1:
        # Create array to store the resulting data
        meanstore = np.zeros(len(unique_times))

        # Calculate the means across each unique time unit
        i = 0
        for myunit in unique_times:
            if processing_required:
                datam = ma.masked_array(data, timeunits != myunit)
                meanstore[i] = ma.average(datam)

            # construct new times list
            yyyy, mm, dd = _create_new_year_month_day(myunit, dates)
            newTimesList.append(datetime.datetime(yyyy, mm, dd, 0, 0, 0, 0))
            i = i + 1

    # 3D data arrays
    if data.ndim == 3:
        # Create array to store the resulting data
        meanstore = np.zeros([len(unique_times), data.shape[1], data.shape[2]])

        # Calculate the means across each unique time unit
        i = 0
        datamask_store = []
        for myunit in unique_times:
            if processing_required:
                mask = np.zeros_like(data)
                mask[timeunits != myunit, :, :] = 1.0
                # Calculate missing data mask within each time unit...
                datamask_at_this_timeunit = np.zeros_like(data)
                datamask_at_this_timeunit[:] = _rcmes_create_mask_using_threshold(
                    data[timeunits == myunit, :, :], threshold=0.75)
                # Store results for masking later
                datamask_store.append(datamask_at_this_timeunit[0])
                # Calculate means for each pixel in this time unit, ignoring
                # missing data (using masked array).
                datam = ma.masked_array(data, np.logical_or(mask,
                                                            datamask_at_this_timeunit))
                meanstore[i, :, :] = ma.average(datam, axis=0)
            # construct new times list
            yyyy, mm, dd = _create_new_year_month_day(myunit, dates)
            newTimesList.append(datetime.datetime(yyyy, mm, dd))
            i += 1

        if not processing_required:
            meanstorem = data

        if processing_required:
            # Create masked array (using missing data mask defined above)
            datamask_store = np.array(datamask_store)
            meanstorem = ma.masked_array(meanstore, datamask_store)

    return meanstorem, newTimesList


def _create_new_year_month_day(time_unit, dates):
    smyunit = str(time_unit)
    if len(smyunit) == 4:  # YYYY
        yyyy = int(smyunit[0:4])
        mm = 1
        dd = 1
    elif len(smyunit) == 6:  # YYYYMM
        yyyy = int(smyunit[0:4])
        mm = int(smyunit[4:6])
        dd = 1
    elif len(smyunit) == 8:  # YYYYMMDD
        yyyy = int(smyunit[0:4])
        mm = int(smyunit[4:6])
        dd = int(smyunit[6:8])
    elif len(smyunit) == 3:  # Full time range
        # Need to set an appropriate time representing the mid-point of the
        # entire time span
        dt = dates[-1] - dates[0]
        halfway = dates[0] + (dt / 2)
        yyyy = int(halfway.year)
        mm = int(halfway.month)
        dd = int(halfway.day)

    return (yyyy, mm, dd)


def _congrid(a, newdims, method='linear', centre=False, minusone=False):
    '''
    This function is from http://wiki.scipy.org/Cookbook/Rebinning - Example 3
    It has been refactored and changed a bit, but the original functionality
    has been preserved.
    Arbitrary resampling of source array to new dimension sizes.
    Currently only supports maintaining the same number of dimensions.
    To use 1-D arrays, first promote them to shape (x,1).

    Uses the same parameters and creates the same co-ordinate lookup points
    as IDL''s congrid routine, which apparently originally came from a VAX/VMS
    routine of the same name.

    method:
    neighbour - closest value from original data
    nearest and linear - uses n x 1-D interpolations using
                         scipy.interpolate.interp1d
    (see Numerical Recipes for validity of use of n 1-D interpolations)
    spline - uses ndimage.map_coordinates

    centre:
    True - interpolation points are at the centres of the bins
    False - points are at the front edge of the bin

    minusone:
    For example- inarray.shape = (i,j) & new dimensions = (x,y)
    False - inarray is resampled by factors of (i/x) * (j/y)
    True - inarray is resampled by(i-1)/(x-1) * (j-1)/(y-1)
    This prevents extrapolation one element beyond bounds of input array.
    '''
    if a.dtype not in [np.float64, np.float32]:
        a = np.cast[float](a)

    # this will merely take the True/False input and convert it to an
    # array(1) or array(0)
    m1 = np.cast[int](minusone)
    # this also casts the True False input into a floating point number
    # of 0.5 or 0.0
    ofs = np.cast[int](centre) * 0.5

    old = np.array(a.shape)
    ndims = len(a.shape)
    if len(newdims) != ndims:
        print("[congrid] dimensions error. "
              "This routine currently only supports "
              "rebinning to the same number of dimensions.")
        return None
    newdims = np.asarray(newdims, dtype=float)
    dimlist = []

    if method == 'neighbour':
        newa = _congrid_neighbor(a, newdims, m1, ofs)

    elif method in ['nearest', 'linear']:
        # calculate new dims
        for i in range(ndims):
            base = np.arange(newdims[i])
            dimlist.append((old[i] - m1) / (newdims[i] - m1) *
                           (base + ofs) - ofs)
        # specify old dims
        olddims = [np.arange(i, dtype=np.float) for i in list(a.shape)]

        # first interpolation - for ndims = any
        mint = scipy.interpolate.interp1d(olddims[-1], a, kind=method)
        newa = mint(dimlist[-1])

        trorder = [ndims - 1] + range(ndims - 1)
        for i in range(ndims - 2, -1, -1):
            newa = newa.transpose(trorder)

            mint = scipy.interpolate.interp1d(olddims[i], newa, kind=method)
            newa = mint(dimlist[i])

        if ndims > 1:
            # need one more transpose to return to original dimensions
            newa = newa.transpose(trorder)

        return newa
    elif method in ['spline']:
        nslices = [slice(0, j) for j in list(newdims)]
        newcoords = np.mgrid[nslices]

        newcoords_dims = range(np.rank(newcoords))
        # make first index last
        newcoords_dims.append(newcoords_dims.pop(0))
        newcoords_tr = newcoords.transpose(newcoords_dims)
        # makes a view that affects newcoords

        newcoords_tr += ofs

        deltas = (np.asarray(old) - m1) / (newdims - m1)
        newcoords_tr *= deltas

        newcoords_tr -= ofs

        newa = scipy.ndimage.map_coordinates(a, newcoords)
        return newa
    else:
        print("Congrid error: Unrecognized interpolation type.\n",
              "Currently only \'neighbour\', \'nearest\',\'linear\',",
              "and \'spline\' are supported.")
        return None


def _check_dataset_shapes(datasets):
    """ If the  datasets are not the same shape throw a ValueError Exception

    :param datasets: OCW Datasets to check for a consistent shape
    :type datasets: List of OCW Dataset Objects

    :raises: ValueError
    """
    dataset_shape = None
    for dataset in datasets:
        if dataset_shape is None:
            dataset_shape = dataset.values.shape
        else:
            if dataset.values.shape != dataset_shape:
                msg = "%s != %s" % (dataset.values.shape, dataset_shape)
                raise ValueError("Input datasets must be the same shape "
                                 "for an ensemble :: ", msg)
            else:
                pass


def _congrid_neighbor(values, new_dims, minus_one, offset):
    """ Use the nearest neighbor to create a new array

    :param values: Array of values that need to be interpolated
    :type values: Numpy ndarray
    :param new_dims: Longitude resolution in degrees
    :type new_dims: float
    :param lat_resolution: Latitude resolution in degrees
    :type lat_resolution: float

    :returns: A new spatially regridded Dataset
    :rtype: Open Climate Workbench Dataset Object
    """
    ndims = len(values.shape)
    dimlist = []
    old_dims = np.array(values.shape)
    for i in range(ndims):
        base = np.indices(new_dims)[i]
        dimlist.append((old_dims[i] - minus_one) / (new_dims[i] - minus_one) *
                       (base + offset) - offset)
    cd = np.array(dimlist).round().astype(int)
    new_values = values[list(cd)]
    return new_values


def _are_bounds_contained_by_dataset(dataset, bounds):
    '''Check if a Dataset fully contains a bounds.

    :param bounds: The Bounds object to check.
    :type bounds: Bounds
    :param dataset: The Dataset that should be fully contain the
        Bounds
    :type dataset: Dataset

    :returns: True if the Bounds are contained by the Dataset, Raises
        a ValueError otherwise
    '''
    lat_min, lat_max, lon_min, lon_max = dataset.spatial_boundaries()
    start, end = dataset.temporal_boundaries()

    errors = []

    # TODO:  THIS IS TERRIBLY inefficent and we need to use a geometry
    # lib instead in the future
    if (lat_min > bounds.lat_max):
        error = ("bounds.lat_max: %s is not between lat_min: %s and"
                 " lat_max: %s" % (bounds.lat_max, lat_min, lat_max))
        errors.append(error)

    if (lon_min > bounds.lon_max):
        error = ("bounds.lon_max: %s is not between lon_min: %s and"
                 " lon_max: %s" % (bounds.lon_max, lon_min, lon_max))
        errors.append(error)
    if not (bounds.start is None):
        if not start <= bounds.start <= end:
            error = ("bounds.start: %s is not between start: %s and end: %s" %
                     (bounds.start, start, end))
            errors.append(error)

    if not (bounds.end is None):
        if not start <= bounds.end <= end:
            error = ("bounds.end: %s is not between start: %s and end: %s" %
                     (bounds.end, start, end))
            errors.append(error)

    if len(errors) == 0:
        return True
    else:
        error_message = '\n'.join(errors)
        raise ValueError(error_message)


def _get_subregion_slice_indices(target_dataset, subregion):
    '''Get the indices for slicing Dataset values to generate the subregion.

    :param subregion: The Bounds that specify the subset of the Dataset
        that should be extracted.
    :type subregion: Bounds
    :param target_dataset: The Dataset to subset.
    :type target_dataset: Dataset

    :returns: The indices to slice the Datasets arrays as a Dictionary.
    '''
    latStart = min(np.nonzero(target_dataset.lats >= subregion.lat_min)[0])
    latEnd = max(np.nonzero(target_dataset.lats <= subregion.lat_max)[0])

    lonStart = min(np.nonzero(target_dataset.lons >= subregion.lon_min)[0])
    lonEnd = max(np.nonzero(target_dataset.lons <= subregion.lon_max)[0])

    if not (subregion.start is None):
        timeStart = min(np.nonzero(target_dataset.times >= subregion.start)[0])
        timeEnd = max(np.nonzero(target_dataset.times <= subregion.end)[0])
    else:
        timeStart = 0
        timeEnd = len(target_dataset.times) -1

    return {
        "lat_start": latStart,
        "lat_end": latEnd,
        "lon_start": lonStart,
        "lon_end": lonEnd,
        "time_start": timeStart,
        "time_end": timeEnd
    }
