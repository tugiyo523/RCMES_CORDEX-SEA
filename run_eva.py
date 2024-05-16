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

from __future__ import print_function
import os
import sys
import ssl
import yaml
import operator
from datetime import datetime
from glob import glob
from getpass import getpass
import numpy as np
import ocw.utils as utils
import ocw.dataset_processor as dsp
from ocw.dataset import Bounds
from ocw.dataset_loader import DatasetLoader
from metrics_and_plots import *
#tambahan
import numpy.ma as ma
import ocw.metrics as metrics
from ocw.evaluation import Evaluation
from scipy.stats import mstats
from numpy import argsort

print('IPB CORDEX-SEA evaluation system v1.0')
print('')

def load_datasets_from_config(*loader_opts):
    '''
    Generic dataset loading function.
    '''
    for opt in loader_opts:
        loader_name = opt['loader_name']
        
    loader = DatasetLoader(*loader_opts)
    loader.load_datasets()
    return loader.datasets

if hasattr(ssl, '_create_unverified_context'):
    ssl._create_default_https_context = ssl._create_unverified_context

config_file = str(sys.argv[1])

reg=config['region']

print('Reading the configuration file ...')
config = yaml.safe_load(open(config_file))

# Get the dataset loader options
data_info = config['datasets']


""" Step 1: Load the datasets """
#print('Loading datasets ...\n{}'.format(data_info))
print('Loading datasets ...')
datasets = load_datasets_from_config(*data_info)

names = [dataset.name for dataset in datasets]
print('Datasets:',names)

# # koding stop running jika diperlukan
# cek = input('Continue ? (y/n):')
# if cek=='n':
    # exit()
# else:
    # print('OK...')

reference_dataset = datasets[0]
target_datasets = datasets[1:]
reference_name = names[0]
target_names = names[1:]


#if config['data_time_ ?']==1:
if config['set_data_annual?']==1:
    #annual subetting
    reference_dataset = dsp.temporal_subset(reference_dataset, month_start, month_end)
    for i, dataset in enumerate(target_datasets):
        target_datasets[i] = dsp.temporal_subset(dataset, month_start, month_end)

'''
#SET DJF etc ...case for wind
if config['temporal_subset']==1:
    month_start=config['start_month']
    month_end=config['end_month']
    
    reference_dataset = dsp.temporal_subset(reference_dataset, month_start, month_end, average_each_year=False)
    for i, dataset in enumerate(target_datasets):
        target_datasets[i] = dsp.temporal_subset(dataset, month_start, month_end,average_each_year=False)


#non-aktifkan jika tidak perlu
#spatial subset Sumatera
reference_dataset = dsp.spatial_slice(reference_dataset, 
                    lat_min=-6, lat_max=6, 
                    lon_min=95, lon_max=107)
for i, dataset in enumerate(target_datasets):
    target_datasets[i] = dsp.spatial_slice(dataset, 
                    lat_min=-6, lat_max=6, 
                    lon_min=95, lon_max=107)

'''
print('')
print('main temporal subsetting =>', reference_dataset.values.shape)

#path
workdir = config['workdir']
if workdir[-1] != '/':
    workdir = workdir+'/'

#PI dipakai yang mengandung independece
#topsis untuk metrik tanpa perlu normalize D,SS,R,RMSE, ... pakai impact +/-

'''
if config['wavelet']== 1:
   print('tes')
   wavelet_14sum_pycwt(reference_dataset, reference_name, target_datasets, target_names, workdir)
#wavelet2(reference_dataset, reference_name, target_datasets, target_names, workdir)

if config['fft']== 1:
    #fft(reference_dataset, reference_name, target_datasets, target_names, workdir)
    #fft_14(reference_dataset, reference_name, target_datasets, target_names, workdir)

    fft_14f(reference_dataset, reference_name, target_datasets, target_names, workdir)

'''

#Fitur evaluasi berikut bisa dipanggil langsung dengan mengaktifkan perintah
#atau mensetting angka 1 pada namelist inputnya pada file konfigurasinya (yaml)

seasons = config['seasons']
if config['mean_map']== 1:
    mean_rainfall_5obs(seasons, reference_dataset, reference_name, target_datasets, target_names, workdir)


#Buat ansambel dengan faktor bobot
#dsp.mmew(reference_dataset, target_datasets, target_names, workdir)
#dsp.mmew2(reference_dataset, target_datasets, target_names, workdir)
#exit()

#ac_station(reference_dataset, reference_name, target_datasets, target_names, workdir)

#ac_taylor(reference_dataset, reference_name, target_datasets, target_names, workdir)

#All_metrics(reference_dataset, reference_name, target_datasets, target_names, workdir)

#annual_cycle(reference_dataset, reference_name, target_datasets, target_names, workdir)
#annual_cycle_sum2R(reference_dataset, reference_name, target_datasets, target_names, workdir)

#cluster_trend3(reference_dataset, reference_name, target_datasets, target_names, workdir)


#corr_enso(reference_dataset, reference_name, target_datasets, target_names, workdir)
#corr_enso(reference_dataset, reference_name, target_datasets, target_names, workdir)
#corr_enso_obs(reference_dataset, reference_name, target_datasets, target_names, workdir)
#corr_enso_tos(reference_dataset, reference_name, target_datasets, target_names, workdir)
#corr_enso_tos_season(reference_dataset, reference_name, target_datasets, target_names, workdir)
#corr_enso3_season(reference_dataset, reference_name, target_datasets, target_names, workdir)
#corr_iod_tos_season(reference_dataset, reference_name, target_datasets, target_names, workdir)

#corr_pc1(reference_dataset, reference_name, target_datasets, target_names, workdir)
#corr_pc1_1(workdir)
#corr_pc1_xr(workdir)

#corr_spatial(reference_dataset, reference_name, target_datasets, target_names, workdir)

#corr_temporal(reference_dataset, reference_name, target_datasets, target_names, workdir)
#corr_temporal_annual_cycle(reference_dataset, reference_name, target_datasets, target_names, workdir)
#corr_temporal_season(reference_dataset, reference_name, target_datasets, target_names, workdir)

#cv_mon(names, reference_dataset, reference_name, target_datasets, target_names, workdir)
#cv_year(names, reference_dataset, reference_name, target_datasets, target_names, workdir)

#dmi_anomaly(reference_dataset, workdir)
#dmi_anomaly(reference_dataset, workdir)
#dmi_corr(reference_dataset, reference_name, target_datasets, target_names, workdir)
#dmi_stdev(reference_dataset, reference_name, target_datasets, target_names, workdir)

#dsc(reference_dataset, reference_name, target_datasets, target_names, workdir)
#dsc2(reference_dataset, reference_name, target_datasets, target_names, workdir)

#enso_partial_corr(reference_dataset, reference_name, target_datasets, target_names, workdir)
#enso_partial_corr2(reference_dataset, reference_name, target_datasets, target_names, workdir)

#enso_stdev(reference_dataset, reference_name, target_datasets, target_names, workdir)
#enso_ts_cmip5(reference_dataset, workdir)
#enso_zomean_cmip5(reference_dataset, workdir)

#eofs_multi(reference_dataset, reference_name, target_datasets, target_names, workdir)
#eofs_multi_tos_nino(reference_dataset, reference_name, target_datasets, target_names, workdir)
#eofs_multi_tos_nino2(reference_dataset, workdir)

#exit()

#fft_15(reference_dataset, reference_name, target_datasets, target_names, workdir)
#fft_15b(reference_dataset, reference_name, target_datasets, target_names, workdir)
#fft_11(reference_dataset, reference_name, target_datasets, target_names, workdir)
if config['fft']== 1:
    fft_5obs(reference_dataset, reference_name, target_datasets, target_names, workdir)

#g=metrics.decompose_Gauch(reference_dataset.values,reference_name, target_datasets[1].values,target_names[1])
#g=metrics.decompose_Gauch(target_datasets[1].values, reference_dataset.values)
#g=metrics.decompose_Gauch2(reference_dataset.values, target_datasets, target_names, workdir)

#hirarki_clustering(reference_dataset, reference_name, target_datasets, target_names, workdir)

#iod_ens2(reference_dataset, reference_name, target_datasets, target_names, workdir)
#iod_ens2(reference_dataset, reference_name, target_datasets, target_names, workdir)
#iod_ts_cmip5(reference_dataset, workdir)

import time as timer
start_time = timer.time()
#dbscan_sea22(reference_dataset, reference_name, target_datasets, target_names, workdir)
#kmeans_metrics(reference_dataset, reference_name, target_datasets, target_names, workdir)
#kmeans_metrics2(reference_dataset, reference_name, target_datasets, target_names, workdir)
#kmeans_sea22(reference_dataset, reference_name, target_datasets, target_names, workdir)
#kmeans_masking(reference_dataset, reference_name, target_datasets, target_names, workdir)
#kmeans2a(reference_dataset, reference_name, target_datasets, target_names, workdir)
#kmeans_dcm(reference_dataset, reference_name, target_datasets, target_names, workdir)
#kmeans_hirarki(reference_dataset, reference_name, target_datasets, target_names, workdir)
#kmeans_sr(reference_dataset, reference_name, target_datasets, target_names, workdir)
#kmeans_season(reference_dataset, reference_name, target_datasets, target_names, workdir)

elapsed_time = timer.time() - start_time
m, s = divmod(elapsed_time, 60)
print('time='+'%.0f'%m+'m:'+'%.2f'%s+'s')


#lat_mean_rainfall2(reference_dataset, reference_name, target_datasets, target_names, workdir)

#Map_plot_bias_of_multiyear_climatology2(reference_dataset, reference_name, target_datasets, target_names,workdir)

#mean_rainfall_15_season(reference_dataset, reference_name, target_datasets, target_names, workdir)
#mean_rainfall_11era_season(reference_dataset, reference_name, target_datasets, target_names, workdir)
#mean_rainfall_14_season_bias(reference_dataset, reference_name, target_datasets, target_names, workdir)

#mean_rainfall_10era_season_bias(reference_dataset, reference_name, target_datasets, target_names, workdir)
#mean_rainfall_season_bias_5obs(reference_dataset, reference_name, target_datasets, target_names, workdir)
#mean_rainfall_5obs(reference_dataset, reference_name, target_datasets, target_names, workdir)
#mean_rainfall_11_jambi(reference_dataset, reference_name, target_datasets, target_names, workdir)

#exit()

#metrik_enso2(reference_dataset, reference_name, target_datasets, target_names, workdir)
#metrik_iod(reference_dataset, reference_name, target_datasets, target_names, workdir)

#nino_corr(reference_dataset, reference_name, target_datasets, target_names, workdir)
#nino_eof(reference_dataset, reference_name, target_datasets, target_names, workdir)
#nino34_anomaly(reference_dataset, workdir)
#nino34_dmi_anomaly(reference_dataset, workdir)
#nino34_ens(reference_dataset, reference_name, target_datasets, target_names, workdir)
#nino34_ens_corr(reference_dataset, reference_name, target_datasets, target_names, workdir)

#obs5_cross_corr(reference_dataset, reference_name, target_datasets, target_names, workdir)

#p_trend(reference_dataset, reference_name, target_datasets, target_names, workdir)
#p_trend2(reference_dataset, reference_name, target_datasets, target_names, workdir)
#p_trend_taylor(reference_dataset, reference_name, target_datasets, target_names, workdir)
#p_trend_taylor2_era(reference_dataset, reference_name, target_datasets, target_names, workdir)
#p_trend_5obs(reference_dataset, reference_name, target_datasets, target_names, workdir)


#per_metrics(reference_dataset, reference_name, target_datasets, target_names)

#pr_extreme_detection2(reference_dataset, reference_name, target_datasets, target_names, names, workdir)

#pr_hist(reference_dataset, reference_name, target_datasets, target_names, names, workdir)
#pr_hist_sea(reference_dataset, reference_name, target_datasets, target_names, names, workdir)

#pr_max_cek(reference_dataset, reference_name, target_datasets, target_names, names, workdir)
#pr_max_map4(reference_dataset, reference_name, target_datasets, target_names, names, workdir)
#pr_min_cek(reference_dataset, reference_name, target_datasets, target_names, names, workdir)

#random_weigths(reference_dataset, reference_name, target_datasets, target_names, workdir)


#rmse_5obs(reference_dataset, reference_name, target_datasets, target_names, workdir)

#spatial_taylor(reference_dataset, reference_name, target_datasets, target_names, workdir)

#ss_annual_cycle(reference_dataset, reference_name, target_datasets, target_names, workdir)
#sst_anomaly(reference_dataset, workdir)

#station2(reference_dataset, reference_name, target_datasets, target_names, workdir)
#station_5obs(reference_dataset, reference_name, target_datasets, target_names, workdir)
#station_5obs2(reference_dataset, reference_name, target_datasets, target_names, workdir)

#taylor_map_5obs(reference_dataset, reference_name, target_datasets, target_names, workdir)
#taylor_map_annual_cycle(reference_dataset, reference_name, target_datasets, target_names, workdir)
#taylor_map_annual_cycle2(reference_dataset, reference_name, target_datasets, target_names, workdir)

#taylor_skill_map(reference_dataset, reference_name, target_datasets, target_names, workdir)
if config['Taylor_diagram_spatial']==1:
    Taylor_diagram_spatial(workdir, seasons, reference_dataset, reference_name, target_datasets, target_names)

#tele_nino_iod_pr(workdir)
#ts_nino_iod(reference_dataset, workdir)
#ts_nino_iod_season(reference_dataset, workdir)

#temporal_corr2(reference_dataset, reference_name, target_datasets, target_names, workdir)
#ts(reference_dataset, reference_name, target_datasets, target_names, workdir)
#ts_c(reference_dataset, reference_name, target_datasets, target_names, workdir)
#ts_black(reference_dataset, reference_name, target_datasets, target_names, workdir)


if config['annual_ts']== 1:
    ts_black_5obs(reference_dataset, reference_name, target_datasets, target_names, workdir)

#ts_taylor(reference_dataset, reference_name, target_datasets, target_names, workdir)

#avelet_14mean(reference_dataset, reference_name, target_datasets, target_names, workdir)
#wavelet_14sum_pycwt(reference_dataset, reference_name, target_datasets, target_names, workdir)
#wavelet_3p(reference_dataset, reference_name, target_datasets, target_names, workdir)
if config['wavelet']== 1:
    #wavelet_14mean(reference_dataset, reference_name, target_datasets, target_names, workdir)
    wavelet_5obs(reference_dataset, reference_name, target_datasets, target_names, workdir)
    #wavelet_cluster(reference_dataset, reference_name, target_datasets, target_names, workdir)
#wavelet_iod(reference_dataset, reference_name, target_datasets, target_names, workdir)
#wavelet_nino(reference_dataset, reference_name, target_datasets, target_names, workdir)
#wind_sum2(reference_dataset, reference_name, target_datasets, target_names, workdir)

#xskillscore(reference_dataset, reference_name, target_datasets, target_names, workdir)
#xskillscore_season(reference_dataset, reference_name, target_datasets, target_names, workdir)

#zonal_mean_rainfall(reference_dataset, reference_name, target_datasets, target_names, workdir)
#zonal_mean_rainfall_15(reference_dataset, reference_name, target_datasets, target_names, workdir)
#zonal_mean_rainfall_noWE(reference_dataset, reference_name, target_datasets, target_names, workdir)

exit()

# #metrik_cpi 
# cpi=metrik_cpi(reference_dataset, reference_name, target_datasets, target_names, workdir) 
# w=np.ma.array(cpi)
# cpi=w.tolist()
# print ('cpi2=',cpi)

# #metrik_zonal_mean
# if config['var_pr']:
   # mbz=bobot_zonal_mean_rainfall(reference_dataset, reference_name, target_datasets, target_names, workdir)
   # #bobot_enso()
   # print('faktor bobot zonal mean=',mbz)
   # w=np.ma.array(mbz)
   # ##print(w)
   # mbz=w.tolist()
   # mbz.append(0)
   # print('w=',mbz)
   # #mbz=np.ma.array(mbz)
 

if config['use_subregions']:
    # sort the subregion by region names and make a list
    subregions= sorted(config['subregions'].items(),key=operator.itemgetter(0))

    # number of subregions
    nsubregion = len(subregions)

    print('Calculating spatial averages and standard deviations of {} subregions ...'
          .format(nsubregion))

    reference_subregion_mean, reference_subregion_std, subregion_array = (
        utils.calc_subregion_area_mean_and_std([reference_dataset], subregions))
    target_subregion_mean, target_subregion_std, subregion_array = (
        utils.calc_subregion_area_mean_and_std(target_datasets, subregions))


# print('Writing a netcdf file ... ',workdir+config['output_netcdf_filename'])
# dsp.write_netcdf_multiple_datasets_with_subregions(
                                # reference_dataset, reference_name, 
                                # target_datasets, target_names,
                                # path=workdir+config['output_netcdf_filename'])
    
""" Step 7: Calculate metrics and draw plots """
nmetrics = config['number_of_metrics_and_plots']

if config['use_subregions']:
    Map_plot_subregion(subregions, reference_dataset, workdir)

if nmetrics > 0:
    print('Calculating metrics and generating plots ...')
    for imetric in np.arange(nmetrics)+1:
        metrics_name = config['metrics'+'%1d' %imetric]
        plot_info = config['plots'+'%1d' %imetric]
        file_name = workdir+plot_info['file_name']
        #file_name2 = workdir+plot_info['file_name2']
        
        print('metrics_name=', metrics_name)

        print('metrics {0}/{1}: {2}'.format(imetric, nmetrics, metrics_name))
        
        if metrics_name == 'Map_plot_bias_of_multiyear_climatology':
            row, column = plot_info['subplots_array']
            if 'map_projection' in plot_info.keys():
                Map_plot_bias_of_multiyear_climatology(
                    reference_dataset, reference_name, target_datasets, target_names,
                    file_name, row, column,
                    map_projection=plot_info['map_projection'])
            else:
                Map_plot_bias_of_multiyear_climatology(
                    reference_dataset, reference_name, target_datasets, target_names,
                    file_name, row, column)
        #Tambahan
        elif metrics_name == 'Map_plot_wind_bias':
            row, column = plot_info['subplots_array']
            if 'map_projection' in plot_info.keys():
                Map_plot_wind_bias(
                    reference_dataset, reference_name, target_datasets, target_names,
                    file_name, row, column,
                    map_projection=plot_info['map_projection'])
            else:
                Map_plot_wind_bias(
                    reference_dataset, reference_name, target_datasets, target_names,
                    file_name, row, column)
                    
        # elif metrics_name == 'Taylor_diagram':
            # if config['spatial or temporal?']= spatial:
                # print('Taylor_diagram_spatial')
                # Taylor_diagram_spatial_pattern_of_multiyear_climatology(
                # reference_dataset, reference_name, target_datasets, target_names,
                # file_name)
            # else:
                # print('Taylor_diagram_temporal')
                # Taylor_diagram_temporal(
                # reference_dataset, reference_name, target_datasets, target_names,
                # file_name)
            
        #elif metrics_name == 'Taylor_diagram_spatial_pattern_of_multiyear_climatology'\           
        elif metrics_name == 'Taylor_diagram_spatial'\
            and config['Weighted based spatial or temporal?']==1:
            Taylor_diagram_spatial(workdir, seasons,
                reference_dataset, reference_name, target_datasets, target_names,
                )
            print('Taylor_diagram_spatial')
        
        elif metrics_name == 'Taylor_diagram_temporal'\
            and config['Weighted based spatial or temporal?']==2:
            Taylor_diagram_temporal(
                reference_dataset, reference_name, target_datasets, target_names,
                file_name, workdir)
            print('Taylor_diagram_temporal')
        
        #Tambahan
        elif metrics_name == 'metrics_season':
            metrics_season(reference_dataset, reference_name, target_datasets, target_names,
                file_name)
        #Tambahan
        elif metrics_name == 'temporal_corr2':
            #print('xxxx')
            temporal_corr2(reference_dataset, reference_name, target_datasets, target_names,
                file_name, workdir)
                
        elif config['use_subregions']:
            #print('xxxx')
            if (metrics_name == 'Timeseries_plot_subregion_interannual_variability'
                and average_each_year):
                row, column = plot_info['subplots_array']
                Time_series_subregion(
                    reference_subregion_mean, reference_name, target_subregion_mean,
                    target_names, False, file_name, row, column,
                    x_tick=['Y'+str(i+1)
                            for i in np.arange(target_subregion_mean.shape[1])])

            if (metrics_name == 'Timeseries_plot_subregion_annual_cycle'
                and not average_each_year and month_start==1 and month_end==12):
                row, column = plot_info['subplots_array']
                #print('tes1')
                Time_series_subregion(
                    reference_subregion_mean, reference_name,
                    target_subregion_mean, target_names, True,
                    file_name, row, column,
                    x_tick=['J','F','M','A','M','J','J','A','S','O','N','D'])
            
            if (metrics_name == 'Timeseries_plot_subregion_annual_cycle_NL'
                and not average_each_year and month_start==1 and month_end==12):
                row, column = plot_info['subplots_array']
                #print('tes1')   
                Time_series_subregion2(
                    reference_subregion_mean, reference_name,
                    target_subregion_mean, target_names, True,
                    file_name, row, column,
                    x_tick=['J','F','M','A','M','J','J','A','S','O','N','D'])
            
            #tes jika month start tidak disyaratkan       
            if (metrics_name == 'Timeseries_plot_subregion_annual_cycle2'
                and not average_each_year): #and month_start==1 and month_end==12):
                row, column = plot_info['subplots_array']
                #print('tes1')
                Time_series_subregion2(
                    reference_subregion_mean, reference_name,
                    target_subregion_mean, target_names, True,
                    file_name, row, column,
                    x_tick=['J','F','M','A','M','J','J','A','S','O','N','D'])
            
            #-----tambahan ini kurang penting
            if (metrics_name == 'Timeseries_plot_subregion_annual_cycle_DJF'
                and not average_each_year):
                row, column = plot_info['subplots_array']
                #print('tes1')
                Time_series_subregion(
                    reference_subregion_mean, reference_name,
                    target_subregion_mean, target_names, True,
                    file_name, row, column,
                    #x_tick=['N','D','J','F','M','A']) #if not config[DJF] else ['D','J','F'])
                    x_tick =['D','J','F'])
                    
            if (metrics_name == 'Timeseries_plot_subregion_annual_cycle_JJA'
                and not average_each_year):
                row, column = plot_info['subplots_array']
                #print('tes1')
                Time_series_subregion(
                    reference_subregion_mean, reference_name,
                    target_subregion_mean, target_names, True,
                    file_name, row, column,
                    x_tick =['J','J','A'])
            #--------------------------------------------------
            
            #Portrait       
            if (metrics_name == 'Portrait_diagram_subregion_interannual_variability'
                and average_each_year):
                Portrait_diagram_subregion(reference_subregion_mean, reference_name,
                                           target_subregion_mean, target_names,
                                           False, file_name)

            if (metrics_name == 'Portrait_diagram_subregion_annual_cycle'
                and not average_each_year and month_start==1 and month_end==12):
                Portrait_diagram_subregion(reference_subregion_mean, reference_name,
                                           target_subregion_mean, target_names,
                                           True, file_name)
            if (metrics_name == 'Portrait_diagram_subregion_annual_cycle2'
                and not average_each_year):
                Portrait_diagram_subregion(reference_subregion_mean, reference_name,
                                           target_subregion_mean, target_names,
                                           True, file_name)
        else:
            print('please check the currently supported metrics')


            

