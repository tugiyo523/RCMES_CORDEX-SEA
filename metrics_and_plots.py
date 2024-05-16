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

#Apache OCW lib immports
import ocw.dataset as ds
import ocw.data_source.local as local
import ocw.plotter as plotter
import ocw.utils as utils
from ocw.evaluation import Evaluation
import ocw.metrics as metrics

# Python libraries
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap 
from matplotlib import rcParams
from matplotlib.patches import Polygon
import string

import ocw.plotter_asli as plotter2
import xarray as xr
import ocw.dataset_processor as dsp

##season
import yaml
import sys
config_file = str(sys.argv[1])
config = yaml.safe_load(open(config_file))


reg=config['region']
#tipe= config['Metric_type']
#ref_name1 = config['ref_name']

def taylor_skill_map(obs_dataset, obs_name, model_datasets, model_names, workdir):
    
    print('Taylor_map_annual_cycle...')
    #ini untuk 2 obs, untuk 5obs dan MOE ini perlu direvisi
    
    
   
    #yg spatial => R di diagram Taylor
    #temporal bulanan, musiman-tahunan perlu?
    #import xarray as xr
    ds = xr.DataArray(obs_dataset.values,
        coords={'time': obs_dataset.times,
        'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
        dims=["time", "lat", "lon"])  
    
    #annual_cycle
    #ds = ds.groupby('time.month').mean()
    #ds = ds.groupby('time.season').mean()
    print('temporal subsetting =>', ds.values.shape)
    
    #Atur ini
    plot='SEA'
    model=14    # maks 10 no MME
    set=7     # set 5 for 10 model 
    tes=0
    
    #geser plot for SEA
    if plot=='SEA':
        lat_min1 = 0.22*3
        lat_max1 = 0.22*6
        lon_min1 = 0.22*4
        lon_max1 = 0.22*3
    
    if plot=='sum':
        lat_min1 = 0.22*1
        lat_max1 = 0.22*3
        lon_min1 = 0.22*3
        lon_max1 = 0.22*1
    
    
    lat_min = ds.lat.min()
    lat_max = ds.lat.max()
    lon_min = ds.lon.min()
    lon_max = ds.lon.max()
   
    x,y = np.meshgrid(ds.lon, ds.lat)
    
    fig, ax = plt.subplots(nrows=2, ncols=7,figsize=(3,4))
    
    from scipy.stats import pearsonr
    import pandas as pd
   
    vmax=1
    vmin=0
    #levels = np.linspace(vmin,vmax,6)
    levels=np.arange(11)/10
    mean=[]
    for i in range(0,model): #0-9 set 10 for 10 model 
        print (model_names[i])
        
        
        dsi = xr.DataArray(model_datasets[i].values,
                coords={'time': obs_dataset.times,
                'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
                dims=["time", "lat", "lon"])       
                
        #dsi = dsi.groupby('time.month').mean()
        #dsi = dsi.groupby('time.season').mean()
        if i<set:
            n=i
            #pergerseran map yang min di + yang max di -
            #untuk koreksi fig map, SEA hanya lat_max-3 => 7
            #                                 lon_min+3 => 4
            m = Basemap(ax=ax[0,n], projection ='cyl', 
                llcrnrlat = lat_min+lat_min1, urcrnrlat = lat_max-lat_max1,
                llcrnrlon = lon_min+lon_min1, urcrnrlon = lon_max-lon_max1, 
                resolution = 'l', fix_aspect=False)
            m.drawcoastlines(linewidth=1)
            m.drawcountries(linewidth=1)
            m.drawstates(linewidth=0.5, color='w')

            map1=ma.zeros((len(ds.lat),len(ds.lon)))
            for ii in np.arange(len(ds.lat)-tes):
                #print(ii)
                for jj in np.arange(len(ds.lon)-tes):
                    x0=dsi[:,ii,jj]
                    y0=ds[:,ii,jj]
                    bad = ~np.logical_or(np.isnan(x0), np.isnan(y0))
                    x1=np.compress(bad, x0) 
                    y1=np.compress(bad, y0)
                    #print(x,y)
                    #print(x.shape,y.shape)
                    if x1.shape==(0,) or y1.shape==(0,):
                        ss='nan'
                    else:
                        c=pearsonr(x1, y1)[0]   
                        sdi=ma.std(x1, ddof=1)
                        sdo=ma.std(y1, ddof=1)
                        s=sdi/sdo
                        r0=1
                        #tt= (4*(1+c))/(((s+1/s)**2)*((1+r0)))
                        ss= (4*(1+c)**4)/(((s+1/s)**2)*((1+r0)**4))
                    map1[ii,jj]=ss    
            #mean
            me=np.nanmean(map1)
            #print(me)        
            #map1=metrics.normalize_2d(map1)
            max = ax[0,n].contourf(x,y,map1,levels=levels, vmin=0,vmax=1)#, extend='both')
            ax[0,n].set_title(model_names[i]+'('+'%.2f'%me+')', pad=3,fontsize=7)
            #khusus SEA
            if plot=='SEA':ax[0,0].set_yticks([-10,0,10,20])
            if plot=='sum':ax[0,0].set_yticks([-5,0,5])
           
            
        else:
            n=i-set
            m = Basemap(ax=ax[1,n], projection ='cyl', 
                llcrnrlat = lat_min+lat_min1, urcrnrlat = lat_max-lat_max1,
                llcrnrlon = lon_min+lon_min1, urcrnrlon = lon_max-lon_max1, 
                resolution = 'l', fix_aspect=False)
            m.drawcoastlines(linewidth=1)
            m.drawcountries(linewidth=1)
            m.drawstates(linewidth=0.5, color='w')
        
           
            map1=ma.zeros((len(ds.lat),len(ds.lon)))

            for ii in np.arange(len(ds.lat)-tes):
                #print(ii)
                for jj in np.arange(len(ds.lon)-tes):
                    x0=dsi[:,ii,jj]
                    y0=ds[:,ii,jj]
                    bad = ~np.logical_or(np.isnan(x0), np.isnan(y0))
                    x1=np.compress(bad, x0) 
                    y1=np.compress(bad, y0)
                    #print(x,y)
                    #print(x.shape,y.shape)
                    if x1.shape==(0,) or y1.shape==(0,):
                        ss='nan'
                    else:
                        c=pearsonr(x1, y1)[0]   
                        sdi=ma.std(x1, ddof=1)
                        sdo=ma.std(y1, ddof=1)
                        s=sdi/sdo
                        r0=1
                        ss= (4*(1+c)**4)/(((s+1/s)**2)*((1+r0)**4))
                       
                    map1[ii,jj]=ss
                     
            me=np.nanmean(map1)
            max = ax[1,n].contourf(x,y,map1,levels=levels, vmin=0,vmax=1)#, extend='both')
            ax[1,n].set_title(model_names[i]+'('+'%.2f'%me+')', pad=3,fontsize=7)
            #khusus SEA ini
            if plot=='SEA':
                ax[1,0].set_yticks([-10,0,10,20])
                ax[1,n].set_xticks([100,120,140])
                ax[1,n].xaxis.set_tick_params(labelsize=6)
                ax[0,0].yaxis.set_tick_params(labelsize=6)
                ax[1,0].yaxis.set_tick_params(labelsize=6)
            if plot=='sum':
                ax[1,0].set_yticks([-5,0,5])
                #ax[1,n] jika semua bawah di kasih ini
                ax[1,n].set_xticks([92,96,100,104]) 
                ax[1,n].xaxis.set_tick_params(labelsize=6)
                ax[0,0].yaxis.set_tick_params(labelsize=6)
                ax[1,0].yaxis.set_tick_params(labelsize=6)
        mean.append(round(me,2))
        
        print('cek_max=', np.nanmax(map1)) 
        print('cek_min=', np.nanmin(map1))     
        print('mean_kopi ke excel',mean)
        
        #if i==1:
            #df = pd.DataFrame(map1)
            #df.to_excel(workdir+'tes2.xlsx')
            #exit()
        
    #df = pd.DataFrame([model_names, mean])
    # 
    #df.T.to_excel(workdir+reg+'mean_s.xlsx', index=False, header=False) 
    
    plt.subplots_adjust(hspace=.12,wspace=.05)
    #cax = fig.add_axes([0.91, 0.5, 0.015, 0.4]) #pas untuk extend both
    cax = fig.add_axes([0.91, 0.53, 0.015, 0.35]) #pas untuk non
    cbar=plt.colorbar(max, cax = cax) 
    cbar.ax.tick_params(labelsize=7)
        
    #file_name='Corr_temporal_season2_SEA'
    file_name='taylor_map_monthly_'+reg
    fig.savefig(workdir+file_name,dpi=300) #,bbox_inches='tight')
    #plt.show()

def taylor_map_annual_cycle(obs_dataset, obs_name, model_datasets, model_names, workdir):
    
    print('Taylor_map_annual_cycle...')
    # pakai similarity_metric
    
    #ini untuk 2 obs, untuk 5obs dan MOE ini perlu direvisi
    
    #Atur ini
    plot='SEA'
   
    #yg spatial => R di diagram Taylor
    #temporal bulanan, musiman-tahunan perlu?
    #import xarray as xr
    ds = xr.DataArray(obs_dataset.values,
        coords={'time': obs_dataset.times,
        'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
        dims=["time", "lat", "lon"])  
    
    #annual_cycle
    ds = ds.groupby('time.month').mean()
    #ds = ds.groupby('time.season').mean()
    print('temporal subsetting =>', ds.values.shape)
    
    #Atur ini
    #plot='sum'
    model=14  # maks 10 no MME
    set=7     # set 5 for 10 model 
    tes=0
    
    #geser plot for SEA
    if plot=='SEA':
        lat_min1 = 0.22*3
        lat_max1 = 0.22*6
        lon_min1 = 0.22*4
        lon_max1 = 0.22*3
    
    if plot=='sum':
        lat_min1 = 0.22*1
        lat_max1 = 0.22*3
        lon_min1 = 0.22*3
        lon_max1 = 0.22*1
    
    
    lat_min = ds.lat.min()
    lat_max = ds.lat.max()
    lon_min = ds.lon.min()
    lon_max = ds.lon.max()
   
    x,y = np.meshgrid(ds.lon, ds.lat)
    
    fig, ax = plt.subplots(nrows=2, ncols=7,figsize=(8,6))
    
    from scipy.stats import pearsonr
    import pandas as pd
   
    
    #model_datasets=np.delete(model_datasets,[-1])
    #model_names=np.delete(model_names,[-1])
    
    vmax=1
    vmin=0
    #levels = np.linspace(vmin,vmax,6)
    levels=np.arange(11)/10
    mean=[]
    for i in range(0,model): #0-9 set 10 for 10 model 
        print (len(model_names),i, model_names[i])
        
        
        dsi = xr.DataArray(model_datasets[i].values,
                coords={'time': obs_dataset.times,
                'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
                dims=["time", "lat", "lon"])       
                
        dsi = dsi.groupby('time.month').mean()
        #dsi = dsi.groupby('time.season').mean()
        if i<set:
            n=i
            #pergerseran map yang min di + yang max di -
            #untuk koreksi fig map, SEA hanya lat_max-3 => 7
            #                                 lon_min+3 => 4
            m = Basemap(ax=ax[0,n], projection ='cyl', 
                llcrnrlat = lat_min+lat_min1, urcrnrlat = lat_max-lat_max1,
                llcrnrlon = lon_min+lon_min1, urcrnrlon = lon_max-lon_max1, 
                resolution = 'l', fix_aspect=False)
            m.drawcoastlines(linewidth=1)
            m.drawcountries(linewidth=1)
            m.drawstates(linewidth=0.5, color='w')

            map1=ma.zeros((len(ds.lat),len(ds.lon)))
            for ii in np.arange(len(ds.lat)-tes):
                print(ii)
                for jj in np.arange(len(ds.lon)-tes):
                    x0=dsi[:,ii,jj]
                    y0=ds[:,ii,jj]
                    bad = ~np.logical_or(np.isnan(x0), np.isnan(y0))
                    x1=np.compress(bad, x0) 
                    y1=np.compress(bad, y0)
                    #print(x,y)
                    #print(x.shape,y.shape)
                    if x1.shape==(0,) or y1.shape==(0,):
                        ss='nan'
                    else:
                        c=pearsonr(x1, y1)[0]   
                        sdi=ma.std(x1, ddof=1)
                        sdo=ma.std(y1, ddof=1)
                        s=sdi/sdo
                        r0=1
                        ss= (4*(1+c)**4)/(((s+1/s)**2)*((1+r0)**4))
                    map1[ii,jj]=ss  
            #similarity_metric
            if i==0: map0=map1 
            if i>0:
                x0=map0.flatten()
                y0=map1.flatten()
                bad = ~np.logical_or(np.isnan(x0), np.isnan(y0))
                x1=np.compress(bad, x0) 
                y1=np.compress(bad, y0)
                #print(x,y)
                #print(x.shape,y.shape)
                
                c=pearsonr(x1, y1)[0]   
                sdi=ma.std(x1, ddof=1)
                sdo=ma.std(y1, ddof=1)
                s=sdi/sdo
                r0=1
                ss2= (4*(1+c)**4)/(((s+1/s)**2)*((1+r0)**4))
         
            #mean
            #me=np.nanmean(map1)
            #print(me)        
            #map1=metrics.normalize_2d(map1)
            max = ax[0,n].contourf(x,y,map1,levels=levels, vmin=0,vmax=1)#, extend='both')
            if i==0:
                ax[0,n].set_title(model_names[i],fontsize=9) 
            else:
                ax[0,n].set_title(model_names[i]+' ('+'%.2f'%ss2+')',fontsize=9)
            
            #khusus SEA
            if plot=='SEA':
                ax[0,0].set_yticks([-10,0,10,20])
                ax[0,0].set_yticklabels(['10$^\circ$S','EQ','10$^\circ$N','20$^\circ$N'])
            if plot=='sum':ax[0,0].set_yticks([-6,-3,0,3,6])
           
            
        else:
            n=i-set
            m = Basemap(ax=ax[1,n], projection ='cyl', 
                llcrnrlat = lat_min+lat_min1, urcrnrlat = lat_max-lat_max1,
                llcrnrlon = lon_min+lon_min1, urcrnrlon = lon_max-lon_max1, 
                resolution = 'l', fix_aspect=False)
            m.drawcoastlines(linewidth=1)
            m.drawcountries(linewidth=1)
            m.drawstates(linewidth=0.5, color='w')
        
           
            map1=ma.zeros((len(ds.lat),len(ds.lon)))

            for ii in np.arange(len(ds.lat)-tes):
                #print(ii)
                for jj in np.arange(len(ds.lon)-tes):
                    x0=dsi[:,ii,jj]
                    y0=ds[:,ii,jj]
                    bad = ~np.logical_or(np.isnan(x0), np.isnan(y0))
                    x1=np.compress(bad, x0) 
                    y1=np.compress(bad, y0)
                    #print(x,y)
                    #print(x.shape,y.shape)
                    if x1.shape==(0,) or y1.shape==(0,):
                        ss='nan'
                    else:
                        c=pearsonr(x1, y1)[0]   
                        sdi=ma.std(x1, ddof=1)
                        sdo=ma.std(y1, ddof=1)
                        s=sdi/sdo
                        r0=1
                        ss= (4*(1+c)**4)/(((s+1/s)**2)*((1+r0)**4))
                       
                    map1[ii,jj]=ss
            
            #similarity metric
            #if i==0: map0=map1 
            if i>0:
                x0=map0.flatten()
                y0=map1.flatten()
                bad = ~np.logical_or(np.isnan(x0), np.isnan(y0))
                x1=np.compress(bad, x0) 
                y1=np.compress(bad, y0)
                #print(x,y)
                #print(x.shape,y.shape)
                
                c=pearsonr(x1, y1)[0]   
                sdi=ma.std(x1, ddof=1)
                sdo=ma.std(y1, ddof=1)
                s=sdi/sdo
                r0=1
                ss2= (4*(1+c)**4)/(((s+1/s)**2)*((1+r0)**4))
         
            #me=np.nanmean(map1)
            max = ax[1,n].contourf(x,y,map1,levels=levels, vmin=0,vmax=1)#, extend='both')
            ax[1,n].set_title(model_names[i]+' ('+'%.2f'%ss2+')',fontsize=9)
            #khusus SEA ini
            if plot=='SEA':
                ax[1,0].set_yticks([-10,0,10,20])
                ax[1,0].set_yticklabels(['10$^\circ$S','EQ','10$^\circ$N','20$^\circ$N'])
                ax[1,n].set_xticks([100,120,140])
                ax[1,n].set_xticklabels(['100$^\circ$E','120$^\circ$E','140$^\circ$E'])
                ax[1,n].xaxis.set_tick_params(labelsize=7)
                ax[0,0].yaxis.set_tick_params(labelsize=7)
                ax[1,0].yaxis.set_tick_params(labelsize=7)
            if plot=='sum':
                ax[1,0].set_yticks([-6,-3,0,3,6])
                #ax[1,n] jika semua bawah di kasih ini
                ax[1,0].set_xticks([94,98,102,106]) 
                ax[1,0].xaxis.set_tick_params(labelsize=6)
                ax[0,0].yaxis.set_tick_params(labelsize=6)
                ax[1,0].yaxis.set_tick_params(labelsize=6)
        #mean.append(round(me,2))
        
        #print('cek_max=', np.nanmax(map1)) 
        #print('cek_min=', np.nanmin(map1))     
        #print('mean_kopi ke excel',mean)
        
        #if i==1:
            #df = pd.DataFrame(map1)
            #df.to_excel(workdir+'tes2.xlsx')
            #exit()
    
    #khusus perbaikan Tian
    #ax[0,0].set_yticks([])
    #ax[0,1].set_yticks([])
    
    df = pd.DataFrame([model_names, mean])
    # 
    df.T.to_excel(workdir+reg+'mean_s.xlsx', index=False, header=False) 
    
    plt.subplots_adjust(hspace=.12,wspace=.05)
    #cax = fig.add_axes([0.91, 0.5, 0.015, 0.4]) #pas untuk extend both
    cax = fig.add_axes([0.91, 0.53, 0.015, 0.35]) #pas untuk non
    plt.colorbar(max, cax = cax) 
        
    #file_name='Corr_temporal_season2_SEA'
    file_name='Taylor map_mon_5obs_saobs_'+reg
    fig.savefig(workdir+file_name,dpi=300,bbox_inches='tight')
    plt.show()

def taylor_map_annual_cycle2(obs_dataset, obs_name, model_datasets, model_names, workdir):
    
    print('Taylor_map_annual_cycle...')
    # pakai mean
    
    plot='SEA'
   
    ds = xr.DataArray(obs_dataset.values,
        coords={'time': obs_dataset.times,
        'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
        dims=["time", "lat", "lon"])  
    
    #annual_cycle
    ds = ds.groupby('time.month').mean()
    #ds = ds.groupby('time.season').mean()
    print('temporal subsetting =>', ds.values.shape)
    
    #Atur ini
    #plot='sum'
    model=14 # maks 10 no MME
    set=7     # set 5 for 10 model 
    tes=0
    
    #geser plot for SEA
    if plot=='SEA':
        lat_min1 = 0.22*3
        lat_max1 = 0.22*6
        lon_min1 = 0.22*4
        lon_max1 = 0.22*3
    
    if plot=='sum':
        lat_min1 = 0.22*1
        lat_max1 = 0.22*3
        lon_min1 = 0.22*3
        lon_max1 = 0.22*1
    
    
    lat_min = ds.lat.min()
    lat_max = ds.lat.max()
    lon_min = ds.lon.min()
    lon_max = ds.lon.max()
   
    x,y = np.meshgrid(ds.lon, ds.lat)
    
    fig, ax = plt.subplots(nrows=2, ncols=7,figsize=(8,6))
    
    from scipy.stats import pearsonr
    import pandas as pd
   
    
    #model_datasets=np.delete(model_datasets,[-1])
    #model_names=np.delete(model_names,[-1])
    
    vmax=1
    vmin=0
    #levels = np.linspace(vmin,vmax,6)
    levels=np.arange(11)/10
    mean=[]
    for i in range(0,model): #0-9 set 10 for 10 model 
        print (len(model_names),i, model_names[i])
        
        
        dsi = xr.DataArray(model_datasets[i].values,
                coords={'time': obs_dataset.times,
                'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
                dims=["time", "lat", "lon"])       
                
        dsi = dsi.groupby('time.month').mean()
        #dsi = dsi.groupby('time.season').mean()
        if i<set:
            n=i
            #pergerseran map yang min di + yang max di -
            #untuk koreksi fig map, SEA hanya lat_max-3 => 7
            #                                 lon_min+3 => 4
            m = Basemap(ax=ax[0,n], projection ='cyl', 
                llcrnrlat = lat_min+lat_min1, urcrnrlat = lat_max-lat_max1,
                llcrnrlon = lon_min+lon_min1, urcrnrlon = lon_max-lon_max1, 
                resolution = 'l', fix_aspect=False)
            m.drawcoastlines(linewidth=1)
            m.drawcountries(linewidth=1)
            m.drawstates(linewidth=0.5, color='w')

            map1=ma.zeros((len(ds.lat),len(ds.lon)))
            for ii in np.arange(len(ds.lat)-tes):
                print(ii)
                for jj in np.arange(len(ds.lon)-tes):
                    x0=dsi[:,ii,jj]
                    y0=ds[:,ii,jj]
                    bad = ~np.logical_or(np.isnan(x0), np.isnan(y0))
                    x1=np.compress(bad, x0) 
                    y1=np.compress(bad, y0)
                    #print(x,y)
                    #print(x.shape,y.shape)
                    if x1.shape==(0,) or y1.shape==(0,):
                        ss='nan'
                    else:
                        c=pearsonr(x1, y1)[0]   
                        sdi=ma.std(x1, ddof=1)
                        sdo=ma.std(y1, ddof=1)
                        s=sdi/sdo
                        r0=1
                        ss= (4*(1+c)**4)/(((s+1/s)**2)*((1+r0)**4))
                    map1[ii,jj]=ss  
            #similarity_metric
            if i==0: map0=map1 
            if i>0:
                x0=map0.flatten()
                y0=map1.flatten()
                bad = ~np.logical_or(np.isnan(x0), np.isnan(y0))
                x1=np.compress(bad, x0) 
                y1=np.compress(bad, y0)
                #print(x,y)
                #print(x.shape,y.shape)
                
                c=pearsonr(x1, y1)[0]   
                sdi=ma.std(x1, ddof=1)
                sdo=ma.std(y1, ddof=1)
                s=sdi/sdo
                r0=1
                ss2= (4*(1+c)**4)/(((s+1/s)**2)*((1+r0)**4))
         
            #mean
            me=np.nanmean(map1)
            #print(me)        
            #map1=metrics.normalize_2d(map1)
            max = ax[0,n].contourf(x,y,map1,levels=levels, vmin=0,vmax=1)#, extend='both')
            if i==0:
                ax[0,n].set_title(model_names[i],fontsize=9) 
            else:
                ax[0,n].set_title(model_names[i]+' ('+'%.2f'%ss2+')',fontsize=9)
                ax[0,n].text(x=94, y=-13, s='m='+'%.2f'%me, fontsize=9)
            
            #khusus SEA
            if plot=='SEA':
                ax[0,0].set_yticks([-10,0,10,20])
                ax[0,0].set_yticklabels(['10$^\circ$S','EQ','10$^\circ$N','20$^\circ$N'])
            if plot=='sum':ax[0,0].set_yticks([-6,-3,0,3,6])
           
            
        else:
            n=i-set
            m = Basemap(ax=ax[1,n], projection ='cyl', 
                llcrnrlat = lat_min+lat_min1, urcrnrlat = lat_max-lat_max1,
                llcrnrlon = lon_min+lon_min1, urcrnrlon = lon_max-lon_max1, 
                resolution = 'l', fix_aspect=False)
            m.drawcoastlines(linewidth=1)
            m.drawcountries(linewidth=1)
            m.drawstates(linewidth=0.5, color='w')
        
           
            map1=ma.zeros((len(ds.lat),len(ds.lon)))

            for ii in np.arange(len(ds.lat)-tes):
                #print(ii)
                for jj in np.arange(len(ds.lon)-tes):
                    x0=dsi[:,ii,jj]
                    y0=ds[:,ii,jj]
                    bad = ~np.logical_or(np.isnan(x0), np.isnan(y0))
                    x1=np.compress(bad, x0) 
                    y1=np.compress(bad, y0)
                    #print(x,y)
                    #print(x.shape,y.shape)
                    if x1.shape==(0,) or y1.shape==(0,):
                        ss='nan'
                    else:
                        c=pearsonr(x1, y1)[0]   
                        sdi=ma.std(x1, ddof=1)
                        sdo=ma.std(y1, ddof=1)
                        s=sdi/sdo
                        r0=1
                        ss= (4*(1+c)**4)/(((s+1/s)**2)*((1+r0)**4))
                       
                    map1[ii,jj]=ss
            
            #similarity metric
            #if i==0: map0=map1 
            if i>0:
                x0=map0.flatten()
                y0=map1.flatten()
                bad = ~np.logical_or(np.isnan(x0), np.isnan(y0))
                x1=np.compress(bad, x0) 
                y1=np.compress(bad, y0)
                #print(x,y)
                #print(x.shape,y.shape)
                
                c=pearsonr(x1, y1)[0]   
                sdi=ma.std(x1, ddof=1)
                sdo=ma.std(y1, ddof=1)
                s=sdi/sdo
                r0=1
                ss2= (4*(1+c)**4)/(((s+1/s)**2)*((1+r0)**4))
         
            me=np.nanmean(map1)
            max = ax[1,n].contourf(x,y,map1,levels=levels, vmin=0,vmax=1)#, extend='both')
            ax[1,n].set_title(model_names[i]+' ('+'%.2f'%ss2+')',fontsize=9)
            ax[1,n].text(x=94, y=-13, s='m='+'%.2f'%me, fontsize=9)
            #khusus SEA ini
            if plot=='SEA':
                ax[1,0].set_yticks([-10,0,10,20])
                ax[1,0].set_yticklabels(['10$^\circ$S','EQ','10$^\circ$N','20$^\circ$N'])
                ax[1,n].set_xticks([100,120,140])
                ax[1,n].set_xticklabels(['100$^\circ$E','120$^\circ$E','140$^\circ$E'])
                ax[1,n].xaxis.set_tick_params(labelsize=7)
                ax[0,0].yaxis.set_tick_params(labelsize=7)
                ax[1,0].yaxis.set_tick_params(labelsize=7)
            if plot=='sum':
                ax[1,0].set_yticks([-6,-3,0,3,6])
                #ax[1,n] jika semua bawah di kasih ini
                ax[1,0].set_xticks([94,98,102,106]) 
                ax[1,0].xaxis.set_tick_params(labelsize=6)
                ax[0,0].yaxis.set_tick_params(labelsize=6)
                ax[1,0].yaxis.set_tick_params(labelsize=6)
       
    
    #khusus perbaikan Tian
    #ax[0,0].set_yticks([])
    #ax[0,1].set_yticks([])
    
    df = pd.DataFrame([model_names, mean])
    # 
    df.T.to_excel(workdir+reg+'mean_s.xlsx', index=False, header=False) 
    
    plt.subplots_adjust(hspace=.12,wspace=.05)
    #cax = fig.add_axes([0.91, 0.5, 0.015, 0.4]) #pas untuk extend both
    cax = fig.add_axes([0.91, 0.53, 0.015, 0.35]) #pas untuk non
    plt.colorbar(max, cax = cax) 
        
    #file_name='Corr_temporal_season2_SEA'
    file_name='Taylor map_ac_'+reg
    fig.savefig(workdir+file_name,dpi=300,bbox_inches='tight')
    plt.show()

def taylor_map_5obs(obs_dataset, obs_name, model_datasets, model_names, workdir):
    
    print('Taylor_map_annual_cycle...')
    
    
    #Atur ini
    plot='sum'
   
    #yg spatial => R di diagram Taylor
    #temporal bulanan, musiman-tahunan perlu?
    #import xarray as xr
    ds = xr.DataArray(obs_dataset.values,
        coords={'time': obs_dataset.times,
        'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
        dims=["time", "lat", "lon"])  
    
    #annual_cycle
    #ds = ds.groupby('time.month').mean()
    #ds = ds.groupby('time.season').mean()
    print('temporal subsetting =>', ds.values.shape)
    
    #Atur ini
    #plot='sum'
    model=4    
   
    tes=50
    
    #geser plot for SEA
    if plot=='SEA':
        lat_min1 = 0.22*3
        lat_max1 = 0.22*6
        lon_min1 = 0.22*4
        lon_max1 = 0.22*3
    
    if plot=='sum':
        lat_min1 = 0.22*1
        lat_max1 = 0.22*3
        lon_min1 = 0.22*3
        lon_max1 = 0.22*1
    
    
    lat_min = ds.lat.min()
    lat_max = ds.lat.max()
    lon_min = ds.lon.min()
    lon_max = ds.lon.max()
   
    x,y = np.meshgrid(ds.lon, ds.lat)
    
    fig, ax = plt.subplots(nrows=1, ncols=4,figsize=(6,4))
    
    from scipy.stats import pearsonr
    import pandas as pd
   
    
    #model_datasets=np.delete(model_datasets,[-1])
    #model_names=np.delete(model_names,[-1])
    
    vmax=1
    vmin=0
    #levels = np.linspace(vmin,vmax,6)
    levels=np.arange(11)/10
    mean=[]
    for i in range(0,model): #0-9 set 10 for 10 model 
        print (len(model_names),i, model_names[i])
        
        
        dsi = xr.DataArray(model_datasets[i].values,
                coords={'time': obs_dataset.times,
                'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
                dims=["time", "lat", "lon"])       
                
        #dsi = dsi.groupby('time.month').mean()
        #dsi = dsi.groupby('time.season').mean()
       
        n=i
        #pergerseran map yang min di + yang max di -
        #untuk koreksi fig map, SEA hanya lat_max-3 => 7
        #                                 lon_min+3 => 4
        m = Basemap(ax=ax[n], projection ='cyl', 
            llcrnrlat = lat_min+lat_min1, urcrnrlat = lat_max-lat_max1,
            llcrnrlon = lon_min+lon_min1, urcrnrlon = lon_max-lon_max1, 
            resolution = 'l', fix_aspect=False)
        m.drawcoastlines(linewidth=1)
        m.drawcountries(linewidth=1)
        m.drawstates(linewidth=0.5, color='w')

        map1=ma.zeros((len(ds.lat),len(ds.lon)))
        for ii in np.arange(len(ds.lat)-tes):
            print(ii)
            for jj in np.arange(len(ds.lon)-tes):
                x0=dsi[:,ii,jj]
                y0=ds[:,ii,jj]
                bad = ~np.logical_or(np.isnan(x0), np.isnan(y0))
                x1=np.compress(bad, x0) 
                y1=np.compress(bad, y0)
                #print(x,y)
                #print(x.shape,y.shape)
                if x1.shape==(0,) or y1.shape==(0,):
                    tt='nan'
                else:
                    c=pearsonr(x1, y1)[0]   
                    sdi=ma.std(x1, ddof=1)
                    sdo=ma.std(y1, ddof=1)
                    s=sdi/sdo
                    r0=1
                    tt= (4*(1+c)**4)/(((s+1/s)**2)*((1+r0)**4))
                map1[ii,jj]=tt    
        #mean
        me=np.nanmean(map1)
        #print(me)        
        #map1=metrics.normalize_2d(map1)
        max = ax[n].contourf(x,y,map1,levels=levels, vmin=0,vmax=1)#, extend='both')
        ax[n].set_title(model_names[i]+'('+'%.2f'%me+')',fontsize=9)
        
        #khusus SEA ini
        if plot=='SEA':
            ax[i].set_yticks([-10,0,10,20])
            ax[i].set_xticks([90,100,110,120,130,140])
            ax[i].xaxis.set_tick_params(labelsize=6)
            ax[i].yaxis.set_tick_params(labelsize=6)
            ax[i].yaxis.set_tick_params(labelsize=6)
        if plot=='sum':
            ax[0].set_yticks([-6,-3,0,3,6])
            ax[0].set_yticklabels(['6$^\circ$S','3$^\circ$S','0','3$^\circ$N','6$^\circ$N'])
            ax[i].set_xticks([95,100,105]) 
            ax[i].set_xticklabels(['95$^\circ$E','100$^\circ$E','105$^\circ$E']) 
            ax[i].xaxis.set_tick_params(labelsize=8)
            ax[0].yaxis.set_tick_params(labelsize=8)
    mean.append(round(me,2))
    
    print('cek_max=', np.nanmax(map1)) 
    print('cek_min=', np.nanmin(map1))     
    print('mean_kopi ke excel',mean)
   
    
       
    #if i==1:
        #df = pd.DataFrame(map1)
        #df.to_excel(workdir+'tes2.xlsx')
        #exit()
        
    df = pd.DataFrame([model_names, mean])
    # 
    df.T.to_excel(workdir+reg+'mean_s.xlsx', index=False, header=False) 
    
    plt.subplots_adjust(hspace=.12,wspace=.05)
    #cax = fig.add_axes([0.91, 0.5, 0.015, 0.4]) #pas untuk extend both
    cax = fig.add_axes([0.91, 0.53, 0.015, 0.35]) #pas untuk non
    plt.colorbar(max, cax = cax) 
        
    #file_name='Corr_temporal_season2_SEA'
    file_name='Taylor map_mon_5obs_saobs_'+reg
    fig.savefig(workdir+file_name,dpi=300,bbox_inches='tight')
    plt.show()
    

def ss_annual_cycle(obs_dataset, obs_name, model_datasets, model_names, workdir):
    #Peirce 2009
    print('ss_map_annual_cycle...')
    #ini untuk 2 obs, untuk 5obs dan MOE ini perlu direvisi
    
    #Atur ini
    plot='sum'
   
    #yg spatial => R di diagram Taylor
    #temporal bulanan, musiman-tahunan perlu?
    #import xarray as xr
    ds = xr.DataArray(obs_dataset.values,
        coords={'time': obs_dataset.times,
        'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
        dims=["time", "lat", "lon"])  
    
    #annual_cycle
    ds = ds.groupby('time.month').mean()
    #ds = ds.groupby('time.season').mean()
    print('temporal subsetting =>', ds.values.shape)
    
    #Atur ini
    #plot='sum'
    model=2    # maks 10 no MME
    set=5     # set 5 for 10 model 
    tes=0
    
    #geser plot for SEA
    if plot=='SEA':
        lat_min1 = 0.22*3
        lat_max1 = 0.22*6
        lon_min1 = 0.22*4
        lon_max1 = 0.22*3
    
    if plot=='sum':
        lat_min1 = 0.22*1
        lat_max1 = 0.22*3
        lon_min1 = 0.22*3
        lon_max1 = 0.22*1
    
    
    lat_min = ds.lat.min()
    lat_max = ds.lat.max()
    lon_min = ds.lon.min()
    lon_max = ds.lon.max()
   
    x,y = np.meshgrid(ds.lon, ds.lat)
    
    fig, ax = plt.subplots(nrows=2, ncols=5,figsize=(8,6))
    
    from scipy.stats import pearsonr
    import pandas as pd
   
    
    model_datasets=np.delete(model_datasets,[-1])
    model_names=np.delete(model_names,[-1])
    
    vmax=1
    vmin=0
    #levels = np.linspace(vmin,vmax,6)
    levels=np.arange(11)/10
    mean=[]
    for i in range(0,model): #0-9 set 10 for 10 model 
        print (model_names[i])
        
        
        dsi = xr.DataArray(model_datasets[i].values,
                coords={'time': obs_dataset.times,
                'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
                dims=["time", "lat", "lon"])       
                
        dsi = dsi.groupby('time.month').mean()
        #dsi = dsi.groupby('time.season').mean()
        if i<set:
            n=i
            #pergerseran map yang min di + yang max di -
            #untuk koreksi fig map, SEA hanya lat_max-3 => 7
            #                                 lon_min+3 => 4
            m = Basemap(ax=ax[0,n], projection ='cyl', 
                llcrnrlat = lat_min+lat_min1, urcrnrlat = lat_max-lat_max1,
                llcrnrlon = lon_min+lon_min1, urcrnrlon = lon_max-lon_max1, 
                resolution = 'l', fix_aspect=False)
            m.drawcoastlines(linewidth=1)
            m.drawcountries(linewidth=1)
            m.drawstates(linewidth=0.5, color='w')

            map1=ma.zeros((len(ds.lat),len(ds.lon)))
            for ii in np.arange(len(ds.lat)-tes):
                #print(ii)
                for jj in np.arange(len(ds.lon)-tes):
                    x0=dsi[:,ii,jj]
                    y0=ds[:,ii,jj]
                    bad = ~np.logical_or(np.isnan(x0), np.isnan(y0))
                    x1=np.compress(bad, x0) 
                    y1=np.compress(bad, y0)
                    #print(x,y)
                    #print(x.shape,y.shape)
                    if x1.shape==(0,) or y1.shape==(0,):
                        ss='nan'
                    else:
                        c=pearsonr(x1, y1)[0]   
                        sdi=ma.std(x1, ddof=1)
                        sdo=ma.std(y1, ddof=1)
                        s=sdi/sdo
                        r0=1
                        #ss= (4*(1+c)**4)/(((s+1/s)**2)*((1+r0)**4))
                        
                        mm=x1.mean()
                        mo=y1.mean()                  
                        ss=c**2 -(c-s)**2 -((mm-mo)/sdo)**2
                        #if ss==0: 
                        #print(ss.values)
                    map1[ii,jj]=ss
            #print('cek_Rmax=', np.nanmax(map1))
            
            #khusus ss2 range -x to 1 => 0,1 ??
            #cara 1 gagal
            m=np.zeros(map1.shape)
            index = np.isnan(map1)
            m[index]=1
            map1 = np.ma.masked_array(map1,mask=m)
            map1=map1+ abs(map1.mean())
            # map1=map1/(map1.max())
            map1[map1<0]= 0
            map1=map1/(map1.max())
            #cara 2 gagal
            # map1=np.nan_to_num(map1, copy=True, nan=0.0, posinf=None, neginf=None)
            # map1=metrics.normalize_2d(map1)
            # map1=map1+abs(np.min(map1))
            # map1=map1/map1.max()   
            # #arr[arr == 0] = 'nan'
            # map1[map1==0]= np.nan
            
            #mean
            me=np.nanmean(map1)
            #print(me)        
            #map1=metrics.normalize_2d(map1)
            max = ax[0,n].contourf(x,y,map1,levels=levels, vmin=0,vmax=1)#, extend='both')
            ax[0,n].set_title(model_names[i]+'('+'%.2f'%me+')',fontsize=9)
            #khusus SEA
            if plot=='SEA':ax[0,0].set_yticks([-10,0,10,20])
            if plot=='sum':ax[0,0].set_yticks([-6,-3,0,3,6])
           
            
        else:
            n=i-set
            m = Basemap(ax=ax[1,n], projection ='cyl', 
                llcrnrlat = lat_min+lat_min1, urcrnrlat = lat_max-lat_max1,
                llcrnrlon = lon_min+lon_min1, urcrnrlon = lon_max-lon_max1, 
                resolution = 'l', fix_aspect=False)
            m.drawcoastlines(linewidth=1)
            m.drawcountries(linewidth=1)
            m.drawstates(linewidth=0.5, color='w')
        
           
            map1=ma.zeros((len(ds.lat),len(ds.lon)))

            for ii in np.arange(len(ds.lat)-tes):
                #print(ii)
                for jj in np.arange(len(ds.lon)-tes):
                    x0=dsi[:,ii,jj]
                    y0=ds[:,ii,jj]
                    bad = ~np.logical_or(np.isnan(x0), np.isnan(y0))
                    x1=np.compress(bad, x0) 
                    y1=np.compress(bad, y0)
                    #print(x,y)
                    #print(x.shape,y.shape)
                    if x1.shape==(0,) or y1.shape==(0,):
                        ss='nan'
                    else:
                        c=pearsonr(x1, y1)[0]   
                        sdi=ma.std(x1, ddof=1)
                        sdo=ma.std(y1, ddof=1)
                        s=sdi/sdo
                        r0=1
                        ss= (4*(1+c)**4)/(((s+1/s)**2)*((1+r0)**4))
                        
                        mm=x1.mean()
                        mo=y1.mean()                  
                        ss=c**2 -(c-s)**2 -((mm-mo)/sdo)**2
                    map1[ii,jj]=ss
            
            m=np.zeros(map1.shape)
            index = np.isnan(map1)
            m[index]=1
            map1 = np.ma.masked_array(map1,mask=m)
            map1=map1+ abs(map1.mean())
            # map1=map1/(map1.max())
            map1[map1<0]= 0
            map1=map1/(map1.max())
            
            me=np.nanmean(map1)
            max = ax[1,n].contourf(x,y,map1,levels=levels, vmin=0,vmax=1)#, extend='both')
            ax[1,n].set_title(model_names[i]+'('+'%.2f'%me+')',fontsize=9)
            #khusus SEA ini
            if plot=='SEA':
                ax[1,0].set_yticks([-10,0,10,20])
                ax[1,0].set_xticks([90,100,110,120,130,140])
                ax[1,0].xaxis.set_tick_params(labelsize=6)
                ax[0,0].yaxis.set_tick_params(labelsize=6)
                ax[1,0].yaxis.set_tick_params(labelsize=6)
            if plot=='sum':
                ax[1,0].set_yticks([-6,-3,0,3,6])
                #ax[1,n] jika semua bawah di kasih ini
                ax[1,0].set_xticks([94,98,102,106]) 
                ax[1,0].xaxis.set_tick_params(labelsize=6)
                ax[0,0].yaxis.set_tick_params(labelsize=6)
                ax[1,0].yaxis.set_tick_params(labelsize=6)
        mean.append(round(me,2))
        
        print('cek_max=', np.nanmax(map1)) 
        print('cek_min=', np.nanmin(map1))     
        print('mean_kopi ke excel',mean)
        
        #if i==1:
            #df = pd.DataFrame(map1)
            #df.to_excel(workdir+'tes2.xlsx')
            #exit()
        
    df = pd.DataFrame([model_names, mean])
    # 
    df.T.to_excel(workdir+'mean_ss2_sum_L-tes.xlsx', index=False, header=False) 
    
    plt.subplots_adjust(hspace=.12,wspace=.05)
    #cax = fig.add_axes([0.91, 0.5, 0.015, 0.4]) #pas untuk extend both
    cax = fig.add_axes([0.91, 0.53, 0.015, 0.35]) #pas untuk non
    plt.colorbar(max, cax = cax) 
        
    #file_name='Corr_temporal_season2_SEA'
    file_name='ss2_'+reg
    #fig.savefig(workdir+file_name,dpi=300,bbox_inches='tight')
    plt.show()

def p_trend(obs_dataset, obs_name, model_datasets, model_names, workdir):
    #metrik pakai mean saja
    #untuk 10 model dan mme, 3mmew = 14
    
    print('p_trend...')
    import pymannkendall as mk

    #yg spatial => R di diagram Taylor
    #temporal bulanan, musiman-tahunan perlu?
    #import xarray as xr
    ds = xr.DataArray(obs_dataset.values,
        coords={'time': obs_dataset.times,
        'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
        dims=["time", "lat", "lon"])  

    print('temporal subsetting =>', ds.values.shape)
    
    #Atur ini
    plot='sum'
    
    model=2  # maks 10 no MME
    set=7     # set 5 for 10 model 
    tes=0 #
    
    #geser plot for SEA
    if plot=='SEA':
        lat_min1 = 0.22*3
        lat_max1 = 0.22*6
        lon_min1 = 0.22*4
        lon_max1 = 0.22*3
    
    if plot=='sum':
        lat_min1 = 0.22*1
        lat_max1 = 0.22*3
        lon_min1 = 0.22*3
        lon_max1 = 0.22*1
    
    
    lat_min = ds.lat.min()
    lat_max = ds.lat.max()
    lon_min = ds.lon.min()
    lon_max = ds.lon.max()
   
    x,y = np.meshgrid(ds.lon, ds.lat)
    
    fig, ax = plt.subplots(nrows=2, ncols=7,figsize=(8,6))
    
    #from scipy.stats import pearsonr
    import pandas as pd
   
    #model_datasets=np.delete(model_datasets,[-1])
    #model_names=np.delete(model_names,[-1])
    
    vmax=3
    vmin=-3
    #levels = np.linspace(vmin,vmax,6)
    levels=[-3,-2,-1,0,1,2,3]
    #levels=np.arange(6)
    norm = plt.Normalize(vmin, vmax)
    mean=[]
    x1=[]
    y1=[]
    x2=[]
    y2=[]
    for i in range(0,model): #0-9 set 10 for 10 model 
        
        print (model_names[i])
        #print(x1)
       
        dsi = xr.DataArray(model_datasets[i].values,
                coords={'time': obs_dataset.times,
                'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
                dims=["time", "lat", "lon"])       
                
        #dsi = dsi.groupby('time.month').mean()
        #dsi = dsi.groupby('time.year').sum()
        print('temporal subsetting =>', dsi.values.shape)
        if i<set:
            n=i
            #pergerseran map yang min di + yang max di -
            #untuk koreksi fig map, SEA hanya lat_max-3 => 7
            #                                 lon_min+3 => 4
            m = Basemap(ax=ax[0,n], projection ='cyl', 
                llcrnrlat = lat_min+lat_min1, urcrnrlat = lat_max-lat_max1,
                llcrnrlon = lon_min+lon_min1, urcrnrlon = lon_max-lon_max1, 
                resolution = 'l', fix_aspect=False)
            m.drawcoastlines(linewidth=1)
            m.drawcountries(linewidth=1)
            m.drawstates(linewidth=0.5, color='w')

            map1=ma.zeros((len(ds.lat),len(ds.lon)))
            for ii in np.arange(len(ds.lat)-tes):
                #print(ii)
                for jj in np.arange(len(ds.lon)-tes):
                    x0=dsi[:,ii,jj]
                   
                    #ds year > tidak ada nan tapi 0 semua, 
                    #monthly ada nan maka 
                    if np.isnan(x0).any() or x0.sum()==0:
                        s='nan'
                    else:
                        k=mk.original_test(x0)
                        #s=k.z
                                       
                        if k.p<0.05:
                            #print(ii,jj, ds.lat[ii].data,ds.lon[jj].data)
                            #x1.append(ds.lon[jj].data)
                            #y1.append(ds.lat[ii].data)
                            # print(k.p)
                            #print(k.z)
                            #print(k.slope)
                            #print(k.trend)
                            s=k.z
                     
                    map1[ii,jj]=s
            
            me=np.nanmean(map1)
            #max = ax[0,n].contourf(x,y,map1) #,levels=levels, vmin=0,vmax=1)#, extend='both')
            #max = ax[0,n].contourf(x,y,map1, vmin=0,vmax=3)#, extend='both')
            max = ax[0,n].contourf(x,y,map1,levels=levels, norm=norm) #, extend='both')
            ax[0,n].set_title(model_names[i]+'('+'%.2f'%me+')',fontsize=8)
            ##khusus SEA
            if plot=='SEA':ax[0,0].set_yticks([-10,0,10,20])
            if plot=='sum':
                ax[0,i].set_yticks([-5,0,5])
                ax[0,i].set_xticks([95,105])
            
            #khusus sumatera, SEA terlalu besar tidak jelas
            #for i in np.arange(len(x1)):
            #    ax[0,n].annotate('+', (x1[i],y1[i]),fontsize=8)
                #ax[0,n].text( x=x1[i], y=y1[i],s='+')
            
        else:
            n=i-set
            m = Basemap(ax=ax[1,n], projection ='cyl', 
                llcrnrlat = lat_min+lat_min1, urcrnrlat = lat_max-lat_max1,
                llcrnrlon = lon_min+lon_min1, urcrnrlon = lon_max-lon_max1, 
                resolution = 'l', fix_aspect=False)
            m.drawcoastlines(linewidth=1)
            m.drawcountries(linewidth=1)
            m.drawstates(linewidth=0.5, color='w')
        
            map1=ma.zeros((len(ds.lat),len(ds.lon)))

            for ii in np.arange(len(ds.lat)-tes):
                #print(ii)
                for jj in np.arange(len(ds.lon)-tes):
                    x0=dsi[:,ii,jj]
                    
                    #ds year > tidak ada nan tapi 0 semua, monthly ada
                    if np.isnan(x0).any() or x0.sum()==0:
                        s='nan'
                        #map1[ii,jj]=s
                        #print(s)
                    else:
                        k=mk.original_test(x0)
                        #s=k.z
                    
                        if k.p<0.05:
                            #print(ii,jj, ds.lat[ii].data,ds.lon[jj].data)
                            #x2.append(ds.lon[jj].data)
                            #y2.append(ds.lat[ii].data)
                            # print(k.p)
                            #print(k.z)
                            #print(k.slope)
                            #print(k.trend)
                            s=k.z
                           
                    #print(s)
                    map1[ii,jj]=s
                                
            me=np.nanmean(map1)
            max = ax[1,n].contourf(x,y,map1,levels=levels, norm=norm) 
            
            ax[1,n].set_title(model_names[i]+'('+'%.2f'%me+')',fontsize=6)
            #khusus SEA ini
            if plot=='SEA':
                ax[1,0].set_yticks([-10,0,10,20])
                ax[1,n].set_xticks([90,100,110,120,130,140])
                ax[1,n].xaxis.set_tick_params(labelsize=6)
                ax[0,0].yaxis.set_tick_params(labelsize=6)
                ax[1,0].yaxis.set_tick_params(labelsize=6)
            if plot=='sum':
                ax[1,0].set_yticks([-5,0,5])
                #ax[1,n] jika semua bawah di kasih ini
                ax[1,n].set_xticks([92,96,100,104]) 
                ax[1,n].xaxis.set_tick_params(labelsize=6)
                ax[0,0].yaxis.set_tick_params(labelsize=6)
                ax[1,0].yaxis.set_tick_params(labelsize=6)
                
            #for i in np.arange(len(x2)):
            #    ax[1,n].annotate('+', (x2[i],y2[i]),fontsize=8)
        mean.append(round(me,2))
        print('mean_kopi ke excel',mean)
        
        print('cek_max=', np.nanmax(map1)) 
        print('cek_min=', np.nanmin(map1))     
        
        
        #if i==1:
            #df = pd.DataFrame(map1)
            #df.to_excel(workdir+'tes2.xlsx')
            #exit()
        x1=[]
        y1=[]
        x2=[]
        y2=[]
    df = pd.DataFrame([model_names, mean])
    # 
    #df.T.to_excel(workdir+'mean_ss2_sum_L-tes.xlsx', index=False, header=False) 
    
    plt.subplots_adjust(hspace=.12,wspace=.05)
    #cax = fig.add_axes([0.91, 0.5, 0.015, 0.4]) #pas untuk extend both
    cax = fig.add_axes([0.91, 0.53, 0.015, 0.35]) #pas untuk non
    plt.colorbar(max, cax = cax) 
        
    #file_name='Corr_temporal_season2_SEA'
    file_name='p_trend_year-sum3mmew_p005'+reg
    fig.savefig(workdir+file_name,dpi=300,bbox_inches='tight')
    plt.show()


def p_trend_taylor(obs_dataset, obs_name, model_datasets, model_names, workdir):
    #with p-value<005 n using //
    #diukur similarity pattern
    
    print('p_trend...')
    import pymannkendall as mk

    #yg spatial => R di diagram Taylor
    #temporal bulanan, musiman-tahunan perlu?
    #import xarray as xr
    ds = xr.DataArray(obs_dataset.values,
        coords={'time': obs_dataset.times,
        'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
        dims=["time", "lat", "lon"])  
    ds = ds.groupby('time.year').sum()
    print('temporal subsetting =>', ds.values.shape)
    
    #Atur ini
    plot='SEA'
    model=14 # maks 10 no MME
    set=7     # set 5 for 10 model 
    tes=0
    fig, ax = plt.subplots(nrows=3, ncols=7,figsize=(3,4))
    #geser plot for SEA
    if plot=='SEA':
        lat_min1 = 0.22*3
        lat_max1 = 0.22*6
        lon_min1 = 0.22*4
        lon_max1 = 0.22*3
    
    if plot=='sum':
        lat_min1 = 0.22*1
        lat_max1 = 0.22*3
        lon_min1 = 0.22*3
        lon_max1 = 0.22*1
    
    
    lat_min = ds.lat.min()
    lat_max = ds.lat.max()
    lon_min = ds.lon.min()
    lon_max = ds.lon.max()
   
    x,y = np.meshgrid(ds.lon, ds.lat)
    
    vmax=3
    vmin=-3
    #levels = np.linspace(vmin,vmax,6)
    levels=[-3,-2,-1,0,1,2,3]
    #levels=np.arange(6)
    norm = plt.Normalize(vmin, vmax)
 
    
    map0=ma.zeros((len(ds.lat),len(ds.lon)))
    map11=ma.zeros((len(ds.lat),len(ds.lon)))
    for ii in np.arange(len(ds.lat)-tes):
        print(ii)
        for jj in np.arange(len(ds.lon)-tes):
            x0=ds[:,ii,jj]
           
            #ds year ==> tidak ada nan tapi 0 semua, 
            #monthly ada nan maka 
            if np.isnan(x0).any() or x0.sum()==0:
                s='nan'; sig='nan'
            else:
                k=mk.original_test(x0)
                s=k.z
                sig=k.p 
                #if k.p<0.05:  s=k.z
                #print(sig)
            map0[ii,jj]=s
            map11[ii,jj]=sig
    print(np.nanmin(map11), np.nanmax(map11))       
    m = Basemap(ax=ax[0,0], projection ='cyl', 
                llcrnrlat = lat_min+lat_min1, urcrnrlat = lat_max-lat_max1,
                llcrnrlon = lon_min+lon_min1, urcrnrlon = lon_max-lon_max1, 
                resolution = 'l', fix_aspect=False)
    m.drawcoastlines(linewidth=1)
    m.drawcountries(linewidth=1)
    m.drawstates(linewidth=0.5, color='w')
    
    max = ax[0,0].contourf(x,y,map0,levels=levels, norm=norm, extend='both')
    #max = ax[0,0].contourf(x,y,map0) #, extend='both')
    ax[0,0].set_title(obs_name, pad=3, fontsize=7)
    #hatching
    ax[0,0].contourf(x, y, map11,levels=[np.nanmin(map11), 0.05, ], 
                        hatches=[5*'/'],
                        extend='lower', alpha = 0)
    
    ##khusus SEA
    if plot=='SEA':
        ax[0,0].set_yticks([-10,0,10,20])
        ax[0,0].set_yticklabels(['10$^\circ$S','EQ','10$^\circ$N','20$^\circ$N'])
    if plot=='sum':ax[0,0].set_yticks([-5,0,5])
    ax[0,0].yaxis.set_tick_params(labelsize=6)
    
    #cbar=plt.colorbar(max, extend='both', orientation='horizontal')
    #plt.show()
    #exit()
    #----------------------------------
    r0=1
    x00=map0.flatten()
    T=[]
    
    from scipy.stats import pearsonr
    import pandas as pd

    mean=[]
 
    for i in range(0,model): #0-9 set 10 for 10 model 
        if i<6: ax[0,1+i].axis('off')
        print (model_names[i])
        #print(x1)
       
        dsi = xr.DataArray(model_datasets[i].values,
                coords={'time': obs_dataset.times,
                'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
                dims=["time", "lat", "lon"])       
                
        #dsi = dsi.groupby('time.month').mean()
        dsi = dsi.groupby('time.year').sum()
        print('temporal subsetting =>', dsi.values.shape)
        if i<set:
            n=i
            #pergerseran map yang min di + yang max di -
            #untuk koreksi fig map, SEA hanya lat_max-3 => 7
            #                                 lon_min+3 => 4
            m = Basemap(ax=ax[1,n], projection ='cyl', 
                llcrnrlat = lat_min+lat_min1, urcrnrlat = lat_max-lat_max1,
                llcrnrlon = lon_min+lon_min1, urcrnrlon = lon_max-lon_max1, 
                resolution = 'l', fix_aspect=False)
            m.drawcoastlines(linewidth=1)
            m.drawcountries(linewidth=1)
            m.drawstates(linewidth=0.5, color='w')

            map1=ma.zeros((len(ds.lat),len(ds.lon)))
            map11=ma.zeros((len(ds.lat),len(ds.lon)))
            for ii in np.arange(len(ds.lat)-tes):
                print(ii)
                for jj in np.arange(len(ds.lon)-tes):
                    x0=dsi[:,ii,jj]
                   
                    #ds year > tidak ada nan tapi 0 semua, 
                    #monthly ada nan maka 
                    if np.isnan(x0).any() or x0.sum()==0:
                        s='nan'; sig='nan'
                    else:
                        k=mk.original_test(x0)
                        s=k.z
                        sig=k.p 
                        #if k.p<0.05:  s=k.z
                     
                    map1[ii,jj]=s
                    map11[ii,jj]=sig

            y0=map1.flatten()
            bad = ~np.logical_or(np.isnan(x00), np.isnan(y0))
            x1=np.compress(bad, x00) 
            y1=np.compress(bad, y0)
            
            
            sd1=x1.std() #(skipna=None)
            #print(sd1)
            sd2=y1.std()
            s=sd2/sd1
            
            c,pp=pearsonr(x1.flatten() , y1.flatten())        
            
            #T= (4*(1+c)**4)/(((s+1/s)**2)*((1+r0)**4))
            # Tanpa pakai **4 T naik dikit
            tt= (4*(1+c))/(((s+1/s)**2)*((1+r0)))
            T.append(round(tt,2))
            #T.append(tt)
            #me=np.nanmean(map1)
            #max = ax[0,n].contourf(x,y,map1) #,levels=levels, vmin=0,vmax=1)#, extend='both')
            #max = ax[0,n].contourf(x,y,map1, vmin=0,vmax=3)#, extend='both')
            max = ax[1,n].contourf(x,y,map1,levels=levels, norm=norm, extend='both')
            ax[1,n].set_title(model_names[i]+' ('+'%.2f'%tt+')', pad=3,fontsize=7)
            #hatching
            ax[1,n].contourf(x, y, map11,levels=[np.nanmin(map11), 0.05, ], 
                        hatches=[5*'/'],
                        extend='lower', alpha = 0)
            
            ##khusus SEA
            if plot=='SEA':
                ax[1,0].set_yticks([-10,0,10,20])
                ax[1,0].set_yticklabels(['10$^\circ$S','EQ','10$^\circ$N','20$^\circ$N'])
            if plot=='sum':ax[1,0].set_yticks([-5,0,5])
            
            #khusus sumatera, SEA terlalu besar tidak jelas
            #for i in np.arange(len(x1)):
            #    ax[0,n].annotate('+', (x1[i],y1[i]),fontsize=8)
                #ax[0,n].text( x=x1[i], y=y1[i],s='+')
            
        else:
            n=i-set
            m = Basemap(ax=ax[2,n], projection ='cyl', 
                llcrnrlat = lat_min+lat_min1, urcrnrlat = lat_max-lat_max1,
                llcrnrlon = lon_min+lon_min1, urcrnrlon = lon_max-lon_max1, 
                resolution = 'l', fix_aspect=False)
            m.drawcoastlines(linewidth=1)
            m.drawcountries(linewidth=1)
            m.drawstates(linewidth=0.5, color='w')
        
            map1=ma.zeros((len(ds.lat),len(ds.lon)))
            map11=ma.zeros((len(ds.lat),len(ds.lon)))
            for ii in np.arange(len(ds.lat)-tes):
                print(ii)
                for jj in np.arange(len(ds.lon)-tes):
                    x0=dsi[:,ii,jj]
                    
                    #ds year > tidak ada nan tapi 0 semua, monthly ada
                    if np.isnan(x0).any() or x0.sum()==0:
                        s='nan' ; sig='nan'
                        #map1[ii,jj]=s
                        #print(s)
                    else:
                        k=mk.original_test(x0)
                        s=k.z
                        sig=k.p 
                        #if k.p<0.05:  s=k.z
                           
                    #print(s)
                    map1[ii,jj]=s
                    map11[ii,jj]=sig            
            y0=map1.flatten()
            bad = ~np.logical_or(np.isnan(x00), np.isnan(y0))
            x1=np.compress(bad, x00) 
            y1=np.compress(bad, y0)
            
            
            sd1=x1.std() #(skipna=None)
            #print(sd1)
            sd2=y1.std()
            s=sd2/sd1
            
            c,pp=pearsonr(x1.flatten() , y1.flatten())        
            
            #T= (4*(1+c)**4)/(((s+1/s)**2)*((1+r0)**4))
            # Tanpa pakai **4 T naik dikit
            tt= (4*(1+c))/(((s+1/s)**2)*((1+r0)))
            T.append(round(tt,2))
            #T.append(tt)
            max = ax[2,n].contourf(x,y,map1,levels=levels, norm=norm, extend='both')            
            ax[2,n].set_title(model_names[i]+' ('+'%.2f'%tt+')', pad=3,fontsize=7)
            
            #hatching
            print(np.nanmin(map11), np.nanmax(map11))       
            if np.nanmin(map11)<=0.05: 
                ax[2,n].contourf(x, y, map11,levels=[np.nanmin(map11), 0.05 ], 
                        hatches=[5*'/'],
                        extend='lower', alpha = 0)
            
            #khusus SEA ini
            if plot=='SEA':
                ax[2,0].set_yticks([-10,0,10,20])
                ax[2,0].set_yticklabels(['10$^\circ$S','EQ','10$^\circ$N','20$^\circ$N'])
                ax[2,n].set_xticks([100,120,140])
                ax[2,n].set_xticklabels(['100$^\circ$E','120$^\circ$E','140$^\circ$E'])
                ax[2,n].xaxis.set_tick_params(labelsize=6)
                ax[1,0].yaxis.set_tick_params(labelsize=6)
                ax[2,0].yaxis.set_tick_params(labelsize=6)
            if plot=='sum':
                ax[2,0].set_yticks([-5,0,5])
                ax[2,0].set_yticklabels(['5S','0','5N'])
                #ax[1,n] jika semua bawah di kasih ini
                ax[2,n].set_xticks([96,100,104])
                ax[2,n].set_xticklabels(['96E','100E','104E'])                 
                ax[2,n].xaxis.set_tick_params(labelsize=6)
                ax[1,0].yaxis.set_tick_params(labelsize=6)
                ax[2,0].yaxis.set_tick_params(labelsize=6)
                
    print('T[]=',T)
    
    #df = pd.DataFrame([model_names, mean])
    # 
    #df.T.to_excel(workdir+'mean_ss2_sum_L-tes.xlsx', index=False, header=False) 
    
    plt.subplots_adjust(hspace=.15,wspace=.05)
    #cax = fig.add_axes([0.91, 0.5, 0.015, 0.4]) #pas untuk extend both
    #cax = fig.add_axes([0.91, 0.53, 0.015, 0.35]) #pas untuk non
    cax = fig.add_axes([0.35, 0.7, 0.4, 0.02]) #horisontal
    #plt.colorbar(max, cax = cax, extend='both', orientation='horizontal') 
    cbar=plt.colorbar(max, cax = cax, extend='both', orientation='horizontal') 
    cbar.ax.tick_params(labelsize=7)
    
    plt.show()
    #file_name='Corr_temporal_season2_SEA'
    file_name='pr_trend_taylor_mon_5obs_'+reg
    fig.savefig(workdir+file_name,dpi=300,bbox_inches='tight')


def p_trend_taylor2(obs_dataset, obs_name, model_datasets, model_names, workdir):
    
    #p-value<005 only 
    #diukur similarity pattern
    
    print('p_trend...')
    import pymannkendall as mk

    ds = xr.DataArray(obs_dataset.values,
        coords={'time': obs_dataset.times,
        'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
        dims=["time", "lat", "lon"])  
    
    ds = ds.groupby('time.year').sum()

    print('temporal subsetting =>', ds.values.shape)
    
    #Atur ini
    plot= 'sum' 
    #plot= 'SEA'
    model=5 #14  # maks 10 no MME
    set=7     # set 5 for 10 model 
    tes=50
    
    #geser plot for SEA
    if plot=='SEA':
        lat_min1 = 0.22*3
        lat_max1 = 0.22*6
        lon_min1 = 0.22*4
        lon_max1 = 0.22*3
    
    if plot=='sum':
        lat_min1 = 0.22*1
        lat_max1 = 0.22*3
        lon_min1 = 0.22*3
        lon_max1 = 0.22*1
    
    
    lat_min = ds.lat.min()
    lat_max = ds.lat.max()
    lon_min = ds.lon.min()
    lon_max = ds.lon.max()
   
    x,y = np.meshgrid(ds.lon, ds.lat)
    
    vmax=3
    vmin=-3
    #levels = np.linspace(vmin,vmax,6)
    levels=[-3,-2,-1,0,1,2,3]
    #levels=np.arange(6)
    norm = plt.Normalize(vmin, vmax)
    
    fig, ax = plt.subplots(nrows=3, ncols=7,figsize=(3,4))
    
    
    map0=ma.zeros((len(ds.lat),len(ds.lon)))
    map11=ma.zeros((len(ds.lat),len(ds.lon)))
    for ii in np.arange(len(ds.lat)-tes):
        print(ii)
        for jj in np.arange(len(ds.lon)-tes):
            x0=ds[:,ii,jj]
           
            #ds year > tidak ada nan tapi 0 semua, 
            #monthly ada nan maka 
            if np.isnan(x0).any() or x0.sum()==0:
                s='nan'; sig='nan'
            else:
                k=mk.original_test(x0)
                #s=k.z
                
                if k.p<=0.05: 
                    #sig=k.p
                    s=k.z
                    
                
            map0[ii,jj]=s
            #map11[ii,jj]=sig
    print('map_z_min_max)',np.nanmin(map0), np.nanmax(map0))  
    m = Basemap(ax=ax[0,0], projection ='cyl', 
                llcrnrlat = lat_min+lat_min1, urcrnrlat = lat_max-lat_max1,
                llcrnrlon = lon_min+lon_min1, urcrnrlon = lon_max-lon_max1, 
                resolution = 'l', fix_aspect=False)
    m.drawcoastlines(linewidth=1)
    m.drawcountries(linewidth=1)
    m.drawstates(linewidth=0.5, color='w')
    
    max = ax[0,0].contourf(x,y,map0,levels=levels, norm=norm , extend='both')
    #ax[0,0].set_title(obs_name, pad=3, fontsize=7)
    #hatching didaratan baru ada jika p=0.5 di lautan p=0.05
    #print('map_sig_min_max)',np.nanmin(map11), np.nanmax(map11))
    #ax[0,0].contourf(x, y, map11,levels=[np.nanmin(map11), 0.05, ], 
    #                    hatches=[7*'/'],
    #                    #extend='lower', 
    #                    alpha = 0)
    
    ##khusus SEA
    if plot=='SEA':
        ax[0,0].set_yticks([-10,0,10,20])
        ax[0,0].set_yticklabels(['10$^\circ$S','EQ','10$^\circ$N','20$^\circ$N'])
    if plot=='sum':ax[0,0].set_yticks([-5,0,5])
    ax[0,0].yaxis.set_tick_params(labelsize=6)
    
    #khsus era5
    ax[0,1].axis('off')
    ax[0,0].set_title(obs_name, pad=3,fontsize=7)
    
    #cbar=plt.colorbar(max, extend='both', orientation='horizontal')
    #plt.show()
    #exit()
    #----------------------------------
    r0=1
    x00=map0.flatten()
    T=[]
    
    from scipy.stats import pearsonr
    import pandas as pd

    mean=[]
 
    for i in range(0,model): #0-9 set 10 for 10 model 
        if i<6: ax[0,1+i].axis('off')
        print (model_names[i])
        #print(x1)
       
        dsi = xr.DataArray(model_datasets[i].values,
                coords={'time': obs_dataset.times,
                'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
                dims=["time", "lat", "lon"])       
                
        #dsi = dsi.groupby('time.month').mean()
        dsi = dsi.groupby('time.year').sum()
        print('temporal subsetting =>', dsi.values.shape)
        if i<set:
            n=i
            #pergerseran map yang min di + yang max di -
            #untuk koreksi fig map, SEA hanya lat_max-3 => 7
            #                                 lon_min+3 => 4
            m = Basemap(ax=ax[1,n], projection ='cyl', 
                llcrnrlat = lat_min+lat_min1, urcrnrlat = lat_max-lat_max1,
                llcrnrlon = lon_min+lon_min1, urcrnrlon = lon_max-lon_max1, 
                resolution = 'l', fix_aspect=False)
            m.drawcoastlines(linewidth=1)
            m.drawcountries(linewidth=1)
            m.drawstates(linewidth=0.5, color='w')

            map1=ma.zeros((len(ds.lat),len(ds.lon)))
            map11=ma.zeros((len(ds.lat),len(ds.lon)))
            for ii in np.arange(len(ds.lat)-tes):
                print(ii)
                for jj in np.arange(len(ds.lon)-tes):
                    x0=dsi[:,ii,jj]
                   
                    #ds year > tidak ada nan tapi 0 semua, 
                    #monthly ada nan maka 
                    if np.isnan(x0).any() or x0.sum()==0:
                        s='nan'; sig='nan'
                    else:
                        k=mk.original_test(x0)
                        #s=k.z
                
                        if k.p<=0.05: 
                            #sig=k.p
                            s=k.z
                        
                     
                    map1[ii,jj]=s
                    #map11[ii,jj]=sig

            y0=map1.flatten()
            bad = ~np.logical_or(np.isnan(x00), np.isnan(y0))
            x1=np.compress(bad, x00) 
            y1=np.compress(bad, y0)
            
            
            sd1=x1.std() #(skipna=None)
            #print(sd1)
            sd2=y1.std()
            s=sd2/sd1
            
            c,pp=pearsonr(x1.flatten() , y1.flatten())        
            
            #T= (4*(1+c)**4)/(((s+1/s)**2)*((1+r0)**4))
            # Tanpa pakai **4 T naik dikit
            tt= (4*(1+c))/(((s+1/s)**2)*((1+r0)))
            T.append(round(tt,2))
            #T.append(tt)
            #me=np.nanmean(map1)
            #max = ax[0,n].contourf(x,y,map1) #,levels=levels, vmin=0,vmax=1)#, extend='both')
            #max = ax[0,n].contourf(x,y,map1, vmin=0,vmax=3)#, extend='both')
            max = ax[1,n].contourf(x,y,map1,levels=levels, norm=norm, extend='both')
            ax[1,n].set_title(model_names[i]+' ('+'%.2f'%tt+')', pad=3,fontsize=7)
            # #hatching
            # max = ax[1,n].contourf(x, y, map11,levels=[np.nanmin(map11), 0.05, ], 
                        # hatches=[5*'/'],
                        # extend='both', 
                        # alpha = 0)
            
            ##khusus SEA
            if plot=='SEA':
                ax[1,0].set_yticks([-10,0,10,20])
                ax[1,0].set_yticklabels(['10$^\circ$S','EQ','10$^\circ$N','20$^\circ$N'])
            if plot=='sum':ax[1,0].set_yticks([-5,0,5])
            
            #khusus sumatera, SEA terlalu besar tidak jelas
            #for i in np.arange(len(x1)):
            #    ax[0,n].annotate('+', (x1[i],y1[i]),fontsize=8)
                #ax[0,n].text( x=x1[i], y=y1[i],s='+')
            
        else:
            n=i-set
            m = Basemap(ax=ax[2,n], projection ='cyl', 
                llcrnrlat = lat_min+lat_min1, urcrnrlat = lat_max-lat_max1,
                llcrnrlon = lon_min+lon_min1, urcrnrlon = lon_max-lon_max1, 
                resolution = 'l', fix_aspect=False)
            m.drawcoastlines(linewidth=1)
            m.drawcountries(linewidth=1)
            m.drawstates(linewidth=0.5, color='w')
        
            map1=ma.zeros((len(ds.lat),len(ds.lon)))
            map11=ma.zeros((len(ds.lat),len(ds.lon)))
            for ii in np.arange(len(ds.lat)-tes):
                print(ii)
                for jj in np.arange(len(ds.lon)-tes):
                    x0=dsi[:,ii,jj]
                    
                    #ds year > tidak ada nan tapi 0 semua, monthly ada
                    if np.isnan(x0).any() or x0.sum()==0:
                        s='nan' ; sig='nan'
                        #map1[ii,jj]=s
                        #print(s)
                    else:
                        k=mk.original_test(x0)
                        if k.p<=0.05:
                            s=k.z
                            #sig=k.p
                       
                    map1[ii,jj]=s
                    #map11[ii,jj]=sig            
            y0=map1.flatten()
            bad = ~np.logical_or(np.isnan(x00), np.isnan(y0))
            x1=np.compress(bad, x00) 
            y1=np.compress(bad, y0)
            
            
            sd1=x1.std() #(skipna=None)
            #print(sd1)
            sd2=y1.std()
            s=sd2/sd1
            
            c,pp=pearsonr(x1.flatten() , y1.flatten())        
            
            #T= (4*(1+c)**4)/(((s+1/s)**2)*((1+r0)**4))
            # Tanpa pakai **4 T naik dikit
            tt= (4*(1+c))/(((s+1/s)**2)*((1+r0)))
            T.append(round(tt,2))
            #T.append(tt)
            max = ax[2,n].contourf(x,y,map1,levels=levels, norm=norm, extend='both')             
            ax[2,n].set_title(model_names[i]+' ('+'%.2f'%tt+')', pad=3,fontsize=7)
            
            # #hatching
            # #print(np.nanmin(map11), np.nanmax(map11))       
            # #if np.nanmin(map11)<=0.05: 
            # max = ax[2,n].contourf(x, y, map11,levels=[np.nanmin(map11), 0.05 ], 
                        # hatches=[5*'/'],
                        # extend='both', 
                        # alpha = 0)
            
            #khusus SEA ini
            if plot=='SEA':
                ax[2,0].set_yticks([-10,0,10,20])
                ax[2,0].set_yticklabels(['10$^\circ$S','EQ','10$^\circ$N','20$^\circ$N'])
                ax[2,n].set_xticks([100,120,140])
                ax[2,n].set_xticklabels(['100$^\circ$E','120$^\circ$E','140$^\circ$E'])
                ax[2,n].xaxis.set_tick_params(labelsize=6)
                ax[1,0].yaxis.set_tick_params(labelsize=6)
                ax[2,0].yaxis.set_tick_params(labelsize=6)
            if plot=='sum':
                ax[2,0].set_yticks([-5,0,5])
                #ax[1,n] jika semua bawah di kasih ini
                ax[2,n].set_xticks([92,96,100,104]) 
                ax[2,n].xaxis.set_tick_params(labelsize=6)
                ax[1,0].yaxis.set_tick_params(labelsize=6)
                ax[2,0].yaxis.set_tick_params(labelsize=6)
                
    print('T[]=',T)
    
    #df = pd.DataFrame([model_names, mean])
    # 
    #df.T.to_excel(workdir+'mean_ss2_sum_L-tes.xlsx', index=False, header=False) 
    
    plt.subplots_adjust(hspace=.15,wspace=.05)
    #cax = fig.add_axes([0.91, 0.5, 0.015, 0.4]) #pas untuk extend both
    #cax = fig.add_axes([0.91, 0.53, 0.015, 0.35]) #pas untuk non
    cax = fig.add_axes([0.35, 0.7, 0.4, 0.02]) #horisontal
    #plt.colorbar(max, cax = cax, extend='both', orientation='horizontal') 
    cbar=plt.colorbar(max, cax = cax, extend='both', orientation='horizontal') 
    cbar.ax.tick_params(labelsize=7)
    
    plt.show()
    #file_name='Corr_temporal_season2_SEA'
    file_name='p_trend_year_3mmew_taylor_era5_z2_'+reg
    fig.savefig(workdir+file_name,dpi=300,bbox_inches='tight')
    
def p_trend_taylor2_era(obs_dataset, obs_name, model_datasets, model_names, workdir):
    
    #p-value<005 only 
    #diukur similarity pattern
    
    print('p_trend...')
    import pymannkendall as mk

    ds = xr.DataArray(obs_dataset.values,
        coords={'time': obs_dataset.times,
        'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
        dims=["time", "lat", "lon"])  
    
    #ds = ds.groupby('time.year').sum()

    print('temporal subsetting =>', ds.values.shape)
    
    #Atur ini
    #plot= 'sum' 
    plot= 'SEA'
    model=10 #14  # maks 10 no MME
    set=5    # set 5 for 10 model 
    tes=0
    
    #geser plot for SEA
    if plot=='SEA':
        lat_min1 = 0.22*3
        lat_max1 = 0.22*6
        lon_min1 = 0.22*4
        lon_max1 = 0.22*3
    
    if plot=='sum':
        lat_min1 = 0.22*1
        lat_max1 = 0.22*3
        lon_min1 = 0.22*3
        lon_max1 = 0.22*1
    
    
    lat_min = ds.lat.min()
    lat_max = ds.lat.max()
    lon_min = ds.lon.min()
    lon_max = ds.lon.max()
   
    x,y = np.meshgrid(ds.lon, ds.lat)
    
    vmax=3
    vmin=-3
    #levels = np.linspace(vmin,vmax,6)
    levels=[-3,-2,-1,0,1,2,3]
    #levels=np.arange(6)
    norm = plt.Normalize(vmin, vmax)
    
    fig, ax = plt.subplots(nrows=3, ncols=5,figsize=(3,4))
    
    
    map0=ma.zeros((len(ds.lat),len(ds.lon)))
    map11=ma.zeros((len(ds.lat),len(ds.lon)))
    for ii in np.arange(len(ds.lat)-tes):
        print(ii)
        for jj in np.arange(len(ds.lon)-tes):
            x0=ds[:,ii,jj]
           
            #ds year > tidak ada nan tapi 0 semua, 
            #monthly ada nan maka 
            if np.isnan(x0).any() or x0.sum()==0:
                s='nan'; sig='nan'
            else:
                k=mk.original_test(x0)
                #s=k.z
                
                if k.p<=0.05: 
                    #sig=k.p
                    s=k.z
                    
                
            map0[ii,jj]=s
            #map11[ii,jj]=sig
    print('map_z_min_max)',np.nanmin(map0), np.nanmax(map0))  
    m = Basemap(ax=ax[0,0], projection ='cyl', 
                llcrnrlat = lat_min+lat_min1, urcrnrlat = lat_max-lat_max1,
                llcrnrlon = lon_min+lon_min1, urcrnrlon = lon_max-lon_max1, 
                resolution = 'l', fix_aspect=False)
    m.drawcoastlines(linewidth=1)
    m.drawcountries(linewidth=1)
    m.drawstates(linewidth=0.5, color='w')
    
    max = ax[0,0].contourf(x,y,map0,levels=levels, norm=norm , extend='both')
    #ax[0,0].set_title(obs_name, pad=3, fontsize=7)
    #hatching didaratan baru ada jika p=0.5 di lautan p=0.05
    #print('map_sig_min_max)',np.nanmin(map11), np.nanmax(map11))
    #ax[0,0].contourf(x, y, map11,levels=[np.nanmin(map11), 0.05, ], 
    #                    hatches=[7*'/'],
    #                    #extend='lower', 
    #                    alpha = 0)
    
    ##khusus SEA
    if plot=='SEA':
        ax[0,0].set_yticks([-10,0,10,20])
        ax[0,0].set_yticklabels(['10$^\circ$S','0','10$^\circ$N','20$^\circ$N'])
    if plot=='sum':ax[0,0].set_yticks([-5,0,5])
    ax[0,0].yaxis.set_tick_params(labelsize=7)
    
    #khsus era5
    ax[0,1].axis('off')
    ax[0,0].set_title(obs_name, pad=3,fontsize=9)
    
    #cbar=plt.colorbar(max, extend='both', orientation='horizontal')
    #plt.show()
    #exit()
    #----------------------------------
    r0=1
    x00=map0.flatten()
    T=[]
    
    from scipy.stats import pearsonr
    import pandas as pd

    mean=[]
  
    for i in range(0,model): #0-9 set 10 for 10 model 
        if i<4: ax[0,1+i].axis('off')
        print (model_names[i])
        #print(x1)
       
        dsi = xr.DataArray(model_datasets[i].values,
                coords={'time': obs_dataset.times,
                'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
                dims=["time", "lat", "lon"])       
                
        #dsi = dsi.groupby('time.month').mean()
        #dsi = dsi.groupby('time.year').sum()
        print('temporal subsetting =>', dsi.values.shape)
        if i<set:
            n=i
            #pergerseran map yang min di + yang max di -
            #untuk koreksi fig map, SEA hanya lat_max-3 => 7
            #                                 lon_min+3 => 4
            m = Basemap(ax=ax[1,n], projection ='cyl', 
                llcrnrlat = lat_min+lat_min1, urcrnrlat = lat_max-lat_max1,
                llcrnrlon = lon_min+lon_min1, urcrnrlon = lon_max-lon_max1, 
                resolution = 'l', fix_aspect=False)
            m.drawcoastlines(linewidth=1)
            m.drawcountries(linewidth=1)
            m.drawstates(linewidth=0.5, color='w')

            map1=ma.zeros((len(ds.lat),len(ds.lon)))
            map11=ma.zeros((len(ds.lat),len(ds.lon)))
            for ii in np.arange(len(ds.lat)-tes):
                print(ii)
                for jj in np.arange(len(ds.lon)-tes):
                    x0=dsi[:,ii,jj]
                   
                    #ds year > tidak ada nan tapi 0 semua, 
                    #monthly ada nan maka 
                    if np.isnan(x0).any() or x0.sum()==0:
                        s='nan'; sig='nan'
                    else:
                        k=mk.original_test(x0)
                        #s=k.z
                
                        if k.p<=0.05: 
                            #sig=k.p
                            s=k.z
                        
                     
                    map1[ii,jj]=s
                    #map11[ii,jj]=sig

            y0=map1.flatten()
            bad = ~np.logical_or(np.isnan(x00), np.isnan(y0))
            x1=np.compress(bad, x00) 
            y1=np.compress(bad, y0)
            
            
            sd1=x1.std() #(skipna=None)
            #print(sd1)
            sd2=y1.std()
            s=sd2/sd1
            
            c,pp=pearsonr(x1.flatten() , y1.flatten())        
            
            #T= (4*(1+c)**4)/(((s+1/s)**2)*((1+r0)**4))
            # Tanpa pakai **4 T naik dikit
            tt= (4*(1+c))/(((s+1/s)**2)*((1+r0)))
            T.append(round(tt,2))
            #T.append(tt)
            #me=np.nanmean(map1)
            #max = ax[0,n].contourf(x,y,map1) #,levels=levels, vmin=0,vmax=1)#, extend='both')
            #max = ax[0,n].contourf(x,y,map1, vmin=0,vmax=3)#, extend='both')
            max = ax[1,n].contourf(x,y,map1,levels=levels, norm=norm, extend='both')
            ax[1,n].set_title(model_names[i]+' ('+'%.2f'%tt+')', pad=3,fontsize=9)
    
            if plot=='SEA':
                ax[1,0].set_yticks([-10,0,10,20])
                ax[1,0].set_yticklabels(['10$^\circ$S','0','10$^\circ$N','20$^\circ$N'])
            if plot=='sum':ax[1,0].set_yticks([-5,0,5])
            
            #khusus sumatera, SEA terlalu besar tidak jelas
            #for i in np.arange(len(x1)):
            #    ax[0,n].annotate('+', (x1[i],y1[i]),fontsize=8)
                #ax[0,n].text( x=x1[i], y=y1[i],s='+')
            
        else:
            n=i-set
            m = Basemap(ax=ax[2,n], projection ='cyl', 
                llcrnrlat = lat_min+lat_min1, urcrnrlat = lat_max-lat_max1,
                llcrnrlon = lon_min+lon_min1, urcrnrlon = lon_max-lon_max1, 
                resolution = 'l', fix_aspect=False)
            m.drawcoastlines(linewidth=1)
            m.drawcountries(linewidth=1)
            m.drawstates(linewidth=0.5, color='w')
        
            map1=ma.zeros((len(ds.lat),len(ds.lon)))
            map11=ma.zeros((len(ds.lat),len(ds.lon)))
            for ii in np.arange(len(ds.lat)-tes):
                print(ii)
                for jj in np.arange(len(ds.lon)-tes):
                    x0=dsi[:,ii,jj]
                    
                    #ds year > tidak ada nan tapi 0 semua, monthly ada
                    if np.isnan(x0).any() or x0.sum()==0:
                        s='nan' ; sig='nan'
                        #map1[ii,jj]=s
                        #print(s)
                    else:
                        k=mk.original_test(x0)
                        if k.p<=0.05:
                            s=k.z
                            #sig=k.p
                       
                    map1[ii,jj]=s
                    #map11[ii,jj]=sig            
            y0=map1.flatten()
            bad = ~np.logical_or(np.isnan(x00), np.isnan(y0))
            x1=np.compress(bad, x00) 
            y1=np.compress(bad, y0)
            
            
            sd1=x1.std() #(skipna=None)
            #print(sd1)
            sd2=y1.std()
            s=sd2/sd1
            
            c,pp=pearsonr(x1.flatten() , y1.flatten())        
            
            #T= (4*(1+c)**4)/(((s+1/s)**2)*((1+r0)**4))
            # Tanpa pakai **4 T naik dikit
            tt= (4*(1+c))/(((s+1/s)**2)*((1+r0)))
            T.append(round(tt,2))
            #T.append(tt)
            max = ax[2,n].contourf(x,y,map1,levels=levels, norm=norm, extend='both')             
            ax[2,n].set_title(model_names[i]+' ('+'%.2f'%tt+')', pad=3,fontsize=9)
            
            # #hatching
            # #print(np.nanmin(map11), np.nanmax(map11))       
            # #if np.nanmin(map11)<=0.05: 
            # max = ax[2,n].contourf(x, y, map11,levels=[np.nanmin(map11), 0.05 ], 
                        # hatches=[5*'/'],
                        # extend='both', 
                        # alpha = 0)
            
            #khusus SEA ini
            if plot=='SEA':
                ax[2,0].set_yticks([-10,0,10,20])
                ax[2,0].set_yticklabels(['10$^\circ$S','0','10$^\circ$N','20$^\circ$N'])
                ax[2,n].set_xticks([100,120,140])
                ax[2,n].set_xticklabels(['100$^\circ$E','120$^\circ$E','140$^\circ$E'])
                ax[2,n].xaxis.set_tick_params(labelsize=7)
                ax[1,0].yaxis.set_tick_params(labelsize=7)
                ax[2,0].yaxis.set_tick_params(labelsize=7)
            if plot=='sum':
                ax[2,0].set_yticks([-5,0,5])
                #ax[1,n] jika semua bawah di kasih ini
                ax[2,n].set_xticks([92,96,100,104]) 
                ax[2,n].xaxis.set_tick_params(labelsize=7)
                ax[1,0].yaxis.set_tick_params(labelsize=7)
                ax[2,0].yaxis.set_tick_params(labelsize=7)
                
    print('T[]=',T)
    
    #df = pd.DataFrame([model_names, mean])
    # 
    #df.T.to_excel(workdir+'mean_ss2_sum_L-tes.xlsx', index=False, header=False) 
    
    plt.subplots_adjust(hspace=.15,wspace=.05)
    #cax = fig.add_axes([0.91, 0.5, 0.015, 0.4]) #pas untuk extend both
    #cax = fig.add_axes([0.91, 0.53, 0.015, 0.35]) #pas untuk non
    cax = fig.add_axes([0.35, 0.7, 0.4, 0.02]) #horisontal
    #plt.colorbar(max, cax = cax, extend='both', orientation='horizontal') 
    cbar=plt.colorbar(max, cax = cax, extend='both', orientation='horizontal') 
    cbar.ax.tick_params(labelsize=7)
    
    plt.show()
    #file_name='Corr_temporal_season2_SEA'
    file_name='p_trend_monthly_taylor_11era_'+reg
    fig.savefig(workdir+file_name,dpi=300,bbox_inches='tight')


def p_trend_5obs(obs_dataset, obs_name, model_datasets, model_names, workdir):
    
    #p-value<005 only 
    #diukur similarity pattern
    
    print('p_trend...')
    import pymannkendall as mk

    ds = xr.DataArray(obs_dataset.values,
        coords={'time': obs_dataset.times,
        'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
        dims=["time", "lat", "lon"])  
    
    ds = ds.groupby('time.year').sum()

    print('temporal subsetting =>', ds.values.shape)
    
    #Atur ini
    plot= 'sum' 
    #plot= 'SEA'
    model= len(model_datasets) #5  # maks 10 no MME
    
    tes=0
    
    #geser plot for SEA
    if plot=='SEA':
        lat_min1 = 0.22*3
        lat_max1 = 0.22*6
        lon_min1 = 0.22*4
        lon_max1 = 0.22*3
    
    if plot=='sum':
        lat_min1 = 0.22*1
        lat_max1 = 0.22*3
        lon_min1 = 0.22*3
        lon_max1 = 0.22*1
    
    
    lat_min = ds.lat.min()
    lat_max = ds.lat.max()
    lon_min = ds.lon.min()
    lon_max = ds.lon.max()
   
    x,y = np.meshgrid(ds.lon, ds.lat)
    
    vmax=3
    vmin=-3
    #levels = np.linspace(vmin,vmax,6)
    levels=[-3,-2,-1,0,1,2,3]
    #levels=np.arange(6)
    norm = plt.Normalize(vmin, vmax)
    
    fig, ax = plt.subplots(nrows=1, ncols=5,figsize=(3,4))
    
    
    map0=ma.zeros((len(ds.lat),len(ds.lon)))
    map11=ma.zeros((len(ds.lat),len(ds.lon)))
    for ii in np.arange(len(ds.lat)-tes):
        #print(ii)
        for jj in np.arange(len(ds.lon)-tes):
            x0=ds[:,ii,jj]
           
            #ds year > tidak ada nan tapi 0 semua, 
            #monthly ada nan maka 
            if np.isnan(x0).any() or x0.sum()==0:
                s='nan'; sig='nan'
            else:
                k=mk.original_test(x0)
                #s=k.z
                
                if k.p<=0.05: 
                    #sig=k.p
                    s=k.z
                
            map0[ii,jj]=s
            #map11[ii,jj]=sig
    print('map_z_min_max)',np.nanmin(map0), np.nanmax(map0))  
    m = Basemap(ax=ax[0], projection ='cyl', 
                llcrnrlat = lat_min+lat_min1, urcrnrlat = lat_max-lat_max1,
                llcrnrlon = lon_min+lon_min1, urcrnrlon = lon_max-lon_max1, 
                resolution = 'l', fix_aspect=False)
    m.drawcoastlines(linewidth=1)
    m.drawcountries(linewidth=1)
    m.drawstates(linewidth=0.5, color='w')
    
    max = ax[0].contourf(x,y,map0,levels=levels, norm=norm , extend='both')
    #ax[0,0].set_title(obs_name, pad=3, fontsize=7)
    #hatching didaratan baru ada jika p=0.5 di lautan p=0.05
    #print('map_sig_min_max)',np.nanmin(map11), np.nanmax(map11))
    #ax[0,0].contourf(x, y, map11,levels=[np.nanmin(map11), 0.05, ], 
    #                    hatches=[7*'/'],
    #                    #extend='lower', 
    #                    alpha = 0)
    
    ##khusus SEA
    if plot=='SEA':
        ax[0,0].set_yticks([-10,0,10,20])
        ax[0,0].set_yticklabels(['10$^\circ$S','EQ','10$^\circ$N','20$^\circ$N'])
    if plot=='sum':
        ax[0].set_yticks([-5,0,5])
        ax[0].yaxis.set_tick_params(labelsize=6)
    
    
    ax[0].set_title(obs_name, pad=3,fontsize=9)
    
   
    r0=1
    x00=map0.flatten()
    T=[]
    
    from scipy.stats import pearsonr
    import pandas as pd

    mean=[]
    n=0
    for i in range(0,model): #0-9 set 10 for 10 model 
        n=0
        print (model_names[i])
        #print(x1)
       
        dsi = xr.DataArray(model_datasets[i].values,
                coords={'time': obs_dataset.times,
                'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
                dims=["time", "lat", "lon"])       
                
        #dsi = dsi.groupby('time.month').mean()
        dsi = dsi.groupby('time.year').sum()
        print('temporal subsetting =>', dsi.values.shape)
       
        
        #pergerseran map yang min di + yang max di -
        #untuk koreksi fig map, SEA hanya lat_max-3 => 7
        #                                 lon_min+3 => 4
        m = Basemap(ax=ax[i], projection ='cyl', 
            llcrnrlat = lat_min+lat_min1, urcrnrlat = lat_max-lat_max1,
            llcrnrlon = lon_min+lon_min1, urcrnrlon = lon_max-lon_max1, 
            resolution = 'l', fix_aspect=False)
        m.drawcoastlines(linewidth=1)
        m.drawcountries(linewidth=1)
        m.drawstates(linewidth=0.5, color='w')

        map1=ma.zeros((len(ds.lat),len(ds.lon)))
        map11=ma.zeros((len(ds.lat),len(ds.lon)))
        for ii in np.arange(len(ds.lat)-tes):
            #print(ii)
            for jj in np.arange(len(ds.lon)-tes):
                x0=dsi[:,ii,jj]
               
                #ds year > tidak ada nan tapi 0 semua, 
                #monthly ada nan maka 
                if np.isnan(x0).any() or x0.sum()==0:
                    s='nan'; sig='nan'
                else:
                    k=mk.original_test(x0)
                    #s=k.z
            
                    if k.p<=0.05: 
                        #sig=k.p
                        s=k.z
                        n=n+1
                        print(n)
                    
                 
                map1[ii,jj]=s
                #map11[ii,jj]=sig
            print('n=',n)
        y0=map1.flatten()
        bad = ~np.logical_or(np.isnan(x00), np.isnan(y0))
        x1=np.compress(bad, x00) 
        y1=np.compress(bad, y0)
        
        
        sd1=x1.std() #(skipna=None)
        #print(sd1)
        sd2=y1.std()
        s=sd2/sd1
        
        c,pp=pearsonr(x1.flatten() , y1.flatten())        
        
        #T= (4*(1+c)**4)/(((s+1/s)**2)*((1+r0)**4))
        # Tanpa pakai **4 T naik dikit
        tt= (4*(1+c))/(((s+1/s)**2)*((1+r0)))
        T.append(round(tt,2))
        #T.append(tt)
        #me=np.nanmean(map1)
        #max = ax[0,n].contourf(x,y,map1) #,levels=levels, vmin=0,vmax=1)#, extend='both')
        #max = ax[0,n].contourf(x,y,map1, vmin=0,vmax=3)#, extend='both')
        max = ax[i].contourf(x,y,map1,levels=levels, norm=norm, extend='both')
        if i>0:
            ax[i].set_title(model_names[i]+' ('+'%.2f'%tt+')', pad=3,fontsize=9)
        # #hatching
        # max = ax[1,n.contourf(x, y, map11,levels=[np.nanmin(map11), 0.05, ], 
                    # hatches=[5*'/'],
                    # extend='both', 
                    # alpha = 0)
        
        if plot=='SEA':
            ax[2,0].set_yticks([-10,0,10,20])
            ax[2,0].set_yticklabels(['10$^\circ$S','EQ','10$^\circ$N','20$^\circ$N'])
            ax[2,i].set_xticks([100,120,140])
            ax[2,n].set_xticklabels(['100$^\circ$E','120$^\circ$E','140$^\circ$E'])
            ax[2,n].xaxis.set_tick_params(labelsize=6)
            ax[1,0].yaxis.set_tick_params(labelsize=6)
            ax[2,0].yaxis.set_tick_params(labelsize=6)
        
        if plot=='sum':
            ax[0].set_yticks([-6,-3,0,3,6])
            ax[0].set_yticklabels(['6$^\circ$S','3$^\circ$S','0','3$^\circ$N','6$^\circ$N'])
            ax[i].set_xticks([95,100,105]) 
            ax[i].set_xticklabels(['95$^\circ$E','100$^\circ$E','105$^\circ$E']) 
            ax[i].xaxis.set_tick_params(labelsize=8)
            ax[0].yaxis.set_tick_params(labelsize=8)
           
                
    print('T[]=',T)
    
    #df = pd.DataFrame([model_names, mean])
    # 
    #df.T.to_excel(workdir+'mean_ss2_sum_L-tes.xlsx', index=False, header=False) 
    plt.subplots_adjust(bottom=.3)
    plt.subplots_adjust(hspace=.15,wspace=.05)
    cax = fig.add_axes([0.91, 0.35, 0.013, 0.5]) #pas untuk extend both
    #cax = fig.add_axes([0.25, 0.22, 0.5, 0.027]) #hori
    #cax = fig.add_axes([0.91, 0.3, 0.015, 0.6])
    #cbar=plt.colorbar(max, cax = cax, extend='both', orientation='horizontal') 
    cbar=plt.colorbar(max, cax = cax, extend='both') 
    cbar.ax.tick_params(labelsize=7)
    
    plt.show()
    #file_name='Corr_temporal_season2_SEA'
    file_name='p_trend_year_5obs_'+reg
    fig.savefig(workdir+file_name,dpi=300,bbox_inches='tight')

def p_trend2(obs_dataset, obs_name, model_datasets, model_names, workdir):
    #untuk 10 model tanpa mme, mmew
    print('p_trend...')
    import pymannkendall as mk

    #yg spatial => R di diagram Taylor
    #temporal bulanan, musiman-tahunan perlu?
    #import xarray as xr
    ds = xr.DataArray(obs_dataset.values,
        coords={'time': obs_dataset.times,
        'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
        dims=["time", "lat", "lon"])  

    print('temporal subsetting =>', ds.values.shape)
    
    #Atur ini
    #plot='sum'
    plot='SEA'
    model=10  # maks 10 no MME
    set=5     # set 5 for 10 model 
    tes=175
    
    #geser plot for SEA
    if plot=='SEA':
        lat_min1 = 0.22*3
        lat_max1 = 0.22*6
        lon_min1 = 0.22*4
        lon_max1 = 0.22*3
    
    if plot=='sum':
        lat_min1 = 0.22*1
        lat_max1 = 0.22*3
        lon_min1 = 0.22*3
        lon_max1 = 0.22*1
    
    
    lat_min = ds.lat.min()
    lat_max = ds.lat.max()
    lon_min = ds.lon.min()
    lon_max = ds.lon.max()
   
    x,y = np.meshgrid(ds.lon, ds.lat)
    
    fig, ax = plt.subplots(nrows=2, ncols=5,figsize=(8,6))
    
    #from scipy.stats import pearsonr
    import pandas as pd
   
    #model_datasets=np.delete(model_datasets,[-1])
    #model_names=np.delete(model_names,[-1])
    
    vmax=3
    vmin=-3
    #levels = np.linspace(vmin,vmax,6)
    levels=[-3,-2,-1,0,1,2,3]
    #levels=np.arange(6)
    norm = plt.Normalize(vmin, vmax)
    mean=[]
    x1=[]
    y1=[]
    x2=[]
    y2=[]
    for i in range(0,model): #0-9 set 10 for 10 model 
        
        print (model_names[i])
        print(x1)
       
        dsi = xr.DataArray(model_datasets[i].values,
                coords={'time': obs_dataset.times,
                'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
                dims=["time", "lat", "lon"])       
                
        #dsi = dsi.groupby('time.month').mean()
        dsi = dsi.groupby('time.year').sum()
        print('temporal subsetting =>', dsi.values.shape)
        if i<set:
            n=i
            #pergerseran map yang min di + yang max di -
            #untuk koreksi fig map, SEA hanya lat_max-3 => 7
            #                                 lon_min+3 => 4
            m = Basemap(ax=ax[0,n], projection ='cyl', 
                llcrnrlat = lat_min+lat_min1, urcrnrlat = lat_max-lat_max1,
                llcrnrlon = lon_min+lon_min1, urcrnrlon = lon_max-lon_max1, 
                resolution = 'l', fix_aspect=False)
            m.drawcoastlines(linewidth=1)
            m.drawcountries(linewidth=1)
            m.drawstates(linewidth=0.5, color='w')

            map1=ma.zeros((len(ds.lat),len(ds.lon)))
            for ii in np.arange(len(ds.lat)-tes):
                print(ii)
                for jj in np.arange(len(ds.lon)-tes):
                    x0=dsi[:,ii,jj]
                   
                    #ds year > tidak ada nan tapi 0 semua, 
                    #monthly ada nan maka 
                    if np.isnan(x0).any() or x0.sum()==0:
                        s='nan'
                    else:
                        k=mk.original_test(x0)
                        s=k.z
                                       
                        if k.p<0.05:
                            #print(ii,jj, ds.lat[ii].data,ds.lon[jj].data)
                            x1.append(ds.lon[jj].data)
                            y1.append(ds.lat[ii].data)
                            # print(k.p)
                            #print(k.z)
                            #print(k.slope)
                            #print(k.trend)
                            #s=k.z
                     
                    map1[ii,jj]=s
            
            me=np.nanmean(map1)
            #max = ax[0,n].contourf(x,y,map1) #,levels=levels, vmin=0,vmax=1)#, extend='both')
            #max = ax[0,n].contourf(x,y,map1, vmin=0,vmax=3)#, extend='both')
            max = ax[0,n].contourf(x,y,map1,levels=levels, norm=norm) #, extend='both')
            ax[0,n].set_title(model_names[i]+'('+'%.2f'%me+')',fontsize=8)
            ##khusus SEA
            if plot=='SEA':ax[0,0].set_yticks([-10,0,10,20])
            if plot=='sum':ax[0,0].set_yticks([-6,-3,0,3,6])
            
            #khusus sumatera, SEA terlalu besar tidak jelas
            for i in np.arange(len(x1)):
                ax[0,n].annotate('+', (x1[i],y1[i]),fontsize=8)
                #ax[0,n].text( x=x1[i], y=y1[i],s='+')
            
        else:
            n=i-set
            m = Basemap(ax=ax[1,n], projection ='cyl', 
                llcrnrlat = lat_min+lat_min1, urcrnrlat = lat_max-lat_max1,
                llcrnrlon = lon_min+lon_min1, urcrnrlon = lon_max-lon_max1, 
                resolution = 'l', fix_aspect=False)
            m.drawcoastlines(linewidth=1)
            m.drawcountries(linewidth=1)
            m.drawstates(linewidth=0.5, color='w')
        
            map1=ma.zeros((len(ds.lat),len(ds.lon)))

            for ii in np.arange(len(ds.lat)-tes):
                print(ii)
                for jj in np.arange(len(ds.lon)-tes):
                    x0=dsi[:,ii,jj]
                    
                    #ds year > tidak ada nan tapi 0 semua, monthly ada
                    if np.isnan(x0).any() or x0.sum()==0:
                        s='nan'
                        #map1[ii,jj]=s
                        #print(s)
                    else:
                        k=mk.original_test(x0)
                        s=k.z
                    
                        if k.p<0.05:
                            #print(ii,jj, ds.lat[ii].data,ds.lon[jj].data)
                            x2.append(ds.lon[jj].data)
                            y2.append(ds.lat[ii].data)
                            # print(k.p)
                            #print(k.z)
                            #print(k.slope)
                            #print(k.trend)
                            #s=k.z
                           
                    #print(s)
                    map1[ii,jj]=s
                                
            me=np.nanmean(map1)
            max = ax[1,n].contourf(x,y,map1,levels=levels, norm=norm) 
            
            ax[1,n].set_title(model_names[i]+'('+'%.2f'%me+')',fontsize=8)
            #khusus SEA ini
            if plot=='SEA':
                ax[1,0].set_yticks([-10,0,10,20])
                ax[1,n].set_xticks([90,100,110,120,130,140])
                ax[1,n].xaxis.set_tick_params(labelsize=6)
                ax[0,0].yaxis.set_tick_params(labelsize=6)
                ax[1,0].yaxis.set_tick_params(labelsize=6)
            if plot=='sum':
                ax[1,0].set_yticks([-6,-3,0,3,6])
                #ax[1,n] jika semua bawah di kasih ini
                ax[1,n].set_xticks([94,98,102,106]) 
                ax[1,n].xaxis.set_tick_params(labelsize=6)
                ax[0,0].yaxis.set_tick_params(labelsize=6)
                ax[1,0].yaxis.set_tick_params(labelsize=6)
                
            for i in np.arange(len(x2)):
                ax[1,n].annotate('+', (x2[i],y2[i]),fontsize=8)
        mean.append(round(me,2))
        
        
        print('cek_max=', np.nanmax(map1)) 
        print('cek_min=', np.nanmin(map1))     
        print('mean_kopi ke excel',mean)
        
        #if i==1:
            #df = pd.DataFrame(map1)
            #df.to_excel(workdir+'tes2.xlsx')
            #exit()
        x1=[]
        y1=[]
        x2=[]
        y2=[]
    df = pd.DataFrame([model_names, mean])
    # 
    #df.T.to_excel(workdir+'mean_ss2_sum_L-tes.xlsx', index=False, header=False) 
    
    plt.subplots_adjust(hspace=.12,wspace=.05)
    #cax = fig.add_axes([0.91, 0.5, 0.015, 0.4]) #pas untuk extend both
    cax = fig.add_axes([0.91, 0.53, 0.015, 0.35]) #pas untuk non
    plt.colorbar(max, cax = cax) 
    
    plt.show()
    #file_name='Corr_temporal_season2_SEA'
    file_name='p_trend_year-sum3mmew_p005_era5'+reg
    fig.savefig(workdir+file_name,dpi=300,bbox_inches='tight')
    

def corr_spatial(obs_dataset, obs_name, model_datasets, model_names, workdir):
    #yg spatial => R di diagram Taylor
    #temporal bulanan, musiman-tahunan perlu?
    #import xarray as xr
    ds = xr.DataArray(model_datasets[0].values,
        coords={'time': obs_dataset.times,
        'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
        dims=["time", "lat", "lon"])  
    dst=ds.mean(dim='time')
    dsm=ds.groupby('time.season')
 
    
    
    from scipy.stats import pearsonr
    
    model_datasets=np.delete(model_datasets,[-1])
    model_names=np.delete(model_names,[-1])
    
   
    ct=[]
    cj=[]
    cd=[]
    
    for i in range(0,10):
        print (i)
        
        if i==0:
            dsi=xr.DataArray(obs_dataset.values,
                coords={'time': obs_dataset.times,
                'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
                dims=["time", "lat", "lon"])  
            model_names[0]='GPCP'
            
            #dsi=dsi .groupby('time.season')
        else:
            dsi=xr.DataArray(model_datasets[i].values,
                coords={'time': obs_dataset.times,
                'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
                dims=["time", "lat", "lon"]) 
            #dsi=dsi .groupby('time.season')
        dsit=dsi.mean(dim='time')
        dsim=dsi.groupby('time.season')
        
        ct.append(xr.corr(dsit,dst).values.tolist()) 
        cd.append(xr.corr(dsim['DJF'],dsm['DJF']).values.tolist())
        cj.append(xr.corr(dsim['JJA'],dsm['JJA']).values.tolist())  
    
    
    print(model_names)
    print('ct=',ct)
    print('cd=',cd)
    print('cj=',cj)
    N = 10
    ind = np.arange(N)
    width = 0.25

    xvals = ct
    bar1 = plt.bar(ind, xvals, width)#, color = 'r')

    yvals = cd
    bar2 = plt.bar(ind+width, yvals, width)#, color='g')

    zvals = cj
    bar3 = plt.bar(ind+width*2, zvals, width)#, color = 'b')

    #model_names=['GPCP','CNRM_a', 'ECE_b', 'IPSL_b', 'HadGEM2_d', 'HadGEM2_c', 'HadGEM2_a',
    #'MPI_c', 'NorESM1_d', 'GFDL_b']
    plt.xticks(ind+width,model_names, fontsize=8)
    plt.xticks(rotation=45)
    plt.legend( (bar1, bar2, bar3), ('Clim', 'DJF', 'JJA') )
    plt.subplots_adjust(bottom=.2)
    file_name='Corr_spatial_'+reg
    plt.savefig(workdir+file_name,dpi=300,bbox_inches='tight')
    plt.show()
         

def corr_temporal_season(obs_dataset, obs_name, model_datasets, model_names, workdir):
    #yg spatial => R di diagram Taylor
    #temporal bulanan, musiman-tahunan perlu?
    #import xarray as xr
    ds = xr.DataArray(model_datasets[0].values,
        coords={'time': obs_dataset.times,
        'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
        dims=["time", "lat", "lon"])  
    
    #ds=ds.groupby('time.season')
 
    lat_min = ds.lat.min()
    lat_max = ds.lat.max()
    lon_min = ds.lon.min()
    lon_max = ds.lon.max()
   
    x,y = np.meshgrid(ds.lon, ds.lat)
    
    fig, ax = plt.subplots(nrows=2, ncols=5,figsize=(8,6))
    
    from scipy.stats import pearsonr
    
    model_datasets=np.delete(model_datasets,[-1])
    model_names=np.delete(model_names,[-1])
    
    vmax=1
    vmin=0
    #levels = np.linspace(vmin,vmax,6)
    levels=np.arange(11)/10
    tes=0
    n=0
    for musim in ['DJF', 'JJA']:
        for i in range(0,5):
            print (i)
            
            if i==0:
                dsi=xr.DataArray(obs_dataset.values,
                    coords={'time': obs_dataset.times,
                    'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
                    dims=["time", "lat", "lon"])  
                model_names[0]='GPCP'
                #dsi=dsi .groupby('time.season')
            else:
                dsi=xr.DataArray(model_datasets[i].values,
                    coords={'time': obs_dataset.times,
                    'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
                    dims=["time", "lat", "lon"]) 
                #dsi=dsi .groupby('time.season')
            
            #pergerseran map yang min di + yang max di -
            #untuk koreksi fig map, SEA hanya lat_max-3 => 7
            #                                 lon_min+3 => 4
            m = Basemap(ax=ax[n,i], projection ='cyl', 
                llcrnrlat = lat_min+1*0.22, urcrnrlat = lat_max-6*0.22,
                llcrnrlon = lon_min+4*0.22, urcrnrlon = lon_max-3*0.22, 
                resolution = 'l', fix_aspect=False)
            m.drawcoastlines(linewidth=1)
            m.drawcountries(linewidth=1)
            m.drawstates(linewidth=0.5, color='w')

            map1=ma.zeros((len(ds.lat),len(ds.lon)))
            for ii in np.arange(len(ds.lat)-tes):
                #print(ii)
                for jj in np.arange(len(ds.lon)-tes):
                    d=dsi[:,ii,jj]
                    dd=ds[:,ii,jj]
                    d=d.groupby('time.season')
                    dd=dd.groupby('time.season')
                    x0=d[musim].values
                    y0=dd[musim].values
                    
                    bad = ~np.logical_or(np.isnan(x0), np.isnan(y0))
                    x1=np.compress(bad, x0) 
                    y1=np.compress(bad, y0)
                    #print(x,y)
                    #print(x.shape,y.shape)
                    if x1.shape==(0,) or y1.shape==(0,):
                        R='nan'
                    else:
                        R=pearsonr(x1, y1)[0]                
                    map1[ii,jj]=R
            #print('cek_Rmax=', np.nanmax(map1))
            #mean
            me=np.nanmean(map1)
            #print(me)        
            
            max = ax[n,i].contourf(x,y,map1,levels=levels, vmin=0,vmax=1)#, extend='both')
            ax[n,i].set_title(model_names[i]+'('+'%.2f'%me+')',fontsize=8.5)
            #ax[0,0].set_yticks([-10,0,10,20])
            #ax[0,1+i].set_xticks([90,100,110,120,130,140])
            ax[n,0].set_ylabel(musim)
        n=n+1
            
    
    plt.subplots_adjust(hspace=.18,wspace=.05)
    #cax = fig.add_axes([0.91, 0.5, 0.015, 0.4]) #pas untuk extend both
    cax = fig.add_axes([0.91, 0.53, 0.015, 0.35]) #pas untuk non
    plt.colorbar(max, cax = cax) 
        
    file_name='Corr_temporal_season_'+reg
    fig.savefig(workdir+file_name,dpi=300,bbox_inches='tight')
    plt.show()

def corr_temporal_annual_cycle(obs_dataset, obs_name, model_datasets, model_names, workdir):
    #yg spatial => R di diagram Taylor
    #temporal bulanan, musiman-tahunan perlu?
    #import xarray as xr
    ds = xr.DataArray(obs_dataset.values,
        coords={'time': obs_dataset.times,
        'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
        dims=["time", "lat", "lon"])  
    ds = ds.groupby('time.month').mean()
    #ds = ds.groupby('time.season').mean()
    print('temporal subsetting =>', ds.values.shape)
    
    #Atur ini
    plot='sum'
    model=1    # maks 10 no MME
    set=5       # set 5 for 10 model 
    tes=0
    
    #geser plot for SEA
    if plot=='SEA':
        lat_min1 = 0.22*3
        lat_max1 = 0.22*6
        lon_min1 = 0.22*4
        lon_max1 = 0.22*3
    
    if plot=='sum':
        lat_min1 = 0.22*1
        lat_max1 = 0.22*3
        lon_min1 = 0.22*3
        lon_max1 = 0.22*1
    
    
    lat_min = ds.lat.min()
    lat_max = ds.lat.max()
    lon_min = ds.lon.min()
    lon_max = ds.lon.max()
   
    x,y = np.meshgrid(ds.lon, ds.lat)
    
    fig, ax = plt.subplots(nrows=2, ncols=5,figsize=(8,6))
    
    from scipy.stats import pearsonr
    
    model_datasets=np.delete(model_datasets,[-1])
    model_names=np.delete(model_names,[-1])
    
    vmax=1
    vmin=0
    #levels = np.linspace(vmin,vmax,6)
    levels=np.arange(11)/10
    
    for i in range(0,model): #0-9 set 10 for 10 model 
        print (i)
        
        # if i==0:
            # dsi = xr.DataArray(obs_dataset.values,
                # coords={'time': obs_dataset.times,
                # 'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
                # dims=["time", "lat", "lon"])  
            # model_names[0]='GPCP'
        # else:
        dsi = xr.DataArray(model_datasets[i].values,
                coords={'time': obs_dataset.times,
                'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
                dims=["time", "lat", "lon"])       
                
        dsi = dsi.groupby('time.month').mean()
        #dsi = dsi.groupby('time.season').mean()
        if i<set:
            n=i
            #pergerseran map yang min di + yang max di -
            #untuk koreksi fig map, SEA hanya lat_max-3 => 7
            #                                 lon_min+3 => 4
            m = Basemap(ax=ax[0,n], projection ='cyl', 
                llcrnrlat = lat_min+lat_min1, urcrnrlat = lat_max-lat_max1,
                llcrnrlon = lon_min+lon_min1, urcrnrlon = lon_max-lon_max1, 
                resolution = 'l', fix_aspect=False)
            m.drawcoastlines(linewidth=1)
            m.drawcountries(linewidth=1)
            m.drawstates(linewidth=0.5, color='w')

            map1=ma.zeros((len(ds.lat),len(ds.lon)))
            for ii in np.arange(len(ds.lat)-tes):
                #print(ii)
                for jj in np.arange(len(ds.lon)-tes):
                    x0=dsi[:,ii,jj]
                    y0=ds[:,ii,jj]
                    bad = ~np.logical_or(np.isnan(x0), np.isnan(y0))
                    x1=np.compress(bad, x0) 
                    y1=np.compress(bad, y0)
                    #print(x,y)
                    #print(x.shape,y.shape)
                    if x1.shape==(0,) or y1.shape==(0,):
                        R='nan'
                    else:
                        R=pearsonr(x1, y1)[0]                
                    map1[ii,jj]=R
            #print('cek_Rmax=', np.nanmax(map1))
            #mean
            me=np.nanmean(map1)
            #print(me)        
            
            max = ax[0,n].contourf(x,y,map1,levels=levels, vmin=0,vmax=1)#, extend='both')
            ax[0,n].set_title(model_names[i]+'('+'%.2f'%me+')',fontsize=9)
            #khusus SEA
            if plot=='SEA':ax[0,0].set_yticks([-10,0,10,20])
            if plot=='sum':ax[0,0].set_yticks([-5,0,5])
           
            
        else:
            n=i-set
            m = Basemap(ax=ax[1,n], projection ='cyl', 
                llcrnrlat = lat_min+lat_min1, urcrnrlat = lat_max-lat_max1,
                llcrnrlon = lon_min+lon_min1, urcrnrlon = lon_max-lon_max1, 
                resolution = 'l', fix_aspect=False)
            m.drawcoastlines(linewidth=1)
            m.drawcountries(linewidth=1)
            m.drawstates(linewidth=0.5, color='w')
        
           
            map1=ma.zeros((len(ds.lat),len(ds.lon)))

            for ii in np.arange(len(ds.lat)-tes):
                #print(ii)
                for jj in np.arange(len(ds.lon)-tes):
                    x0=dsi[:,ii,jj]
                    y0=ds[:,ii,jj]
                    bad = ~np.logical_or(np.isnan(x0), np.isnan(y0))
                    x1=np.compress(bad, x0) 
                    y1=np.compress(bad, y0)
                    #print(x,y)
                    #print(x.shape,y.shape)
                    if x1.shape==(0,) or y1.shape==(0,):
                        R='nan'
                    else:
                        R=pearsonr(x1, y1)[0]
                    map1[ii,jj]=R
                   
            me=np.nanmean(map1)
            max = ax[1,n].contourf(x,y,map1,levels=levels, vmin=0,vmax=1)#, extend='both')
            ax[1,n].set_title(model_names[i]+'('+'%.2f'%me+')',fontsize=9)
            #khusus SEA ini
            if plot=='SEA':
                ax[1,0].set_yticks([-10,0,10,20])
                ax[1,n].set_xticks([90,100,110,120,130,140])
                ax[1,n].xaxis.set_tick_params(labelsize=6)
                ax[0,0].yaxis.set_tick_params(labelsize=6)
                ax[1,0].yaxis.set_tick_params(labelsize=6)
            if plot=='sum':
                ax[1,0].set_yticks([-5,0,5])
                ax[1,n].set_xticks([92,96,100,104])
                ax[1,n].xaxis.set_tick_params(labelsize=6)
                ax[0,0].yaxis.set_tick_params(labelsize=6)
                ax[1,0].yaxis.set_tick_params(labelsize=6)
    
    plt.subplots_adjust(hspace=.12,wspace=.05)
    #cax = fig.add_axes([0.91, 0.5, 0.015, 0.4]) #pas untuk extend both
    cax = fig.add_axes([0.91, 0.53, 0.015, 0.35]) #pas untuk non
    plt.colorbar(max, cax = cax) 
        
    #file_name='Corr_temporal_season2_SEA'
    file_name='Corr_temporal_annual_cycle_'+reg
    #fig.savefig(workdir+file_name,dpi=300,bbox_inches='tight')
    plt.show()

def corr_temporal(obs_dataset, obs_name, model_datasets, model_names, workdir):
    #yg spatial => R di diagram Taylor
    #temporal bulanan, musiman-tahunan perlu?
    #import xarray as xr
    ds = xr.DataArray(model_datasets[0].values,
        coords={'time': obs_dataset.times,
        'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
        dims=["time", "lat", "lon"])  
 
    lat_min = ds.lat.min()
    lat_max = ds.lat.max()
    lon_min = ds.lon.min()
    lon_max = ds.lon.max()
   
    x,y = np.meshgrid(ds.lon, ds.lat)
    
    fig, ax = plt.subplots(nrows=2, ncols=5,figsize=(8,6))
    
    from scipy.stats import pearsonr
    
    model_datasets=np.delete(model_datasets,[-1])
    model_names=np.delete(model_names,[-1])
    
    vmax=1
    vmin=0
    #levels = np.linspace(vmin,vmax,6)
    levels=np.arange(11)/10
    set=5
    for i in range(0,10):
        print (i)
        
        if i==0:
            dsi = xr.DataArray(obs_dataset.values,
                coords={'time': obs_dataset.times,
                'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
                dims=["time", "lat", "lon"])  
            model_names[0]='GPCP'
        else:
            dsi = xr.DataArray(model_datasets[i].values,
                coords={'time': obs_dataset.times,
                'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
                dims=["time", "lat", "lon"])         
        
        if i<set:
            n=i
            #pergerseran map yang min di + yang max di -
            #untuk koreksi fig map, SEA hanya lat_max-3 => 7
            #                                 lon_min+3 => 4
            m = Basemap(ax=ax[0,n], projection ='cyl', 
                llcrnrlat = lat_min+1*0.22, urcrnrlat = lat_max-6*0.22,
                llcrnrlon = lon_min+4*0.22, urcrnrlon = lon_max-3*0.22, 
                resolution = 'l', fix_aspect=False)
            m.drawcoastlines(linewidth=1)
            m.drawcountries(linewidth=1)
            m.drawstates(linewidth=0.5, color='w')

            map1=ma.zeros((len(ds.lat),len(ds.lon)))
            for ii in np.arange(len(ds.lat)):
                #print(ii)
                for jj in np.arange(len(ds.lon)):
                    x0=dsi[:,ii,jj]
                    y0=ds[:,ii,jj]
                    bad = ~np.logical_or(np.isnan(x0), np.isnan(y0))
                    x1=np.compress(bad, x0) 
                    y1=np.compress(bad, y0)
                    #print(x,y)
                    #print(x.shape,y.shape)
                    if x1.shape==(0,) or y1.shape==(0,):
                        R='nan'
                    else:
                        R=pearsonr(x1, y1)[0]                
                    map1[ii,jj]=R
            #print('cek_Rmax=', np.nanmax(map1))
            #mean
            me=np.nanmean(map1)
            #print(me)        
            
            max = ax[0,n].contourf(x,y,map1,levels=levels, vmin=0,vmax=1)#, extend='both')
            ax[0,n].set_title(model_names[i]+'('+'%.2f'%me+')',fontsize=9)
            #ax[0,0].set_yticks([-10,0,10,20])
            #ax[0,1+i].set_xticks([90,100,110,120,130,140])
            
        else:
            n=i-set
            m = Basemap(ax=ax[1,n], projection ='cyl', 
                llcrnrlat = lat_min+1*0.22, urcrnrlat = lat_max-6*0.22,
                llcrnrlon = lon_min+4*0.22, urcrnrlon = lon_max-3*0.22, 
                resolution = 'l', fix_aspect=False)
            m.drawcoastlines(linewidth=1)
            m.drawcountries(linewidth=1)
            m.drawstates(linewidth=0.5, color='w')
        
           
            map1=ma.zeros((len(ds.lat),len(ds.lon)))

            for ii in np.arange(len(ds.lat)):
                #print(ii)
                for jj in np.arange(len(ds.lon)):
                    x0=dsi[:,ii,jj]
                    y0=ds[:,ii,jj]
                    bad = ~np.logical_or(np.isnan(x0), np.isnan(y0))
                    x1=np.compress(bad, x0) 
                    y1=np.compress(bad, y0)
                    #print(x,y)
                    #print(x.shape,y.shape)
                    if x1.shape==(0,) or y1.shape==(0,):
                        R='nan'
                    else:
                        R=pearsonr(x1, y1)[0]
                    map1[ii,jj]=R
                   
            me=np.nanmean(map1)
            max = ax[1,n].contourf(x,y,map1,levels=levels, vmin=0,vmax=1)#, extend='both')
            ax[1,n].set_title(model_names[i]+'('+'%.2f'%me+')',fontsize=9)
            #khusus SEA ini
            #ax[1,0].set_yticks([-10,0,10,20])
            #ax[1,n].set_xticks([90,100,110,120,130,140])
            
    
    plt.subplots_adjust(hspace=.18,wspace=.05)
    #cax = fig.add_axes([0.91, 0.5, 0.015, 0.4]) #pas untuk extend both
    cax = fig.add_axes([0.91, 0.53, 0.015, 0.35]) #pas untuk non
    plt.colorbar(max, cax = cax) 
        
    file_name='Corr_temporal_'+reg
    #fig.savefig(workdir+file_name,dpi=300,bbox_inches='tight')
    plt.show()

def pr_max_map(obs_dataset, obs_name, model_datasets, model_names,  names, workdir):
  
    for i in np.arange(len(model_datasets)):
        if i==2:
            print(model_names[i])
            #if i==3: 
            dsi = xr.DataArray(model_datasets[i].values,
            coords={'time': obs_dataset.times,
            'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
            dims=["time", "lat", "lon"]) 
           
           #d=xr.where(dsi.values >275,dsi.values,0)
            max=dsi.max().values
            print('max=',max)
            print('mean=',dsi.mean().values)
            t, lat_i, lon_i = np.nonzero(xr.where(dsi>=max, 1, 0).data)
            #lats = dsi.isel(lat=lat_i).data
            #lons = dsi.isel(lon=lon_i).data
            time = dsi.time.isel(time=t).data
            #pairs = list(zip(lats, lons))
            #print(lons)
            print('t=',t,time)
            #print('mean_t=',dsi[t,:,:].mean().values)
            
    
            lat_min = dsi.lat.min()
            lat_max = dsi.lat.max()
            lon_min = dsi.lon.min()
            lon_max = dsi.lon.max()
            
            fig, ax = plt.subplots(nrows=1, ncols=1 ,figsize=(6,5))
    
            x,y = np.meshgrid(dsi.lon, dsi.lat)
            # n=0
            # map1=dsi[t-1,:,:].mean(dim='time')
            map2=dsi[t,:,:].mean(dim='time')
            # map3=dsi[t+1,:,:].mean(dim='time')
            # map=[map1,map3,map2]
            
            levels=np.arange(5)
            #norm = plt.Normalize(vmin=300, vmax=500)
            cmaps = ['RdBu_r', 'viridis']
           
            m = Basemap(ax=ax, projection ='cyl', llcrnrlat = lat_min+1*.22, urcrnrlat = lat_max-3*.22,
            llcrnrlon = lon_min+7*.22, urcrnrlon = lon_max, resolution = 'l', fix_aspect=False)
            m.drawcoastlines(linewidth=1)
            m.drawcountries(linewidth=1)
            m.drawstates(linewidth=0.5, color='w')
           
            #print(map.shape)
            #max1 = ax[n,0].contourf(x,y,dsi[t,:,:].mean(dim='time'))
            pcm = ax.contourf(x,y,map2)#,norm=norm,cmap=cmaps[0])
            ax.set_title(model_names[i],fontsize=8)
            #ax[n,0].set_ylabel(musim)
            #ax[0,0].set_yticks([-10,0,10,20])
            #ax[0,0].set_xticks([90,100,110,120,130,140])
           
            cax = fig.add_axes([0.92, 0.5, 0.02, 0.35])
            cax.tick_params(labelsize=6)
            fig.colorbar(pcm, cax = cax) 
    #ax.clabel(pcm, inline=True, fontsize=10)
    plt.subplots_adjust(hspace=.05,wspace=.05)
    
    #file_name='cek_max_map_SEA_new_HadGEM_d_3'
    file_name='cek_max_map_'+reg
    
    fig.savefig(workdir+file_name,dpi=300)
    plt.show()


        
def pr_extreme_detection(obs_dataset, obs_name, model_datasets, model_names,  names, workdir):
    #adopted from ...
    
    
    i=9
    print(model_names[i])
    
    # Initialisation
    slp=model_datasets[i].values
    lon=obs_dataset.lons
    lat=obs_dataset.lats
    #print('lon', lon)
    print('lat', lat)

    lon_storms_a = []
    lat_storms_a = []
    amp_storms_a = []
    lon_storms_c = []
    lat_storms_c = []
    amp_storms_c = []
    

    for tt in np.arange(len(obs_dataset.times)):
        #
        print ('tt=', tt)
        #
        # Detect lon and lat coordinates 
        #anticyclonic
        
        lon_storms, lat_storms, amp = detect_extreme(slp[tt,:,:], lon, lat, res=2, Npix_min=9,
                                                          cyc='anticyclonic', globe=False)
        lon_storms_a.append(lon_storms)
        lat_storms_a.append(lat_storms)
        amp_storms_a.append(amp)
        # print('lon_storms, lat_storms, amp', lon_storms_a, lat_storms_a, amp_storms_a)
        print('lon_storms, lat_storms, amp', lon_storms, lat_storms, amp)
        
        #cyclonic
        lon_storms, lat_storms, amp = detect_extreme(slp[tt,:,:], lon, lat, res=2, Npix_min=9,
                                                          cyc='cyclonic', globe=False)
        lon_storms_c.append(lon_storms)
        lat_storms_c.append(lat_storms)
        amp_storms_c.append(amp)
        #print('lon_storms, lat_storms, amp', lon_storms_c, lat_storms_c, amp_storms_c)
        
        print('lon_storms, lat_storms, amp', lon_storms, lat_storms, amp)
        
def detect_extreme(field, lon, lat, res, Npix_min, cyc, globe=False):
    print('def detect_storms ...')
    import scipy.ndimage as ndimage
    '''
    Detect storms present in field which satisfy the criteria.
    Algorithm is an adaptation of an eddy detection algorithm,
    outlined in Chelton et al., Prog. ocean., 2011, App. B.2,
    with modifications needed for storm detection.

    field is a 2D array specified on grid defined by lat and lon.

    res is the horizontal grid resolution in degrees of field

    Npix_min is the minimum number of pixels within which an
    extremum of field must lie (recommended: 9).

    cyc = 'cyclonic' or 'anticyclonic' specifies type of system
    to be detected (cyclonic storm or high-pressure systems)

    globe is an option to detect storms on a globe, i.e. with periodic
    boundaries in the West/East. Note that if using this option the 
    supplied longitudes must be positive only (i.e. 0..360 not -180..+180).

    Function outputs lon, lat coordinates of detected storms
    '''

  
    llon, llat = np.meshgrid(lon, lat)

    lon_storms = np.array([])
    lat_storms = np.array([])
    amp_storms = np.array([])

    # ssh_crits is an array of ssh levels over which to perform storm detection loop
    # ssh_crits increasing for 'cyclonic', decreasing for 'anticyclonic'
    ssh_crits = np.linspace(np.nanmin(field), np.nanmax(field), 5)
    print('ssh_crits=', ssh_crits)
    print(np.nanmax(field)-np.nanmin(field))
    ssh_crits.sort()
    if cyc == 'anticyclonic':
        ssh_crits = np.flipud(ssh_crits)

    # loop over ssh_crits and remove interior pixels of detected storms from subsequent loop steps
    for ssh_crit in ssh_crits:
        print('ssh_crit', ssh_crit)
        ## 1. Find all regions with eta greater (less than) than ssh_crit for anticyclonic (cyclonic) storms (Chelton et al. 2011, App. B.2, criterion 1)
        if cyc == 'anticyclonic':
            regions, nregions = ndimage.label((field > ssh_crit))#.astype(int))
        elif cyc == 'cyclonic':
            regions, nregions = ndimage.label((field < ssh_crit))#.astype(int))
        #print('regions=', regions)
        print('nregions=', nregions)
        #RuntimeWarning: invalid value encountered in less (juga in greater)
        #regions, nregions = ndimage.label((field < ssh_crit).astype(int))
        
        for iregion in range(nregions):       
            ## 2. Calculate number of pixels comprising detected region, 
            #     reject if not within >= Npix_min
            region = (regions==iregion+1).astype(int)
            region_Npix = region.sum()
            storm_area_within_limits = (region_Npix >= Npix_min)
 
            ## 3. Detect presence of local maximum (minimum) for anticylones (cyclones), 
            #     reject if non-existent
            interior = ndimage.binary_erosion(region)
            #print('interior=',interior )
            exterior = region - interior
            #print('exterior=',exterior)
            #print('interior.sum() == 0 ?', interior.sum())
            
            if interior.sum() == 0: continue
            
            if cyc == 'anticyclonic':
                has_internal_ext = field[interior].max() > field[exterior].max()
            elif cyc == 'cyclonic':
                has_internal_ext = field[interior].min() < field[exterior].min()
 
            ## 4. Find amplitude of region, reject if < amp_thresh
            if cyc == 'anticyclonic':
                amp_abs = field[interior].max()
                amp = amp_abs - field[exterior].mean()
                #print("ac=", amp)
            elif cyc == 'cyclonic':
                amp_abs = field[interior].min()
                amp = field[exterior].mean() - amp_abs
                
            amp_thresh = np.abs(np.diff(ssh_crits)[0])
            #print("amT=", amp_thresh)
            is_tall_storm = amp >= amp_thresh
            print("c=", amp_abs, amp, amp_thresh)
            # Quit loop if these are not satisfied
            if np.logical_not(storm_area_within_limits * has_internal_ext * is_tall_storm):
                continue
                print("quit_loop")
            
            print("ada")
            
            print(storm_area_within_limits, has_internal_ext, is_tall_storm)
            print(storm_area_within_limits * has_internal_ext * is_tall_storm)
            
            # Detected storms:
            if storm_area_within_limits * has_internal_ext * is_tall_storm:
                print('Detected storms................................................')
                # find centre of mass of storm
                storm_object_with_mass = field * region
                storm_object_with_mass[np.isnan(storm_object_with_mass)] = 0
                j_cen, i_cen = ndimage.center_of_mass(storm_object_with_mass)
                lon_cen = np.interp(i_cen, range(0,len(lon)), lon)
                lat_cen = np.interp(j_cen, range(0,len(lat)), lat)
                # Remove storms detected outside global domain (lon < 0, > 360)
                #if globe * (lon_cen >= 0.) * (lon_cen <= 360.):
                #if (lon_cen >= 0.) * (lon_cen <= 360.):
                    # Save storm
                lon_storms = np.append(lon_storms, lon_cen)
                lat_storms = np.append(lat_storms, lat_cen)
                    # assign (and calculated) amplitude, area, and scale of storms
                amp_storms = np.append(amp_storms, amp_abs)
                # remove its interior pixels from further storm detection
                storm_mask = np.ones(field.shape)
                storm_mask[interior.astype(int)==1] = np.nan
                field = field * storm_mask
                #print(lon_storms, lat_storms, amp_storms)
                #print(lon_storms)
    return lon_storms, lat_storms, amp_storms
        
def pr_extreme_detection2(obs_dataset, obs_name, model_datasets, model_names,  names, workdir):
    #adopted from ...
    
    
    i=0
    print(model_names[i])
    
    dsi = xr.DataArray(model_datasets[i].values,
            coords={'time': obs_dataset.times,
            'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
            dims=["time", "lat", "lon"]) 
    
    # Initialisation
    #slp=model_datasets[i].values
    slp=dsi.values
    lon=obs_dataset.lons
    lat=obs_dataset.lats
    #print('lon', lon)
    print('lat', lat)

    lon_storms_a = []
    lat_storms_a = []
    amp_storms_a = []
    
    

    for tt in np.arange(len(obs_dataset.times)):
        #tt=262
        print ('tt=', tt)
        #
        # Detect lon and lat coordinates 
        #anticyclonic
        
        detect_extreme2(slp[tt,:,:], lon, lat, res=2, Npix_min=9,
                                cyc='anticyclonic', globe=False)
       
       
       

def detect_extreme2(field, lon, lat, res, Npix_min, cyc, globe=False):
    #print('def detect_storms ...')
    import scipy.ndimage as ndimage
  
  
    llon, llat = np.meshgrid(lon, lat)

    lon_storms = np.array([])
    lat_storms = np.array([])
    amp_storms = np.array([])

    # ssh_crits is an array of ssh levels over which to perform storm detection loop
    # ssh_crits increasing for 'cyclonic', decreasing for 'anticyclonic'
    ssh_crits = np.linspace(np.nanmin(field), np.nanmax(field), 5)
    
 
    ssh_crits.sort()
    ssh_crits = np.flipud(ssh_crits)
    #print('ssh_crits=', ssh_crits)
    
    #ssh_crits=ssh_crits[0:4]
    
    # loop over ssh_crits and remove interior pixels of detected storms from subsequent loop steps
    for ssh_crit in ssh_crits:
        #print('ssh_crit', ssh_crit)
        
        regions, nregions = ndimage.label((field >= ssh_crit))#.astype(int))
        #print(field)
        #print('nregions=', nregions)
       
        
        for iregion in range(nregions):       
            ## 2. Calculate number of pixels comprising detected region, 
            #     reject if not within >= Npix_min
            region = (regions==iregion+1).astype(int)
            #print(region)
            if region.sum()>=70 and ssh_crit >=5:
                print('ssh_crit', ssh_crit)
                
                print('region.sum', region.sum())
                #region_Npix = region.sum()
                #storm_area_within_limits = (region_Npix >= Npix_min) 
                #print('storm_area_within_limits',storm_area_within_limits)
            #else:
            #    print ('tidak ada') 
            
    
    return 

def detect_extremeX(field, lon, lat, res, Npix_min, cyc, globe=False):
    print('def detect_storms ...')
    import scipy.ndimage as ndimage
    '''
    Detect storms present in field which satisfy the criteria.
    Algorithm is an adaptation of an eddy detection algorithm,
    outlined in Chelton et al., Prog. ocean., 2011, App. B.2,
    with modifications needed for storm detection.

    field is a 2D array specified on grid defined by lat and lon.

    res is the horizontal grid resolution in degrees of field

    Npix_min is the minimum number of pixels within which an
    extremum of field must lie (recommended: 9).

    cyc = 'cyclonic' or 'anticyclonic' specifies type of system
    to be detected (cyclonic storm or high-pressure systems)

    globe is an option to detect storms on a globe, i.e. with periodic
    boundaries in the West/East. Note that if using this option the 
    supplied longitudes must be positive only (i.e. 0..360 not -180..+180).

    Function outputs lon, lat coordinates of detected storms
    '''

  
    llon, llat = np.meshgrid(lon, lat)

    lon_storms = np.array([])
    lat_storms = np.array([])
    amp_storms = np.array([])

    # ssh_crits is an array of ssh levels over which to perform storm detection loop
    # ssh_crits increasing for 'cyclonic', decreasing for 'anticyclonic'
    ssh_crits = np.linspace(np.nanmin(field), np.nanmax(field), 5)
    print('ssh_crits=', ssh_crits)
    print(np.nanmax(field)-np.nanmin(field))
    ssh_crits.sort()
    if cyc == 'anticyclonic':
        ssh_crits = np.flipud(ssh_crits)

    # loop over ssh_crits and remove interior pixels of detected storms from subsequent loop steps
    for ssh_crit in ssh_crits:
        print('ssh_crit', ssh_crit)
        ## 1. Find all regions with eta greater (less than) than ssh_crit for anticyclonic (cyclonic) storms (Chelton et al. 2011, App. B.2, criterion 1)
        if cyc == 'anticyclonic':
            regions, nregions = ndimage.label((field > ssh_crit))#.astype(int))
        elif cyc == 'cyclonic':
            regions, nregions = ndimage.label((field < ssh_crit))#.astype(int))
        #print('regions=', regions)
        print('nregions=', nregions)
        #RuntimeWarning: invalid value encountered in less (juga in greater)
        #regions, nregions = ndimage.label((field < ssh_crit).astype(int))
        
        for iregion in range(nregions):       
            ## 2. Calculate number of pixels comprising detected region, 
            #     reject if not within >= Npix_min
            region = (regions==iregion+1).astype(int)
            region_Npix = region.sum()
            storm_area_within_limits = (region_Npix >= Npix_min)
 
            ## 3. Detect presence of local maximum (minimum) for anticylones (cyclones), 
            #     reject if non-existent
            interior = ndimage.binary_erosion(region)
            #print('interior=',interior )
            exterior = region - interior
            #print('exterior=',exterior)
            #print('interior.sum() == 0 ?', interior.sum())
            
            if interior.sum() == 0: continue
            
            if cyc == 'anticyclonic':
                has_internal_ext = field[interior].max() > field[exterior].max()
            elif cyc == 'cyclonic':
                has_internal_ext = field[interior].min() < field[exterior].min()
 
            ## 4. Find amplitude of region, reject if < amp_thresh
            if cyc == 'anticyclonic':
                amp_abs = field[interior].max()
                amp = amp_abs - field[exterior].mean()
                #print("ac=", amp)
            elif cyc == 'cyclonic':
                amp_abs = field[interior].min()
                amp = field[exterior].mean() - amp_abs
                
            amp_thresh = np.abs(np.diff(ssh_crits)[0])
            #print("amT=", amp_thresh)
            is_tall_storm = amp >= amp_thresh
            print("c=", amp_abs, amp, amp_thresh)
            # Quit loop if these are not satisfied
            if np.logical_not(storm_area_within_limits * has_internal_ext * is_tall_storm):
                continue
                print("quit_loop")
            
            print("ada")
            
            print(storm_area_within_limits, has_internal_ext, is_tall_storm)
            print(storm_area_within_limits * has_internal_ext * is_tall_storm)
            
            # Detected storms:
            if storm_area_within_limits * has_internal_ext * is_tall_storm:
                print('Detected storms................................................')
                # find centre of mass of storm
                storm_object_with_mass = field * region
                storm_object_with_mass[np.isnan(storm_object_with_mass)] = 0
                j_cen, i_cen = ndimage.center_of_mass(storm_object_with_mass)
                lon_cen = np.interp(i_cen, range(0,len(lon)), lon)
                lat_cen = np.interp(j_cen, range(0,len(lat)), lat)
                # Remove storms detected outside global domain (lon < 0, > 360)
                #if globe * (lon_cen >= 0.) * (lon_cen <= 360.):
                #if (lon_cen >= 0.) * (lon_cen <= 360.):
                
                # Save storm
                lon_storms = np.append(lon_storms, lon_cen)
                lat_storms = np.append(lat_storms, lat_cen)
                
                # assign (and calculated) amplitude, area, and scale of storms
                amp_storms = np.append(amp_storms, amp_abs)
                # remove its interior pixels from further storm detection
                storm_mask = np.ones(field.shape)
                storm_mask[interior.astype(int)==1] = np.nan
                field = field * storm_mask
                #print(lon_storms, lat_storms, amp_storms)
                #print(lon_storms)
    return lon_storms, lat_storms, amp_storms

def pr_max_map2(obs_dataset, obs_name, model_datasets, model_names,  names, workdir):
    tt=[]
    fig, ax = plt.subplots(nrows=1, ncols=1 ,figsize=(6,5))
    #for i in np.arange(len(model_datasets)):
    i=9
    print(model_names[i])
    #if i==3: 
    dsi = xr.DataArray(model_datasets[i].values,
    coords={'time': obs_dataset.times,
    'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
    dims=["time", "lat", "lon"]) 
   
   #d=xr.where(dsi.values >275,dsi.values,0)
    #max=dsi.max().values
    max=200
    print('max=',max)
    print('mean=',dsi.mean().values)
    t, lat_i, lon_i = np.nonzero(xr.where(dsi>=max, 1, 0).data)
    lats = dsi.isel(lat=lat_i).data
    lons = dsi.isel(lon=lon_i).data
    #print(lats,lons)
    time = dsi.time.isel(time=t).data
    #pairs = list(zip(lats, lons))
    print(lons)
    print('t=',t,time)
    #print('mean_t=',dsi[t,:,:].mean().values)
    

    lat_min = dsi.lat.min()
    lat_max = dsi.lat.max()
    lon_min = dsi.lon.min()
    lon_max = dsi.lon.max()
    
    
    #t=np.array(t)
    
    print('t=',t) 
    print('len(t)=',len(t))    
    x,y = np.meshgrid(dsi.lon, dsi.lat)
    map2=ma.zeros((len(dsi.time),len(dsi.lat),len(dsi.lon)))
   
    for it in t:
        print(it)
        map2[it,:,:]=dsi[it,:,:] #.mean(dim='time')
    
    #print(map2.shape)  
    map2=map2.sum(axis=0)
    
    levels=np.arange(5)
    #norm = plt.Normalize(vmin=300, vmax=500)
    cmaps = ['RdBu_r', 'viridis']
   
    m = Basemap(ax=ax, projection ='cyl', llcrnrlat = lat_min+1*.22, urcrnrlat = lat_max-3*.22,
    llcrnrlon = lon_min+7*.22, urcrnrlon = lon_max, resolution = 'l', fix_aspect=False)
    m.drawcoastlines(linewidth=1)
    m.drawcountries(linewidth=1)
    m.drawstates(linewidth=0.5, color='w')
   
    #print(map.shape)
    #max1 = ax[n,0].contourf(x,y,dsi[t,:,:].mean(dim='time'))
    pcm = ax.contourf(x,y,map2)#,norm=norm,cmap=cmaps[0])
    ax.set_title(model_names[i],fontsize=8)
    #ax[n,0].set_ylabel(musim)
    #ax[0,0].set_yticks([-10,0,10,20])
    #ax[0,0].set_xticks([90,100,110,120,130,140])
   
    cax = fig.add_axes([0.92, 0.5, 0.02, 0.35])
    cax.tick_params(labelsize=6)
    fig.colorbar(pcm, cax = cax) 
    #ax.clabel(pcm, inline=True, fontsize=10)
    plt.subplots_adjust(hspace=.05,wspace=.05)
    
    #file_name='cek_max_map_SEA_new_HadGEM_d_3'
    file_name='cek_max_map_'+reg
    
    fig.savefig(workdir+file_name,dpi=300)
    plt.show()

def pr_max_map3(obs_dataset, obs_name, model_datasets, model_names,  names, workdir):
    tt=[]
    
    #for i in np.arange(len(model_datasets)):
    i=9 # ECE i=2 IPSL=3 GFDL=9
    print(model_names[i])
    #if i==3: 
    dsi = xr.DataArray(model_datasets[i].values,
    coords={'time': obs_dataset.times,
    'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
    dims=["time", "lat", "lon"]) 
   
   #d=xr.where(dsi.values >275,dsi.values,0)
    #----set sesuai kondisi
    max=dsi.max().values
    print('max=',max)
    max=max-max/3
    #max=200
    #max=[,200]
    print('max=',max)
    #print('mean=',dsi.mean().values)
    
    t, lat_i, lon_i = np.nonzero(xr.where(dsi>=max, 1, 0).data)
    lats = dsi.lat.isel(lat=lat_i).data
    lons = dsi.lon.isel(lon=lon_i).data
    #print
    time = dsi.time.isel(time=t).data
    pairs = list(zip(lons,lats))
    print(pairs)
    print('t=',t)
    print('time=',time)
    #from datetime import datetime
    #datetime(time.isoformat())

    #datetime.strptime(str(time),'%Y-%m-%d %H:%M:%S')
    
    #time.astype("datetime64[ns]")

    #print('time=',time)
    #print('mean_t=',dsi[t,:,:].mean().values)
    

    lat_min = dsi.lat.min()
    lat_max = dsi.lat.max()
    lon_min = dsi.lon.min()
    lon_max = dsi.lon.max()
    
    
    #t=np.array(t)
    
    print('t=',t) 
    print('len(t)=',len(t))    
    x,y = np.meshgrid(dsi.lon, dsi.lat)
    #map2=ma.zeros((len(dsi.time),len(dsi.lat),len(dsi.lon)))
    
    levels=np.arange(5)
    #norm = plt.Normalize(vmin=300, vmax=500)
    cmaps = ['RdBu_r', 'viridis']
    #fig, ax = plt.subplots(nrows=1, ncols=len(t) ,figsize=(6,5))
    '''
    n=0
    for it in t:
        print(it)
        map2=dsi[it,:,:] #.mean(dim='time')
        
        #print(map2.shape)  
        #map2=map2.sum(axis=0)
        mak=map2.max()
        m = Basemap(ax=ax[n], projection ='cyl', llcrnrlat = lat_min+1*.22, urcrnrlat = lat_max-3*.22,
        llcrnrlon = lon_min+7*.22, urcrnrlon = lon_max, resolution = 'l', fix_aspect=False)
        m.drawcoastlines(linewidth=1)
        m.drawcountries(linewidth=1)
        m.drawstates(linewidth=0.5, color='w')
       
        #print(map.shape)
        #max1 = ax[n,0].contourf(x,y,dsi[t,:,:].mean(dim='time'))
        pcm = ax[n].contourf(x,y,map2)#,norm=norm,cmap=cmaps[0])
        #ax[n].set_title('max=('+'%.2f'%mak+')',fontsize=8)
        ax[n].set_title('t='+str(it),fontsize=8)
        #ax[n+1].set_title('('+'%.2f'%mak+')',fontsize=8)
        #ax[n].set_xlabel('t='+str(it))
        #ax[0,0].set_yticks([-10,0,10,20])
        #ax[0,0].set_xticks([90,100,110,120,130,140])
        ax[n].set_xticks([95,100,105])
        ax[n].set_xticklabels(['95E','100E','105E'], fontsize=7)
        ax[n].annotate('max('+'%.2f'%mak+')',xy=pairs[n],xytext=(99,-5),
             arrowprops={"width":1,"headwidth":5,'headlength':7},
             horizontalalignment='center',fontsize=7)
        n=n+1
    ax[0].set_yticks([-5,0,5])
    ax[0].set_yticklabels(['5S','EQ','5N'], fontsize=7)
    cax = fig.add_axes([0.92, 0.5, 0.02, 0.35])
    cax.tick_params(labelsize=6)
    fig.colorbar(pcm, cax = cax) 
    #ax.clabel(pcm, inline=True, fontsize=10)
    plt.subplots_adjust(hspace=.05,wspace=.05)
    
    #file_name='cek_max_map_SEA_new_HadGEM_d_3'
    file_name=model_names[i]+'_max_map_'+reg
    plt.show()
    fig.savefig(workdir+file_name,dpi=300)
    '''
    n=0
    fig, ax = plt.subplots(nrows=1, ncols=len(t) ,figsize=(6,8))
    for it in t:
        print(it)
        map2=dsi[it,:,:] #.mean(dim='time')
        
        #print(map2.shape)  
        #map2=map2.sum(axis=0)
        mak=map2.max()
        m = Basemap(ax=ax[n], projection ='cyl', llcrnrlat = lat_min+1*.22, urcrnrlat = lat_max-3*.22,
        llcrnrlon = lon_min+7*.22, urcrnrlon = lon_max, resolution = 'l', fix_aspect=True)
        m.drawcoastlines(linewidth=1)
        m.drawcountries(linewidth=1)
        m.drawstates(linewidth=0.5, color='w')
       
        #print(map.shape)
        #max1 = ax[n,0].contourf(x,y,dsi[t,:,:].mean(dim='time'))
        pcm = ax[n].contour(x,y,map2)#,norm=norm,cmap=cmaps[0])
        #ax[n].set_title('max=('+'%.2f'%mak+')',fontsize=8)
        ax[n].set_title('t='+str(it),fontsize=8)
        #ax[n+1].set_title('('+'%.2f'%mak+')',fontsize=8)
        #ax[n].set_xlabel('t='+str(it))
        #ax[0,0].set_yticks([-10,0,10,20])
        #ax[0,0].set_xticks([90,100,110,120,130,140])
        ax[n].set_xticks([95,100,105])
        ax[n].set_xticklabels(['95E','100E','105E'], fontsize=7)
        ax[n].annotate('max='+'%.2f'%mak,xy=pairs[n],xytext=(99,-5),
             arrowprops={"width":1,"headwidth":5,'headlength':7},
             horizontalalignment='center',fontsize=5)
        n=n+1
    ax[0].set_yticks([-5,0,5])
    ax[0].set_yticklabels(['5S','EQ','5N'], fontsize=7)
    #cax = fig.add_axes([0.92, 0.5, 0.02, 0.35])
    #cax.tick_params(labelsize=6)
    #fig.colorbar(pcm, cax = cax) 
    
    plt.subplots_adjust(hspace=.05,wspace=.05)
    
    #file_name='cek_max_map_SEA_new_HadGEM_d_3'
    file_name=model_names[i]+'_max_contour_3'+reg
    plt.show()
    fig.savefig(workdir+file_name,dpi=300)
    
def pr_max_map4(obs_dataset, obs_name, model_datasets, model_names,  names, workdir):
    
    i=7 # ECE i=2 IPSL=3 GFDL=9
    
    print(model_names[i])
    #if i==3: 
    dsi = xr.DataArray(model_datasets[i].values,
    coords={'time': obs_dataset.times,
    'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
    dims=["time", "lat", "lon"]) 
   
   #d=xr.where(dsi.values >275,dsi.values,0)
    #----set sesuai kondisi
    max=dsi.max().values
    print('max=',max)
    max=max-max/5
    #max=200
    #max=[,200]
    print('max=',max)
    #print('mean=',dsi.mean().values)
    
    t, lat_i, lon_i = np.nonzero(xr.where(dsi>=max, 1, 0).data)
    lats = dsi.lat.isel(lat=lat_i).data
    lons = dsi.lon.isel(lon=lon_i).data
    #print
    time = dsi.time.isel(time=t).data
    pairs = list(zip(lons,lats))
    print(pairs)
    print('t=',t)
    print('time=',time)
    #from datetime import datetime
    #datetime(time.isoformat())

    #datetime.strptime(str(time),'%Y-%m-%d %H:%M:%S')
    
    #time.astype("datetime64[ns]")

    #print('time=',time)
    #print('mean_t=',dsi[t,:,:].mean().values)
    

    lat_min = dsi.lat.min()
    lat_max = dsi.lat.max()
    lon_min = dsi.lon.min()
    lon_max = dsi.lon.max()
    
    
    #t=np.array(t)
    
    print('t=',t) 
    print('len(t)=',len(t))    
    x,y = np.meshgrid(dsi.lon, dsi.lat)
    #map2=ma.zeros((len(dsi.time),len(dsi.lat),len(dsi.lon)))
    
    levels=np.arange(5)
    #norm = plt.Normalize(vmin=300, vmax=500)
    cmaps = ['RdBu_r', 'viridis']
   
    n=0
    fig, ax = plt.subplots(nrows=1, ncols=len(t) ,figsize=(6,5))
    #plt.subplots_adjust(left=.2)
    
    for it in t:
        print(it)
        map2=dsi[it,:,:] #.mean(dim='time')
        
        #print(map2.shape)  
        #map2=map2.sum(axis=0)
        mak=map2.max()
        m = Basemap(ax=ax[n], projection ='cyl', llcrnrlat = lat_min+1*.22, urcrnrlat = lat_max-3*.22,
        llcrnrlon = lon_min+7*.22, urcrnrlon = lon_max, resolution = 'l', fix_aspect=True)
        m.drawcoastlines(linewidth=1)
        m.drawcountries(linewidth=1)
        m.drawstates(linewidth=0.5, color='w')
       
        #print(map.shape)
        #max1 = ax[n,0].contourf(x,y,dsi[t,:,:].mean(dim='time'))
        pcm = ax[n].contour(x,y,map2)#,norm=norm,cmap=cmaps[0])
        #ax[n].set_title('max=('+'%.2f'%mak+')',fontsize=8)

        ax[n].set_title('t='+str(it),fontsize=8)
        #ax[n+1].set_title('('+'%.2f'%mak+')',fontsize=8)
        #ax[n].set_xlabel('t='+str(it))
        #ax[0,0].set_yticks([-10,0,10,20])
        #ax[0,0].set_xticks([90,100,110,120,130,140])
        ax[0].set_xticks([95,100,105])
        ax[0].set_xticklabels(['95E','100E','105E'], fontsize=7)
        ax[n].annotate('max='+'%.2f'%mak,xy=pairs[n],xytext=(99,-5),
             arrowprops={"width":1,"headwidth":5,'headlength':7},
             horizontalalignment='center',fontsize=5)
        #if n==0: plt.ylabel(model_names[i], fontsize=7)
        n=n+1
    ax[0].set_yticks([-5,0,5])
    ax[0].set_yticklabels(['5S','EQ','5N'], fontsize=7)
   
    #plt.ylabel(model_names[i], fontsize=7)
    
    plt.subplots_adjust(hspace=.1,wspace=.05)
    
    #file_name='cek_max_map_SEA_new_HadGEM_d_3'
    file_name=model_names[i]+'_max_contour_3'+reg
    plt.show()
    fig.savefig(workdir+file_name,dpi=300)
    

def pr_hist(obs_dataset, obs_name, model_datasets, model_names,  names, workdir):
    fig, ax = plt.subplots(nrows=1, ncols= len(model_datasets)+1,figsize=(6,5))
    
    ds = xr.DataArray(obs_dataset.values,
    coords={'time': obs_dataset.times,
    'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
    dims=["time", "lat", "lon"])
    mak=ds.max().data
    ds.plot.hist(ax=ax[0])
    #untuk models
    #ax[0].set_ylim(0, 300000)
    
    #untuk 5obs
    ax[0].set_ylim(0, 200000)
    
    #ax[0].ticklabel_format(axis='y', style='sci', scilimits=(4,4))
    ax[0].set_title(obs_name, fontsize=7)
    
    #ax[0].text(10, 180000, s='max='+'%.2f'%mak, fontsize=6)
    ax[0].set_xlabel('max='+'%.2f'%mak, fontsize=7) 
    #ax[0].text(17, 13000, s='max='+'%.2f'%mak, fontsize=7)
    
    
    
    ax[0].yaxis.set_tick_params(labelsize=7)
    ax[0].xaxis.set_tick_params(labelsize=7)
    
    for i in np.arange(len(model_datasets)):
        
        print(model_names[i])
        #if i==3: 
        dsi = xr.DataArray(model_datasets[i].values,
        coords={'time': obs_dataset.times,
        'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
        dims=["time", "lat", "lon"]) 
        mak=dsi.max().data
        print(mak)
        dsi.plot.hist(ax=ax[i+1])
        #ax[i+1].set_ylim(0, 300000)
        #untuk 5obs
        ax[i+1].set_ylim(0, 200000)
        #ax[i].ticklabel_format(axis='y', style='sci', scilimits=(4,4))
        ax[i+1].set_title(model_names[i], fontsize=7)
        #ax[i+1].annotate('max='+'%.2f'%mak, fontsize=5, xy=(20,180000), backgroundcolor='0.95',alpha=1)
        
        #ax[i+1].text(10, 180000, s='max='+'%.2f'%mak, fontsize=6)
        #ax[i+1].text(17, 13000, s='max='+'%.2f'%mak, fontsize=7)
        
        ax[i+1].set_yticks([])         
        ax[i+1].set_xlabel('max='+'%.2f'%mak, fontsize=7)
    
        
        
        ax[i+1].xaxis.set_tick_params(labelsize=7)       
    #plt.ticklabel_format(axis='y', style='sci', scilimits=(4,4))
    ax[0].set_ylabel('Number of occurance')
    #plt.xlabel('Precipitation intensity(mm)')
    file_name=reg+'_hist_5obs'
    plt.show()
    fig.savefig(workdir+file_name,dpi=300)
    
    
def pr_hist2(obs_dataset, obs_name, model_datasets, model_names,  names, workdir):
    fig, ax = plt.subplots(nrows=1, ncols= len(model_datasets)+1,figsize=(6,5))
    
    ds = xr.DataArray(obs_dataset.values,
    coords={'time': obs_dataset.times,
    'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
    dims=["time", "lat", "lon"])
    mak=ds.max().data
    ds.plot.hist(ax=ax[0])
    ax[0].set_ylim(0, 300000)
    ax[0].set_xlim(0, 50)
    #ax[0].ticklabel_format(axis='y', style='sci', scilimits=(4,4))
    ax[0].set_title(obs_name, fontsize=7)
    #ax[0].annotate('max='+'%.2f'%mak, fontsize=5, xy=(0,180000), backgroundcolor='0.95',alpha=1)
    #ax[0].text(10, 180000, s='max='+'%.2f'%mak, fontsize=6)
    ax[0].set_xlabel('max='+'%.2f'%mak, fontsize=7) 
    ax[0].yaxis.set_tick_params(labelsize=7)
    ax[0].xaxis.set_tick_params(labelsize=7)
    
    for i in np.arange(len(model_datasets)):
        
        print(model_names[i])
        #if i==3: 
        dsi = xr.DataArray(model_datasets[i].values,
        coords={'time': obs_dataset.times,
        'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
        dims=["time", "lat", "lon"]) 
        mak=dsi.max().data
        print(mak)
        dsi.plot.hist(ax=ax[i+1])
        ax[i+1].set_ylim(0, 300000)
        ax[i+1].set_xlim(0, 50)
        #ax[i].ticklabel_format(axis='y', style='sci', scilimits=(4,4))
        ax[i+1].set_title(model_names[i], fontsize=7)
        #ax[i+1].annotate('max='+'%.2f'%mak, fontsize=5, xy=(20,180000), backgroundcolor='0.95',alpha=1)
        #ax[i+1].text(10, 180000, s='max='+'%.2f'%mak, fontsize=6)
        ax[i+1].set_yticks([])         
        ax[i+1].set_xlabel('max='+'%.2f'%mak, fontsize=7)
        ax[i+1].xaxis.set_tick_params(labelsize=7)       
    #plt.ticklabel_format(axis='y', style='sci', scilimits=(4,4))
    ax[0].set_ylabel('Number of occurance')
    #plt.xlabel('Precipitation intensity(mm)')
    file_name=reg+'_hist'
    plt.show()
    fig.savefig(workdir+file_name,dpi=300)
    

def pr_hist_sea(obs_dataset, obs_name, model_datasets, model_names,  names, workdir):
    fig, ax = plt.subplots(nrows=1, ncols= 9,figsize=(6,5))
    
       
    #for i in np.arange(len(model_datasets)):
    for i in [1,2,3,4,5,6,7,8,9]:
        
        print(model_names[i])
        #if i==3: 
        dsi = xr.DataArray(model_datasets[i].values,
        coords={'time': obs_dataset.times,
        'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
        dims=["time", "lat", "lon"]) 
        mak=dsi.max().data
        print(mak)
        dsi.plot.hist(ax=ax[i-1])
        ax[i-1].set_ylim(0, 3000)
        ax[i-1].set_xlim(100,300)
        #ax[i].ticklabel_format(axis='y', style='sci', scilimits=(4,4))
        ax[i-1].set_title(model_names[i], fontsize=7)
        #ax[i+1].annotate('max='+'%.2f'%mak, fontsize=5, xy=(20,180000), backgroundcolor='0.95',alpha=1)
        #ax[i+1].text(10, 180000, s='max='+'%.2f'%mak, fontsize=6)
        if i<9: 
            ax[i].set_yticks([])         
            ax[i].set_xticklabels(['','---','---'])
        ax[i-1].set_xlabel('max='+'%.2f'%mak, fontsize=7)
        ax[i-1].xaxis.set_tick_params(labelsize=7)       
    #plt.ticklabel_format(axis='y', style='sci', scilimits=(4,4))
    ax[0].set_ylabel('Number of occurance')
    #plt.xlabel('Precipitation intensity(mm)')
    file_name=reg+'_hist'
    plt.show()
    fig.savefig(workdir+file_name,dpi=300)

def xpr_max_cek(obs_dataset, obs_name, model_datasets, model_names,  names, workdir):
  
    for i in np.arange(len(model_datasets)):
        if i==9:
            print(model_names[i])
            #if i==3: 
            dsi = xr.DataArray(model_datasets[i].values,
            coords={'time': obs_dataset.times,
            'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
            dims=["time", "lat", "lon"]) 
           
           #d=xr.where(dsi.values >275,dsi.values,0)
            max=dsi.max().values
            print('max=',max)
            t, lat_i, lon_i = np.nonzero(xr.where(dsi>=max, 1, 0).data)
            # lats = dsi.isel(lat=lat_i).data
            # lons = dsi.isel(lon=lon_i).data
            time = dsi.time.isel(time=t).data
            # pairs = list(zip(time,lats, lons))
            print(time)
            print('t=',t)
           
            lat_min = dsi.lat.min().values
            lat_max = dsi.lat.max().values
            lon_min = dsi.lon.min().values
            lon_max = dsi.lon.max().values
            
            #print('SEA RCMES: -15.14, 27.26, 89.26, 146.96')
            print('SEA data ini:',lat_min, lat_max, lon_min, lon_max)
            
            fig, ax = plt.subplots(nrows=1, ncols=3 ,figsize=(6,5))
    
            x,y = np.meshgrid(dsi.lon, dsi.lat)
            #print(x,y)
            n=0
            map1=dsi[t-1,:,:].mean(dim='time')
            map2=dsi[t,:,:].mean(dim='time')
            map3=dsi[t+1,:,:].mean(dim='time')
            map=[map1,map3,map2]
            
            for mapi in map:
                m = Basemap(ax=ax[n], projection ='cyl', llcrnrlat = lat_min+4*.22, urcrnrlat = lat_max,
                llcrnrlon = lon_min+4*.22, urcrnrlon = lon_max, resolution = 'l', fix_aspect=False)
                m.drawcoastlines(linewidth=1)
                m.drawcountries(linewidth=1)
                m.drawstates(linewidth=0.5, color='w')
               
                #print(map.shape)
                #max1 = ax[n,0].contourf(x,y,dsi[t,:,:].mean(dim='time'))
                max1 = m.contourf(x,y,mapi)
                ax[n].set_title(model_names[i],fontsize=8)
                #ax[n,0].set_ylabel(musim)
                #ax[0,0].set_yticks([-10,0,10,20])
                #ax[0,0].set_xticks([90,100,110,120,130,140])
                n=n+1
    cax = fig.add_axes([0.91, 0.5, 0.02, 0.35])
    cax.tick_params(labelsize=6)
    plt.colorbar(max1, cax = cax) 
    plt.subplots_adjust(hspace=.05,wspace=.05)
    
    file_name='cek_max_map_SEA2'
    fig.savefig(workdir+file_name,dpi=300)
    plt.show()

def pr_max_cek(obs_dataset, obs_name, model_datasets, model_names,  names, workdir):
  
    #next plot pr mean max min untuk cek hal aneh pada grafik lat_mean
    #boxplot dan mean tidak perlu disini 
    import math
   
    fig,ax=plt.subplots(figsize=[8,6])
    plt.subplots_adjust(bottom=.2)
    #fig=plt.figure()
    #waktu nya lama
    #data=obs_dataset.values.flatten()
    #fdata = data[~np.isnan(data)]
    #data = [value for value in data if not math.isnan(value)]
    #ax=fig.add_axes([0,0,1,1])
    #ax.boxplot(data)
    
    #plt.show()
    #exit()
    #plt.figure()
    #max=np.max(obs_dataset.values)
    #mean=np.nanmedian(obs_dataset.values)
    #print(mean)
    #plt.scatter(1,max, label = 'GPCP',marker='o',color='black')
    #plt.scatter(1,mean, label = 'GPCP',marker='o',color='black')
   
    #tebal garis
    #lws=[2,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,4]
    #t=[]
    for i in np.arange(len(model_datasets)):
        print(model_names[i])
        #if i==3: 
        dsi = xr.DataArray(model_datasets[i].values,
        coords={'time': obs_dataset.times,
        'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
        dims=["time", "lat", "lon"])  
        
        print('min_val=', np.min(model_datasets[i].values))
      
        max=np.max(model_datasets[i].values)
        #mean=np.nanmedian(model_datasets[i].values)
        #print(max)
        ax.scatter(i+1,max, label = model_names[i],marker='o',color='black')
        #plt.scatter(i+1,mean, label = model_names[i],marker='o',color='black')
    #plt.legend(bbox_to_anchor=(1, .8), loc='best', prop={'size':7}, frameon=False) 
    #plt.ylabel('Max and Mean rainfall (mm/month)')
        #data=model_datasets[i].values.flatten()
    #fdata = data[~np.isnan(data)]
        #data = [value for value in data if not math.isnan(value)]
        #t.append(data)
    #ax.boxplot(t)
    ax.set_xticks(np.arange(len(model_datasets)+1))
    names[0]=''
    ax.set_xticklabels(names,fontsize=8.5)
    plt.xticks(rotation=45)
    plt.ylabel('Max rainfall (mm/month)')
    #plt.title('Max rainfall: Sumatera/CORDEX-SEA/res.25km/bias-corrected')
    #plt.title('Max rainfall: SEA/CORDEX-SEA/res.25km/unbias-corrected')
        
    file_name='_pr_max'
    plt.savefig(workdir+reg+file_name,dpi=300,bbox_inches='tight')
    plt.show()
    
def pr_min_cek(obs_dataset, obs_name, model_datasets, model_names,  names, workdir):
   
    min=[]
    max=[]
    for i in np.arange(len(model_datasets)):
        print(model_names[i])
       
        dsi = xr.DataArray(model_datasets[i].values)
     
        
        min1=np.min(model_datasets[i].values)
        print('min_val=', min1)
        min.append(min1)
        
        max1=np.max(model_datasets[i].values)
        print('max_val=', max1)
        max.append(max1)
        
    import pandas as pd
    df = pd.DataFrame([model_names, min, max])
    
    df.T.to_excel(workdir+reg+'min_max_cek.xlsx', index=False, header=False) 
    

def lat_mean_rainfall2(obs_dataset, obs_name, model_datasets, model_names, workdir):
   
    ds = xr.DataArray(obs_dataset.values,
    coords={'time': obs_dataset.times,
    'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
    dims=["time", "lat", "lon"])
        
    #ds = xr.DataArray(obs_dataset)
    #ds = ds.groupby('time.month').mean() 
    #print('xx=',ds.shape, ds.lat.shape)
      
    ds0= ds.mean(dim=('time','lat'))
    
    fig, ax = plt.subplots(nrows=1, ncols=2,figsize=(6,6))
    
    
    ax[0].plot(ds0.lon, ds0.values, color='black', lw=2, label = obs_name)
    #tebal garis
    #lws=[2,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,4]
       
    for i in np.arange(len(model_datasets)):
        #i=10
        print(model_names[i])
        dsi = xr.DataArray(model_datasets[i].values,
        coords={'time': obs_dataset.times,
        'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
        dims=["time", "lat", "lon"])  
                
        dsi= dsi.mean(dim=('time','lat'))
    
        #plot zm hujan mean() SEA
        #plt.figure(figsize=[12,7])
        
        ax[0].plot(dsi.lon, dsi.values, lw=1, label = model_names[i])
    #plt.legend(loc='best', prop={'size':8}, frameon=False) 
    ax[0].set_ylabel('Mean rainfall (mm/day)')
    ax[0].set_xlabel('Longitude (E)')
    
    file_name='lat mean_old_sizeL8'
    plt.savefig(workdir+file_name,dpi=300,bbox_inches='tight')
 
    #pakai xr
    ds = xr.DataArray(obs_dataset.values,
    coords={'time': obs_dataset.times,
    'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
    dims=["time", "lat", "lon"])
        
    #ds = xr.DataArray(obs_dataset)
    #ds = ds.groupby('time.month').mean() 
    #print('xx=',ds.shape, ds.lat.shape)
      
    ds0= ds.mean(dim=('time','lon'))
   
    ax[1].plot(ds0.values, ds0.lat, color='black', lw=2, label = obs_name)
    #tebal garis
    #lws=[2,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,4]
       
    for i in np.arange(len(model_datasets)):
        #i=10
        #print(model_names[i])
        dsi = xr.DataArray(model_datasets[i].values,
        coords={'time': obs_dataset.times,
        'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
        dims=["time", "lat", "lon"])  
        
        dsi= dsi.mean(dim=('time','lon'))
    
        #plot zm hujan mean() SEA
        #plt.figure(figsize=[12,7])
        ax[1].plot(dsi.values, dsi.lat, lw=1, label = model_names[i])
    #plt.legend(bbox_to_anchor=(1, .7), loc='best', prop={'size':8}, frameon=False) 
    plt.legend(loc='best', prop={'size':8}, frameon=False) 
    ax[1].set_xlabel('Mean rainfall (mm/day)')
    ax[1].set_ylabel('Latitude')
    
    plt.subplots_adjust(bottom=.3)
    
    file_name='lon mean_'+reg
    plt.savefig(workdir+file_name,dpi=300,bbox_inches='tight')
    plt.show()
    
    return


def cv_year(names, obs_dataset, obs_name, model_datasets, model_names, workdir):
    #coefficient of variation (CV), 
    #also known as relative standard deviation (RSD)
    #It is often expressed as a percentage, and is defined as 
    #the ratio of the standard deviation to the mean 
    ds = xr.DataArray(obs_dataset.values,
    coords={'time': obs_dataset.times,
    'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
    dims=["time", "lat", "lon"])
    ds = ds.groupby('time.year').sum()
    #print(ds.mean(dim))
    #exit()
    ds=ds.mean(dim=("lat", "lon"))
    
    sd0=metrics.calc_stddev(ds.values, axis=0)
    print(sd0)
    m0=utils.calc_temporal_mean(ds).mean()
    print(m0)
    plt.scatter(sd0/m0*100,m0, label = 'GPCP',marker='$%d$' % 1,color='black')
    #plt.scatter(sdi/m*100,m, label = model_names[i], marker='$%d$' % (i + 2), color='black')
    for i in np.arange(len(model_datasets)):
        dsi = xr.DataArray(model_datasets[i].values,
                coords={'time': obs_dataset.times,
                'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
                dims=["time", "lat", "lon"]) 
        dsi = dsi.groupby('time.year').sum()
        dsi=dsi.mean(dim=("lat", "lon"))
        sdi=metrics.calc_stddev(dsi.values, axis=0)
        m=utils.calc_temporal_mean(dsi).mean()
        plt.scatter(sdi/m*100,m, label = model_names[i], 
                    marker='$%d$' % (i + 2),s=40, color='black')
       
    plt.ylabel('Mean rainfall (mm/year)')
    plt.xlabel('Coefficient of variation (%)')
    #plt.ylim(40,110)
    plt.legend(bbox_to_anchor=(0.99, .6), loc='best', prop={'size':8.5}, frameon=True, handletextpad=0) 
       
    figname='cv_year_'+reg
    plt.savefig(workdir+figname, dpi=300, bbox_inches='tight')
    plt.show()


def cv_mon(names, obs_dataset, obs_name, model_datasets, model_names, workdir):
    #coefficient of variation (CV), 
    #also known as relative standard deviation (RSD)
    #It is often expressed as a percentage, and is defined as 
    #the ratio of the standard deviation to the mean 
    #ax=plt.figure()
    
    ds = xr.DataArray(obs_dataset.values,
    coords={'time': obs_dataset.times,
    'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
    dims=["time", "lat", "lon"])
    ds = ds.groupby('time.year').sum()
    #print(ds.mean(dim))
    #exit()
    ds=ds.mean(dim=("lat", "lon"))
    
    sd0=metrics.calc_stddev(ds.values, axis=0)
    print(sd0)
    m0=utils.calc_temporal_mean(ds).mean()
    print(m0)
    
    #bulanan
    #sd0=metrics.calc_stddev(obs_dataset.values, axis=0).mean()
    #m0=utils.calc_temporal_mean(obs_dataset).mean()
    cv=np.zeros(len(model_datasets)+1)
    mm=np.zeros(len(model_datasets)+1)
    mm[0]=m0 #-5 
    cv[0]=sd0/m0*100
    for i in np.arange(len(model_datasets)):
        dsi = xr.DataArray(model_datasets[i].values,
                coords={'time': model_datasets[i].times,
                'lat': model_datasets[i].lats, 'lon': model_datasets[i].lons},
                dims=["time", "lat", "lon"]) 
        dsi = dsi.groupby('time.year').sum()
        dsi=dsi.mean(dim=("lat", "lon"))
        sdi=metrics.calc_stddev(dsi.values, axis=0)
        m=utils.calc_temporal_mean(dsi).mean()
        #sdi=metrics.calc_stddev(model_datasets[i].values, axis=0).mean() 
        #m=utils.calc_temporal_mean(model_datasets[i]).mean()
        mm[i+1]=m
        cv[i+1]=sdi/m*100
    plt.scatter(cv,mm, marker='')
    #n=np.arange(1,13)
    n=names
    #print(model_names)
    #n=model_names.insert(0,'GPCP')
    #n=np. concatenate(obs_name,model_names)
    #n=obs_name+model_names
    #print(n)
    for i, txt in enumerate(n):
        plt.annotate(txt, (cv[i], mm[i]))
    #marker='$%d$' %np.arange(1,13)
    plt.ylabel('Precipitation (mm/year)')
    plt.xlabel('Coefficient of variation (%)')
    #plt.xlim(right=12)
    plt.xlim(2,12)
   
    figname='cv_year_'+reg
    plt.savefig(workdir+figname, dpi=300, bbox_inches='tight')
    plt.show()
    
# the difference in SSTanomaly between the tropical western Indian Ocean 
# (50' E - 70' E, 10 S - 10 N) and the tropical south-eastern Indian Ocean 
# (90' E- 110' E, 10 S - 0). The strong correlation (>0.7) 
# between this index, referred to as the dipole mode index (DMI)   

def mjo(obs_dataset, obs_name, model_datasets, model_names, workdir):
    from scipy import signal
    
    ds = xr.DataArray(obs_dataset.values,
    coords={'time': obs_dataset.times,
    'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
    dims=["time", "lat", "lon"])
    
    # Load observational data (e.g., reanalysis data)
    obs_data = ds
    
    import xarray as xr

    # Load OLR data (replace with your data file and variable)
    olr_data = xr.open_dataset('path/to/olr_data.nc')['olr']

    # Calculate MJO index based on OLR anomalies (replace with your specific calculation)
    mjo_index = olr_data.groupby('time.month').mean(dim='time')  # Monthly mean
    mjo_index_anomaly = mjo_index - mjo_index.mean(dim='month')   # Calculate anomalies

    # You can then use mjo_index_anomaly for further analysis

    
    for i in np.arange(len(model_datasets)):
        dsi = xr.DataArray(model_datasets[i].values,
                coords={'time': model_datasets[i].times,
                'lat': model_datasets[i].lats, 'lon': model_datasets[i].lons},
                dims=["time", "lat", "lon"]) 
    
   
    # Load model output data
    model_data = dsi

    # Define MJO indices or relevant variables
    obs_mjo_index = obs_data['mjo_index']
    model_mjo_index = model_data['mjo_index']

    # Time-frequency analysis using wavelet
    def wavelet_analysis(data, dt):
        _, _, power = signal.morlet(data, dt)
        return power

    obs_wavelet_power = wavelet_analysis(obs_mjo_index, dt=1)  # Adjust dt based on your data time step
    model_wavelet_power = wavelet_analysis(model_mjo_index, dt=1)

    # Spectral analysis
    def spectral_analysis(data, dt):
        freq, power = signal.welch(data, fs=1/dt, nperseg=len(data))
        return freq, power

    obs_freq, obs_power = spectral_analysis(obs_mjo_index, dt=1)
    model_freq, model_power = spectral_analysis(model_mjo_index, dt=1)

    # Spatial patterns - Plot phase diagrams
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.contourf(obs_data['longitude'], obs_data['time'], obs_data['mjo_index'])
    plt.title('Observational MJO Phase Diagram')

    plt.subplot(1, 2, 2)
    plt.contourf(model_data['longitude'], model_data['time'], model_data['mjo_index'])
    plt.title('Model MJO Phase Diagram')

    plt.show()

    # Other evaluation metrics and visualizations can be added based on your specific requirements


def dmi_anomaly(obs_dataset, workdir):
      
    import xarray as xr 
    import pandas as pd
    
    filepath=[
            'D:/data1/tos_Omon_CNRM-CM5_historical_r1i1p1_198101-200512.nc',
            'D:/data1/tos_Omon_IPSL-CM5A-LR_historicalNat_r1i1p1_1981-2005.nc',
            'D:/data1/tos_Omon_HadGEM2-ES_esmHistorical_r1i1p1_1981-2005_11.nc',
            'D:/data1/tos_Omon_NorESM1-M_historical_r1i1p1_198101-200512.nc',
            'D:/data1/4tos_Omon_GFDL-ESM2M_historical_r1i1p1_198101-200512.nc'
            ]
            
    names=['CNRM', 'IPSL', 'HadGEM2', 'NorESM1', 'GFDL' ]
    
    ds = xr.open_dataset(filepath[0])
    
    # w=[-10, 10, 50, 70]
    # e=[-10, 0, 90, 110]
    # n=[-5, 5, 190, 240]
    # # (50' E - 70' E, 10 S - 10 N)
    # #(90' E- 110' E, 10 S - 0 )
    # #lons, lats = np.meshgrid(ref_dataset.lons, ref_dataset.lats) 
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # #m = Basemap(ax=ax, projection='cyl',llcrnrlat = ds.lat.min(), urcrnrlat = ds.lat.max(),
    # #            llcrnrlon = ds.lon.min(), urcrnrlon = ds.lon.max(), resolution = 'h')
    # m = Basemap(ax=ax, projection='cyl',llcrnrlat = -30, 
                                        # urcrnrlat = 30,
                                        # llcrnrlon = 20, 
                                        # urcrnrlon = 300, resolution = 'h')
    # m.drawcoastlines(linewidth=0.75)
    # m.drawcountries(linewidth=0.75)
    # #m.etopo()  
    # #x, y = m(lons, lats) 
    # #subregion_array = ma.masked_equal(subregion_array, 0)
    # #max=m.contourf(x, y, subregion_array, alpha=0.7, cmap='Accent')

    # draw_screen_poly(w, m, 'b') 
    # draw_screen_poly(e, m, 'b') 
    # draw_screen_poly(n, m, 'b') 
    
    # plt.savefig(workdir+'nino34_dmi2.png', dpi=300, bbox_inches='tight')
    # plt.show()
    
    # exit()
 
    #fig, ax = plt.subplots(nrows=4, ncols=1 ) #,figsize=(6,4))
    
    #--------------------------------------- SEA 200412 
    d= pd.read_excel('D:/Disertasi3/dmi_1981-2005.xlsx')
    
    index_nino34 = xr.DataArray( d['Value'],
    coords={'time': obs_dataset.times})
   
    fig, ax = plt.subplots(nrows=1, ncols=1 )
    #fig=plt.figure(figsize=[8,6])
    fig.subplots_adjust(right=.7)
    
    plt.plot(obs_dataset.times, d['Value'], color='black', label = 'Obs')
    plt.axhline(.5, linestyle = 'dashed', color='red')
    plt.axhline(-.5, linestyle = 'dashed', color='red')
    #ax.set_xticklabels(obs_dataset.times) error
    #exit()

    
    for i in np.arange(len(filepath)):
      
        print(i, names[i])
           
        dsx = xr.open_dataset(filepath[i])
        #print(dsx)
        #slice ini nans
        #tos_nino34 = ds.sel(i=slice(-5, 5), j=slice(190, 240))
        # (50' E - 70' E, 10 S - 10 N)
        try:
            tos_w = dsx.where(
                    (dsx.lat < 10) & (dsx.lat > -10) & 
                    (dsx.lon > 50) & (dsx.lon < 70), drop=True)
        #GFDL pakai rlat,rlon            
        except:
            tos_w = dsx.where(
                    (dsx.rlat < 10) & (dsx.rlat > -10) & 
                    (dsx.rlon > 50) & (dsx.rlon < 70), drop=True)
                    
                    
        try:
            tos_e = dsx.where(
                    (dsx.lat < 0) & (dsx.lat > -10) & 
                    (dsx.lon > 90) & (dsx.lon < 110), drop=True)
                    
        except:
            tos_e = dsx.where(
                    (dsx.rlat < 0) & (dsx.rlat > -10) & 
                    (dsx.rlon > 90) & (dsx.rlon < 110), drop=True)            
                    
                    
        # #(50' E - 70' E, 10 S - 10 N)            
        # tos_w = dsx.where(
                    # (dsx.lat < 10) & (dsx.lat > -10) & 
                    # (dsx.lon > 50) & (dsx.lon < 70), drop=True)
        # #(90' E- 110' E, 10 S - 0 )
        # tos_e = dsx.where(
                    # (dsx.lat < 0) & (dsx.lat > -10) & 
                    # (dsx.lon > 90) & (dsx.lon < 110), drop=True)

        tos_nino34=tos_w 
        
        gb = tos_nino34.tos.groupby('time.month')
        tos_nino34_anom = gb - gb.mean(dim='time')
        #print(tos_nino34_anom)
        try: index_nino34 = tos_nino34_anom.mean(dim=['rlat', 'rlon'])
        except: 
            try: index_nino34 = tos_nino34_anom.mean(dim=['i', 'j'])
            except: index_nino34 = tos_nino34_anom.mean(dim=['lat', 'lon'])
        # print('index_nino34=',index_nino34)
        # #smooth the anomalies with a 5-month running m.vvaleuean:
        #index_nino34_rolling_mean = index_nino34.rolling(time=5, center=True).mean()
        ir5_w = index_nino34.rolling(time=5, center=True).mean()
        
        tos_nino34=tos_e 
        
        gb = tos_nino34.tos.groupby('time.month')
        tos_nino34_anom = gb - gb.mean(dim='time')
        #print(tos_nino34_anom)
        try: index_nino34 = tos_nino34_anom.mean(dim=['rlat', 'rlon'])
        except: 
            try: index_nino34 = tos_nino34_anom.mean(dim=['i', 'j'])
            except: index_nino34 = tos_nino34_anom.mean(dim=['lat', 'lon'])
        # print('index_nino34=',index_nino34)
        # #smooth the anomalies with a 5-month running m.vvaleuean:
        #index_nino34_rolling_mean = index_nino34.rolling(time=5, center=True).mean()
        ir5_e = index_nino34.rolling(time=5, center=True).mean()
        
        ir5=ir5_w -ir5_e
        
        #karna i=2 hanya sampai 2005-11 maka
        times=obs_dataset.times
        if i !=2: 
            ir5=np.delete(ir5,-1)
            times=np.delete(times,-1)
            print(len(ir5),len(times))
        else: 
            times=np.delete(times,-1)
            print(len(ir5),len(times))
        
        
        #plt.plot(obs_dataset.times, ir5.values, label = names[i])
        plt.plot(times, ir5.values, label = names[i])
    
    #ax.set_xticks([1985, 1990,1995,2000, 2005])
    plt.legend(bbox_to_anchor=(1, .5), loc='best', prop={'size':10}, frameon=False) 
    plt.ylabel('STT_IOD anomaly (K)')
    #plt.set_xticks([1981,1988,2005])
    #plt.title('SST anomaly obs & models over the Niño3.4 region');
    figname='SST_IOD_1981-2005'
    plt.savefig(workdir+figname) #, dpi=300, bbox_inches='tight')
    plt.show()

def dmi_stdev(obs_dataset, obs_name, model_datasets, model_names, workdir):
      
    import xarray as xr 
    import pandas as pd
    
    fig, ax = plt.subplots(nrows=1, ncols=3 ,figsize=(6,4))
    #--------------------------------------- SEA 
    d= pd.read_excel('D:/Disertasi3/dmi_1981-2005.xlsx')
    index_nino34 = xr.DataArray( d['Value'],
    coords={'time': obs_dataset.times})
    
    x0= index_nino34
    
    sdo= index_nino34.std()
    #print(index_nino34.std())
    ax[0].scatter(0, sdo, label='Obs',color='black')
    ax[0].axhline(sdo, linestyle = 'dashed', color='black')
 
    from scipy.stats import pearsonr
    
    
    #era
    ds = xr.DataArray(model_datasets[0].values,
    coords={'time': obs_dataset.times,
    'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
    dims=["time", "lat", "lon"])
    
    ds0= ds.mean(dim=('lat','lon'))
    #print(ds0)
    
    md = ds0.groupby('time.month')
    d_anom = md - md.mean(dim='time')
    pr5 = d_anom.rolling(time=5, center=True).mean()
    y0=pr5.values  
    
    
    bad = ~np.logical_or(np.isnan(x0), np.isnan(y0))
    x1=np.compress(bad, x0) 
    y1=np.compress(bad, y0)
    
    r=pearsonr(x1, y1)[0]*-1
    
    ax[1].scatter(0, r, label='ERA5',color='black')
    ax[1].axhline(r, linestyle = 'dashed', color='black')
    
    ax[2].scatter(0, r, label='ERA5',color='black')
    ax[2].axhline(r, linestyle = 'dashed', color='black')
    
    #GPCP
    ds = xr.DataArray(obs_dataset.values,
    coords={'time': obs_dataset.times,
    'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
    dims=["time", "lat", "lon"])
    
    ds0= ds.mean(dim=('lat','lon'))
    #print(ds0)
    
    md = ds0.groupby('time.month')
    d_anom = md - md.mean(dim='time')
    pr5 = d_anom.rolling(time=5, center=True).mean()
    y0=pr5.values  
    
    
    bad = ~np.logical_or(np.isnan(x0), np.isnan(y0))
    x1=np.compress(bad, x0) 
    y1=np.compress(bad, y0)
    
    r=pearsonr(x1, y1)[0]*-1
    ax[1].scatter(0, r, label='GPCP',color='blue')
    ax[1].axhline(r, linestyle = 'dashed', color='blue')
    
    ax[2].scatter(0, r, label='GPCP',color='blue')
    ax[2].axhline(r, linestyle = 'dashed', color='blue')

    filepath=[
            'D:/data1/tos_Omon_CNRM-CM5_historical_r1i1p1_198101-200512.nc',
            'D:/data1/tos_Omon_IPSL-CM5A-LR_historicalNat_r1i1p1_1981-2005.nc',
            'D:/data1/tos_Omon_HadGEM2-ES_esmHistorical_r1i1p1_1981-2005_11.nc',
            'D:/data1/tos_Omon_NorESM1-M_historical_r1i1p1_198101-200512.nc',
            'D:/data1/4tos_Omon_GFDL-ESM2M_historical_r1i1p1_198101-200512.nc'
            ]
    
    names=['CNRM', 'IPSL', 'HadGEM2','NorESM1', 'GFDL']
    mm=['s','^', '*','D','v','P','H']
    for i in np.arange(len(filepath)):
        print(i)
        dsx = xr.open_dataset(filepath[i])
  
        try:
            tos_w = dsx.where(
                    (dsx.lat < 10) & (dsx.lat > -10) & 
                    (dsx.lon > 50) & (dsx.lon < 70), drop=True)
        #GFDL pakai rlat,rlon             
        except:
            try:
                tos_w = dsx.where(
                    (dsx.rlat < 10) & (dsx.rlat > -10) & 
                    (dsx.rlon > 50) & (dsx.rlon < 70), drop=True)
            except:
                #ini tidak perlu
                print('i,j terpakai')
                tos_w = dsx.where(
                    (dsx.j < 10) & (dsx.j > -10) & 
                    (dsx.i > 50) & (dsx.i < 70), drop=True)   
        try:
            tos_e = dsx.where(
                    (dsx.lat < 0) & (dsx.lat > -10) & 
                    (dsx.lon > 90) & (dsx.lon < 110), drop=True)         
        except:
            try:
                tos_e = dsx.where(
                    (dsx.rlat < 0) & (dsx.rlat > -10) & 
                    (dsx.rlon > 90) & (dsx.rlon < 110), drop=True) 
            except:
                print('i,j terpakai')
                tos_e = dsx.where(
                    (dsx.j < 0) & (dsx.j > -10) & 
                    (dsx.i > 90) & (dsx.i < 110), drop=True) 
        
        tos_nino34=tos_w 
        
        gb = tos_nino34.tos.groupby('time.month')
        tos_nino34_anom = gb - gb.mean(dim='time')
        #print(tos_nino34_anom)
        try: index_nino34 = tos_nino34_anom.mean(dim=['rlat', 'rlon'])
        except:
            try: index_nino34 = tos_nino34_anom.mean(dim=['lat', 'lon'])
            except: index_nino34 = tos_nino34_anom.mean(dim=['i', 'j'])

        ir5_w = index_nino34.rolling(time=5, center=True).mean()
        
        tos_nino34=tos_e 
        
        gb = tos_nino34.tos.groupby('time.month')
        tos_nino34_anom = gb - gb.mean(dim='time')
        #print(tos_nino34_anom)
        try: index_nino34 = tos_nino34_anom.mean(dim=['rlat', 'rlon'])
        except:
            try: index_nino34 = tos_nino34_anom.mean(dim=['lat', 'lon'])
            except: index_nino34 = tos_nino34_anom.mean(dim=['i', 'j'])

        ir5_e = index_nino34.rolling(time=5, center=True).mean()
        
        ir5=ir5_w -ir5_e
        
        sdi= ir5.std()
         
        ax[0].scatter(i+1, sdi, label=names[i],marker=mm[i])
        ax[0].set_xticks([])
        ax[0].legend(bbox_to_anchor=(.98, .4), loc='best', prop={'size':10}, frameon=True) 
    
    
    n =[0,1,2,2,2,3,4] 
    nn=0
    for i in [1,3,4,5,6,8,9]:
        print(i)
        dsx = xr.open_dataset(filepath[n[nn]])
  
        try:
            tos_w = dsx.where(
                    (dsx.lat < 10) & (dsx.lat > -10) & 
                    (dsx.lon > 50) & (dsx.lon < 70), drop=True)
        #GFDL pakai rlat,rlon             
        except:
            try:
                tos_w = dsx.where(
                    (dsx.rlat < 10) & (dsx.rlat > -10) & 
                    (dsx.rlon > 50) & (dsx.rlon < 70), drop=True)
            except:
                #ini tidak perlu
                print('i,j terpakai')
                tos_w = dsx.where(
                    (dsx.j < 10) & (dsx.j > -10) & 
                    (dsx.i > 50) & (dsx.i < 70), drop=True)   
        try:
            tos_e = dsx.where(
                    (dsx.lat < 0) & (dsx.lat > -10) & 
                    (dsx.lon > 90) & (dsx.lon < 110), drop=True)         
        except:
            try:
                tos_e = dsx.where(
                    (dsx.rlat < 0) & (dsx.rlat > -10) & 
                    (dsx.rlon > 90) & (dsx.rlon < 110), drop=True) 
            except:
                print('i,j terpakai')
                tos_e = dsx.where(
                    (dsx.j < 0) & (dsx.j > -10) & 
                    (dsx.i > 90) & (dsx.i < 110), drop=True) 
        
        tos_nino34=tos_w 
        
        gb = tos_nino34.tos.groupby('time.month')
        tos_nino34_anom = gb - gb.mean(dim='time')
        #print(tos_nino34_anom)
        try: index_nino34 = tos_nino34_anom.mean(dim=['rlat', 'rlon'])
        except:
            try: index_nino34 = tos_nino34_anom.mean(dim=['lat', 'lon'])
            except: index_nino34 = tos_nino34_anom.mean(dim=['i', 'j'])

        ir5_w = index_nino34.rolling(time=5, center=True).mean()
        
        tos_nino34=tos_e 
        
        gb = tos_nino34.tos.groupby('time.month')
        tos_nino34_anom = gb - gb.mean(dim='time')
        #print(tos_nino34_anom)
        try: index_nino34 = tos_nino34_anom.mean(dim=['rlat', 'rlon'])
        except:
            try: index_nino34 = tos_nino34_anom.mean(dim=['lat', 'lon'])
            except: index_nino34 = tos_nino34_anom.mean(dim=['i', 'j'])

        ir5_e = index_nino34.rolling(time=5, center=True).mean()
        
        ir5=ir5_w -ir5_e
        
               
        #for ii in [1,8,9]: #in np.arange(len(model_datasets)-):#
        dsi = xr.DataArray(model_datasets[i].values,
        coords={'time': obs_dataset.times,
        'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
        dims=["time", "lat", "lon"])  
        
        dsi= dsi.mean(dim=('lat','lon'))
        
        md = dsi.groupby('time.month')
        d_anom = md - md.mean(dim='time')
        pr5 = d_anom.rolling(time=5, center=True).mean()
        y0=pr5.values
              
        
        bad = ~np.logical_or(np.isnan(x0), np.isnan(y0))
        x1=np.compress(bad, x0) 
        y1=np.compress(bad, y0)
        r=pearsonr(x1, y1)[0]*-1 
        ax[1].scatter(nn+1, r, label=model_names[i], marker=mm[nn])
        ax[1].set_xticks([])
        
        if i in [4,5,6]: 
            pr5=np.delete(pr5,-1)
            y0=pr5.values 
           
        else: 
            y0=pr5.values
        
        x01= ir5
        bad = ~np.logical_or(np.isnan(x01), np.isnan(y0))
        x1=np.compress(bad, x01) 
        y1=np.compress(bad, y0)
        r=pearsonr(x1, y1)[0]*-1 
        ax[2].scatter(nn+1, r, label=model_names[i], marker=mm[nn])
        ax[2].set_xticks([])
        
        nn=nn+1
 
    
    #ax[1].legend(bbox_to_anchor=(.99, .5), loc='best', prop={'size':10}, frameon=False)
    #ax[2].legend(bbox_to_anchor=(.99, .5), loc='best', prop={'size':10}, frameon=False)     
    ax[2].legend(bbox_to_anchor=(.93, .5), loc='best', prop={'size':10}, frameon=False, handletextpad=0) 
    #ax[0].legend(loc=0)
   
    
    ax[0].set_ylabel('Standart deviation (IOD index) (K)')
    ax[1].set_ylabel('Correlation (IOD index vs SEA rainfall anomaly)', labelpad=0)
    ax[0].set_xlabel('(a)')
    ax[1].set_xlabel('(b)')
    ax[2].set_xlabel('(c)')
    
    #plt.set_xticks([1981,1988,2005])
    #plt.title('SST anomaly obs & models over the Niño3.4 region');
   
    fig.subplots_adjust(hspace=.05,wspace=.3)
    fig.subplots_adjust(right=.85)
  
    
    #plt.show()
    figname='dmi_stdev_corr_3'
    plt.savefig(workdir+figname+reg, dpi=300, bbox_inches='tight')
    plt.show()
    
    
def iod_ens(obs_dataset, obs_name, model_datasets, model_names, workdir):
    from scipy.stats import pearsonr
    import xarray as xr
    import pandas as pd
    
    fig, ax = plt.subplots(nrows=1, ncols=2 ,figsize=(6,4))
    #--------------------------------------- SEA 
    d= pd.read_excel('D:/Disertasi3/dmi_1981-2005.xlsx')
   
    index_nino34 = xr.DataArray( d['Value'])
    #coords={'time': obs_dataset.times})
    
    
    sdo= index_nino34.std()
    #print(index_nino34.std())
    ax[0].scatter(0, sdo, label='Obs',color='black')
    ax[0].axhline(sdo, linestyle = 'dashed', color='black')
    
   
    for i in [0,1,2,3,4,5]:
        print(i, model_names[i])
        
        dsi = xr.DataArray(model_datasets[i].values,
        coords={'time': obs_dataset.times,
        'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
        dims=["time", "lat", "lon"])  
        
        dsi= dsi.mean(dim=('lat','lon'))
        
        md = dsi.groupby('time.month')
        d_anom = md - md.mean(dim='time')
        ir5 = d_anom.rolling(time=5, center=True).mean()
        
        sdi= ir5.std()
        if i==5: 
            x0=ir5
            print(model_names[i])
            ax[0].scatter(i, sdi, label=model_names[i],color='blue')
            ax[0].axhline(sdi, linestyle = 'dashed', color='blue')
        else:    
            ax[0].scatter(i, sdi, label=model_names[i])
        ax[0].set_xticks([])
        
    # y0=index_nino34.values
    # y0=np.delete(y0,-1)
                
    # bad = ~np.logical_or(np.isnan(x0), np.isnan(y0))
    # x1=np.compress(bad, x0) 
    # y1=np.compress(bad, y0)
           
    # r=pearsonr(x1, y1)[0]*-1 
    
    # ax[1].scatter(0, r, label='MME')
  
    for i in [6,7,8,9]: #mmew pi ss t 678
        dsi = xr.DataArray(model_datasets[i].values,
        coords={'time': model_datasets[i].times,
        'lat': model_datasets[i].lats, 'lon': model_datasets[i].lons},
        dims=["time", "lat", "lon"])  
        
                
        dsi= dsi.mean(dim=('lat','lon'))
        
        md = dsi.groupby('time.month')
        d_anom = md - md.mean(dim='time')
        pr5 = d_anom.rolling(time=5, center=True).mean()
       
        pr5=np.delete(pr5,-1)
        y0=pr5.values
                
        bad = ~np.logical_or(np.isnan(x0), np.isnan(y0))
        x1=np.compress(bad, x0) 
        y1=np.compress(bad, y0)
               
        r=pearsonr(x1, y1)[0]*-1 
        
        
        if i==6:
            ax[1].scatter(0, r, label='MME',color='black')
            ax[1].axhline(r, linestyle = 'dashed', color='black')
        else:
            ax[1].scatter(i, r, label=model_names[i])
    ax[1].set_xticks([])
    ax[0].legend(bbox_to_anchor=(1, .5), loc='best', prop={'size':10}, frameon=True) 
    ax[0].legend(loc=0)
    ax[1].legend(loc=0)
    
    ax[0].set_ylabel('SSTA IOD stdev (K)')
    ax[1].set_ylabel('Correlation (SSTA vs prA)')
    #plt.set_xticks([1981,1988,2005])
    #plt.title('SST anomaly obs & models over the Niño3.4 region');
   
    fig.subplots_adjust(hspace=.05,wspace=.3)
  
    
    #plt.show()
    figname='iod_stdev_corr_mme'
    plt.savefig(workdir+figname+reg, dpi=300, bbox_inches='tight')
    plt.show()
    
    
def iod_ens2(obs_dataset, obs_name, model_datasets, model_names, workdir):
    from scipy.stats import pearsonr
    import xarray as xr
    import pandas as pd
    
    fig, ax = plt.subplots(nrows=1, ncols=2 ,figsize=(6,4))
    #--------------------------------------- SEA 
    d= pd.read_excel('D:/Disertasi3/dmi_1981-2005.xlsx')
   
    iod_obs = xr.DataArray( d['Value'], 
        coords={'time': obs_dataset.times})
        
    gb = iod_obs.groupby('time.month')
    iod_obs = gb - gb.mean(dim='time')
    ir5_obs = iod_obs.rolling(time=5, center=True).mean()
    sdo= ir5_obs.std()
    
    
    ax[0].scatter(0, sdo, label='Obs',color='black')
    ax[0].axhline(sdo, linestyle = 'dashed', color='black')
 
    for i in [0,1,2,3,4,5]:
        print(i, model_names[i])
        
        dsx = xr.DataArray(model_datasets[i].values,
        coords={'time': model_datasets[i].times,
        'lat': model_datasets[i].lats, 'lon': model_datasets[i].lons},
        dims=["time", "lat", "lon"])
        
        print('dsx.mean()=',dsx.mean())
        #try:
        tos_w = dsx.where(
                    (dsx.lat < 10) & (dsx.lat > -10) & 
                    (dsx.lon > 50) & (dsx.lon < 70), drop=True)
        
        #try:
        tos_e = dsx.where(
                    (dsx.lat < 0) & (dsx.lat > -10) & 
                    (dsx.lon > 90) & (dsx.lon < 110), drop=True)         
        
        print('tos_e.mean()=',tos_e.mean())
        print('tos_w.mean()=',tos_w.mean())
        
        tos_nino34=tos_w 
        
        gb = tos_nino34.groupby('time.month')
        tos_nino34_anom = gb - gb.mean(dim='time')
       
        index_nino34 = tos_nino34_anom.mean(dim=['lat', 'lon'])
        

        ir5_w = index_nino34.rolling(time=5, center=True).mean()
        
        tos_nino34=tos_e 
        
        gb = tos_nino34.groupby('time.month')
        tos_nino34_anom = gb - gb.mean(dim='time')
        #print(tos_nino34_anom)
        index_nino34 = tos_nino34_anom.mean(dim=['lat', 'lon'])
            
        ir5_e = index_nino34.rolling(time=5, center=True).mean()
        
        #ir5=ir5_w -ir5_e
        
        ir5=ir5_e #-ir5_e
        
        sdi= ir5.std()
        if i==5: 
            x0=ir5
            print(model_names[i])
            ax[0].scatter(i, sdi, label=model_names[i],color='blue')
            ax[0].axhline(sdi, linestyle = 'dashed', color='blue')
        else:    
            ax[0].scatter(i, sdi, label=model_names[i])
        ax[0].set_xticks([])
  
  
    for i in [6,7,8,9]: #mmew pi ss t 678
        dsi = xr.DataArray(model_datasets[i].values,
        coords={'time': model_datasets[i].times,
        'lat': model_datasets[i].lats, 'lon': model_datasets[i].lons},
        dims=["time", "lat", "lon"])  
                       
        dsi= dsi.mean(dim=('lat','lon'))
        
        md = dsi.groupby('time.month')
        d_anom = md - md.mean(dim='time')
        pr5 = d_anom.rolling(time=5, center=True).mean()
       
        pr5=np.delete(pr5,-1)
        y0=pr5.values
                
        bad = ~np.logical_or(np.isnan(x0), np.isnan(y0))
        x1=np.compress(bad, x0) 
        y1=np.compress(bad, y0)
               
        r=pearsonr(x1, y1)[0]*-1 
               
        if i==6:
            ax[1].scatter(0, r, label='MME',color='black')
            ax[1].axhline(r, linestyle = 'dashed', color='black')
        else:
            ax[1].scatter(i, r, label=model_names[i])
    ax[1].set_xticks([])
    ax[0].legend(bbox_to_anchor=(1, .5), loc='best', prop={'size':10}, frameon=True) 
    ax[0].legend(loc=0)
    ax[1].legend(loc=0)
    
    ax[0].set_ylabel('SSTA IOD stdev (K)')
    ax[1].set_ylabel('Correlation (SSTA vs prA)')
 
    fig.subplots_adjust(hspace=.05,wspace=.3)
 
    figname='iod_stdev_mme_east'
    plt.savefig(workdir+figname+reg, dpi=300, bbox_inches='tight')
    plt.show()

def iod_ens_corr(obs_dataset, obs_name, model_datasets, model_names, workdir):
   
    
    import xarray as xr
    import pandas as pd
    from scipy.stats import pearsonr
    
    fig, ax = plt.subplots(nrows=2, ncols=4 ,figsize=(8,6))
    
    d= pd.read_excel('D:/Disertasi3/dmi_1981-2005.xlsx')
    
    nino_obs = xr.DataArray( d['Value'], coords={'time': obs_dataset.times})
    gb = nino_obs.groupby('time.month')
    nino_obs = gb - gb.mean(dim='time')
    ir5_obs = nino_obs.rolling(time=5, center=True).mean()
    x0=ir5_obs.values
   
   
    ds = xr.DataArray(obs_dataset.values,
    coords={'time': obs_dataset.times,
    'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
    dims=["time", "lat", "lon"])
    
    lat_min = ds.lat.min()
    lat_max = ds.lat.max()
    lon_min = ds.lon.min()
    lon_max = ds.lon.max()
   
    x,y = np.meshgrid(ds.lon, ds.lat)
    
    tes=0
    
    map0=ma.zeros((len(ds.lat),len(ds.lon)))
   
    
    for ii in np.arange(len(ds.lat)-tes):
        print(ii)
        for jj in np.arange(len(ds.lon)-tes):
            d=ds[:,ii,jj]
            md = d.groupby('time.month')
            d_anom = md - md.mean(dim='time')
            pr5 = d_anom.rolling(time=5, center=True).mean()
            y0=pr5.values
          
            bad = ~np.logical_or(np.isnan(x0), np.isnan(y0))
            x1=np.compress(bad, x0) 
            y1=np.compress(bad, y0)
       
            if x1.shape==(0,) or y1.shape==(0,):
                R='nan'
            else:
                R=pearsonr(x1, y1)[0]*-1
                #R=pearsonr(pr5, ir5)[0]*-1
            map0[ii,jj]=R
    m0=map0.flatten()
    
    m = Basemap(ax=ax[0,0], projection ='cyl', llcrnrlat = lat_min, urcrnrlat = lat_max,
                llcrnrlon = lon_min, urcrnrlon = lon_max, resolution = 'l', fix_aspect=False)
    m.drawcoastlines(linewidth=1)
    m.drawcountries(linewidth=1)
    m.drawstates(linewidth=0.5, color='w')
    
    max = ax[0,0].contourf(x,y,map0)
    #ax[0,nn].set_title(model_names[i])
    ax[0,0].set_title('GPCP', pad=3,fontsize=10)
    ax[0,0].set_yticks([-10,0,10,20])
    #ax[0].set_xticks([100,120,140])
    ax[0,0].yaxis.set_tick_params(labelsize=7)
    #ax[0].xaxis.set_tick_params(labelsize=7)
    for i in [1,2,3]: ax[0,i].axis('off')
    
    #tos mme
    i=5
    dsx = xr.DataArray(model_datasets[i].values,
        coords={'time': model_datasets[i].times,
        'lat': model_datasets[i].lats, 'lon': model_datasets[i].lons},
        dims=["time", "lat", "lon"])
        
    print('dsx.mean()=',dsx.mean())
    #try:
    tos_w = dsx.where(
                (dsx.lat < 10) & (dsx.lat > -10) & 
                (dsx.lon > 50) & (dsx.lon < 70), drop=True)
    
    #try:
    tos_e = dsx.where(
                (dsx.lat < 0) & (dsx.lat > -10) & 
                (dsx.lon > 90) & (dsx.lon < 110), drop=True)         
    
    print('tos_e.mean()=',tos_e.mean())
    print('tos_w.mean()=',tos_w.mean())
    
    tos_nino34=tos_w 
    
    gb = tos_nino34.groupby('time.month')
    tos_nino34_anom = gb - gb.mean(dim='time')
   
    index_nino34 = tos_nino34_anom.mean(dim=['lat', 'lon'])
    

    ir5_w = index_nino34.rolling(time=5, center=True).mean()
    
    tos_nino34=tos_e 
    
    gb = tos_nino34.groupby('time.month')
    tos_nino34_anom = gb - gb.mean(dim='time')
    #print(tos_nino34_anom)
    index_nino34 = tos_nino34_anom.mean(dim=['lat', 'lon'])
        
    ir5_e = index_nino34.rolling(time=5, center=True).mean()
    
    ir5=ir5_w -ir5_e
    x0=ir5.values
    
    map1=ma.zeros((len(ds.lat),len(ds.lon)))
  
    r0=1
    #for n in pilih_nino:
    for i in [6,7,8,9]:
        
        print(model_names[i])
        
        dsi = xr.DataArray(model_datasets[i].values,
        coords={'time': model_datasets[i].times,
        'lat': model_datasets[i].lats, 'lon': model_datasets[i].lons},
        dims=["time", "lat", "lon"])  
        
               
        
        m = Basemap(ax=ax[1,i-6], projection ='cyl', llcrnrlat = lat_min, urcrnrlat = lat_max,
            llcrnrlon = lon_min, urcrnrlon = lon_max, resolution = 'l', fix_aspect=False)
        m.drawcoastlines(linewidth=1)
        m.drawcountries(linewidth=1)
        m.drawstates(linewidth=0.5, color='w')
            
        map1=ma.zeros((len(dsi.lat),len(dsi.lon)))
        
        for ii in np.arange(len(dsi.lat)-tes):
            print(ii)
            for jj in np.arange(len(dsi.lon)-tes):
                d=dsi[:,ii,jj]
                md = d.groupby('time.month')
                d_anom = md - md.mean(dim='time')
                pr5 = d_anom.rolling(time=5, center=True).mean()
               
                pr5=np.delete(pr5,-1)
                y0=pr5.values
              
                bad = ~np.logical_or(np.isnan(x0), np.isnan(y0))
                x1=np.compress(bad, x0) 
                y1=np.compress(bad, y0)
                #print(x,y)
                #print(x.shape,y.shape)
                if x1.shape==(0,) or y1.shape==(0,):
                    R='nan'
                else:
                    R=pearsonr(x1, y1)[0]*-1
                    #R=pearsonr(pr5, ir5)[0]*-1
                map1[ii,jj]=R
                #if i==0: m0=map1.flatten()
                
        m1=map1.flatten()
        bad = ~np.logical_or(np.isnan(m0), np.isnan(m1))
        x1=np.compress(bad, m0) 
        y1=np.compress(bad, m1)
        
        sd1=x1.std() #(skipna=None)
        #print(sd1)
        sd2=y1.std()
        s=sd2/sd1
        
        c,pp=pearsonr(x1.flatten() , y1.flatten())        
        
        #T= (4*(1+c)**4)/(((s+1/s)**2)*((1+r0)**4))
        # Tanpa pakai **4 T naik dikit
        tt= (4*(1+c))/(((s+1/s)**2)*((1+r0)))
        #T.append(round(tt,2))#taylor score
                
                
        #print('map1=',map1)
        max = ax[1,i-6].contourf(x,y,map1)
        #ax[0,nn].set_title(model_names[i])
        ax[1,i-6].set_title(model_names[i]+'('+'%.2f'%tt+')', pad=5,fontsize=10)
        ax[1,0].set_yticks([-10,0,10,20])
        ax[1,i-6].set_xticks([100,120,140])
        ax[1,i-6].xaxis.set_tick_params(labelsize=7)
        ax[1,0].yaxis.set_tick_params(labelsize=7)
 
    plt.subplots_adjust(hspace=.15,wspace=.12)
    #cax = fig.add_axes([0.91, 0.5, 0.015, 0.4])
    cax = fig.add_axes([0.4, 0.6, 0.4, 0.04]) #horisontal
    #plt.colorbar(max, cax = cax) 
    cbar=plt.colorbar(max, cax = cax, extend='both', orientation='horizontal') 
    cbar.ax.tick_params(labelsize=7)
    
    file_name='Corr_iod_tos_mme2_'+reg
    fig.savefig(workdir+file_name,dpi=300,bbox_inches='tight')
    plt.show()



def dmi_corr_pv(obs_dataset, obs_name, model_datasets, model_names, workdir):
    #dmi obs vs rainfall model
    #dmi cmip5 model vs rainfall model 
 
    import xarray as xr
    import pandas as pd
    fig, ax = plt.subplots(nrows=3, ncols=4 ,figsize=(8,6))
    
    d= pd.read_excel('D:/Disertasi3/dmi_1981-2005.xlsx')
    nino_obs = xr.DataArray( d['Value'], coords={'time': obs_dataset.times})
    gb = nino_obs.groupby('time.month')
    nino_obs = gb - gb.mean(dim='time')
    ir5_obs = nino_obs.rolling(time=5, center=True).mean()
    
    filepath=[
            'D:/data1/tos_Omon_CNRM-CM5_historical_r1i1p1_198101-200512.nc',
            'D:/data1/tos_Omon_IPSL-CM5A-LR_historicalNat_r1i1p1_1981-2005.nc',
            'D:/data1/tos_Omon_HadGEM2-ES_esmHistorical_r1i1p1_1981-2005_11.nc',
            'D:/data1/tos_Omon_NorESM1-M_historical_r1i1p1_198101-200512.nc',
            'D:/data1/4tos_Omon_GFDL-ESM2M_historical_r1i1p1_198101-200512.nc'
            ]
   
    ds = xr.DataArray(obs_dataset.values,
    coords={'time': obs_dataset.times,
    'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
    dims=["time", "lat", "lon"])
  
    lat_min = ds.lat.min()
    lat_max = ds.lat.max()
    lon_min = ds.lon.min()
    lon_max = ds.lon.max()
   
    x,y = np.meshgrid(ds.lon, ds.lat)
    
    from scipy.stats import pearsonr
    
    map0=ma.zeros((2, len(ds.lat),len(ds.lon)))
   
    tes=150
    #print('GPCP')
    
    x0=ir5_obs.values
    for ii in np.arange(len(ds.lat)-tes):
        print(ii)
        for jj in np.arange(len(ds.lon)-tes):
            d=ds[:,ii,jj]
            md = d.groupby('time.month')
            d_anom = md - md.mean(dim='time')
            pr5 = d_anom.rolling(time=5, center=True).mean()
            y0=pr5.values
           
            bad = ~np.logical_or(np.isnan(x0), np.isnan(y0))
            x1=np.compress(bad, x0) 
            y1=np.compress(bad, y0)
          
            if x1.shape==(0,) or y1.shape==(0,):
                R='nan'
                pv='nan'
            else:
                R=pearsonr(x1, y1)[0]*-1
                pv=pearsonr(x1, y1)[1]
                print('pv=',pv)
            map0[0,ii,jj]=R
            map0[1,ii,jj]=pv
            
            
    m0=map0.flatten()
    
    m = Basemap(ax=ax[0,0], projection ='cyl', llcrnrlat = lat_min, urcrnrlat = lat_max,
                llcrnrlon = lon_min, urcrnrlon = lon_max, resolution = 'l', fix_aspect=False)
    m.drawcoastlines(linewidth=1)
    m.drawcountries(linewidth=1)
    m.drawstates(linewidth=0.5, color='w')
    
    max = ax[0,0].contourf(x,y,map0[0,:,:])
    #ax[0,nn].set_title(model_names[i])
    ax[0,0].set_title('GPCP', pad=3,fontsize=10)
    for i in [1,2,3]: ax[0,i].axis('off')
    alpha=0.01
    density=4
    
    ax[0,0].contourf(x, y, map0[1,:,:],levels=[np.nanmin(map0[1,:,:]), alpha, ], 
                        hatches=[density*'/'],
                        extend='lower', alpha = 0)
       
    
    rpv = xr.DataArray(map0,
    coords={'rpv':[0,1],
    'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
    dims=["rpv", "lat", "lon"])
    rpv.to_netcdf('tes_pv.nc')
    print(rpv)
  
    plt.show()
    #agar tidak error saat saving buang titik pada alpha  
    #extensiom conflict
    alpha2=str(alpha)
    alpha2=alpha2.replace('.', '')
    file_name='Corr_iod_pv_'+alpha2+reg
    fig.savefig(workdir+file_name,dpi=300,bbox_inches='tight')
    exit()
    
    
    
def dmi_corr(obs_dataset, obs_name, model_datasets, model_names, workdir):
    #dmi obs vs rainfall model
    #dmi cmip5 model vs rainfall model 
 
    import xarray as xr
    import pandas as pd
    fig, ax = plt.subplots(nrows=3, ncols=4 ,figsize=(8,6))
    
    d= pd.read_excel('D:/Disertasi3/dmi_1981-2005.xlsx')
    nino_obs = xr.DataArray( d['Value'], coords={'time': obs_dataset.times})
    gb = nino_obs.groupby('time.month')
    nino_obs = gb - gb.mean(dim='time')
    ir5_obs = nino_obs.rolling(time=5, center=True).mean()
    
    filepath=[
            'D:/data1/tos_Omon_CNRM-CM5_historical_r1i1p1_198101-200512.nc',
            'D:/data1/tos_Omon_IPSL-CM5A-LR_historicalNat_r1i1p1_1981-2005.nc',
            'D:/data1/tos_Omon_HadGEM2-ES_esmHistorical_r1i1p1_1981-2005_11.nc',
            'D:/data1/tos_Omon_NorESM1-M_historical_r1i1p1_198101-200512.nc',
            'D:/data1/4tos_Omon_GFDL-ESM2M_historical_r1i1p1_198101-200512.nc'
            ]
   
    ds = xr.DataArray(obs_dataset.values,
    coords={'time': obs_dataset.times,
    'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
    dims=["time", "lat", "lon"])
  
    lat_min = ds.lat.min()
    lat_max = ds.lat.max()
    lon_min = ds.lon.min()
    lon_max = ds.lon.max()
   
    x,y = np.meshgrid(ds.lon, ds.lat)
    
    from scipy.stats import pearsonr
    
    
    tes=0
    alpha=0.01
    density=4
    
    print('len(ds.lat)', len(ds.lat))
    print(obs_name)
    
    x0=ir5_obs.values
    map0=ma.zeros((2, len(ds.lat),len(ds.lon)))
    for ii in np.arange(len(ds.lat)-tes):
        print(ii)
        for jj in np.arange(len(ds.lon)-tes):
            d=ds[:,ii,jj]
            md = d.groupby('time.month')
            d_anom = md - md.mean(dim='time')
            pr5 = d_anom.rolling(time=5, center=True).mean()
            y0=pr5.values
            #print(y0)
            #print(x0.shape)
            #print(y0.shape)
            #print(x0)
            #print(y0)
            bad = ~np.logical_or(np.isnan(x0), np.isnan(y0))
            x1=np.compress(bad, x0) 
            y1=np.compress(bad, y0)
            #print(x,y)
            #print(x.shape,y.shape)
            if x1.shape==(0,) or y1.shape==(0,):
                R='nan'
                pv='nan'
            else:
                R=pearsonr(x1, y1)[0]*-1
                pv=pearsonr(x1, y1)[1]
                #print('pv=',pv)
            map0[0,ii,jj]=R
            map0[1,ii,jj]=pv
            
            
    m0=map0.flatten()
    
    m = Basemap(ax=ax[0,0], projection ='cyl', llcrnrlat = lat_min, urcrnrlat = lat_max,
                llcrnrlon = lon_min, urcrnrlon = lon_max, resolution = 'l', fix_aspect=False)
    m.drawcoastlines(linewidth=1)
    m.drawcountries(linewidth=1)
    m.drawstates(linewidth=0.5, color='w')
    
    max = ax[0,0].contourf(x,y,map0[0,:,:])
    #ax[0,nn].set_title(model_names[i])
    ax[0,0].set_title(obs_name, pad=3,fontsize=10)
    for i in [1,2,3]: ax[0,i].axis('off')
    
    ax[0,0].contourf(x, y, map0[1,:,:],levels=[np.nanmin(map0[1,:,:]), alpha, ], 
                        hatches=[density*'/'],
                        extend='lower', alpha = 0)
    
    
    map1=ma.zeros((2,len(ds.lat),len(ds.lon)))
    pilih_nino   =[0,1,2,2,2,3,4] 
    pilih_model=[0,1,3,4,5,6,8,9] 
    
    
    nn=0
    r0=1
    #for n in pilih_nino:
    for i in pilih_model:
        print(model_names[i])
        if i==0: 
        #if i<10: #dmi obs only
            ir5=ir5_obs
        else: 
            print(pilih_nino[nn-1],i)
            #tes for data 3=> 2
            dsx = xr.open_dataset(filepath[pilih_nino[nn-1]]) 
            #nn-1 agar saat nn=1 hasilnya nino file tos ke 0 
            try:
                tos_w = dsx.where(
                        (dsx.lat < 10) & (dsx.lat > -10) & 
                        (dsx.lon > 50) & (dsx.lon < 70), drop=True)
            #GFDL pakai rlat,rlon             
            except:
                try:
                    tos_w = dsx.where(
                        (dsx.rlat < 10) & (dsx.rlat > -10) & 
                        (dsx.rlon > 50) & (dsx.rlon < 70), drop=True)
                except:
                    #ini tidak perlu
                    print('i,j terpakai')
                    tos_w = dsx.where(
                        (dsx.j < 10) & (dsx.j > -10) & 
                        (dsx.i > 50) & (dsx.i < 70), drop=True)   
            try:
                tos_e = dsx.where(
                        (dsx.lat < 0) & (dsx.lat > -10) & 
                        (dsx.lon > 90) & (dsx.lon < 110), drop=True)         
            except:
                try:
                    tos_e = dsx.where(
                        (dsx.rlat < 0) & (dsx.rlat > -10) & 
                        (dsx.rlon > 90) & (dsx.rlon < 110), drop=True) 
                except:
                    print('i,j terpakai')
                    tos_e = dsx.where(
                        (dsx.j < 0) & (dsx.j > -10) & 
                        (dsx.i > 90) & (dsx.i < 110), drop=True) 
            
            tos_nino34=tos_w 
            
            gb = tos_nino34.tos.groupby('time.month')
            tos_nino34_anom = gb - gb.mean(dim='time')
            #print(tos_nino34_anom)
            try: index_nino34 = tos_nino34_anom.mean(dim=['rlat', 'rlon'])
            except:
                try: index_nino34 = tos_nino34_anom.mean(dim=['lat', 'lon'])
                except: index_nino34 = tos_nino34_anom.mean(dim=['i', 'j'])

            ir5_w = index_nino34.rolling(time=5, center=True).mean()
            
            tos_nino34=tos_e 
            
            gb = tos_nino34.tos.groupby('time.month')
            tos_nino34_anom = gb - gb.mean(dim='time')
            #print(tos_nino34_anom)
            try: index_nino34 = tos_nino34_anom.mean(dim=['rlat', 'rlon'])
            except:
                try: index_nino34 = tos_nino34_anom.mean(dim=['lat', 'lon'])
                except: index_nino34 = tos_nino34_anom.mean(dim=['i', 'j'])

            ir5_e = index_nino34.rolling(time=5, center=True).mean()
            
            ir5=ir5_w -ir5_e
        
        dsi = xr.DataArray(model_datasets[i].values,
        coords={'time': obs_dataset.times,
        'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
        dims=["time", "lat", "lon"]) 
        
        if i<5:
            m = Basemap(ax=ax[1,nn], projection ='cyl', llcrnrlat = lat_min, urcrnrlat = lat_max,
                llcrnrlon = lon_min, urcrnrlon = lon_max, resolution = 'l', fix_aspect=False)
            m.drawcoastlines(linewidth=1)
            m.drawcountries(linewidth=1)
            m.drawstates(linewidth=0.5, color='w')
                
            map1=ma.zeros((2,len(ds.lat),len(ds.lon)))
            x0=ir5.values
            for ii in np.arange(len(ds.lat)-tes):
                print(ii)
                for jj in np.arange(len(ds.lon)-tes):
                    d=dsi[:,ii,jj]
                    md = d.groupby('time.month')
                    d_anom = md - md.mean(dim='time')
                    pr5 = d_anom.rolling(time=5, center=True).mean()
                    if i in [4,5,6]: 
                        pr5=np.delete(pr5,-1) #ini khusus dmi dari model
                        y0=pr5.values 
                        #print(len(pr5))
                        #print(y0)
                    else: 
                        y0=pr5.values
                        #print(y0)
                    #print(x0.shape)
                    #print(y0.shape)
                    #print(x0)
                    #print(y0)
                    bad = ~np.logical_or(np.isnan(x0), np.isnan(y0))
                    x1=np.compress(bad, x0) 
                    y1=np.compress(bad, y0)
                    #print(x,y)
                    #print(x.shape,y.shape)
                    if x1.shape==(0,) or y1.shape==(0,):
                        R='nan'
                        pv='nan'
                    else:
                        R=pearsonr(x1, y1)[0]*-1
                        pv=pearsonr(x1, y1)[1]
                    map1[0,ii,jj]=R
                    map1[1,ii,jj]=pv
                    
                    
                    if i==0: m02=map1.flatten()
                    
            m1=map1.flatten()
            bad = ~np.logical_or(np.isnan(m0), np.isnan(m1))
            x1=np.compress(bad, m0) 
            y1=np.compress(bad, m1)
            
            sd1=x1.std() #(skipna=None)
            #print(sd1)
            sd2=y1.std()
            s=sd2/sd1
            
            c,pp=pearsonr(x1.flatten() , y1.flatten())        
            
            #T= (4*(1+c)**4)/(((s+1/s)**2)*((1+r0)**4))
            # Tanpa pakai **4 T naik dikit
            tt= (4*(1+c))/(((s+1/s)**2)*((1+r0)))
            #T.append(round(tt,2))#taylor score
            
            #m02
            bad = ~np.logical_or(np.isnan(m02), np.isnan(m1))
            x1=np.compress(bad, m02) 
            y1=np.compress(bad, m1)
            
            sd1=x1.std() #(skipna=None)
            #print(sd1)
            sd2=y1.std()
            s=sd2/sd1
            
            c,pp=pearsonr(x1.flatten() , y1.flatten())        
            
            #T= (4*(1+c)**4)/(((s+1/s)**2)*((1+r0)**4))
            # Tanpa pakai **4 T naik dikit
            tt2= (4*(1+c))/(((s+1/s)**2)*((1+r0)))
                    
                    
            #print('map1=',map1)
            max = ax[1,nn].contourf(x,y,map1[0,:,:])
            #ax[0,nn].set_title(model_names[i])
            ax[1,nn].set_title(model_names[i]+'('+'%.2f'%tt+')'+'('+'%.2f'%tt2+')', pad=5,fontsize=10)
            #ax[0,0].set_yticks([-10,0,10,20])
            #ax[0,1+i].set_xticks([90,100,110,120,130,140])
            
            ax[1,nn].contourf(x, y, map1[1,:,:],levels=[np.nanmin(map1[1,:,:]), alpha, ], 
                        hatches=[density*'/'],
                        extend='lower', alpha = 0)
            
        else:
            m = Basemap(ax=ax[2,nn-4], projection ='cyl', llcrnrlat = lat_min, urcrnrlat = lat_max,
                llcrnrlon = lon_min, urcrnrlon = lon_max, resolution = 'l', fix_aspect=False)
            m.drawcoastlines(linewidth=1)
            m.drawcountries(linewidth=1)
            m.drawstates(linewidth=0.5, color='w')
                
            map1=ma.zeros((2, len(ds.lat),len(ds.lon)))
            x0=ir5.values
            for ii in np.arange(len(ds.lat)-tes):
                print(ii)
                for jj in np.arange(len(ds.lon)-tes):
                    d=dsi[:,ii,jj]
                    md = d.groupby('time.month')
                    d_anom = md - md.mean(dim='time')
                    pr5 = d_anom.rolling(time=5, center=True).mean()
                    if i in [4,5,6]: 
                        pr5=np.delete(pr5,-1) #ini khusus dmi dari model
                        y0=pr5.values 
                        #print(len(pr5))
                        #print(y0)
                    else: 
                        y0=pr5.values
                    #print(x0.shape)
                    #print(y0.shape)
                    #print(x0)
                    #print(y0)
                    bad = ~np.logical_or(np.isnan(x0), np.isnan(y0))
                    x1=np.compress(bad, x0) 
                    y1=np.compress(bad, y0)
                    #print(x,y)
                    #print(x.shape,y.shape)
                    if x1.shape==(0,) or y1.shape==(0,):
                        R='nan'
                        pv='nan'
                    else:
                        R=pearsonr(x1, y1)[0]*-1
                        pv=pearsonr(x1, y1)[1]
                    map1[0,ii,jj]=R
                    map1[1,ii,jj]=pv
            
            m1=map1.flatten()
            bad = ~np.logical_or(np.isnan(m0), np.isnan(m1))
            x1=np.compress(bad, m0) 
            y1=np.compress(bad, m1)
            
            sd1=x1.std() #(skipna=None)
            #print(sd1)
            sd2=y1.std()
            s=sd2/sd1
            
            c,pp=pearsonr(x1.flatten() , y1.flatten())        
            
            #T= (4*(1+c)**4)/(((s+1/s)**2)*((1+r0)**4))
            # Tanpa pakai **4 T naik dikit
            tt= (4*(1+c))/(((s+1/s)**2)*((1+r0)))
            #T.append(round(tt,2))#taylor score
            #print('map1=',map1)
            
            #m02
            bad = ~np.logical_or(np.isnan(m02), np.isnan(m1))
            x1=np.compress(bad, m02) 
            y1=np.compress(bad, m1)
            
            sd1=x1.std() #(skipna=None)
            #print(sd1)
            sd2=y1.std()
            s=sd2/sd1
            
            c,pp=pearsonr(x1.flatten() , y1.flatten())        
            
            #T= (4*(1+c)**4)/(((s+1/s)**2)*((1+r0)**4))
            # Tanpa pakai **4 T naik dikit
            tt2= (4*(1+c))/(((s+1/s)**2)*((1+r0)))
           
            max = ax[2,nn-4].contourf(x,y,map1[0,:,:])
            
            ax[2,nn-4].contourf(x, y, map1[1,:,:],levels=[np.nanmin(map1[1,:,:]), alpha, ], 
                        hatches=[density*'/'],
                        extend='lower', alpha = 0)
            #ax[1,nn-4].set_title(model_names[i])
            ax[2,nn-4].set_title(model_names[i]+'('+'%.2f'%tt+')'+'('+'%.2f'%tt2+')', pad=5,fontsize=10)
            #ax[0,0].set_yticks([-10,0,10,20])
            #ax[0,1+i].set_xticks([90,100,110,120,130,140])
            ax[2,nn-4].set_xticks([100,120,140])
            ax[2,nn-4].xaxis.set_tick_params(labelsize=7)
        nn=nn+1    
    
    ax[0,0].set_yticks([-10,0,10,20])
    ax[1,0].set_yticks([-10,0,10,20])
    ax[2,0].set_yticks([-10,0,10,20])
    ax[0,0].yaxis.set_tick_params(labelsize=7)
    ax[1,0].yaxis.set_tick_params(labelsize=7)
    ax[2,0].yaxis.set_tick_params(labelsize=7)
    
    plt.subplots_adjust(hspace=.25,wspace=.12)
    #cax = fig.add_axes([0.91, 0.5, 0.015, 0.4])
    cax = fig.add_axes([0.4, 0.7, 0.4, 0.04]) #horisontal
    #plt.colorbar(max, cax = cax) 
    cbar=plt.colorbar(max, cax = cax, extend='both', orientation='horizontal') 
    cbar.ax.tick_params(labelsize=7) 
    
    #agar tidak error saat saving buang titik pada alpha  
    #extensiom conflict
    alpha2=str(alpha)
    alpha2=alpha2.replace('.', '')
   
    file_name='Corr_IOD_tos_pv'+alpha2+reg
    fig.savefig(workdir+file_name,dpi=300,bbox_inches='tight')
    plt.show()

def nino_eof(obs_dataset, obs_name, model_datasets, model_names, workdir):
  
    import xarray as xr
    import pandas as pd
    from eofs2.xarray import Eof
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    
    fig, ax = plt.subplots(nrows=3, ncols=4 ,figsize=(8,6))
    
    d= pd.read_excel('D:/Disertasi3/enso_mon_1981-2005.xlsx')
    nino_obs = xr.DataArray( d['Value'], coords={'time': obs_dataset.times})
    gb = nino_obs.groupby('time.month')
    nino_obs = gb - gb.mean(dim='time')
    ir5_obs = nino_obs.rolling(time=5, center=True).mean()
    
    filepath=[
            'D:/data1/tos_Omon_CNRM-CM5_historical_r1i1p1_198101-200512.nc',
            'D:/data1/tos_Omon_IPSL-CM5A-LR_historicalNat_r1i1p1_1981-2005.nc',
            'D:/data1/tos_Omon_HadGEM2-ES_esmHistorical_r1i1p1_1981-2005_11.nc',
            'D:/data1/tos_Omon_NorESM1-M_historical_r1i1p1_198101-200512.nc',
            'D:/data1/4tos_Omon_GFDL-ESM2M_historical_r1i1p1_198101-200512.nc'
            ]
    dsx = xr.open_dataset(filepath[1])        
    
   
    try:
        tos_nino34 = dsx.where(
                (dsx.lat < 5) & (dsx.lat > -5) & 
                (dsx.lon > 190) & (dsx.lon < 240), drop=True)
            
    except:
     
        tos_nino34 = dsx.where(
                (dsx.rlat < 5) & (dsx.rlat > -5) & 
                (dsx.rlon > 190) & (dsx.rlon < 240), drop=True)
    print(tos_nino34)
    ds=tos_nino34
    climatology_mean = ds.groupby("time.month").mean("time")
    climatology_std = ds.groupby("time.month").std("time")
    
    ds=ds.drop_vars('bnds')
    print(ds)
    exit()
    ds = xr.apply_ufunc(
        lambda x, m, s: 
        (x - m) / s,
        ds.groupby("time.month"),
        climatology_mean,
        climatology_std,
    )
    
    #ds=ds.drop_vars('time')
    #print('3 ds.min_max=',ds.min().data,ds.max().data)
    #ds=ds.rename({'month':'time'}) #if climatology anomalies used
    #ds=ds.assign_coords(time=np.arange(300)) #1981,2006))
    ds=ds.rename('SEAR')
    solver = Eof(ds)
    fn='monthly'
    #print('ds=',ds)
    
    eof = solver.eofsAsCorrelation(neofs=1)
    pc = solver.pcs(npcs=1, pcscaling=0) #1 with scaling
    #print('eof=',eof)
    #print('solver.varianceFraction=', solver.varianceFraction())
    
    #print('pc=',pc)
    
      
    clevs = np.linspace(0.5, 1, 5)
    
    for i in [0]: #len(eof):
        plt.figure()
        ax = plt.axes(projection=ccrs.PlateCarree())
        fill = eof[i].plot.contourf(ax=ax, 
                             #levels=clevs, 
                             #title=musim,
                             #add_colorbar=False, 
                             transform=ccrs.PlateCarree(),
                             #colors='brown'
                             )
        
        #plt.clabel(fill, inline=True, fontsize=8)
        #ax.set_title(musim)
        
        ax.add_feature(cfeature.LAND, facecolor='w', edgecolor='k')
        #cb = plt.colorbar(fill, orientation='horizontal')
        #cb.set_label('correlation coefficient', fontsize=12)
        ax.set_title('EOF'+str(i)+' expressed as correlation_'+fn, pad=20, fontsize=16)
        #ax.set_xlabel(fn)
        plt.savefig(workdir+reg+'_tos_EOF_'+fn,dpi=300,bbox_inches='tight')
    
    for i in [0]: #np.arange(len(pc)-1):
        print('i=', i, len(pc))
        plt.figure()
        pc[:, i].plot(color='b', linewidth=2)
        ax = plt.gca()
        ax.axhline(0, color='k')
        #ax.set_ylim(-3, 3)
        ax.set_xlabel('month')
        ax.set_ylabel('Normalized Units')
        ax.set_title('PC'+str(i)+' Time Series_'+fn, pad=20, fontsize=16)
       
        plt.savefig(workdir+reg+'_tos_PCA_ts_'+fn,dpi=300,bbox_inches='tight')
    
    plt.show()

def nino_corr(obs_dataset, obs_name, model_datasets, model_names, workdir):
    #dmi obs vs rainfall model
    #dmi cmip5 model vs rainfall model 
 
    import xarray as xr
    import pandas as pd
    fig, ax = plt.subplots(nrows=3, ncols=4 ,figsize=(8,6))
    
    d= pd.read_excel('D:/Disertasi3/enso_mon_1981-2005.xlsx')
    nino_obs = xr.DataArray( d['Value'], coords={'time': obs_dataset.times})
    gb = nino_obs.groupby('time.month')
    nino_obs = gb - gb.mean(dim='time')
    ir5_obs = nino_obs.rolling(time=5, center=True).mean()
    
    filepath=[
            'D:/data1/tos_Omon_CNRM-CM5_historical_r1i1p1_198101-200512.nc',
            'D:/data1/tos_Omon_IPSL-CM5A-LR_historicalNat_r1i1p1_1981-2005.nc',
            'D:/data1/tos_Omon_HadGEM2-ES_esmHistorical_r1i1p1_1981-2005_11.nc',
            'D:/data1/tos_Omon_NorESM1-M_historical_r1i1p1_198101-200512.nc',
            'D:/data1/4tos_Omon_GFDL-ESM2M_historical_r1i1p1_198101-200512.nc'
            ]
   
    ds = xr.DataArray(obs_dataset.values,
    coords={'time': obs_dataset.times,
    'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
    dims=["time", "lat", "lon"])
  
    lat_min = ds.lat.min()
    lat_max = ds.lat.max()
    lon_min = ds.lon.min()
    lon_max = ds.lon.max()
   
    x,y = np.meshgrid(ds.lon, ds.lat)
    
    from scipy.stats import pearsonr
    
    
    tes=0  #len(ds.lat) 161
    alpha=0.01
    density=4
    
    print('len(ds.lat)', len(ds.lat))
    print(obs_name)
    
    x0=ir5_obs.values
    map0=ma.zeros((2, len(ds.lat),len(ds.lon)))
    for ii in np.arange(len(ds.lat)-tes):
        print(ii)
        for jj in np.arange(len(ds.lon)-tes):
            d=ds[:,ii,jj]
            md = d.groupby('time.month')
            d_anom = md - md.mean(dim='time')
            pr5 = d_anom.rolling(time=5, center=True).mean()
            y0=pr5.values
            #print(y0)
            #print(x0.shape)
            #print(y0.shape)
            #print(x0)
            #print(y0)
            bad = ~np.logical_or(np.isnan(x0), np.isnan(y0))
            x1=np.compress(bad, x0) 
            y1=np.compress(bad, y0)
            #print(x,y)
            #print(x.shape,y.shape)
            if x1.shape==(0,) or y1.shape==(0,):
                R='nan'
                pv='nan'
            else:
                R=pearsonr(x1, y1)[0]*-1
                pv=pearsonr(x1, y1)[1]
                #print('pv=',pv)
            map0[0,ii,jj]=R
            map0[1,ii,jj]=pv
            
            
    m0=map0.flatten()
    
    m = Basemap(ax=ax[0,0], projection ='cyl', llcrnrlat = lat_min, urcrnrlat = lat_max,
                llcrnrlon = lon_min, urcrnrlon = lon_max, resolution = 'l', fix_aspect=False)
    m.drawcoastlines(linewidth=1)
    m.drawcountries(linewidth=1)
    m.drawstates(linewidth=0.5, color='w')
    
    max = ax[0,0].contourf(x,y,map0[0,:,:])
    #ax[0,nn].set_title(model_names[i])
    ax[0,0].set_title(obs_name, pad=3,fontsize=10)
    for i in [1,2,3]: ax[0,i].axis('off')
    
    ax[0,0].contourf(x, y, map0[1,:,:],levels=[np.nanmin(map0[1,:,:]), alpha, ], 
                        hatches=[density*'/'],
                        extend='lower', alpha = 0)
    
    
    map1=ma.zeros((2,len(ds.lat),len(ds.lon)))
    pilih_nino   =[0,1,2,2,2,3,4] 
    pilih_model=[0,1,3,4,5,6,8,9] 
    
    
    nn=0
    r0=1
    #for n in pilih_nino:
    for i in pilih_model:
        print(model_names[i])
        if i==0: 
        #if i<10: #dmi obs only
            ir5=ir5_obs
        else: 
            print(pilih_nino[nn-1],i)
            #tes for data 3=> 2
            dsx = xr.open_dataset(filepath[pilih_nino[nn-1]]) 
            #nn-1 agar saat nn=1 hasilnya nino file tos ke 0 
            try:
                tos_nino34 = dsx.where(
                        (dsx.lat < 5) & (dsx.lat > -5) & 
                        (dsx.lon > 190) & (dsx.lon < 240), drop=True)
                    
            except:
             
                tos_nino34 = dsx.where(
                        (dsx.rlat < 5) & (dsx.rlat > -5) & 
                        (dsx.rlon > 190) & (dsx.rlon < 240), drop=True)
             
            
             
            gb = tos_nino34.tos.groupby('time.month')
            tos_nino34_anom = gb - gb.mean(dim='time')
            #print(tos_nino34_anom)
            try: index_nino34 = tos_nino34_anom.mean(dim=['rlat', 'rlon'])
            except:
                try: index_nino34 = tos_nino34_anom.mean(dim=['lat', 'lon'])
                except: index_nino34 = tos_nino34_anom.mean(dim=['i', 'j'])

            ir5 = index_nino34.rolling(time=5, center=True).mean()
            
           
        
        dsi = xr.DataArray(model_datasets[i].values,
        coords={'time': obs_dataset.times,
        'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
        dims=["time", "lat", "lon"]) 
        
        if i<5:
            m = Basemap(ax=ax[1,nn], projection ='cyl', llcrnrlat = lat_min, urcrnrlat = lat_max,
                llcrnrlon = lon_min, urcrnrlon = lon_max, resolution = 'l', fix_aspect=False)
            m.drawcoastlines(linewidth=1)
            m.drawcountries(linewidth=1)
            m.drawstates(linewidth=0.5, color='w')
                
            map1=ma.zeros((2,len(ds.lat),len(ds.lon)))
            x0=ir5.values
            for ii in np.arange(len(ds.lat)-tes):
                print(ii)
                for jj in np.arange(len(ds.lon)-tes):
                    d=dsi[:,ii,jj]
                    md = d.groupby('time.month')
                    d_anom = md - md.mean(dim='time')
                    pr5 = d_anom.rolling(time=5, center=True).mean()
                    if i in [4,5,6]: 
                        pr5=np.delete(pr5,-1) #ini khusus dmi dari model
                        y0=pr5.values 
                        #print(len(pr5))
                        #print(y0)
                    else: 
                        y0=pr5.values
                        #print(y0)
                    #print(x0.shape)
                    #print(y0.shape)
                    #print(x0)
                    #print(y0)
                    bad = ~np.logical_or(np.isnan(x0), np.isnan(y0))
                    x1=np.compress(bad, x0) 
                    y1=np.compress(bad, y0)
                    #print(x,y)
                    #print(x.shape,y.shape)
                    if x1.shape==(0,) or y1.shape==(0,):
                        R='nan'
                        pv='nan'
                    else:
                        R=pearsonr(x1, y1)[0]*-1
                        pv=pearsonr(x1, y1)[1]
                    map1[0,ii,jj]=R
                    map1[1,ii,jj]=pv
                    
                    
                    if i==0: m02=map1.flatten()
                    
            m1=map1.flatten()
            bad = ~np.logical_or(np.isnan(m0), np.isnan(m1))
            x1=np.compress(bad, m0) 
            y1=np.compress(bad, m1)
            
            sd1=x1.std() #(skipna=None)
            #print(sd1)
            sd2=y1.std()
            s=sd2/sd1
            
            c,pp=pearsonr(x1.flatten() , y1.flatten())        
            
            #T= (4*(1+c)**4)/(((s+1/s)**2)*((1+r0)**4))
            # Tanpa pakai **4 T naik dikit
            tt= (4*(1+c))/(((s+1/s)**2)*((1+r0)))
            #T.append(round(tt,2))#taylor score
            
            #m02
            bad = ~np.logical_or(np.isnan(m02), np.isnan(m1))
            x1=np.compress(bad, m02) 
            y1=np.compress(bad, m1)
            
            sd1=x1.std() #(skipna=None)
            #print(sd1)
            sd2=y1.std()
            s=sd2/sd1
            
            c,pp=pearsonr(x1.flatten() , y1.flatten())        
            
            #T= (4*(1+c)**4)/(((s+1/s)**2)*((1+r0)**4))
            # Tanpa pakai **4 T naik dikit
            tt2= (4*(1+c))/(((s+1/s)**2)*((1+r0)))
                    
                    
            #print('map1=',map1)
            max = ax[1,nn].contourf(x,y,map1[0,:,:])
            #ax[0,nn].set_title(model_names[i])
            ax[1,nn].set_title(model_names[i]+'('+'%.2f'%tt+')'+'('+'%.2f'%tt2+')', pad=5,fontsize=10)
            #ax[0,0].set_yticks([-10,0,10,20])
            #ax[0,1+i].set_xticks([90,100,110,120,130,140])
            
            ax[1,nn].contourf(x, y, map1[1,:,:],levels=[np.nanmin(map1[1,:,:]), alpha, ], 
                        hatches=[density*'/'],
                        extend='lower', alpha = 0)
            
        else:
            m = Basemap(ax=ax[2,nn-4], projection ='cyl', llcrnrlat = lat_min, urcrnrlat = lat_max,
                llcrnrlon = lon_min, urcrnrlon = lon_max, resolution = 'l', fix_aspect=False)
            m.drawcoastlines(linewidth=1)
            m.drawcountries(linewidth=1)
            m.drawstates(linewidth=0.5, color='w')
                
            map1=ma.zeros((2, len(ds.lat),len(ds.lon)))
            x0=ir5.values
            for ii in np.arange(len(ds.lat)-tes):
                print(ii)
                for jj in np.arange(len(ds.lon)-tes):
                    d=dsi[:,ii,jj]
                    md = d.groupby('time.month')
                    d_anom = md - md.mean(dim='time')
                    pr5 = d_anom.rolling(time=5, center=True).mean()
                    if i in [4,5,6]: 
                        pr5=np.delete(pr5,-1) #ini khusus dmi dari model
                        y0=pr5.values 
                        #print(len(pr5))
                        #print(y0)
                    else: 
                        y0=pr5.values
                    #print(x0.shape)
                    #print(y0.shape)
                    #print(x0)
                    #print(y0)
                    bad = ~np.logical_or(np.isnan(x0), np.isnan(y0))
                    x1=np.compress(bad, x0) 
                    y1=np.compress(bad, y0)
                    #print(x,y)
                    #print(x.shape,y.shape)
                    if x1.shape==(0,) or y1.shape==(0,):
                        R='nan'
                        pv='nan'
                    else:
                        R=pearsonr(x1, y1)[0]*-1
                        pv=pearsonr(x1, y1)[1]
                    map1[0,ii,jj]=R
                    map1[1,ii,jj]=pv
            
            m1=map1.flatten()
            bad = ~np.logical_or(np.isnan(m0), np.isnan(m1))
            x1=np.compress(bad, m0) 
            y1=np.compress(bad, m1)
            
            sd1=x1.std() #(skipna=None)
            #print(sd1)
            sd2=y1.std()
            s=sd2/sd1
            
            c,pp=pearsonr(x1.flatten() , y1.flatten())        
            
            #T= (4*(1+c)**4)/(((s+1/s)**2)*((1+r0)**4))
            # Tanpa pakai **4 T naik dikit
            tt= (4*(1+c))/(((s+1/s)**2)*((1+r0)))
            #T.append(round(tt,2))#taylor score
            #print('map1=',map1)
            
            #m02
            bad = ~np.logical_or(np.isnan(m02), np.isnan(m1))
            x1=np.compress(bad, m02) 
            y1=np.compress(bad, m1)
            
            sd1=x1.std() #(skipna=None)
            #print(sd1)
            sd2=y1.std()
            s=sd2/sd1
            
            c,pp=pearsonr(x1.flatten() , y1.flatten())        
            
            #T= (4*(1+c)**4)/(((s+1/s)**2)*((1+r0)**4))
            # Tanpa pakai **4 T naik dikit
            tt2= (4*(1+c))/(((s+1/s)**2)*((1+r0)))
           
            max = ax[2,nn-4].contourf(x,y,map1[0,:,:])
            
            ax[2,nn-4].contourf(x, y, map1[1,:,:],levels=[np.nanmin(map1[1,:,:]), alpha, ], 
                        hatches=[density*'/'],
                        extend='lower', alpha = 0)
            #ax[1,nn-4].set_title(model_names[i])
            ax[2,nn-4].set_title(model_names[i]+'('+'%.2f'%tt+')'+'('+'%.2f'%tt2+')', pad=5,fontsize=10)
            #ax[0,0].set_yticks([-10,0,10,20])
            #ax[0,1+i].set_xticks([90,100,110,120,130,140])
            ax[2,nn-4].set_xticks([100,120,140])
            ax[2,nn-4].xaxis.set_tick_params(labelsize=7)
        nn=nn+1    
    
    ax[0,0].set_yticks([-10,0,10,20])
    ax[1,0].set_yticks([-10,0,10,20])
    ax[2,0].set_yticks([-10,0,10,20])
    ax[0,0].yaxis.set_tick_params(labelsize=7)
    ax[1,0].yaxis.set_tick_params(labelsize=7)
    ax[2,0].yaxis.set_tick_params(labelsize=7)
    
    plt.subplots_adjust(hspace=.25,wspace=.12)
    #cax = fig.add_axes([0.91, 0.5, 0.015, 0.4])
    cax = fig.add_axes([0.4, 0.7, 0.4, 0.04]) #horisontal
    #plt.colorbar(max, cax = cax) 
    cbar=plt.colorbar(max, cax = cax, extend='both', orientation='horizontal') 
    cbar.ax.tick_params(labelsize=9) 
    
    #agar tidak error saat saving buang titik pada alpha  
    #extensiom conflict
    alpha2=str(alpha)
    alpha2=alpha2.replace('.', '')
   
    file_name='Corr_nino_tos_pv'+alpha2+reg
    fig.savefig(workdir+file_name,dpi=300,bbox_inches='tight')
    plt.show()


def enso_partial_corr(obs_dataset, obs_name, model_datasets, model_names, workdir):
        
    import xarray as xr     
    import pandas as pd  
    import math
    '''
    #partial correlation r
    #p=precipitation  #n=Nino3.4  #i=IOD
    rpn.i = rpn - (rpi*rin) / (sqrt((1-rpi**2))*sqrt((1-rin**2)))
    '''
    #partial correlation r 
    #install and import pingouin package 
    from pingouin.correlation import _correl_pvalue
    from scipy.stats import pearsonr
    
    #korelasi hujan dan nino34 index disini tidak dikali negatif
    #maka jika hasil - show hal yg berlawanan hujan - SST +
    
    d= pd.read_excel('D:/Disertasi3/enso_mon_1981-2005.xlsx')
    d2= pd.read_excel('D:/Disertasi3/dmi_1981-2005.xlsx')
    
    ds1 = xr.DataArray( d['Value'], coords={'time': obs_dataset.times})
    nino_obs=ds1.groupby('time.season')
    
    ds2 = xr.DataArray( d2['Value'], coords={'time': obs_dataset.times})
    iod_obs=ds2.groupby('time.season')
    
    #korelasi keduanya pada tiap musim
    #bandingkan dgn GCMs
    
    kor=pearsonr(ds1.values,ds2.values)[0]
    korp=pearsonr(ds1.values,ds2.values)[1]
    print('kor=',kor)
    kor2=[]
    kor2p=[]
    #p p-value
    '''
    for musim in ['DJF','MAM','JJA','SON']:
        kor2.append(pearsonr(nino_obs[musim],iod_obs[musim])[0])
        kor2p.append(pearsonr(nino_obs[musim],iod_obs[musim])[1])
        
    print(kor2)
    print(kor2p)
    musim=['DJF','MAM','JJA','SON']
    plt.scatter(musim,kor2, label = 'seasonal')
    plt.scatter(musim,kor2p, label = 'p-value')
    plt.axhline(kor, linestyle = 'dashed', color='black', label = 'annual')
    plt.axhline(korp, linestyle = 'dashed', color='orange', label = 'p-value')
    plt.ylabel('Correlation SSTA Nino3.4 vs DMI')
    plt.legend(bbox_to_anchor=(0.75, .75), loc='best', prop={'size':8.5}, frameon=True, handletextpad=1) 
    plt.show()
    figname='obs_corr_nino-iod'
    plt.savefig(workdir+figname)
    exit()
    '''
    
    
    #penjumlahan ?? boleh ??  index nino34 + MDI
    mix=xr.DataArray(ds1.values+ds2.values, coords={'time': obs_dataset.times})
    mix=mix.groupby('time.season')
    #print('mix',mix)
    
    ds = xr.DataArray(obs_dataset.values,
    coords={'time': obs_dataset.times,
    'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
    dims=["time", "lat", "lon"])
    #print(ds)
    
    musim='DJF'
    
    '''
    #plot ts
    fig, ax = plt.subplots(nrows=2, ncols=1 ,figsize=(6,4))
    
   
    mds = ds.groupby('time.month')
    ds_anom = mds - mds.mean(dim='time')
  
    ds_anom = ds_anom.mean(dim=['lat', 'lon'])
    ds_anom=ds_anom.groupby('time.season')
    ras=ds_anom[musim]
    print(len(ras))
    print(len(nino_obs[musim]))
  
    
    years = ras.time.sel(time=slice('1981-01-01','2005-12-31')).values
    print(years)
    ax[0].plot(years, nino_obs[musim])
    ax[1].plot(years, ras)
    
    #ax[0].plot(nino_obs[musim])
    #ax[1].plot(ras)
    #ax[1].set_xticks([1985,1995, 2005])
    #ax[1].set_xticklabels(years,rotation=-90)
    ax[0].set_title('Nino34 index anomaly')
    ax[1].set_title('Rainfall anomaly')
    
    plt.show()
    
    exit()
    '''
    lat_min = ds.lat.min()
    lat_max = ds.lat.max()
    lon_min = ds.lon.min()
    lon_max = ds.lon.max()
   
    fig, ax = plt.subplots(nrows=2, ncols=2 ,figsize=(6,4))
    
    
    x,y = np.meshgrid(ds.lon, ds.lat)
    
    
    #OBS
    
    mapx=ma.zeros((len(ds.lat),len(ds.lon)))
    mapxx=ma.zeros((len(ds.lat),len(ds.lon)))
    map1=ma.zeros((len(ds.lat),len(ds.lon)))
    map2=ma.zeros((len(ds.lat),len(ds.lon)))
    map22=ma.zeros((len(ds.lat),len(ds.lon)))
    map3=ma.zeros((len(ds.lat),len(ds.lon)))
    map4=ma.zeros((len(ds.lat),len(ds.lon)))
    map44=ma.zeros((len(ds.lat),len(ds.lon)))
    map11=ma.zeros((len(ds.lat),len(ds.lon)))
    map33=ma.zeros((len(ds.lat),len(ds.lon)))
        
   
    #print('mix',mix)
    #exit()
    #ax[2,1].axis('off')
    for i in np.arange(2):
        for j in np.arange(2):
            m = Basemap(ax=ax[i,j], projection ='cyl', llcrnrlat = lat_min, urcrnrlat = lat_max,
                llcrnrlon = lon_min, urcrnrlon = lon_max, resolution = 'l', fix_aspect=False)
            m.drawcoastlines(linewidth=1)
            m.drawcountries(linewidth=1)
            m.drawstates(linewidth=0.5, color='w')
    '''        
    m = Basemap(ax=ax[2,0], projection ='cyl', llcrnrlat = lat_min, urcrnrlat = lat_max,
                llcrnrlon = lon_min, urcrnrlon = lon_max, resolution = 'l', fix_aspect=False)
    m.drawcoastlines(linewidth=1)
    m.drawcountries(linewidth=1)
    m.drawstates(linewidth=0.5, color='w')
    '''
    
    
    tes=0 #hitung dikurangi 50 step untuk tes
    #cmap='viridis'
    cmap=None
    
    #p-value dengan pingouin
    n = obs_dataset.times.shape[0]  # Number of samples
    num_var=3
    k1 = num_var - 2  # Number of covariates ==> variable
    alpha=0.05
    #-------------------------
    
    vmax=5
    vmin=-5
    #levels = np.linspace(vmin,vmax,6)
    levels=np.linspace(vmin,vmax,11)/10
    #levels=np.arange(6)
    norm = plt.Normalize(vmin, vmax)
    
    x0=nino_obs[musim].values
    x02=iod_obs[musim].values
    mix=mix[musim].values
    
    for i in np.arange(len(ds.lat)-tes):
        print (i)
        for j in np.arange(len(ds.lon)-tes):
            d=ds[:,i,j]
            #hitung anomaly
            md = d.groupby('time.month')          
            d_anom = md - md.mean(dim='time')
            pr=d_anom .groupby('time.season')
           
            #Nan error
            y0=pr[musim].values      
            bad = ~np.logical_or(np.isnan(x0), np.isnan(y0))
            x1=np.compress(bad, x0)
            x12=np.compress(bad, x02)
            mix1=np.compress(bad, mix)
            y1=np.compress(bad, y0)
            #jika panjang data y1 0 maka isi nan
            #boleh juga jika panjang data < 2 juga nan
            #karna minimal korelasi 3 data
            if x1.shape==(0,) or y1.shape==(0,):
                rpn='nan'
                rpni='nan'
                rpnic='nan'
                rpi='nan'
                rpin='nan'
                rpinc='nan'
                rin='nan'
                rinc='nan'                
                rpnc='nan'
                rpic='nan'
                mx='nan'
                mxc='nan'
            else:
                rpn=pearsonr(x1, y1)[0]
                rpnc=pearsonr(x1, y1)[1]
                #c untuk p-value
                
                rpi=pearsonr(x12, y1)[0]
                rpic=pearsonr(x12, y1)[1]
                
                #mx=pearsonr(mix1, y1)[0]
                #mxc=pearsonr(mix1, y1)[1]
                
                # if rpnc <=alpha:
                    # #print(i,j, ds.lat[i].data,ds.lon[j].data)
                    # x2.append(ds.lon[j].data)
                    # y2.append(ds.lat[i].data)
                
                
                # if rpic <=alpha:
                    # #print(i,j, ds.lat[i].data,ds.lon[j].data)
                    # x3.append(ds.lon[j].data)
                    # y3.append(ds.lat[i].data)          
                
                rin=pearsonr(x12, x1)[0]
                rinc=pearsonr(x12, x1)[1]
                
                #rpn.i = (rpn - rpi*rin)/ (sqrt((1-rpi**2)) * sqrt((1-rin**2)))
                               
                rpni = (rpn - (rpi*rin)) / (math.sqrt((1-rpi**2)) * math.sqrt((1-rin**2)))    
                #print(rpn,rpi,rin,rpni)
                #print(round(rpn/rpni,2))
                
                #penenetuan p-value pada partial correlation
                rpnic=_correl_pvalue(rpni, n, k=k1, alternative="two-sided")
                #print('cek', n,rpnic)
                
                
                rpin = (rpi - (rpn*rin)) / (math.sqrt((1-rpn**2))*math.sqrt((1-rin**2)))                    
                #print(rpi,rpn,rin,rpin)
                #print(round(rpin/rpi,2))
                rpinc=_correl_pvalue(rpin, n, k=k1, alternative="two-sided")
                #print('cek', n,rpnic)
                
            map1[i,j]=rpn
            map11[i,j]=rpnc
            
            map2[i,j]=rpni
            map22[i,j]=rpnic
            
            map3[i,j]=rpi
            map33[i,j]=rpic
            
            map4[i,j]=rpin
            map44[i,j]=rpinc
            
            #mapx[i,j]=mx
            #mapxx[i,j]=mxc
    #print(np.nanmin(map11), np.nanmax(map11))
    #print('map1.shp=',map1.shape)
    density=3
    
    max1 = ax[0,0].contourf(x,y,map1, cmap=cmap, extend='both') #=levels,  norm =norm)
    
    ax[0,0].set_title(obs_name+'_ENSO',fontsize=9)
    #hatch ini ok
    ax[0,0].contourf(x, y, map11,levels=[np.nanmin(map11), alpha, ], 
                        hatches=[density*'/'],
                        extend='lower', alpha = 0)
    #for i in np.arange(len(x2)):
        # ax[0,0].text( x=x2[i], y=y2[i],s='.',alpha=0.2)
    #for i in np.arange(len(x3)):
        # ax[1,0].text( x=x3[i], y=y3[i],s='.',alpha=0.2)
    #for i in np.arange(len(x2)):
    #ini ok
    #ax[0,0].scatter(x2[i], y2[i], marker = '.', s = 0.5, c = 'k', alpha = 0.2)
   
    
    max1 = ax[0,1].contourf(x,y,map2, cmap=cmap,  extend='both') #=levels,  norm =norm)
    ax[0,1].set_title(obs_name+'_pENSO',fontsize=9)
    #ax[0,0].set_ylabel(musim)
    ax[0,1].contourf(x, y, map22,levels=[np.nanmin(map11), alpha, ], 
                        hatches=[density*'/'],
                        extend='lower', alpha = 0)
    
    
    max1 = ax[1,0].contourf(x,y,map3, cmap=cmap, extend='both') #=levels,  norm =norm)
    ax[1,0].set_title(obs_name+'_IOD',fontsize=9)
    
    ax[1,0].contourf(x, y, map33,levels=[np.nanmin(map33), alpha, ], 
                        hatches=[density*'/'],
                        extend='lower', alpha = 0)
    
    max1 = ax[1,1].contourf(x,y,map4, cmap=cmap, extend='both') #=levels,  norm =norm)
    ax[1,1].set_title(obs_name+'_pIOD',fontsize=9)
    ax[1,1].contourf(x, y, map44,levels=[np.nanmin(map11), alpha, ], 
                        hatches=[density*'/'],
                        extend='lower', alpha = 0)
    
    #Mix tidak dipakai dulu... tidak baku
    '''                   
    max1 = ax[2,0].contourf(x,y,mapx, extend='both') #=levels,  norm =norm)
    ax[2,0].set_title('GPCP_mix',fontsize=8)
    
    ax[2,0].contourf(x, y, mapxx,levels=[np.nanmin(mapxx), alpha, ], 
                        hatches=[density*'/'],
                        extend='lower', alpha = 0)
    #ax[2,1].axis('off')                
    '''
    
    '''             
    fs=8
    ax[0,0].set_yticks([-10,0,10,20],fontsize=fs)
    ax[1,0].set_yticks([-10,0,10,20],fontsize=fs)
    
    ax[0,0].set_yticklabels(['10S','0','10N','20N'],fontsize=fs)
    ax[1,0].set_yticklabels(['10S','0','10N','20N'],fontsize=fs)
    
    ax[1,0].set_xticks([100,120,140],fontsize=fs)
    ax[1,0].set_xticklabels(['100E','120E','140E'],fontsize=fs)
    ax[1,1].set_xticks([100,120,140],fontsize=fs)
    ax[1,1].set_xticklabels(['100E','120E','140E'],fontsize=fs)
    
    '''
    
    #ax[2,0].set_xticks([90,100,110,120,130,140],fontsize=fs)
    
    #ax[0,0].yaxis.set_tick_params(labelsize=6)
    #ax[1,0].yaxis.set_tick_params(labelsize=6)
    #ax[2,0].yaxis.set_tick_params(labelsize=6)
    
    #ax[2,0].xaxis.set_tick_params(labelsize=6)
    #ax[1,1].xaxis.set_tick_params(labelsize=6)
    
    
    #ax[n,0].set_ylabel(musim)
    #n=n+1
    cax = fig.add_axes([0.91, 0.15, 0.015, 0.7])
    cax.tick_params(labelsize=6)
    plt.colorbar(max1, cax = cax, extend='both') 
    plt.subplots_adjust(hspace=.2,wspace=.05)
    
    plt.show()
    fig.savefig(workdir+reg+'_2_GPCC_nino34_iod_p'+str(alpha)+musim+'.png',dpi=300,bbox_inches='tight')
    #fig.savefig(workdir+reg+'_nino34_iod_p_tes',dpi=300,bbox_inches='tight')
    exit()
    
    # ini ?? khusus zonal jika ingin obs=2 dan MMEW not included
    #model_datasets=np.delete(model_datasets,[1, -1])
    #model_names=np.delete(model_names,[1, -1])
    #model_datasets=np.delete(model_datasets,[-1])
    #model_names=np.delete(model_names,[-1])
    #Datasets: ['GPCP', 'ERA5', 1'CNRM_a', 2'ECE_b', 3'IPSL_b', 
    #4'HadGEM2_d', 5'HadGEM2_c', 6'HadGEM2_a', 7'MPI_c', 8'NorESM1_d', 
    #9'GFDL_b', 10'MME']
    #MODEL
    n=0
    from scipy.stats import pearsonr
       #nino34 model
    # filepath=[
            # 'D:/data1/tos_Omon_CNRM-CM5_historical_r1i1p1_198101-200512.nc',
            # 'D:/data1/tos_Omon_NorESM1-M_historical_r1i1p1_198101-200512.nc',
            # 'D:/data1/2tos_Omon_GFDL-ESM2M_historical_r1i1p1_198101-200512.nc']
    filepath=[
            'D:/data1/tos_Omon_CNRM-CM5_historical_r1i1p1_198101-200412.nc',
            'D:/data1/2tos_Omon_NorESM1-M_historical_r1i1p1_198101-200412.nc',
            'D:/data1/2tos_Omon_GFDL-ESM2M_historical_r1i1p1_198101-200412.nc']
    #cek=0
    #hapus md selain 0189 
    for musim in ['DJF', 'JJA']:
        for i in np.arange(len(model_datasets)):
            if i==0 or i==1 or i==8 or i==9:  
                print(i)
                print ('model=',model_names[i])
             
                if i>0:
                    f=0
                    if i==8: f=1
                    if i==9: f=2
                    #print('f=',f)
                    dsx = xr.open_dataset(filepath[f])
                    #print(dsx)
                    #slice ini nans
                    #tos_nino34 = ds.sel(i=slice(-5, 5), j=slice(190, 240))
                    try:
                        tos_nino34 = dsx.where(
                                (dsx.lat < 5) & (dsx.lat > -5) & (dsx.lon > 190) & 
                                (dsx.lon < 240), drop=True)
                    except:
                        tos_nino34 = dsx.where(
                                (dsx.lat < 5) & (dsx.lat > -5) & (dsx.lon > -170) & 
                                (dsx.lon < -120), drop=True)

                    
                    gb = tos_nino34.tos.groupby('time.month')
                    tos_nino34_anom = gb - gb.mean(dim='time')
                    #print(tos_nino34_anom)
                    try: index_nino34 = tos_nino34_anom.mean(dim=['rlat', 'rlon'])
                    except: index_nino34 = tos_nino34_anom.mean(dim=['i', 'j'])
                    # print('index_nino34=',index_nino34)
                    s34=index_nino34.groupby('time.season')
               
                
                #model
                dsi = xr.DataArray(model_datasets[i].values,
                coords={'time': obs_dataset.times,
                'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
                dims=["time", "lat", "lon"])  
                
                f2=0
                if i==1: f2=1
                if i==8: f2=2
                if i==9: f2=3
                m = Basemap(ax=ax[n,f2+1], projection ='cyl', llcrnrlat = lat_min, urcrnrlat = lat_max,
                    llcrnrlon = lon_min, urcrnrlon = lon_max, resolution = 'l', fix_aspect=False)
                m.drawcoastlines(linewidth=1)
                m.drawcountries(linewidth=1)
                m.drawstates(linewidth=0.5, color='w')
            
                #x,y = np.meshgrid(ds.lon, ds.lat)
            
                for ii in np.arange(len(ds.lat)-tes):
                    print(ii)
                    for jj in np.arange(len(ds.lon)-tes):
                        d=dsi[:,ii,jj]
                        md = d.groupby('time.month')
                        #print(md)
                        d_anom = md - md.mean(dim='time')
                        #pr5 = d_anom.rolling(time=5, center=True).mean()
                        pr=d_anom .groupby('time.season')
                       
                        ipr=pr[musim]
                        #nt = 12 if not config['season'] else nmon
                        #x0=is34.values  #if not i==0 x0=nino_obs[musim].values   
                        if i==0: x0=nino_obs[musim].values 
                        else: x0=s34[musim].values 
                        y0=ipr.values
                        #print(x0.shape)
                        #print(y0.shape)
                        #print(x0)
                        #print(y0)
                        bad = ~np.logical_or(np.isnan(x0), np.isnan(y0))
                        x1=np.compress(bad, x0) 
                        y1=np.compress(bad, y0)
                        #print(x,y)
                        #print(x.shape,y.shape)
                        if x1.shape==(0,) or y1.shape==(0,):
                            R='nan'
                        else:
                            R=pearsonr(x1, y1)[0]*-1
                            #R=pearsonr(pr5, ir5)[0]*-1
                        map1[ii,jj]=R
                #print('map1=',map1)
                max1 = ax[n,f2+1].contourf(x,y,map1)
                ax[0,f2+1].set_title(model_names[i],fontsize=8)
                ax[n,0].set_ylabel(musim)
           
            #ax[n,0].set_yticks([-10,0,10,20])
            #ax[1,i].set_xticks([90,100,110,120,130,140])
        n=n+1
                
    
    cax = fig.add_axes([0.91, 0.5, 0.02, 0.35])
    cax.tick_params(labelsize=6)
    plt.colorbar(max1, cax = cax) 
    plt.subplots_adjust(hspace=.05,wspace=.05)
    
    file_name='SEA_Corr_nino34_3'
    fig.savefig(workdir+file_name+reg,dpi=300)
                                       

def enso_partial_corr2(obs_dataset, obs_name, model_datasets, model_names, workdir):
    #partial corr pakai pingouin package 
    #corr nino34 index vs rainfall anomaly
    #nino34 index = SST anomaly   
    #nino34 index > 0 elnino ==>  rainfall anomaly < 0 ==> dark blue on map
    #nino34 index < 0 lanina ==>  rainfall anomaly > 0 ==> dark red on map
    
    ### tidak dipilih tahun2 tertentu ENSO/IOD

    import xarray as xr     
    import pandas as pd  
    import math

    #partial correlation r
    #install and import pingouin package 
    import pingouin as pg

    d= pd.read_excel('D:/Disertasi3/enso_mon_1981-2005.xlsx')
    d2= pd.read_excel('D:/Disertasi3/dmi_1981-2005.xlsx')
    
    ds = xr.DataArray( d['Value'],
    coords={'time': obs_dataset.times})
    nino_obs=ds.groupby('time.season')
    
    ds = xr.DataArray( d2['Value'],
    coords={'time': obs_dataset.times})
    iod_obs=ds.groupby('time.season')
   
    
    ds = xr.DataArray(obs_dataset.values,
    coords={'time': obs_dataset.times,
    'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
    dims=["time", "lat", "lon"])
    
    #ds = ds.where((ds.lon > 90) & (ds.lon < 130), drop=True)
                          
    
    #print(ds)
                
    # #ds = ds.groupby('time.year').sum() 
    # mds = ds.groupby('time.month')
    # ds_anom = mds - mds.mean(dim='time')
    # #print(tos_nino34_anom)
    # index_nino34 = ds_anom.mean(dim=['lat', 'lon'])
    
    lat_min = ds.lat.min()
    lat_max = ds.lat.max()
    lon_min = ds.lon.min()
    lon_max = ds.lon.max()
   
    fig, ax = plt.subplots(nrows=2, ncols=2 ,figsize=(6,4))
    
    
    x,y = np.meshgrid(ds.lon, ds.lat)
    
    #OBS
    map1=ma.zeros((len(ds.lat),len(ds.lon)))
    map2=ma.zeros((len(ds.lat),len(ds.lon)))
    map22=ma.zeros((len(ds.lat),len(ds.lon)))
    map3=ma.zeros((len(ds.lat),len(ds.lon)))
    map4=ma.zeros((len(ds.lat),len(ds.lon)))
    map44=ma.zeros((len(ds.lat),len(ds.lon)))
    map11=ma.zeros((len(ds.lat),len(ds.lon)))
    map33=ma.zeros((len(ds.lat),len(ds.lon)))
    #n=0
   
    
    for i in np.arange(2):
        for j in np.arange(2):
            m = Basemap(ax=ax[i,j], projection ='cyl', llcrnrlat = lat_min, urcrnrlat = lat_max,
                llcrnrlon = lon_min, urcrnrlon = lon_max, resolution = 'l', fix_aspect=False)
            m.drawcoastlines(linewidth=1)
            m.drawcountries(linewidth=1)
            m.drawstates(linewidth=0.5, color='w')
  
    musim='DJF'
    
    alpha=0.05
    tes=0 
    density=3
    #cmap='viridis'
    cmap=None
   
    x0=nino_obs[musim].values
    x02=iod_obs[musim].values
    
    for i in np.arange(len(ds.lat)-tes):
        print (i)
        for j in np.arange(len(ds.lon)-tes):
            d=ds[:,i,j]
            md = d.groupby('time.month')
           
            d_anom = md - md.mean(dim='time')
           
            pr=d_anom.groupby('time.season')
            
            y0=pr[musim].values
         
            bad = ~np.logical_or(np.isnan(x0), np.isnan(y0))
            x1=np.compress(bad, x0)     #Nino
            x12=np.compress(bad, x02)   #IOD
            y1=np.compress(bad, y0)     #rainfall
           
            if x1.shape==(0,) or y1.shape==(0,):
                #r=corr, p=precipitation, i=iod, n=nino               
                rpn='nan'
                rpni='nan'
                rpnic='nan'
                
                rpi='nan'
                rpin='nan'
                rpinc='nan'
                
                rin='nan'
                rinc='nan'                
                rpnc='nan'
                rpic='nan'
            
            else:
                df= pd.DataFrame({
                    'Ra': y1,
                    'Ni': x1,
                    'IO': x12
                })
                r=pg.partial_corr(data=df, x='Ni', y='Ra', covar=None)
                rpn=r['r'].values
                rpnc=r['p-val'].values
                
                r=pg.partial_corr(data=df, x='IO', y='Ra', covar=None)
                rpi=r['r'].values
                rpic=r['p-val'].values
                
                r=pg.partial_corr(data=df, x='Ni', y='Ra', covar='IO')
                rpni=r['r'].values
                rpnic=r['p-val'].values
                
                r=pg.partial_corr(data=df, x='IO', y='Ra', covar='Ni')
                rpin=r['r'].values
                rpinc=r['p-val'].values
           
            map1[i,j]=rpn
            map11[i,j]=rpnc
            
            map2[i,j]=rpni
            map22[i,j]=rpnic
            
            map3[i,j]=rpi 
            map33[i,j]=rpic
            
            map4[i,j]=rpin 
            map44[i,j]=rpinc
            
            
    
    print(np.nanmin(map11), np.nanmax(map11))
    
    
    max1 = ax[0,0].contourf(x,y,map1, cmap=cmap, extend='both') #=levels,  norm =norm)
    
    ax[0,0].set_title(obs_name+'_ENSO',fontsize=9)
    #hatch ini ok
    ax[0,0].contourf(x, y, map11,levels=[np.nanmin(map11), alpha, ], 
                        hatches=[density*'/'],
                        extend='lower', alpha = 0)
    #for i in np.arange(len(x2)):
        # ax[0,0].text( x=x2[i], y=y2[i],s='.',alpha=0.2)
    #for i in np.arange(len(x3)):
        # ax[1,0].text( x=x3[i], y=y3[i],s='.',alpha=0.2)
    #for i in np.arange(len(x2)):
    #ini ok
    #ax[0,0].scatter(x2[i], y2[i], marker = '.', s = 0.5, c = 'k', alpha = 0.2)
   
    
    max1 = ax[0,1].contourf(x,y,map2, cmap=cmap,  extend='both') #=levels,  norm =norm)
    ax[0,1].set_title(obs_name+'_pENSO',fontsize=9)
    #ax[0,0].set_ylabel(musim)
    ax[0,1].contourf(x, y, map22,levels=[np.nanmin(map11), alpha, ], 
                        hatches=[density*'/'],
                        extend='lower', alpha = 0)
    
    
    max1 = ax[1,0].contourf(x,y,map3, cmap=cmap, extend='both') #=levels,  norm =norm)
    ax[1,0].set_title(obs_name+'_IOD',fontsize=9)
    
    ax[1,0].contourf(x, y, map33,levels=[np.nanmin(map33), alpha, ], 
                        hatches=[density*'/'],
                        extend='lower', alpha = 0)
    
    max1 = ax[1,1].contourf(x,y,map4, cmap=cmap, extend='both') #=levels,  norm =norm)
    ax[1,1].set_title(obs_name+'_pIOD',fontsize=9)
    ax[1,1].contourf(x, y, map44,levels=[np.nanmin(map11), alpha, ], 
                        hatches=[density*'/'],
                        extend='lower', alpha = 0)
    
    #Mix tidak dipakai dulu... tidak baku
    '''                   
    max1 = ax[2,0].contourf(x,y,mapx, extend='both') #=levels,  norm =norm)
    ax[2,0].set_title('GPCP_mix',fontsize=8)
    
    ax[2,0].contourf(x, y, mapxx,levels=[np.nanmin(mapxx), alpha, ], 
                        hatches=[density*'/'],
                        extend='lower', alpha = 0)
    #ax[2,1].axis('off')                
    '''
    
    '''        
    fs=8
    ax[0,0].set_yticks([-10,0,10,20],fontsize=fs)
    ax[1,0].set_yticks([-10,0,10,20],fontsize=fs)
    
    ax[0,0].set_yticklabels(['10S','0','10N','20N'],fontsize=fs)
    ax[1,0].set_yticklabels(['10S','0','10N','20N'],fontsize=fs)
    
    ax[1,0].set_xticks([100,120,140],fontsize=fs)
    ax[1,0].set_xticklabels(['100E','120E','140E'],fontsize=fs)
    ax[1,1].set_xticks([100,120,140],fontsize=fs)
    ax[1,1].set_xticklabels(['100E','120E','140E'],fontsize=fs)
    '''
    
    
    #ax[2,0].set_xticks([90,100,110,120,130,140],fontsize=fs)
    
    #ax[0,0].yaxis.set_tick_params(labelsize=6)
    #ax[1,0].yaxis.set_tick_params(labelsize=6)
    #ax[2,0].yaxis.set_tick_params(labelsize=6)
    
    #ax[2,0].xaxis.set_tick_params(labelsize=6)
    #ax[1,1].xaxis.set_tick_params(labelsize=6)
    
    
    #ax[n,0].set_ylabel(musim)
    #n=n+1
    cax = fig.add_axes([0.91, 0.15, 0.015, 0.7])
    cax.tick_params(labelsize=6)
    plt.colorbar(max1, cax = cax, extend='both') 
    plt.subplots_adjust(hspace=.2,wspace=.05)
    
    plt.show()
    fig.savefig(workdir+reg+obs_name+str(alpha)+musim+'.png',dpi=300,bbox_inches='tight')
    #fig.savefig(workdir+reg+'_nino34_iod_p_tes',dpi=300,bbox_inches='tight')

def enso_partial_corr3(obs_dataset, obs_name, model_datasets, model_names, workdir):
   
    #dipilih tahun2 tertentu ENSO/IOD

    import xarray as xr     
    import pandas as pd  
    import math
    #partial correlation import pingouin package 
    import pingouin as pg

    d= pd.read_excel('D:/Disertasi3/enso_mon_1981-2005.xlsx')
    d2= pd.read_excel('D:/Disertasi3/dmi_1981-2005.xlsx')
    
    ds = xr.DataArray( d['Value'],
    coords={'time': obs_dataset.times})
    
    #dipilih tahun2 tertentu ENSO/IOD
    years_of_interest = ['1982', '1983', '1987', '1988', '1997', '1998', '2002', '2003']
    sliced_data = []
    for year in years_of_interest:
        year_data = ds.sel(time=slice(f'{year}-01-01', f'{year}-12-31'))
        sliced_data.append(year_data)
    # Combine the sliced data into a new xarray dataset or dictionary
    ds =  xr.concat(sliced_data, dim='time')

    nino_obs=ds.groupby('time.season')
    
    ds = xr.DataArray( d2['Value'],
    coords={'time': obs_dataset.times})
    
    sliced_data = []
    for year in years_of_interest:
        # Use a slice that covers the entire year (January to December)
        year_data = ds.sel(time=slice(f'{year}-01-01', f'{year}-12-31'))
        sliced_data.append(year_data)

    # Combine the sliced data into a new xarray dataset or dictionary
    ds =  xr.concat(sliced_data, dim='time')
    iod_obs=ds.groupby('time.season')
   
    ds = xr.DataArray(obs_dataset.values,
    coords={'time': obs_dataset.times,
    'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
    dims=["time", "lat", "lon"])
    
    sliced_data = []
    for year in years_of_interest:
        # Use a slice that covers the entire year (January to December)
        year_data = ds.sel(time=slice(f'{year}-01-01', f'{year}-12-31'))
        sliced_data.append(year_data)

    # Combine the sliced data into a new xarray dataset or dictionary
    ds =  xr.concat(sliced_data, dim='time')
    
    ds = ds.where((ds.lon > 90) & (ds.lon < 130), drop=True)
                          
    #print(ds)
                
    # #ds = ds.groupby('time.year').sum() 
    # mds = ds.groupby('time.month')
    # ds_anom = mds - mds.mean(dim='time')
    # #print(tos_nino34_anom)
    # index_nino34 = ds_anom.mean(dim=['lat', 'lon'])
    
    lat_min = ds.lat.min()
    lat_max = ds.lat.max()
    lon_min = ds.lon.min()
    lon_max = ds.lon.max()
   
    fig, ax = plt.subplots(nrows=2, ncols=2 ,figsize=(6,4))
    
    x,y = np.meshgrid(ds.lon, ds.lat)
    
    #OBS
    map1=ma.zeros((len(ds.lat),len(ds.lon)))
    map2=ma.zeros((len(ds.lat),len(ds.lon)))
    map22=ma.zeros((len(ds.lat),len(ds.lon)))
    map3=ma.zeros((len(ds.lat),len(ds.lon)))
    map4=ma.zeros((len(ds.lat),len(ds.lon)))
    map44=ma.zeros((len(ds.lat),len(ds.lon)))
    map11=ma.zeros((len(ds.lat),len(ds.lon)))
    map33=ma.zeros((len(ds.lat),len(ds.lon)))
    #n=0
    
    #plot domain
    for i in np.arange(2):
        for j in np.arange(2):
            m = Basemap(ax=ax[i,j], projection ='cyl', llcrnrlat = lat_min, urcrnrlat = lat_max,
                llcrnrlon = lon_min, urcrnrlon = lon_max, resolution = 'l', fix_aspect=False)
            m.drawcoastlines(linewidth=1)
            m.drawcountries(linewidth=1)
            m.drawstates(linewidth=0.5, color='w')
  
    musim='MAM'
    
    alpha=0.05
    tes=40 
    density=3
    #cmap='viridis'
    cmap=None
   
    x0=nino_obs[musim].values
    x02=iod_obs[musim].values
    
    #corr at grids 
    for i in np.arange(len(ds.lat)-tes):
        print (i)
        for j in np.arange(len(ds.lon)-tes):
            d=ds[:,i,j]
            md = d.groupby('time.month')
           
            d_anom = md - md.mean(dim='time')
           
            pr=d_anom.groupby('time.season')
            
            y0=pr[musim].values
            
            bad = ~np.logical_or(np.isnan(x0), np.isnan(y0))
            x1=np.compress(bad, x0)     #Nino
            x12=np.compress(bad, x02)   #IOD
            y1=np.compress(bad, y0)     #rainfall
           
            #if x1.shape==(0,) or y1.shape==(0,) or y1.shape <=(2,):
            if y1.shape <=(2,):                   
                rpn='nan'
                
                rpni='nan'
                rpnic='nan'
                
                rpi='nan'
                
                rpin='nan'
                rpinc='nan'
                rin='nan'
                rinc='nan'                
                rpnc='nan'
                rpic='nan'
            
            else:
                
                df= pd.DataFrame({
                    'Rainfall': y1,
                    'Ni': x1,
                    'IO': x12
                })
                #print(df)
                r=pg.partial_corr(data=df, x='Ni', y='Rainfall', covar=None)
                rpn=r['r'].values
                rpnc=r['p-val'].values
                
                r=pg.partial_corr(data=df, x='IO', y='Rainfall', covar=None)
                rpi=r['r'].values
                rpic=r['p-val'].values
                
                r=pg.partial_corr(data=df, x='Ni', y='Rainfall', covar='IO')
                rpni=r['r'].values
                rpnic=r['p-val'].values
                
                r=pg.partial_corr(data=df, x='IO', y='Rainfall', covar='Ni')
                rpin=r['r'].values
                rpinc=r['p-val'].values
           
            map1[i,j]=rpn
            map11[i,j]=rpnc
            
            map2[i,j]=rpni
            map22[i,j]=rpnic
            
            map3[i,j]=rpi 
            map33[i,j]=rpic
            
            map4[i,j]=rpin 
            map44[i,j]=rpinc
            
            
    
    print(np.nanmin(map11), np.nanmax(map11))
    
    
    max1 = ax[0,0].contourf(x,y,map1, cmap=cmap, extend='both') #=levels,  norm =norm)
    
    ax[0,0].set_title(obs_name+'_ENSO',fontsize=9)
    #hatch ini ok
    ax[0,0].contourf(x, y, map11,levels=[np.nanmin(map11), alpha, ], 
                        hatches=[density*'/'],
                        extend='lower', alpha = 0)
    #for i in np.arange(len(x2)):
        # ax[0,0].text( x=x2[i], y=y2[i],s='.',alpha=0.2)
    #for i in np.arange(len(x3)):
        # ax[1,0].text( x=x3[i], y=y3[i],s='.',alpha=0.2)
    #for i in np.arange(len(x2)):
    #ini ok
    #ax[0,0].scatter(x2[i], y2[i], marker = '.', s = 0.5, c = 'k', alpha = 0.2)
   
    
    max1 = ax[0,1].contourf(x,y,map2, cmap=cmap,  extend='both') #=levels,  norm =norm)
    ax[0,1].set_title(obs_name+'_pENSO',fontsize=9)
    #ax[0,0].set_ylabel(musim)
    ax[0,1].contourf(x, y, map22,levels=[np.nanmin(map11), alpha, ], 
                        hatches=[density*'/'],
                        extend='lower', alpha = 0)
    
    
    max1 = ax[1,0].contourf(x,y,map3, cmap=cmap, extend='both') #=levels,  norm =norm)
    ax[1,0].set_title(obs_name+'_IOD',fontsize=9)
    
    ax[1,0].contourf(x, y, map33,levels=[np.nanmin(map33), alpha, ], 
                        hatches=[density*'/'],
                        extend='lower', alpha = 0)
    
    max1 = ax[1,1].contourf(x,y,map4, cmap=cmap, extend='both') #=levels,  norm =norm)
    ax[1,1].set_title(obs_name+'_pIOD',fontsize=9)
    ax[1,1].contourf(x, y, map44,levels=[np.nanmin(map11), alpha, ], 
                        hatches=[density*'/'],
                        extend='lower', alpha = 0)
    
    #Mix tidak dipakai dulu... tidak baku
    '''                   
    max1 = ax[2,0].contourf(x,y,mapx, extend='both') #=levels,  norm =norm)
    ax[2,0].set_title('GPCP_mix',fontsize=8)
    
    ax[2,0].contourf(x, y, mapxx,levels=[np.nanmin(mapxx), alpha, ], 
                        hatches=[density*'/'],
                        extend='lower', alpha = 0)
    #ax[2,1].axis('off')                
    '''
    
    '''             
    fs=8
    ax[0,0].set_yticks([-10,0,10,20],fontsize=fs)
    ax[1,0].set_yticks([-10,0,10,20],fontsize=fs)
    
    ax[0,0].set_yticklabels(['10S','0','10N','20N'],fontsize=fs)
    ax[1,0].set_yticklabels(['10S','0','10N','20N'],fontsize=fs)
    
    ax[1,0].set_xticks([100,120,140],fontsize=fs)
    ax[1,0].set_xticklabels(['100E','120E','140E'],fontsize=fs)
    ax[1,1].set_xticks([100,120,140],fontsize=fs)
    ax[1,1].set_xticklabels(['100E','120E','140E'],fontsize=fs)
    
    '''
    
    #ax[2,0].set_xticks([90,100,110,120,130,140],fontsize=fs)
    
    #ax[0,0].yaxis.set_tick_params(labelsize=6)
    #ax[1,0].yaxis.set_tick_params(labelsize=6)
    #ax[2,0].yaxis.set_tick_params(labelsize=6)
    
    #ax[2,0].xaxis.set_tick_params(labelsize=6)
    #ax[1,1].xaxis.set_tick_params(labelsize=6)
    
    
    #ax[n,0].set_ylabel(musim)
    #n=n+1
    cax = fig.add_axes([0.91, 0.15, 0.015, 0.7])
    cax.tick_params(labelsize=6)
    plt.colorbar(max1, cax = cax, extend='both') 
    plt.subplots_adjust(hspace=.2,wspace=.05)
    
    plt.show()
    fig.savefig(workdir+reg+'_'+obs_name+str(alpha)+musim+'.png',dpi=300,bbox_inches='tight')
    #fig.savefig(workdir+reg+'_nino34_iod_p_tes',dpi=300,bbox_inches='tight')


def nino34_anomaly(obs_dataset, workdir):
      
    import xarray as xr 
    import pandas as pd
    #fig, ax = plt.subplots(nrows=1, ncols=1 ,figsize=(6,4))
    
    #--------------------------------------- SEA 200412 
    d= pd.read_excel('D:/Disertasi3/enso_mon_1981-2005.xlsx')
    #SEA:
    #d= pd.read_excel('D:/Disertasi3/enso_mon_1981-2004.xlsx')
    index_nino34 = xr.DataArray( d['Value'],
    coords={'time': obs_dataset.times})
    
        
    #index_nino34.plot(size=8)
    #plt.legend(['anomaly', 'sst'])
    #plt.title('SST anomaly over the Niño 3.4 region');
    
    #figname='sst'
    #plt.savefig(workdir+figname, dpi=300, bbox_inches='tight')
    
    fig=plt.figure(figsize=[8,6])
    fig.subplots_adjust(right=.7)
    
    plt.plot(obs_dataset.times, d['Value'], lw=2, color='black', label = 'Obs')
    plt.axhline(.5, linestyle = 'dashed', color='red')
    plt.axhline(-.5, linestyle = 'dashed', color='red')
    #ax.set_xticklabels(obs_dataset.times) error
    #exit()

    filepath=[
    'D:/data1/tos_Omon_CNRM-CM5_historical_r1i1p1_198101-200512.nc',
    'D:/data1/tos_Omon_IPSL-CM5A-LR_historicalNat_r1i1p1_1981-2005_nino34_ok.nc',
    'D:/data1/tos_Omon_HadGEM2-ES_esmHistorical_r1i1p1_1981-2005_nino34_ok.nc',
    'D:/data1/tos_Omon_NorESM1-M_historical_r1i1p1_198101-200512.nc',
    'D:/data1/4tos_Omon_GFDL-ESM2M_historical_r1i1p1_198101-200512.nc'
    ]
    
    names=['CNRM', 'IPSL', 'HadGEM2', 'NorESM1', 'GFDL' ]
    for i in np.arange(len(filepath)):
      
        print(i, names[i])
           
        dsx = xr.open_dataset(filepath[i])
        #print(dsx)
        #slice ini nans
        #tos_nino34 = ds.sel(i=slice(-5, 5), j=slice(190, 240))
        try:
            tos_nino34 = dsx.where(
                    (dsx.lat < 5) & (dsx.lat > -5) & 
                    (dsx.lon > 190) & (dsx.lon < 240), drop=True)
                    
        except:
            #jika sistem bukan 0-360 maka 
            #ubah 190 ke -170 dan 240 ke -120 
            #rumus 190-360=-170
            #ini salah unutk GFDL -280 to 80
            #tos_nino34 = dsx.where(
            #        (dsx.lat < 5) & (dsx.lat > -5) & (dsx.lon > -170) & 
            #        (dsx.lon < -120), drop=True)
            tos_nino34 = dsx.where(
                    (dsx.rlat < 5) & (dsx.rlat > -5) & 
                    (dsx.rlon > 190) & (dsx.rlon < 240), drop=True)
                    
        
        gb = tos_nino34.tos.groupby('time.month')
        tos_nino34_anom = gb - gb.mean(dim='time')
        #print(tos_nino34_anom)
        try: index_nino34 = tos_nino34_anom.mean(dim=['rlat', 'rlon'])
        except: 
            try: index_nino34 = tos_nino34_anom.mean(dim=['i', 'j'])
            except: index_nino34 = tos_nino34_anom.mean(dim=['lat', 'lon'])
        
        
        # print('index_nino34=',index_nino34)
        # #smooth the anomalies with a 5-month running m.vvaleuean:
        #index_nino34_rolling_mean = index_nino34.rolling(time=5, center=True).mean()
        ir5 = index_nino34.rolling(time=5, center=True).mean()
       
       #karna i=2 hanya sampai 2005-11 maka
        times=obs_dataset.times
        if i !=2: 
            ir5=np.delete(ir5,-1)
            times=np.delete(times,-1)
            print(len(ir5),len(times))
        else: 
            times=np.delete(times,-1)
            print(len(ir5),len(times))
        
        sdi= ir5.std()
        
        #plt.plot(obs_dataset.times, ir5.values, label = names[i])
        plt.plot(times, ir5.values, label = names[i])
       
    #plt.axvline(x=1982,ymin=0, ymax=sse[c-1]/100000, ls=':', color='black') 
    plt.legend(bbox_to_anchor=(1, .5), loc='best', prop={'size':10}, frameon=False) 
    
    plt.ylabel('SST Nino3.4 anomaly (K)')
    #plt.set_xticks([1981,1988,2005])
    #plt.title('SST anomaly obs & models over the Niño3.4 region');
    figname='sstN34A'
    
    #plt.show()
    plt.savefig(workdir+figname) #, dpi=300, bbox_inches='tight')
    plt.show()

def nino34_dmi_anomaly(obs_dataset, workdir):
      
    import xarray as xr 
    import pandas as pd
    fig, ax = plt.subplots(nrows=1, ncols=2 ,figsize=(6,4))
    
    #--------------------------------------- SEA 200412 
    d= pd.read_excel('D:/Disertasi3/enso_mon_1981-2005.xlsx')
    #SEA:
    #d= pd.read_excel('D:/Disertasi3/enso_mon_1981-2004.xlsx')
    index_nino34 = xr.DataArray( d['Value'],
    coords={'time': obs_dataset.times})
    
        
    #index_nino34.plot(size=8)
    #plt.legend(['anomaly', 'sst'])
    #plt.title('SST anomaly over the Niño 3.4 region');
    
    #figname='sst'
    #plt.savefig(workdir+figname, dpi=300, bbox_inches='tight')
    
    
    #fig.subplots_adjust(right=.7)
    
    ax[0].plot(obs_dataset.times, d['Value'], lw=2, color='black', label = 'Obs')
    ax[0].axhline(.5, linestyle = 'dashed', color='red')
    ax[0].axhline(-.5, linestyle = 'dashed', color='red')
    #ax.set_xticklabels(obs_dataset.times) error
    #exit()

    filepath=[
    'D:/data1/tos_Omon_CNRM-CM5_historical_r1i1p1_198101-200512.nc',
    'D:/data1/tos_Omon_IPSL-CM5A-LR_historicalNat_r1i1p1_1981-2005_nino34_ok.nc',
    'D:/data1/tos_Omon_HadGEM2-ES_esmHistorical_r1i1p1_1981-2005_nino34_ok.nc',
    'D:/data1/tos_Omon_NorESM1-M_historical_r1i1p1_198101-200512.nc',
    'D:/data1/4tos_Omon_GFDL-ESM2M_historical_r1i1p1_198101-200512.nc'
    ]
    
    names=['CNRM', 'IPSL', 'HadGEM2', 'NorESM1', 'GFDL' ]
    for i in np.arange(len(filepath)):
      
        print(i, names[i])
           
        dsx = xr.open_dataset(filepath[i])
        #print(dsx)
        #slice ini nans
        #tos_nino34 = ds.sel(i=slice(-5, 5), j=slice(190, 240))
        try:
            tos_nino34 = dsx.where(
                    (dsx.lat < 5) & (dsx.lat > -5) & 
                    (dsx.lon > 190) & (dsx.lon < 240), drop=True)
                    
        except:
            #jika sistem bukan 0-360 maka 
            #ubah 190 ke -170 dan 240 ke -120 
            #rumus 190-360=-170
            #ini salah unutk GFDL -280 to 80
            #tos_nino34 = dsx.where(
            #        (dsx.lat < 5) & (dsx.lat > -5) & (dsx.lon > -170) & 
            #        (dsx.lon < -120), drop=True)
            tos_nino34 = dsx.where(
                    (dsx.rlat < 5) & (dsx.rlat > -5) & 
                    (dsx.rlon > 190) & (dsx.rlon < 240), drop=True)
                    
        
        gb = tos_nino34.tos.groupby('time.month')
        tos_nino34_anom = gb - gb.mean(dim='time')
        #print(tos_nino34_anom)
        try: index_nino34 = tos_nino34_anom.mean(dim=['rlat', 'rlon'])
        except: 
            try: index_nino34 = tos_nino34_anom.mean(dim=['i', 'j'])
            except: index_nino34 = tos_nino34_anom.mean(dim=['lat', 'lon'])
        
        
        # print('index_nino34=',index_nino34)
        # #smooth the anomalies with a 5-month running m.vvaleuean:
        #index_nino34_rolling_mean = index_nino34.rolling(time=5, center=True).mean()
        ir5 = index_nino34.rolling(time=5, center=True).mean()
       
       #karna i=2 hanya sampai 2005-11 maka
        times=obs_dataset.times
        if i !=2: 
            ir5=np.delete(ir5,-1)
            times=np.delete(times,-1)
            print(len(ir5),len(times))
        else: 
            times=np.delete(times,-1)
            print(len(ir5),len(times))
        
        sdi= ir5.std()
        
        #plt.plot(obs_dataset.times, ir5.values, label = names[i])
        ax[0].plot(times, ir5.values, label = names[i])
       
    
    #ax[0].legend(bbox_to_anchor=(1, .5), loc='best', prop={'size':10}, frameon=False) 
    
    ax[0].set_ylabel('SST_Nino3.4 anomaly (K)')
    #plt.set_xticks([1981,1988,2005])
    #plt.title('SST anomaly obs & models over the Niño3.4 region');
    
    
    filepath=[
            'D:/data1/tos_Omon_CNRM-CM5_historical_r1i1p1_198101-200512.nc',
            'D:/data1/tos_Omon_IPSL-CM5A-LR_historicalNat_r1i1p1_1981-2005.nc',
            'D:/data1/tos_Omon_HadGEM2-ES_esmHistorical_r1i1p1_1981-2005_11.nc',
            'D:/data1/tos_Omon_NorESM1-M_historical_r1i1p1_198101-200512.nc',
            'D:/data1/4tos_Omon_GFDL-ESM2M_historical_r1i1p1_198101-200512.nc'
            ]
            
    names=['CNRM', 'IPSL', 'HadGEM2', 'NorESM1', 'GFDL' ]
    
    #fig, ax = plt.subplots(nrows=4, ncols=1 ) #,figsize=(6,4))
    
    #--------------------------------------- SEA 200412 
    d= pd.read_excel('D:/Disertasi3/dmi_1981-2005.xlsx')
    
    index_nino34 = xr.DataArray( d['Value'],
    coords={'time': obs_dataset.times})
   
   
    
    ax[1].plot(obs_dataset.times, d['Value'], lw=2, color='black', label = 'Obs')
    ax[1].axhline(.5, linestyle = 'dashed', color='red')
    ax[1].axhline(-.5, linestyle = 'dashed', color='red')
    #ax.set_xticklabels(obs_dataset.times) error
    #exit()

    
    for i in np.arange(len(filepath)):
      
        print(i, names[i])
           
        dsx = xr.open_dataset(filepath[i])
        #print(dsx)
        #slice ini nans
        #tos_nino34 = ds.sel(i=slice(-5, 5), j=slice(190, 240))
        # (50' E - 70' E, 10 S - 10 N)
        try:
            tos_w = dsx.where(
                    (dsx.lat < 10) & (dsx.lat > -10) & 
                    (dsx.lon > 50) & (dsx.lon < 70), drop=True)
        #GFDL pakai rlat,rlon            
        except:
            tos_w = dsx.where(
                    (dsx.rlat < 10) & (dsx.rlat > -10) & 
                    (dsx.rlon > 50) & (dsx.rlon < 70), drop=True)
                    
                    
        try:
            tos_e = dsx.where(
                    (dsx.lat < 0) & (dsx.lat > -10) & 
                    (dsx.lon > 90) & (dsx.lon < 110), drop=True)
                    
        except:
            tos_e = dsx.where(
                    (dsx.rlat < 0) & (dsx.rlat > -10) & 
                    (dsx.rlon > 90) & (dsx.rlon < 110), drop=True)            
                    
                    
        # #(50' E - 70' E, 10 S - 10 N)            
        # tos_w = dsx.where(
                    # (dsx.lat < 10) & (dsx.lat > -10) & 
                    # (dsx.lon > 50) & (dsx.lon < 70), drop=True)
        # #(90' E- 110' E, 10 S - 0 )
        # tos_e = dsx.where(
                    # (dsx.lat < 0) & (dsx.lat > -10) & 
                    # (dsx.lon > 90) & (dsx.lon < 110), drop=True)

        tos_nino34=tos_w 
        
        gb = tos_nino34.tos.groupby('time.month')
        tos_nino34_anom = gb - gb.mean(dim='time')
        #print(tos_nino34_anom)
        try: index_nino34 = tos_nino34_anom.mean(dim=['rlat', 'rlon'])
        except: 
            try: index_nino34 = tos_nino34_anom.mean(dim=['i', 'j'])
            except: index_nino34 = tos_nino34_anom.mean(dim=['lat', 'lon'])
        # print('index_nino34=',index_nino34)
        # #smooth the anomalies with a 5-month running m.vvaleuean:
        #index_nino34_rolling_mean = index_nino34.rolling(time=5, center=True).mean()
        ir5_w = index_nino34.rolling(time=5, center=True).mean()
        
        tos_nino34=tos_e 
        
        gb = tos_nino34.tos.groupby('time.month')
        tos_nino34_anom = gb - gb.mean(dim='time')
        #print(tos_nino34_anom)
        try: index_nino34 = tos_nino34_anom.mean(dim=['rlat', 'rlon'])
        except: 
            try: index_nino34 = tos_nino34_anom.mean(dim=['i', 'j'])
            except: index_nino34 = tos_nino34_anom.mean(dim=['lat', 'lon'])
        # print('index_nino34=',index_nino34)
        # #smooth the anomalies with a 5-month running m.vvaleuean:
        #index_nino34_rolling_mean = index_nino34.rolling(time=5, center=True).mean()
        ir5_e = index_nino34.rolling(time=5, center=True).mean()
        
        ir5=ir5_w -ir5_e
        
        #karna i=2 hanya sampai 2005-11 maka
        times=obs_dataset.times
        if i !=2: 
            ir5=np.delete(ir5,-1)
            times=np.delete(times,-1)
            print(len(ir5),len(times))
        else: 
            times=np.delete(times,-1)
            print(len(ir5),len(times))
        
        
        #plt.plot(obs_dataset.times, ir5.values, label = names[i])
        ax[1].plot(times, ir5.values, label = names[i])
    #print(times) 
    #ax[1].set_xticks([1985, 1990,1995,2000, 2005])
    plt.legend(bbox_to_anchor=(1, .5), loc='best', prop={'size':10}, frameon=False) 
    ax[1].set_ylabel('STT_IOD anomaly (K)')
    plt.subplots_adjust(right=0.8)
    ax[0].set_ylim(-3,3)
    ax[1].set_ylim(-3,3)
    
    #plt.set_xticks([1981,1988,2005])
    #plt.title('SST anomaly obs & models over the Niño3.4 region');
    figname='SST_Nino34-IOD_1981-2005'
    plt.savefig(workdir+figname) #, dpi=300, bbox_inches='tight')
    plt.show()






def dmi_anomaly(obs_dataset, workdir):
      
    import xarray as xr 
    import pandas as pd
    
    filepath=[
            'D:/data1/tos_Omon_CNRM-CM5_historical_r1i1p1_198101-200512.nc',
            'D:/data1/tos_Omon_IPSL-CM5A-LR_historicalNat_r1i1p1_1981-2005.nc',
            'D:/data1/tos_Omon_HadGEM2-ES_esmHistorical_r1i1p1_1981-2005_11.nc',
            'D:/data1/tos_Omon_NorESM1-M_historical_r1i1p1_198101-200512.nc',
            'D:/data1/4tos_Omon_GFDL-ESM2M_historical_r1i1p1_198101-200512.nc'
            ]
            
    names=['CNRM', 'IPSL', 'HadGEM2', 'NorESM1', 'GFDL' ]
    
    ds = xr.open_dataset(filepath[0])
    
    # w=[-10, 10, 50, 70]
    # e=[-10, 0, 90, 110]
    # n=[-5, 5, 190, 240]
    # # (50' E - 70' E, 10 S - 10 N)
    # #(90' E- 110' E, 10 S - 0 )
    # #lons, lats = np.meshgrid(ref_dataset.lons, ref_dataset.lats) 
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # #m = Basemap(ax=ax, projection='cyl',llcrnrlat = ds.lat.min(), urcrnrlat = ds.lat.max(),
    # #            llcrnrlon = ds.lon.min(), urcrnrlon = ds.lon.max(), resolution = 'h')
    # m = Basemap(ax=ax, projection='cyl',llcrnrlat = -30, 
                                        # urcrnrlat = 30,
                                        # llcrnrlon = 20, 
                                        # urcrnrlon = 300, resolution = 'h')
    # m.drawcoastlines(linewidth=0.75)
    # m.drawcountries(linewidth=0.75)
    # #m.etopo()  
    # #x, y = m(lons, lats) 
    # #subregion_array = ma.masked_equal(subregion_array, 0)
    # #max=m.contourf(x, y, subregion_array, alpha=0.7, cmap='Accent')

    # draw_screen_poly(w, m, 'b') 
    # draw_screen_poly(e, m, 'b') 
    # draw_screen_poly(n, m, 'b') 
    
    # plt.savefig(workdir+'nino34_dmi2.png', dpi=300, bbox_inches='tight')
    # plt.show()
    
    # exit()
 
    #fig, ax = plt.subplots(nrows=4, ncols=1 ) #,figsize=(6,4))
    
    #--------------------------------------- SEA 200412 
    d= pd.read_excel('D:/Disertasi3/dmi_1981-2005.xlsx')
    
    index_nino34 = xr.DataArray( d['Value'],
    coords={'time': obs_dataset.times})
   
    fig, ax = plt.subplots(nrows=1, ncols=1 )
    #fig=plt.figure(figsize=[8,6])
    fig.subplots_adjust(right=.7)
    
    plt.plot(obs_dataset.times, d['Value'], lw=3, color='black', label = 'Obs')
    plt.axhline(.5, linestyle = 'dashed', color='red')
    plt.axhline(-.5, linestyle = 'dashed', color='red')
    #ax.set_xticklabels(obs_dataset.times) error
    #exit()

    
    for i in np.arange(len(filepath)):
      
        print(i, names[i])
           
        dsx = xr.open_dataset(filepath[i])
        #print(dsx)
        #slice ini nans
        #tos_nino34 = ds.sel(i=slice(-5, 5), j=slice(190, 240))
        # (50' E - 70' E, 10 S - 10 N)
        try:
            tos_w = dsx.where(
                    (dsx.lat < 10) & (dsx.lat > -10) & 
                    (dsx.lon > 50) & (dsx.lon < 70), drop=True)
        #GFDL pakai rlat,rlon            
        except:
            tos_w = dsx.where(
                    (dsx.rlat < 10) & (dsx.rlat > -10) & 
                    (dsx.rlon > 50) & (dsx.rlon < 70), drop=True)
                    
                    
        try:
            tos_e = dsx.where(
                    (dsx.lat < 0) & (dsx.lat > -10) & 
                    (dsx.lon > 90) & (dsx.lon < 110), drop=True)
                    
        except:
            tos_e = dsx.where(
                    (dsx.rlat < 0) & (dsx.rlat > -10) & 
                    (dsx.rlon > 90) & (dsx.rlon < 110), drop=True)            
                    
                    
        # #(50' E - 70' E, 10 S - 10 N)            
        # tos_w = dsx.where(
                    # (dsx.lat < 10) & (dsx.lat > -10) & 
                    # (dsx.lon > 50) & (dsx.lon < 70), drop=True)
        # #(90' E- 110' E, 10 S - 0 )
        # tos_e = dsx.where(
                    # (dsx.lat < 0) & (dsx.lat > -10) & 
                    # (dsx.lon > 90) & (dsx.lon < 110), drop=True)

        tos_nino34=tos_w 
        
        gb = tos_nino34.tos.groupby('time.month')
        tos_nino34_anom = gb - gb.mean(dim='time')
        #print(tos_nino34_anom)
        try: index_nino34 = tos_nino34_anom.mean(dim=['rlat', 'rlon'])
        except: 
            try: index_nino34 = tos_nino34_anom.mean(dim=['i', 'j'])
            except: index_nino34 = tos_nino34_anom.mean(dim=['lat', 'lon'])
        # print('index_nino34=',index_nino34)
        # #smooth the anomalies with a 5-month running m.vvaleuean:
        #index_nino34_rolling_mean = index_nino34.rolling(time=5, center=True).mean()
        ir5_w = index_nino34.rolling(time=5, center=True).mean()
        
        tos_nino34=tos_e 
        
        gb = tos_nino34.tos.groupby('time.month')
        tos_nino34_anom = gb - gb.mean(dim='time')
        #print(tos_nino34_anom)
        try: index_nino34 = tos_nino34_anom.mean(dim=['rlat', 'rlon'])
        except: 
            try: index_nino34 = tos_nino34_anom.mean(dim=['i', 'j'])
            except: index_nino34 = tos_nino34_anom.mean(dim=['lat', 'lon'])
        # print('index_nino34=',index_nino34)
        # #smooth the anomalies with a 5-month running m.vvaleuean:
        #index_nino34_rolling_mean = index_nino34.rolling(time=5, center=True).mean()
        ir5_e = index_nino34.rolling(time=5, center=True).mean()
        
        ir5=ir5_w -ir5_e
        
        #karna i=2 hanya sampai 2005-11 maka
        times=obs_dataset.times
        if i !=2: 
            ir5=np.delete(ir5,-1)
            times=np.delete(times,-1)
            print(len(ir5),len(times))
        else: 
            times=np.delete(times,-1)
            print(len(ir5),len(times))
        
        
        #plt.plot(obs_dataset.times, ir5.values, label = names[i])
        plt.plot(times, ir5.values, label = names[i])
    
    #ax.set_xticks([1985, 1990,1995,2000, 2005])
    plt.legend(bbox_to_anchor=(1, .5), loc='best', prop={'size':10}, frameon=False) 
    plt.ylabel('STT_IOD anomaly (K)')
    #plt.set_xticks([1981,1988,2005])
    #plt.title('SST anomaly obs & models over the Niño3.4 region');
    figname='SST_IOD_1981-2005'
    plt.savefig(workdir+figname) #, dpi=300, bbox_inches='tight')
    plt.show()
    


def eofs_multi_tos_nino2(obs_dataset, workdir):
      
    import xarray as xr 
    from eofs.xarray import Eof
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    
    
    anom=1
    
    n_eofs=1 #1,2,3..
    n_ke=0 #0,1,2..
    
    fig, ax = plt.subplots(2,3) #, figsize=(8,8))
    '''
    filepath=[#'D:/data1/tos/tos_Omon_CNRM-CM5_historical_r1i1p1_1950-2005.nc',
    'D:/data1/tos/tos_Omon_CNRM-CM5_historical_r1i1p1_1950-1975.nc',
    'D:/data1/tos/tos_Omon_CNRM-CM5_historical_r1i1p1_1976-2005.nc',
    'D:/data1/tos/tos_Omon_CNRM-CM5_historical_r1i1p1_1981-2005.nc',
    ]
    names=['CNRM-1','CNRM-2','CNRM-3']
    
    '''
    filepath=['D:/data1/cobe_sst.mon.nc',
    'D:/data1/cobe_sst_mon_1891-1960.nc',
    'D:/data1/cobe_sst_mon_1961-2023.nc',
    #'D:/data1/cobe_sst_mon_1981-2005.nc',
    ]
    names=['COBE-1','COBE-2','COBE-3']
    
    filepath2=['D:/data1/cobe_sst.mon.nc',
    'D:/data1/tos/tos_Omon_CNRM-CM5_historical_r1i1p1_1950-2005.nc',
    'D:/data1/tos/tos_Omon_IPSL-CM5A-LR_historicalNat_r1i1p1_185001-201212.nc',
    'D:/data1/tos/tos_Omon_HadGEM2-ES_esmHistorical_r1i1p1_195912-200511.nc',
    'D:/data1/tos/tos_Omon_NorESM1-M_historical_r1i1p1_185001-200512.nc',
    ]
    
    #names=['COBE','CNRM', 'IPSL', 'HadGEM2', 'NorESM1', 'GFDL' ]
    
    for i in np.arange(len(filepath)):
      
        print(i, names[i])
           
        dsx = xr.open_dataset(filepath[i])
        #print(dsx)
        #slice ini nans
        #tos_nino34 = ds.sel(i=slice(-5, 5), j=slice(190, 240))
        try:
            tos_nino34 = dsx.where(
                    (dsx.lat < 5) & (dsx.lat > -5) & 
                    (dsx.lon > 190) & (dsx.lon < 240), drop=True)
                    
        except:
          
            tos_nino34 = dsx.where(
                    (dsx.rlat < 5) & (dsx.rlat > -5) & 
                    (dsx.rlon > 190) & (dsx.rlon < 240), drop=True)
        
        ds=tos_nino34
        #print(ds)
        
        try:
            ds = xr.DataArray(ds.sst.values,
            coords={'time': ds.time,
                    'lat': ds.lat.values, 
                    'lon': ds.lon.values},
            dims=["time", "lat", "lon"])
            
        except:
            ds = xr.DataArray(ds.tos.values,
            coords={'time': ds.time,
                    'lat': ds.lat[:,0].values, 
                    'lon': ds.lon[0,:].values},
            dims=["time", "lat", "lon"])
        
        if anom==1:
            climatology_mean = ds.groupby("time.month").mean("time")
            climatology_std = ds.groupby("time.month").std("time")
            ds = xr.apply_ufunc(
                lambda x, m, s: 
                (x - m) / s,
                ds.groupby("time.month"),
                climatology_mean,
                climatology_std,
            )
        
        #print(ds)
           
        ds = ds.groupby('time.year').sum()
        ds=ds.rename({'year':'time'})
        #print('ds=',ds)
        #ds=ds.rename('SEA-RA')
        solver = Eof(ds)
        fn='yearly'
        
        print('ds.lon.min(),ds.lon.max()=', ds.lon.min().values,ds.lon.max().values)
        #print('dm.shape', dm)
        #print('dm.values.shape', dm.values)
        
        #eof = solver.eofsAsCorrelation(neofs=n_eofs)
        eof = solver.eofs(neofs=n_eofs)
        
        pc = solver.pcs(npcs=n_eofs, pcscaling=0) #1 with scaling
        
        #print('eof=',eof)
        print('solver.varianceFraction=', solver.varianceFraction(3))
        
        #print('pc=',pc)
        levels =np.arange(0.8,1,.02)
        p = eof[n_ke].plot.contourf(ax=ax[1,i], add_colorbar=False)
        #p = eof[n_ke].plot.contourf(ax=ax[1,i], levels =levels , vmin=-1, vmax=1,cmap='rainbow', add_colorbar=False)
       
        #ax.add_feature(cfeature.LAND, facecolor='w', edgecolor='k')
        #cb = plt.colorbar(fill, orientation='horizontal')
        #cb.set_label('correlation coefficient', fontsize=12)
        
        #ax.set_xlabel(fn)
        
    
       
        pc[:, n_ke].plot(ax=ax[0,i], color='b', linewidth=2)
        #ax[0,i] = plt.gca()
        ax[0,i].axhline(0, color='k')
        #ax.set_ylim(-3, 3)
        #ax.set_xlabel('month')
        if anom==1: 
            ax[0,0].set_ylabel('Normalized Units')
        else:
            ax[0,0].set_ylabel('Un-normalized Units')
        #ax.set_title('PC'+str(i)+' Time Series_'+fn, pad=20, fontsize=16)
        
        ax[0,i].tick_params(axis='x', pad=1,labelsize=8)
        ax[0,i].tick_params(axis='y', pad=1,labelsize=8)
        
        #tt=' ['+str(n_ke+1)+']'+' ['+str(np.round(solver.varianceFraction()[n_ke].data,2)*100)+'%]'   
        
        tt='  Mode='+str(n_ke+1)+'  EV='+str(np.round(solver.varianceFraction()[n_ke].data*100,2))+'%'
        
        ax[0,i].set_title(names[i]+tt, fontsize=9)
        
        ax[1,i].set_xlabel('')
        ax[1,i].set_ylabel('')
        ax[0,i].set_ylabel('')
        ax[0,i].set_xlabel('')
        
        #190-240 => -170 to -120
        #ax[1,i].set_xticks([-160,-140,-120])
        #ax[1,i].set_xticklabels(['160W','140W','120W'])
        #ax[1,i].tick_params(axis='x', pad=1,labelsize=8)
        #ax[1,i].set_title('')
        if i>0: ax[0,i].set_yticks([])
    #-5 5    
    ax[1,0].set_yticks([-5,0,5])
    ax[1,0].set_yticklabels(['5S','0','5N'])
    ax[1,0].tick_params(axis='y', pad=1,labelsize=8)
    
    
    
    #plt.subplots_adjust(right=.5)
    plt.subplots_adjust(hspace=.2,wspace=.1)
    
    cax = fig.add_axes([0.91, 0.11, 0.015, 0.345])
   
    cbar = plt.colorbar(p,cax=cax)
    
    
    plt.draw()
    #cbar.ax.get_yaxis().set_ticks([])
        
    #ax[0].set_title('EOF'+str(1)+' expressed as correlation_'+fn, pad=20, fontsize=16)
    #plt.savefig(workdir+reg+'_EOF_'+str(n_ke)+'_3tos_new_'+fn+, dpi=300,bbox_inches='tight')
    plt.show()        
            
            
            
            
def enso_ts_cmip5(obs_dataset, workdir):
      
    import xarray as xr 
    import pandas as pd
    from scipy.stats import pearsonr
   
    d= pd.read_excel('D:/Disertasi3/enso_mon_1981-2005.xlsx')
    dd= pd.read_excel('D:/Disertasi3/dmi_1981-2005.xlsx')
    
    nino_obs = xr.DataArray( d['Value'],
    coords={'time': obs_dataset.times})
    gb = nino_obs.groupby('time.month')
    nino_obs = gb - gb.mean(dim='time')
    #nino_obs = nino_obs.rolling(time=5, center=True).mean()
    
    iod_obs = xr.DataArray( dd['Value'],
    coords={'time': obs_dataset.times})
    gb = iod_obs.groupby('time.month')
    iod_obs = gb - gb.mean(dim='time')
    #iod_obs = iod_obs.rolling(time=5, center=True).mean()
    
    fig, ax = plt.subplots(nrows=6, ncols=1 ,figsize=(6,4))
    
    #fig.subplots_adjust(bottom=.3)
    
    #ax[0].plot(obs_dataset.times, d['Value'], color='black', label = 'Nino34')    
    #ax[1].plot(obs_dataset.times, dd['Value'], color='blue', label = 'IOD') 
    ax[0].plot(obs_dataset.times, nino_obs.values, color='black', label = 'Nino34r')   
    ax[0].set_title('Obs', loc='left', y=0.85, x=0.005, fontsize='medium')
    #ax[1].plot(obs_dataset.times, iod_obs.values, color='blue', label = 'IODr') 
    #[ax[i].set_xticks([]) for i in [0,1,2]] 
    #for i in [0,1,2]: ax[i].set_xticks([]) ini gak mau aneh
    for i in [0,1,2,3,4]:ax[i].xaxis.set_major_locator(plt.NullLocator())
    
  
    filepath=[
    'D:/data1/tos_Omon_CNRM-CM5_historical_r1i1p1_198101-200512.nc',
    'D:/data1/tos_Omon_IPSL-CM5A-LR_historicalNat_r1i1p1_1981-2005_nino34_ok.nc',
    'D:/data1/tos_Omon_HadGEM2-ES_esmHistorical_r1i1p1_1981-2005_nino34_ok.nc',
    'D:/data1/tos_Omon_NorESM1-M_historical_r1i1p1_198101-200512.nc',
    'D:/data1/4tos_Omon_GFDL-ESM2M_historical_r1i1p1_198101-200512.nc'
    ]
    
    names=['CNRM', 'IPSL', 'HadGEM2', 'NorESM1', 'GFDL' ]
    
    for i in np.arange(len(filepath)):
      
        print(i, names[i])
           
        dsx = xr.open_dataset(filepath[i])
        #print(dsx)
        #slice ini nans
        #tos_nino34 = ds.sel(i=slice(-5, 5), j=slice(190, 240))
        try:
            tos_nino34 = dsx.where(
                    (dsx.lat < 5) & (dsx.lat > -5) & 
                    (dsx.lon > 190) & (dsx.lon < 240), drop=True)
                    
        except:
          
            tos_nino34 = dsx.where(
                    (dsx.rlat < 5) & (dsx.rlat > -5) & 
                    (dsx.rlon > 190) & (dsx.rlon < 240), drop=True)
                    
        
        gb = tos_nino34.tos.groupby('time.month')
        tos_nino34_anom = gb - gb.mean(dim='time')
        #print(tos_nino34_anom)
        try: index_nino34 = tos_nino34_anom.mean(dim=['rlat', 'rlon'])
        except: 
            try: index_nino34 = tos_nino34_anom.mean(dim=['i', 'j'])
            except: index_nino34 = tos_nino34_anom.mean(dim=['lat', 'lon'])
        
      
        ir5 = index_nino34 #.rolling(time=5, center=True).mean()
       
       #karna i=2 hanya sampai 2005-11 maka
        times=obs_dataset.times
        if i==2: 
            nino_obs1=np.delete(nino_obs,-1)
            times=np.delete(times,-1)
            print( len(nino_obs),len(ir5))
        else:
            nino_obs1=nino_obs
            print(len(iod_obs),len(ir5))
        
        m0=nino_obs1.values
        m1=ir5.values
        #print(iod_obs)
        #print(ir5)
        bad = ~np.logical_or(np.isnan(m0), np.isnan(m1))
        x1=np.compress(bad, m0) 
        y1=np.compress(bad, m1)
        #print(x1)
        #print(y1)
        
        sd1=x1.std() #(skipna=None)
        #print(sd1)
        sd2=y1.std()
        s=sd2/sd1
        
        c,pp=pearsonr(x1, y1)        
        
        #T= (4*(1+c)**4)/(((s+1/s)**2)*((1+r0)**4))
        # Tanpa pakai **4 T naik dikit
        r0=1
        tt= (4*(1+c))/(((s+1/s)**2)*((1+r0)))
        
        ax[i+1].plot(times, ir5.values, label = names[i]) 
        #ax[i+1].set_title(names[i]+' ('+'%.2f'%tt+')', loc='left', y=0.5, x=0.005, fontsize='medium')
        ax[i+1].set_title(names[i]+' (sr='+'%.2f'%s + ', r=' + '%.2f'%c + ', T=' + '%.2f'%tt + ')', loc='left', y=0.85, x=0.005, fontsize='medium')
    
    import matplotlib.dates as mdates
    
    ax[5].xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 7)))
    #ax[3].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%b'))
    # Rotates and right-aligns the x labels so they don't crowd each other.
    for label in ax[5].get_xticklabels(which='major'):
        label.set(rotation=65, horizontalalignment='right')
    
    fig.subplots_adjust(bottom=.25)
    plt.subplots_adjust(hspace=.8,wspace=.05)
    
    
    plt.ylabel('                                                        SST Nino3.4 anomaly (K)')
    #plt.set_xticks([1981,1988,2005])
    #plt.title('SST anomaly obs & models over the Niño3.4 region');
    figname='ts_Nino_obs_5model_2'
    
    #plt.show()
    plt.savefig(workdir+figname) #, dpi=300, bbox_inches='tight')
    plt.show()

def ts_taylor(obs_dataset, workdir):
      
    import xarray as xr 
    import pandas as pd
    from scipy.stats import pearsonr
   
    d= pd.read_excel('D:/Disertasi3/enso_mon_1981-2005.xlsx')
    dd= pd.read_excel('D:/Disertasi3/dmi_1981-2005.xlsx')
    
    nino_obs = xr.DataArray( d['Value'],
    coords={'time': obs_dataset.times})
    gb = nino_obs.groupby('time.month')
    nino_obs = gb - gb.mean(dim='time')
    #nino_obs = nino_obs.rolling(time=5, center=True).mean()
    
    iod_obs = xr.DataArray( dd['Value'],
    coords={'time': obs_dataset.times})
    gb = iod_obs.groupby('time.month')
    iod_obs = gb - gb.mean(dim='time')
    #iod_obs = iod_obs.rolling(time=5, center=True).mean()
    
    fig, ax = plt.subplots(nrows=6, ncols=1 ,figsize=(6,4))
    
    #fig.subplots_adjust(bottom=.3)
    
    #ax[0].plot(obs_dataset.times, d['Value'], color='black', label = 'Nino34')    
    #ax[1].plot(obs_dataset.times, dd['Value'], color='blue', label = 'IOD') 
    ax[0].plot(obs_dataset.times, nino_obs.values, color='black', label = 'Nino34r')   
    ax[0].set_title('Obs', loc='left', y=0.85, x=0.005, fontsize='medium')
    #ax[1].plot(obs_dataset.times, iod_obs.values, color='blue', label = 'IODr') 
    #[ax[i].set_xticks([]) for i in [0,1,2]] 
    #for i in [0,1,2]: ax[i].set_xticks([]) ini gak mau aneh
    for i in [0,1,2,3,4]:ax[i].xaxis.set_major_locator(plt.NullLocator())
    
  
    filepath=[
    'D:/data1/tos_Omon_CNRM-CM5_historical_r1i1p1_198101-200512.nc',
    'D:/data1/tos_Omon_IPSL-CM5A-LR_historicalNat_r1i1p1_1981-2005_nino34_ok.nc',
    'D:/data1/tos_Omon_HadGEM2-ES_esmHistorical_r1i1p1_1981-2005_nino34_ok.nc',
    'D:/data1/tos_Omon_NorESM1-M_historical_r1i1p1_198101-200512.nc',
    'D:/data1/4tos_Omon_GFDL-ESM2M_historical_r1i1p1_198101-200512.nc'
    ]
    
    names=['CNRM', 'IPSL', 'HadGEM2', 'NorESM1', 'GFDL' ]
    
    for i in np.arange(len(filepath)):
      
        print(i, names[i])
           
        dsx = xr.open_dataset(filepath[i])
        #print(dsx)
        #slice ini nans
        #tos_nino34 = ds.sel(i=slice(-5, 5), j=slice(190, 240))
        try:
            tos_nino34 = dsx.where(
                    (dsx.lat < 5) & (dsx.lat > -5) & 
                    (dsx.lon > 190) & (dsx.lon < 240), drop=True)
                    
        except:
          
            tos_nino34 = dsx.where(
                    (dsx.rlat < 5) & (dsx.rlat > -5) & 
                    (dsx.rlon > 190) & (dsx.rlon < 240), drop=True)
                    
        
        gb = tos_nino34.tos.groupby('time.month')
        tos_nino34_anom = gb - gb.mean(dim='time')
        #print(tos_nino34_anom)
        try: index_nino34 = tos_nino34_anom.mean(dim=['rlat', 'rlon'])
        except: 
            try: index_nino34 = tos_nino34_anom.mean(dim=['i', 'j'])
            except: index_nino34 = tos_nino34_anom.mean(dim=['lat', 'lon'])
        
      
        ir5 = index_nino34 #.rolling(time=5, center=True).mean()
       
       #karna i=2 hanya sampai 2005-11 maka
        times=obs_dataset.times
        if i==2: 
            nino_obs1=np.delete(nino_obs,-1)
            times=np.delete(times,-1)
            print( len(nino_obs),len(ir5))
        else:
            nino_obs1=nino_obs
            print(len(iod_obs),len(ir5))
        
        m0=nino_obs1.values
        m1=ir5.values
        #print(iod_obs)
        #print(ir5)
        bad = ~np.logical_or(np.isnan(m0), np.isnan(m1))
        x1=np.compress(bad, m0) 
        y1=np.compress(bad, m1)
        #print(x1)
        #print(y1)
        
        sd1=x1.std() #(skipna=None)
        #print(sd1)
        sd2=y1.std()
        s=sd2/sd1
        
        c,pp=pearsonr(x1, y1)        
        
        #T= (4*(1+c)**4)/(((s+1/s)**2)*((1+r0)**4))
        # Tanpa pakai **4 T naik dikit
        r0=1
        tt= (4*(1+c))/(((s+1/s)**2)*((1+r0)))
        
        ax[i+1].plot(times, ir5.values, label = names[i]) 
        #ax[i+1].set_title(names[i]+' ('+'%.2f'%tt+')', loc='left', y=0.5, x=0.005, fontsize='medium')
        ax[i+1].set_title(names[i]+' (sr='+'%.2f'%s + ', r=' + '%.2f'%c + ', T=' + '%.2f'%tt + ')', loc='left', y=0.85, x=0.005, fontsize='medium')
    
    import matplotlib.dates as mdates
    
    ax[5].xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 7)))
    #ax[3].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%b'))
    # Rotates and right-aligns the x labels so they don't crowd each other.
    for label in ax[5].get_xticklabels(which='major'):
        label.set(rotation=65, horizontalalignment='right')
    
    fig.subplots_adjust(bottom=.25)
    plt.subplots_adjust(hspace=.8,wspace=.05)
    
    
    plt.ylabel('                                                        SST Nino3.4 anomaly (K)')
    #plt.set_xticks([1981,1988,2005])
    #plt.title('SST anomaly obs & models over the Niño3.4 region');
    figname='ts_Nino_obs_5model_2'
    
    #plt.show()
    plt.savefig(workdir+figname) #, dpi=300, bbox_inches='tight')
    plt.show()

def ts_nino_iod(obs_dataset, workdir):
      
    import xarray as xr 
    import pandas as pd
    #from scipy.stats import pearsonr
    
    #jika ini gunakan cordex/obs untuk time yg 81-2005
    #d= pd.read_excel('D:/Disertasi3/enso_mon_1981-2005.xlsx')
    #dd= pd.read_excel('D:/Disertasi3/dmi_1981-2005.xlsx')
    
    #jika ini gunakan cordex/obs untuk time yg 76-2005
    d= pd.read_excel('D:/Disertasi3/enso2_1976-2005.xlsx')
    dd= pd.read_excel('D:/Disertasi3/iod2_1976-2005.xlsx')
    
    nino_obs = xr.DataArray( d['Value'],
    coords={'time': obs_dataset.times})
    
    gb = nino_obs.groupby('time.month')
    nino_obs = gb - gb.mean(dim='time')
    #nino_obs = nino_obs.rolling(time=5, center=True).mean()
    #nino_obs=nino_obs.groupby('time.season')['DJF']
    
    
    iod_obs = xr.DataArray( dd['Value'],
    coords={'time': obs_dataset.times})
    
    gb = iod_obs.groupby('time.month')
    iod_obs = gb - gb.mean(dim='time')
    #iod_obs = iod_obs.rolling(time=5, center=True).mean()
    #iod_obs=iod_obs.groupby('time.season')['DJF']
    
    
    fig, ax = plt.subplots(nrows=2, ncols=1 ,figsize=(6,4))
    
    #fig.subplots_adjust(bottom=.3)
    
    #ax[0].plot(obs_dataset.times, d['Value'], color='black', label = 'Nino34')    
    #ax[1].plot(obs_dataset.times, dd['Value'], color='blue', label = 'IOD') 
    ax[0].plot(obs_dataset.times, nino_obs.values, color='black', label = 'Nino34r')   
    ax[0].set_title('Nino3.4', loc='left', y=0.85, x=0.005, fontsize='medium')
    ax[0].axhline(0, color='k')
    ax[1].plot(obs_dataset.times, iod_obs.values, color='blue', label = 'IODr') 
    ax[1].set_title('DMI', loc='left', y=0.85, x=0.005, fontsize='medium')
    ax[1].axhline(0, color='k')
    #[ax[i].set_xticks([]) for i in [0,1,2]] 
    #for i in [0,1,2]: ax[i].set_xticks([]) #ini gak mau aneh
    plt.show()

def ts_nino_iod_season(obs_dataset, workdir):
      
    import xarray as xr 
    import pandas as pd
    #from scipy.stats import pearsonr
    
    #jika ini gunakan cordex/obs untuk time yg 81-2005
    #d= pd.read_excel('D:/Disertasi3/enso_mon_1981-2005.xlsx')
    #dd= pd.read_excel('D:/Disertasi3/dmi_1981-2005.xlsx')
    
    #jika ini gunakan cordex/obs untuk time yg 76-2005
    d= pd.read_excel('D:/Disertasi3/enso2_1976-2005.xlsx')
    dd= pd.read_excel('D:/Disertasi3/iod2_1976-2005.xlsx')
    
    nino_obs = xr.DataArray( d['Value'],
    coords={'time': obs_dataset.times})
    
    gb = nino_obs.groupby('time.month')
    nino_obs = gb - gb.mean(dim='time')
    #nino_obs = nino_obs.rolling(time=5, center=True).mean()
    nino_obs=nino_obs.groupby('time.season')['DJF']
    #print(nino_obs)
    
    iod_obs = xr.DataArray( dd['Value'],
    coords={'time': obs_dataset.times})
    
    gb = iod_obs.groupby('time.month')
    iod_obs = gb - gb.mean(dim='time')
    #iod_obs = iod_obs.rolling(time=5, center=True).mean()
    iod_obs=iod_obs.groupby('time.season')['DJF']
    
    
    fig, ax = plt.subplots(nrows=2, ncols=1 ,figsize=(6,4))
    
    #fig.subplots_adjust(bottom=.3)
    
    #ax[0].plot(obs_dataset.times, d['Value'], color='black', label = 'Nino34')    
    #ax[1].plot(obs_dataset.times, dd['Value'], color='blue', label = 'IOD') 
    ax[0].plot(nino_obs.month, nino_obs.values, color='black', label = 'Nino34r')   
    ax[0].set_title('Nino3.4', loc='left', y=0.85, x=0.005, fontsize='medium')
    ax[0].axhline(0, color='k')
    
    ax[1].plot(nino_obs.month, iod_obs.values, color='blue', label = 'IODr') 
    ax[1].set_title('DMI', loc='left', y=0.85, x=0.005, fontsize='medium')
    ax[1].axhline(0, color='k')
    #[ax[i].set_xticks([]) for i in [0,1,2]] 
    #for i in [0,1,2]: ax[i].set_xticks([]) #ini gak mau aneh
    plt.show()

def enso_zomean_cmip5(obs_dataset, workdir):
      
    import xarray as xr 
    import pandas as pd
    from scipy.stats import pearsonr
   
    d= pd.read_excel('D:/Disertasi3/enso_mon_1981-2005.xlsx')
    dd= pd.read_excel('D:/Disertasi3/dmi_1981-2005.xlsx')
    
    nino_obs = xr.DataArray( d['Value'],
    coords={'time': obs_dataset.times})
    gb = nino_obs.groupby('time.month')
    nino_obs = gb - gb.mean(dim='time')
    #nino_obs = nino_obs.rolling(time=5, center=True).mean()
    
    iod_obs = xr.DataArray( dd['Value'],
    coords={'time': obs_dataset.times})
    gb = iod_obs.groupby('time.month')
    iod_obs = gb - gb.mean(dim='time')
    #iod_obs = iod_obs.rolling(time=5, center=True).mean()
    
    fig, ax = plt.subplots(nrows=6, ncols=1 ,figsize=(6,4))
    fig2, ax2 = plt.subplots(nrows=6, ncols=1 ,figsize=(6,4))
    
    #fig.subplots_adjust(bottom=.3)
    
    #ax[0].plot(obs_dataset.times, d['Value'], color='black', label = 'Nino34')    
    #ax[1].plot(obs_dataset.times, dd['Value'], color='blue', label = 'IOD') 
    ax[0].plot(obs_dataset.times, nino_obs.values, color='black', label = 'Nino34r')   
    ax[0].set_title('Obs', loc='left', y=0.85, x=0.005, fontsize='medium')
    #ax[1].plot(obs_dataset.times, iod_obs.values, color='blue', label = 'IODr') 
    #[ax[i].set_xticks([]) for i in [0,1,2]] 
    #for i in [0,1,2]: ax[i].set_xticks([]) ini gak mau aneh
    for i in [0,1,2,3,4]:ax[i].xaxis.set_major_locator(plt.NullLocator())
    for i in [0,1,2,3,4]:ax2[i].xaxis.set_major_locator(plt.NullLocator())
    
    
  
    filepath=[
    'D:/data1/tos_Omon_CNRM-CM5_historical_r1i1p1_198101-200512.nc',
    'D:/data1/tos_Omon_IPSL-CM5A-LR_historicalNat_r1i1p1_1981-2005_nino34_ok.nc',
    'D:/data1/tos_Omon_HadGEM2-ES_esmHistorical_r1i1p1_1981-2005_nino34_ok.nc',
    'D:/data1/tos_Omon_NorESM1-M_historical_r1i1p1_198101-200512.nc',
    'D:/data1/4tos_Omon_GFDL-ESM2M_historical_r1i1p1_198101-200512.nc'
    ]
    
    names=['CNRM', 'IPSL', 'HadGEM2', 'NorESM1', 'GFDL' ]
    
    for i in np.arange(len(filepath)):
      
        print(i, names[i])
           
        dsx = xr.open_dataset(filepath[i])
        #print(dsx)
        #slice ini nans
        #tos_nino34 = ds.sel(i=slice(-5, 5), j=slice(190, 240))
        try:
            tos_nino34 = dsx.where(
                    (dsx.lat < 5) & (dsx.lat > -5) & 
                    (dsx.lon > 190) & (dsx.lon < 240), drop=True)
                    
        except:
          
            tos_nino34 = dsx.where(
                    (dsx.rlat < 5) & (dsx.rlat > -5) & 
                    (dsx.rlon > 190) & (dsx.rlon < 240), drop=True)
                    
        
        ds = tos_nino34.tos
      
        try: index_nino34 = ds.mean(dim=['time','rlat'])
        except: 
            try: index_nino34 = ds.mean(dim=['time','j'])
            except: index_nino34 = ds.mean(dim=['time','lat'])
        
        print(index_nino34)
        try: ax[i+1].plot( index_nino34.i, index_nino34.values, lw=1, label = names[i])
        except:
            try: ax[i+1].plot(index_nino34.rlon,index_nino34.values, lw=1, label = names[i])
            except: ax[i+1].plot(index_nino34.lon, index_nino34.values, lw=1, label = names[i])
       
        #lat
        try: index_nino34 = ds.mean(dim=['time','rlon'])
        except: 
            try: index_nino34 = ds.mean(dim=['time','i'])
            except: index_nino34 = ds.mean(dim=['time','lon'])
        
        print(index_nino34)
        try: ax2[i+1].plot(index_nino34.values, index_nino34.j, lw=1, label = names[i])
        except:
            try: ax2[i+1].plot(index_nino34.values, index_nino34.rlat, lw=1, label = names[i])
            except: ax2[i+1].plot(index_nino34.values, index_nino34.lat, lw=1, label = names[i])
        
    plt.subplots_adjust(hspace=.2,wspace=.05)
    figname='zm_Nino_5model'
    
    #plt.show()
    plt.savefig(workdir+figname) #, dpi=300, bbox_inches='tight')
    plt.show()

def iod_ts_cmip5(obs_dataset, workdir):
      
    import xarray as xr 
    import pandas as pd
    from scipy.stats import pearsonr
 
    dd= pd.read_excel('D:/Disertasi3/dmi_1981-2005.xlsx')
      
    iod_obs = xr.DataArray( dd['Value'],
    coords={'time': obs_dataset.times})
    gb = iod_obs.groupby('time.month')
    iod_obs = gb - gb.mean(dim='time')
    iod_obs = iod_obs.rolling(time=5, center=True).mean()
    
    fig, ax = plt.subplots(nrows=6, ncols=1)
    
    ax[0].plot(obs_dataset.times, iod_obs.values, color='blue', label = 'IODr') 
    ax[0].set_title('Obs', loc='left', y=0.85, x=0.005, fontsize='medium')
   
    #[ax[i].set_xticks([]) for i in [0,1,2]] 
    #for i in [0,1,2]: ax[i].set_xticks([]) ini gak mau aneh
    for i in [0,1,2,3,4]:ax[i].xaxis.set_major_locator(plt.NullLocator())
    
  
    filepath=[
            'D:/data1/tos_Omon_CNRM-CM5_historical_r1i1p1_198101-200512.nc',
            'D:/data1/tos_Omon_IPSL-CM5A-LR_historicalNat_r1i1p1_1981-2005.nc',
            'D:/data1/tos_Omon_HadGEM2-ES_esmHistorical_r1i1p1_1981-2005_11.nc',
            'D:/data1/tos_Omon_NorESM1-M_historical_r1i1p1_198101-200512.nc',
            'D:/data1/4tos_Omon_GFDL-ESM2M_historical_r1i1p1_198101-200512.nc'
            ]
    
    names=['CNRM', 'IPSL', 'HadGEM2','NorESM1', 'GFDL']
    
    for i in np.arange(len(filepath)):
        print(i, names[i])
        dsx = xr.open_dataset(filepath[i])
  
        try:
            tos_w = dsx.where(
                    (dsx.lat < 10) & (dsx.lat > -10) & 
                    (dsx.lon > 50) & (dsx.lon < 70), drop=True)
        #GFDL pakai rlat,rlon             
        except:
            try:
                tos_w = dsx.where(
                    (dsx.rlat < 10) & (dsx.rlat > -10) & 
                    (dsx.rlon > 50) & (dsx.rlon < 70), drop=True)
            except:
                #ini tidak perlu
                print('i,j terpakai')
                tos_w = dsx.where(
                    (dsx.j < 10) & (dsx.j > -10) & 
                    (dsx.i > 50) & (dsx.i < 70), drop=True)   
        try:
            tos_e = dsx.where(
                    (dsx.lat < 0) & (dsx.lat > -10) & 
                    (dsx.lon > 90) & (dsx.lon < 110), drop=True)         
        except:
            try:
                tos_e = dsx.where(
                    (dsx.rlat < 0) & (dsx.rlat > -10) & 
                    (dsx.rlon > 90) & (dsx.rlon < 110), drop=True) 
            except:
                print('i,j terpakai')
                tos_e = dsx.where(
                    (dsx.j < 0) & (dsx.j > -10) & 
                    (dsx.i > 90) & (dsx.i < 110), drop=True) 
        
        tos_nino34=tos_w 
        
        gb = tos_nino34.tos.groupby('time.month')
        tos_nino34_anom = gb - gb.mean(dim='time')
        #print(tos_nino34_anom)
        try: index_nino34 = tos_nino34_anom.mean(dim=['rlat', 'rlon'])
        except:
            try: index_nino34 = tos_nino34_anom.mean(dim=['lat', 'lon'])
            except: index_nino34 = tos_nino34_anom.mean(dim=['i', 'j'])

        ir5_w = index_nino34.rolling(time=5, center=True).mean()
        
        tos_nino34=tos_e 
        
        gb = tos_nino34.tos.groupby('time.month')
        tos_nino34_anom = gb - gb.mean(dim='time')
        #print(tos_nino34_anom)
        try: index_nino34 = tos_nino34_anom.mean(dim=['rlat', 'rlon'])
        except:
            try: index_nino34 = tos_nino34_anom.mean(dim=['lat', 'lon'])
            except: index_nino34 = tos_nino34_anom.mean(dim=['i', 'j'])

        ir5_e = index_nino34.rolling(time=5, center=True).mean()
        
        ir5=ir5_w -ir5_e
       
        #karna i=2 hanya sampai 2005-11 maka
        times=obs_dataset.times
        if i==2: 
            iod_obs1=np.delete(iod_obs,-1)
            times=np.delete(times,-1)
            print( len(iod_obs),len(ir5))
        else:
            iod_obs1=iod_obs
            print(len(iod_obs),len(ir5))
        
        m0=iod_obs1.values
        m1=ir5.values
        #print(iod_obs)
        #print(ir5)
        bad = ~np.logical_or(np.isnan(m0), np.isnan(m1))
        x1=np.compress(bad, m0) 
        y1=np.compress(bad, m1)
        #print(x1)
        #print(y1)
        
        sd1=x1.std() #(skipna=None)
        #print(sd1)
        sd2=y1.std()
        s=sd2/sd1
        
        c,pp=pearsonr(x1, y1)        
        
        #T= (4*(1+c)**4)/(((s+1/s)**2)*((1+r0)**4))
        # Tanpa pakai **4 T naik dikit
        r0=1
        tt= (4*(1+c))/(((s+1/s)**2)*((1+r0)))
        
        ax[i+1].plot(times, ir5.values, label = names[i]) 
        #ax[i+1].set_title(names[i]+' ('+'%.2f'%tt+')', loc='left', y=0.5, x=0.005, fontsize='medium')
        ax[i+1].set_title(names[i]+' (sr='+'%.2f'%s + ', r=' + '%.2f'%c + ', T=' + '%.2f'%tt + ')', loc='left', y=0.85, x=0.005, fontsize='medium')
    import matplotlib.dates as mdates
    
    ax[5].xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 7)))
    #ax[3].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%b'))
    # Rotates and right-aligns the x labels so they don't crowd each other.
    for label in ax[5].get_xticklabels(which='major'):
        label.set(rotation=65, horizontalalignment='right')
    
    fig.subplots_adjust(bottom=.25)
    plt.subplots_adjust(hspace=.8,wspace=.05)
    
    plt.ylabel('                                                           SST IOD anomaly (K)')
    ax[5].yaxis.labelpad = 20

    
    figname='ts_IOD_obs_5model'
    
    #plt.show()
    plt.savefig(workdir+figname) #, dpi=300, bbox_inches='tight')
    plt.show()
   
   
   
def enso(obs_dataset, obs_name, model_datasets, model_names, workdir):
      
    import xarray as xr 
    import pandas as pd
    fig, ax = plt.subplots(nrows=1, ncols=3 ,figsize=(6,4))
    
    #--------------------------------------- SEA 200412 
    d= pd.read_excel('D:/Disertasi3/enso_mon_1981-2005.xlsx')
    #SEA:
    #d= pd.read_excel('D:/Disertasi3/enso_mon_1981-2004.xlsx')
    index_nino34 = xr.DataArray( d['Value'],
    coords={'time': obs_dataset.times})
    
    sdo= index_nino34.std()
    #print(index_nino34.std())
    ax[1].scatter(0, sdo, label='Obs',color='black')
    ax[1].axhline(sdo, linestyle = 'dashed', color='black')
    
    from scipy.stats import pearsonr
    ds = xr.DataArray(obs_dataset.values,
    coords={'time': obs_dataset.times,
    'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
    dims=["time", "lat", "lon"])
    
    ds0= ds.mean(dim=('lat','lon'))
    #print(ds0)
    
    md = ds0.groupby('time.month')
    d_anom = md - md.mean(dim='time')
    pr5 = d_anom.rolling(time=5, center=True).mean()
    y0=pr5.values  
    
    x0= index_nino34
    bad = ~np.logical_or(np.isnan(x0), np.isnan(y0))
    x1=np.compress(bad, x0) 
    y1=np.compress(bad, y0)
    
    r=pearsonr(x1, y1)[0]*-1
    ax[2].scatter(0, r, label='GPCP',color='black')

    
    #index_nino34.plot(size=8)
    #plt.legend(['anomaly', 'sst'])
    #plt.title('SST anomaly over the Niño 3.4 region');
    
    #figname='sst'
    #plt.savefig(workdir+figname, dpi=300, bbox_inches='tight')
    
    #fig=plt.figure(figsize=[8,6])
    #fig.subplots_adjust(right=.7)
    
    ax[0].plot(obs_dataset.times, d['Value'], color='black', label = 'Obs')
    ax[0].axhline(.5, linestyle = 'dashed', color='red')
    ax[0].axhline(-.5, linestyle = 'dashed', color='red')
    #ax.set_xticklabels(obs_dataset.times) error
    #exit()

    filepath=[
            'D:/data1/tos_Omon_CNRM-CM5_historical_r1i1p1_198101-200512.nc',
            'D:/data1/tos_Omon_HadGEM2-ES_esmHistorical_r1i1p1_1981-2005_nino34_ok.nc',
            'D:/data1/tos_Omon_NorESM1-M_historical_r1i1p1_198101-200512.nc',
            'D:/data1/4tos_Omon_GFDL-ESM2M_historical_r1i1p1_198101-200512.nc'
            ]
    # filepath=[
            # 'D:/data1/tos_Omon_CNRM-CM5_historical_r1i1p1_198101-200412.nc',
            # 'D:/data1/2tos_Omon_NorESM1-M_historical_r1i1p1_198101-200412.nc',
            # 'D:/data1/2tos_Omon_GFDL-ESM2M_historical_r1i1p1_198101-200412.nc']
    names=['CNRM', 'NorESM1', 'GFDL' ]
    n=[1,8,9]
    for i in np.arange(len(filepath)):
      
      
        print(i)
           
        dsx = xr.open_dataset(filepath[i])
        #print(dsx)
        #slice ini nans
        #tos_nino34 = ds.sel(i=slice(-5, 5), j=slice(190, 240))
        try:
            tos_nino34 = dsx.where(
                    (dsx.lat < 5) & (dsx.lat > -5) & 
                    (dsx.lon > 190) & (dsx.lon < 240), drop=True)
                    
        except:
            #jika sistem bukan 0-360 maka 
            #ubah 190 ke -170 dan 240 ke -120 
            #rumus 190-360=-170
            
            #ini salah unutk GFDL -280 to 80
            #tos_nino34 = dsx.where(
            #        (dsx.lat < 5) & (dsx.lat > -5) & 
            #         (dsx.lon > -170) & (dsx.lon < -120), drop=True)
                  
            tos_nino34 = dsx.where(
                    (dsx.rlat < 5) & (dsx.rlat > -5) & 
                    (dsx.rlon > 190) & (dsx.rlon < 240), drop=True)
                    
        
        gb = tos_nino34.tos.groupby('time.month')
        tos_nino34_anom = gb - gb.mean(dim='time')
        #print(tos_nino34_anom)
        try: index_nino34 = tos_nino34_anom.mean(dim=['rlat', 'rlon'])
        except: index_nino34 = tos_nino34_anom.mean(dim=['i', 'j'])
        # print('index_nino34=',index_nino34)
        # #smooth the anomalies with a 5-month running m.vvaleuean:
        #index_nino34_rolling_mean = index_nino34.rolling(time=5, center=True).mean()
        ir5 = index_nino34.rolling(time=5, center=True).mean()
        
        sdi= ir5.std()
        
        ax[0].plot(obs_dataset.times, ir5.values, label = names[i])
        ax[1].scatter(i+1, sdi, label=names[i])
        
        #for ii in [1,8,9]: #in np.arange(len(model_datasets)-):#
        dsi = xr.DataArray(model_datasets[n[i]].values,
        coords={'time': obs_dataset.times,
        'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
        dims=["time", "lat", "lon"])  
        
        dsi= dsi.mean(dim=('lat','lon'))
        
        md = dsi.groupby('time.month')
        d_anom = md - md.mean(dim='time')
        pr5 = d_anom.rolling(time=5, center=True).mean()
        y0=pr5.values
        
        x0= ir5
        bad = ~np.logical_or(np.isnan(x0), np.isnan(y0))
        x1=np.compress(bad, x0) 
        y1=np.compress(bad, y0)
        r=pearsonr(x1, y1)[0]*-1 
        ax[2].scatter(i+1, r, label=model_names[n[i]])
    
    #ax[0].legend(bbox_to_anchor=(1, .5), loc='best', prop={'size':10}, frameon=False) 
    
    #ax[0].legend(loc=0)
    ax[1].legend(loc=0)
    ax[2].legend(loc=0)
    ax[0].set_ylabel('SSTA Nino3.4 (K)')
    ax[1].set_ylabel('SSTA Nino3.4 stdev (K)')
    ax[2].set_ylabel('Correlation (SSTA vs prA)')
    #plt.set_xticks([1981,1988,2005])
    #plt.title('SST anomaly obs & models over the Niño3.4 region');
   
    plt.subplots_adjust(hspace=.05,wspace=.3)
    
    #plt.show()
    figname='sst4_'
    plt.savefig(workdir+figname+reg, dpi=300, bbox_inches='tight')   

def enso2(obs_dataset, obs_name, model_datasets, model_names, workdir):
      
    import xarray as xr 
    import pandas as pd
    
    fig, ax = plt.subplots(nrows=1, ncols=2 ,figsize=(6,4))
    #--------------------------------------- SEA 
    d= pd.read_excel('D:/Disertasi3/enso_mon_1981-2005.xlsx')
    #SEA:
    #d= pd.read_excel('D:/Disertasi3/enso_mon_1981-2004.xlsx')
    index_nino34 = xr.DataArray( d['Value'],
    coords={'time': obs_dataset.times})
    
    sdo= index_nino34.std()
    #print(index_nino34.std())
    ax[0].scatter(0, sdo, label='Obs',color='black')
    ax[0].axhline(sdo, linestyle = 'dashed', color='black')
 
    from scipy.stats import pearsonr
    ds = xr.DataArray(obs_dataset.values,
    coords={'time': obs_dataset.times,
    'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
    dims=["time", "lat", "lon"])
    
    ds0= ds.mean(dim=('lat','lon'))
    #print(ds0)
    
    md = ds0.groupby('time.month')
    d_anom = md - md.mean(dim='time')
    pr5 = d_anom.rolling(time=5, center=True).mean()
    y0=pr5.values  
    
    x0= index_nino34
    bad = ~np.logical_or(np.isnan(x0), np.isnan(y0))
    x1=np.compress(bad, x0) 
    y1=np.compress(bad, y0)
    
    r=pearsonr(x1, y1)[0]*-1
    ax[1].scatter(0, r, label='GPCP',color='black')
    ax[1].axhline(r, linestyle = 'dashed', color='black')

    
    #index_nino34.plot(size=8)
    #plt.legend(['anomaly', 'sst'])
    #plt.title('SST anomaly over the Niño 3.4 region');
    
    
    
    #fig2=plt.subplots(nrows=1, ncols=1 ,figsize=(6,4))
    #fig.subplots_adjust(right=.7)
    
    
    
    filepath=[
    'D:/data1/tos_Omon_CNRM-CM5_historical_r1i1p1_198101-200512.nc',
    'D:/data1/tos_Omon_IPSL-CM5A-LR_historicalNat_r1i1p1_1981-2005_nino34_ok.nc',
    'D:/data1/tos_Omon_HadGEM2-ES_esmHistorical_r1i1p1_1981-2005_nino34_ok.nc',
    'D:/data1/tos_Omon_NorESM1-M_historical_r1i1p1_198101-200512.nc',
    'D:/data1/4tos_Omon_GFDL-ESM2M_historical_r1i1p1_198101-200512.nc'
    ]
    
    names=['CNRM', 'IPSL', 'HadGEM2', 'NorESM1', 'GFDL']
    n=[1,3,4,8,9] #1,3,4,5,6,8,9
    for i in np.arange(len(filepath)):
      
      
        print(i)
           
        dsx = xr.open_dataset(filepath[i])
        #print(dsx)
        #slice ini nans
        #tos_nino34 = ds.sel(i=slice(-5, 5), j=slice(190, 240))
        try:
            tos_nino34 = dsx.where(
                    (dsx.lat < 5) & (dsx.lat > -5) & 
                    (dsx.lon > 190) & (dsx.lon < 240), drop=True)
                    
        except:
            #jika sistem bukan 0-360 maka 
            #ubah 190 ke -170 dan 240 ke -120 
            #rumus 190-360=-170
            #ini salah unutk GFDL -280 to 80
            #tos_nino34 = dsx.where(
            #        (dsx.lat < 5) & (dsx.lat > -5) & (dsx.lon > -170) & 
            #        (dsx.lon < -120), drop=True)
            tos_nino34 = dsx.where(
                    (dsx.rlat < 5) & (dsx.rlat > -5) & 
                    (dsx.rlon > 190) & (dsx.rlon < 240), drop=True)
                    
        
        gb = tos_nino34.tos.groupby('time.month')
        tos_nino34_anom = gb - gb.mean(dim='time')
        #print(tos_nino34_anom)
        try: index_nino34 = tos_nino34_anom.mean(dim=['rlat', 'rlon'])
        except:
              try: index_nino34 = tos_nino34_anom.mean(dim=['lat', 'lon'])
              except: index_nino34 = tos_nino34_anom.mean(dim=['i', 'j'])
        # print('index_nino34=',index_nino34)
        # #smooth the anomalies with a 5-month running m.vvaleuean:
        #index_nino34_rolling_mean = index_nino34.rolling(time=5, center=True).mean()
        ir5 = index_nino34.rolling(time=5, center=True).mean()
        
        sdi= ir5.std()
        
        
        ax[0].scatter(i+1, sdi, label=names[i])
        ax[0].set_xticks([])
        #ax[0].set_xlabel(names[i], fontsize=6,rotation=30)
        
        #for ii in [1,8,9]: #in np.arange(len(model_datasets)-):#
        dsi = xr.DataArray(model_datasets[n[i]].values,
        coords={'time': obs_dataset.times,
        'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
        dims=["time", "lat", "lon"])  
        
        dsi= dsi.mean(dim=('lat','lon'))
        
        md = dsi.groupby('time.month')
        d_anom = md - md.mean(dim='time')
        pr5 = d_anom.rolling(time=5, center=True).mean()
        #y0=pr5.values
        if i in [2]: 
            pr5=np.delete(pr5,-1)
            y0=pr5.values 
           
        else: 
            y0=pr5.values
        
        x0= ir5
        bad = ~np.logical_or(np.isnan(x0), np.isnan(y0))
        x1=np.compress(bad, x0) 
        y1=np.compress(bad, y0)
        r=pearsonr(x1, y1)[0]*-1 
        ax[1].scatter(i+1, r, label=model_names[n[i]])
 
    ax[1].set_xticks([])
    #ax[1].set_xlabel(names, fontsize=6,rotation=30)
    
    #ax[0].legend(bbox_to_anchor=(1, .5), loc='best', prop={'size':10}, frameon=False) 
    
    #ax[0].legend(loc=0)
    ax[0].legend(loc=0)
    #ax[1].legend(loc=0)
    
    ax[0].set_ylabel('SSTA Nino3.4 stdev (K)')
    ax[1].set_ylabel('Correlation (SSTA vs prA)')
    #plt.set_xticks([1981,1988,2005])
    #plt.title('SST anomaly obs & models over the Niño3.4 region');
   
    fig.subplots_adjust(hspace=.05,wspace=.3)
  
    
    #plt.show()
    figname='sst_stdev_corr_2'
    plt.savefig(workdir+figname+reg, dpi=300, bbox_inches='tight')
    plt.show()
    
def enso_stdev(obs_dataset, obs_name, model_datasets, model_names, workdir):
    #pr_seaLO no MME no EW
    
    import xarray as xr 
    import pandas as pd
    
    fig, ax = plt.subplots(nrows=1, ncols=3 ,figsize=(6,4))
    #--------------------------------------- SEA 
    d= pd.read_excel('D:/Disertasi3/enso_mon_1981-2005.xlsx')
    #SEA:
    #d= pd.read_excel('D:/Disertasi3/enso_mon_1981-2004.xlsx')
    index_nino34 = xr.DataArray( d['Value'],
    coords={'time': obs_dataset.times})
    
    x0= index_nino34
    
    sdo= index_nino34.std()
    #print(index_nino34.std())
    ax[0].scatter(0, sdo, label='Obs',color='black')
    ax[0].axhline(sdo, linestyle = 'dashed', color='black')
 
    from scipy.stats import pearsonr
    
    
    #era
    ds = xr.DataArray(model_datasets[0].values,
    coords={'time': obs_dataset.times,
    'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
    dims=["time", "lat", "lon"])
    
    ds0= ds.mean(dim=('lat','lon'))
    #print(ds0)
    
    md = ds0.groupby('time.month')
    d_anom = md - md.mean(dim='time')
    pr5 = d_anom.rolling(time=5, center=True).mean()
    y0=pr5.values  
    
    
    bad = ~np.logical_or(np.isnan(x0), np.isnan(y0))
    x1=np.compress(bad, x0) 
    y1=np.compress(bad, y0)
    
    r=pearsonr(x1, y1)[0]*-1
    
    ax[1].scatter(0, r, label='ERA5',color='black')
    ax[1].axhline(r, linestyle = 'dashed', color='black')
    
    ax[2].scatter(0, r, label='ERA5',color='black')
    ax[2].axhline(r, linestyle = 'dashed', color='black')
    
    #GPCP
    ds = xr.DataArray(obs_dataset.values,
    coords={'time': obs_dataset.times,
    'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
    dims=["time", "lat", "lon"])
    
    ds0= ds.mean(dim=('lat','lon'))
    #print(ds0)
    
    md = ds0.groupby('time.month')
    d_anom = md - md.mean(dim='time')
    pr5 = d_anom.rolling(time=5, center=True).mean()
    y0=pr5.values  
    
    
    bad = ~np.logical_or(np.isnan(x0), np.isnan(y0))
    x1=np.compress(bad, x0) 
    y1=np.compress(bad, y0)
    
    r=pearsonr(x1, y1)[0]*-1
    ax[1].scatter(0, r, label='GPCP',color='blue')
    ax[1].axhline(r, linestyle = 'dashed', color='blue')
    ax[2].scatter(0, r, label='GPCP',color='blue')
    ax[2].axhline(r, linestyle = 'dashed', color='blue')

    filepath=[
    'D:/data1/tos_Omon_CNRM-CM5_historical_r1i1p1_198101-200512.nc',
    'D:/data1/tos_Omon_IPSL-CM5A-LR_historicalNat_r1i1p1_1981-2005_nino34_ok.nc',
    'D:/data1/tos_Omon_HadGEM2-ES_esmHistorical_r1i1p1_1981-2005_nino34_ok.nc',
    'D:/data1/tos_Omon_NorESM1-M_historical_r1i1p1_198101-200512.nc',
    'D:/data1/4tos_Omon_GFDL-ESM2M_historical_r1i1p1_198101-200512.nc'
    ]
    
    names=['CNRM', 'IPSL', 'HadGEM2','NorESM1', 'GFDL']
    mm=['s','^', '*','D','v','P','H']
    for i in np.arange(len(filepath)):
        print(i)
        dsx = xr.open_dataset(filepath[i])
  
        try:
            tos_nino34 = dsx.where(
                    (dsx.lat < 5) & (dsx.lat > -5) & 
                    (dsx.lon > 190) & (dsx.lon < 240), drop=True)
                    
        except:
   
            tos_nino34 = dsx.where(
                    (dsx.rlat < 5) & (dsx.rlat > -5) & 
                    (dsx.rlon > 190) & (dsx.rlon < 240), drop=True)
                   
        gb = tos_nino34.tos.groupby('time.month')
        tos_nino34_anom = gb - gb.mean(dim='time')
        #print(tos_nino34_anom)
        try: index_nino34 = tos_nino34_anom.mean(dim=['rlat', 'rlon'])
        except:
              try: index_nino34 = tos_nino34_anom.mean(dim=['lat', 'lon'])
              except: index_nino34 = tos_nino34_anom.mean(dim=['i', 'j'])
 
        ir5 = index_nino34.rolling(time=5, center=True).mean()
        
        sdi= ir5.std()
         
        ax[0].scatter(i+1, sdi, label=names[i], marker=mm[i])
        ax[0].set_xticks([])
    
    
    n =[0,1,2,2,2,3,4] 
    nn=0
    #untuk mmew pakai nino mana? nino ew
    for i in [1,3,4,5,6,8,9]:
        print(i)
        dsx = xr.open_dataset(filepath[n[nn]])
  
        try:
            tos_nino34 = dsx.where(
                    (dsx.lat < 5) & (dsx.lat > -5) & 
                    (dsx.lon > 190) & (dsx.lon < 240), drop=True)
                    
        except:
   
            tos_nino34 = dsx.where(
                    (dsx.rlat < 5) & (dsx.rlat > -5) & 
                    (dsx.rlon > 190) & (dsx.rlon < 240), drop=True)
                   
        gb = tos_nino34.tos.groupby('time.month')
        tos_nino34_anom = gb - gb.mean(dim='time')
        #print(tos_nino34_anom)
        try: index_nino34 = tos_nino34_anom.mean(dim=['rlat', 'rlon'])
        except:
              try: index_nino34 = tos_nino34_anom.mean(dim=['lat', 'lon'])
              except: index_nino34 = tos_nino34_anom.mean(dim=['i', 'j'])
 
        ir5 = index_nino34.rolling(time=5, center=True).mean()
        
               
        #for ii in [1,8,9]: #in np.arange(len(model_datasets)-):#
        dsi = xr.DataArray(model_datasets[i].values,
        coords={'time': obs_dataset.times,
        'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
        dims=["time", "lat", "lon"])  
        
        dsi= dsi.mean(dim=('lat','lon'))
        
        md = dsi.groupby('time.month')
        d_anom = md - md.mean(dim='time')
        
        pr5 = d_anom.rolling(time=5, center=True).mean()
        y0=pr5.values
              
        
        bad = ~np.logical_or(np.isnan(x0), np.isnan(y0))
        x1=np.compress(bad, x0) 
        y1=np.compress(bad, y0)
        r=pearsonr(x1, y1)[0]*-1 
        ax[1].scatter(nn+1, r, label=model_names[i], marker=mm[nn])
        ax[1].set_xticks([])
        
        if i in [4,5,6]: 
            pr5=np.delete(pr5,-1)
            y0=pr5.values 
           
        else: 
            y0=pr5.values
        
        x01= ir5
        bad = ~np.logical_or(np.isnan(x01), np.isnan(y0))
        x1=np.compress(bad, x01) 
        y1=np.compress(bad, y0)
        r=pearsonr(x1, y1)[0]*-1 
        ax[2].scatter(nn+1, r, label=model_names[i], marker=mm[nn])
        ax[2].set_xticks([])
        
        nn=nn+1
   
    ax[2].legend(bbox_to_anchor=(.93, .5), loc='best', prop={'size':10}, frameon=False, handletextpad=0) 
    
    #ax[0].legend(loc=0)
    ax[0].legend(loc=0)
    #ax[1].legend(loc=0)
    
    ax[0].set_ylabel('Standart deviation (Nino.34 index) (K)')
    ax[1].set_ylabel('Correlation (Nino.34 index vs SEA rainfall anomaly)', labelpad=0)
    ax[0].set_xlabel('(a)')
    ax[1].set_xlabel('(b)')
    ax[2].set_xlabel('(c)')
    
    #plt.set_xticks([1981,1988,2005])
    #plt.title('SST anomaly obs & models over the Niño3.4 region');
   
    fig.subplots_adjust(hspace=.05,wspace=.3)
    fig.subplots_adjust(right=.85)
    #ax[1].subplots_adjust(left=.3) err
  
    
    #plt.show()
    figname='sst_stdev_corr_3'
    plt.savefig(workdir+figname+reg, dpi=300, bbox_inches='tight')
    plt.show()
    
def nino34_ens(obs_dataset, obs_name, model_datasets, model_names, workdir):
    from scipy.stats import pearsonr
    import xarray as xr
    import pandas as pd
    
    fig, ax = plt.subplots(nrows=1, ncols=2 ,figsize=(6,4))
    #--------------------------------------- SEA 
    d= pd.read_excel('D:/Disertasi3/enso_mon_1981-2005.xlsx')
    
    
    #index_nino34 = xr.DataArray(d['Value'])
    #sdo= index_nino34.std()
    
    
    nino_obs = xr.DataArray( d['Value'], 
        coords={'time': obs_dataset.times})
    gb = nino_obs.groupby('time.month')
    nino_obs = gb - gb.mean(dim='time')
    ir5_obs = nino_obs.rolling(time=5, center=True).mean()
    sdo= ir5_obs.std()
    
    
    #print(index_nino34.std())
    ax[0].scatter(0, sdo, label='Obs',color='black')
    ax[0].axhline(sdo, linestyle = 'dashed', color='black')
    
    
           
    
    for i in [0,1,2,3,4,5]:
        print(i, model_names[i])
        
        dsi = xr.DataArray(model_datasets[i].values,
        coords={'time': model_datasets[i].times,
        'lat': model_datasets[i].lats, 'lon': model_datasets[i].lons},
        dims=["time", "lat", "lon"])  
        
        dsi= dsi.mean(dim=('lat','lon'))
        
        md = dsi.groupby('time.month')
        d_anom = md - md.mean(dim='time')
        ir5 = d_anom.rolling(time=5, center=True).mean()
        
        sdi= ir5.std()
        if i==5: 
            x0=ir5
            print(model_names[i])
            ax[0].scatter(i, sdi, label=model_names[i],color='blue')
            ax[0].axhline(sdi, linestyle = 'dashed', color='blue')
        else:    
            ax[0].scatter(i, sdi, label=model_names[i])
        ax[0].set_xticks([])
   
  
    for i in [6,7,8,9]: #mmew pi ss t 678
        dsi = xr.DataArray(model_datasets[i].values,
        coords={'time': model_datasets[i].times,
        'lat': model_datasets[i].lats, 'lon': model_datasets[i].lons},
        dims=["time", "lat", "lon"])  
        
                
        dsi= dsi.mean(dim=('lat','lon'))
        
        md = dsi.groupby('time.month')
        d_anom = md - md.mean(dim='time')
        pr5 = d_anom.rolling(time=5, center=True).mean()
       
        pr5=np.delete(pr5,-1)
        y0=pr5.values
                
        bad = ~np.logical_or(np.isnan(x0), np.isnan(y0))
        x1=np.compress(bad, x0) 
        y1=np.compress(bad, y0)
               
        r=pearsonr(x1, y1)[0]*-1 
        
        
        if i==6:
            ax[1].scatter(0, r, label='MME',color='black')
            ax[1].axhline(r, linestyle = 'dashed', color='black')
        else:
            ax[1].scatter(i, r, label=model_names[i])
    ax[1].set_xticks([])
    ax[1].legend(bbox_to_anchor=(.99, .5), loc='best', prop={'size':10}, frameon=True) 
    ax[0].legend(loc=0)
    ax[1].legend(loc=0)
    
    ax[0].set_ylabel('SSTA Nino3.4 stdev (K)')
    ax[1].set_ylabel('Correlation (SSTA vs prA)')
    #plt.set_xticks([1981,1988,2005])
    #plt.title('SST anomaly obs & models over the Niño3.4 region');
   
    fig.subplots_adjust(hspace=.05,wspace=.3)
  
    
    #plt.show()
    figname='nino34_stdev_corr_mme2'
    plt.savefig(workdir+figname+reg, dpi=300, bbox_inches='tight')
    plt.show()

def nino34_ens_corr(obs_dataset, obs_name, model_datasets, model_names, workdir):
   
    
    import xarray as xr
    import pandas as pd
    from scipy.stats import pearsonr
    
    fig, ax = plt.subplots(nrows=2, ncols=4 ,figsize=(8,6))
    
    d= pd.read_excel('D:/Disertasi3/enso_mon_1981-2005.xlsx')
    
    nino_obs = xr.DataArray( d['Value'], coords={'time': obs_dataset.times})
    gb = nino_obs.groupby('time.month')
    nino_obs = gb - gb.mean(dim='time')
    ir5_obs = nino_obs.rolling(time=5, center=True).mean()
    x0=ir5_obs.values
   
   
    ds = xr.DataArray(obs_dataset.values,
    coords={'time': obs_dataset.times,
    'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
    dims=["time", "lat", "lon"])
    
    lat_min = ds.lat.min()
    lat_max = ds.lat.max()
    lon_min = ds.lon.min()
    lon_max = ds.lon.max()
   
    x,y = np.meshgrid(ds.lon, ds.lat)
    
    tes=0
    
    map0=ma.zeros((len(ds.lat),len(ds.lon)))
   
    
    for ii in np.arange(len(ds.lat)-tes):
        print(ii)
        for jj in np.arange(len(ds.lon)-tes):
            d=ds[:,ii,jj]
            md = d.groupby('time.month')
            d_anom = md - md.mean(dim='time')
            pr5 = d_anom.rolling(time=5, center=True).mean()
            y0=pr5.values
          
            bad = ~np.logical_or(np.isnan(x0), np.isnan(y0))
            x1=np.compress(bad, x0) 
            y1=np.compress(bad, y0)
       
            if x1.shape==(0,) or y1.shape==(0,):
                R='nan'
            else:
                R=pearsonr(x1, y1)[0]*-1
                #R=pearsonr(pr5, ir5)[0]*-1
            map0[ii,jj]=R
    m0=map0.flatten()
    
    m = Basemap(ax=ax[0,0], projection ='cyl', llcrnrlat = lat_min, urcrnrlat = lat_max,
                llcrnrlon = lon_min, urcrnrlon = lon_max, resolution = 'l', fix_aspect=False)
    m.drawcoastlines(linewidth=1)
    m.drawcountries(linewidth=1)
    m.drawstates(linewidth=0.5, color='w')
    
    max = ax[0,0].contourf(x,y,map0)
    #ax[0,nn].set_title(model_names[i])
    ax[0,0].set_title('GPCP', pad=3,fontsize=10)
    ax[0,0].set_yticks([-10,0,10,20])
    #ax[0].set_xticks([100,120,140])
    ax[0,0].yaxis.set_tick_params(labelsize=7)
    #ax[0].xaxis.set_tick_params(labelsize=7)
    for i in [1,2,3]: ax[0,i].axis('off')
    
    #tos mme
    i=5
    dsi = xr.DataArray(model_datasets[i].values,
        coords={'time': model_datasets[i].times,
        'lat': model_datasets[i].lats, 'lon': model_datasets[i].lons},
        dims=["time", "lat", "lon"])  
        
    dsi= dsi.mean(dim=('lat','lon'))
        
    md = dsi.groupby('time.month')
    d_anom = md - md.mean(dim='time')
    ir5 = d_anom.rolling(time=5, center=True).mean()
    x0=ir5.values
    
    map1=ma.zeros((len(ds.lat),len(ds.lon)))
  
    r0=1
    #for n in pilih_nino:
    for i in [6,7,8,9]:
        
        print(model_names[i])
        
        dsi = xr.DataArray(model_datasets[i].values,
        coords={'time': model_datasets[i].times,
        'lat': model_datasets[i].lats, 'lon': model_datasets[i].lons},
        dims=["time", "lat", "lon"])  
        
               
        
        m = Basemap(ax=ax[1,i-6], projection ='cyl', llcrnrlat = lat_min, urcrnrlat = lat_max,
            llcrnrlon = lon_min, urcrnrlon = lon_max, resolution = 'l', fix_aspect=False)
        m.drawcoastlines(linewidth=1)
        m.drawcountries(linewidth=1)
        m.drawstates(linewidth=0.5, color='w')
            
        map1=ma.zeros((len(dsi.lat),len(dsi.lon)))
        
        for ii in np.arange(len(dsi.lat)-tes):
            print(ii)
            for jj in np.arange(len(dsi.lon)-tes):
                d=dsi[:,ii,jj]
                md = d.groupby('time.month')
                d_anom = md - md.mean(dim='time')
                pr5 = d_anom.rolling(time=5, center=True).mean()
               
                pr5=np.delete(pr5,-1)
                y0=pr5.values
              
                bad = ~np.logical_or(np.isnan(x0), np.isnan(y0))
                x1=np.compress(bad, x0) 
                y1=np.compress(bad, y0)
                #print(x,y)
                #print(x.shape,y.shape)
                if x1.shape==(0,) or y1.shape==(0,):
                    R='nan'
                else:
                    R=pearsonr(x1, y1)[0]*-1
                    #R=pearsonr(pr5, ir5)[0]*-1
                map1[ii,jj]=R
                #if i==0: m0=map1.flatten()
                
        m1=map1.flatten()
        bad = ~np.logical_or(np.isnan(m0), np.isnan(m1))
        x1=np.compress(bad, m0) 
        y1=np.compress(bad, m1)
        
        sd1=x1.std() #(skipna=None)
        #print(sd1)
        sd2=y1.std()
        s=sd2/sd1
        
        c,pp=pearsonr(x1.flatten() , y1.flatten())        
        
        #T= (4*(1+c)**4)/(((s+1/s)**2)*((1+r0)**4))
        # Tanpa pakai **4 T naik dikit
        tt= (4*(1+c))/(((s+1/s)**2)*((1+r0)))
        #T.append(round(tt,2))#taylor score
                
                
        #print('map1=',map1)
        max = ax[1,i-6].contourf(x,y,map1)
        #ax[0,nn].set_title(model_names[i])
        ax[1,i-6].set_title(model_names[i]+'('+'%.2f'%tt+')', pad=5,fontsize=10)
        ax[1,0].set_yticks([-10,0,10,20])
        ax[1,i-6].set_xticks([100,120,140])
        ax[1,i-6].xaxis.set_tick_params(labelsize=7)
        ax[1,0].yaxis.set_tick_params(labelsize=7)
 
    plt.subplots_adjust(hspace=.15,wspace=.12)
    #cax = fig.add_axes([0.91, 0.5, 0.015, 0.4])
    cax = fig.add_axes([0.4, 0.6, 0.4, 0.04]) #horisontal
    #plt.colorbar(max, cax = cax) 
    cbar=plt.colorbar(max, cax = cax, extend='both', orientation='horizontal') 
    cbar.ax.tick_params(labelsize=7)
    
    file_name='Corr_nino34_tos_mme_'+reg
    fig.savefig(workdir+file_name,dpi=300,bbox_inches='tight')
    plt.show()
        
        


def corr_enso(obs_dataset, obs_name, model_datasets, model_names, workdir):
    
    #Enso model vs nino obs
    import pandas as pd
    import xarray as xr
       
    d= pd.read_excel('D:/Disertasi3/enso_mon_1981-2005.xlsx')
    
    ds = xr.DataArray( d['Value'],
    coords={'time': obs_dataset.times})
    
    gb = ds.groupby('time.month')
    index_nino34 = gb - gb.mean(dim='time')
    #print(nino34_anom)
    
    # #smooth the anomalies with a 5-month running mean:
    #index_nino34_rolling_mean = index_nino34.rolling(time=5, center=True).mean()
    ir5 = index_nino34.rolling(time=5, center=True).mean()
    
    index_nino34.plot(size=8)
    ir5.plot()
    plt.legend(['anomaly', '5-month running mean anomaly'])
    plt.title('SST anomaly over the Niño 3.4 region');
    
    plt.savefig(workdir+'nino34_obs', dpi=300, bbox_inches='tight')
    #nino34yr=index_nino34.groupby('time.year').mean()
    #plt.show()
    #exit()
    
    ds = xr.DataArray(obs_dataset.values,
    coords={'time': obs_dataset.times,
    'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
    dims=["time", "lat", "lon"])
                
    # #ds = ds.groupby('time.year').sum() 
    
    lat_min = ds.lat.min()
    lat_max = ds.lat.max()
    lon_min = ds.lon.min()
    lon_max = ds.lon.max()
    x,y = np.meshgrid(ds.lon, ds.lat)
   
    from scipy.stats import pearsonr
    
    tes=0
    plot='SEA'
    #geser plot for SEA
    if plot=='SEA':
        lat_min1 = 0.22*3
        lat_max1 = 0.22*6
        lon_min1 = 0.22*4
        lon_max1 = 0.22*3
    
    if plot=='sum':
        lat_min1 = 0.22*1
        lat_max1 = 0.22*3
        lon_min1 = 0.22*3
        lon_max1 = 0.22*1
    
    #khusus obs gpcp
    fig, ax = plt.subplots(nrows=3, ncols=5,figsize=(8,6))
    
    m = Basemap(ax=ax[0,0], projection ='cyl', 
        llcrnrlat = lat_min+lat_min1, 
        urcrnrlat = lat_max-lat_max1,
        llcrnrlon = lon_min+lon_min1, 
        urcrnrlon = lon_max-lon_max1, 
        resolution = 'l', fix_aspect=False)
    m.drawcoastlines(linewidth=1)
    m.drawcountries(linewidth=1)
    m.drawstates(linewidth=0.5, color='w')

    map0=ma.zeros((len(ds.lat),len(ds.lon)))
    vmin=-0.5
    vmax=1
    levels = np.linspace(vmin,vmax,16)
    norm = plt.Normalize(vmin, vmax)
    mean=[]
    
    x0=ir5.values
    #x1=index_nino34.values
    r0=1
    print('GPCP')
    for i in np.arange(len(ds.lat)-tes):
        print (i)
        for j in np.arange(len(ds.lon)-tes):
            dd=ds[:,i,j]
            md = dd.groupby('time.month')
            d_anom = md - md.mean(dim='time')
            #untuk hujan mean bisa sum ?? untuk suhu tidak bisa sum
            pr5 = d_anom.rolling(time=5, center=True).mean()
            y0=pr5.values
                   
            bad = ~np.logical_or(np.isnan(x0), np.isnan(y0))
            x1=np.compress(bad, x0) 
            y1=np.compress(bad, y0)
     
            if x1.shape==(0,) or y1.shape==(0,):
                R='nan'
            else:
                R=pearsonr(x1, y1)[0]*-1
          
            map0[i,j]=R
    m0=map0.flatten()
    #me=np.nanmean(map1)
    #mak=np.nanmax(map1)
    #print('max_R=','%.2f'%mak)
    max = ax[0,0].contourf(x,y,map0,levels=levels,norm=norm)
    ax[0,0].set_title('GPCP',fontsize=8)
    ax[0,0].set_yticks([-10,0,10,20])
    #ax.set_xticks([90,100,110,120,130,140])
    ax[0,0].yaxis.set_tick_params(labelsize=6)
    
    #cax = fig.add_axes([0.91, 0.45, 0.015, 0.4])
    #plt.colorbar(max, cax = cax) 
    
    #fig.savefig(workdir+reg+'gpcp_enso',dpi=300,bbox_inches='tight')
    #plt.show()
    #exit()
    # ini ?? khusus zonal jika ingin obs=2 dan MMEW not included
    #model_datasets=np.delete(model_datasets,[1, -1])
    #model_names=np.delete(model_names,[1, -1])
    
    #fig, ax = plt.subplots(nrows=2, ncols=5,figsize=(16,6))
    model_datasets=np.delete(model_datasets,[-1])
    model_names=np.delete(model_names,[-1])
    mean=[]
    #-3 karna mengadung mmew 3
    for i in np.arange(len(model_datasets)-3):#
        print(i, len(model_datasets)-3)
        print (model_names[i])
        map1=ma.zeros((len(ds.lat),len(ds.lon)))
        if i<4: ax[0,1+i].axis('off')
        #i=1
        dsi = xr.DataArray(model_datasets[i].values,
        coords={'time': obs_dataset.times,
        'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
        dims=["time", "lat", "lon"])  
        
        #dsi = dsi.groupby('time.year').sum() 
        
        #ds = ds.groupby('time.year').sum() 
   
        if i<5:
            m = Basemap(ax=ax[1,i], projection ='cyl', llcrnrlat = lat_min, urcrnrlat = lat_max,
                llcrnrlon = lon_min, urcrnrlon = lon_max, resolution = 'l', fix_aspect=False)
            m.drawcoastlines(linewidth=1)
            m.drawcountries(linewidth=1)
            m.drawstates(linewidth=0.5, color='w')
        
            x,y = np.meshgrid(ds.lon, ds.lat)
        
            #from scipy.stats import pearsonr
        
            
            x0=ir5.values
            for ii in np.arange(len(ds.lat)-tes):
                print(ii)
                for jj in np.arange(len(ds.lon)-tes):
                    d=dsi[:,ii,jj]
                    md = d.groupby('time.month')
                    d_anom = md - md.mean(dim='time')
                    pr5 = d_anom.rolling(time=5, center=True).mean()
                    y0=pr5.values
                   
                    bad = ~np.logical_or(np.isnan(x0), np.isnan(y0))
                    x1=np.compress(bad, x0) 
                    y1=np.compress(bad, y0)
                  
                    if x1.shape==(0,) or y1.shape==(0,):
                        R='nan'
                    else:
                        R=pearsonr(x1, y1)[0]*-1
                        #R=pearsonr(pr5, ir5)[0]*-1
                    map1[ii,jj]=R
            
            
            m1=map1.flatten()
            bad = ~np.logical_or(np.isnan(m0), np.isnan(m1))
            x1=np.compress(bad, m0) 
            y1=np.compress(bad, m1)
            
            sd1=x1.std() #(skipna=None)
            #print(sd1)
            sd2=y1.std()
            s=sd2/sd1
            
            c,pp=pearsonr(x1.flatten() , y1.flatten())        
            
            #T= (4*(1+c)**4)/(((s+1/s)**2)*((1+r0)**4))
            # Tanpa pakai **4 T naik dikit
            tt= (4*(1+c))/(((s+1/s)**2)*((1+r0)))
           
            max = ax[1,i].contourf(x,y,map1,levels=levels,norm=norm)
            ax[1,i].set_title(model_names[i]+'('+'%.2f'%tt+')',fontsize=8)
            ax[1,0].set_yticks([-10,0,10,20])
            ax[1,0].yaxis.set_tick_params(labelsize=6)
            #ax[0,1+i].set_xticks([90,100,110,120,130,140])
            
        else:
            m = Basemap(ax=ax[2,i-5], projection ='cyl', llcrnrlat = lat_min, urcrnrlat = lat_max,
                llcrnrlon = lon_min, urcrnrlon = lon_max, resolution = 'l', fix_aspect=False)
            m.drawcoastlines(linewidth=1)
            m.drawcountries(linewidth=1)
            m.drawstates(linewidth=0.5, color='w')
        
            x,y = np.meshgrid(ds.lon, ds.lat)
        
            from scipy.stats import pearsonr
        
            map1=ma.zeros((len(ds.lat),len(ds.lon)))

            for ii in np.arange(len(ds.lat)-tes):
                print(ii)
                for jj in np.arange(len(ds.lon)-tes):
                    d=dsi[:,ii,jj]
                    md = d.groupby('time.month')
                    d_anom = md - md.mean(dim='time')
                    pr5 = d_anom.rolling(time=5, center=True).mean()
                    y0=pr5.values
                   
                    bad = ~np.logical_or(np.isnan(x0), np.isnan(y0))
                    x1=np.compress(bad, x0) 
                    y1=np.compress(bad, y0)
                  
                    if x1.shape==(0,) or y1.shape==(0,):
                        R='nan'
                    else:
                        R=pearsonr(x1, y1)[0]*-1
                        #R=pearsonr(pr5, ir5)[0]*-1
                    map1[ii,jj]=R
            m1=map1.flatten()
            bad = ~np.logical_or(np.isnan(m0), np.isnan(m1))
            x1=np.compress(bad, m0) 
            y1=np.compress(bad, m1)
            
            sd1=x1.std() #(skipna=None)
            #print(sd1)
            sd2=y1.std()
            s=sd2/sd1
            
            c,pp=pearsonr(x1.flatten() , y1.flatten())        
            
            #T= (4*(1+c)**4)/(((s+1/s)**2)*((1+r0)**4))
            # Tanpa pakai **4 T naik dikit
            tt= (4*(1+c))/(((s+1/s)**2)*((1+r0)))
            
            max = ax[2,i-5].contourf(x,y,map1,levels=levels,norm=norm)
            ax[2,i-5].set_title(model_names[i]+'('+'%.2f'%tt+')',fontsize=8)
            ax[2,0].set_yticks([-10,0,10,20])
            ax[2,i-5].set_xticks([100,120,140])
            ax[2,i-5].xaxis.set_tick_params(labelsize=6)
            ax[2,0].yaxis.set_tick_params(labelsize=6)
    
        #mean.append(round(me,2))
        #print('mean_kopi ke excel',mean)
    # print('s_Taylor=',mean)
    # plt.subplots_adjust(hspace=.2,wspace=.025)
    # cax = fig.add_axes([0.91, 0.48, 0.015, 0.4])
    # plt.colorbar(max, cax = cax) 
    
    plt.subplots_adjust(hspace=.25,wspace=.12)
    #cax = fig.add_axes([0.91, 0.5, 0.015, 0.4])
    cax = fig.add_axes([0.4, 0.7, 0.4, 0.04]) #horisontal
    #plt.colorbar(max, cax = cax) 
    cbar=plt.colorbar(max, cax = cax, extend='both', orientation='horizontal') 
    cbar.ax.tick_params(labelsize=7)
    
    file_name=reg+'Corr_nino34_ir5_2'
    fig.savefig(workdir+file_name,dpi=300,bbox_inches='tight')
    plt.show()
              
 
def corr_enso3_season(obs_dataset, obs_name, model_datasets, model_names, workdir):
    nino_obs=[0.09,1.55,-0.47,-0.68,-0.33,0.93,0.63,-1.23,0.02,0.38,1.11,0.23,0.21,
          0.57,-0.65,-0.08,1.63,-1.22,-1.18,-0.49,0.00,0.64,0.28,0.53]
    #temporal
    #nino34_model => obs 
    #bulanan ? 3bulanan?
    #nino34r5 dan obsr5?
    #resolusi mod-obs sama? res beda hasil sama?
    
    import xarray as xr
    #'D:/data1/tos_Omon_CNRM_sellonlatbox185_245_-10_10.nc',
    
    
        
        #print(s34)
        #is34=s34['JJA']
        # #smooth the anomalies with a 5-month running mean:
        #index_nino34_rolling_mean = index_nino34.rolling(time=5, center=True).mean()
        #ir5 = index_nino34.rolling(time=5, center=True).mean()
    
    # index_nino34.plot(size=8)
    # index_nino34_rolling_mean.plot()
    # plt.legend(['anomaly', '5-month running mean anomaly'])
    # plt.title('SST anomaly over the Niño 3.4 region');
    # figname='nino34'
    # plt.savefig(workdir+figname, dpi=300, bbox_inches='tight')
    #nino34yr=index_nino34.groupby('time.year').mean()
   
    import pandas as pd
    #--------------------------------------- SEA 200412 
    d= pd.read_excel('D:/Disertasi3/enso_mon_1981-2005.xlsx')
    #SEA:
    #d= pd.read_excel('D:/Disertasi3/enso_mon_1981-2004.xlsx')
    ds = xr.DataArray( d['Value'],
    coords={'time': obs_dataset.times})
    nino_obs=ds.groupby('time.season')
    #print(nino_obs)
    
    ds = xr.DataArray(obs_dataset.values,
    coords={'time': obs_dataset.times,
    'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
    dims=["time", "lat", "lon"])
    print(ds)
                
    # #ds = ds.groupby('time.year').sum() 
    # mds = ds.groupby('time.month')
    # ds_anom = mds - mds.mean(dim='time')
    # #print(tos_nino34_anom)
    # index_nino34 = ds_anom.mean(dim=['lat', 'lon'])
    
    lat_min = ds.lat.min()
    lat_max = ds.lat.max()
    lon_min = ds.lon.min()
    lon_max = ds.lon.max()
   
    fig, ax = plt.subplots(nrows=2, ncols=5 ,figsize=(6,4))
    
    x,y = np.meshgrid(ds.lon, ds.lat)
    
    from scipy.stats import pearsonr
    
    
    tes=0 #hitung dikurangi 50 step untuk tes
    #OBS
    map1=ma.zeros((len(ds.lat),len(ds.lon)))
    n=0
    for musim in ['DJF', 'JJA']:
        m = Basemap(ax=ax[n,0], projection ='cyl', llcrnrlat = lat_min, urcrnrlat = lat_max,
            llcrnrlon = lon_min, urcrnrlon = lon_max, resolution = 'l', fix_aspect=False)
        m.drawcoastlines(linewidth=1)
        m.drawcountries(linewidth=1)
        m.drawstates(linewidth=0.5, color='w')
        for i in np.arange(len(ds.lat)-tes):
            print (i)
            for j in np.arange(len(ds.lon)-tes):
                d=ds[:,i,j]
                md = d.groupby('time.month')
                #print(md)
                d_anom = md - md.mean(dim='time')
               
                pr=d_anom .groupby('time.season')
                
                x0=nino_obs[musim].values
                y0=pr[musim].values
                #print(x0.shape)
                #print(y0.shape)
                #print(x0)
                #print(y0)
                bad = ~np.logical_or(np.isnan(x0), np.isnan(y0))
                x1=np.compress(bad, x0) 
                y1=np.compress(bad, y0)
                #print(x,y)
                #print(x.shape,y.shape)
                if x1.shape==(0,) or y1.shape==(0,):
                    R='nan'
                else:
                    R=pearsonr(x1, y1)[0]*-1
                    #R=pearsonr(pr5, ir5)[0]*-1
                map1[i,j]=R
        #print('map1=',map1)
        #print('map1.shp=',map1.shape)
        
        max1 = ax[n,0].contourf(x,y,map1)
        ax[0,0].set_title('GPCP',fontsize=8)
        ax[n,0].set_ylabel(musim)
        #ax[0,0].set_yticks([-10,0,10,20])
        #ax[0,0].set_xticks([90,100,110,120,130,140])
        n=n+1
    
    #fig.savefig(workdir+'nino34_tes',dpi=300,bbox_inches='tight')
    #exit()
    # ini ?? khusus zonal jika ingin obs=2 dan MMEW not included
    #model_datasets=np.delete(model_datasets,[1, -1])
    #model_names=np.delete(model_names,[1, -1])
    #model_datasets=np.delete(model_datasets,[-1])
    #model_names=np.delete(model_names,[-1])
    #Datasets: ['GPCP', 'ERA5', 1'CNRM_a', 2'ECE_b', 3'IPSL_b', 
    #4'HadGEM2_d', 5'HadGEM2_c', 6'HadGEM2_a', 7'MPI_c', 8'NorESM1_d', 
    #9'GFDL_b', 10'MME']
    #MODEL
    n=0
    from scipy.stats import pearsonr
       #nino34 model
    # filepath=[
            # 'D:/data1/tos_Omon_CNRM-CM5_historical_r1i1p1_198101-200512.nc',
            # 'D:/data1/tos_Omon_NorESM1-M_historical_r1i1p1_198101-200512.nc',
            # 'D:/data1/2tos_Omon_GFDL-ESM2M_historical_r1i1p1_198101-200512.nc']
    filepath=[
            'D:/data1/tos_Omon_CNRM-CM5_historical_r1i1p1_198101-200412.nc',
            'D:/data1/2tos_Omon_NorESM1-M_historical_r1i1p1_198101-200412.nc',
            'D:/data1/2tos_Omon_GFDL-ESM2M_historical_r1i1p1_198101-200412.nc']
    #cek=0
    #hapus md selain 0189 
    for musim in ['DJF', 'JJA']:
        for i in np.arange(len(model_datasets)):
            if i==0 or i==1 or i==8 or i==9:  
                print(i)
                print ('model=',model_names[i])
             
                if i>0:
                    f=0
                    if i==8: f=1
                    if i==9: f=2
                    #print('f=',f)
                    dsx = xr.open_dataset(filepath[f])
                    #print(dsx)
                    #slice ini nans
                    #tos_nino34 = ds.sel(i=slice(-5, 5), j=slice(190, 240))
                    try:
                        tos_nino34 = dsx.where(
                                (dsx.lat < 5) & (dsx.lat > -5) & (dsx.lon > 190) & 
                                (dsx.lon < 240), drop=True)
                    except:
                        tos_nino34 = dsx.where(
                                (dsx.lat < 5) & (dsx.lat > -5) & (dsx.lon > -170) & 
                                (dsx.lon < -120), drop=True)

                    
                    gb = tos_nino34.tos.groupby('time.month')
                    tos_nino34_anom = gb - gb.mean(dim='time')
                    #print(tos_nino34_anom)
                    try: index_nino34 = tos_nino34_anom.mean(dim=['rlat', 'rlon'])
                    except: index_nino34 = tos_nino34_anom.mean(dim=['i', 'j'])
                    # print('index_nino34=',index_nino34)
                    s34=index_nino34.groupby('time.season')
               
                
                #model
                dsi = xr.DataArray(model_datasets[i].values,
                coords={'time': obs_dataset.times,
                'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
                dims=["time", "lat", "lon"])  
                
                f2=0
                if i==1: f2=1
                if i==8: f2=2
                if i==9: f2=3
                m = Basemap(ax=ax[n,f2+1], projection ='cyl', llcrnrlat = lat_min, urcrnrlat = lat_max,
                    llcrnrlon = lon_min, urcrnrlon = lon_max, resolution = 'l', fix_aspect=False)
                m.drawcoastlines(linewidth=1)
                m.drawcountries(linewidth=1)
                m.drawstates(linewidth=0.5, color='w')
            
                #x,y = np.meshgrid(ds.lon, ds.lat)
            
                for ii in np.arange(len(ds.lat)-tes):
                    print(ii)
                    for jj in np.arange(len(ds.lon)-tes):
                        d=dsi[:,ii,jj]
                        md = d.groupby('time.month')
                        #print(md)
                        d_anom = md - md.mean(dim='time')
                        #pr5 = d_anom.rolling(time=5, center=True).mean()
                        pr=d_anom .groupby('time.season')
                       
                        ipr=pr[musim]
                        #nt = 12 if not config['season'] else nmon
                        #x0=is34.values  #if not i==0 x0=nino_obs[musim].values   
                        if i==0: x0=nino_obs[musim].values 
                        else: x0=s34[musim].values 
                        y0=ipr.values
                        #print(x0.shape)
                        #print(y0.shape)
                        #print(x0)
                        #print(y0)
                        bad = ~np.logical_or(np.isnan(x0), np.isnan(y0))
                        x1=np.compress(bad, x0) 
                        y1=np.compress(bad, y0)
                        #print(x,y)
                        #print(x.shape,y.shape)
                        if x1.shape==(0,) or y1.shape==(0,):
                            R='nan'
                        else:
                            R=pearsonr(x1, y1)[0]*-1
                            #R=pearsonr(pr5, ir5)[0]*-1
                        map1[ii,jj]=R
                #print('map1=',map1)
                max1 = ax[n,f2+1].contourf(x,y,map1)
                ax[0,f2+1].set_title(model_names[i],fontsize=8)
                ax[n,0].set_ylabel(musim)
           
            #ax[n,0].set_yticks([-10,0,10,20])
            #ax[1,i].set_xticks([90,100,110,120,130,140])
        n=n+1
                
    
    cax = fig.add_axes([0.91, 0.5, 0.02, 0.35])
    cax.tick_params(labelsize=6)
    plt.colorbar(max1, cax = cax) 
    plt.subplots_adjust(hspace=.05,wspace=.05)
    
    file_name='SEA_Corr_nino34_3'
    fig.savefig(workdir+file_name+reg,dpi=300)

def fft(obs_dataset, obs_name, model_datasets, model_names, workdir):
    plt.style.use('seaborn')
    
    ds = xr.DataArray(obs_dataset.values,
    coords={'time': obs_dataset.times,
    'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
    dims=["time", "lat", "lon"])
    
    print('ds_xarray cek mak=',ds.max())
    #exit()            
    ds = ds.mean(dim=("lat", "lon"))
   
    N=ds.time.shape[0]
    t0 = [int(s) for s in str(ds.time[0].values.astype('datetime64[D]')).split('-') if s.isdigit()][0]
    dt = .09
    time = np.arange(0, N) * dt + t0
  
    signal = ds.values.squeeze() 
    signal = signal - np.mean(signal)
   
    dt = time[1] - time[0]
    N = len(signal)
    fs = 1/dt

    T=dt
    y_values=signal
    
    #def get_fft_values(y_values, T, N, f_s):
    f_values = np.linspace(0.0, 1.0/(2.0*T), N//2)
    fft_values_ = np.fft.fft(y_values)
    fft_values = 2.0/N * np.abs(fft_values_[0:N//2])
    #return f_values, fft_values

    #def plot_fft_plus_power(time, signal, figname=None):
    
    variance = np.std(signal)**2
    #f_values, fft_values = get_fft_values(signal, dt, N, fs)
    fft_power = variance * abs(fft_values) ** 2  # FFT power spectrum
    
    f_values=1/f_values
    print(f_values)
    
    fig, axes = plt.subplots(nrows=3, ncols=10,figsize=(5,3))
    
    axes[0,0].plot(f_values, fft_values, 'r-', label='Fourier Transform')
    axes[0,1].plot(f_values, fft_power, 'k--',
               linewidth=1, label='FFT Power Spectrum')
    # #ax[1].set_xlabel('Frequency [Hz / year]', fontsize=18)
    # ax[1].set_ylabel('Amplitude', fontsize=12)
    # ax[0].set_ylabel('Amplitude', fontsize=12)
    # ax[0].legend()
    # ax[1].legend()
    # plt.show()
    
    axes[0,0].tick_params(axis='x', pad=1, labelsize=7)   
    axes[0,0].tick_params(axis='y', pad=1,labelsize=7)
    axes[0,1].tick_params(axis='x', pad=1, labelsize=7)   
    axes[0,1].tick_params(axis='y', pad=1,labelsize=7)
    axes[0,0].set_title(obs_name, fontsize=7)
    
    axes[0,0].set_ylabel('Amplitude', fontsize=7)
    axes[1,0].set_ylabel('Amplitude', fontsize=7)
    axes[2,0].set_ylabel('Amplitude', fontsize=7)
    axes[2,0].set_xlabel('Periode(year)', fontsize=7)
    
   
    for i in np.arange(len(model_datasets)):
        #print(i)
        if i<8: axes[0,2+i].axis('off')
        dsi = xr.DataArray(model_datasets[i].values,
        coords={'time': obs_dataset.times,
        'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
        dims=["time", "lat", "lon"])
        
        dsi = dsi.mean(dim=("lat", "lon"))
        
        signal = dsi.values.squeeze() 
        signal = signal - np.mean(signal)
        dt = time[1] - time[0]
        N = len(signal)
        fs = 1/dt

        T=dt
        y_values=signal
        
        #def get_fft_values(y_values, T, N, f_s):
        f_values = np.linspace(0.0, 1.0/(2.0*T), N//2)
        fft_values_ = np.fft.fft(y_values)
        fft_values = 2.0/N * np.abs(fft_values_[0:N//2])

        variance = np.std(signal)**2
        #f_values, fft_values = get_fft_values(signal, dt, N, fs)
        fft_power = variance * abs(fft_values) ** 2  # FFT power spectrum
        
        f_values=1/f_values
      
        if i<10:
            axes[1,i].plot(f_values, fft_values, 'r-', label='Fourier Transform')
            axes[2,i].plot(f_values, fft_power, 'k--',
                    linewidth=1, label='FFT Power Spectrum')
            axes[1,i].set_title(model_names[i], fontsize=8)
            axes[1,i].tick_params(axis='x', pad=1, labelsize=7)   
            axes[1,i].tick_params(axis='y', pad=1, labelsize=7)
            axes[2,i].tick_params(axis='x', pad=1, labelsize=7)   
            axes[2,i].tick_params(axis='y', pad=1, labelsize=7)
    
    plt.subplots_adjust(hspace=.3,wspace=.4)
    plt.show()
    file_name='fft'
    fig.savefig(workdir+reg+file_name,dpi=300,bbox_inches='tight')
 
 
def fft_11(obs_dataset, obs_name, model_datasets, model_names, workdir):
   
    plt.style.use('seaborn')
    
    fig, axes = plt.subplots(nrows=3, ncols=5,figsize=(5,3))
    
    ds = xr.DataArray(obs_dataset.values,
    coords={'time': obs_dataset.times,
    'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
    dims=["time", "lat", "lon"])
    
    print('ds_xarray cek mak=',ds.max())
    #exit()            
    ds = ds.mean(dim=("lat", "lon"))
   
    N=ds.time.shape[0]
    t0 = [int(s) for s in str(ds.time[0].values.astype('datetime64[D]')).split('-') if s.isdigit()][0]
    dt = .09
    time = np.arange(0, N) * dt + t0
  
    
    signal = ds.values.squeeze() 
    signal = signal - np.mean(signal)
   
    dt = time[1] - time[0]
    N = len(signal)
    fs = 1/dt


    T=dt
    y_values=signal
    
    #def get_fft_values(y_values, T, N, f_s):
    f_values = np.linspace(0.0, 1.0/(2.0*T), N//2)
    fft_values_ = np.fft.fft(y_values)
    fft_values1 = 2.0/N * np.abs(fft_values_[0:N//2])
   
    
    f_values1=1/f_values
    #f_values1=f_values
   
    axes[0,0].plot(f_values1, fft_values1, 'k-', label='Fourier Transform')
    #plt.ylim(top=2) # axes[0,0].ylim(top=2) tidak ada ,error??
    #ini mengatur skala sb_x
    #axes[0,0].set_xticks([0,10,20])
    #axes[0,0].set_xticks([0.5, 1, 1.5, 2])
    
    axes[0,0].tick_params(axis='x', pad=1,labelsize=7)   
    axes[0,0].tick_params(axis='y', pad=1,labelsize=7)
   
    axes[0,0].set_title(obs_name, pad=3, fontsize=7)
    
    #axes[0,0].set_ylabel('Amplitude', fontsize=7)
    axes[1,0].set_ylabel('Amplitude', fontsize=9)
    #axes[2,0].set_ylabel('Amplitude', fontsize=7)
    axes[2,2].set_xlabel('Period (year)', fontsize=9)
    #axes[2,0].set_xlabel('Frequency (Hz)', fontsize=7)
    
    axes[0,0].set_xlim(left=0.3, right=8)
    
    r1=fft_values1
    #print(r1)
   
    r0=1
    TT=[]
    from scipy.stats import pearsonr
    for i in np.arange(len(model_datasets)):
        print(i)
        if i<4: axes[0,1+i].axis('off')
        dsi = xr.DataArray(model_datasets[i].values,
        coords={'time': obs_dataset.times,
        'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
        dims=["time", "lat", "lon"])
        
        dsi = dsi.mean(dim=("lat", "lon"))
        
        signal = dsi.values.squeeze() 
        signal = signal - np.mean(signal)
        dt = time[1] - time[0]
        N = len(signal)
        fs = 1/dt

        T=dt
        y_values=signal
        
        #def get_fft_values(y_values, T, N, f_s):
        f_values = np.linspace(0.0, 1.0/(2.0*T), N//2)
        fft_values_ = np.fft.fft(y_values)
        fft_values = 2.0/N * np.abs(fft_values_[0:N//2])

        f_values1=1/f_values
        #f_values1=f_values
        
        r2=fft_values
        #print(r1)
   
        # bad = ~np.logical_or(np.isinf(r1), np.isinf(r2))

        sd1=r1.std() 
        sd2=r2.std() 
        s=sd2/sd1      
        c,pp=pearsonr(r1.flatten(), r2.flatten())    
        print(s,c)
        #T= (4*(1+c)**4)/(((s+1/s)**2)*((1+r0)**4))
        # Tanpa pakai **4 T naik dikit
        tt= (4*(1+c))/(((s+1/s)**2)*((1+r0)))
        TT.append(round(tt,2))
      
        if i<5:
            axes[1,i].plot(f_values1, fft_values, 'r--', label='Fourier Transform')
            axes[1,i].plot(f_values1, fft_values1, 'k--', label='Fourier Transform')
            #axes[1,i].set_xticks([0,10,20])
            #axes[1,i].set_xticks([0.5, 1, 1.5, 2])
            #plt.ylim(top=2)
            axes[1,i].tick_params(axis='x', pad=1, labelsize=7)   
            axes[1,i].tick_params(axis='y', pad=1, labelsize=7)
            axes[1,i].set_title(model_names[i]+' ('+'%.2f'%tt+')',pad=3,fontsize=8)
            #axes[i].set_ylim(top=0.8)
            axes[1,i].set_xlim(left=0.3, right=8)
        if i>=5:
           
            axes[2,i-5].plot(f_values1, fft_values, 'r--', label='Fourier Transform')
            axes[2,i-5].plot(f_values1, fft_values1, 'k--', label='Fourier Transform')
            #axes[2,i-7].set_xticks([0,10,20])
            #axes[2,i-7].set_xticks([0.5, 1, 1.5, 2])
            #plt.ylim(top=2)
            axes[2,i-5].tick_params(axis='x', pad=1, labelsize=7)   
            axes[2,i-5].tick_params(axis='y', pad=1, labelsize=7)
            axes[2,i-5].set_title(model_names[i]+' ('+'%.2f'%tt+')',pad=3,fontsize=8)
            axes[2,i-5].set_xlim(left=0.3, right=8)
    print('T[]=',TT)
    plt.subplots_adjust(hspace=.3,wspace=.2)
    
    #set limit sb_x untuk apa?
    #plt.setp(axes, xlim=(0,2))
    plt.show()
    file_name='fft_14_land-only'
    fig.savefig(workdir+reg+file_name,dpi=300,bbox_inches='tight')


def fft_15b(obs_dataset, obs_name, model_datasets, model_names, workdir):
    #pakai scipy.signal import periodogram
    
    #bagus untuk mode < 1 tahun, pada Sum mode ENSO 2-8 tahun tidak nampak
    from scipy.signal import periodogram
    fig, ax = plt.subplots(5,3, figsize=(3,2))
    important_periods = [0.25, 0.5, 1, 2, 4, 7, 10]
    for i in np.arange(len(model_datasets)):
        print()
        dsi = xr.DataArray(model_datasets[i].values,
        coords={'time': model_datasets[i].times,
        'lat': model_datasets[i].lats, 'lon': model_datasets[i].lons},
        dims=["time", "lat", "lon"])
        
        #for Sumatera R1 monsunal dan equatorial 2 puncak
        #dsi = dsi.where((dsi.lat > 0) & (dsi.lat < 6) & 
        #            (dsi.lon > 95) & (dsi.lon < 104), drop=True)
                    
        #R2 bawah
        #dsi = dsi.where((dsi.lat > -6) & (dsi.lat < 0) & 
        #            (dsi.lon > 99) & (dsi.lon < 106.5), drop=True)
        
        dsi = dsi.mean(dim=("lat", "lon"))
        
        # Perform spectral analysis
        frequencies, psd = periodogram(dsi, fs=12)  
        # Sampling frequency assumed to be 12 month
        if i<5:
            ax[i,0].plot(1/frequencies, psd)
            ax[i,0].set_xscale('log')
            ax[i,0].set_xticks(important_periods, labels=[f'{period}' for period in important_periods])
            ax[i,0].set_xlim(left=0.2, right=8)
            ax[i,0].set_title(model_names[i], pad=4,fontsize=8)
        if 5<=i<10:
            i=i-5
            ax[i,1].plot(1/frequencies, psd)
            ax[i,1].set_xscale('log')
            ax[i,1].set_xticks(important_periods, labels=[f'{period}' for period in important_periods])
            ax[i,1].set_xlim(left=0.2, right=8)
            ax[i,1].set_title(model_names[i+5], pad=4,fontsize=8)
        if 15>i>=10:
            i=i-10
            ax[i,2].plot(1/frequencies, psd)
            ax[i,2].set_xscale('log')
            ax[i,2].set_xticks(important_periods, labels=[f'{period}' for period in important_periods])
            ax[i,2].set_xlim(left=0.2, right=8)
            ax[i,2].set_title(model_names[i+10],pad=4, fontsize=8)
        
        ax[i,0].tick_params(axis='x', pad=1,labelsize=7)   
        ax[i,1].tick_params(axis='x', pad=1,labelsize=7)
        ax[i,2].tick_params(axis='x', pad=1,labelsize=7)    
        ax[i,0].tick_params(axis='y', pad=1,labelsize=7)   
        ax[i,1].tick_params(axis='y', pad=1,labelsize=7)
        ax[i,2].tick_params(axis='y', pad=1,labelsize=7)   
        
        ax[2,0].set_ylabel('Power spectral density', fontsize=8)
        ax[4,1].set_xlabel('Period (year)', fontsize=8)
        
        #plt.yscale('log')
        #ax[2].set_xlabel('Period (years)')
        #ax[1].set_ylabel('Power spectral density')
        #ax[i].set_title(cluster_names[i])
       
    plt.subplots_adjust(hspace=.7,wspace=.1)
    #plt.subplots_adjust(right=.45)
    #plt.subplots_adjust(bottom=.3)
    #plt.tight_layout()
    plt.show()    

   
def fft_15(obs_dataset, obs_name, model_datasets, model_names, workdir):
    
    Sum=True
    plt.style.use('seaborn')
    
    fig, axes = plt.subplots(nrows=3, ncols=7,figsize=(5,3))
    
    ds = xr.DataArray(obs_dataset.values,
    coords={'time': obs_dataset.times,
    'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
    dims=["time", "lat", "lon"])
    
    if Sum:
    #Sum bagi 2
    #for Sumatera R1 monsunal dan equatorial 2 puncak
    #for Sumatera R2 monsunal 1 puncak
        ds = ds.where(
                        #R1
                        #(ds.lat > 0) & (ds.lat < 6) & 
                        #(ds.lon > 95) & (ds.lon < 104), 
                        #R2
                        (ds.lat > -6) & (ds.lat < 0) & 
                        (ds.lon > 99) & (ds.lon < 106), 
                        #Sum
                        #(ds.lat > -6) & (ds.lat < 6) & 
                        #(ds.lon > 95) & (ds.lon < 106), 
                        drop=True)
        
    print('ds_xarray cek mak=',ds.max())
    #exit()            
    ds = ds.mean(dim=("lat", "lon"))
   
    N=ds.time.shape[0]
    t0 = [int(s) for s in str(ds.time[0].values.astype('datetime64[D]')).split('-') if s.isdigit()][0]
    dt = .09
    time = np.arange(0, N) * dt + t0
  
    
    signal = ds.values.squeeze() 
    signal = signal - np.mean(signal)
   
    dt = time[1] - time[0]
    N = len(signal)
    fs = 1/dt


    T=dt
    y_values=signal
    
    #def get_fft_values(y_values, T, N, f_s):
    f_values = np.linspace(0.0, 1.0/(2.0*T), N//2)
    fft_values_ = np.fft.fft(y_values)
    fft_values1 = 2.0/N * np.abs(fft_values_[0:N//2])
    #return f_values, fft_values

    #def plot_fft_plus_power(time, signal, figname=None):
    
    #variance = np.std(signal)**2
    #f_values, fft_values = get_fft_values(signal, dt, N, fs)
    #fft_power = variance * abs(fft_values) ** 2  # FFT power spectrum
    
    f_values1=1/f_values
    #f_values1=f_values
    #print(f_values1)
    
    axes[0,0].plot(f_values1, fft_values1, 'k-', label='Fourier Transform')
    #plt.ylim(top=2) # axes[0,0].ylim(top=2) tidak ada ,error??
    #ini mengatur skala sb_x
    #axes[0,0].set_xticks([0,10,20])
    #axes[0,0].set_xticks([0.5, 1, 1.5, 2])
    
    axes[0,0].tick_params(axis='x', pad=1,labelsize=7)   
    axes[0,0].tick_params(axis='y', pad=1,labelsize=7)
   
    axes[0,0].set_title(obs_name, pad=3, fontsize=7)
    
    #axes[0,0].set_ylabel('Amplitude', fontsize=7)
    axes[1,0].set_ylabel('Amplitude', fontsize=9)
    #axes[2,0].set_ylabel('Amplitude', fontsize=7)
    axes[2,3].set_xlabel('Period (year)', fontsize=9)
    #axes[2,0].set_xlabel('Frequency (Hz)', fontsize=7)
    axes[0,0].set_xlim(left=0.3, right=10)
    
    
    r1=fft_values1
    #print(r1)
   
    r0=1
    TT=[]
    from scipy.stats import pearsonr
    for i in np.arange(len(model_datasets)):
        print(i)
        if i<5: axes[0,2+i].axis('off')
        dsi = xr.DataArray(model_datasets[i].values,
        coords={'time': obs_dataset.times,
        'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
        dims=["time", "lat", "lon"])
        
        if Sum:
            #for Sumatera R1 monsunal dan equatorial 2 puncak
            dsi = dsi.where(
                        #R1
                        #(dsi.lat > 0) & (dsi.lat < 6) & 
                        #(dsi.lon > 95) & (dsi.lon < 104), 
                        #R2
                        (dsi.lat > -6) & (dsi.lat < 0) & 
                        (dsi.lon > 99) & (dsi.lon < 106), 
                                        
                        #Sum
                        #(dsi.lat > -6) & (dsi.lat < 6) & 
                        #(dsi.lon > 95) & (dsi.lon < 106), 
                        
                        drop=True)
            
        dsi = dsi.mean(dim=("lat", "lon"))
        
        signal = dsi.values.squeeze() 
        signal = signal - np.mean(signal)
        dt = time[1] - time[0]
        N = len(signal)
        fs = 1/dt

        T=dt
        y_values=signal
        
        #def get_fft_values(y_values, T, N, f_s):
        f_values = np.linspace(0.0, 1.0/(2.0*T), N//2)
        fft_values_ = np.fft.fft(y_values)
        fft_values = 2.0/N * np.abs(fft_values_[0:N//2])

        f_values1=1/f_values
        #f_values1=f_values
        
        r2=fft_values
        #print(r1)
   
        # bad = ~np.logical_or(np.isinf(r1), np.isinf(r2))

        sd1=r1.std() 
        sd2=r2.std() 
        s=sd2/sd1      
        c,pp=pearsonr(r1.flatten(), r2.flatten())    
        print(s,c)
        #T= (4*(1+c)**4)/(((s+1/s)**2)*((1+r0)**4))
        # Tanpa pakai **4 T naik dikit
        tt= (4*(1+c))/(((s+1/s)**2)*((1+r0)))
        TT.append(round(tt,2))
        if i==0:
            ii=1
            axes[0,ii].plot(f_values1, fft_values, 'r--', label='Fourier Transform')
            axes[0,ii].plot(f_values1, fft_values1, 'k--', label='Fourier Transform')
           
            axes[0,ii].tick_params(axis='x', pad=1, labelsize=7)   
            axes[0,ii].tick_params(axis='y', pad=1, labelsize=7)
            axes[0,ii].set_title(model_names[i]+' ('+'%.2f'%tt+')',pad=3,fontsize=8)
            axes[0,ii].set_xlim(left=0.3, right=10)
        #1-7
        if 0<i<8:
            axes[1,i-1].plot(f_values1, fft_values, 'r--', label='Fourier Transform')
            axes[1,i-1].plot(f_values1, fft_values1, 'k--', label='Fourier Transform')
            #axes[1,i].set_xticks([0,10,20])
            #axes[1,i].set_xticks([0.5, 1, 1.5, 2])
            #plt.ylim(top=2)
            axes[1,i-1].tick_params(axis='x', pad=1, labelsize=7)   
            axes[1,i-1].tick_params(axis='y', pad=1, labelsize=7)
            axes[1,i-1].set_title(model_names[i]+' ('+'%.2f'%tt+')',pad=3,fontsize=8)
            axes[1,i-1].set_xlim(left=0.3, right=10)
        if i>=8:
           
            axes[2,i-8].plot(f_values1, fft_values, 'r--', label='Fourier Transform')
            axes[2,i-8].plot(f_values1, fft_values1, 'k--', label='Fourier Transform')
            #axes[2,i-7].set_xticks([0,10,20])
            #axes[2,i-7].set_xticks([0.5, 1, 1.5, 2])
            #plt.ylim(top=2)
            axes[2,i-8].tick_params(axis='x', pad=1, labelsize=7)   
            axes[2,i-8].tick_params(axis='y', pad=1, labelsize=7)
            axes[2,i-8].set_title(model_names[i]+' ('+'%.2f'%tt+')',pad=3,fontsize=8)
            axes[2,i-8].set_xlim(left=0.3, right=10)
    print('T[]=',TT)
    plt.subplots_adjust(hspace=.3,wspace=.2)
    
    #set limit sb_x untuk apa?
    #plt.setp(axes, xlim=(0,2))
    plt.show()
    file_name='fft_14_R2'
    fig.savefig(workdir+reg+file_name,dpi=300,bbox_inches='tight')

def fft_5obs(obs_dataset, obs_name, model_datasets, model_names, workdir):
   
    plt.style.use('seaborn')
    
    fig, axes = plt.subplots(nrows=5, ncols=1,figsize=(5,3))
    
    ds = xr.DataArray(obs_dataset.values,
    coords={'time': obs_dataset.times,
    'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
    dims=["time", "lat", "lon"])
    
    print('ds_xarray cek mak=',ds.max())
    #exit()            
    ds = ds.mean(dim=("lat", "lon"))
   
    N=ds.time.shape[0]
    t0 = [int(s) for s in str(ds.time[0].values.astype('datetime64[D]')).split('-') if s.isdigit()][0]
    dt = .09
    time = np.arange(0, N) * dt + t0
  
    
    signal = ds.values.squeeze() 
    signal = signal - np.mean(signal)
   
    dt = time[1] - time[0]
    N = len(signal)
    fs = 1/dt

    T=dt
    y_values=signal
    
    #def get_fft_values(y_values, T, N, f_s):
    f_values = np.linspace(0.0, 1.0/(2.0*T), N//2)
    fft_values_ = np.fft.fft(y_values)
    fft_values1 = 2.0/N * np.abs(fft_values_[0:N//2])
   
    f_values1=1/f_values
    #f_values1=f_values
    print(f_values1)
    
    axes[0].plot(f_values1, fft_values1, 'k-', label='Fourier Transform')
  
    r1=fft_values1
    #print(r1)
   
    r0=1
    TT=[]
    from scipy.stats import pearsonr
    for i in np.arange(len(model_datasets)):
       
        dsi = xr.DataArray(model_datasets[i].values,
        coords={'time': obs_dataset.times,
        'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
        dims=["time", "lat", "lon"])
        
        dsi = dsi.mean(dim=("lat", "lon"))
        
        signal = dsi.values.squeeze() 
        signal = signal - np.mean(signal)
        dt = time[1] - time[0]
        N = len(signal)
        fs = 1/dt

        T=dt
        y_values=signal
        
        #def get_fft_values(y_values, T, N, f_s):
        f_values = np.linspace(0.0, 1.0/(2.0*T), N//2)
        fft_values_ = np.fft.fft(y_values)
        fft_values = 2.0/N * np.abs(fft_values_[0:N//2])

        f_values1=1/f_values
        #f_values1=f_values
        
        r2=fft_values
        #print(r1)
   
        # bad = ~np.logical_or(np.isinf(r1), np.isinf(r2))

        sd1=r1.std() 
        sd2=r2.std() 
        s=sd2/sd1      
        c,pp=pearsonr(r1.flatten(), r2.flatten())    
        print(s,c)
        #T= (4*(1+c)**4)/(((s+1/s)**2)*((1+r0)**4))
        # Tanpa pakai **4 T naik dikit
        tt= (4*(1+c))/(((s+1/s)**2)*((1+r0)))
        TT.append(round(tt,2))
      
        axes[i].plot(f_values1, fft_values, 'r-', label='Fourier Transform')
        axes[i].plot(f_values1, fft_values1, 'k-', label='Fourier Transform')
        if i<len(model_datasets)-1:
            axes[i].set_xticks([])
        #axes[1,i].set_xticks([0.5, 1, 1.5, 2])
        axes[i].set_ylim(top=0.8)
        axes[i].set_xlim(left=0.3, right=3)
        
        axes[i].tick_params(axis='y', pad=1, labelsize=10)
        if i==0:
            axes[i].set_title(model_names[i])
        else:
            axes[i].set_title(model_names[i]+' ['+'%.2f'%tt+']',pad=3)
           
    axes[2].set_ylabel('Amplitude', fontsize=10)
    axes[len(model_datasets)-1].tick_params(axis='x', pad=1, labelsize=10)   
    axes[len(model_datasets)-1].set_xlabel('Period (year)', fontsize=10)     
    print('T[]=',TT)
    plt.subplots_adjust(hspace=.3,wspace=.2)
    
    #set limit sb_x untuk apa?
    #plt.setp(axes, xlim=(0,2))
    plt.show()
    file_name='fft_5obs'
    fig.savefig(workdir+reg+file_name,dpi=300,bbox_inches='tight')

def fft_14f(obs_dataset, obs_name, model_datasets, model_names, workdir):
   
    
    plt.style.use('seaborn')
    
    ds = xr.DataArray(obs_dataset.values,
    coords={'time': obs_dataset.times,
    'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
    dims=["time", "lat", "lon"])
    
    print('ds_xarray cek mak=',ds.max())
    #exit()            
    ds = ds.mean(dim=("lat", "lon"))
   
    N=ds.time.shape[0]
    t0 = [int(s) for s in str(ds.time[0].values.astype('datetime64[D]')).split('-') if s.isdigit()][0]
    dt = .09
    time = np.arange(0, N) * dt + t0
  
    
    signal = ds.values.squeeze() 
    signal = signal - np.mean(signal)
   
    dt = time[1] - time[0]
    N = len(signal)
    fs = 1/dt


    T=dt
    y_values=signal
    
    #def get_fft_values(y_values, T, N, f_s):
    f_values = np.linspace(0.0, 1.0/(2.0*T), N//2)
    fft_values_ = np.fft.fft(y_values)
    fft_values1 = 2.0/N * np.abs(fft_values_[0:N//2])
    #return f_values, fft_values

    #def plot_fft_plus_power(time, signal, figname=None):
    

    
    #variance = np.std(signal)**2
    #f_values, fft_values = get_fft_values(signal, dt, N, fs)
    #fft_power = variance * abs(fft_values) ** 2  # FFT power spectrum
    
    #f_values1=1/f_values
    f_values1=f_values
    
    fig, axes = plt.subplots(nrows=3, ncols=5,figsize=(5,3))
    
    axes[0,0].plot(f_values1, fft_values1, 'k-', label='Fourier Transform')
    #plt.ylim(top=2) # axes[0,0].ylim(top=2) tidak ada ,error??
    #axes[0,0].set_xticks([0,10,20])
    axes[0,0].set_xticks([1,3,5])
    
    axes[0,0].tick_params(axis='x', pad=1,labelsize=7)   
    axes[0,0].tick_params(axis='y', pad=1,labelsize=7)
   
    axes[0,0].set_title(obs_name, fontsize=7)
    
    axes[0,0].set_ylabel('Amplitude', fontsize=7)
    axes[1,0].set_ylabel('Amplitude', fontsize=7)
    axes[2,0].set_ylabel('Amplitude', fontsize=7)
    #axes[2,0].set_xlabel('Period (year)', fontsize=7)
    axes[2,0].set_xlabel('Frequency (Hz)', fontsize=7)
    
    
    r1=fft_values1
    #print(r1)
   
    r0=1
    TT=[]
    from scipy.stats import pearsonr
    for i in np.arange(len(model_datasets)):
        #print(i)
        #if i<6: axes[0,1+i].axis('off')
        dsi = xr.DataArray(model_datasets[i].values,
        coords={'time': obs_dataset.times,
        'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
        dims=["time", "lat", "lon"])
        
        dsi = dsi.mean(dim=("lat", "lon"))
        
        signal = dsi.values.squeeze() 
        signal = signal - np.mean(signal)
        dt = time[1] - time[0]
        N = len(signal)
        fs = 1/dt

        T=dt
        y_values=signal
        
        #def get_fft_values(y_values, T, N, f_s):
        f_values = np.linspace(0.0, 1.0/(2.0*T), N//2)
        fft_values_ = np.fft.fft(y_values)
        fft_values = 2.0/N * np.abs(fft_values_[0:N//2])

        f_values1=1/f_values
        #f_values1=f_values
        
        r2=fft_values
        #print(r1)
   
        # bad = ~np.logical_or(np.isinf(r1), np.isinf(r2))

        sd1=r1.std() 
        sd2=r2.std() 
        s=sd2/sd1      
        c,pp=pearsonr(r1.flatten(), r2.flatten())    
        print(s,c)
        #T= (4*(1+c)**4)/(((s+1/s)**2)*((1+r0)**4))
        # Tanpa pakai **4 T naik dikit
        tt= (4*(1+c))/(((s+1/s)**2)*((1+r0)))
        TT.append(round(tt,2))
      
        if 0<=i<4:
            axes[0,i+1].plot(f_values, fft_values, 'r--', label='Fourier Transform')
            #axes[0,i+1].plot(f_values1, fft_values1, 'k-', label='Fourier Transform')
            #axes[1,i].set_xticks([0,10,20])
            axes[0,i+1].set_xticks([1,3,5])
            #plt.ylim(top=2)
            axes[0,i+1].tick_params(axis='x', pad=1, labelsize=7)   
            axes[0,i+1].tick_params(axis='y', pad=1, labelsize=7)
            axes[0,i+1].set_title(model_names[i]+'('+'%.2f'%tt+')',fontsize=8)
            
        if 9>i>=4:
           
            axes[1,i-4].plot(f_values, fft_values, 'r--', label='Fourier Transform')
            #axes[1,i-4].plot(f_values1, fft_values1, 'k-', label='Fourier Transform')
            #axes[2,i-7].set_xticks([0,10,20])
            axes[1,i-4].set_xticks([1,3,5])
            #plt.ylim(top=2)
            axes[1,i-4].tick_params(axis='x', pad=1, labelsize=7)   
            axes[1,i-4].tick_params(axis='y', pad=1, labelsize=7)
            axes[1,i-4].set_title(model_names[i]+'('+'%.2f'%tt+')',fontsize=8)
    
        if i>=9:
           
            axes[2,i-9].plot(f_values, fft_values, 'r--', label='Fourier Transform')
            #axes[2,i-9].plot(f_values1, fft_values1, 'k-', label='Fourier Transform')
            #axes[2,i-7].set_xticks([0,10,20])
            axes[2,i-9].set_xticks([1,3,5])
            #plt.ylim(top=2)
            axes[2,i-9].tick_params(axis='x', pad=1, labelsize=7)   
            axes[2,i-9].tick_params(axis='y', pad=1, labelsize=7)
            axes[2,i-9].set_title(model_names[i]+'('+'%.2f'%tt+')',fontsize=8)
    
    print('T[]=',TT)
    plt.subplots_adjust(hspace=.3,wspace=.4)
    #plt.ylim(top=2)
    
    plt.show()
    file_name='fft_14-freq2tesssss'
    fig.savefig(workdir+reg+file_name,dpi=300,bbox_inches='tight')                              

def wavelet(obs_dataset, obs_name, model_datasets, model_names, workdir):
    import pywt
    from scipy.stats import pearsonr
    plt.style.use('seaborn')
    
    ds = xr.DataArray(obs_dataset.values,
    coords={'time': obs_dataset.times,
    'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
    dims=["time", "lat", "lon"])
    
    print('ds_xarray cek mak=',ds.max())
    #exit()            
    ds = ds.mean(dim=("lat", "lon"))
    #ini bgm sb x tahun sb y hujan/bulan ya?
    
    #N = df_nino.shape[0]
    N=ds.time.shape[0]
    t0 = [int(s) for s in str(ds.time[0].values.astype('datetime64[D]')).split('-') if s.isdigit()][0]
    dt = .09
    time = np.arange(0, N) * dt + t0
    print(N)
    print(t0)
    print(time)
    
    signal = ds.values.squeeze() 
    signal = signal - np.mean(signal)

    scales = np.arange(1, 222) 
    
    # #using matplotlib
    
    # fs = 1000.0 # 1 kHz sampling frequency
    # (S, f) = plt.psd(signal, Fs=fs)
    # #print(signal,S,f)
    # p = 1. / f
    # plt.semilogy(p, S)
    # plt.xlim([0, .01])
    # plt.xlabel('frequency [Hz]')
    # plt.ylabel('PSD [V**2/Hz]')
    # plt.show()
    # exit()

    waveletname='cmor1.5-1.0' 
    cmap=plt.cm.seismic 
    title='Wavelet Transform'
    ylabel='Period (years)'
    xlabel='Time'

    dt = time[1] - time[0]
    [coefficients, frequencies] = pywt.cwt(signal, scales, 
                                    waveletname, dt)
    power = (abs(coefficients)) ** 2
    #print(power)
    period = 1. / frequencies

    scale0 = 0.03
    numlevels = 10

    levels = [scale0]
    for ll in range(1, numlevels):
        scale0 *= 2
        levels.append(scale0)

    contourlevels = np.log2(levels)
    fig, axes = plt.subplots(nrows=1, ncols=11) #,figsize=(16,6))
    
    # fig, ax = plt.subplots(figsize=(8, 6))
    im = axes[0].contourf(time, np.log2(period), np.log2(power),
                     contourlevels, extend='both', cmap=cmap)
    #----ini akan di R kan dgn model
    #r1=np.log2(power)
    r1=power  
    
    sd1=r1.std() #(skipna=None)
    r0=1
    # ax.set_title(title, fontsize=20)
    # ax.set_ylabel(ylabel, fontsize=18)
    axes[0].set_xlabel('GPCP', fontsize=6)
    # yticks = 2**np.arange(np.ceil(np.log2(period.min())),
                          # np.ceil(np.log2(period.max())))
    # ax.set_yticks(np.log2(yticks))
    # ax.set_yticklabels(yticks)
    # ax.invert_yaxis()
    # ylim = ax.get_ylim()
    # ax.set_ylim(ylim[0], -1)
    # cbar_ax = fig.add_axes([0.95, 0.15, 0.03, 0.7])
    # fig.colorbar(im, cax=cbar_ax, orientation="vertical")
    
    ##########
    T=[]
    for i in np.arange(len(model_datasets)-1):
        dsi = xr.DataArray(model_datasets[i].values,
        coords={'time': obs_dataset.times,
        'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
        dims=["time", "lat", "lon"])
                    
        dsi = dsi.mean(dim=("lat", "lon"))
        
        signal = dsi.values.squeeze() 
        signal = signal - np.mean(signal)

              
        [coefficients, frequencies] = pywt.cwt(signal, scales, waveletname, dt)
        power = (abs(coefficients)) ** 2
        #print(power)
        period = 1. / frequencies

       
        #fig, ax = plt.subplots(figsize=(8, 6))
        im = axes[1+i].contourf(time, np.log2(period), np.log2(power),
                         contourlevels, extend='both', cmap=cmap)
        
        #----ini akan di R kan dgn obs
        #r2=np.log2(power)  
        r2=power  
        sd2=r2.std() #(skipna=None)
        s=sd2/sd1
        c,pp=pearsonr(r1.flatten() , r2.flatten())        
        
        #T= (4*(1+c)**4)/(((s+1/s)**2)*((1+r0)**4))
        # Tanpa pakai **4 T naik dikit
        tt= (4*(1+c))/(((s+1/s)**2)*((1+r0)))
        T.append(round(tt,2))
        
        #ax.set_title(title, fontsize=20)
        axes[0].set_ylabel(ylabel, fontsize=10)
        if i<11: axes[1+i].set_yticks([]) 
        
        axes[i].set_xticks([])
        axes[10].set_xticks([])
        axes[1+i].set_xlabel(model_names[i], fontsize=6)
        yticks = 2**np.arange(np.ceil(np.log2(period.min())),
                              np.ceil(np.log2(period.max())))
        axes[0].set_yticks(np.log2(yticks))
        #axes[0].set_yticklabels(yticks, fontsize=8)
        axes[0].set_yticklabels(['1/8','1/4','1/2','1','2','4','8','16'],fontsize=8)
        #axes[0].invert_yaxis()
        #ylim = axes[0].get_ylim()
        #axes[0].set_ylim(ylim[0], -1)
        axes[1+i].annotate('S='+'%.2f'%tt, fontsize=6, xy=(1988,.93*np.log2(period).max()), backgroundcolor='0.85',alpha=1)
        
        cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.7])
        cbar_ax.tick_params(labelsize=8)
        fig.colorbar(im, cax=cbar_ax, ticks=[-5,-4,-3,-2,-1,0,1,2,3,4])  #, orientation="vertical")
    print(T)
    plt.subplots_adjust(hspace=.2,wspace=.04)
    plt.savefig(workdir+reg+'_wavelet2.png', dpi=300, bbox_inches='tight')
    #plt.show()

def wavelet_14mean(obs_dataset, obs_name, model_datasets, model_names, workdir):
    import pywt
    from scipy.stats import pearsonr
    
    from sklearn.metrics.pairwise import cosine_similarity
    
    plt.style.use('seaborn')
    
    fig, axes = plt.subplots(nrows=1, ncols=16) #,figsize=(16,6))
    
    ds = xr.DataArray(obs_dataset.values,
    coords={'time': obs_dataset.times,
    'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
    dims=["time", "lat", "lon"])

    '''
    #Sum bagi 2
    #for Sumatera R1 monsunal dan equatorial 2 puncak
    #for Sumatera R2 monsunal 1 puncak
    ds = ds.where(
                    #R1
                    #(ds.lat > 0) & (ds.lat < 6) & 
                    #(ds.lon > 95) & (ds.lon < 104), 
                    #R2
                    (ds.lat > -6) & (ds.lat < 0) & 
                    (ds.lon > 99) & (ds.lon < 106), 
                    #Sum
                    #(ds.lat > -6) & (ds.lat < 6) & 
                    #(ds.lon > 95) & (ds.lon < 106), 
                    drop=True)
    '''
    '''
    #Sum kmeans clustering
    from xlearn22.cluster import KMeans
    ds0=ds
    n_clusters=3
    annual_cycle=True
    with_pca=False
    ds= ds.stack(z=("lat", "lon")) #s= s.unstack()
    dm = ds.groupby('time.month').mean()
    km = KMeans(n_clusters=n_clusters, random_state=0).fit2(dm, annual_cycle, with_pca)
           
    n=km.cluster_centers_da.sel(cluster=0)
       
    #xarray masking ==> hasil kmeans clustering sbg mask 
    ds0.coords['mask'] = (('lat', 'lon'), n.values)
    
    #ERA5 zone 0 biru R2, 1 merah zone 1
    ds=ds0.where(ds0.mask == 0) #.mean(dim=("lat", "lon"))
    '''
    
    #--------------------------
    
    print('ds_xarray cek mak=',ds.max())
    # ada 2 pilihan           
    ds = ds.mean(dim=("lat", "lon"))
    #ds = ds.sum(dim=("lat", "lon"))
    
    #ini bgm sb x tahun sb y hujan/bulan ya?
    
    #N = df_nino.shape[0]
    N=ds.time.shape[0]
    t0 = [int(s) for s in str(ds.time[0].values.astype('datetime64[D]')).split('-') if s.isdigit()][0]
    dt = .09
    time = np.arange(0, N) * dt + t0
    
    signal = ds.values.squeeze() 
    signal = signal - np.mean(signal)
    
    # #detrending > data berkurang 1 di ujung 
    # X = signal
    # diff = list()
    # for i in range(1, len(X)):
        # value = X[i] - X[i - 1]
        # diff.append(value)
    
    # #plt.plot(X,'r')
    # #plt.plot(diff)
    # #plt.show()
    # #exit()
    # signal=diff
    # time=np.delete(time,-1)
    
    scales = np.arange(1, 222) 
  
    waveletname='cmor1.5-1.0' 
    cmap=plt.cm.seismic 
    title='Wavelet Transform'
    ylabel='Period (years)'
    xlabel='Time'

    dt = time[1] - time[0]
    [coefficients, frequencies] = pywt.cwt(signal, scales, 
                                    waveletname, dt)
    power = (abs(coefficients)) ** 2
    #print(power)
    period = 1. / frequencies

    scale0 = 0.03
    numlevels = 10

    levels = [scale0]
    for ll in range(1, numlevels):
        scale0 *= 2
        levels.append(scale0)
    
    print(power.min(), power.max())
    print('levels=', levels)

    contourlevels = np.log2(levels)
   
    
    # fig, ax = plt.subplots(figsize=(8, 6))
    im = axes[0].contourf(time, np.log2(period), np.log2(power),
                     contourlevels, extend='both', cmap=cmap)
    axes[0].set_xticks([])
    axes[0].set_xlabel(obs_name, fontsize=8,rotation=30)
    
    r1=power  
    sd1=r1.std() #(skipna=None)
    r0=1
    T=[]
    for i in np.arange(len(model_datasets)):
        dsi = xr.DataArray(model_datasets[i].values,
        coords={'time': model_datasets[i].times,
        'lat': model_datasets[i].lats, 'lon': model_datasets[i].lons},
        dims=["time", "lat", "lon"])
        
        '''
        #for Sumatera R1 monsunal dan equatorial 2 puncak
        dsi = dsi.where(
                    #R1
                    #(dsi.lat > -6) & (dsi.lat < 0) & 
                    #(dsi.lon > 99) & (dsi.lon < 106), 
                    #R2
                    (dsi.lat > 0) & (dsi.lat < 6) & 
                    (dsi.lon > 95) & (dsi.lon < 104), 
                                      
                    #Sum
                    #(dsi.lat > -6) & (dsi.lat < 6) & 
                    #(dsi.lon > 95) & (dsi.lon < 106), 
                    
                    drop=True)
                    
        '''
        
        dsi = dsi.mean(dim=("lat", "lon"))
        
        signal = dsi.values.squeeze() 
        signal = signal - np.mean(signal)
        
        # #detrending
        # X = signal
        # diff = list()
        # for ii in range(1, len(X)):
            # value = X[ii] - X[ii - 1]
            # diff.append(value)
        
        # #plt.plot(X,'r')
        # #plt.plot(diff)
        # #plt.show()
        # #exit()
        # signal=diff
       
              
        [coefficients, frequencies] = pywt.cwt(signal, scales, waveletname, dt)
        power = (abs(coefficients)) ** 2
        #print(power)
        period = 1. / frequencies

       
        #fig, ax = plt.subplots(figsize=(8, 6))
        if i<len(model_datasets):
            im = axes[1+i].contourf(time, np.log2(period), np.log2(power),
                         contourlevels, extend='both', cmap=cmap)
        
        #----ini akan di R kan dgn obs
        #r2=np.log2(power)  
        r2=power  
        sd2=r2.std() #(skipna=None)
        s=sd2/sd1
        c,pp=pearsonr(r1.flatten() , r2.flatten())        
        
        #T= (4*(1+c)**4)/(((s+1/s)**2)*((1+r0)**4))
        # Tanpa pakai **4 T naik dikit
        tt= (4*(1+c))/(((s+1/s)**2)*((1+r0)))
        T.append(round(tt,2))
        
        #similarity score
        #from sklearn.metrics.pairwise import cosine_similarity
    
        features_graph1 = np.array(r1)  # Replace with your feature data
        features_graph2 = np.array(r2)  # Replace with your feature data

        # Reshape feature arrays if needed
        features_graph1 = features_graph1.reshape(1, -1)
        features_graph2 = features_graph2.reshape(1, -1)
        
        
        # Calculate cosine similarity
        #sc = cosine_similarity(features_graph1, features_graph2)[0,0]
        #sc = euclidean_distances(features_graph1, features_graph2)[0,0]
        
        #print("Similarity Score:", sc)
        
        #ax.set_title(title, fontsize=20)
        axes[0].set_ylabel(ylabel, fontsize=10)
        
        if i<15: axes[1+i].set_yticks([]) 
        
        axes[i].set_xticks([])
        #axes[14].set_xticks([])
        axes[1+i].set_xlabel(model_names[i], fontsize=8,rotation=30)
        #khusus noIPSL
        #axes[12].set_xticks([])
       # axes[13].set_xticks([])
        #axes[13].set_yticks([])
        axes[len(model_datasets)-1].set_yticks([])
        
        
        
        yticks = 2**np.arange(np.ceil(np.log2(period.min())),
                              np.ceil(np.log2(period.max())))
        axes[0].set_yticks(np.log2(yticks))
        #axes[0].set_yticklabels(yticks, fontsize=8)
        axes[0].set_yticklabels(['1/8','1/4','1/2','1','2','4','8','16'],fontsize=8)
        #axes[0].invert_yaxis()
        #ylim = axes[0].get_ylim()
        #axes[0].set_ylim(ylim[0], -1)S
        if i<len(model_datasets):
            axes[1+i].annotate('T='+'%.2f'%tt, fontsize=8, xy=(1985,0.9*np.log2(period).min()), backgroundcolor='0.85',alpha=1)
            axes[1+i].set_yticks([])
            axes[1+i].set_xticks([])
            axes[1+i].set_xlabel(model_names[i], fontsize=8,rotation=30)
            #axes[1+i].set_title('SC='+str(round(sc,2)), fontsize=6)
        
        cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.7])
        cbar_ax.tick_params(labelsize=8)
        #plt.xlabel(rotation=45)
        #fig.colorbar(im, cax=cbar_ax)
        fig.colorbar(im, cax=cbar_ax, ticks=[-5,-4,-3,-2,-1,0,1,2,3,4])  #, orientation="vertical")
    print(T)
    
    plt.subplots_adjust(hspace=.3,wspace=.04)
    #plt.savefig(workdir+reg+'_wavelet_14_era5_noIPSL.png', dpi=300, bbox_inches='tight')
    plt.show()

def wavelet_5obs(obs_dataset, obs_name, model_datasets, model_names, workdir):
    import pywt
    from scipy.stats import pearsonr
    from sklearn.metrics.pairwise import cosine_similarity
    from skimage.metrics import structural_similarity as ssim
    
    plt.style.use('seaborn')
    
    fig, axes = plt.subplots(nrows=1, ncols=5) #,figsize=(16,6))
    
    ds = xr.DataArray(obs_dataset.values,
    coords={'time': obs_dataset.times,
    'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
    dims=["time", "lat", "lon"])
    
    print('ds_xarray cek mak=',ds.max())
    # ada 2 pilihan           
    ds = ds.mean(dim=("lat", "lon"))
    #ds = ds.sum(dim=("lat", "lon"))
    
    #ini bgm sb x tahun sb y hujan/bulan ya?
    
    #N = df_nino.shape[0]
    N=ds.time.shape[0]
    t0 = [int(s) for s in str(ds.time[0].values.astype('datetime64[D]')).split('-') if s.isdigit()][0]
    dt = .09
    time = np.arange(0, N) * dt + t0
    
    signal = ds.values.squeeze() 
    signal = signal - np.mean(signal)
    
 
    scales = np.arange(1, 222) 
  
    waveletname='cmor1.5-1.0' 
    cmap=plt.cm.seismic 
    title='Wavelet Transform'
    ylabel='Period (years)'
    xlabel='Time'

    dt = time[1] - time[0]
    [coefficients, frequencies] = pywt.cwt(signal, scales, 
                                    waveletname, dt)
    power = (abs(coefficients)) ** 2
    #print(power)
    period = 1. / frequencies

    scale0 = 0.03
    numlevels = 10

    levels = [scale0]
    for ll in range(1, numlevels):
        scale0 *= 2
        levels.append(scale0)
    
    print(power.min(), power.max())
    print('levels=', levels)

    contourlevels = np.log2(levels)
   
    
    # fig, ax = plt.subplots(figsize=(8, 6))
    im = axes[0].contourf(time, np.log2(period), np.log2(power),
                     contourlevels, extend='both', cmap=cmap)
    #print('len time',len(time))
    #axes[0].set_xticks(np.arange(len(time)))
    r1=power  
    sd1=r1.std() #(skipna=None)
    r0=1
    T=[]
    for i in np.arange(len(model_datasets)):
        print(i, model_names[i])
        dsi = xr.DataArray(model_datasets[i].values,
        coords={'time': obs_dataset.times,
        'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
        dims=["time", "lat", "lon"])
                    
        dsi = dsi.mean(dim=("lat", "lon"))
        
        signal = dsi.values.squeeze() 
        signal = signal - np.mean(signal)
        
    
        [coefficients, frequencies] = pywt.cwt(signal, scales, waveletname, dt)
        power = (abs(coefficients)) ** 2
        #print(power)
        period = 1. / frequencies

        if i<len(model_datasets):
            im = axes[1+i].contourf(time, np.log2(period), np.log2(power),
                         contourlevels, extend='both', cmap=cmap)
        
        #axes[i].set_xticks(time, fontsize=1, rotation='vertical')
        #----ini akan di R kan dgn obs
        #r2=np.log2(power)  
        r2=power  
        sd2=r2.std() #(skipna=None)
        s=sd2/sd1
        c,pp=pearsonr(r1.flatten() , r2.flatten())        
      
        tt= (4*(1+c))/(((s+1/s)**2)*((1+r0)))
        T.append(round(tt,2))
        
        #similarity score
        #from sklearn.metrics.pairwise import cosine_similarity
    
        features_graph1 = np.array(r1)  # Replace with your feature data
        features_graph2 = np.array(r2)  # Replace with your feature data

        # Reshape feature arrays if needed
        features_graph1 = features_graph1.reshape(1, -1)
        features_graph2 = features_graph2.reshape(1, -1)
        
        #ss = ssim(r1, r2)
        
        
        # Calculate cosine similarity
        #cs = cosine_similarity(features_graph1, features_graph2)[0,0]
        #ed = euclidean_distances(features_graph1, features_graph2)[0,0]
        
        #print("Similarity Score cs, ss:", cs,ss)
        
        #ax.set_title(title, fontsize=20)
        axes[0].set_ylabel(ylabel, fontsize=10)
              
        
        yticks = 2**np.arange(np.ceil(np.log2(period.min())),
                              np.ceil(np.log2(period.max())))
        axes[0].set_yticks(np.log2(yticks))
        #axes[0].set_yticklabels(yticks, fontsize=8)
        axes[0].set_yticklabels(['1/8','1/4','1/2','1','2','4','8','16'],fontsize=8)
        
        if i<len(model_datasets):
            #axes[1+i].annotate('T='+'%.2f'%tt, fontsize=6, xy=(1985,0.9*np.log2(period).min()), backgroundcolor='0.85',alpha=1)
            axes[1+i].set_yticks([])
            #axes[1+i].set_xticks([])
            axes[1+i].set_title(model_names[i]+' ['+'%.2f'%tt+']',fontsize=9)
        
        axes[0].set_title(obs_name, fontsize=8)
        cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.7])
        cbar_ax.tick_params(labelsize=8)
        #this no effect
        plt.xticks(rotation=45)
        
        #fig.colorbar(im, cax=cbar_ax)
        fig.colorbar(im, cax=cbar_ax, ticks=[-5,-4,-3,-2,-1,0,1,2,3,4])  #, orientation="vertical")
    print(T)
    
    plt.subplots_adjust(hspace=.3,wspace=.04)
    plt.show()
    plt.savefig(workdir+reg+'_wavelet_5obs.png', dpi=300, bbox_inches='tight')

def cluster_wavelet(model_name, clusters, cluster_names, workdir):
    import pywt
    from scipy.stats import pearsonr
    from sklearn.metrics.pairwise import cosine_similarity
    from skimage.metrics import structural_similarity as ssim
    
    plt.style.use('seaborn')
    
    fig, axes = plt.subplots(nrows=1, ncols=len(cluster_names)) #,figsize=(16,6))
    
    ds=clusters[0]
    
    #ini bgm sb x tahun sb y hujan/bulan ya?
    
    #N = df_nino.shape[0]
    N=ds.time.shape[0]
    t0 = [int(s) for s in str(ds.time[0].values.astype('datetime64[D]')).split('-') if s.isdigit()][0]
    dt = .09
    time = np.arange(0, N) * dt + t0
    
    signal = ds.values.squeeze() 
    signal = signal - np.mean(signal)
    
 
    scales = np.arange(1, 222) 
  
    waveletname='cmor1.5-1.0' 
    cmap=plt.cm.seismic 
    title='Wavelet Transform'
    ylabel='Period (years)'
    xlabel='Time'

    dt = time[1] - time[0]
    [coefficients, frequencies] = pywt.cwt(signal, scales, 
                                    waveletname, dt)
    power = (abs(coefficients)) ** 2
    #print(power)
    period = 1. / frequencies

    scale0 = 0.03
    numlevels = 10

    levels = [scale0]
    for ll in range(1, numlevels):
        scale0 *= 2
        levels.append(scale0)
    
    print(power.min(), power.max())
    print('levels=', levels)

    contourlevels = np.log2(levels)
   
    
   
    for i in np.arange(len(cluster_names)):
        print(i, cluster_names[i])
        dsi=clusters[i]
        
        signal = dsi.values.squeeze() 
        signal = signal - np.mean(signal)
        
    
        [coefficients, frequencies] = pywt.cwt(signal, scales, waveletname, dt)
        power = (abs(coefficients)) ** 2
        #print(power)
        period = 1. / frequencies

        if i<len(cluster_names):
            im = axes[i].contourf(time, np.log2(period), np.log2(power),
                         contourlevels, extend='both', cmap=cmap)
        
        #axes[i].set_xticks(time, fontsize=1, rotation='vertical')
        #ax.set_title(title, fontsize=20)
        axes[0].set_ylabel(ylabel, fontsize=10)
              
        
        yticks = 2**np.arange(np.ceil(np.log2(period.min())),
                              np.ceil(np.log2(period.max())))
        axes[0].set_yticks(np.log2(yticks))
        #axes[0].set_yticklabels(yticks, fontsize=8)
        axes[0].set_yticklabels(['1/8','1/4','1/2','1','2','4','8','16'],fontsize=8)
        
        if i<len(cluster_names):
            #axes[1+i].annotate('T='+'%.2f'%tt, fontsize=6, xy=(1985,0.9*np.log2(period).min()), backgroundcolor='0.85',alpha=1)
            axes[i].set_yticks([])
            #axes[1+i].set_xticks([])
            axes[i].set_title('cluster '+str(i+1),fontsize=10)
            axes[i].set_xticks([])
            axes[i].set_xlabel('1976-2005',fontsize=8)
            
            #Tahhun dimiringkan sulit time terlalu banyak
            #axes[i].set_xticks(time, rotation=45)
        
        #axes[0].set_title(model_names[0], fontsize=8)
        #cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.7])
        #cbar_ax.tick_params(labelsize=8)
        #this no effect
        #plt.xticks(rotation=45)
        
        #fig.colorbar(im, cax=cbar_ax)
        #fig.colorbar(im, cax=cbar_ax, ticks=[-5,-4,-3,-2,-1,0,1,2,3,4])  #, orientation="vertical")
   
    plt.subplots_adjust(right=.35)
    plt.subplots_adjust(bottom=.3)
    plt.subplots_adjust(hspace=.3,wspace=.04)
    plt.show()
    plt.savefig(workdir+reg+'_wavelet_xxxx.png', dpi=300, bbox_inches='tight') 

def wavelet_14sum(obs_dataset, obs_name, model_datasets, model_names, workdir):
    import pywt
    from scipy.stats import pearsonr
    plt.style.use('seaborn')
    
    ds = xr.DataArray(obs_dataset.values,
    coords={'time': obs_dataset.times,
    'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
    dims=["time", "lat", "lon"])
    
    print('ds_xarray cek mak=',ds.max())
    #exit()

    #rata-rata hujan bulanan di SEA
    #ds = ds.mean(dim=("lat", "lon"))
    
    #Jumlah hujan bulan di SEA
    ds = ds.sum(dim=("lat", "lon"))
    
    #ini bgm sb x tahun sb y hujan/bulan ya?
    
    #N = df_nino.shape[0]
    N=ds.time.shape[0]
    t0 = [int(s) for s in str(ds.time[0].values.astype('datetime64[D]')).split('-') if s.isdigit()][0]
    dt = .09
    time = np.arange(0, N) * dt + t0
    
    signal = ds.values.squeeze() 
    signal = signal - np.mean(signal)
    
    #plt.plot(signal)
    #plt.show()
    #exit()
    # #detrending > data berkurang 1 di ujung 
    # X = signal
    # diff = list()
    # for i in range(1, len(X)):
        # value = X[i] - X[i - 1]
        # diff.append(value)
    
    # #plt.plot(X,'r')
    # #plt.plot(diff)
    # #plt.show()
    # #exit()
    # signal=diff
    # time=np.delete(time,-1)
    
    scales = np.arange(1, 222) 
  
    waveletname='cmor1.5-1.0' 
    cmap=plt.cm.seismic 
    title='Wavelet Transform'
    ylabel='Period (years)'
    xlabel='Time'

    dt = time[1] - time[0]
    [coefficients, frequencies] = pywt.cwt(signal, scales, 
                                    waveletname, dt)
    power = (abs(coefficients)) ** 2
    print('power.min_max_mean=', power.min(), power.max(), power.mean())
    
    period = 1. / frequencies

    scale0 = power.mean()/50 #2500000
    #pendekatan power.mean()/50 untuk interval warna/kontur lumayan pas
    numlevels = 10

    levels = [scale0]
    for ll in range(1, numlevels):
        scale0 *= 2
        levels.append(scale0)
   
    print(levels)

    contourlevels = np.log2(levels)
    fig, axes = plt.subplots(nrows=1, ncols=15) #,figsize=(16,6))
    
    # fig, ax = plt.subplots(figsize=(8, 6))
    im = axes[0].contourf(time, np.log2(period), np.log2(power),
                     contourlevels, extend='both', cmap=cmap)
  
    axes[0].set_xlabel('GPCP', fontsize=6)
    
    r1=power  
    sd1=r1.std() #(skipna=None)
    r0=1
    T=[]
    for i in np.arange(len(model_datasets)):
        dsi = xr.DataArray(model_datasets[i].values,
        coords={'time': obs_dataset.times,
        'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
        dims=["time", "lat", "lon"])
                    
        dsi = dsi.sum(dim=("lat", "lon"))
        
        signal = dsi.values.squeeze() 
        signal = signal - np.mean(signal)
        
        # #detrending
        # X = signal
        # diff = list()
        # for ii in range(1, len(X)):
            # value = X[ii] - X[ii - 1]
            # diff.append(value)
        
        # #plt.plot(X,'r')
        # #plt.plot(diff)
        # #plt.show()
        # #exit()
        # signal=diff
       
              
        [coefficients, frequencies] = pywt.cwt(signal, scales, waveletname, dt)
        power = (abs(coefficients)) ** 2
        #print(power)
        period = 1. / frequencies

       
        #fig, ax = plt.subplots(figsize=(8, 6))
        im = axes[1+i].contourf(time, np.log2(period), np.log2(power),
                         contourlevels, extend='both', cmap=cmap)
        
        #----ini akan di R kan dgn obs
        #r2=np.log2(power)  
        r2=power  
        sd2=r2.std() #(skipna=None)
        s=sd2/sd1
        c,pp=pearsonr(r1.flatten() , r2.flatten())        
        
        #T= (4*(1+c)**4)/(((s+1/s)**2)*((1+r0)**4))
        # Tanpa pakai **4 T naik dikit
        tt= (4*(1+c))/(((s+1/s)**2)*((1+r0)))
        T.append(round(tt,2))
        
        #ax.set_title(title, fontsize=20)
        axes[0].set_ylabel(ylabel, fontsize=10)
        if i<15: axes[1+i].set_yticks([]) 
        
        axes[i].set_xticks([])
        axes[14].set_xticks([])
        axes[1+i].set_xlabel(model_names[i], fontsize=6,rotation=30)
        #khusus noIPSL
        axes[12].set_xticks([])
        axes[13].set_xticks([])
        axes[13].set_yticks([])
        axes[14].set_yticks([])
        
        
        yticks = 2**np.arange(np.ceil(np.log2(period.min())),
                              np.ceil(np.log2(period.max())))
        axes[0].set_yticks(np.log2(yticks))
        #axes[0].set_yticklabels(yticks, fontsize=8)
        axes[0].set_yticklabels(['1/8','1/4','1/2','1','2','4','8','16'],fontsize=8)
        #axes[0].invert_yaxis()
        #ylim = axes[0].get_ylim()
        #axes[0].set_ylim(ylim[0], -1)
        axes[1+i].annotate('S='+'%.2f'%tt, fontsize=6, xy=(1985,0.9*np.log2(period).min()), backgroundcolor='0.85',alpha=1)
        
        
        
        cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.7])
        cbar_ax.tick_params(labelsize=8)
        #plt.xlabel(rotation=45)
        fig.colorbar(im, cax=cbar_ax)
        #fig.colorbar(im, cax=cbar_ax, ticks=[-5,-4,-3,-2,-1,0,1,2,3,4])  #, orientation="vertical")
    print(T)
    
    plt.subplots_adjust(hspace=.3,wspace=.04)
    #plt.savefig(workdir+reg+'_wavelet_14_.png', dpi=300, bbox_inches='tight')
    plt.show()
    
def wavelet_14sum_pycwt(obs_dataset, obs_name, model_datasets, model_names, workdir):
    print('wavelet_14sum_pycwt...')
    
    import pycwt as wavelet
    from scipy.stats import pearsonr
    from matplotlib import pyplot
    import numpy
    #pyplot.close('all')
    #pyplot.ioff()
    #figprops = dict(figsize=(11, 8), dpi=72)
    #fig = pyplot.figure(**figprops)
    
    
    ds = xr.DataArray(obs_dataset.values,
    coords={'time': obs_dataset.times,
    'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
    dims=["time", "lat", "lon"])
    
    print('ds_xarray cek mak=',ds.max())
    #exit()            
    ds = ds.mean(dim=("lat", "lon"))
    #ds = ds.sum(dim=("lat", "lon"))
    
    #ini bgm sb x tahun sb y hujan/bulan ya?
    
    #N = df_nino.shape[0]
    N=ds.time.shape[0]
    t0 = [int(s) for s in str(ds.time[0].values.astype('datetime64[D]')).split('-') if s.isdigit()][0]
    dt = .09
    t = np.arange(0, N) * dt + t0
    
   
    signal = ds.values.squeeze() 
    signal = signal - np.mean(signal)
    
    dat=signal
    p = numpy.polyfit(t - t0, dat, 1)
    dat_notrend = dat - numpy.polyval(p, t - t0)
    std = dat_notrend.std()  # Standard deviation
    var = std ** 2  # Variance
    dat_norm = dat_notrend / std  # Normalized dataset
    
    
    mother = wavelet.Morlet(6)
    s0 = 2 * dt  # Starting scale, in this case 2 * 0.25 years = 6 months
    dj = 1 / 12  # Twelve sub-octaves per octaves
    J = 7 / dj  # Seven powers of two with dj sub-octaves
    alpha, _, _ = wavelet.ar1(dat)  # Lag-1 autocorrelation for red noise

    # The following routines perform the wavelet transform and inverse wavelet
    # transform using the parameters defined above. Since we have normalized our
    # input time-series, we multiply the inverse transform by the standard
    # deviation.
    wave, scales, freqs, coi, fft, fftfreqs = wavelet.cwt(dat_norm, dt, dj, s0, J,
                                                          mother)
    iwave = wavelet.icwt(wave, scales, dt, dj, mother) * std

    # We calculate the normalized wavelet and Fourier power spectra, as well as
    # the Fourier equivalent periods for each wavelet scale.
    power = (numpy.abs(wave)) ** 2
    fft_power = numpy.abs(fft) ** 2
    period = 1 / freqs

    #Optionally, we could also rectify the power spectrum according to the 
    #suggestions proposed by Liu et al. (2007)[2]
    #power /= scales[:, None]

    # We could stop at this point and plot our results. However we are also
    # interested in the power spectra significance test. The power is significant
    # where the ratio ``power / sig95 > 1``.
    signif, fft_theor = wavelet.significance(1.0, dt, scales, 0, alpha,
                                             significance_level=0.95,
                                             wavelet=mother)
    sig95 = numpy.ones([1, N]) * signif[:, None]
    sig95 = power / sig95
   
    fig, axes = plt.subplots(nrows=1, ncols=15)
    #bx = pyplot.axes([0.1, 0.37, 0.65, 0.28])
    levels = [1/32, 0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16]
    im=axes[0].contourf(t, numpy.log2(period), numpy.log2(power), numpy.log2(levels),
                extend='both', 
                #cmap=pyplot.cm.viridis)
                cmap=pyplot.cm.seismic)
    extent = [t.min(), t.max(), 0, max(period)]
    axes[0].contour(t, numpy.log2(period), sig95, [-99, 1], colors='k', linewidths=2,
               extent=extent)
    
    # bx.fill(numpy.concatenate([t, t[-1:] + dt, t[-1:] + dt,
                               # t[:1] - dt, t[:1] - dt]),
            # numpy.concatenate([numpy.log2(coi), [1e-9], numpy.log2(period[-1:]),
                               # numpy.log2(period[-1:]), [1e-9]]),
            # 'k', alpha=0.3, hatch='x')
    #bx.set_title('b) {} Wavelet Power Spectrum ({})'.format(label='tes', mother.name))
    
    axes[0].set_ylabel('Period (years)')
    #
    Yticks = 2 ** numpy.arange(numpy.ceil(numpy.log2(period.min())),
                               numpy.ceil(numpy.log2(period.max())))
    axes[0].set_yticks(numpy.log2(Yticks))
    #axes[0].set_yticklabels(Yticks)
    axes[0].set_yticklabels(['1/4','1/2','1','2','4','8','16'],fontsize=8)
  
    axes[0].set_xlabel('GPCP', fontsize=6)
    
    r1=power  
    sd1=r1.std() #(skipna=None)
    r0=1
    T=[]
    for i in np.arange(len(model_datasets)):
        dsi = xr.DataArray(model_datasets[i].values,
        coords={'time': obs_dataset.times,
        'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
        dims=["time", "lat", "lon"])
                    
        dsi = dsi.mean(dim=("lat", "lon"))
        
        signal = dsi.values.squeeze() 
        signal = signal - np.mean(signal)
        
        dat=signal
        p = numpy.polyfit(t - t0, dat, 1)
        dat_notrend = dat - numpy.polyval(p, t - t0)
        std = dat_notrend.std()  # Standard deviation
        var = std ** 2  # Variance
        dat_norm = dat_notrend / std  # Normalized dataset
        
        
        mother = wavelet.Morlet(6)
        s0 = 2 * dt  # Starting scale, in this case 2 * 0.25 years = 6 months
        dj = 1 / 12  # Twelve sub-octaves per octaves
        J = 7 / dj  # Seven powers of two with dj sub-octaves
        alpha, _, _ = wavelet.ar1(dat)  # Lag-1 autocorrelation for red noise

        # The following routines perform the wavelet transform and inverse wavelet
        # transform using the parameters defined above. Since we have normalized our
        # input time-series, we multiply the inverse transform by the standard
        # deviation.
        wave, scales, freqs, coi, fft, fftfreqs = wavelet.cwt(dat_norm, dt, dj, s0, J,
                                                              mother)
        iwave = wavelet.icwt(wave, scales, dt, dj, mother) * std

        # We calculate the normalized wavelet and Fourier power spectra, as well as
        # the Fourier equivalent periods for each wavelet scale.
        power = (numpy.abs(wave)) ** 2
        fft_power = numpy.abs(fft) ** 2
        period = 1 / freqs

        #Optionally, we could also rectify the power spectrum according to the 
        #suggestions proposed by Liu et al. (2007)[2]
        #power /= scales[:, None]

        # We could stop at this point and plot our results. However we are also
        # interested in the power spectra significance test. The power is significant
        # where the ratio ``power / sig95 > 1``.
        signif, fft_theor = wavelet.significance(1.0, dt, scales, 0, alpha,
                                                 significance_level=0.95,
                                                 wavelet=mother)
        sig95 = numpy.ones([1, N]) * signif[:, None]
        sig95 = power / sig95
       
        
        
        im=axes[i+1].contourf(t, numpy.log2(period), numpy.log2(power), numpy.log2(levels),
                    extend='both', 
                    #cmap=pyplot.cm.viridis)
                    cmap=pyplot.cm.seismic)
        extent = [t.min(), t.max(), 0, max(period)]
        axes[i+1].contour(t, numpy.log2(period), sig95, [-99, 1], colors='k', linewidths=2,
                   extent=extent)
        
        #----ini akan di R kan dgn obs
        #r2=np.log2(power)  
        r2=power  
        sd2=r2.std() #(skipna=None)
        s=sd2/sd1
        c,pp=pearsonr(r1.flatten() , r2.flatten())        
        
        #T= (4*(1+c)**4)/(((s+1/s)**2)*((1+r0)**4))
        # Tanpa pakai **4 T naik dikit
        tt= (4*(1+c))/(((s+1/s)**2)*((1+r0)))
        T.append(round(tt,2))
        
        #ax.set_title(title, fontsize=20)
        #axes[0].set_ylabel(ylabel, fontsize=10)
        if i<15: axes[1+i].set_yticks([]) 
        
        axes[i].set_xticks([])
        axes[14].set_xticks([])
        axes[1+i].set_xlabel(model_names[i], fontsize=6,rotation=30)
        #khusus noIPSL
        axes[12].set_xticks([])
        axes[13].set_xticks([])
        axes[13].set_yticks([])
        axes[14].set_yticks([])
        
        
        yticks = 2**np.arange(np.ceil(np.log2(period.min())),
                              np.ceil(np.log2(period.max())))
        #axes[0].set_yticks(np.log2(yticks))
        #axes[0].set_yticklabels(yticks, fontsize=8)
        #axes[0].set_yticklabels(['1/8','1/4','1/2','1','2','4','8','16'],fontsize=8)
        #axes[0].invert_yaxis()
        #ylim = axes[0].get_ylim()
        #axes[0].set_ylim(ylim[0], -1)
        axes[1+i].annotate('S='+'%.2f'%tt, fontsize=4, xy=(1985,0.9*np.log2(period).min()), backgroundcolor='0.85',alpha=1)
        
        
        
        cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.7])
        cbar_ax.tick_params(labelsize=8)
        #plt.xlabel(rotation=45)
        fig.colorbar(im, cax=cbar_ax)
        #fig.colorbar(im, cax=cbar_ax, ticks=[-5,-4,-3,-2,-1,0,1,2,3,4])  #, orientation="vertical")
    print(T)
    
    plt.subplots_adjust(hspace=.3,wspace=.06)
    plt.savefig(workdir+reg+'_wavelet_14_mean_pycwt2_xxtes.png', dpi=300, bbox_inches='tight')
    plt.show()

def wavelet_14sum_pycwt_2(obs_dataset, obs_name, model_datasets, model_names, workdir):
    print('wavelet_14sum_pycwt...graph susun ke bawah')
    
    import pycwt as wavelet
    from scipy.stats import pearsonr
    from matplotlib import pyplot
    import numpy
    #pyplot.close('all')
    #pyplot.ioff()
    #figprops = dict(figsize=(11, 8), dpi=72)
    #fig = pyplot.figure(**figprops)
    
    fig, axes = plt.subplots(nrows=4, ncols=1)
    
    ds = xr.DataArray(obs_dataset.values,
    coords={'time': obs_dataset.times,
    'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
    dims=["time", "lat", "lon"])
    
    print('ds_xarray cek mak=',ds.max())
    #exit()            
    ds = ds.mean(dim=("lat", "lon"))
    #ds = ds.sum(dim=("lat", "lon"))
    
    #ini bgm sb x tahun sb y hujan/bulan ya?
    
    #N = df_nino.shape[0]
    N=ds.time.shape[0]
    t0 = [int(s) for s in str(ds.time[0].values.astype('datetime64[D]')).split('-') if s.isdigit()][0]
    dt = .09
    t = np.arange(0, N) * dt + t0
    
   
    signal = ds.values.squeeze() 
    signal = signal - np.mean(signal)
    
    dat=signal
    p = numpy.polyfit(t - t0, dat, 1)
    dat_notrend = dat - numpy.polyval(p, t - t0)
    std = dat_notrend.std()  # Standard deviation
    var = std ** 2  # Variance
    dat_norm = dat_notrend / std  # Normalized dataset
    
    
    mother = wavelet.Morlet(6)
    s0 = 2 * dt  # Starting scale, in this case 2 * 0.25 years = 6 months
    dj = 1 / 12  # Twelve sub-octaves per octaves
    J = 7 / dj  # Seven powers of two with dj sub-octaves
    alpha, _, _ = wavelet.ar1(dat)  # Lag-1 autocorrelation for red noise

    # The following routines perform the wavelet transform and inverse wavelet
    # transform using the parameters defined above. Since we have normalized our
    # input time-series, we multiply the inverse transform by the standard
    # deviation.
    wave, scales, freqs, coi, fft, fftfreqs = wavelet.cwt(dat_norm, dt, dj, s0, J,
                                                          mother)
    iwave = wavelet.icwt(wave, scales, dt, dj, mother) * std

    # We calculate the normalized wavelet and Fourier power spectra, as well as
    # the Fourier equivalent periods for each wavelet scale.
    power = (numpy.abs(wave)) ** 2
    fft_power = numpy.abs(fft) ** 2
    period = 1 / freqs

    #Optionally, we could also rectify the power spectrum according to the 
    #suggestions proposed by Liu et al. (2007)[2]
    #power /= scales[:, None]

    # We could stop at this point and plot our results. However we are also
    # interested in the power spectra significance test. The power is significant
    # where the ratio ``power / sig95 > 1``.
    signif, fft_theor = wavelet.significance(1.0, dt, scales, 0, alpha,
                                             significance_level=0.95,
                                             wavelet=mother)
    sig95 = numpy.ones([1, N]) * signif[:, None]
    sig95 = power / sig95
   
    
    #bx = pyplot.axes([0.1, 0.37, 0.65, 0.28])
    levels = [1/32, 0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16]
    im=axes[0].contourf(t, numpy.log2(period), numpy.log2(power), numpy.log2(levels),
                extend='both', 
                #cmap=pyplot.cm.viridis)
                cmap=pyplot.cm.seismic)
    extent = [t.min(), t.max(), 0, max(period)]
    axes[0].contour(t, numpy.log2(period), sig95, [-99, 1], colors='k', linewidths=2,
               extent=extent)
    
    # bx.fill(numpy.concatenate([t, t[-1:] + dt, t[-1:] + dt,
                               # t[:1] - dt, t[:1] - dt]),
            # numpy.concatenate([numpy.log2(coi), [1e-9], numpy.log2(period[-1:]),
                               # numpy.log2(period[-1:]), [1e-9]]),
            # 'k', alpha=0.3, hatch='x')
    #bx.set_title('b) {} Wavelet Power Spectrum ({})'.format(label='tes', mother.name))
    
    #axes[0].set_ylabel('Period (years)')
    #
    #Yticks = 2 ** numpy.arange(numpy.ceil(numpy.log2(period.min())),
    #                           numpy.ceil(numpy.log2(period.max())))
    #axes[0].set_yticks(numpy.log2(Yticks))
    #axes[0].set_yticklabels(Yticks)
    #axes[0].set_yticklabels(['1/4','1/2','1','2','4','8','16'],fontsize=8)
  
    #axes[0].set_xlabel(obs_name, fontsize=6)
    
    r1=power  
    sd1=r1.std() #(skipna=None)
    r0=1
    T=[]
    for i in np.arange(len(model_datasets)-11):
        dsi = xr.DataArray(model_datasets[i].values,
        coords={'time': obs_dataset.times,
        'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
        dims=["time", "lat", "lon"])
                    
        dsi = dsi.mean(dim=("lat", "lon"))
        
        signal = dsi.values.squeeze() 
        signal = signal - np.mean(signal)
        
        dat=signal
        p = numpy.polyfit(t - t0, dat, 1)
        dat_notrend = dat - numpy.polyval(p, t - t0)
        std = dat_notrend.std()  # Standard deviation
        var = std ** 2  # Variance
        dat_norm = dat_notrend / std  # Normalized dataset
        
        
        mother = wavelet.Morlet(6)
        s0 = 2 * dt  # Starting scale, in this case 2 * 0.25 years = 6 months
        dj = 1 / 12  # Twelve sub-octaves per octaves
        J = 7 / dj  # Seven powers of two with dj sub-octaves
        alpha, _, _ = wavelet.ar1(dat)  # Lag-1 autocorrelation for red noise

        # The following routines perform the wavelet transform and inverse wavelet
        # transform using the parameters defined above. Since we have normalized our
        # input time-series, we multiply the inverse transform by the standard
        # deviation.
        wave, scales, freqs, coi, fft, fftfreqs = wavelet.cwt(dat_norm, dt, dj, s0, J,
                                                              mother)
        iwave = wavelet.icwt(wave, scales, dt, dj, mother) * std

        # We calculate the normalized wavelet and Fourier power spectra, as well as
        # the Fourier equivalent periods for each wavelet scale.
        power = (numpy.abs(wave)) ** 2
        fft_power = numpy.abs(fft) ** 2
        period = 1 / freqs

        #Optionally, we could also rectify the power spectrum according to the 
        #suggestions proposed by Liu et al. (2007)[2]
        #power /= scales[:, None]

        # We could stop at this point and plot our results. However we are also
        # interested in the power spectra significance test. The power is significant
        # where the ratio ``power / sig95 > 1``.
        signif, fft_theor = wavelet.significance(1.0, dt, scales, 0, alpha,
                                                 significance_level=0.95,
                                                 wavelet=mother)
        sig95 = numpy.ones([1, N]) * signif[:, None]
        sig95 = power / sig95
       
        
        
        im=axes[i+1].contourf(t, numpy.log2(period), numpy.log2(power), numpy.log2(levels),
                    extend='both', 
                    #cmap=pyplot.cm.viridis)
                    cmap=pyplot.cm.seismic)
        extent = [t.min(), t.max(), 0, max(period)]
        axes[i+1].contour(t, numpy.log2(period), sig95, [-99, 1], colors='k', linewidths=2,
                   extent=extent)
        
        #----ini akan di R kan dgn obs
        #r2=np.log2(power)  
        r2=power  
        sd2=r2.std() #(skipna=None)
        s=sd2/sd1
        c,pp=pearsonr(r1.flatten() , r2.flatten())        
        
        #T= (4*(1+c)**4)/(((s+1/s)**2)*((1+r0)**4))
        # Tanpa pakai **4 T naik dikit
        tt= (4*(1+c))/(((s+1/s)**2)*((1+r0)))
        T.append(round(tt,2))
        
        #ax.set_title(title, fontsize=20)
        #axes[0].set_ylabel(ylabel, fontsize=10)
        if i<15: axes[1+i].set_yticks([]) 
        
        #axes[i].set_xticks([])
        #axes[14].set_xticks([])
       # axes[1+i].set_xlabel(model_names[i], fontsize=6,rotation=30)
        #khusus noIPSL
        #axes[12].set_xticks([])
        #axes[13].set_xticks([])
        #axes[13].set_yticks([])
        #axes[14].set_yticks([])
        
        
       # yticks = 2**np.arange(np.ceil(np.log2(period.min())),
          #                    np.ceil(np.log2(period.max())))
        #axes[0].set_yticks(np.log2(yticks))
        #axes[0].set_yticklabels(yticks, fontsize=8)
        #axes[0].set_yticklabels(['1/8','1/4','1/2','1','2','4','8','16'],fontsize=8)
        #axes[0].invert_yaxis()
        #ylim = axes[0].get_ylim()
        #axes[0].set_ylim(ylim[0], -1)
        #axes[1+i].annotate('S='+'%.2f'%tt, fontsize=4, xy=(1985,0.9*np.log2(period).min()), backgroundcolor='0.85',alpha=1)
        
        
        
        cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.7])
        cbar_ax.tick_params(labelsize=8)
        #plt.xlabel(rotation=45)
        fig.colorbar(im, cax=cbar_ax)
        #fig.colorbar(im, cax=cbar_ax, ticks=[-5,-4,-3,-2,-1,0,1,2,3,4])  #, orientation="vertical")
    print(T)
    
    plt.subplots_adjust(hspace=.3,wspace=.06)
    plt.savefig(workdir+reg+'_wavelet_14_mean_pycwt2_xxtes.png', dpi=300, bbox_inches='tight')
    plt.show()

  
def wavelet_nino(obs_dataset, obs_name, model_datasets, model_names, workdir):
    print('wavelet_14sum_pycwt...')
    
    import pywt
    from scipy.stats import pearsonr
    from matplotlib import pyplot
    import numpy
    import xarray as xr 
    import pandas as pd
    fig, axes = plt.subplots(nrows=1, ncols=6)
    
    d= pd.read_excel('D:/Disertasi3/enso_mon_1981-2005.xlsx')
    
    
    nino_obs1 = xr.DataArray( d['Value'],
    coords={'time': obs_dataset.times})
    
    '''
    gb = nino_obs1.groupby('time.month')
    nino_obs = gb - gb.mean(dim='time')
    nino_obs = nino_obs.rolling(time=5, center=True).mean()
   
    ds=nino_obs1
    #print(ds.time.values)
    '''
    ds=nino_obs1
    time=np.delete(ds.time.values,-1)
    
    N=time.shape[0]
    t0 = [int(s) for s in str(ds.time[0].values.astype('datetime64[D]')).split('-') if s.isdigit()][0]
    dt = .09
    t = np.arange(0, N) * dt + t0
    #print('t0=',t0)
   
    signal=np.delete(nino_obs1.values,-1)
    

    signal = signal - np.mean(signal)
    
    scales = np.arange(1, 222) 
  
    waveletname='cmor1.5-1.0' 
    cmap=plt.cm.seismic 
    title='Wavelet Transform'
    ylabel='Period (years)'
    xlabel='Time'

    dt = t[1] - t[0]
    [coefficients, frequencies] = pywt.cwt(signal, scales, 
                                    waveletname, dt)
    power = (abs(coefficients)) ** 2
    print(power)
    period = 1. / frequencies

    scale0 = 0.05
    numlevels = 10

    levels = [scale0]
    for ll in range(1, numlevels):
        scale0 *= 2
        levels.append(scale0)
    #print(power.min(), power.max())
    #print(levels)
    
    contourlevels = np.log2(levels)
   
    
    # fig, ax = plt.subplots(figsize=(8, 6))
    im = axes[0].contourf(t, np.log2(period), np.log2(power),
                     contourlevels, extend='both', cmap=cmap)
  
    
    axes[0].set_ylabel('Period (years)')
     
    axes[0].set_xlabel('Obs', fontsize=8) #,rotation=30)
    
    r1=power  
    sd1=r1.std() #(skipna=None)
    
    r0=1
    T=[]
    
    filepath=[
    'D:/data1/tos_Omon_CNRM-CM5_historical_r1i1p1_198101-200512.nc',
    'D:/data1/tos_Omon_IPSL-CM5A-LR_historicalNat_r1i1p1_1981-2005_nino34_ok.nc',
    'D:/data1/tos_Omon_HadGEM2-ES_esmHistorical_r1i1p1_1981-2005_nino34_ok.nc',
    'D:/data1/tos_Omon_NorESM1-M_historical_r1i1p1_198101-200512.nc',
    'D:/data1/4tos_Omon_GFDL-ESM2M_historical_r1i1p1_198101-200512.nc'
    ]
    
    names=['CNRM', 'IPSL', 'HadGEM2', 'NorESM1', 'GFDL' ]
    
    
    import networkx as nx
    from sklearn.metrics.pairwise import cosine_similarity
        
    for i in np.arange(len(filepath)):
      
        print(i, names[i])
           
        dsx = xr.open_dataset(filepath[i])
        #print(dsx)
        #slice ini nans
        #tos_nino34 = ds.sel(i=slice(-5, 5), j=slice(190, 240))
        try:
            tos_nino34 = dsx.where(
                    (dsx.lat < 5) & (dsx.lat > -5) & 
                    (dsx.lon > 190) & (dsx.lon < 240), drop=True)
                    
        except:
          
            tos_nino34 = dsx.where(
                    (dsx.rlat < 5) & (dsx.rlat > -5) & 
                    (dsx.rlon > 190) & (dsx.rlon < 240), drop=True)
                    
        
        
        try: index_nino34 = tos_nino34.tos.mean(dim=['rlat', 'rlon'])
        except: 
            try: index_nino34 = tos_nino34.tos.mean(dim=['i', 'j'])
            except: index_nino34 = tos_nino34.tos.mean(dim=['lat', 'lon'])
        
      
        
        
        #signal = index_nino34.values #.squeeze() 
        #print(signal)
        if i!=2:
            signal=np.delete(index_nino34.values,-1)
        else:
            signal=index_nino34.values
        
        print(len(signal))
        signal = signal - np.mean(signal)
        
        [coefficients, frequencies] = pywt.cwt(signal, scales, waveletname, dt)
        power = (abs(coefficients)) ** 2
        #print(power)
        period = 1. / frequencies
       
       
        
        #fig, ax = plt.subplots(figsize=(8, 6))
        im = axes[1+i].contourf(t, np.log2(period), np.log2(power),
                         contourlevels, extend='both', cmap=cmap)
        
        #----ini akan di R kan dgn obs
        #r2=np.log2(power)  
        r2=power  
        sd2=r2.std() #(skipna=None)
        s=sd2/sd1
        
        print(len(r1.flatten()),len(r2.flatten()))
        
        c,pp=pearsonr(r1.flatten() , r2.flatten())        
        
        #T= (4*(1+c)**4)/(((s+1/s)**2)*((1+r0)**4))
        # Tanpa pakai **4 T naik dikit
        tt= (4*(1+c))/(((s+1/s)**2)*((1+r0)))
        T.append(round(tt,2))
        
        #similarity score
     
        features_graph1 = np.array(r1)  # Replace with your feature data
        features_graph2 = np.array()  # Replace with your feature data

        # Reshape feature arrays if needed
        features_graph1 = features_graph1.reshape(1, -1)
        features_graph2 = features_graph2.reshape(1, -1)

        # Calculate cosine similarity
        similarity_score = cosine_similarity(features_graph1, features_graph2)

        print("Similarity Score:", similarity_score)

        #ax.set_title(title, fontsize=20)
        axes[0].set_ylabel(ylabel, fontsize=10)
        if i<15: axes[1+i].set_yticks([]) 
        
        axes[i].set_xticks([])
        axes[5].set_xticks([])
        
        axes[1+i].set_xlabel(names[i], fontsize=8) #,rotation=30)
       
        
        yticks = 2**np.arange(np.ceil(np.log2(period.min())),
                              np.ceil(np.log2(period.max())))
        axes[0].set_yticks(np.log2(yticks))
        #axes[0].set_yticklabels(yticks, fontsize=8)
        axes[0].set_yticklabels(['1/8','1/4','1/2','1','2','4','8','16'],fontsize=8)
        #axes[0].invert_yaxis()
        #ylim = axes[0].get_ylim()
        #axes[0].set_ylim(ylim[0], -1)
        axes[1+i].annotate('S='+'%.2f'%tt, fontsize=6, xy=(1988,0.9*np.log2(period).min()), backgroundcolor='0.85',alpha=1)
        
        
        
        cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.7])
        cbar_ax.tick_params(labelsize=8)
        #plt.xlabel(rotation=45)
        fig.colorbar(im, cax=cbar_ax)
        fig.colorbar(im, cax=cbar_ax, ticks=[-4,-3,-2,-1,0,1,2,3,4])  #, orientation="vertical")
    print(T)
    
    plt.subplots_adjust(hspace=.3,wspace=.04)
    plt.savefig(workdir+reg+'_wavelet_nino_cmip5.png', dpi=300, bbox_inches='tight')
    plt.show()
    
def wavelet_iod(obs_dataset, obs_name, model_datasets, model_names, workdir):
    print('wavelet_14sum_pycwt...')
    
    import pywt
    from scipy.stats import pearsonr
    from matplotlib import pyplot
    import numpy
    import xarray as xr 
    import pandas as pd
    fig, axes = plt.subplots(nrows=1, ncols=6)
    
    d= pd.read_excel('D:/Disertasi3/dmi_1981-2005.xlsx')
    
    nino_obs1 = xr.DataArray( d['Value'],
    coords={'time': obs_dataset.times})
    
    '''
    gb = nino_obs1.groupby('time.month')
    nino_obs = gb - gb.mean(dim='time')
    nino_obs = nino_obs.rolling(time=5, center=True).mean()
    '''
   
    ds=nino_obs1
    #print(ds.time.values)
    
    time=np.delete(ds.time.values,-1)
    
    N=time.shape[0]
    t0 = [int(s) for s in str(ds.time[0].values.astype('datetime64[D]')).split('-') if s.isdigit()][0]
    dt = .09
    t = np.arange(0, N) * dt + t0
    #print('t0=',t0)
   
    signal=np.delete(nino_obs1.values,-1)
    

    signal = signal - np.mean(signal)
    
    scales = np.arange(1, 222) 
  
    waveletname='cmor1.5-1.0' 
    cmap=plt.cm.seismic 
    title='Wavelet Transform'
    ylabel='Period (years)'
    xlabel='Time'

    dt = t[1] - t[0]
    [coefficients, frequencies] = pywt.cwt(signal, scales, 
                                    waveletname, dt)
    power = (abs(coefficients)) ** 2
    print(power)
    print(power.min(), power.max())
    
    
    period = 1. / frequencies

    scale0 = 0.01
    numlevels = 10

    levels = [scale0]
    for ll in range(1, numlevels):
        scale0 *= 2
        levels.append(scale0)
    #print(power.min(), power.max())
    #print(levels)
    
    contourlevels = np.log2(levels)
   
    
    # fig, ax = plt.subplots(figsize=(8, 6))
    im = axes[0].contourf(t, np.log2(period), np.log2(power),
                     contourlevels, extend='both', cmap=cmap)
  
    
    axes[0].set_ylabel('Period (years)')
     
    axes[0].set_xlabel('Obs', fontsize=8) #,rotation=30)
    
    r1=power  
    sd1=r1.std() #(skipna=None)
    
    r0=1
    T=[]
    
    filepath=[
            'D:/data1/tos_Omon_CNRM-CM5_historical_r1i1p1_198101-200512.nc',
            'D:/data1/tos_Omon_IPSL-CM5A-LR_historicalNat_r1i1p1_1981-2005.nc',
            'D:/data1/tos_Omon_HadGEM2-ES_esmHistorical_r1i1p1_1981-2005_11.nc',
            'D:/data1/tos_Omon_NorESM1-M_historical_r1i1p1_198101-200512.nc',
            'D:/data1/4tos_Omon_GFDL-ESM2M_historical_r1i1p1_198101-200512.nc'
            ]
    
    names=['CNRM', 'IPSL', 'HadGEM2','NorESM1', 'GFDL']
    
    for i in np.arange(len(filepath)):
        print(i)
        dsx = xr.open_dataset(filepath[i])
  
        try:
            tos_w = dsx.where(
                    (dsx.lat < 10) & (dsx.lat > -10) & 
                    (dsx.lon > 50) & (dsx.lon < 70), drop=True)
        #GFDL pakai rlat,rlon             
        except:
            try:
                tos_w = dsx.where(
                    (dsx.rlat < 10) & (dsx.rlat > -10) & 
                    (dsx.rlon > 50) & (dsx.rlon < 70), drop=True)
            except:
                #ini tidak perlu
                print('i,j terpakai')
                tos_w = dsx.where(
                    (dsx.j < 10) & (dsx.j > -10) & 
                    (dsx.i > 50) & (dsx.i < 70), drop=True)   
        try:
            tos_e = dsx.where(
                    (dsx.lat < 0) & (dsx.lat > -10) & 
                    (dsx.lon > 90) & (dsx.lon < 110), drop=True)         
        except:
            try:
                tos_e = dsx.where(
                    (dsx.rlat < 0) & (dsx.rlat > -10) & 
                    (dsx.rlon > 90) & (dsx.rlon < 110), drop=True) 
            except:
                print('i,j terpakai')
                tos_e = dsx.where(
                    (dsx.j < 0) & (dsx.j > -10) & 
                    (dsx.i > 90) & (dsx.i < 110), drop=True) 
        
        tos_nino34=tos_w 
        
        try: index_nino34w = tos_nino34.tos.mean(dim=['rlat', 'rlon'])
        except: 
            try: index_nino34w = tos_nino34.tos.mean(dim=['i', 'j'])
            except: index_nino34w = tos_nino34.tos.mean(dim=['lat', 'lon'])
        
        tos_nino34=tos_e
        
        try: index_nino34e = tos_nino34.tos.mean(dim=['rlat', 'rlon'])
        except: 
            try: index_nino34e = tos_nino34.tos.mean(dim=['i', 'j'])
            except: index_nino34e = tos_nino34.tos.mean(dim=['lat', 'lon'])
      
        index_nino34 = index_nino34w - index_nino34e
        
        #signal = index_nino34.values #.squeeze() 
        #print(signal)
        if i!=2:
            signal=np.delete(index_nino34.values,-1)
        else:
            signal=index_nino34.values
        
        print(len(signal))
        signal = signal - np.mean(signal)
        
        [coefficients, frequencies] = pywt.cwt(signal, scales, waveletname, dt)
        power = (abs(coefficients)) ** 2
        #print(power)
        period = 1. / frequencies
       
       
        
        #fig, ax = plt.subplots(figsize=(8, 6))
        im = axes[1+i].contourf(t, np.log2(period), np.log2(power),
                         contourlevels, extend='both', cmap=cmap)
        
        #----ini akan di R kan dgn obs
        #r2=np.log2(power)  
        r2=power  
        sd2=r2.std() #(skipna=None)
        s=sd2/sd1
        
        print(len(r1.flatten()),len(r2.flatten()))
        
        c,pp=pearsonr(r1.flatten() , r2.flatten())        
        
        #T= (4*(1+c)**4)/(((s+1/s)**2)*((1+r0)**4))
        # Tanpa pakai **4 T naik dikit
        tt= (4*(1+c))/(((s+1/s)**2)*((1+r0)))
        T.append(round(tt,2))
        
        #ax.set_title(title, fontsize=20)
        axes[0].set_ylabel(ylabel, fontsize=10)
        if i<15: axes[1+i].set_yticks([]) 
        
        axes[i].set_xticks([])
        axes[5].set_xticks([])
        
        axes[1+i].set_xlabel(names[i], fontsize=8) #,rotation=30)
       
        
        yticks = 2**np.arange(np.ceil(np.log2(period.min())),
                              np.ceil(np.log2(period.max())))
        axes[0].set_yticks(np.log2(yticks))
        #axes[0].set_yticklabels(yticks, fontsize=8)
        axes[0].set_yticklabels(['1/8','1/4','1/2','1','2','4','8','16'],fontsize=8)
        #axes[0].invert_yaxis()
        #ylim = axes[0].get_ylim()
        #axes[0].set_ylim(ylim[0], -1)
        axes[1+i].annotate('S='+'%.2f'%tt, fontsize=6, xy=(1988,0.9*np.log2(period).min()), backgroundcolor='0.85',alpha=1)
        
        
        
        cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.7])
        cbar_ax.tick_params(labelsize=8)
        #plt.xlabel(rotation=45)
        fig.colorbar(im, cax=cbar_ax)
        fig.colorbar(im, cax=cbar_ax, ticks=np.arange(-6,3))  #, orientation="vertical")
    
    print(T)
    
    plt.subplots_adjust(hspace=.3,wspace=.04)
    plt.savefig(workdir+reg+'_wavelet_iod_cmip5.png', dpi=300, bbox_inches='tight')
    plt.show()
    
def wavelet_3p(obs_dataset, obs_name, model_datasets, model_names, workdir):
    import pywt
    plt.style.use('seaborn')
    
    ds = xr.DataArray(obs_dataset.values,
    coords={'time': obs_dataset.times,
    'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
    dims=["time", "lat", "lon"])
                
    ds = ds.mean(dim=("lat", "lon"))
    
    
    
    #N = df_nino.shape[0]
    N=ds.time.shape[0]
    t0 = [int(s) for s in str(ds.time[0].values.astype('datetime64[D]')).split('-') if s.isdigit()][0]
    dt = .09
    time = np.arange(0, N) * dt + t0
   

    signal = ds.values.squeeze() 
    signal = signal - np.mean(signal)

    scales = np.arange(1, 222) 
       
    waveletname='cmor1.5-1.0' 
    cmap=plt.cm.seismic 
    title='Wavelet Transform'
    ylabel='Period (years)'
    xlabel='Time'

    dt = time[1] - time[0]
    [coefficients, frequencies] = pywt.cwt(signal, scales, waveletname, dt)
    power = (abs(coefficients)) ** 2
    #print(power)
    period = 1. / frequencies

    scale0 = 0.03
    numlevels = 10

    levels = [scale0]
    for ll in range(1, numlevels):
        scale0 *= 2
        levels.append(scale0)

    contourlevels = np.log2(levels)
    fig, axes = plt.subplots(nrows=4, ncols=1) #,figsize=(16,6))
    
    # fig, ax = plt.subplots(figsize=(8, 6))
    im = axes[0].contourf(time, np.log2(period), np.log2(power),
                     contourlevels, extend='both', cmap=cmap)
    # ax.set_title(title, fontsize=20)
    # ax.set_ylabel(ylabel, fontsize=18)
    axes[0].set_ylabel(obs_name, fontsize=8)
    f=1
    for i in [0, 9, 6]: #np.arange(len(model_datasets)-1):
      
        dsi = xr.DataArray(model_datasets[i].values,
        coords={'time': obs_dataset.times,
        'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
        dims=["time", "lat", "lon"])
                    
        dsi = dsi.mean(dim=("lat", "lon"))
        
        signal = dsi.values.squeeze() 
        signal = signal - np.mean(signal)

        [coefficients, frequencies] = pywt.cwt(signal, scales, waveletname, dt)
        power = (abs(coefficients)) ** 2
        #print(power)
        period = 1. / frequencies

        
        #fig, ax = plt.subplots(figsize=(8, 6))
        im = axes[f].contourf(time, np.log2(period), np.log2(power),
                         contourlevels, extend='both', cmap=cmap)
        #ax.set_title(title, fontsize=20)
        axes[f].set_ylabel(model_names[i], fontsize=8)
        #if i<11: axes[f].set_yticks([]) 
        
        #axes[i].set_xticks([])
        #axes[10].set_xticks([])
        #axes[f].set_xlabel(model_names[i], fontsize=6)
        yticks = 2**np.arange(np.ceil(np.log2(period.min())),
                              np.ceil(np.log2(period.max())))
        axes[f].set_yticks(np.log2(yticks))
        #axes[0].set_yticklabels(yticks, fontsize=8)
        axes[f].set_yticklabels(['1/8','1/4','1/2','1','2','4','8','16'],fontsize=8)
        axes[0].set_yticks(np.log2(yticks))
        axes[0].set_yticklabels(['1/8','1/4','1/2','1','2','4','8','16'],fontsize=8)
        #axes[0].invert_yaxis()
        #ylim = axes[0].get_ylim()
        #axes[0].set_ylim(ylim[0], -1)
        
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        cbar_ax.tick_params(labelsize=8)
        fig.colorbar(im, cax=cbar_ax, ticks=[-5,-4,-3,-2,-1,0,1,2,3,4])  #, orientation="vertical")
        f=f+1
    axes[0].set_xticks([])
    axes[1].set_xticks([])
    axes[2].set_xticks([])
    plt.show()
    plt.savefig(workdir+reg+'wavelet_3pilihan', dpi=300, bbox_inches='tight')
    
def corr_enso3(obs_dataset, obs_name, model_datasets, model_names, workdir):
    nino_obs=[0.09,1.55,-0.47,-0.68,-0.33,0.93,0.63,-1.23,0.02,0.38,1.11,0.23,0.21,
          0.57,-0.65,-0.08,1.63,-1.22,-1.18,-0.49,0.00,0.64,0.28,0.53]
    #temporal
    #nino34_model => obs 
    #bulanan ? 3bulanan?
    #nino34r5 dan obsr5?
    #resolusi mod-obs sama? res beda hasil sama?
    
    import xarray as xr
    fig = plt.figure(figsize=(12, 6))
    
    filepath=[
    'D:/data1/tos_Omon_CNRM-CM5_historical_r1i1p1_198101-200512.nc',
    'D:/data1/tos_Omon_HadGEM2-ES_esmHistorical_r1i1p1_1981-2005_nino34_ok.nc',
    'D:/data1/tos_Omon_NorESM1-M_historical_r1i1p1_198101-200512.nc',
    'D:/data1/4tos_Omon_GFDL-ESM2M_historical_r1i1p1_198101-200512.nc'
    ]
    
    #tes for data 3=> 2
    ds = xr.open_dataset(filepath[2])
    #slice ini nans
    #tos_nino34 = ds.sel(i=slice(-5, 5), j=slice(190, 240))
    tos_nino34 = ds.where(
                (ds.lat < 5) & (ds.lat > -5) & (ds.lon > 190) & 
                (ds.lon < 240), drop=True)
    gb = tos_nino34.tos.groupby('time.month')
    tos_nino34_anom = gb - gb.mean(dim='time')
    #print(tos_nino34_anom)
    index_nino34 = tos_nino34_anom.mean(dim=['lat', 'lon'])
    #index_nino34 = tos_nino34_anom.mean(dim=['i', 'j'])
    # print('index_nino34=',index_nino34)
    
    # #smooth the anomalies with a 5-month running mean:
    #index_nino34_rolling_mean = index_nino34.rolling(time=5, center=True).mean()
    ir5 = index_nino34.rolling(time=5, center=True).mean()
    # index_nino34.plot(size=8)
    # index_nino34_rolling_mean.plot()
    # plt.legend(['anomaly', '5-month running mean anomaly'])
    # plt.title('SST anomaly over the Niño 3.4 region');
    # figname='nino34'
    # plt.savefig(workdir+figname, dpi=300, bbox_inches='tight')
    #nino34yr=index_nino34.groupby('time.year').mean()
   
    ds = xr.DataArray(obs_dataset.values,
    coords={'time': obs_dataset.times,
    'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
    dims=["time", "lat", "lon"])
                
    # #ds = ds.groupby('time.year').sum() 
    # mds = ds.groupby('time.month')
    # ds_anom = mds - mds.mean(dim='time')
    # #print(tos_nino34_anom)
    # index_nino34 = ds_anom.mean(dim=['lat', 'lon'])
    
    lat_min = ds.lat.min()
    lat_max = ds.lat.max()
    lon_min = ds.lon.min()
    lon_max = ds.lon.max()
   
    
    fig, ax = plt.subplots(nrows=2, ncols=4,figsize=(16,6))
    m = Basemap(ax=ax[0,0], projection ='cyl', llcrnrlat = lat_min, urcrnrlat = lat_max,
            llcrnrlon = lon_min, urcrnrlon = lon_max, resolution = 'l', fix_aspect=True)
    m.drawcoastlines(linewidth=1)
    m.drawcountries(linewidth=1)
    m.drawstates(linewidth=0.5, color='w')
    
    x,y = np.meshgrid(ds.lon, ds.lat)
    
    from scipy.stats import pearsonr
    
    map1=ma.zeros((len(ds.lat),len(ds.lon)))

    # for i in np.arange(len(ds.lat)):
        # #print (i)
        # for j in np.arange(len(ds.lon)):
            # dd=ds[:,i,j]
            # #nan problem 
            # #R=pearsonr(nino_obs, dd)[0]*-1
            # #x1=index_nino34.values
            # x1=ir5.values
            
            # y1=dd.values
            # bad = ~np.logical_or(np.isnan(x1), np.isnan(y1))
            # x1=np.compress(bad, x1) 
            # y1=np.compress(bad, y1)
            # #print(x,y)
            # #print(x.shape,y.shape)
            # if x1.shape==(0,) or y1.shape==(0,):
                # R='nan'
            # else:
                # R=pearsonr(x1, y1)[0]*-1
            # #R=pearsonr(nino34yr.values, dd.values)[0]*-1
            
            # # ini ok tapi lambat
            # #R=xr.corr(index_nino34,dd)*-1
            
            # map1[i,j]=R
    # print('map1=',map1)
    # print('map1.shp=',map1.shape)
    
    # max = ax[0,0].contourf(x,y,map1)
    # ax[0,0].set_title('GPCP')
    # ax[0,0].set_yticks([-10,0,10,20])
    # #ax[0,0].set_xticks([90,100,110,120,130,140])
    
    #fig.savefig(workdir+'nino34_5',dpi=300,bbox_inches='tight')
    #exit()
    # ini ?? khusus zonal jika ingin obs=2 dan MMEW not included
    #model_datasets=np.delete(model_datasets,[1, -1])
    #model_names=np.delete(model_names,[1, -1])
    #model_datasets=np.delete(model_datasets,[-1])
    #model_names=np.delete(model_names,[-1])
    
    for i in np.arange(1):
        print (i)
        i=1
        print(model_names[1])
        dsi = xr.DataArray(model_datasets[i].values,
        coords={'time': obs_dataset.times,
        'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
        dims=["time", "lat", "lon"])  
        
        #dsi = dsi.groupby('time.year').sum() 
        
        #ds = ds.groupby('time.year').sum() 
        
        
        if i<3:
            m = Basemap(ax=ax[0,1+i], projection ='cyl', llcrnrlat = lat_min, urcrnrlat = lat_max,
                llcrnrlon = lon_min, urcrnrlon = lon_max, resolution = 'l', fix_aspect=True)
            m.drawcoastlines(linewidth=1)
            m.drawcountries(linewidth=1)
            m.drawstates(linewidth=0.5, color='w')
        
            x,y = np.meshgrid(ds.lon, ds.lat)
        
            #from scipy.stats import pearsonr
        
            map1=ma.zeros((len(ds.lat),len(ds.lon)))
            x0=ir5.values
            for ii in np.arange(len(ds.lat)):
                print(ii)
                for jj in np.arange(len(ds.lon)):
                    d=dsi[:,ii,jj]
                    md = d.groupby('time.month')
                    d_anom = md - md.mean(dim='time')
                    pr5 = d_anom.rolling(time=5, center=True).mean()
                    y0=pr5.values
                    #print(x0.shape)
                    #print(y0.shape)
                    #print(x0)
                    #print(y0)
                    bad = ~np.logical_or(np.isnan(x0), np.isnan(y0))
                    x1=np.compress(bad, x0) 
                    y1=np.compress(bad, y0)
                    #print(x,y)
                    #print(x.shape,y.shape)
                    if x1.shape==(0,) or y1.shape==(0,):
                        R='nan'
                    else:
                        R=pearsonr(x1, y1)[0]*-1
                        #R=pearsonr(pr5, ir5)[0]*-1
                    map1[ii,jj]=R
            #print('map1=',map1)
            max = ax[0,1+i].contourf(x,y,map1)
            ax[0,1+i].set_title(model_names[i])
            #ax[0,0].set_yticks([-10,0,10,20])
            #ax[0,1+i].set_xticks([90,100,110,120,130,140])
            
        else:
            m = Basemap(ax=ax[1,i-3], projection ='cyl', llcrnrlat = lat_min, urcrnrlat = lat_max,
                llcrnrlon = lon_min, urcrnrlon = lon_max, resolution = 'l', fix_aspect=True)
            m.drawcoastlines(linewidth=1)
            m.drawcountries(linewidth=1)
            m.drawstates(linewidth=0.5, color='w')
        
            x,y = np.meshgrid(ds.lon, ds.lat)
        
            from scipy.stats import pearsonr
        
            map1=ma.zeros((len(ds.lat),len(ds.lon)))

            for ii in np.arange(len(ds.lat)):
                for jj in np.arange(len(ds.lon)):
                    dd=dsi[:,ii,jj]
                    #R=pearsonr(index_nino34, dd)[0]*-1
                    R=xr.corr(index_nino34,dd)*-1
                    map1[ii,jj]=R
        
            max = ax[1,i-3].contourf(x,y,map1)
            ax[1,i-3].set_title(model_names[i])
            ax[1,0].set_yticks([-10,0,10,20])
            ax[1,i-3].set_xticks([90,100,110,120,130,140])
    
    
    plt.subplots_adjust(hspace=.2,wspace=.01)
    cax = fig.add_axes([0.91, 0.5, 0.015, 0.4])
    plt.colorbar(max, cax = cax) 
    
    file_name='Corr_nino34_pr5'
    fig.savefig(workdir+file_name,dpi=300,bbox_inches='tight')
    
def corr_enso_tos(obs_dataset, obs_name, model_datasets, model_names, workdir):
   
    #temporal
    #nino34_model => obs 
    #bulanan ? 3bulanan?
    #nino34r5 dan obsr5?
    #resolusi mod-obs sama? res beda hasil sama?
    
    import xarray as xr
    import pandas as pd
    from scipy.stats import pearsonr
    
    fig, ax = plt.subplots(nrows=3, ncols=4 ,figsize=(8,6))
    
    d= pd.read_excel('D:/Disertasi3/enso_mon_1981-2005.xlsx')
    
    nino_obs = xr.DataArray( d['Value'], coords={'time': obs_dataset.times})
    gb = nino_obs.groupby('time.month')
    nino_obs = gb - gb.mean(dim='time')
    ir5_obs = nino_obs.rolling(time=5, center=True).mean()
    
    filepath=[
    'D:/data1/tos_Omon_CNRM-CM5_historical_r1i1p1_198101-200512.nc',
    'D:/data1/tos_Omon_HadGEM2-ES_esmHistorical_r1i1p1_1981-2005_nino34_ok.nc',
    'D:/data1/tos_Omon_NorESM1-M_historical_r1i1p1_198101-200512.nc',
    'D:/data1/4tos_Omon_GFDL-ESM2M_historical_r1i1p1_198101-200512.nc'
    ]
       
    ds = xr.DataArray(obs_dataset.values,
    coords={'time': obs_dataset.times,
    'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
    dims=["time", "lat", "lon"])
    
    lat_min = ds.lat.min()
    lat_max = ds.lat.max()
    lon_min = ds.lon.min()
    lon_max = ds.lon.max()
   
    x,y = np.meshgrid(ds.lon, ds.lat)
    
    tes=0
    
    map0=ma.zeros((len(ds.lat),len(ds.lon)))
    x0=ir5_obs.values
    for ii in np.arange(len(ds.lat)-tes):
        print(ii)
        for jj in np.arange(len(ds.lon)-tes):
            d=ds[:,ii,jj]
            md = d.groupby('time.month')
            d_anom = md - md.mean(dim='time')
            pr5 = d_anom.rolling(time=5, center=True).mean()
            y0=pr5.values
            
            bad = ~np.logical_or(np.isnan(x0), np.isnan(y0))
            x1=np.compress(bad, x0) 
            y1=np.compress(bad, y0)
            #print(x,y)
            #print(x.shape,y.shape)
            if x1.shape==(0,) or y1.shape==(0,):
                R='nan'
            else:
                R=pearsonr(x1, y1)[0]*-1
                #R=pearsonr(pr5, ir5)[0]*-1
            map0[ii,jj]=R
    m0=map0.flatten()
    
    m = Basemap(ax=ax[0,0], projection ='cyl', llcrnrlat = lat_min, urcrnrlat = lat_max,
                llcrnrlon = lon_min, urcrnrlon = lon_max, resolution = 'l', fix_aspect=False)
    m.drawcoastlines(linewidth=1)
    m.drawcountries(linewidth=1)
    m.drawstates(linewidth=0.5, color='w')
    
    max = ax[0,0].contourf(x,y,map0)
    #ax[0,nn].set_title(model_names[i])
    ax[0,0].set_title(obs_name, pad=3,fontsize=10)
    for i in [1,2,3]: ax[0,i].axis('off')
    
    map1=ma.zeros((len(ds.lat),len(ds.lon)))
    pilih_nino   =[0,1,2,2,2,3,4] 
    pilih_model=[0,1,3,4,5,6,8,9] 
    
    print(len(ds.lat),len(ds.lon))
    nn=0
    r0=1
    #for n in pilih_nino:
    for i in pilih_model:
        
        print(model_names[i])
        
        if i<10: 
            ir5=ir5_obs
            print('')
        else: 
            print(pilih_nino[nn-1],i)
            #tes for data 3=> 2
            dsx = xr.open_dataset(filepath[pilih_nino[nn-1]]) 
            #nn-1 agar saat nn=1 hasilnya nino file tos ke 0
            
            try:
                tos_nino34 = dsx.where(
                        (dsx.lat < 5) & (dsx.lat > -5) & 
                        (dsx.lon > 190) & (dsx.lon < 240), drop=True)
                    
            except:
             
                tos_nino34 = dsx.where(
                        (dsx.rlat < 5) & (dsx.rlat > -5) & 
                        (dsx.rlon > 190) & (dsx.rlon < 240), drop=True)
                        
            gb = tos_nino34.tos.groupby('time.month')
            tos_nino34_anom = gb - gb.mean(dim='time')
            #print(tos_nino34_anom)
            try: index_nino34 = tos_nino34_anom.mean(dim=['rlat', 'rlon'])
            except:
                try: index_nino34 = tos_nino34_anom.mean(dim=['lat', 'lon'])
                except: index_nino34 = tos_nino34_anom.mean(dim=['i', 'j'])
       
            ir5 = index_nino34.rolling(time=5, center=True).mean()
        
        dsi = xr.DataArray(model_datasets[i].values,
        coords={'time': obs_dataset.times,
        'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
        dims=["time", "lat", "lon"]) 
        
        if i<5:
            m = Basemap(ax=ax[1,nn], projection ='cyl', llcrnrlat = lat_min, urcrnrlat = lat_max,
                llcrnrlon = lon_min, urcrnrlon = lon_max, resolution = 'l', fix_aspect=False)
            m.drawcoastlines(linewidth=1)
            m.drawcountries(linewidth=1)
            m.drawstates(linewidth=0.5, color='w')
                
            map1=ma.zeros((len(ds.lat),len(ds.lon)))
            x0=ir5.values
            for ii in np.arange(len(ds.lat)-tes):
                print(ii)
                for jj in np.arange(len(ds.lon)-tes):
                    d=dsi[:,ii,jj]
                    md = d.groupby('time.month')
                    d_anom = md - md.mean(dim='time')
                    pr5 = d_anom.rolling(time=5, center=True).mean()
                    if i in [4,5,6]: 
                        pr5=np.delete(pr5,-1)
                        y0=pr5.values 
                        #print(len(pr5))
                        #print(y0)
                    else: 
                        y0=pr5.values
                        #print(y0)
                    #print(x0.shape)
                    #print(y0.shape)
                    #print(x0)
                    #print(y0)
                    bad = ~np.logical_or(np.isnan(x0), np.isnan(y0))
                    x1=np.compress(bad, x0) 
                    y1=np.compress(bad, y0)
                    #print(x,y)
                    #print(x.shape,y.shape)
                    if x1.shape==(0,) or y1.shape==(0,):
                        R='nan'
                    else:
                        R=pearsonr(x1, y1)[0]*-1
                        #R=pearsonr(pr5, ir5)[0]*-1
                    map1[ii,jj]=R
                    if i==0: m02=map1.flatten()
                    
            m1=map1.flatten()
            bad = ~np.logical_or(np.isnan(m0), np.isnan(m1))
            x1=np.compress(bad, m0) 
            y1=np.compress(bad, m1)
            
            sd1=x1.std() #(skipna=None)
            #print(sd1)
            sd2=y1.std()
            s=sd2/sd1
            
            c,pp=pearsonr(x1.flatten() , y1.flatten())        
            
            #T= (4*(1+c)**4)/(((s+1/s)**2)*((1+r0)**4))
            # Tanpa pakai **4 T naik dikit
            tt= (4*(1+c))/(((s+1/s)**2)*((1+r0)))
            #T.append(round(tt,2))#taylor score
            
            #m02
            bad = ~np.logical_or(np.isnan(m02), np.isnan(m1))
            x1=np.compress(bad, m02) 
            y1=np.compress(bad, m1)
            
            sd1=x1.std() #(skipna=None)
            #print(sd1)
            sd2=y1.std()
            s=sd2/sd1
            
            c,pp=pearsonr(x1.flatten() , y1.flatten())        
            
            #T= (4*(1+c)**4)/(((s+1/s)**2)*((1+r0)**4))
            # Tanpa pakai **4 T naik dikit
            tt2= (4*(1+c))/(((s+1/s)**2)*((1+r0)))
                    
                    
            #print('map1=',map1)
            max = ax[1,nn].contourf(x,y,map1)
            #ax[0,nn].set_title(model_names[i])
            ax[1,nn].set_title(model_names[i]+'('+'%.2f'%tt+')'+'('+'%.2f'%tt2+')', pad=5,fontsize=10)
            #ax[0,0].set_yticks([-10,0,10,20])
            #ax[0,1+i].set_xticks([90,100,110,120,130,140])
            
            
        else:
            m = Basemap(ax=ax[2,nn-4], projection ='cyl', llcrnrlat = lat_min, urcrnrlat = lat_max,
                llcrnrlon = lon_min, urcrnrlon = lon_max, resolution = 'l', fix_aspect=False)
            m.drawcoastlines(linewidth=1)
            m.drawcountries(linewidth=1)
            m.drawstates(linewidth=0.5, color='w')
                
            map1=ma.zeros((len(ds.lat),len(ds.lon)))
            x0=ir5.values
            for ii in np.arange(len(ds.lat)-tes):
                print(ii)
                for jj in np.arange(len(ds.lon)-tes):
                    d=dsi[:,ii,jj]
                    md = d.groupby('time.month')
                    d_anom = md - md.mean(dim='time')
                    pr5 = d_anom.rolling(time=5, center=True).mean()
                    if i in [4,5,6]: 
                        pr5=np.delete(pr5,-1)
                        y0=pr5.values 
                        #print(len(pr5))
                        #print(y0)
                    else: 
                        y0=pr5.values
                    #print(x0.shape)
                    #print(y0.shape)
                    #print(x0)
                    #print(y0)
                    bad = ~np.logical_or(np.isnan(x0), np.isnan(y0))
                    x1=np.compress(bad, x0) 
                    y1=np.compress(bad, y0)
                    #print(x,y)
                    #print(x.shape,y.shape)
                    if x1.shape==(0,) or y1.shape==(0,):
                        R='nan'
                    else:
                        R=pearsonr(x1, y1)[0]*-1
                        #R=pearsonr(pr5, ir5)[0]*-1
                    map1[ii,jj]=R
            
            m1=map1.flatten()
            bad = ~np.logical_or(np.isnan(m0), np.isnan(m1))
            x1=np.compress(bad, m0) 
            y1=np.compress(bad, m1)
            
            sd1=x1.std() #(skipna=None)
            #print(sd1)
            sd2=y1.std()
            s=sd2/sd1
            
            c,pp=pearsonr(x1.flatten() , y1.flatten())        
            
            #T= (4*(1+c)**4)/(((s+1/s)**2)*((1+r0)**4))
            # Tanpa pakai **4 T naik dikit
            tt= (4*(1+c))/(((s+1/s)**2)*((1+r0)))
            #T.append(round(tt,2))#taylor score
            #print('map1=',map1)
            
            #m02
            bad = ~np.logical_or(np.isnan(m02), np.isnan(m1))
            x1=np.compress(bad, m02) 
            y1=np.compress(bad, m1)
            
            sd1=x1.std() #(skipna=None)
            #print(sd1)
            sd2=y1.std()
            s=sd2/sd1
            
            c,pp=pearsonr(x1.flatten() , y1.flatten())        
            
            #T= (4*(1+c)**4)/(((s+1/s)**2)*((1+r0)**4))
            # Tanpa pakai **4 T naik dikit
            tt2= (4*(1+c))/(((s+1/s)**2)*((1+r0)))
           
            max = ax[2,nn-4].contourf(x,y,map1)
            #ax[1,nn-4].set_title(model_names[i])
            ax[2,nn-4].set_title(model_names[i]+'('+'%.2f'%tt+')'+'('+'%.2f'%tt2+')', pad=5,fontsize=10)
            #ax[0,0].set_yticks([-10,0,10,20])
            #ax[0,1+i].set_xticks([90,100,110,120,130,140])
            ax[2,nn-4].set_xticks([100,120,140])
            ax[2,nn-4].xaxis.set_tick_params(labelsize=7)
        nn=nn+1    
    
    ax[0,0].set_yticks([-10,0,10,20])
    ax[1,0].set_yticks([-10,0,10,20])
    ax[2,0].set_yticks([-10,0,10,20])
    ax[0,0].yaxis.set_tick_params(labelsize=7)
    ax[1,0].yaxis.set_tick_params(labelsize=7)
    ax[2,0].yaxis.set_tick_params(labelsize=7)
    
    
    plt.subplots_adjust(hspace=.25,wspace=.12)
    #cax = fig.add_axes([0.91, 0.5, 0.015, 0.4])
    cax = fig.add_axes([0.4, 0.7, 0.4, 0.04]) #horisontal
    #plt.colorbar(max, cax = cax) 
    cbar=plt.colorbar(max, cax = cax, extend='both', orientation='horizontal') 
    cbar.ax.tick_params(labelsize=7)
    
    file_name='Corr_nino34_pr5_tos3_'+reg
    fig.savefig(workdir+file_name,dpi=300,bbox_inches='tight')
    plt.show()

def corr_enso_tos_season(obs_dataset, obs_name, model_datasets, model_names, workdir):
   
   
    import xarray as xr
    import pandas as pd
    from scipy.stats import pearsonr
    
    musim='MAM'
    print('corr_enso_tos_season', musim)
    
    fig, ax = plt.subplots(nrows=3, ncols=4 ,figsize=(8,6))
    
    d= pd.read_excel('D:/Disertasi3/enso_mon_1981-2005.xlsx')
    
    ds = xr.DataArray( d['Value'], coords={'time': obs_dataset.times})
    nino_obs=ds.groupby('time.season')
    x0=nino_obs[musim].values
    
    #gb = nino_obs.groupby('time.month')
    #nino_obs = gb - gb.mean(dim='time')
    #ir5_obs = nino_obs.rolling(time=5, center=True).mean()
    
    filepath=[
    'D:/data1/tos_Omon_CNRM-CM5_historical_r1i1p1_198101-200512.nc',
    'D:/data1/tos_Omon_IPSL-CM5A-LR_historicalNat_r1i1p1_1981-2005_nino34_ok.nc',
    'D:/data1/tos_Omon_HadGEM2-ES_esmHistorical_r1i1p1_1981-2005_nino34_ok.nc',
    'D:/data1/tos_Omon_NorESM1-M_historical_r1i1p1_198101-200512.nc',
    'D:/data1/4tos_Omon_GFDL-ESM2M_historical_r1i1p1_198101-200512.nc'
    ]
       
    ds = xr.DataArray(obs_dataset.values,
    coords={'time': obs_dataset.times,
    'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
    dims=["time", "lat", "lon"])
    
    lat_min = ds.lat.min()
    lat_max = ds.lat.max()
    lon_min = ds.lon.min()
    lon_max = ds.lon.max()
   
    x,y = np.meshgrid(ds.lon, ds.lat)
    
    tes=0
    
    map0=ma.zeros((len(ds.lat),len(ds.lon)))
    
    for ii in np.arange(len(ds.lat)-tes):
        print(ii)
        for jj in np.arange(len(ds.lon)-tes):
            d=ds[:,ii,jj]
            md = d.groupby('time.month')
            d_anom = md - md.mean(dim='time')
            
            #pr5 = d_anom.rolling(time=5, center=True).mean()
            #y0=pr5.values
            pr=d_anom .groupby('time.season')
            y0=pr[musim].values
            
            bad = ~np.logical_or(np.isnan(x0), np.isnan(y0))
            x1=np.compress(bad, x0) 
            y1=np.compress(bad, y0)
            #print(x,y)
            #print(x.shape,y.shape)
            if x1.shape==(0,) or y1.shape==(0,):
                R='nan'
            else:
                R=pearsonr(x1, y1)[0]*-1
                #R=pearsonr(pr5, ir5)[0]*-1
            map0[ii,jj]=R
    m0=map0.flatten()
    
    m = Basemap(ax=ax[0,0], projection ='cyl', llcrnrlat = lat_min, urcrnrlat = lat_max,
                llcrnrlon = lon_min, urcrnrlon = lon_max, resolution = 'l', fix_aspect=False)
    m.drawcoastlines(linewidth=1)
    m.drawcountries(linewidth=1)
    m.drawstates(linewidth=0.5, color='w')
    
    max = ax[0,0].contourf(x,y,map0)
    #ax[0,nn].set_title(model_names[i])
    ax[0,0].set_title(obs_name, pad=3,fontsize=10)
    for i in [1,2,3]: ax[0,i].axis('off')
    #khusus NorESM1
    for i in [0,1]: ax[1,i].axis('off')
    for i in [1]: ax[2,i].axis('off')
    
    map1=ma.zeros((len(ds.lat),len(ds.lon)))
    pilih_nino   =[0,1,2,2,2,3,4] 
    pilih_model=[0,1,3,4,5,6,8,9] 
    
    print(len(ds.lat),len(ds.lon))
    nn=0
    r0=1
    #for n in pilih_nino:
    for i in [8]: #pilih_model:
        
        print(model_names[i])
        
        if i==0: 
            ir5=x0
            print('')
        else: 
            print(pilih_nino[nn-1],i)
            #tes for data 3=> 2
            dsx = xr.open_dataset(filepath[pilih_nino[nn-1]]) 
            #nn-1 agar saat nn=1 hasilnya nino file tos ke 0
            
            try:
                tos_nino34 = dsx.where(
                        (dsx.lat < 5) & (dsx.lat > -5) & 
                        (dsx.lon > 190) & (dsx.lon < 240), drop=True)
                    
            except:
             
                tos_nino34 = dsx.where(
                        (dsx.rlat < 5) & (dsx.rlat > -5) & 
                        (dsx.rlon > 190) & (dsx.rlon < 240), drop=True)
                        
            gb = tos_nino34.tos.groupby('time.month')
            tos_nino34_anom = gb - gb.mean(dim='time')
            #print(tos_nino34_anom)
            try: index_nino34 = tos_nino34_anom.mean(dim=['rlat', 'rlon'])
            except:
                try: index_nino34 = tos_nino34_anom.mean(dim=['lat', 'lon'])
                except: index_nino34 = tos_nino34_anom.mean(dim=['i', 'j'])
       
            #ir5 = index_nino34.rolling(time=5, center=True).mean()
            ir5=index_nino34.groupby('time.season')
            ir5=ir5[musim].values
            
        dsi = xr.DataArray(model_datasets[i].values,
        coords={'time': obs_dataset.times,
        'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
        dims=["time", "lat", "lon"]) 
        
        if i<5:
            m = Basemap(ax=ax[1,nn], projection ='cyl', llcrnrlat = lat_min, urcrnrlat = lat_max,
                llcrnrlon = lon_min, urcrnrlon = lon_max, resolution = 'l', fix_aspect=False)
            m.drawcoastlines(linewidth=1)
            m.drawcountries(linewidth=1)
            m.drawstates(linewidth=0.5, color='w')
                
            map1=ma.zeros((len(ds.lat),len(ds.lon)))
            
            for ii in np.arange(len(ds.lat)-tes):
                print(ii)
                for jj in np.arange(len(ds.lon)-tes):
                    d=dsi[:,ii,jj]
                    md = d.groupby('time.month')
                    d_anom = md - md.mean(dim='time')
                    #pr5 = d_anom.rolling(time=5, center=True).mean()
                    pr=d_anom .groupby('time.season')
                    y0=pr[musim].values
                    
                    bad = ~np.logical_or(np.isnan(x0), np.isnan(y0))
                    x1=np.compress(bad, x0) 
                    y1=np.compress(bad, y0)
                    #print(x,y)
                    #print(x.shape,y.shape)
                    if x1.shape==(0,) or y1.shape==(0,):
                        R='nan'
                    else:
                        R=pearsonr(x1, y1)[0]*-1
                        #R=pearsonr(pr5, ir5)[0]*-1
                    map1[ii,jj]=R
                    if i==0: m02=map1.flatten()
                    
            m1=map1.flatten()
            bad = ~np.logical_or(np.isnan(m0), np.isnan(m1))
            x1=np.compress(bad, m0) 
            y1=np.compress(bad, m1)
            
            sd1=x1.std() #(skipna=None)
            #print(sd1)
            sd2=y1.std()
            s=sd2/sd1
            
            c,pp=pearsonr(x1.flatten() , y1.flatten())        
            
            #T= (4*(1+c)**4)/(((s+1/s)**2)*((1+r0)**4))
            # Tanpa pakai **4 T naik dikit
            tt= (4*(1+c))/(((s+1/s)**2)*((1+r0)))
            #T.append(round(tt,2))#taylor score
            
            #m02
            m02=m0
            bad = ~np.logical_or(np.isnan(m02), np.isnan(m1))
            x1=np.compress(bad, m02) 
            y1=np.compress(bad, m1)
            
            sd1=x1.std() #(skipna=None)
            #print(sd1)
            sd2=y1.std()
            s=sd2/sd1
            
            c,pp=pearsonr(x1.flatten() , y1.flatten())        
            
            #T= (4*(1+c)**4)/(((s+1/s)**2)*((1+r0)**4))
            # Tanpa pakai **4 T naik dikit
            tt2= (4*(1+c))/(((s+1/s)**2)*((1+r0)))
                    
                    
            #print('map1=',map1)
            max = ax[1,nn].contourf(x,y,map1)
            #ax[0,nn].set_title(model_names[i])
            ax[1,nn].set_title(model_names[i]+'('+'%.2f'%tt+')'+'('+'%.2f'%tt2+')', pad=5,fontsize=10)
            #ax[0,0].set_yticks([-10,0,10,20])
            #ax[0,1+i].set_xticks([90,100,110,120,130,140])
            
            
        else:
            m = Basemap(ax=ax[2,nn-4], projection ='cyl', llcrnrlat = lat_min, urcrnrlat = lat_max,
                llcrnrlon = lon_min, urcrnrlon = lon_max, resolution = 'l', fix_aspect=False)
            m.drawcoastlines(linewidth=1)
            m.drawcountries(linewidth=1)
            m.drawstates(linewidth=0.5, color='w')
                
            map1=ma.zeros((len(ds.lat),len(ds.lon)))
            
            for ii in np.arange(len(ds.lat)-tes):
                print(ii)
                for jj in np.arange(len(ds.lon)-tes):
                    d=dsi[:,ii,jj]
                    md = d.groupby('time.month')
                    d_anom = md - md.mean(dim='time')
                    #pr5 = d_anom.rolling(time=5, center=True).mean()
                    pr=d_anom .groupby('time.season')
                    y0=pr[musim].values
                    
                    bad = ~np.logical_or(np.isnan(x0), np.isnan(y0))
                    x1=np.compress(bad, x0) 
                    y1=np.compress(bad, y0)
                    #print(x,y)
                    #print(x.shape,y.shape)
                    if x1.shape==(0,) or y1.shape==(0,):
                        R='nan'
                    else:
                        R=pearsonr(x1, y1)[0]*-1
                        #R=pearsonr(pr5, ir5)[0]*-1
                    map1[ii,jj]=R
            
            m1=map1.flatten()
            bad = ~np.logical_or(np.isnan(m0), np.isnan(m1))
            x1=np.compress(bad, m0) 
            y1=np.compress(bad, m1)
            
            sd1=x1.std() #(skipna=None)
            #print(sd1)
            sd2=y1.std()
            s=sd2/sd1
            
            c,pp=pearsonr(x1.flatten() , y1.flatten())        
            
            #T= (4*(1+c)**4)/(((s+1/s)**2)*((1+r0)**4))
            # Tanpa pakai **4 T naik dikit
            tt= (4*(1+c))/(((s+1/s)**2)*((1+r0)))
            #T.append(round(tt,2))#taylor score
            #print('map1=',map1)
            
            #m02
            m02=m0
            bad = ~np.logical_or(np.isnan(m02), np.isnan(m1))
            x1=np.compress(bad, m02) 
            y1=np.compress(bad, m1)
            
            sd1=x1.std() #(skipna=None)
            #print(sd1)
            sd2=y1.std()
            s=sd2/sd1
            
            c,pp=pearsonr(x1.flatten() , y1.flatten())        
            
            #T= (4*(1+c)**4)/(((s+1/s)**2)*((1+r0)**4))
            # Tanpa pakai **4 T naik dikit
            tt2= (4*(1+c))/(((s+1/s)**2)*((1+r0)))
           
            max = ax[2,nn-4].contourf(x,y,map1)
            #ax[1,nn-4].set_title(model_names[i])
            ax[2,nn-4].set_title(model_names[i]+'('+'%.2f'%tt+')'+'('+'%.2f'%tt2+')', pad=5,fontsize=10)
            #ax[0,0].set_yticks([-10,0,10,20])
            #ax[0,1+i].set_xticks([90,100,110,120,130,140])
            ax[2,nn-4].set_xticks([100,120,140])
            ax[2,nn-4].xaxis.set_tick_params(labelsize=7)
        nn=nn+1    
    
    ax[0,0].set_yticks([-10,0,10,20])
    ax[1,0].set_yticks([-10,0,10,20])
    ax[2,0].set_yticks([-10,0,10,20])
    ax[0,0].yaxis.set_tick_params(labelsize=7)
    ax[1,0].yaxis.set_tick_params(labelsize=7)
    ax[2,0].yaxis.set_tick_params(labelsize=7)
    
    
    plt.subplots_adjust(hspace=.25,wspace=.12)
    #cax = fig.add_axes([0.91, 0.5, 0.015, 0.4])
    cax = fig.add_axes([0.4, 0.7, 0.4, 0.04]) #horisontal
    #plt.colorbar(max, cax = cax) 
    cbar=plt.colorbar(max, cax = cax, extend='both', orientation='horizontal') 
    cbar.ax.tick_params(labelsize=7)
    
    file_name='Corr_nino34_pr5_tos_'+musim
    plt.title('File: '+file_name , y=6, x=.5)
    
    print(file_name)
    plt.show()
    fig.savefig(workdir+file_name,dpi=300,bbox_inches='tight')
    

def corr_iod_tos_season(obs_dataset, obs_name, model_datasets, model_names, workdir):
   
   
    import xarray as xr
    import pandas as pd
    from scipy.stats import pearsonr
    
    musim='JJA'
    print('corr_iod_tos_season', musim)
    
    fig, ax = plt.subplots(nrows=3, ncols=4 ,figsize=(8,6))
    
    d= pd.read_excel('D:/Disertasi3/dmi_1981-2005.xlsx')
    
    ds = xr.DataArray( d['Value'], coords={'time': obs_dataset.times})
    nino_obs=ds.groupby('time.season')
    x0=nino_obs[musim].values
    
    #gb = nino_obs.groupby('time.month')
    #nino_obs = gb - gb.mean(dim='time')
    #ir5_obs = nino_obs.rolling(time=5, center=True).mean()
    
    filepath=[
            'D:/data1/tos_Omon_CNRM-CM5_historical_r1i1p1_198101-200512.nc',
            'D:/data1/tos_Omon_IPSL-CM5A-LR_historicalNat_r1i1p1_1981-2005.nc',
            'D:/data1/tos_Omon_HadGEM2-ES_esmHistorical_r1i1p1_1981-2005_11.nc',
            'D:/data1/tos_Omon_NorESM1-M_historical_r1i1p1_198101-200512.nc',
            'D:/data1/4tos_Omon_GFDL-ESM2M_historical_r1i1p1_198101-200512.nc'
            ]
       
    ds = xr.DataArray(obs_dataset.values,
    coords={'time': obs_dataset.times,
    'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
    dims=["time", "lat", "lon"])
    
    lat_min = ds.lat.min()
    lat_max = ds.lat.max()
    lon_min = ds.lon.min()
    lon_max = ds.lon.max()
   
    x,y = np.meshgrid(ds.lon, ds.lat)
    
    tes=0
    
    map0=ma.zeros((len(ds.lat),len(ds.lon)))
    
    for ii in np.arange(len(ds.lat)-tes):
        print(ii)
        for jj in np.arange(len(ds.lon)-tes):
            d=ds[:,ii,jj]
            md = d.groupby('time.month')
            d_anom = md - md.mean(dim='time')
            
            #pr5 = d_anom.rolling(time=5, center=True).mean()
            #y0=pr5.values
            pr=d_anom .groupby('time.season')
            y0=pr[musim].values
            
            bad = ~np.logical_or(np.isnan(x0), np.isnan(y0))
            x1=np.compress(bad, x0) 
            y1=np.compress(bad, y0)
            #print(x,y)
            #print(x.shape,y.shape)
            if x1.shape==(0,) or y1.shape==(0,):
                R='nan'
            else:
                R=pearsonr(x1, y1)[0]*-1
                #R=pearsonr(pr5, ir5)[0]*-1
            map0[ii,jj]=R
    m0=map0.flatten()
    
    m = Basemap(ax=ax[0,0], projection ='cyl', llcrnrlat = lat_min, urcrnrlat = lat_max,
                llcrnrlon = lon_min, urcrnrlon = lon_max, resolution = 'l', fix_aspect=False)
    m.drawcoastlines(linewidth=1)
    m.drawcountries(linewidth=1)
    m.drawstates(linewidth=0.5, color='w')
    
    max = ax[0,0].contourf(x,y,map0)
    #ax[0,nn].set_title(model_names[i])
    ax[0,0].set_title(obs_name, pad=3,fontsize=10)
    for i in [1,2,3]: ax[0,i].axis('off')
    
    map1=ma.zeros((len(ds.lat),len(ds.lon)))
    pilih_nino   =[0,1,2,2,2,3,4] 
    pilih_model=[0,1,3,4,5,6,8,9] 
    
    print(len(ds.lat),len(ds.lon))
    nn=0
    r0=1
    tay=[]
    model=[]
    for i in pilih_model:
        
        print(model_names[i])
        
        if i==0: 
            ir5=x0
            print('')
        else: 
            print(pilih_nino[nn-1],i)
            #tes for data 3=> 2
            dsx = xr.open_dataset(filepath[pilih_nino[nn-1]]) 
            #nn-1 agar saat nn=1 hasilnya nino file tos ke 0
            
            try:
                tos_w = dsx.where(
                        (dsx.lat < 10) & (dsx.lat > -10) & 
                        (dsx.lon > 50) & (dsx.lon < 70), drop=True)
            #GFDL pakai rlat,rlon             
            except:
                try:
                    tos_w = dsx.where(
                        (dsx.rlat < 10) & (dsx.rlat > -10) & 
                        (dsx.rlon > 50) & (dsx.rlon < 70), drop=True)
                except:
                    #ini tidak perlu
                    print('i,j terpakai')
                    tos_w = dsx.where(
                        (dsx.j < 10) & (dsx.j > -10) & 
                        (dsx.i > 50) & (dsx.i < 70), drop=True)   
            try:
                tos_e = dsx.where(
                        (dsx.lat < 0) & (dsx.lat > -10) & 
                        (dsx.lon > 90) & (dsx.lon < 110), drop=True)         
            except:
                try:
                    tos_e = dsx.where(
                        (dsx.rlat < 0) & (dsx.rlat > -10) & 
                        (dsx.rlon > 90) & (dsx.rlon < 110), drop=True) 
                except:
                    print('i,j terpakai')
                    tos_e = dsx.where(
                        (dsx.j < 0) & (dsx.j > -10) & 
                        (dsx.i > 90) & (dsx.i < 110), drop=True) 
            
            tos_nino34=tos_w 
            
            gb = tos_nino34.tos.groupby('time.month')
            tos_nino34_anom = gb - gb.mean(dim='time')
            #print(tos_nino34_anom)
            try: index_nino34 = tos_nino34_anom.mean(dim=['rlat', 'rlon'])
            except:
                try: index_nino34 = tos_nino34_anom.mean(dim=['lat', 'lon'])
                except: index_nino34 = tos_nino34_anom.mean(dim=['i', 'j'])

            ir5_w = index_nino34 #.rolling(time=5, center=True).mean()
            
            tos_nino34=tos_e 
            
            gb = tos_nino34.tos.groupby('time.month')
            tos_nino34_anom = gb - gb.mean(dim='time')
            #print(tos_nino34_anom)
            try: index_nino34 = tos_nino34_anom.mean(dim=['rlat', 'rlon'])
            except:
                try: index_nino34 = tos_nino34_anom.mean(dim=['lat', 'lon'])
                except: index_nino34 = tos_nino34_anom.mean(dim=['i', 'j'])

            ir5_e = index_nino34 #.rolling(time=5, center=True).mean()
            
            ir5=ir5_w -ir5_e
       
            #ir5 = index_nino34.rolling(time=5, center=True).mean()
            ir5=ir5.groupby('time.season')
            ir5=ir5[musim].values
            
        dsi = xr.DataArray(model_datasets[i].values,
        coords={'time': obs_dataset.times,
        'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
        dims=["time", "lat", "lon"]) 
        
        if i<5:
            m = Basemap(ax=ax[1,nn], projection ='cyl', llcrnrlat = lat_min, urcrnrlat = lat_max,
                llcrnrlon = lon_min, urcrnrlon = lon_max, resolution = 'l', fix_aspect=False)
            m.drawcoastlines(linewidth=1)
            m.drawcountries(linewidth=1)
            m.drawstates(linewidth=0.5, color='w')
                
            map1=ma.zeros((len(ds.lat),len(ds.lon)))
            
            for ii in np.arange(len(ds.lat)-tes):
                print(ii)
                for jj in np.arange(len(ds.lon)-tes):
                    d=dsi[:,ii,jj]
                    md = d.groupby('time.month')
                    d_anom = md - md.mean(dim='time')
                    #pr5 = d_anom.rolling(time=5, center=True).mean()
                    pr=d_anom .groupby('time.season')
                    y0=pr[musim].values
                    
                    bad = ~np.logical_or(np.isnan(x0), np.isnan(y0))
                    x1=np.compress(bad, x0) 
                    y1=np.compress(bad, y0)
                    #print(x,y)
                    #print(x.shape,y.shape)
                    if x1.shape==(0,) or y1.shape==(0,):
                        R='nan'
                    else:
                        R=pearsonr(x1, y1)[0]*-1
                        #R=pearsonr(pr5, ir5)[0]*-1
                    map1[ii,jj]=R
                    if i==0: m02=map1.flatten()
                    
            m1=map1.flatten()
            bad = ~np.logical_or(np.isnan(m0), np.isnan(m1))
            x1=np.compress(bad, m0) 
            y1=np.compress(bad, m1)
            
            sd1=x1.std() #(skipna=None)
            #print(sd1)
            sd2=y1.std()
            s=sd2/sd1
            
            c,pp=pearsonr(x1.flatten() , y1.flatten())        
            
            #T= (4*(1+c)**4)/(((s+1/s)**2)*((1+r0)**4))
            # Tanpa pakai **4 T naik dikit
            tt= (4*(1+c))/(((s+1/s)**2)*((1+r0)))
            tay.append(round(tt,2))
            model.append(model_names[i])
            
            #m02
            bad = ~np.logical_or(np.isnan(m02), np.isnan(m1))
            x1=np.compress(bad, m02) 
            y1=np.compress(bad, m1)
            
            sd1=x1.std() #(skipna=None)
            #print(sd1)
            sd2=y1.std()
            s=sd2/sd1
            
            c,pp=pearsonr(x1.flatten() , y1.flatten())        
            
            #T= (4*(1+c)**4)/(((s+1/s)**2)*((1+r0)**4))
            # Tanpa pakai **4 T naik dikit
            tt2= (4*(1+c))/(((s+1/s)**2)*((1+r0)))
                    
                    
            #print('map1=',map1)
            max = ax[1,nn].contourf(x,y,map1)
            #ax[0,nn].set_title(model_names[i])
            ax[1,nn].set_title(model_names[i]+'('+'%.2f'%tt+')'+'('+'%.2f'%tt2+')', pad=5,fontsize=10)
            #ax[0,0].set_yticks([-10,0,10,20])
            #ax[0,1+i].set_xticks([90,100,110,120,130,140])
            
            
        else:
            m = Basemap(ax=ax[2,nn-4], projection ='cyl', llcrnrlat = lat_min, urcrnrlat = lat_max,
                llcrnrlon = lon_min, urcrnrlon = lon_max, resolution = 'l', fix_aspect=False)
            m.drawcoastlines(linewidth=1)
            m.drawcountries(linewidth=1)
            m.drawstates(linewidth=0.5, color='w')
                
            map1=ma.zeros((len(ds.lat),len(ds.lon)))
            
            for ii in np.arange(len(ds.lat)-tes):
                print(ii)
                for jj in np.arange(len(ds.lon)-tes):
                    d=dsi[:,ii,jj]
                    md = d.groupby('time.month')
                    d_anom = md - md.mean(dim='time')
                    #pr5 = d_anom.rolling(time=5, center=True).mean()
                    pr=d_anom .groupby('time.season')
                    y0=pr[musim].values
                    
                    bad = ~np.logical_or(np.isnan(x0), np.isnan(y0))
                    x1=np.compress(bad, x0) 
                    y1=np.compress(bad, y0)
               
                    if x1.shape==(0,) or y1.shape==(0,):
                        R='nan'
                    else:
                        R=pearsonr(x1, y1)[0]*-1
                        #R=pearsonr(pr5, ir5)[0]*-1
                    map1[ii,jj]=R
            
            m1=map1.flatten()
            bad = ~np.logical_or(np.isnan(m0), np.isnan(m1))
            x1=np.compress(bad, m0) 
            y1=np.compress(bad, m1)
            
            sd1=x1.std() 
            #print(sd1)
            sd2=y1.std()
            s=sd2/sd1
            
            c,pp=pearsonr(x1.flatten() , y1.flatten())        
            
            tt= (4*(1+c))/(((s+1/s)**2)*((1+r0)))
            tay.append(round(tt,2))
            model.append(model_names[i])
            
            #m02
            bad = ~np.logical_or(np.isnan(m02), np.isnan(m1))
            x1=np.compress(bad, m02) 
            y1=np.compress(bad, m1)
            
            sd1=x1.std()
           
            sd2=y1.std()
            s=sd2/sd1
            
            c,pp=pearsonr(x1.flatten() , y1.flatten())        
          
            tt2= (4*(1+c))/(((s+1/s)**2)*((1+r0)))
           
            max = ax[2,nn-4].contourf(x,y,map1)
            #ax[1,nn-4].set_title(model_names[i])
            ax[2,nn-4].set_title(model_names[i]+'('+'%.2f'%tt+')'+'('+'%.2f'%tt2+')', pad=5,fontsize=10)
        
            ax[2,nn-4].set_xticks([100,120,140])
            ax[2,nn-4].xaxis.set_tick_params(labelsize=7)
        nn=nn+1    
     
    ax[0,0].set_yticks([-10,0,10,20])
    ax[1,0].set_yticks([-10,0,10,20])
    ax[2,0].set_yticks([-10,0,10,20])
    ax[0,0].yaxis.set_tick_params(labelsize=7)
    ax[1,0].yaxis.set_tick_params(labelsize=7)
    ax[2,0].yaxis.set_tick_params(labelsize=7)
    
    
    plt.subplots_adjust(hspace=.25,wspace=.12)
    #cax = fig.add_axes([0.91, 0.5, 0.015, 0.4])
    cax = fig.add_axes([0.4, 0.7, 0.4, 0.04]) #horisontal
    #plt.colorbar(max, cax = cax) 
    cbar=plt.colorbar(max, cax = cax, extend='both', orientation='horizontal') 
    cbar.ax.tick_params(labelsize=7)
    
    file_name='Corr_iod_pr5_tos_'+musim
    print(file_name)
    
    plt.title('File: '+file_name , y=6, x=.5)
    
    plt.show()
    fig.savefig(workdir+file_name,dpi=300,bbox_inches='tight')
    
    print('Tay=',tay)
    print('Model=',model)
    import pandas as pd
    result = pd.DataFrame(tay,model)
    result.to_excel(workdir+file_name+'.xlsx')
    

def corr_enso_obs(obs_dataset, obs_name, model_datasets, model_names, workdir):
    print('corr_enso_obs')
    #temporal
    #nino34_model => obs 
    #bulanan ? 3bulanan?
    #nino34r5 dan obsr5?
    #resolusi mod-obs sama? res beda hasil sama?
    
    import xarray as xr
    import pandas as pd
    from scipy.stats import pearsonr
    
    fig, ax = plt.subplots(nrows=3, ncols=4 ,figsize=(8,6))
    
    d= pd.read_excel('D:/Disertasi3/enso_mon_1981-2005.xlsx')
    
    nino_obs = xr.DataArray( d['Value'], coords={'time': obs_dataset.times})
    gb = nino_obs.groupby('time.month')
    nino_obs = gb - gb.mean(dim='time')
    ir5_obs = nino_obs.rolling(time=5, center=True).mean()
    
    ds = xr.DataArray(obs_dataset.values,
    coords={'time': obs_dataset.times,
    'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
    dims=["time", "lat", "lon"])
    
    lat_min = ds.lat.min()
    lat_max = ds.lat.max()
    lon_min = ds.lon.min()
    lon_max = ds.lon.max()
   
    x,y = np.meshgrid(ds.lon, ds.lat)
    
    tes=0
    
    map0=ma.zeros((len(ds.lat),len(ds.lon)))
    x0=ir5_obs.values
    for ii in np.arange(len(ds.lat)-tes):
        print(ii)
        for jj in np.arange(len(ds.lon)-tes):
            d=ds[:,ii,jj]
            md = d.groupby('time.month')
            d_anom = md - md.mean(dim='time')
            pr5 = d_anom.rolling(time=5, center=True).mean()
            y0=pr5.values
          
            bad = ~np.logical_or(np.isnan(x0), np.isnan(y0))
            x1=np.compress(bad, x0) 
            y1=np.compress(bad, y0)
          
            if x1.shape==(0,) or y1.shape==(0,):
                R='nan'
            else:
                R=pearsonr(x1, y1)[0]*-1
                #R=pearsonr(pr5, ir5)[0]*-1
            map0[ii,jj]=R
    m0=map0.flatten()
    
    m = Basemap(ax=ax[0,0], projection ='cyl', llcrnrlat = lat_min, urcrnrlat = lat_max,
                llcrnrlon = lon_min, urcrnrlon = lon_max, resolution = 'l', fix_aspect=False)
    m.drawcoastlines(linewidth=1)
    m.drawcountries(linewidth=1)
    m.drawstates(linewidth=0.5, color='w')
    
    max = ax[0,0].contourf(x,y,map0)
    #ax[0,nn].set_title(model_names[i])
    ax[0,0].set_title(obs_name, pad=3,fontsize=10)
    for i in [1,2,3]: ax[0,i].axis('off')
    
    map1=ma.zeros((len(ds.lat),len(ds.lon)))
  
    pilih_model=[0,1,3,4,5,6,8,9] 
    
    nn=0   
    r0=1
   
    for i in pilih_model:
        
        print(model_names[i])
   
        dsi = xr.DataArray(model_datasets[i].values,
        coords={'time': obs_dataset.times,
        'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
        dims=["time", "lat", "lon"]) 
        
        if i<5:
            m = Basemap(ax=ax[1,nn], projection ='cyl', llcrnrlat = lat_min, urcrnrlat = lat_max,
                llcrnrlon = lon_min, urcrnrlon = lon_max, resolution = 'l', fix_aspect=False)
            m.drawcoastlines(linewidth=1)
            m.drawcountries(linewidth=1)
            m.drawstates(linewidth=0.5, color='w')
                
            map1=ma.zeros((len(ds.lat),len(ds.lon)))
           
            for ii in np.arange(len(ds.lat)-tes):
                print(ii)
                for jj in np.arange(len(ds.lon)-tes):
                    d=dsi[:,ii,jj]
                    md = d.groupby('time.month')
                    d_anom = md - md.mean(dim='time')
                    pr5 = d_anom.rolling(time=5, center=True).mean()
                    
                    y0=pr5.values
                    
                    bad = ~np.logical_or(np.isnan(x0), np.isnan(y0))
                    x1=np.compress(bad, x0) 
                    y1=np.compress(bad, y0)
                  
                    if x1.shape==(0,) or y1.shape==(0,):
                        R='nan'
                    else:
                        R=pearsonr(x1, y1)[0]*-1
                    
                    map1[ii,jj]=R
                    if i==0: m02=map1.flatten()
                    
            m1=map1.flatten()
            bad = ~np.logical_or(np.isnan(m0), np.isnan(m1))
            x1=np.compress(bad, m0) 
            y1=np.compress(bad, m1)
            
            sd1=x1.std() 
           
            sd2=y1.std()
            s=sd2/sd1
            
            c,pp=pearsonr(x1.flatten() , y1.flatten())        
         
            tt= (4*(1+c))/(((s+1/s)**2)*((1+r0)))
          
            #m02
            bad = ~np.logical_or(np.isnan(m02), np.isnan(m1))
            x1=np.compress(bad, m02) 
            y1=np.compress(bad, m1)
            
            sd1=x1.std() #(skipna=None)
            #print(sd1)
            sd2=y1.std()
            s=sd2/sd1
            
            c,pp=pearsonr(x1.flatten() , y1.flatten())        
            
            tt2= (4*(1+c))/(((s+1/s)**2)*((1+r0)))
          
            max = ax[1,nn].contourf(x,y,map1)
          
            ax[1,nn].set_title(model_names[i]+'('+'%.2f'%tt+')'+'('+'%.2f'%tt2+')', pad=5,fontsize=10)
           
            
        else:
            m = Basemap(ax=ax[2,nn-4], projection ='cyl', llcrnrlat = lat_min, urcrnrlat = lat_max,
                llcrnrlon = lon_min, urcrnrlon = lon_max, resolution = 'l', fix_aspect=False)
            m.drawcoastlines(linewidth=1)
            m.drawcountries(linewidth=1)
            m.drawstates(linewidth=0.5, color='w')
                
            map1=ma.zeros((len(ds.lat),len(ds.lon)))
            
            for ii in np.arange(len(ds.lat)-tes):
                print(ii)
                for jj in np.arange(len(ds.lon)-tes):
                    d=dsi[:,ii,jj]
                    md = d.groupby('time.month')
                    d_anom = md - md.mean(dim='time')
                    pr5 = d_anom.rolling(time=5, center=True).mean()
                   
                    y0=pr5.values
                    
                    bad = ~np.logical_or(np.isnan(x0), np.isnan(y0))
                    x1=np.compress(bad, x0) 
                    y1=np.compress(bad, y0)
                   
                    if x1.shape==(0,) or y1.shape==(0,):
                        R='nan'
                    else:
                        R=pearsonr(x1, y1)[0]*-1
                        #R=pearsonr(pr5, ir5)[0]*-1
                    map1[ii,jj]=R
            
            m1=map1.flatten()
            bad = ~np.logical_or(np.isnan(m0), np.isnan(m1))
            x1=np.compress(bad, m0) 
            y1=np.compress(bad, m1)
            
            sd1=x1.std() #(skipna=None)
            #print(sd1)
            sd2=y1.std()
            s=sd2/sd1
            
            c,pp=pearsonr(x1.flatten() , y1.flatten())        
            
            tt= (4*(1+c))/(((s+1/s)**2)*((1+r0)))
           
            #m02
            bad = ~np.logical_or(np.isnan(m02), np.isnan(m1))
            x1=np.compress(bad, m02) 
            y1=np.compress(bad, m1)
            
            sd1=x1.std() #(skipna=None)
            #print(sd1)
            sd2=y1.std()
            s=sd2/sd1
            
            c,pp=pearsonr(x1.flatten() , y1.flatten())        
            
            #T= (4*(1+c)**4)/(((s+1/s)**2)*((1+r0)**4))
            # Tanpa pakai **4 T naik dikit
            tt2= (4*(1+c))/(((s+1/s)**2)*((1+r0)))
           
            max = ax[2,nn-4].contourf(x,y,map1)
            #ax[1,nn-4].set_title(model_names[i])
            ax[2,nn-4].set_title(model_names[i]+'('+'%.2f'%tt+')'+'('+'%.2f'%tt2+')', pad=5,fontsize=10)
           
            ax[2,nn-4].set_xticks([100,120,140])
            ax[2,nn-4].xaxis.set_tick_params(labelsize=7)
        nn=nn+1    
    
    ax[0,0].set_yticks([-10,0,10,20])
    ax[1,0].set_yticks([-10,0,10,20])
    ax[2,0].set_yticks([-10,0,10,20])
    ax[0,0].yaxis.set_tick_params(labelsize=7)
    ax[1,0].yaxis.set_tick_params(labelsize=7)
    ax[2,0].yaxis.set_tick_params(labelsize=7)
    
    
    plt.subplots_adjust(hspace=.25,wspace=.12)
    #cax = fig.add_axes([0.91, 0.5, 0.015, 0.4])
    cax = fig.add_axes([0.4, 0.7, 0.4, 0.04]) #horisontal
    #plt.colorbar(max, cax = cax) 
    cbar=plt.colorbar(max, cax = cax, extend='both', orientation='horizontal') 
    cbar.ax.tick_params(labelsize=7)
    
    file_name='Corr_nino34_pr5_obs_'+reg
    print(file_name)
    fig.savefig(workdir+file_name,dpi=300,bbox_inches='tight')
    plt.show()
      
def metrik_enso2(obs_dataset, obs_name, model_datasets, model_names, workdir):
    nino_obs=[0.09,1.55,-0.47,-0.68,-0.33,0.93,0.63,-1.23,0.02,0.38,1.11,0.23,0.21,
          0.57,-0.65,-0.08,1.63,-1.22,-1.18,-0.49,0.00,0.64,0.28,0.53]
    
    #nino_model => tiap model
    #bulanan ? 3bulanan?
    import xarray as xr
    fig = plt.figure(figsize=(12, 6))
    
    filepath=['D:/data1/tos_Omon_CNRM-CM5_historical_r1i1p1_198101-200412.nc',
    'D:/data1/tos_Omon_NorESM1-M_historical_r1i1p1_1976001-200512.nc']
    ds = xr.open_dataset(filepath[0])
    #slice ini nans
    #tos_nino34 = ds.sel(i=slice(-5, 5), j=slice(190, 240))
    tos_nino34 = ds.where(
                (ds.lat < 5) & (ds.lat > -5) & (ds.lon > 190) & 
                (ds.lon < 240), drop=True)
    gb = tos_nino34.tos.groupby('time.month')
    tos_nino34_anom = gb - gb.mean(dim='time')
    #print(tos_nino34_anom)
    index_nino34 = tos_nino34_anom.mean(dim=['i', 'j'])
    # print('index_nino34=',index_nino34)
    
    # #smooth the anomalies with a 5-month running mean:
    # index_nino34_rolling_mean = index_nino34.rolling(time=5, center=True).mean()
    # index_nino34.plot(size=8)
    # index_nino34_rolling_mean.plot()
    # plt.legend(['anomaly', '5-month running mean anomaly'])
    # plt.title('SST anomaly over the Niño 3.4 region');
    # figname='nino34'
    # plt.savefig(workdir+figname, dpi=300, bbox_inches='tight')
    #nino34yr=index_nino34.groupby('time.year').mean()
   
    ds = xr.DataArray(obs_dataset.values,
    coords={'time': obs_dataset.times,
    'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
    dims=["time", "lat", "lon"])
                
    #ds = ds.groupby('time.year').sum() 
    mds = ds.groupby('time.month')
    ds_anom = mds - mds.mean(dim='time')
    #print(tos_nino34_anom)
    index_nino34 = ds_anom.mean(dim=['lat', 'lon'])
    
    lat_min = ds.lat.min()
    lat_max = ds.lat.max()
    lon_min = ds.lon.min()
    lon_max = ds.lon.max()
   
    
    fig, ax = plt.subplots(nrows=2, ncols=4,figsize=(16,6))
    m = Basemap(ax=ax[0,0], projection ='cyl', llcrnrlat = lat_min, urcrnrlat = lat_max,
            llcrnrlon = lon_min, urcrnrlon = lon_max, resolution = 'l', fix_aspect=True)
    m.drawcoastlines(linewidth=1)
    m.drawcountries(linewidth=1)
    m.drawstates(linewidth=0.5, color='w')
    
    x,y = np.meshgrid(ds.lon, ds.lat)
    
    from scipy.stats import pearsonr
    
    map1=ma.zeros((len(ds.lat),len(ds.lon)))

    for i in np.arange(len(ds.lat)):
        #print (i)
        for j in np.arange(len(ds.lon)):
            dd=ds[:,i,j]
            #nan problem 
            #R=pearsonr(nino_obs, dd)[0]*-1
            x1=index_nino34.values
            
            y1=dd.values
            bad = ~np.logical_or(np.isnan(x1), np.isnan(y1))
            x1=np.compress(bad, x1) 
            y1=np.compress(bad, y1)
            #print(x,y)
            #print(x.shape,y.shape)
            if x1.shape==(0,) or y1.shape==(0,):
                R='nan'
            else:
                R=pearsonr(x1, y1)[0]*-1
            #R=pearsonr(nino34yr.values, dd.values)[0]*-1
            
            # ini ok tapi lambat
            #R=xr.corr(index_nino34,dd)*-1
            
            map1[i,j]=R
    print('map1=',map1)
    print('map1.shp=',map1.shape)
    
    max = ax[0,0].contourf(x,y,map1)
    ax[0,0].set_title('GPCP')
    ax[0,0].set_yticks([-10,0,10,20])
    #ax[0,0].set_xticks([90,100,110,120,130,140])
    
    fig.savefig(workdir+'nino34',dpi=300,bbox_inches='tight')
    exit()
    
    # ini ?? khusus zonal jika ingin obs=2 dan MMEW not included
    #model_datasets=np.delete(model_datasets,[1, -1])
    #model_names=np.delete(model_names,[1, -1])
    model_datasets=np.delete(model_datasets,[-1])
    model_names=np.delete(model_names,[-1])
       
    for i in np.arange(1):
        print (i)
        i=1
        dsi = xr.DataArray(model_datasets[i].values,
        coords={'time': obs_dataset.times,
        'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
        dims=["time", "lat", "lon"])  
        
        dsi = dsi.groupby('time.year').sum() 
        
        if i<3:
            m = Basemap(ax=ax[0,1+i], projection ='cyl', llcrnrlat = lat_min, urcrnrlat = lat_max,
                llcrnrlon = lon_min, urcrnrlon = lon_max, resolution = 'l', fix_aspect=True)
            m.drawcoastlines(linewidth=1)
            m.drawcountries(linewidth=1)
            m.drawstates(linewidth=0.5, color='w')
        
            x,y = np.meshgrid(ds.lon, ds.lat)
        
            from scipy.stats import pearsonr
        
            map1=ma.zeros((len(ds.lat),len(ds.lon)))

            for ii in np.arange(len(ds.lat)):
                for jj in np.arange(len(ds.lon)):
                    dd=dsi[:,ii,jj].values
                    #R=pearsonr(index_nino34, dd)[0]*-1
                    #R=xr.corr(index_nino34,dd)*-1
                    R=pearsonr(nino34yr.values, dd)[0]*-1
                    map1[ii,jj]=R
        
            max = ax[0,1+i].contourf(x,y,map1)
            ax[0,1+i].set_title(model_names[i])
            #ax[0,0].set_yticks([-10,0,10,20])
            #ax[0,1+i].set_xticks([90,100,110,120,130,140])
            
        else:
            m = Basemap(ax=ax[1,i-3], projection ='cyl', llcrnrlat = lat_min, urcrnrlat = lat_max,
                llcrnrlon = lon_min, urcrnrlon = lon_max, resolution = 'l', fix_aspect=True)
            m.drawcoastlines(linewidth=1)
            m.drawcountries(linewidth=1)
            m.drawstates(linewidth=0.5, color='w')
        
            x,y = np.meshgrid(ds.lon, ds.lat)
        
            from scipy.stats import pearsonr
        
            map1=ma.zeros((len(ds.lat),len(ds.lon)))

            for ii in np.arange(len(ds.lat)):
                for jj in np.arange(len(ds.lon)):
                    dd=dsi[:,ii,jj]
                    #R=pearsonr(index_nino34, dd)[0]*-1
                    R=xr.corr(index_nino34,dd)*-1
                    map1[ii,jj]=R
        
            max = ax[1,i-3].contourf(x,y,map1)
            ax[1,i-3].set_title(model_names[i])
            ax[1,0].set_yticks([-10,0,10,20])
            ax[1,i-3].set_xticks([90,100,110,120,130,140])
    
    
    plt.subplots_adjust(hspace=.2,wspace=.01)
    cax = fig.add_axes([0.91, 0.5, 0.015, 0.4])
    plt.colorbar(max, cax = cax) 
    
    file_name='Corr_nino34yr'
    fig.savefig(workdir+file_name,dpi=300,bbox_inches='tight')

def dsc(obs_dataset, obs_name, model_datasets, model_names, workdir):
    print('dsc.Delta_correction...')
    #downscaling delta correction cepat 10menit, QM lambat 25menit/model
    #hasil stdev ratio 1 ==> model=obs dan => R=1  karena keduanya dilakukan spatial?
    #kalo temporal bgm ==> 1 lokasi atau wilayah kecil dirata2 menjadi 1
    #bisa wilayah luas SEA dgn rata2 temporal => 
    #std dan R temporal =>  diagram taylor pada model ori dan dsc: sama! 
    
    
    #statistical_downscaling ok yang 2 malah nan?
    import ocw.statistical_downscaling as down
    
    ds = xr.DataArray(obs_dataset.values,
    coords={'time': obs_dataset.times,
            'lat': obs_dataset.lats, 
            'lon': obs_dataset.lons},
    dims=["time", "lat", "lon"],
    attrs=dict(
        description="Rainfall", var='pr',
        units="mm/month")
        )
    #rename blank var_name to 'pr')
    #ds = ds.to_dataset(name='pr') # ini error di r =ds[:,ii,jj].values
    #print('obs_dataset.times=',obs_dataset.times)
    #print('obs_dataset=',obs_dataset)
    
    #ds.to_netcdf(path=workdir+'tes.nc', mode='w', format ="NETCDF4_CLASSIC") #"NETCDF4")
    #exit() 
    import time as timer
    dsc2=ma.zeros((len(ds.time), len(ds.lat),len(ds.lon)))
    for i in np.arange(len(model_datasets)):
        
        start_time = timer.time()
        if i>0: 
        #if i==1: 
            print (i)
            print(model_names[i])
            dsi = xr.DataArray(model_datasets[i].values,
            coords={'time': obs_dataset.times,
            'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
            dims=["time", "lat", "lon"])  
            
            for ii in np.arange(len(ds.lat)):
                print(ii)
                for jj in np.arange(len(ds.lon)):
                    #print(ii)
                    r =ds[:,ii,jj].values
                    mp = dsi[:,ii,jj].values
                    dsc = down.Downscaling(r, mp, mp)
                        #R=pearsonr(r, mp)[0]
                        #Rd=pearsonr(r, dsc)[0]
                    #print(dsc)
                    #dscp, dscf = dsc.Delta_correction() 
                    dscp = dsc.Delta_correction() 
                    # data nan error?? # sudah bisa namun lambat 
                    #dscp, dscf = dsc.Quantile_mapping() 
                    #print(dscp)
                    #print('dsc max mean=',np.max(dsc),np.mean(dsc))
                    #print(dscp)
                    #print('dscp.min()',dscp.min())
                    #print(dscp.ndim)
                    dsc2[:,ii,jj] = dscp
                    
                    #print('dsc2 max mean=',dsc2.max(),dsc2.mean())
            dsc3 = xr.DataArray(dsc2,
            coords={'time': obs_dataset.times,
                    'lat': obs_dataset.lats, 
                    'lon': obs_dataset.lons},
            dims=["time", "lat", "lon"],
            attrs=dict( description="Rainfall",
                       units="mm/month"))
            print('dsc3 max, mean=',dsc3.max().values,dsc3.mean().values)
            #rename
            dsc3 = dsc3.to_dataset(name='pr')
            dsc3.to_netcdf(path=workdir+model_names[i]+'_delcor2_'+reg+'.nc', mode='w', format ="NETCDF4_CLASSIC")        
            
            #print("Selesai")
            elapsed_time = timer.time() - start_time
            print("time_menit=",elapsed_time/60)

def dsc2(obs_dataset, obs_name, model_datasets, model_names, workdir):
    from bias_correction import BiasCorrection, XBiasCorrection
    
    ds = xr.DataArray(obs_dataset.values,
    coords={'time': obs_dataset.times,
            'lat': obs_dataset.lats, 
            'lon': obs_dataset.lons},
    dims=["time", "lat", "lon"],
    attrs=dict(
        description="Rainfall", var='pr',
        units="mm/month")
        )
    #rename blank var_name to 'pr')
    #ds = ds.to_dataset(name='pr') # ini error di r =ds[:,ii,jj].values
    #print('obs_dataset.times=',obs_dataset.times)
    #print('obs_dataset=',obs_dataset)
    
    #ds.to_netcdf(path=workdir+'tes.nc', mode='w', format ="NETCDF4_CLASSIC") #"NETCDF4")
    #exit() 
    
    dsc2=ma.zeros((len(ds.time), len(ds.lat),len(ds.lon)))
    for i in np.arange(len(model_datasets)):
        
        #if i>9: 
        if i==10: 
            #print (i)
           
            print(model_names[i])
            dsi = xr.DataArray(model_datasets[i].values,
            coords={'time': obs_dataset.times,
            'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
            dims=["time", "lat", "lon"])  
            
            for ii in np.arange(len(ds.lat)):
                print(ii)
                for jj in np.arange(len(ds.lon)):
                    #print(ii)
                    r =ds[:,ii,jj].values
                    mp = dsi[:,ii,jj].values
                    
                    if not np.isnan(r).any():
                    
                        #print(r)
                        #print(mp)
                        
                        #not ok
                        #r=np.array(r)
                        #mp=np.array(mp)
                                               
                        bc = BiasCorrection(r, mp, mp)
                        #corrected = bc.correct(method='gamma_mapping')
                        #corrected = bc.correct(method='normal_mapping')
                        corrected = bc.correct(method='basic_quantile')
                        #corrected = bc.correct(method='modified_quantile')
               
                    else:
                        corrected= 'nan'
                    #print('corrected=', corrected)
                    #dsc2[:,ii,jj] = corrected
                    dsc2[:,ii,jj] = np.array(corrected)
                    #print('dsc2 max mean=',dsc2.max(),dsc2.mean())
            dsc3 = xr.DataArray(dsc2,
            coords={'time': obs_dataset.times,
                    'lat': obs_dataset.lats, 
                    'lon': obs_dataset.lons},
            dims=["time", "lat", "lon"],
            attrs=dict( description="Rainfall",
                       units="mm/month"))
            #print('dsc3 max mean=',dsc3.max(),dsc3.mean())
            #rename
            dsc3 = dsc3.to_dataset(name='pr')
            dsc3.to_netcdf(path=workdir+model_names[i]+'bc_gamma_sum.nc', 
                mode='w', format ="NETCDF4_CLASSIC")        


def metrik_cpi(obs_dataset, obs_name, model_datasets, model_names, workdir):
                
    ds = xr.DataArray(obs_dataset.values,
    coords={'time': obs_dataset.times,
    'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
    dims=["time", "lat", "lon"])
    
    #for cpi=exp[-1/2(s-o)^2 /sigma^2]
    so=ds.mean(dim=("time", "lat", "lon"))
        
    cpi=ma.zeros(len(model_datasets)-1) 
    import math
    for i in np.arange(len(model_datasets)):
        #print (i)
   
        dsi = xr.DataArray(model_datasets[i].values,
        coords={'time': obs_dataset.times,
        'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
        dims=["time", "lat", "lon"])  
        
        si=dsi.mean(dim=("time", "lat", "lon"))
        si2=dsi.mean()
        #print('cek', si.values, si2.values) #hasil sama
        
        sig=dsi.mean(dim="time").var(skipna=True) # spatial
        sig=dsi.var(skipna=True) # bisa juga
        
        #sig=dsi.mean(dim=("lat", "lon")).var(skipna=True) # temporal
        #print ('var model=', sig.values)
        #print('mean bias = ', (si-so).values)
        if i>0: #obs 2 tidak ikut
            cpi[i-1]=math.exp(-1*0.5*(si-so)**2/sig)
    #print('CPI = ',cpi)    
    return cpi #[-11] #obs2 -11 tidak ikut...gagal
        
 

def bobot_zonal_mean_rainfall(obs_dataset, obs_name, model_datasets, model_names, workdir):
    #Ini perlu dihitung dengan skor Taylor
    
    # calculate climatology of observation data
    #temporal mean per grid
    obs_clim = utils.calc_temporal_mean(obs_dataset)
    # determine the metrics
    map_of_bias = metrics.TemporalMeanBias()

    # create the Evaluation object
    bias_evaluation = Evaluation(obs_dataset, # Reference dataset for the evaluation
                                 model_datasets, # list of target datasets for the evaluation
                                 [map_of_bias, map_of_bias])
    # run the evaluation (bias calculation)
    bias_evaluation.run() 

    rcm_bias = bias_evaluation.results[0]
        
    
    fig, axes = plt.subplots(nrows=3, ncols=10,figsize=(16,6))
    #Tmax = 15; Tmin =0 ; delT = 5
    #clevels = np.arange(Tmin,Tmax+delT,delT)
    
    clevs = plotter2._nice_intervals(rcm_bias, 11)
    
    cmap2='RdBu_r'
    x_tick=['J','F','M','A','M','J','J','A','S','O','N','D']
    
    ds = xr.DataArray(obs_dataset.values,
    coords={'time': obs_dataset.times,
    'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
    dims=["time", "lat", "lon"])
        
    #ds = xr.DataArray(obs_dataset)
    ds = ds.groupby('time.month').mean() 
    #print('xx=',ds.shape, ds.lat.shape)
      
    ds0= ds.mean(dim='lon')
    #print('ds0=',ds0)
    
    lat=ds.lat
    lon=ds.lon
    lat_min=ds.lat.min()
    lon_min=ds.lon.min()
    
    max = axes[0,0].contourf(np.arange(12)+.5, lat, 
                      ds0.transpose(), 
                      levels=clevs, extend='both',
                      cmap=cmap2)
    axes[0,0].set_title('GPCP')
    axes[0,0].set_yticks([-10,0,10,20])
    axes[0,0].set_xticks([])
    
    # ini ?? khusus zonal jika ingin obs=2 dan MMEW not included
    #model_datasets=np.delete(model_datasets,[1, -1])
    #model_names=np.delete(model_names,[1, -1])
    model_datasets=np.delete(model_datasets,[-1])
    model_names=np.delete(model_names,[-1])
    
    from scipy.stats import pearsonr
    #c1 = ds0.dropna(dim="lat", how="any").stack(z=("month", "lat"))
    #sd1=ma.std(c1)
    
    sd1=ds0.std(skipna=None) #spatial??
    #print('sd1 =', sd1.values )
    mbz=ma.zeros(len(model_datasets)-1) 
    r0=1
    for i in np.arange(len(model_datasets)):
        #print(i)
        if i<9: axes[0,1+i].axis('off')
        ds = xr.DataArray(model_datasets[i].values,
        coords={'time': obs_dataset.times,
        'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
        dims=["time", "lat", "lon"])
        ds = ds.groupby('time.month').mean()
        ds= ds.mean(dim='lon')
        
        ##Untuk faktor bobot zonal mean
        ##Hasilnya naik naik sedikit (jika digabungkan ke bobot RAC 
        ## justru menurunkan T RAC yg sudah naik tinggi)
        ##pearsonr memerlukan dropna dan stack pada ds
        # c2 = ds.dropna(dim="lat", how="any").stack(z=("month", "lat"))
        # c=pearsonr(c1,c2)[0]
        # #stdev
        # sd2=ma.std(c2)
        # #print('bobot zonal sd2 , c2 =', sd2, c )
        # r0=1 #set
        # s=sd2/sd1
        
        #pakai xarray corr, std  
        #DataArray.std(dim=None, *, skipna=None)
        #xarray.corr(da_a, da_b, dim=None)
        c=xr.corr(ds0,ds)
        sd2=ds.std(skipna=True)
        s=sd2/sd1
        
        #print('corr=',c.values)
        #print('sd2 =',sd2.values)
        
        
        #T= (4*(1+c)**4)/(((s+1/s)**2)*((1+r0)**4))
        # Tanpa pakai **4 T naik dikit
        T= (4*(1+c))/(((s+1/s)**2)*((1+r0)))
        #Obs2 tidak ikut bobot 
        if i>0:
            #if i< len(model_datasets)-1:
            mbz[i-1]=T
        #####################################
        
        if i<10:
            cax = axes[1,i].contourf(np.arange(12)+.5, lat, 
                      ds.transpose(), 
                      levels=clevs, extend='both',
                      cmap=cmap2)
            axes[1,i].set_xticks(np.arange(12)+.5)
            axes[1,i].set_xticklabels(x_tick)
            axes[1,i].set_title(model_names[i])
            axes[1,0].set_yticks([-10,0,10,20])
            #axes[1,i].set_xticks([])
            if i<9: axes[1,1+i].set_yticks([])
                       
            #bias
            ds=ds-ds0
            #ds= ds.mean(dim='lon')
            bias=ds.mean()          
            
            cax = axes[2,i].contourf(np.arange(12)+.5, lat, 
                      ds.transpose(), 
                      levels=clevs, extend='both',
                      cmap=cmap2)
            #axes[2,i].set_title('bias='+'%.2f'%bias, 'bottom')
            axes[2,i].set_xticks([])
            axes[2,i].annotate('b='+'%.2f'%bias,xy=(3, lat_min), backgroundcolor='0.85',alpha=1)
            axes[2,0].set_yticks([-10,0,10,20])
            if i<9: axes[2,1+i].set_yticks([])
    print('faktor bobot zonal mean=',mbz)        
            
        
    #The dimensions [left, bottom, width, height] of the new axes
    cax = fig.add_axes([0.91, 0.5, 0.015, 0.4])
    plt.colorbar(max, cax = cax) 
    #plt.colorbar(cax).ax.set_title('mm')

    plt.subplots_adjust(hspace=.25,wspace=.15)
    
    file_name='Zonal mean'
    fig.savefig(workdir+file_name+reg,dpi=600,bbox_inches='tight')
    return mbz

def zonal_mean_rainfall(obs_dataset, obs_name, model_datasets, model_names, workdir):
    #dihitung dengan skor Taylor 
    #Khusus SEA
    
    # calculate climatology of observation data
    #temporal mean per grid
    obs_clim = utils.calc_temporal_mean(obs_dataset)
    # determine the metrics
    map_of_bias = metrics.TemporalMeanBias()

    # create the Evaluation object
    bias_evaluation = Evaluation(obs_dataset, # Reference dataset for the evaluation
                                 model_datasets, # list of target datasets for the evaluation
                                 [map_of_bias, map_of_bias])
    # run the evaluation (bias calculation)
    bias_evaluation.run() 

    rcm_bias = bias_evaluation.results[0]
        
    
    fig, axes = plt.subplots(nrows=3, ncols=10,figsize=(16,6))
    #Tmax = 15; Tmin =0 ; delT = 5
    #clevels = np.arange(Tmin,Tmax+delT,delT)
    
    clevs = plotter2._nice_intervals(rcm_bias, 11)
    
    cmap2='RdBu_r'
    x_tick=['J','F','M','A','M','J','J','A','S','O','N','D']
    
    ds = xr.DataArray(obs_dataset.values,
    coords={'time': obs_dataset.times,
    'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
    dims=["time", "lat", "lon"])
        
    #ds = xr.DataArray(obs_dataset)
    ds = ds.groupby('time.month').mean() 
    #print('xx=',ds.shape, ds.lat.shape)
      
    ds0= ds.mean(dim='lon')
    #print('ds0=',ds0)
    
    lat=ds.lat
    lon=ds.lon
    lat_min=ds.lat.min()
    lon_min=ds.lon.min()
    
    max = axes[0,0].contourf(np.arange(12)+.5, lat, 
                      ds0.transpose(), 
                      levels=clevs, extend='both',
                      cmap=cmap2)
    axes[0,0].set_title('GPCP', fontsize=6)
    axes[0,0].set_yticks([-10,0,10,20])
    axes[0,0].yaxis.set_tick_params(labelsize=6)
    axes[0,0].set_xticks([])
    
    # ini ?? khusus zonal jika ingin obs=2 dan MMEW not included
    #model_datasets=np.delete(model_datasets,[1, -1])
    #model_names=np.delete(model_names,[1, -1])
    model_datasets=np.delete(model_datasets,[-1])
    model_names=np.delete(model_names,[-1])
    
    from scipy.stats import pearsonr
    #c1 = ds0.dropna(dim="lat", how="any").stack(z=("month", "lat"))
    #sd1=ma.std(c1)
    
    sd1=ds0.std(skipna=None) 
    #print('sd1 =', sd1.values )
    mbz=ma.zeros(len(model_datasets)) 
    r0=1
    for i in np.arange(len(model_datasets)):
        #print(i)
        if i<9: axes[0,1+i].axis('off')
        ds = xr.DataArray(model_datasets[i].values,
        coords={'time': obs_dataset.times,
        'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
        dims=["time", "lat", "lon"])
        ds = ds.groupby('time.month').mean()
        ds= ds.mean(dim='lon')
        
        ##Untuk faktor bobot zonal mean
        ##Hasilnya naik naik sedikit (jika digabungkan ke bobot RAC 
        ## justru menurunkan T RAC yg sudah naik tinggi)
        ##pearsonr memerlukan dropna dan stack pada ds
        # c2 = ds.dropna(dim="lat", how="any").stack(z=("month", "lat"))
        # c=pearsonr(c1,c2)[0]
        # #stdev
        # sd2=ma.std(c2)
        # #print('bobot zonal sd2 , c2 =', sd2, c )
        # r0=1 #set
        # s=sd2/sd1
        
        #pakai xarray corr, std  
        #DataArray.std(dim=None, *, skipna=None)
        #xarray.corr(da_a, da_b, dim=None)
        c=xr.corr(ds0,ds)
        sd2=ds.std(skipna=True)
        s=sd2/sd1
        
        #print('corr=',c.values)
        #print('sd2 =',sd2.values)
        
        
        #T= (4*(1+c)**4)/(((s+1/s)**2)*((1+r0)**4))
        # Tanpa pakai **4 T naik dikit
        T= (4*(1+c))/(((s+1/s)**2)*((1+r0)))
        #Obs2 tidak ikut bobot 
        #if i>0:
            #if i< len(model_datasets)-1:
            #mbz[i-1]=T
        mbz[i]=T
        #####################################
        
        if i<10:
            cax = axes[1,i].contourf(np.arange(12)+.5, lat, 
                      ds.transpose(), 
                      levels=clevs, extend='both',
                      cmap=cmap2)
            axes[1,i].set_xticks(np.arange(12)+.5)
            axes[1,i].set_xticklabels(x_tick,fontsize=5)
            axes[1,i].set_title(model_names[i]+'('+'%.2f'%mbz[i]+')',fontsize=6)
            axes[1,0].set_yticks([-10,0,10,20])
            axes[1,0].yaxis.set_tick_params(labelsize=6)
            #axes[1,i].set_xticks([])
            if i<9: axes[1,1+i].set_yticks([])
                       
            #bias
            ds=ds-ds0
            #ds= ds.mean(dim='lon')
            bias=ds.mean()          
            
            cax = axes[2,i].contourf(np.arange(12)+.5, lat, 
                      ds.transpose(), 
                      levels=clevs, extend='both',
                      cmap=cmap2)
            #axes[2,i].set_title('bias='+'%.2f'%bias, 'bottom')
            axes[2,i].set_xticks([])
            print(i)
            #axes[2,i].annotate('S='+'%.2f'%mbz[i-1],xy=(3, lat_min), backgroundcolor='0.85',alpha=1)
            axes[2,0].set_yticks([-10,0,10,20])
            axes[2,0].yaxis.set_tick_params(labelsize=6)
            if i<9: axes[2,1+i].set_yticks([])
    print('S_Taylor zonal mean=',mbz)        
            
        
    #The dimensions [left, bottom, width, height] of the new axes
    cax = fig.add_axes([0.91, 0.2, 0.015, 0.4])
    cbar=plt.colorbar(max, cax = cax)
    cbar.ax.tick_params(labelsize=7)
    #plt.colorbar(cax).ax.set_title('mm')

    plt.subplots_adjust(hspace=.2,wspace=.15)
    
    file_name='Zonal mean_STaylor'
    fig.savefig(workdir+file_name+reg,dpi=300,bbox_inches='tight')
    return mbz

def zonal_mean_rainfall_noWE2(obs_dataset, obs_name, model_datasets, model_names, workdir):
    #dihitung dengan skor Taylor 
    #Khusus SEA dan LO
    
    # # calculate climatology of observation data
    # #temporal mean per grid
    # obs_clim = utils.calc_temporal_mean(obs_dataset)
    # # determine the metrics
    # map_of_bias = metrics.TemporalMeanBias()

    # # create the Evaluation object
    # bias_evaluation = Evaluation(obs_dataset, # Reference dataset for the evaluation
                                 # model_datasets, # list of target datasets for the evaluation
                                 # [map_of_bias, map_of_bias])
    # # run the evaluation (bias calculation)
    # bias_evaluation.run() 

    # rcm_bias = bias_evaluation.results[0]
    
    #ini tidak cocok untuk dsc perlu di set yg tepat
    #akibat bias menjadi sangat kecil 
    #clevs = plotter2._nice_intervals(rcm_bias, 11)
        
    
    fig, axes = plt.subplots(nrows=2, ncols=7) #,figsize=(5,6))
    Tmax = 12; Tmin =0 ; delT = 5
    clevs = np.arange(Tmin,Tmax+delT,delT)
    
   
    cmap2='rainbow'
    x_tick=['J','F','M','A','M','J','J','A','S','O','N','D']
    
    ds = xr.DataArray(obs_dataset.values,
    coords={'time': obs_dataset.times,
    'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
    dims=["time", "lat", "lon"])
        
    #ds = xr.DataArray(obs_dataset)
    ds = ds.groupby('time.month').mean() 
    #print('xx=',ds.shape, ds.lat.shape)
      
    ds0= ds.mean(dim='lon')
    #print('ds0=',ds0)
    
    lat=ds.lat
    lon=ds.lon
    lat_min=ds.lat.min()
    lon_min=ds.lon.min()
    
    max = axes[0,0].contourf(np.arange(12)+.5, lat, 
                      ds0.transpose(), 
                      levels=clevs, extend='both',
                      cmap=cmap2)
    axes[0,0].set_title(obs_name, fontsize=7)
    axes[0,0].set_yticks([-10,0,10,20])
    axes[0,0].set_yticklabels(['10$^\circ$S','EQ','10$^\circ$N','20$^\circ$N'])
    axes[0,0].yaxis.set_tick_params(labelsize=7)
    axes[0,0].set_xticks([])
    
   
    
    from scipy.stats import pearsonr
  
    sd1=ds0.std(skipna=None) 
    #print('sd1 =', sd1.values )
    mbz=ma.zeros(len(model_datasets)) 
    r0=1
    for i in np.arange(len(model_datasets)):
        print(i)
        print(model_names[i])
       
        dsi = xr.DataArray(model_datasets[i].values,
        coords={'time': model_datasets[i].times,
        'lat': model_datasets[i].lats, 'lon': model_datasets[i].lons},
        dims=["time", "lat", "lon"])
        lat=dsi.lat
        dsi = dsi.groupby('time.month').mean()
        dsi= dsi.mean(dim='lon')
        
        
        print(ds0.shape, dsi.shape)
        #print(ds0.lat.values, dsi.lat.values)
        c=xr.corr(ds0,dsi)
        sd2=dsi.std(skipna=True)
        s=sd2/sd1
        print(c.values,s.values)
   
        #T= (4*(1+c)**4)/(((s+1/s)**2)*((1+r0)**4))
        # Tanpa pakai **4 T naik dikit
        T= (4*(1+c))/(((s+1/s)**2)*((1+r0)))
     
        mbz[i]=T
        
        #-----------------------------------
        if i==0: 
            ds11=dsi
            sd11=sd2
        
        c=xr.corr(ds11,dsi)
        sd2=dsi.std(skipna=True)
        s=sd2/sd11        
        T2= (4*(1+c))/(((s+1/s)**2)*((1+r0)))
        
        
        axes[0,6].axis('off')
        #model datasets from 0 to 10 and 11 MME
        if i in [0,1,2,3,4]:
            cax = axes[0,i+1].contourf(np.arange(12)+.5, lat, 
                      dsi.transpose(), 
                      levels=clevs, extend='both',
                      cmap=cmap2)
            axes[0,i+1].set_xticks([])
            axes[0,i+1].set_xticklabels([])
            axes[0,i+1].set_yticks([])
            axes[0,i+1].set_yticklabels([])
            if i==0:
                axes[0,i+1].set_title(model_names[i]+' ('+'%.2f'%mbz[i]+')',fontsize=7)
            if i>0:
                axes[0,i+1].set_title(model_names[i]+' ('+'%.2f'%mbz[i]+')'+'('+'%.2f'%T2+')',fontsize=7)

                
        if i in [5,6,7,8,9,10,11] :
            n=5
            cax = axes[1,i-n].contourf(np.arange(12)+.5, lat, 
                      dsi.transpose(), 
                      levels=clevs, extend='both',
                      cmap=cmap2)
            axes[1,i-n].set_xticks(np.arange(12)+.5)
            axes[1,i-n].set_xticklabels(x_tick, fontsize=7)
            axes[1,i-n].set_title(model_names[i]+' ('+'%.2f'%mbz[i]+')'+'('+'%.2f'%T2+')',fontsize=7)
            axes[1,0].set_yticks([-10,0,10,20])
            axes[1,0].set_yticklabels(['10$^\circ$S','EQ','10$^\circ$N','20$^\circ$N'])
            axes[1,0].yaxis.set_tick_params(labelsize=7)
            #axes[1,i].set_xticks([])
            axes[1,i-n].set_yticks([])
            
        if len(model_names)==11: axes[1,6].axis('off')
        
              
    print('S_Taylor zonal mean=',mbz)        
   
        
    #The dimensions [left, bottom, width, height] of the new axes
    cax = fig.add_axes([0.80, 0.54, 0.015, 0.35])
    
    plt.colorbar(max, cax = cax).ax.tick_params(labelsize=7)
    #plt.colorbar(cax).ax.set_title('mm')

    plt.subplots_adjust(hspace=.15,wspace=.08)
    
    file_name='14Zonal mean_STaylor_dsc'
    fig.savefig(workdir+file_name+reg,dpi=300,bbox_inches='tight')
    plt.show()
    return mbz
    
def zonal_mean_rainfall_noWE(obs_dataset, obs_name, model_datasets, model_names, workdir):
    #dihitung dengan skor Taylor 
    #Khusus SEA dan LO bisa L
    
       
    fig, axes = plt.subplots(nrows=3, ncols=4,figsize=(16,6))
    Tmax = 12; Tmin =0 ; delT = 5
    clevs = np.arange(Tmin,Tmax+delT,delT)
    
   
    cmap2='rainbow'
    x_tick=['J','F','M','A','M','J','J','A','S','O','N','D']
    
    ds = xr.DataArray(obs_dataset.values,
    coords={'time': obs_dataset.times,
    'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
    dims=["time", "lat", "lon"])
        
    #ds = xr.DataArray(obs_dataset)
    ds = ds.groupby('time.month').mean() 
    #print('xx=',ds.shape, ds.lat.shape)
      
    ds0= ds.mean(dim='lon')
    #print('ds0=',ds0)
    
    lat=ds.lat
    lon=ds.lon
    lat_min=ds.lat.min()
    lon_min=ds.lon.min()
    
    max = axes[0,0].contourf(np.arange(12)+.5, lat, 
                      ds0.transpose(), 
                      levels=clevs, extend='both',
                      cmap=cmap2)
    axes[0,0].set_title(obs_name, fontsize=9)
    axes[0,0].set_yticks([-10,0,10,20])
    axes[0,0].yaxis.set_tick_params(labelsize=7)
    axes[0,0].set_xticks([])
    
   
    
    from scipy.stats import pearsonr
  
    sd1=ds0.std(skipna=None) 
    #print('sd1 =', sd1.values )
    mbz=ma.zeros(len(model_datasets)) 
    r0=1
    nn=3
    for i in np.arange(len(model_datasets)):
        print(i)
        print(model_names[i])
        #if i<6: axes[0,1+i].axis('off')
        dsi = xr.DataArray(model_datasets[i].values,
        coords={'time': model_datasets[i].times,
        'lat': model_datasets[i].lats, 'lon': model_datasets[i].lons},
        dims=["time", "lat", "lon"])
        lat=dsi.lat
        dsi = dsi.groupby('time.month').mean()
        dsi= dsi.mean(dim='lon')
        
        
        print(ds0.shape, dsi.shape)
        #print(ds0.lat.values, dsi.lat.values)
        c=xr.corr(ds0,dsi)
        sd2=dsi.std(skipna=True)
        s=sd2/sd1
        print(c.values,s.values)
   
        #T= (4*(1+c)**4)/(((s+1/s)**2)*((1+r0)**4))
        # Tanpa pakai **4 T naik dikit
        T= (4*(1+c))/(((s+1/s)**2)*((1+r0)))
     
        mbz[i]=T
        #####################################
        
        
        if i<nn:
            cax = axes[0,i+1].contourf(np.arange(12)+.5, lat, 
                      dsi.transpose(), 
                      levels=clevs, extend='both',
                      cmap=cmap2)
            axes[0,i+1].set_xticks([])
            axes[0,i+1].set_xticklabels([])
            axes[0,i+1].set_title(model_names[i]+'('+'%.2f'%mbz[i]+')',fontsize=9)
            axes[0,0].set_yticks([-10,0,10,20])
            axes[0,0].set_yticklabels(['10S','0','10N','20N'])
            axes[0,0].yaxis.set_tick_params(labelsize=7)
            #axes[1,i].set_xticks([])
            axes[0,1+i].set_yticks([])
        
        if nn+3>=i>=nn:
            cax = axes[1,i-nn].contourf(np.arange(12)+.5, lat, 
                      dsi.transpose(), 
                      levels=clevs, extend='both',
                      cmap=cmap2)
            axes[1,i-nn].set_xticks(np.arange(12)+.5)
            axes[1,i-nn].set_xticklabels(x_tick, fontsize=7)
            axes[1,i-nn].set_title(model_names[i]+'('+'%.2f'%mbz[i]+')',fontsize=9)
            axes[1,0].set_yticks([-10,0,10,20])
            axes[1,0].set_yticklabels(['10S','0','10N','20N'])
            axes[1,0].yaxis.set_tick_params(labelsize=7)
            axes[1,i-nn].set_xticks([])
            axes[1,i-nn].set_yticks([])                     
   
        if i>nn+3:
            cax = axes[2,i-nn-4].contourf(np.arange(12)+.5, lat, 
                      dsi.transpose(), 
                      levels=clevs, extend='both',
                      cmap=cmap2)
            axes[2,i-nn-4].set_xticks(np.arange(12)+.5)
            axes[2,i-nn-4].set_xticklabels(x_tick, fontsize=7)
            axes[2,i-nn-4].set_title(model_names[i]+'('+'%.2f'%mbz[i]+')',fontsize=9)
            axes[2,0].set_yticks([-10,0,10,20])
            axes[2,0].set_yticklabels(['10S','0','10N','20N'])
            axes[2,0].yaxis.set_tick_params(labelsize=7)
            axes[2,i-nn-4].set_yticks([])    
            #axes[2,i-nn-4].set_xticks([])



    print('S_Taylor zonal mean=',mbz)        
    axes[2,3].axis('off')
    axes[1,3].set_xticks(np.arange(12)+.5)
    axes[1,3].set_xticklabels(x_tick, fontsize=7)
    #cax = fig.add_axes([0.91, 0.4, 0.015, 0.5])
    #cbar=plt.colorbar(max, cax = cax) 
    
    cax = fig.add_axes([0.715, 0.2, 0.19, 0.023])
    #cax = fig.add_axes([0.5, 0.97, 0.6, 0.02])
    cbar=plt.colorbar(max, cax = cax, orientation='horizontal')
    
    cbar.ax.tick_params(labelsize=7)
    
    #cbar.ax.set_ylabel('mm/month', rotation=270, labelpad=7)
    cbar.ax.set_xlabel('mm/day', labelpad=3)

    plt.subplots_adjust(hspace=.25,wspace=.15)
    
    file_name='Zonal mean_STaylor_'
    fig.savefig(workdir+file_name+reg,dpi=300,bbox_inches='tight')
    plt.show()
    

def zonal_mean_rainfall_15(obs_dataset, obs_name, model_datasets, model_names, workdir):
    #dihitung dengan skor Taylor 
    #Khusus SEA dan LO bisa L
    
       
    fig, axes = plt.subplots(nrows=3, ncols=7,figsize=(6,6))
    Tmax = 12; Tmin =0 ; delT = 5
    clevs = np.arange(Tmin,Tmax+delT,delT)
    
   
    cmap2='rainbow'
    x_tick=['J','F','M','A','M','J','J','A','S','O','N','D']
    
    ds = xr.DataArray(obs_dataset.values,
    coords={'time': obs_dataset.times,
    'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
    dims=["time", "lat", "lon"])
        
    #ds = xr.DataArray(obs_dataset)
    ds = ds.groupby('time.month').mean() 
    #print('xx=',ds.shape, ds.lat.shape)
      
    ds0= ds.mean(dim='lon')
    #print('ds0=',ds0)
    
    lat=ds.lat
    lon=ds.lon
    lat_min=ds.lat.min()
    lon_min=ds.lon.min()
    
    max = axes[0,0].contourf(np.arange(12)+.5, lat, 
                      ds0.transpose(), 
                      levels=clevs, extend='both',
                      cmap=cmap2)
    axes[0,0].set_title(obs_name, fontsize=9)
    axes[0,0].set_yticks([-10,0,10,20])
    axes[0,0].set_yticklabels(['10S','EQ','10N','20N'])
    axes[0,0].yaxis.set_tick_params(labelsize=7)
    axes[0,0].set_xticks([])
    
   
    
    from scipy.stats import pearsonr
  
    sd1=ds0.std(skipna=None) 
    #print('sd1 =', sd1.values )
    mbz=ma.zeros(len(model_datasets)) 
    r0=1
    nn=7
    for i in np.arange(len(model_datasets)):
        print(i)
        print(model_names[i])
        if i<5: axes[0,2+i].axis('off')
        
        dsi = xr.DataArray(model_datasets[i].values,
        coords={'time': model_datasets[i].times,
        'lat': model_datasets[i].lats, 'lon': model_datasets[i].lons},
        dims=["time", "lat", "lon"])
        lat=dsi.lat
        dsi = dsi.groupby('time.month').mean()
        dsi= dsi.mean(dim='lon')
        
        
        print(ds0.shape, dsi.shape)
        #print(ds0.lat.values, dsi.lat.values)
        c=xr.corr(ds0,dsi)
        sd2=dsi.std(skipna=True)
        s=sd2/sd1
        print(c.values,s.values)
   
        #T= (4*(1+c)**4)/(((s+1/s)**2)*((1+r0)**4))
        # Tanpa pakai **4 T naik dikit
        T= (4*(1+c))/(((s+1/s)**2)*((1+r0)))
     
        mbz[i]=T
        #####################################
        
        if i==0:
            
            cax = axes[0,1].contourf(np.arange(12)+.5, lat, 
                      dsi.transpose(), 
                      levels=clevs, extend='both',
                      cmap=cmap2)
            axes[0,1].set_xticks(np.arange(12)+.5)
            axes[0,1].set_xticklabels(x_tick, fontsize=7)
            axes[0,1].set_title(model_names[i]+'('+'%.2f'%mbz[i]+')',fontsize=9)
            axes[0,1].set_yticks([])
            axes[0,1].set_yticklabels([])
           
            
        #1-7
        if 0<i<8:
            nn=1
            cax = axes[1,i-nn].contourf(np.arange(12)+.5, lat, 
                      dsi.transpose(), 
                      levels=clevs, extend='both',
                      cmap=cmap2)
            axes[1,i-nn].set_xticks(np.arange(12)+.5)
            axes[1,i-nn].set_xticklabels(x_tick, fontsize=7)
            axes[1,i-nn].set_title(model_names[i]+'('+'%.2f'%mbz[i]+')',fontsize=9)
           
            axes[1,0].yaxis.set_tick_params(labelsize=7)
            axes[1,i-nn].set_xticks([])
            axes[1,i-nn].set_yticks([])                     
   
        if i>=8:
            nn=8
            print(i)
            cax = axes[2,i-nn].contourf(np.arange(12)+.5, lat, 
                      dsi.transpose(), 
                      levels=clevs, extend='both',
                      cmap=cmap2)
            axes[2,i-nn].set_xticks(np.arange(12)+.5)
            axes[2,i-nn].set_xticklabels(x_tick, fontsize=7)
            axes[2,i-nn].set_title(model_names[i]+'('+'%.2f'%mbz[i]+')',fontsize=9)
            
            axes[2,i-nn].set_yticks([])    
            #axes[2,i-nn-4].set_xticks([])


    axes[1,0].set_yticks([-10,0,10,20])
    axes[1,0].set_yticklabels(['10S','EQ','10N','20N'])
    axes[2,0].set_yticks([-10,0,10,20])
    axes[2,0].set_yticklabels(['10S','EQ','10N','20N'])
    axes[2,0].yaxis.set_tick_params(labelsize=7)
    print('S_Taylor zonal mean=',mbz)        
    #khusus noIPSL
    #axes[2,6].set_yticks([])
    axes[0,1].set_xticks([])  
        
    #The dimensions [left, bottom, width, height] of the new axes
    #cax = fig.add_axes([0.91, 0.4, 0.015, 0.5])
    #plt.colorbar(max, cax = cax) 
    #plt.colorbar(cax).ax.set_title('mm')
    cax = fig.add_axes([0.43, 0.7, 0.4, 0.02]) #horisontal
    #plt.colorbar(max, cax = cax, extend='both', orientation='horizontal') 
    cbar=plt.colorbar(max, cax = cax, extend='both', orientation='horizontal') 
    cbar.ax.tick_params(labelsize=7)
    cbar.ax.set_title ('mm/day')

    plt.subplots_adjust(hspace=.25,wspace=.15)
    
    file_name='Zonal mean_STaylor_'
    fig.savefig(workdir+file_name+reg,dpi=300,bbox_inches='tight')
    plt.show()
    return mbz

def mean_rainfall_14(obs_dataset, obs_name, model_datasets, model_names, workdir):
  
    #Khusus SEA dan LO
    #dihitung dengan skor Taylor 
    # # calculate climatology of observation data
    # #temporal mean per grid
    # obs_clim = utils.calc_temporal_mean(obs_dataset)
    # # determine the metrics
    # map_of_bias = metrics.TemporalMeanBias()

    # # create the Evaluation object
    # bias_evaluation = Evaluation(obs_dataset, # Reference dataset for the evaluation
                                 # model_datasets, # list of target datasets for the evaluation
                                 # [map_of_bias, map_of_bias])
    # # run the evaluation (bias calculation)
    # bias_evaluation.run() 

    # rcm_bias = bias_evaluation.results[0]
    
    #ini tidak cocok untuk dsc perlu di set yg tepat
    #akibat bias menjadi sangat kecil 
    #clevs = plotter2._nice_intervals(rcm_bias, 11)
        
    
    fig, axes = plt.subplots(nrows=3, ncols=5,figsize=(4,6))
    Tmax = 15; Tmin =0 ; delT = 3
    clevs = np.arange(Tmin,Tmax+delT,delT)
    
    #cmap2='viridis_r'
    cmap2='rainbow'
    x_tick=['J','F','M','A','M','J','J','A','S','O','N','D']
    
    ds = xr.DataArray(obs_dataset.values,
    coords={'time': obs_dataset.times,
    'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
    dims=["time", "lat", "lon"])
        
         
    #not season
    ds0= ds.mean(dim='time')
    #print('ds0=',ds0)
    
    plot='SEA'
    
    if plot=='SEA':
        lat_min1 = 0.22*3
        lat_max1 = 0.22*6
        lon_min1 = 0.22*4
        lon_max1 = 0.22*3
    
    if plot=='sum':
        lat_min1 = 0.22*1
        lat_max1 = 0.22*3
        lon_min1 = 0.22*3
        lon_max1 = 0.22*1
    
    
    
    lat=ds.lat
    lon=ds.lon
    lat_min = ds.lat.min()
    lat_max = ds.lat.max()
    lon_min = ds.lon.min()
    lon_max = ds.lon.max()
    x,y = np.meshgrid(ds.lon, ds.lat)
    
    m = Basemap(ax=axes[0,0], projection ='cyl', 
                llcrnrlat = lat_min+lat_min1, urcrnrlat = lat_max-lat_max1,
                llcrnrlon = lon_min+lon_min1, urcrnrlon = lon_max-lon_max1, 
                resolution = 'l', fix_aspect=False)
    m.drawcoastlines(linewidth=1)
    m.drawcountries(linewidth=1)
    m.drawstates(linewidth=0.5, color='w')
    
    max = axes[0,0].contourf(lon, lat, 
                      ds0, 
                      levels=clevs, 
                      extend='both',
                      cmap=cmap2
                      )
    axes[0,0].set_title(obs_name, pad=3, fontsize=9)
    axes[0,0].set_xticks([])
    axes[0,0].set_xticklabels([])
    
    for i in [0,1,2]:
        axes[i,0].set_yticks([-10,0,10,20])   
        axes[i,0].set_yticklabels(['10$^\circ$S','EQ','10$^\circ$N','20$^\circ$N'])
        axes[i,0].tick_params(axis='y', pad=1,labelsize=7)
     
   
    
    from scipy.stats import pearsonr
  
    sd1=ds0.std(skipna=None) 
    #print('sd1 =', sd1.values )
    mbz=ma.zeros(len(model_datasets)) 
    r0=1
    for i in np.arange(len(model_datasets)):
        #print(i)
        #if i<6: axes[0,1+i].axis('off')
        ds = xr.DataArray(model_datasets[i].values,
        coords={'time': obs_dataset.times,
        'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
        dims=["time", "lat", "lon"])
        #ds = ds.groupby('time.month').mean()
        ds= ds.mean(dim='time')
        
       
        c=xr.corr(ds0,ds)
        sd2=ds.std(skipna=True)
        s=sd2/sd1
   
        #T= (4*(1+c)**4)/(((s+1/s)**2)*((1+r0)**4))
        # Tanpa pakai **4 T naik dikit
        T= (4*(1+c))/(((s+1/s)**2)*((1+r0)))
     
        mbz[i]=T
        #####################################
        
        if 0<=i<4:
            
            m = Basemap(ax=axes[0,i+1], projection ='cyl', 
                llcrnrlat = lat_min+lat_min1, urcrnrlat = lat_max-lat_max1,
                llcrnrlon = lon_min+lon_min1, urcrnrlon = lon_max-lon_max1, 
                resolution = 'l', fix_aspect=False)
            m.drawcoastlines(linewidth=1)
            m.drawcountries(linewidth=1)
            m.drawstates(linewidth=0.5, color='w')
            
             
            cax = axes[0,i+1].contourf(lon, lat, 
                      ds, 
                      levels=clevs, extend='both',
                      cmap=cmap2)
            axes[0,i+1].set_xticks([])
            axes[0,i+1].set_xticklabels([])
            axes[0,i+1].set_title(model_names[i]+'('+'%.2f'%mbz[i]+')',pad=3,fontsize=9)
            
          
            axes[0,1+i].set_yticks([])
        
        if 9>i>=4:
            m = Basemap(ax=axes[1,i-4], projection ='cyl', 
                llcrnrlat = lat_min+lat_min1, urcrnrlat = lat_max-lat_max1,
                llcrnrlon = lon_min+lon_min1, urcrnrlon = lon_max-lon_max1, 
                resolution = 'l', fix_aspect=False)
            m.drawcoastlines(linewidth=1)
            m.drawcountries(linewidth=1)
            m.drawstates(linewidth=0.5, color='w')
                       
            cax = axes[1,i-4].contourf(lon, lat, 
                      ds, 
                      levels=clevs, extend='both',
                      cmap=cmap2)
            #axes[1,i-4].set_xticks([100,120,140])
            #axes[1,i-4].xaxis.set_tick_params(labelsize=6)
            axes[1,i-4].set_title(model_names[i]+'('+'%.2f'%mbz[i]+')',pad=3,fontsize=9)
            axes[1,0].set_yticks([-10,0,10,20])
            #axes[1,0].yaxis.set_tick_params(labelsize=6)
            axes[1,0].set_yticklabels(['10S','EQ','10N','20N'], fontsize=7)
            axes[1,i-4].set_xticks([])
            if i>4: axes[1,i-4].set_yticks([])                     
    
        if i>=9:
        
            m = Basemap(ax=axes[2,i-9], projection ='cyl', 
                llcrnrlat = lat_min+lat_min1, urcrnrlat = lat_max-lat_max1,
                llcrnrlon = lon_min+lon_min1, urcrnrlon = lon_max-lon_max1, 
                resolution = 'l', fix_aspect=False)
            m.drawcoastlines(linewidth=1)
            m.drawcountries(linewidth=1)
            m.drawstates(linewidth=0.5, color='w')
            
            cax = axes[2,i-9].contourf(lon, lat, 
                      ds, 
                      levels=clevs, extend='both',
                      cmap=cmap2)
            axes[2,i-9].set_xticks([100,120,140])
            #axes[2,i-9].xaxis.set_tick_params(labelsize=6)
            axes[2,i-9].set_xticklabels(['100$^\circ$E','120$^\circ$E','140$^\circ$E'], fontsize=7)
            axes[2,i-9].tick_params(axis='x', pad=1,labelsize=7)   
            axes[2,i-9].set_title(model_names[i]+'('+'%.2f'%mbz[i]+')',pad=3,fontsize=9)
            axes[2,0].set_yticks([-10,0,10,20])
            #axes[2,0].yaxis.set_tick_params(labelsize=6)
            axes[2,0].set_yticklabels(['10S','EQ','10N','20N'], fontsize=7)
           
            if i>9: axes[2,i-9].set_yticks([])
  
    print('S_Taylor zonal mean=',mbz)
    import pandas as pd
    df = pd.DataFrame([model_names, mbz])
    df.T.to_excel(workdir+reg+'mean_pre_ann.xlsx', index=False, header=False) 
    
    axes[2,3].set_yticks([])
    axes[2,4].set_yticks([])
    #axes[2,3].set_xticks([])
    #axes[2,4].set_xticks([])  
    for i in [0,1,2]:
        axes[i,0].set_yticks([-10,0,10,20])   
        axes[i,0].set_yticklabels(['10$^\circ$S','EQ','10$^\circ$N','20$^\circ$N'])
        axes[i,0].tick_params(axis='y', pad=1,labelsize=7)
        
    #The dimensions [left, bottom, width, height] of the new axes
    cax = fig.add_axes([0.91, 0.5, 0.015, 0.4])
    cbar=plt.colorbar(max, cax = cax) 
    #plt.colorbar(cax).ax.set_title('mm')
    cbar.ax.tick_params(labelsize=7)

    plt.subplots_adjust(hspace=.15,wspace=.05)
    plt.show()
    file_name='mean_rainfall_Taylor_14_dsc'
    fig.savefig(workdir+file_name+reg,dpi=300) #,bbox_inches='tight')
    return mbz

def mean_rainfall_14_season_sig(obs_dataset, obs_name, model_datasets, model_names, workdir):
    import xarray as xr
    import cartopy.crs as ccrs
    import matplotlib.pyplot as plt
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    from scipy import stats
  
    musim='DJF'
   
    
    ds = xr.DataArray(obs_dataset.values,
    coords={'time': obs_dataset.times,
    'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
    dims=["time", "lat", "lon"])
        
    #season
    #dss = ds.groupby('time.season')
    #ds0= dss[musim].mean(dim='time')
   
    ds0=ds.mean(dim='time')
     
    # Step 6: Interpret the results and plot on a map
    significance_level = 0.05

    # Create a grid of latitude and longitude coordinates
    lons, lats = np.meshgrid(ds.lon.values, ds.lat.values)

    # Create a figure and axis using Cartopy
    fig = plt.figure(figsize=(10, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
   
    for i in np.arange(len(model_datasets)):
        print('i=',i)
        #if i<6: axes[0,1+i].axis('off')
        dsi = xr.DataArray(model_datasets[i].values,
        coords={'time': obs_dataset.times,
        'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
        dims=["time", "lat", "lon"])
        
        #dss= dsi.groupby('time.season')
        #dsi= dss[musim].mean(dim='time')
        dsi=dsi.mean(dim='time')
        
        ds=dsi-ds0
        ds.plot()
        plt.show()
        
    exit()
    '''    
        print(dsi)
        print(dsi.shape)
        x0=ds0.stack(z=("lat", "lon"))
        y0=dsi.stack(z=("lat", "lon"))
        
        bad = ~np.logical_or(np.isnan(x0), np.isnan(y0))
        x1=np.compress(bad, x0) 
        y1=np.compress(bad, y0)
        
        observed=x1.values
        predicted=y1.values
        
        #observed=ds0
        #predicted=dsi
        
        # Step 3: Calculate mean absolute error (MAE)
        mae = mean_absolute_error(observed, predicted)

        # Step 4: Calculate root mean squared error (RMSE)
        rmse = np.sqrt(mean_squared_error(observed, predicted))
        
        print('mae, rmse', mae, rmse)
        
        # Step 5: Conduct a paired t-test for significance level
        t_statistic, p_value = stats.ttest_rel(observed, predicted)
        
        print('t_statistic, p_value', t_statistic, p_value)
      

        # Plot the results on a map
        im = ax.pcolormesh(lons, lats, p_value < significance_level, cmap='coolwarm', transform=ccrs.PlateCarree())

        # Add coastlines and a colorbar
        ax.coastlines()
        cbar = plt.colorbar(im, ax=ax, label='Significant')

        # Mark the significant grids with a backslash
        significant_indices = np.where(p_value < significance_level)
        for lon, lat in zip(lons[significant_indices], lats[significant_indices]):
            ax.text(lon, lat, '\\', fontsize=12, ha='center', va='center', transform=ccrs.PlateCarree())

        # Set the title and show the plot
        plt.title('Significant Grids')
        plt.show()
    '''  
        


def mean_rainfall_14_season(obs_dataset, obs_name, model_datasets, model_names, workdir):
    
    #no EW 14-3 
    #1 obs 9 mod 4 we
    musim='DJF'
    file_name='mean_rainfall_Taylor_'+musim
        
    
    fig, axes = plt.subplots(nrows=3, ncols=5,figsize=(4,6))
    Tmax = 15; Tmin =0 ; delT = 3
    clevs = np.arange(Tmin,Tmax+delT,delT)
    
   
    cmap2='rainbow'
    x_tick=['J','F','M','A','M','J','J','A','S','O','N','D']
    
    ds = xr.DataArray(obs_dataset.values,
    coords={'time': obs_dataset.times,
    'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
    dims=["time", "lat", "lon"])
        
    #season
    dss = ds.groupby('time.season')
    ds0= dss[musim].mean(dim='time')
     
        
    plot='SEA'
    
    if plot=='SEA':
        lat_min1 = 0.22*3
        lat_max1 = 0.22*6
        lon_min1 = 0.22*4
        lon_max1 = 0.22*3
    
    if plot=='sum':
        lat_min1 = 0.22*1
        lat_max1 = 0.22*3
        lon_min1 = 0.22*3
        lon_max1 = 0.22*1
    
    
    
    lat=ds.lat
    lon=ds.lon
    lat_min = ds.lat.min()
    lat_max = ds.lat.max()
    lon_min = ds.lon.min()
    lon_max = ds.lon.max()
    x,y = np.meshgrid(ds.lon, ds.lat)
    
    m = Basemap(ax=axes[0,0], projection ='cyl', 
                llcrnrlat = lat_min+lat_min1, urcrnrlat = lat_max-lat_max1,
                llcrnrlon = lon_min+lon_min1, urcrnrlon = lon_max-lon_max1, 
                resolution = 'l', fix_aspect=False)
    m.drawcoastlines(linewidth=1)
    m.drawcountries(linewidth=1)
    m.drawstates(linewidth=0.5, color='w')

    
    max = axes[0,0].contourf(lon, lat, 
                      ds0, 
                      levels=clevs, extend='both',
                      cmap=cmap2)
    axes[0,0].set_title(obs_name,pad=3, fontsize=9)
    axes[0,0].set_xticks([])
    axes[0,0].set_xticklabels([])
    for i in [0,1,2]:
        axes[i,0].set_yticks([-10,0,10,20])   
        axes[i,0].set_yticklabels(['10$^\circ$S','EQ','10$^\circ$N','20$^\circ$N'])
        axes[i,0].tick_params(axis='y', pad=1,labelsize=7)
    
    
    axes[0,0].set_ylabel(musim,fontsize=8)
    axes[1,0].set_ylabel(musim,fontsize=8)
    axes[2,0].set_ylabel(musim,fontsize=8)
   
    
    from scipy.stats import pearsonr
  
    sd1=ds0.std(skipna=None) 
    #print('sd1 =', sd1.values )
    mbz=ma.zeros(len(model_datasets)) 
    r0=1
    for i in np.arange(len(model_datasets)):
        #print(i)
        #if i<6: axes[0,1+i].axis('off')
        ds = xr.DataArray(model_datasets[i].values,
        coords={'time': obs_dataset.times,
        'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
        dims=["time", "lat", "lon"])
        
        dss= ds.groupby('time.season')
        dsi= dss[musim].mean(dim='time')
        
        c=xr.corr(ds0,dsi)
        sd2=dsi.std(skipna=True)
        s=sd2/sd1
        
        #T= (4*(1+c)**4)/(((s+1/s)**2)*((1+r0)**4))
        # Tanpa pakai **4 T naik dikit
        T= (4*(1+c))/(((s+1/s)**2)*((1+r0)))
     
        mbz[i]=T
        #-----------------------------------
        
        if i==0: 
            ds11=dsi
            sd11=sd2
        
        c=xr.corr(ds11,dsi)
        sd2=dsi.std(skipna=True)
        s=sd2/sd11        
        T2= (4*(1+c))/(((s+1/s)**2)*((1+r0)))
        
        '''
        # T2 diganti Tian
        ds=ds0
        
        vr= ds.var() #mstats.tvar(ds.data)
        vf= dsi.var() #vf=mstats.tvar(dsi.data)
        #(ma.mean((calc_bias(target_array, reference_array))**2))**0.5
        #rmse=sqrt(mean(m-o)^2)
        rmse=(((dsi-ds)**2).mean())**0.5
        b=dsi.mean()-ds.mean()
        mse=rmse.data**2
        b=b.data
        #print(r,b) #,vr,vf)
        #tambahkan Tian
        #Tian=(1+c)/2*(1-(r**2/(b**2+vr+vf)))
        T2=(1+c)/2*(1-(mse/(b**2+vr+vf)))
        #T.append(np.round(Tian.data,4))
        '''
        
        if 0<=i<4:
            m = Basemap(ax=axes[0,i+1], projection ='cyl', 
                llcrnrlat = lat_min+lat_min1, urcrnrlat = lat_max-lat_max1,
                llcrnrlon = lon_min+lon_min1, urcrnrlon = lon_max-lon_max1, 
                resolution = 'l', fix_aspect=False)
            m.drawcoastlines(linewidth=1)
            m.drawcountries(linewidth=1)
            m.drawstates(linewidth=0.5, color='w')
        
            cax = axes[0,i+1].contourf(lon, lat, 
                      dsi, 
                      levels=clevs, extend='both',
                      cmap=cmap2)
            axes[0,i+1].set_xticks([])
            axes[0,i+1].set_xticklabels([])
            if i==0:
                axes[0,i+1].set_title(model_names[i]+' ('+'%.2f'%mbz[i]+')',pad=3,fontsize=9)
            else:
                axes[0,i+1].set_title(model_names[i]+' ('+'%.2f'%mbz[i]+')'+'('+'%.2f'%T2+')',pad=3,fontsize=9)
            
          
            axes[0,1+i].set_yticks([])
        
        
        if 9>i>=4:
            m = Basemap(ax=axes[1,i-4], projection ='cyl', 
                llcrnrlat = lat_min+lat_min1, urcrnrlat = lat_max-lat_max1,
                llcrnrlon = lon_min+lon_min1, urcrnrlon = lon_max-lon_max1, 
                resolution = 'l', fix_aspect=False)
            m.drawcoastlines(linewidth=1)
            m.drawcountries(linewidth=1)
            m.drawstates(linewidth=0.5, color='w')
            
            cax = axes[1,i-4].contourf(lon, lat, 
                      dsi, 
                      levels=clevs, extend='both',
                      cmap=cmap2)
            #axes[1,i-4].set_xticks([100,120,140])
            #axes[1,i-4].xaxis.set_tick_params(labelsize=6)
            axes[1,i-4].set_title(model_names[i]+' ('+'%.2f'%mbz[i]+')'+'('+'%.2f'%T2+')',pad=3,fontsize=9)
            axes[1,0].set_yticks([-10,0,10,20])
            #axes[1,0].yaxis.set_tick_params(labelsize=6)
            axes[1,0].set_yticklabels(['10S','EQ','10N','20N'], fontsize=6)
            axes[1,i-4].set_xticks([])
            if i>4: axes[1,i-4].set_yticks([])                     
    
        if i>=9:
            m = Basemap(ax=axes[2,i-9], projection ='cyl', 
                llcrnrlat = lat_min+lat_min1, urcrnrlat = lat_max-lat_max1,
                llcrnrlon = lon_min+lon_min1, urcrnrlon = lon_max-lon_max1, 
                resolution = 'l', fix_aspect=False)
            m.drawcoastlines(linewidth=1)
            m.drawcountries(linewidth=1)
            m.drawstates(linewidth=0.5, color='w')
        
            
            cax = axes[2,i-9].contourf(lon, lat, 
                      dsi, 
                      levels=clevs, extend='both',
                      cmap=cmap2)
            axes[2,i-9].set_xticks([100,120,140])
            #axes[2,i-9].xaxis.set_tick_params(labelsize=6)
            axes[2,i-9].set_xticklabels(['100$^\circ$E','120$^\circ$E','140$^\circ$E'], fontsize=7)
            axes[2,i-9].tick_params(axis='x', pad=1,labelsize=7)   
            axes[2,i-9].set_title(model_names[i]+' ('+'%.2f'%mbz[i]+')'+'('+'%.2f'%T2+')',pad=3,fontsize=9)
            axes[2,0].set_yticks([-10,0,10,20])
            #axes[2,0].yaxis.set_tick_params(labelsize=6)
            axes[2,0].set_yticklabels(['10S','EQ','10N','20N'], fontsize=6)
           
            if i>9: axes[2,i-9].set_yticks([])
 
    print('S_Taylor zonal mean=',mbz)  
    import pandas as pd
    df = pd.DataFrame([model_names, mbz])
    df.T.to_excel(workdir+reg+'mean_pre_'+musim+'_Tian.xlsx', index=False, header=False) 
    axes[2,3].set_yticks([])
    axes[2,4].set_yticks([])
    #axes[2,3].set_xticks([])
    #axes[2,4].set_xticks([])  
    for i in [0,1,2]:
        axes[i,0].set_yticks([-10,0,10,20])   
        axes[i,0].set_yticklabels(['10$^\circ$S','EQ','10$^\circ$N','20$^\circ$N'])
        axes[i,0].tick_params(axis='y', pad=1,labelsize=7)
        
    #The dimensions [left, bottom, width, height] of the new axes
    cax = fig.add_axes([0.91, 0.5, 0.015, 0.4])
    cax.tick_params(labelsize=7)
    plt.colorbar(max, cax = cax) 
    #error ini
    #plt.colorbar(cax).ax.set_title('mm/month')
    
    plt.subplots_adjust(hspace=.2,wspace=.05)
    #plt.tight_layout()
    #plt.title(file_name, y=0, x=0)
    
    
    
    plt.show()
    fig.savefig(workdir+file_name+reg+musim,dpi=300,bbox_inches='tight')
    return mbz

def mean_rainfall_15_season(obs_dataset, obs_name, model_datasets, model_names, workdir):
    
    #no EW 14-3 
    #1 obs 9 mod 4 we
    musim='JJA'
    file_name='mean_rainfall_Taylor_'+musim
        
    
    fig, axes = plt.subplots(nrows=4, ncols=5,figsize=(4,6))
    Tmax = 15; Tmin =0 ; delT = 3
    clevs = np.arange(Tmin,Tmax+delT,delT)
    
   
    cmap2='rainbow'
    x_tick=['J','F','M','A','M','J','J','A','S','O','N','D']
    
    ds = xr.DataArray(obs_dataset.values,
    coords={'time': obs_dataset.times,
    'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
    dims=["time", "lat", "lon"])
        
    #season
    dss = ds.groupby('time.season')
    ds0= dss[musim].mean(dim='time')
     
        
    plot='SEA'
    
    if plot=='SEA':
        lat_min1 = 0.22*3
        lat_max1 = 0.22*6
        lon_min1 = 0.22*4
        lon_max1 = 0.22*3
    
    if plot=='sum':
        lat_min1 = 0.22*1
        lat_max1 = 0.22*3
        lon_min1 = 0.22*3
        lon_max1 = 0.22*1
    
    
    
    lat=ds.lat
    lon=ds.lon
    lat_min = ds.lat.min()
    lat_max = ds.lat.max()
    lon_min = ds.lon.min()
    lon_max = ds.lon.max()
    x,y = np.meshgrid(ds.lon, ds.lat)
    
    m = Basemap(ax=axes[0,0], projection ='cyl', 
                llcrnrlat = lat_min+lat_min1, urcrnrlat = lat_max-lat_max1,
                llcrnrlon = lon_min+lon_min1, urcrnrlon = lon_max-lon_max1, 
                resolution = 'l', fix_aspect=False)
    m.drawcoastlines(linewidth=1)
    m.drawcountries(linewidth=1)
    m.drawstates(linewidth=0.5, color='w')

    
    max = axes[0,0].contourf(lon, lat, 
                      ds0, 
                      levels=clevs, extend='both',
                      cmap=cmap2)
    axes[0,0].set_title(obs_name,pad=3, fontsize=9)
    axes[0,0].set_xticks([])
    axes[0,0].set_xticklabels([])
    for i in [0,1,2]:
        axes[i,0].set_yticks([-10,0,10,20])   
        axes[i,0].set_yticklabels(['10$^\circ$S','EQ','10$^\circ$N','20$^\circ$N'])
        axes[i,0].tick_params(axis='y', pad=1,labelsize=7)
    
    
    from scipy.stats import pearsonr
  
    sd1=ds0.std(skipna=None) 
    #print('sd1 =', sd1.values )
    mbz=ma.zeros(len(model_datasets)) 
    r0=1
    for i in np.arange(len(model_datasets)):
        print(i)
        if i<4: axes[0,1+i].axis('off')
        ds = xr.DataArray(model_datasets[i].values,
        coords={'time': obs_dataset.times,
        'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
        dims=["time", "lat", "lon"])
        
        dss= ds.groupby('time.season')
        dsi= dss[musim].mean(dim='time')
        
        c=xr.corr(ds0,dsi)
        sd2=dsi.std(skipna=True)
        s=sd2/sd1
       
        T= (4*(1+c))/(((s+1/s)**2)*((1+r0)))
     
        mbz[i]=T
       
        
        if 0<=i<5:
            m = Basemap(ax=axes[1,i], projection ='cyl', 
                llcrnrlat = lat_min+lat_min1, urcrnrlat = lat_max-lat_max1,
                llcrnrlon = lon_min+lon_min1, urcrnrlon = lon_max-lon_max1, 
                resolution = 'l', fix_aspect=False)
            m.drawcoastlines(linewidth=1)
            m.drawcountries(linewidth=1)
            m.drawstates(linewidth=0.5, color='w')
        
            cax = axes[1,i].contourf(lon, lat, 
                      dsi, 
                      levels=clevs, extend='both',
                      cmap=cmap2)
            axes[1,i].set_xticks([])
            axes[1,i].set_xticklabels([])
            
            axes[1,i].set_title(model_names[i]+' ('+'%.2f'%mbz[i]+')',pad=3,fontsize=9)
            
          
            axes[1,i].set_yticks([])
        
        
        if 9>i>=4:
            m = Basemap(ax=axes[2,i-4], projection ='cyl', 
                llcrnrlat = lat_min+lat_min1, urcrnrlat = lat_max-lat_max1,
                llcrnrlon = lon_min+lon_min1, urcrnrlon = lon_max-lon_max1, 
                resolution = 'l', fix_aspect=False)
            m.drawcoastlines(linewidth=1)
            m.drawcountries(linewidth=1)
            m.drawstates(linewidth=0.5, color='w')
            
            cax = axes[2,i-4].contourf(lon, lat, 
                      dsi, 
                      levels=clevs, extend='both',
                      cmap=cmap2)
            #axes[1,i-4].set_xticks([100,120,140])
            #axes[1,i-4].xaxis.set_tick_params(labelsize=6)
            axes[2,i-4].set_title(model_names[i]+' ('+'%.2f'%mbz[i]+')',pad=3,fontsize=9)
            axes[2,0].set_yticks([-10,0,10,20])
            #axes[1,0].yaxis.set_tick_params(labelsize=6)
            axes[2,0].set_yticklabels(['10S','EQ','10N','20N'], fontsize=6)
            axes[2,i-4].set_xticks([])
            if i>4: axes[2,i-4].set_yticks([])                     
    
        if i>=9:
            m = Basemap(ax=axes[3,i-10], projection ='cyl', 
                llcrnrlat = lat_min+lat_min1, urcrnrlat = lat_max-lat_max1,
                llcrnrlon = lon_min+lon_min1, urcrnrlon = lon_max-lon_max1, 
                resolution = 'l', fix_aspect=False)
            m.drawcoastlines(linewidth=1)
            m.drawcountries(linewidth=1)
            m.drawstates(linewidth=0.5, color='w')
        
            
            cax = axes[3,i-10].contourf(lon, lat, 
                      dsi, 
                      levels=clevs, extend='both',
                      cmap=cmap2)
            axes[3,i-10].set_xticks([100,120,140])
            #axes[2,i-9].xaxis.set_tick_params(labelsize=6)
            axes[3,i-10].set_xticklabels(['100$^\circ$E','120$^\circ$E','140$^\circ$E'], fontsize=7)
            axes[3,i-10].tick_params(axis='x', pad=1,labelsize=7)   
            axes[3,i-10].set_title(model_names[i]+' ('+'%.2f'%mbz[i]+')',pad=3,fontsize=9)
            axes[3,0].set_yticks([-10,0,10,20])
            #axes[2,0].yaxis.set_tick_params(labelsize=6)
            axes[3,0].set_yticklabels(['10S','EQ','10N','20N'], fontsize=6)
           
            if i>9: axes[3,i-10].set_yticks([])
        
    print('S_Taylor zonal mean=',mbz)  
    import pandas as pd
    df = pd.DataFrame([model_names, mbz])
    df.T.to_excel(workdir+reg+'mean_pre_'+musim+'_Tian.xlsx', index=False, header=False) 
    axes[3,3].set_yticks([])
    axes[3,4].set_yticks([])
    #axes[2,3].set_xticks([])
    #axes[2,4].set_xticks([])  
    for i in [0,1,2]:
        axes[i,0].set_yticks([-10,0,10,20])   
        axes[i,0].set_yticklabels(['10$^\circ$S','EQ','10$^\circ$N','20$^\circ$N'])
        axes[i,0].tick_params(axis='y', pad=1,labelsize=7)
        
    #The dimensions [left, bottom, width, height] of the new axes
    #cax = fig.add_axes([0.91, 0.2, 0.015, 0.4])
    #cax.tick_params(labelsize=7)
    #plt.colorbar(max, cax = cax) 
    
    cax = fig.add_axes([0.35, 0.8, 0.4, 0.02]) #horisontal
    cax.tick_params(labelsize=7)
    cbar=plt.colorbar(max, cax = cax, extend='both', orientation='horizontal') 
    #cbar.ax.set_xlabel('mm/day', labelpad=7)
    cbar.ax.set_title('season '+musim+' (mm/day)')
    
    plt.subplots_adjust(hspace=.2,wspace=.05)
    #plt.tight_layout()
    #plt.title(file_name, y=0, x=0)
    
    
    
    plt.show()
    fig.savefig(workdir+file_name+reg+musim,dpi=300,bbox_inches='tight')
    return mbz

def mean_rainfall_11_season(obs_dataset, obs_name, model_datasets, model_names, workdir):
    
    #no EW 14-3
    musim='DJF'
    file_name='mean_rainfall_Taylor_'+musim
        
    
    fig, axes = plt.subplots(nrows=3, ncols=4,figsize=(4,6))
    Tmax = 15; Tmin =0 ; delT = 3
    clevs = np.arange(Tmin,Tmax+delT,delT)
    
   
    cmap2='rainbow'
    x_tick=['J','F','M','A','M','J','J','A','S','O','N','D']
    
    ds = xr.DataArray(obs_dataset.values,
    coords={'time': obs_dataset.times,
    'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
    dims=["time", "lat", "lon"])
        
    #season
    dss = ds.groupby('time.season')
    ds0= dss[musim].mean(dim='time')
     
        
    plot='SEA'
    
    if plot=='SEA':
        lat_min1 = 0.22*3
        lat_max1 = 0.22*6
        lon_min1 = 0.22*4
        lon_max1 = 0.22*3
    
    if plot=='sum':
        lat_min1 = 0.22*1
        lat_max1 = 0.22*3
        lon_min1 = 0.22*3
        lon_max1 = 0.22*1
    
    
    
    lat=ds.lat
    lon=ds.lon
    lat_min = ds.lat.min()
    lat_max = ds.lat.max()
    lon_min = ds.lon.min()
    lon_max = ds.lon.max()
    x,y = np.meshgrid(ds.lon, ds.lat)
    
    m = Basemap(ax=axes[0,0], projection ='cyl', 
                llcrnrlat = lat_min+lat_min1, urcrnrlat = lat_max-lat_max1,
                llcrnrlon = lon_min+lon_min1, urcrnrlon = lon_max-lon_max1, 
                resolution = 'l', fix_aspect=False)
    m.drawcoastlines(linewidth=1)
    m.drawcountries(linewidth=1)
    m.drawstates(linewidth=0.5, color='w')

    
    max = axes[0,0].contourf(lon, lat, 
                      ds0, 
                      levels=clevs, extend='both',
                      cmap=cmap2)
    axes[0,0].set_title(obs_name,pad=3, fontsize=9)
    axes[0,0].set_xticks([])
    axes[0,0].set_xticklabels([])
    for i in [0,1,2]:
        axes[i,0].set_yticks([-10,0,10,20])   
        axes[i,0].set_yticklabels(['10$^\circ$S','EQ','10$^\circ$N','20$^\circ$N'])
        axes[i,0].tick_params(axis='y', pad=1,labelsize=7)
    
    
    axes[0,0].set_ylabel(musim,fontsize=8)
    axes[1,0].set_ylabel(musim,fontsize=8)
    axes[2,0].set_ylabel(musim,fontsize=8)
   
    
    from scipy.stats import pearsonr
  
    sd1=ds0.std(skipna=None) 
    #print('sd1 =', sd1.values )
    s1=[]
    s2=[]
    r0=1
    for i in np.arange(len(model_datasets)):
        #print(i)
        #if i<6: axes[0,1+i].axis('off')
        ds = xr.DataArray(model_datasets[i].values,
        coords={'time': obs_dataset.times,
        'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
        dims=["time", "lat", "lon"])
        
        dss= ds.groupby('time.season')
        dsi= dss[musim].mean(dim='time')
        
        c=xr.corr(ds0,dsi)
        sd2=dsi.std(skipna=True)
        s=sd2/sd1
        
        #T= (4*(1+c)**4)/(((s+1/s)**2)*((1+r0)**4))
        # Tanpa pakai **4 T naik dikit
        T1= (4*(1+c))/(((s+1/s)**2)*((1+r0)))
        s1.append(np.round(T1.values,2))
        #mbz[i]=T
        #-----------------------------------
        
        if i==0: 
            ds11=dsi
            sd11=sd2
        
        c=xr.corr(ds11,dsi)
        sd2=dsi.std(skipna=True)
        s=sd2/sd11        
        T2= (4*(1+c))/(((s+1/s)**2)*((1+r0)))
        s2.append(np.round(T2.values,2))
        
        
        #012
        if 0<=i<3:
            m = Basemap(ax=axes[0,i+1], projection ='cyl', 
                llcrnrlat = lat_min+lat_min1, urcrnrlat = lat_max-lat_max1,
                llcrnrlon = lon_min+lon_min1, urcrnrlon = lon_max-lon_max1, 
                resolution = 'l', fix_aspect=False)
            m.drawcoastlines(linewidth=1)
            m.drawcountries(linewidth=1)
            m.drawstates(linewidth=0.5, color='w')
        
            cax = axes[0,i+1].contourf(lon, lat, 
                      dsi, 
                      levels=clevs, extend='both',
                      cmap=cmap2)
            axes[0,i+1].set_xticks([])
            axes[0,i+1].set_xticklabels([])
            if i==0:
                axes[0,i+1].set_title(model_names[i]+' ('+'%.2f'%T1+')',pad=3,fontsize=9)
            else:
                axes[0,i+1].set_title(model_names[i]+' ('+'%.2f'%T1+')'+'('+'%.2f'%T2+')',pad=3,fontsize=9)
            
          
            axes[0,1+i].set_yticks([])
        
        #3456
        if 7>i>=3:
            m = Basemap(ax=axes[1,i-3], projection ='cyl', 
                llcrnrlat = lat_min+lat_min1, urcrnrlat = lat_max-lat_max1,
                llcrnrlon = lon_min+lon_min1, urcrnrlon = lon_max-lon_max1, 
                resolution = 'l', fix_aspect=False)
            m.drawcoastlines(linewidth=1)
            m.drawcountries(linewidth=1)
            m.drawstates(linewidth=0.5, color='w')
            
            cax = axes[1,i-3].contourf(lon, lat, 
                      dsi, 
                      levels=clevs, extend='both',
                      cmap=cmap2)
            #axes[1,i-4].set_xticks([100,120,140])
            #axes[1,i-4].xaxis.set_tick_params(labelsize=6)
            axes[1,i-3].set_title(model_names[i]+' ('+'%.2f'%T1+')'+'('+'%.2f'%T2+')',pad=3,fontsize=9)
            axes[1,0].set_yticks([-10,0,10,20])
            #axes[1,0].yaxis.set_tick_params(labelsize=6)
            axes[1,0].set_yticklabels(['10S','EQ','10N','20N'], fontsize=6)
            axes[1,i-3].set_xticks([])
            if i>3: axes[1,i-3].set_yticks([])                     
        n=7
        #78910
        if 11>i>=n:
            print(i)
            m = Basemap(ax=axes[2,i-n], projection ='cyl', 
                llcrnrlat = lat_min+lat_min1, urcrnrlat = lat_max-lat_max1,
                llcrnrlon = lon_min+lon_min1, urcrnrlon = lon_max-lon_max1, 
                resolution = 'l', fix_aspect=False)
            m.drawcoastlines(linewidth=1)
            m.drawcountries(linewidth=1)
            m.drawstates(linewidth=0.5, color='w')
        
            
            cax = axes[2,i-n].contourf(lon, lat, 
                      dsi, 
                      levels=clevs, extend='both',
                      cmap=cmap2)
            axes[2,i-n].set_xticks([100,120,140])
            #axes[2,i-9].xaxis.set_tick_params(labelsize=6)
            axes[2,i-n].set_xticklabels(['100$^\circ$E','120$^\circ$E','140$^\circ$E'], fontsize=7)
            axes[2,i-n].tick_params(axis='x', pad=1,labelsize=7)   
            axes[2,i-n].set_title(model_names[i]+' ('+'%.2f'%T1+')'+'('+'%.2f'%T2+')',pad=3,fontsize=9)
            axes[2,0].set_yticks([-10,0,10,20])
            #axes[2,0].yaxis.set_tick_params(labelsize=6)
            axes[2,0].set_yticklabels(['10S','EQ','10N','20N'], fontsize=6)
           
            if i>n: axes[2,i-n].set_yticks([])
 
    print('S_Taylor zonal mean=',s1,s2)  
    import pandas as pd
    df = pd.DataFrame([model_names, s1,s2])
    df.T.to_excel(workdir+reg+'mean_pre_'+musim+'_Tian.xlsx', index=False, header=False) 
    axes[2,3].set_yticks([])
    #axes[2,4].set_yticks([])
    #axes[2,3].set_xticks([])
    #axes[2,4].set_xticks([])  
    for i in [0,1,2]:
        axes[i,0].set_yticks([-10,0,10,20])   
        axes[i,0].set_yticklabels(['10$^\circ$S','EQ','10$^\circ$N','20$^\circ$N'])
        axes[i,0].tick_params(axis='y', pad=1,labelsize=7)
        
    #The dimensions [left, bottom, width, height] of the new axes
    cax = fig.add_axes([0.91, 0.5, 0.015, 0.4])
  
    cbar=plt.colorbar(max, cax = cax)
    cbar.ax.tick_params(labelsize=7)
    #label pad tidak pengaruh
    cbar.ax.set_ylabel('mm/month', rotation=270, labelpad=7)
    #ini tidak nampak
    #cbar.ax.set_xlabel('mm/month', rotation=270)

    
    plt.subplots_adjust(hspace=.2,wspace=.05)
    #plt.tight_layout()
    #plt.title(file_name, y=0, x=0)
    
    
    
    plt.show()
    fig.savefig(workdir+file_name+reg+musim,dpi=300,bbox_inches='tight')
    
def mean_rainfall_11era_season(obs_dataset, obs_name, model_datasets, model_names, workdir):
    
    #no EW 14-3
    musim='DJF'
    file_name='mean_rainfall_Taylor_'+musim
        
    
    fig, axes = plt.subplots(nrows=3, ncols=4,figsize=(4,6))
    Tmax = 15; Tmin =0 ; delT = 3
    clevs = np.arange(Tmin,Tmax+delT,delT)
    
   
    cmap2='rainbow'
    x_tick=['J','F','M','A','M','J','J','A','S','O','N','D']
    
    ds = xr.DataArray(obs_dataset.values,
    coords={'time': obs_dataset.times,
    'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
    dims=["time", "lat", "lon"])
        
    #season
    dss = ds.groupby('time.season')
    ds0= dss[musim].mean(dim='time')
     
        
    plot='SEA'
    
    if plot=='SEA':
        lat_min1 = 0.22*3
        lat_max1 = 0.22*6
        lon_min1 = 0.22*4
        lon_max1 = 0.22*3
    
    if plot=='sum':
        lat_min1 = 0.22*1
        lat_max1 = 0.22*3
        lon_min1 = 0.22*3
        lon_max1 = 0.22*1
    
    
    
    lat=ds.lat
    lon=ds.lon
    lat_min = ds.lat.min()
    lat_max = ds.lat.max()
    lon_min = ds.lon.min()
    lon_max = ds.lon.max()
    x,y = np.meshgrid(ds.lon, ds.lat)
    
    m = Basemap(ax=axes[0,0], projection ='cyl', 
                llcrnrlat = lat_min+lat_min1, urcrnrlat = lat_max-lat_max1,
                llcrnrlon = lon_min+lon_min1, urcrnrlon = lon_max-lon_max1, 
                resolution = 'l', fix_aspect=False)
    m.drawcoastlines(linewidth=1)
    m.drawcountries(linewidth=1)
    m.drawstates(linewidth=0.5, color='w')

    
    max = axes[0,0].contourf(lon, lat, 
                      ds0, 
                      levels=clevs, extend='both',
                      cmap=cmap2)
    axes[0,0].set_title(obs_name,pad=3, fontsize=9)
    axes[0,0].set_xticks([])
    axes[0,0].set_xticklabels([])
    if plot=='SEA':
        axes[1,0].set_yticks([-10,0,10,20])
        #axes[1,0].yaxis.set_tick_params(labelsize=6)
        axes[1,0].set_yticklabels(['10S','EQ','10N','20N'], fontsize=6)
        for i in [0,1,2]:
            axes[i,0].set_yticks([-10,0,10,20])   
            axes[i,0].set_yticklabels(['10$^\circ$S','EQ','10$^\circ$N','20$^\circ$N'])
            axes[i,0].tick_params(axis='y', pad=1,labelsize=7)
    
    
    #axes[0,0].set_ylabel(musim,fontsize=8)
    axes[1,0].set_ylabel('Boreal winter (DJF)',fontsize=10)
    #axes[2,0].set_ylabel(musim,fontsize=8)
   
    
    from scipy.stats import pearsonr
  
    sd1=ds0.std(skipna=None) 
    #print('sd1 =', sd1.values )
    s1=[]
    s2=[]
    r0=1
    for i in np.arange(len(model_datasets)):
        print(model_names[i])
        #if i<6: axes[0,1+i].axis('off')
        ds = xr.DataArray(model_datasets[i].values,
        coords={'time': obs_dataset.times,
        'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
        dims=["time", "lat", "lon"])
        
        dss= ds.groupby('time.season')
        dsi= dss[musim].mean(dim='time')
        
        c=xr.corr(ds0,dsi)
        sd2=dsi.std(skipna=True)
        s=sd2/sd1
        
        #T= (4*(1+c)**4)/(((s+1/s)**2)*((1+r0)**4))
        # Tanpa pakai **4 T naik dikit
        T1= (4*(1+c))/(((s+1/s)**2)*((1+r0)))
        s1.append(np.round(T1.values,2))
        #mbz[i]=T
        #-----------------------------------
      
        #012
        if 0<=i<3:
            m = Basemap(ax=axes[0,i+1], projection ='cyl', 
                llcrnrlat = lat_min+lat_min1, urcrnrlat = lat_max-lat_max1,
                llcrnrlon = lon_min+lon_min1, urcrnrlon = lon_max-lon_max1, 
                resolution = 'l', fix_aspect=False)
            m.drawcoastlines(linewidth=1)
            m.drawcountries(linewidth=1)
            m.drawstates(linewidth=0.5, color='w')
        
            cax = axes[0,i+1].contourf(lon, lat, 
                      dsi, 
                      levels=clevs, extend='both',
                      cmap=cmap2)
            axes[0,i+1].set_xticks([])
            axes[0,i+1].set_xticklabels([])
          
            #axes[0,i+1].set_title(model_names[i], pad=3,fontsize=9)
            axes[0,i+1].set_title(model_names[i]+' ('+'%.2f'%T1+')',pad=3,fontsize=9)
          
            axes[0,1+i].set_yticks([])
        
        #3456
        if 7>i>=3:
            m = Basemap(ax=axes[1,i-3], projection ='cyl', 
                llcrnrlat = lat_min+lat_min1, urcrnrlat = lat_max-lat_max1,
                llcrnrlon = lon_min+lon_min1, urcrnrlon = lon_max-lon_max1, 
                resolution = 'l', fix_aspect=False)
            m.drawcoastlines(linewidth=1)
            m.drawcountries(linewidth=1)
            m.drawstates(linewidth=0.5, color='w')
            
            cax = axes[1,i-3].contourf(lon, lat, 
                      dsi, 
                      levels=clevs, extend='both',
                      cmap=cmap2)
            #axes[1,i-4].set_xticks([100,120,140])
            #axes[1,i-4].xaxis.set_tick_params(labelsize=6)
            axes[1,i-3].set_title(model_names[i]+' ('+'%.2f'%T1+')',pad=3,fontsize=9)
            
            
            axes[1,i-3].set_xticks([])
            if i>3: axes[1,i-3].set_yticks([])                     
        n=7
        #78910
        if 11>i>=n:
            print(i)
            m = Basemap(ax=axes[2,i-n], projection ='cyl', 
                llcrnrlat = lat_min+lat_min1, urcrnrlat = lat_max-lat_max1,
                llcrnrlon = lon_min+lon_min1, urcrnrlon = lon_max-lon_max1, 
                resolution = 'l', fix_aspect=False)
            m.drawcoastlines(linewidth=1)
            m.drawcountries(linewidth=1)
            m.drawstates(linewidth=0.5, color='w')
        
            
            cax = axes[2,i-n].contourf(lon, lat, 
                      dsi, 
                      levels=clevs, extend='both',
                      cmap=cmap2)
            axes[2,i-n].set_title(model_names[i]+' ('+'%.2f'%T1+')',pad=3,fontsize=9)
            if plot=='SEA':
                axes[2,i-n].set_xticks([100,120,140])
                #axes[2,i-9].xaxis.set_tick_params(labelsize=6)
                axes[2,i-n].set_xticklabels(['100$^\circ$E','120$^\circ$E','140$^\circ$E'], fontsize=7)
                axes[2,i-n].tick_params(axis='x', pad=1,labelsize=7)   
                
                axes[2,0].set_yticks([-10,0,10,20])
                #axes[2,0].yaxis.set_tick_params(labelsize=6)
                axes[2,0].set_yticklabels(['10S','EQ','10N','20N'], fontsize=6)
           
            if i>n: axes[2,i-n].set_yticks([])
 
    print('S_Taylor zonal mean=',s1)  
    import pandas as pd
    df = pd.DataFrame([model_names, s1])
    df.T.to_excel(workdir+reg+'mean_pre_'+musim+'.xlsx', index=False, header=False) 
    axes[2,3].set_yticks([])
    #axes[2,4].set_yticks([])
    axes[2,3].axis('off')
    #axes[2,4].set_xticks([])  
    
        
    #cax = fig.add_axes([0.91, 0.5, 0.015, 0.4])
    #cbar=plt.colorbar(max, cax = cax, )
    cax = fig.add_axes([0.715, 0.25, 0.19, 0.023])
    #cax = fig.add_axes([0.5, 0.97, 0.6, 0.02])
    cbar=plt.colorbar(max, cax = cax, orientation='horizontal')
    
    cbar.ax.tick_params(labelsize=7)
    #label pad tidak pengaruh
    cbar.ax.set_xlabel('mm/day', labelpad=7)
   

    
    plt.subplots_adjust(hspace=.2,wspace=.05)
    #plt.tight_layout()
    #plt.title(file_name, y=0, x=0)
    
    
    
    plt.show()
    fig.savefig(workdir+file_name+reg+musim,dpi=300,bbox_inches='tight')    

def mean_rainfall_11_jambi(obs_dataset, obs_name, model_datasets, model_names, workdir):
    
    #no EW 14-3
    musim='JJA'
    file_name='mean_rainfall_Taylor_'+musim
        
    
    fig, axes = plt.subplots(nrows=3, ncols=4,figsize=(4,6))
    Tmax = 15; Tmin =0 ; delT = 3
    clevs = np.arange(Tmin,Tmax+delT,delT)
    
   
    cmap2='rainbow'
    x_tick=['J','F','M','A','M','J','J','A','S','O','N','D']
    
    ds = xr.DataArray(obs_dataset.values,
    coords={'time': obs_dataset.times,
    'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
    dims=["time", "lat", "lon"])
    
    ds = ds.where(
                    (ds.lat > -3) & (ds.lat < -0.5) & 
                    (ds.lon > 99) & (ds.lon < 106), drop=True)
      
    #season
    dss = ds.groupby('time.season')
    ds0= dss[musim].mean(dim='time')
     
        
    plot='jambi'
    
    if plot=='SEA':
        lat_min1 = 0.22*3
        lat_max1 = 0.22*6
        lon_min1 = 0.22*4
        lon_max1 = 0.22*3
    
    if plot=='sum':
        lat_min1 = 0.22*1
        lat_max1 = 0.22*3
        lon_min1 = 0.22*3
        lon_max1 = 0.22*1
    
    if plot=='jambi':
        lat_min1 = 0
        lat_max1 = 0
        lon_min1 = 0
        lon_max1 = 0
    
    lat=ds.lat
    lon=ds.lon
    lat_min = ds.lat.min()
    lat_max = ds.lat.max()
    lon_min = ds.lon.min()
    lon_max = ds.lon.max()
    x,y = np.meshgrid(ds.lon, ds.lat)
    
    m = Basemap(ax=axes[0,0], projection ='cyl', 
                llcrnrlat = lat_min+lat_min1, urcrnrlat = lat_max-lat_max1,
                llcrnrlon = lon_min+lon_min1, urcrnrlon = lon_max-lon_max1, 
                resolution = 'h', fix_aspect=False)
    m.drawcoastlines(linewidth=1)
    m.drawcountries(linewidth=1)
    m.drawstates(linewidth=0.5, color='w')

    
    max = axes[0,0].contourf(lon, lat, 
                      ds0, 
                      levels=clevs, extend='both',
                      cmap=cmap2)
    axes[0,0].set_title(obs_name,pad=3, fontsize=9)
    axes[0,0].set_xticks([])
    axes[0,0].set_xticklabels([])
    
            
    #axes[0,0].set_ylabel(musim,fontsize=8)
    #axes[1,0].set_ylabel('Boreal winter (DJF)',fontsize=10)
    #axes[2,0].set_ylabel(musim,fontsize=8)
   
    
    from scipy.stats import pearsonr
  
    sd1=ds0.std(skipna=None) 
    #print('sd1 =', sd1.values )
    s1=[]
    s2=[]
    r0=1
    for i in np.arange(len(model_datasets)):
        #print(i)
        #if i<6: axes[0,1+i].axis('off')
        ds = xr.DataArray(model_datasets[i].values,
        coords={'time': obs_dataset.times,
        'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
        dims=["time", "lat", "lon"])
        
        ds = ds.where(
                    (ds.lat > -3) & (ds.lat < -0.5) & 
                    (ds.lon > 99) & (ds.lon < 106), drop=True)
        
        dss= ds.groupby('time.season')
        dsi= dss[musim].mean(dim='time')
        
        c=xr.corr(ds0,dsi)
        sd2=dsi.std(skipna=True)
        s=sd2/sd1
        
        #T= (4*(1+c)**4)/(((s+1/s)**2)*((1+r0)**4))
        # Tanpa pakai **4 T naik dikit
        T1= (4*(1+c))/(((s+1/s)**2)*((1+r0)))
        s1.append(np.round(T1.values,2))
        #mbz[i]=T
        #-----------------------------------
      
        #012
        if 0<=i<3:
            m = Basemap(ax=axes[0,i+1], projection ='cyl', 
                llcrnrlat = lat_min+lat_min1, urcrnrlat = lat_max-lat_max1,
                llcrnrlon = lon_min+lon_min1, urcrnrlon = lon_max-lon_max1, 
                resolution = 'h', fix_aspect=False)
            m.drawcoastlines(linewidth=1)
            m.drawcountries(linewidth=1)
            m.drawstates(linewidth=0.5, color='w')
        
            cax = axes[0,i+1].contourf(lon, lat, 
                      dsi, 
                      levels=clevs, extend='both',
                      cmap=cmap2)
            axes[0,i+1].set_xticks([])
            axes[0,i+1].set_xticklabels([])
          
            #axes[0,i+1].set_title(model_names[i], pad=3,fontsize=9)
            axes[0,i+1].set_title(model_names[i]+' ('+'%.2f'%T1+')',pad=3,fontsize=9)
          
            axes[0,1+i].set_yticks([])
        
        #3456
        if 7>i>=3:
            m = Basemap(ax=axes[1,i-3], projection ='cyl', 
                llcrnrlat = lat_min+lat_min1, urcrnrlat = lat_max-lat_max1,
                llcrnrlon = lon_min+lon_min1, urcrnrlon = lon_max-lon_max1, 
                resolution = 'h', fix_aspect=False)
            m.drawcoastlines(linewidth=1)
            m.drawcountries(linewidth=1)
            m.drawstates(linewidth=0.5, color='w')
            
            cax = axes[1,i-3].contourf(lon, lat, 
                      dsi, 
                      levels=clevs, extend='both',
                      cmap=cmap2)
            #axes[1,i-4].set_xticks([100,120,140])
            #axes[1,i-4].xaxis.set_tick_params(labelsize=6)
            axes[1,i-3].set_title(model_names[i]+' ('+'%.2f'%T1+')',pad=3,fontsize=9)
            
            
            axes[1,i-3].set_xticks([])
            #if i>4: axes[1,i-3].set_yticks([])                     
        n=7
        #78910
        if 11>i>=n:
            print(i)
            m = Basemap(ax=axes[2,i-n], projection ='cyl', 
                llcrnrlat = lat_min+lat_min1, urcrnrlat = lat_max-lat_max1,
                llcrnrlon = lon_min+lon_min1, urcrnrlon = lon_max-lon_max1, 
                resolution = 'h', fix_aspect=False)
            m.drawcoastlines(linewidth=1)
            m.drawcountries(linewidth=1)
            m.drawstates(linewidth=0.5, color='w')
        
            
            cax = axes[2,i-n].contourf(lon, lat, 
                      dsi, 
                      levels=clevs, extend='both',
                      cmap=cmap2)
            axes[2,i-n].set_title(model_names[i]+' ('+'%.2f'%T1+')',pad=3,fontsize=9)
            if plot=='SEA':
                axes[2,i-n].set_xticks([100,120,140])
                #axes[2,i-9].xaxis.set_tick_params(labelsize=6)
                axes[2,i-n].set_xticklabels(['100$^\circ$E','120$^\circ$E','140$^\circ$E'], fontsize=7)
                axes[2,i-n].tick_params(axis='x', pad=1,labelsize=7)   
                
                axes[2,0].set_yticks([-10,0,10,20])
                #axes[2,0].yaxis.set_tick_params(labelsize=6)
                axes[2,0].set_yticklabels(['10S','EQ','10N','20N'], fontsize=6)
           
            if plot=='jambi':
                axes[2,i-n].set_xticks([100,102,104])
                #axes[2,i-9].xaxis.set_tick_params(labelsize=6)
                axes[2,i-n].set_xticklabels(['100$^\circ$E','102$^\circ$E','104$^\circ$E'], fontsize=7)
                axes[2,i-n].tick_params(axis='x', pad=1,labelsize=7)   
                
                
            
            if i>n: axes[2,i-n].set_yticks([])
    
    if plot=='SEA':
        axes[1,0].set_yticks([-10,0,10,20])
        #axes[1,0].yaxis.set_tick_params(labelsize=6)
        axes[1,0].set_yticklabels(['10S','EQ','10N','20N'], fontsize=6)
        for i in [0,1,2]:
            axes[i,0].set_yticks([-10,0,10,20])   
            axes[i,0].set_yticklabels(['10$^\circ$S','EQ','10$^\circ$N','20$^\circ$N'])
            axes[i,0].tick_params(axis='y', pad=1,labelsize=7)
    
    if plot=='jambi':
       
        for i in [0,1,2]:
            #print('i=',i)
            axes[i,0].set_yticks([-3, -2,-1])   
            axes[i,0].set_yticklabels(['3$^\circ$S','2$^\circ$S','1$^\circ$S'])
            axes[i,0].tick_params(axis='y', pad=1,labelsize=7)
    
    print('S_Taylor zonal mean=',s1)  
    import pandas as pd
    df = pd.DataFrame([model_names, s1])
    df.T.to_excel(workdir+reg+'mean_pre_'+musim+'.xlsx', index=False, header=False) 
    axes[2,3].set_yticks([])
    #axes[2,4].set_yticks([])
    axes[2,3].axis('off')
    #axes[2,4].set_xticks([])  
    
        
    #cax = fig.add_axes([0.91, 0.5, 0.015, 0.4])
    #cbar=plt.colorbar(max, cax = cax, )
    cax = fig.add_axes([0.715, 0.25, 0.19, 0.023])
    #cax = fig.add_axes([0.5, 0.97, 0.6, 0.02])
    cbar=plt.colorbar(max, cax = cax, orientation='horizontal')
    
    cbar.ax.tick_params(labelsize=7)
    #label pad tidak pengaruh
    cbar.ax.set_xlabel('mm/day', labelpad=7)
       
    plt.subplots_adjust(hspace=.2,wspace=.05)
    #plt.tight_layout()
    #plt.title(file_name, y=0, x=0)
       
    
    plt.show()
    fig.savefig(workdir+file_name+reg+musim,dpi=300,bbox_inches='tight')    
  
def mean_rainfall_14_season_bias(obs_dataset, obs_name, model_datasets, model_names, workdir):
  
    musim='JJA'
    file_name='mean_rainfall_Taylor_'+musim
        
    
    fig, axes = plt.subplots(nrows=3, ncols=5,figsize=(4,6))
    Tmax = 15; Tmin =0 ; delT = 3
    clevs = np.arange(Tmin,Tmax+delT,delT)
    
   
    cmap2='rainbow'
    x_tick=['J','F','M','A','M','J','J','A','S','O','N','D']
    
    ds = xr.DataArray(obs_dataset.values,
    coords={'time': obs_dataset.times,
    'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
    dims=["time", "lat", "lon"])
        
    #season
    dss = ds.groupby('time.season')
    ds0= dss[musim].mean(dim='time')
     
        
    plot='sum'
    
    if plot=='SEA':
        lat_min1 = 0.22*3
        lat_max1 = 0.22*6
        lon_min1 = 0.22*4
        lon_max1 = 0.22*3
    
    if plot=='sum':
        lat_min1 = 0.22*1
        lat_max1 = 0.22*3
        lon_min1 = 0.22*3
        lon_max1 = 0.22*1
  
    lat_min = ds.lat.min()
    lat_max = ds.lat.max()
    lon_min = ds.lon.min()
    lon_max = ds.lon.max()
    x,y = np.meshgrid(ds.lon, ds.lat)
    
    m = Basemap(ax=axes[0,0], projection ='cyl', 
                llcrnrlat = lat_min+lat_min1, urcrnrlat = lat_max-lat_max1,
                llcrnrlon = lon_min+lon_min1, urcrnrlon = lon_max-lon_max1, 
                resolution = 'l', fix_aspect=False)
    m.drawcoastlines(linewidth=1)
    m.drawcountries(linewidth=1)
    m.drawstates(linewidth=0.5, color='w')

    
    max1 = m.contourf(x, y, 
                      ds0, 
                      levels=clevs, extend='both',
                      cmap=cmap2)
    axes[0,0].set_title(obs_name,pad=4, fontsize=9)
    axes[0,0].set_xticks([])
    axes[0,0].set_xticklabels([])
    
    axes[0,0].set_yticks([-10,0,10,20])
           
    axes[0,0].set_yticklabels(['10$^\circ$S','EQ','10$^\circ$N','20$^\circ$N'])
    
    axes[0,0].tick_params(axis='y', pad=1,labelsize=7)
    axes[1,0].tick_params(axis='y', pad=1,labelsize=7)
    axes[2,0].tick_params(axis='y', pad=1,labelsize=7)
    
    #axes[0,0].set_ylabel(musim,fontsize=8)
    axes[1,0].set_ylabel(musim,fontsize=8)
    #axes[2,0].set_ylabel(musim,fontsize=8)
   
    from scipy.stats import pearsonr
  
    sd1=ds0.std(skipna=None) 
    #print('sd1 =', sd1.values )
    mbz=ma.zeros(len(model_datasets)) 
    pb=[]
    pcc=[]
    t=[]
    
    r0=1
    cmap2='RdBu_r'
    Tmax = 15; Tmin =-15 ; delT = 3
    clevs = np.arange(Tmin,Tmax+delT,delT)
    for i in np.arange(len(model_datasets)):
        print(i)
        #if i<6: axes[0,1+i].axis('off')
        ds = xr.DataArray(model_datasets[i].values,
        coords={'time': obs_dataset.times,
        'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
        dims=["time", "lat", "lon"])
        
        dss= ds.groupby('time.season')
        dsi= dss[musim].mean(dim='time')
        
        #percent bias = (model mean – observed mean) / |model mean|  x100%
        bias=(dsi.mean() - ds0.mean())/abs(dsi.mean()) *100
        
        
        dbi=dsi-ds0
        # print('dsi.mean()=',dsi.mean().data)
        #print('abs(dsi).mean()=',abs(dsi).mean().data)
        #bias=abs(dsi).mean()
       
        #jadikan GPCP baseline bias
        if i==0: 
            db0=dbi
            sd0=dbi.std(skipna=True)
            #axes[0,1].set_title(model_names[i],pad=3, fontsize=9)
        
        c=xr.corr(dbi,db0)
        sdi=dbi.std(skipna=True)
        s=sdi/sd0       
        T= (4*(1+c))/(((s+1/s)**2)*((1+r0)))
        
        pb.append(np.round(bias.data,2))
        pcc.append(np.round(c.data,2))
        t.append(np.round(T.data,2))
        
        import pandas as pd
        #df = pd.DataFrame([pb,pcc,t],model_names, columns=['PB','PCC','T'])
        df = pd.DataFrame([model_names, pb,pcc,t]) #, columns=['PB','PCC','T'])
        #df.T.to_excel(workdir+reg+'_bias_'+musim+'.xlsx', index=False, header=False) 
        
        
        if 0<=i<4:
            m = Basemap(ax=axes[0,i+1], projection ='cyl', 
                llcrnrlat = lat_min+lat_min1, urcrnrlat = lat_max-lat_max1,
                llcrnrlon = lon_min+lon_min1, urcrnrlon = lon_max-lon_max1, 
                resolution = 'l', fix_aspect=False)
            m.drawcoastlines(linewidth=1)
            m.drawcountries(linewidth=1)
            m.drawstates(linewidth=0.5, color='w')
        
            max = m.contourf(x, y, 
                      dbi, 
                      levels=clevs, 
                      extend='both',
                      cmap=cmap2)
            axes[0,i+1].set_xticks([])
            axes[0,i+1].set_xticklabels([])
            if i==0: 
                axes[0,i+1].set_title(model_names[i]+' [PB='+'%.2f'%bias+'%]',pad=4,fontsize=9)
            if i>=1:
                axes[0,i+1].set_title(model_names[i]+' [PB='+'%.2f'%bias+'%]',pad=4,fontsize=9)
                axes[0,i+1].text(x=91, y=-13, s=' PCC='+'%.2f'%c+'  T='+'%.2f'%T, fontsize=8)
            axes[0,1+i].set_yticks([])
        
        
        if 9>i>=4:
            m = Basemap(ax=axes[1,i-4], projection ='cyl', 
                llcrnrlat = lat_min+lat_min1, urcrnrlat = lat_max-lat_max1,
                llcrnrlon = lon_min+lon_min1, urcrnrlon = lon_max-lon_max1, 
                resolution = 'l', fix_aspect=False)
            m.drawcoastlines(linewidth=1)
            m.drawcountries(linewidth=1)
            m.drawstates(linewidth=0.5, color='w')
            
            max = m.contourf(x, y, 
                      dbi, 
                      levels=clevs, 
                      extend='both',
                      cmap=cmap2)
            #axes[1,i-4].set_xticks([100,120,140])
            #axes[1,i-4].xaxis.set_tick_params(labelsize=6)
            axes[1,i-4].set_title(model_names[i]+' [PB='+'%.2f'%bias+'%]',pad=4,fontsize=9)
            #for JJA
            #if i==4 or i==6:
            #    axes[1,i-4].text(x=128, y=10, s='T='+'%.2f'%T, fontsize=9)
            #else:
            axes[1,i-4].text(x=91, y=-13, s=' PCC='+'%.2f'%c+'  T='+'%.2f'%T, fontsize=8)
            axes[1,0].set_yticks([-10,0,10,20])
            #axes[1,0].yaxis.set_tick_params(labelsize=6)
            axes[1,0].set_yticklabels(['10$^\circ$S','EQ','10$^\circ$N','20$^\circ$N'], fontsize=7)
            axes[1,i-4].set_xticks([])
            if i>4: axes[1,i-4].set_yticks([])                     
    
        if i>=9:
            m = Basemap(ax=axes[2,i-9], projection ='cyl', 
                llcrnrlat = lat_min+lat_min1, urcrnrlat = lat_max-lat_max1,
                llcrnrlon = lon_min+lon_min1, urcrnrlon = lon_max-lon_max1, 
                resolution = 'l', fix_aspect=False)
            m.drawcoastlines(linewidth=1)
            m.drawcountries(linewidth=1)
            m.drawstates(linewidth=0.5, color='w')
           
            max = m.contourf(x, y, 
                      dbi, 
                      levels=clevs, 
                      extend='both',
                      cmap=cmap2)
            axes[2,i-9].set_xticks([100,120,140])
            #axes[2,i-9].xaxis.set_tick_params(labelsize=6)
            axes[2,i-9].set_xticklabels(['100$^\circ$E','120$^\circ$E','140$^\circ$E'], fontsize=7)
            axes[2,i-9].tick_params(axis='x', pad=1,labelsize=7)   
            axes[2,i-9].set_title(model_names[i]+' [PB='+'%.2f'%bias+'%]',pad=4,fontsize=9)
            axes[2,i-9].text(x=91, y=-13, s=' PCC='+'%.2f'%c+'  T='+'%.2f'%T, fontsize=8)
            axes[2,0].set_yticks([-10,0,10,20])
            #axes[2,0].yaxis.set_tick_params(labelsize=6)
            axes[2,0].set_yticklabels(['10$^\circ$S','EQ','10$^\circ$N','20$^\circ$N'], fontsize=7)
           
            if i>9: axes[2,i-9].set_yticks([])
        
    #print('S_Taylor zonal mean=',mbz)  
    
    '''   
    #The dimensions [left, bottom, width, height] of the new axes
    cax1 = fig.add_axes([0.12, 0.92, 0.15, 0.02])
    cbar1=plt.colorbar(max1, cax = cax1, orientation='horizontal')
    cbar1.ax.tick_params(labelsize=7)
    cbar1.ax.xaxis.set_ticks_position('top')
    cbar1.ax.xaxis.set_label_position('top')   
    '''
    
    cax = fig.add_axes([0.91, 0.2, 0.01, 0.6])
    #cax = fig.add_axes([0.5, 0.97, 0.6, 0.02])
    cbar=plt.colorbar(max, cax = cax)
    cbar.ax.tick_params(labelsize=7)

    #plt.colorbar(cax).ax.set_title('mm')
 
    plt.subplots_adjust(hspace=.2,wspace=.05)
    #plt.tight_layout()
    #plt.title(file_name, y=0, x=0)
  
    plt.show()
    fig.savefig(workdir+file_name+reg+musim,dpi=300,bbox_inches='tight')
    
def mean_rainfall_11_season_bias(obs_dataset, obs_name, model_datasets, model_names, workdir):
  
    musim='DJF'
    file_name='mean_rainfall_Taylor_'+musim
        
    
    fig, axes = plt.subplots(nrows=3, ncols=4,figsize=(4,6))
    Tmax = 15; Tmin =0 ; delT = 3
    clevs = np.arange(Tmin,Tmax+delT,delT)
    
   
    cmap2='rainbow'
    x_tick=['J','F','M','A','M','J','J','A','S','O','N','D']
    
    ds = xr.DataArray(obs_dataset.values,
    coords={'time': obs_dataset.times,
    'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
    dims=["time", "lat", "lon"])
        
    #season
    dss = ds.groupby('time.season')
    ds0= dss[musim].mean(dim='time')
     
        
    plot='sum'
    
    if plot=='SEA':
        lat_min1 = 0.22*3
        lat_max1 = 0.22*6
        lon_min1 = 0.22*4
        lon_max1 = 0.22*3
    
    if plot=='sum':
        lat_min1 = 0.22*1
        lat_max1 = 0.22*3
        lon_min1 = 0.22*3
        lon_max1 = 0.22*1
  
    lat_min = ds.lat.min()
    lat_max = ds.lat.max()
    lon_min = ds.lon.min()
    lon_max = ds.lon.max()
    x,y = np.meshgrid(ds.lon, ds.lat)
    
    m = Basemap(ax=axes[0,0], projection ='cyl', 
                llcrnrlat = lat_min+lat_min1, urcrnrlat = lat_max-lat_max1,
                llcrnrlon = lon_min+lon_min1, urcrnrlon = lon_max-lon_max1, 
                resolution = 'l', fix_aspect=False)
    m.drawcoastlines(linewidth=1)
    m.drawcountries(linewidth=1)
    m.drawstates(linewidth=0.5, color='w')

    
    max1 = m.contourf(x, y, 
                      ds0, 
                      #levels=clevs, 
                      extend='both',
                      cmap=cmap2)
    axes[0,0].set_title(obs_name,pad=4, fontsize=9)
    axes[0,0].set_xticks([])
    axes[0,0].set_xticklabels([])
    
    axes[0,0].set_yticks([-10,0,10,20])
           
    axes[0,0].set_yticklabels(['10$^\circ$S','EQ','10$^\circ$N','20$^\circ$N'])
    
    axes[0,0].tick_params(axis='y', pad=1,labelsize=7)
    axes[1,0].tick_params(axis='y', pad=1,labelsize=7)
    axes[2,0].tick_params(axis='y', pad=1,labelsize=7)
    
    
   
    from scipy.stats import pearsonr
  
    sd1=ds0.std(skipna=None) 
    #print('sd1 =', sd1.values )
    mbz=ma.zeros(len(model_datasets)) 
    pb=[]
    pcc=[]
    t=[]
    
    r0=1
    cmap2='RdBu_r'
    Tmax = 15; Tmin =-15 ; delT = 3
    clevs = np.arange(Tmin,Tmax+delT,delT)
    for i in np.arange(len(model_datasets)):
        print(i)
        #if i<6: axes[0,1+i].axis('off')
        ds = xr.DataArray(model_datasets[i].values,
        coords={'time': obs_dataset.times,
        'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
        dims=["time", "lat", "lon"])
        
        dss= ds.groupby('time.season')
        dsi= dss[musim].mean(dim='time')
        
        #percent bias = (model mean – observed mean) / |model mean|  x100%
        bias=(dsi.mean() - ds0.mean())/abs(dsi.mean()) *100
        
        
        dbi=dsi-ds0
        # print('dsi.mean()=',dsi.mean().data)
        #print('abs(dsi).mean()=',abs(dsi).mean().data)
        #bias=abs(dsi).mean()
       
        #jadikan GPCP baseline bias
        if i==0: 
            db0=dbi
            sd0=dbi.std(skipna=True)
            #axes[0,1].set_title(model_names[i],pad=3, fontsize=9)
        
        c=xr.corr(dbi,db0)
        sdi=dbi.std(skipna=True)
        s=sdi/sd0       
        T= (4*(1+c))/(((s+1/s)**2)*((1+r0)))
        
        pb.append(np.round(bias.data,2))
        pcc.append(np.round(c.data,2))
        t.append(np.round(T.data,2))
        
        import pandas as pd
        #df = pd.DataFrame([pb,pcc,t],model_names, columns=['PB','PCC','T'])
        df = pd.DataFrame([model_names, pb,pcc,t]) #, columns=['PB','PCC','T'])
        #df.T.to_excel(workdir+reg+'_bias_'+musim+'.xlsx', index=False, header=False) 
        
        
        if 0<=i<3:
            m = Basemap(ax=axes[0,i+1], projection ='cyl', 
                llcrnrlat = lat_min+lat_min1, urcrnrlat = lat_max-lat_max1,
                llcrnrlon = lon_min+lon_min1, urcrnrlon = lon_max-lon_max1, 
                resolution = 'l', fix_aspect=False)
            m.drawcoastlines(linewidth=1)
            m.drawcountries(linewidth=1)
            m.drawstates(linewidth=0.5, color='w')
        
            max = m.contourf(x, y, 
                      dbi, 
                      #levels=clevs, 
                      extend='both',
                      cmap=cmap2)
            axes[0,i+1].set_xticks([])
            axes[0,i+1].set_xticklabels([])
            if i==0: 
                axes[0,i+1].set_title(model_names[i]+' [PB='+'%.2f'%bias+'%]',pad=4,fontsize=9)
            if i>=1:
                axes[0,i+1].set_title(model_names[i]+' [PB='+'%.2f'%bias+'%]',pad=4,fontsize=9)
                axes[0,i+1].text(x=91, y=-13, s=' PCC='+'%.2f'%c+'  T='+'%.2f'%T, fontsize=8)
            axes[0,1+i].set_yticks([])
        
        
        if 7>i>=3:
            m = Basemap(ax=axes[1,i-4], projection ='cyl', 
                llcrnrlat = lat_min+lat_min1, urcrnrlat = lat_max-lat_max1,
                llcrnrlon = lon_min+lon_min1, urcrnrlon = lon_max-lon_max1, 
                resolution = 'l', fix_aspect=False)
            m.drawcoastlines(linewidth=1)
            m.drawcountries(linewidth=1)
            m.drawstates(linewidth=0.5, color='w')
            
            max = m.contourf(x, y, 
                      dbi, 
                      levels=clevs, 
                      extend='both',
                      cmap=cmap2)
            #axes[1,i-4].set_xticks([100,120,140])
            #axes[1,i-4].xaxis.set_tick_params(labelsize=6)
            axes[1,i-4].set_title(model_names[i]+' [PB='+'%.2f'%bias+'%]',pad=4,fontsize=9)
            #for JJA
            #if i==4 or i==6:
            #    axes[1,i-4].text(x=128, y=10, s='T='+'%.2f'%T, fontsize=9)
            #else:
            axes[1,i-4].text(x=91, y=-13, s=' PCC='+'%.2f'%c+'  T='+'%.2f'%T, fontsize=8)
            axes[1,0].set_yticks([-10,0,10,20])
            #axes[1,0].yaxis.set_tick_params(labelsize=6)
            axes[1,0].set_yticklabels(['10$^\circ$S','EQ','10$^\circ$N','20$^\circ$N'], fontsize=7)
            axes[1,i-4].set_xticks([])
            if i>4: axes[1,i-4].set_yticks([])                     
        n=7
        if i>=n:
            m = Basemap(ax=axes[2,i-n], projection ='cyl', 
                llcrnrlat = lat_min+lat_min1, urcrnrlat = lat_max-lat_max1,
                llcrnrlon = lon_min+lon_min1, urcrnrlon = lon_max-lon_max1, 
                resolution = 'l', fix_aspect=False)
            m.drawcoastlines(linewidth=1)
            m.drawcountries(linewidth=1)
            m.drawstates(linewidth=0.5, color='w')
           
            max = m.contourf(x, y, 
                      dbi, 
                      levels=clevs, 
                      extend='both',
                      cmap=cmap2)
            axes[2,i-n].set_xticks([100,120,140])
            #axes[2,i-n].xaxis.set_tick_params(labelsize=6)
            axes[2,i-n].set_xticklabels(['100$^\circ$E','120$^\circ$E','140$^\circ$E'], fontsize=7)
            axes[2,i-n].tick_params(axis='x', pad=1,labelsize=7)   
            axes[2,i-n].set_title(model_names[i]+' [PB='+'%.2f'%bias+'%]',pad=4,fontsize=n)
            axes[2,i-n].text(x=91, y=-13, s=' PCC='+'%.2f'%c+'  T='+'%.2f'%T, fontsize=8)
            axes[2,0].set_yticks([-10,0,10,20])
            #axes[2,0].yaxis.set_tick_params(labelsize=6)
            axes[2,0].set_yticklabels(['10$^\circ$S','EQ','10$^\circ$N','20$^\circ$N'], fontsize=7)
           
            if i>n: axes[2,i-n].set_yticks([])
        
    axes[1,0].set_ylabel('Boreal winter (DJF)',fontsize=10)
    axes[0,0].set_ylabel('')
    axes[2,0].set_ylabel('')
    #The dimensions [left, bottom, width, height] of the new axes
    cax1 = fig.add_axes([0.12, 0.92, 0.17, 0.02])
    cbar1=plt.colorbar(max1, cax = cax1, orientation='horizontal')
    cbar1.ax.tick_params(labelsize=7)
    cbar1.ax.xaxis.set_tick_params(pad=0)
    cbar1.ax.xaxis.set_ticks_position('top')
    cbar1.ax.xaxis.set_label_position('top')
    cbar1.ax.set_xlabel('mm/day', labelpad=3)

    #vertikal
    cax = fig.add_axes([0.91, 0.2, 0.01, 0.6])
    #cax = fig.add_axes([0.5, 0.97, 0.6, 0.02])
    cbar=plt.colorbar(max, cax = cax)
    cbar.ax.tick_params(labelsize=7)
    cbar.ax.yaxis.set_tick_params(pad=3)
    cbar.ax.set_ylabel('mm/day', rotation=270, labelpad=4)

    #plt.colorbar(cax).ax.set_title('mm')
 
    plt.subplots_adjust(hspace=.2,wspace=.05)
    #plt.tight_layout()
    #plt.title(file_name, y=0, x=0)
  
    plt.show()
    fig.savefig(workdir+file_name+reg+musim,dpi=300,bbox_inches='tight')

def mean_rainfall_10era_season_bias(obs_dataset, obs_name, model_datasets, model_names, workdir):
  
    musim='JJA'
    file_name='mean_rainfall_Taylor_'+musim
        
    
    fig, axes = plt.subplots(nrows=3, ncols=4,figsize=(4,6))
    Tmax = 15; Tmin =0 ; delT = 3
    clevs = np.arange(Tmin,Tmax+delT,delT)
    
   
    cmap2='rainbow'
    x_tick=['J','F','M','A','M','J','J','A','S','O','N','D']
    
    ds = xr.DataArray(obs_dataset.values,
    coords={'time': obs_dataset.times,
    'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
    dims=["time", "lat", "lon"])
        
    #season
    dss = ds.groupby('time.season')
    ds0= dss[musim].mean(dim='time')
     
        
    plot='sum'
    
    if plot=='SEA':
        lat_min1 = 0.22*3
        lat_max1 = 0.22*6
        lon_min1 = 0.22*4
        lon_max1 = 0.22*3
    
    if plot=='sum':
        lat_min1 = 0.22*1
        lat_max1 = 0.22*3
        lon_min1 = 0.22*3
        lon_max1 = 0.22*1
  
    lat_min = ds.lat.min()
    lat_max = ds.lat.max()
    lon_min = ds.lon.min()
    lon_max = ds.lon.max()
    
    x,y = np.meshgrid(ds.lon, ds.lat)
    
   
    from scipy.stats import pearsonr
  
   
    pb=[]
    pcc=[]
    t=[]
    
    r0=1
    cmap2='RdBu_r'
    Tmax = 15; Tmin =-15 ; delT = 3
    clevs = np.arange(Tmin,Tmax+delT,delT)
    for i in np.arange(len(model_datasets)):
        print(i)
     
        ds = xr.DataArray(model_datasets[i].values,
        coords={'time': obs_dataset.times,
        'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
        dims=["time", "lat", "lon"])
        
        dss= ds.groupby('time.season')
        dsi= dss[musim].mean(dim='time')
        
        #percent bias = (model mean – observed mean) / |model mean|  x100%
        bias=(dsi.mean() - ds0.mean())/abs(dsi.mean()) *100
        
        
        dbi=dsi-ds0
        # print('dsi.mean()=',dsi.mean().data)
        #print('abs(dsi).mean()=',abs(dsi).mean().data)
        #bias=abs(dsi).mean()
       
        if i==0: 
           db0=dbi
           sd0=dbi.std(skipna=True)
        
        c=xr.corr(dbi,db0)
        sdi=dbi.std(skipna=True)
        s=sdi/sd0       
        T= (4*(1+c))/(((s+1/s)**2)*((1+r0)))
        
        pb.append(np.round(bias.data,2))
        pcc.append(np.round(c.data,2))
        t.append(np.round(T.data,2))
        
        import pandas as pd
        #df = pd.DataFrame([pb,pcc,t],model_names, columns=['PB','PCC','T'])
        df = pd.DataFrame([model_names, pb,pcc,t]) #, columns=['PB','PCC','T'])
        #df.T.to_excel(workdir+reg+'_bias_'+musim+'.xlsx', index=False, header=False) 
        
        
        if 0<=i<4:
            m = Basemap(ax=axes[0,i], projection ='cyl', 
                llcrnrlat = lat_min+lat_min1, urcrnrlat = lat_max-lat_max1,
                llcrnrlon = lon_min+lon_min1, urcrnrlon = lon_max-lon_max1, 
                resolution = 'l', fix_aspect=False)
            m.drawcoastlines(linewidth=1)
            m.drawcountries(linewidth=1)
            m.drawstates(linewidth=0.5, color='w')
        
            max = m.contourf(x, y, 
                      dbi, 
                      levels=clevs, 
                      extend='both',
                      cmap=cmap2)
            axes[0,i].set_xticks([])
            axes[0,i].set_xticklabels([])
            
            axes[0,i].set_title(model_names[i]+' [PB='+'%.2f'%bias+'%]',pad=4,fontsize=9)
            #axes[0,i].text(x=91, y=-13, s=' PCC='+'%.2f'%c+'  T='+'%.2f'%T, fontsize=8)
            axes[0,1].set_yticks([])
        
        
        if 8>i>=4:
            m = Basemap(ax=axes[1,i-4], projection ='cyl', 
                llcrnrlat = lat_min+lat_min1, urcrnrlat = lat_max-lat_max1,
                llcrnrlon = lon_min+lon_min1, urcrnrlon = lon_max-lon_max1, 
                resolution = 'l', fix_aspect=False)
            m.drawcoastlines(linewidth=1)
            m.drawcountries(linewidth=1)
            m.drawstates(linewidth=0.5, color='w')
            
            max = m.contourf(x, y, 
                      dbi, 
                      levels=clevs, 
                      extend='both',
                      cmap=cmap2)
            #axes[1,i-4].set_xticks([100,120,140])
            #axes[1,i-4].xaxis.set_tick_params(labelsize=6)
            axes[1,i-4].set_title(model_names[i]+' [PB='+'%.2f'%bias+'%]',pad=4,fontsize=9)
            #for JJA
            #if i==4 or i==6:
            #    axes[1,i-4].text(x=128, y=10, s='T='+'%.2f'%T, fontsize=9)
            #else:
            #axes[1,i-4].text(x=91, y=-13, s=' PCC='+'%.2f'%c+'  T='+'%.2f'%T, fontsize=8)
            axes[1,0].set_yticks([-10,0,10,20])
            #axes[1,0].yaxis.set_tick_params(labelsize=6)
            axes[1,0].set_yticklabels(['10$^\circ$S','0','10$^\circ$N','20$^\circ$N'], fontsize=7)
            axes[1,i-4].set_xticks([])
            if i>4: axes[1,i-4].set_yticks([])                     
        n=8
        if i>=n:
            m = Basemap(ax=axes[2,i-n], projection ='cyl', 
                llcrnrlat = lat_min+lat_min1, urcrnrlat = lat_max-lat_max1,
                llcrnrlon = lon_min+lon_min1, urcrnrlon = lon_max-lon_max1, 
                resolution = 'l', fix_aspect=False)
            m.drawcoastlines(linewidth=1)
            m.drawcountries(linewidth=1)
            m.drawstates(linewidth=0.5, color='w')
           
            max = m.contourf(x, y, 
                      dbi, 
                      levels=clevs, 
                      extend='both',
                      cmap=cmap2)
            axes[2,i-n].set_xticks([100,120,140])
            #axes[2,i-n].xaxis.set_tick_params(labelsize=6)
            axes[2,i-n].set_xticklabels(['100$^\circ$E','120$^\circ$E','140$^\circ$E'], fontsize=7)
            axes[2,i-n].tick_params(axis='x', pad=1,labelsize=7)   
            axes[2,i-n].set_title(model_names[i]+' [PB='+'%.2f'%bias+'%]',pad=4,fontsize=n)
            #axes[2,i-n].text(x=91, y=-13, s=' PCC='+'%.2f'%c+'  T='+'%.2f'%T, fontsize=8)
            axes[2,0].set_yticks([-10,0,10,20])
            #axes[2,0].yaxis.set_tick_params(labelsize=6)
            axes[2,0].set_yticklabels(['10$^\circ$S','EQ','10$^\circ$N','20$^\circ$N'], fontsize=7)
           
            if i>n: 
                axes[2,i-n].set_yticks([])
            
        
    
    axes[2,2].axis('off')
    axes[2,3].axis('off')
    
    
    axes[0,0].set_yticks([-10,0,10,20])
           
    axes[0,0].set_yticklabels(['10$^\circ$S','0','10$^\circ$N','20$^\circ$N'])
    
    axes[0,0].tick_params(axis='y', pad=1,labelsize=7)
    axes[1,0].tick_params(axis='y', pad=1,labelsize=7)
    axes[2,0].tick_params(axis='y', pad=1,labelsize=7)
    
    #axes[0,0].set_ylabel(musim,fontsize=8)
    axes[1,0].set_ylabel('Boreal summer (JJA)',fontsize=10)
    #axes[2,0].set_ylabel(musim,fontsize=8)
    
    cax = fig.add_axes([0.55, 0.2, 0.3, 0.02])
    #cax = fig.add_axes([0.5, 0.97, 0.6, 0.02])
    cbar=plt.colorbar(max, cax = cax, orientation='horizontal')
    cbar.ax.tick_params(labelsize=7)
    cbar.ax.yaxis.set_tick_params(pad=3)
    cbar.ax.set_xlabel('mm/day', labelpad=4)

    #plt.colorbar(cax).ax.set_title('mm')
 
    plt.subplots_adjust(hspace=.2,wspace=.05)
    #plt.tight_layout()
    #plt.title(file_name, y=0, x=0)
  
    plt.show()
    fig.savefig(workdir+file_name+reg+musim,dpi=300,bbox_inches='tight')

def mean_rainfall_season_bias_5obs(obs_dataset, obs_name, model_datasets, model_names, workdir):
  
    musim='JJA'
    file_name='mean_rainfall_Taylor_'+musim
        
    
    fig, axes = plt.subplots(nrows=1, ncols=5,figsize=(6,4))
    Tmax = 15; Tmin =0 ; delT = 3
    clevs = np.arange(Tmin,Tmax+delT,delT)
    
   
    cmap2='rainbow'
    x_tick=['J','F','M','A','M','J','J','A','S','O','N','D']
    
    ds = xr.DataArray(obs_dataset.values,
    coords={'time': obs_dataset.times,
    'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
    dims=["time", "lat", "lon"])
        
    #season
    dss = ds.groupby('time.season')
    ds0= dss[musim].mean(dim='time')
     
        
    plot='sum'
    
   
    if plot=='sum':
        lat_min1 = 0.22*1
        lat_max1 = 0.22*3
        lon_min1 = 0.22*3
        lon_max1 = 0.22*1
  
    lat_min = ds.lat.min()
    lat_max = ds.lat.max()
    lon_min = ds.lon.min()
    lon_max = ds.lon.max()
    x,y = np.meshgrid(ds.lon, ds.lat)
    
    m = Basemap(ax=axes[0], projection ='cyl', 
                llcrnrlat = lat_min+lat_min1, urcrnrlat = lat_max-lat_max1,
                llcrnrlon = lon_min+lon_min1, urcrnrlon = lon_max-lon_max1, 
                resolution = 'l', fix_aspect=False)
    m.drawcoastlines(linewidth=1)
    m.drawcountries(linewidth=1)
    m.drawstates(linewidth=0.5, color='w')

    
    max1 = m.contourf(x, y, 
                      ds0, 
                      levels=clevs, extend='both',
                      cmap=cmap2)
    axes[0].set_title(obs_name,pad=4, fontsize=9)
      
    axes[0].set_yticks([-5,0,5])  
    axes[0].set_yticklabels(['5$^\circ$S','0','5$^\circ$N'])
    axes[0].tick_params(axis='y', pad=1,labelsize=9)
    
    axes[0].set_xticks([95,100,105])  
    axes[0].set_xticklabels(['95$^\circ$E','100$^\circ$E','105$^\circ$E'])
    axes[0].tick_params(axis='x', pad=1,labelsize=9)
   
    axes[0].set_ylabel(musim,fontsize=9)
  
   
    cmap2='RdBu_r'
    Tmax = 15; Tmin =-15 ; delT = 3
    clevs = np.arange(Tmin,Tmax+delT,delT)
    for i in np.arange(len(model_datasets)):
        print(i)
        #if i<6: axes[0,1+i].axis('off')
        ds = xr.DataArray(model_datasets[i].values,
        coords={'time': obs_dataset.times,
        'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
        dims=["time", "lat", "lon"])
        
        dss= ds.groupby('time.season')
        dsi= dss[musim].mean(dim='time')
        
        #percent bias = (model mean – observed mean) / |model mean|  x100%
        bias=(dsi.mean() - ds0.mean())/abs(dsi.mean()) *100
       
        dbi=dsi-ds0
        #akibat axes[i+1] maka harus diatur i<4 agar tidak out of ...
        if i<4:
            m = Basemap(ax=axes[i+1], projection ='cyl', 
                llcrnrlat = lat_min+lat_min1, urcrnrlat = lat_max-lat_max1,
                llcrnrlon = lon_min+lon_min1, urcrnrlon = lon_max-lon_max1, 
                resolution = 'l', fix_aspect=False)
            m.drawcoastlines(linewidth=1)
            m.drawcountries(linewidth=1)
            m.drawstates(linewidth=0.5, color='w')
        
            max = m.contourf(x, y, 
                      dbi, 
                      levels=clevs, 
                      extend='both',
                      cmap=cmap2)
          
                
            
            axes[i+1].set_title(model_names[i]+' [PB='+'%.2f'%bias+'%]',pad=4,fontsize=9)
            
            axes[i+1].set_yticks([])
             
            axes[i+1].set_xticks([95,100,105])  
            axes[i+1].set_xticklabels(['95$^\circ$E','100$^\circ$E','105$^\circ$E'])
            axes[i+1].tick_params(axis='x', pad=1,labelsize=9)
            
        
       
    #The dimensions [left, bottom, width, height] of the new axes
    cax1 = fig.add_axes([0.12, 0.92, 0.15, 0.02])
    cbar1=plt.colorbar(max1, cax = cax1, orientation='horizontal')
    cbar1.ax.tick_params(labelsize=7)
    cbar1.ax.xaxis.set_ticks_position('top')
    cbar1.ax.xaxis.set_label_position('top')
    #cbar1.ax.set_title('         mm/month', fontsize=7)   
    #cbar1.ax.set_xlabel('mm/month')

    
    cax = fig.add_axes([0.91, 0.2, 0.01, 0.6])
    #cax = fig.add_axes([0.5, 0.97, 0.6, 0.02])
    cbar=plt.colorbar(max, cax = cax)
    cbar.ax.tick_params(labelsize=7)
    cbar.ax.set_ylabel('mm/month', rotation=270)

    #plt.colorbar(cax).ax.set_title('mm')
 
    plt.subplots_adjust(hspace=.2,wspace=.05)
    #plt.tight_layout()
    #plt.title(file_name, y=0, x=0)
  
    plt.show()
    #fig.savefig(workdir+file_name+reg+musim,dpi=300,bbox_inches='tight')
   
def mean_rainfall_5obs(seasons, obs_dataset, obs_name, model_datasets, model_names, workdir):
    #Sumatera
    plot='sum'
    print(seasons)
    for musim in seasons: 
       
        fig, axes = plt.subplots(nrows=1, ncols=len(model_datasets)+1, figsize=(10,6))
        Tmax = 15; Tmin =0 ; delT = 3
        clevs = np.arange(Tmin,Tmax+delT,delT)
       
        cmap='rainbow'
        x_tick=['J','F','M','A','M','J','J','A','S','O','N','D']
        
        ds = xr.DataArray(obs_dataset.values,
        coords={'time': obs_dataset.times,
        'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
        dims=["time", "lat", "lon"])
            
        #season
        dss = ds.groupby('time.season')
        ds0= dss[musim].mean(dim='time')
      
            
        if plot=='sum':
            lat_min1 = 0.22*1
            lat_max1 = 0.22*3
            lon_min1 = 0.22*3
            lon_max1 = 0.22*1
      
        lat_min = ds.lat.min()
        lat_max = ds.lat.max()
        lon_min = ds.lon.min()
        lon_max = ds.lon.max()
        x,y = np.meshgrid(ds.lon, ds.lat)
        
        m = Basemap(ax=axes[0], projection ='cyl', 
                    llcrnrlat = lat_min+lat_min1, urcrnrlat = lat_max-lat_max1,
                    llcrnrlon = lon_min+lon_min1, urcrnrlon = lon_max-lon_max1, 
                    resolution = 'l', fix_aspect=False)
        m.drawcoastlines(linewidth=1)
        m.drawcountries(linewidth=1)
        m.drawstates(linewidth=0.5, color='w')

        
        max1 = m.contourf(x, y, 
                          ds0, 
                          #levels=clevs, 
                          extend='both',
                          cmap=cmap)
        axes[0].set_title(obs_name, fontsize=9)
          
        axes[0].set_yticks([-5,0,5])  
        axes[0].set_yticklabels(['5S','0','5N'])
        axes[0].tick_params(axis='y', pad=1,labelsize=9)
        
        axes[0].set_xticks([95,100,105])  
        axes[0].set_xticklabels(['95E','100E','105E'])
        axes[0].tick_params(axis='x', pad=1, labelsize=8)
       
        axes[0].set_ylabel(musim,fontsize=9, labelpad=0)
      
        Tmax = 15; Tmin =-15 ; delT = 3
        clevs = np.arange(Tmin,Tmax+delT,delT)
        for i in np.arange(len(model_datasets)):
            print(i)
            #if i<6: axes[0,1+i].axis('off')
            ds = xr.DataArray(model_datasets[i].values,
            coords={'time': obs_dataset.times,
            'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
            dims=["time", "lat", "lon"])
            
            dss= ds.groupby('time.season')
            dsi= dss[musim].mean(dim='time')
           
            #akibat axes[i+1] maka harus diatur i<4 agar tidak out of ...
            if i<len(model_datasets):
                m = Basemap(ax=axes[i+1], projection ='cyl', 
                    llcrnrlat = lat_min+lat_min1, urcrnrlat = lat_max-lat_max1,
                    llcrnrlon = lon_min+lon_min1, urcrnrlon = lon_max-lon_max1, 
                    resolution = 'l', fix_aspect=False)
                m.drawcoastlines(linewidth=1)
                m.drawcountries(linewidth=1)
                m.drawstates(linewidth=0.5, color='w')
            
                max = m.contourf(x, y, 
                          dsi, 
                          #levels=clevs, 
                          extend='both',
                          cmap=cmap)
              
                axes[i+1].set_title(model_names[i],fontsize=9)
                
                axes[i+1].set_yticks([])
                 
                axes[i+1].set_xticks([95,100,105])  
                axes[i+1].set_xticklabels(['95E','100E','105E'])
                axes[i+1].tick_params(axis='x', pad=1,labelsize=8)
       
        cax = fig.add_axes([0.81, 0.3, 0.01, 0.6])
        #cax = fig.add_axes([0.5, 0.97, 0.6, 0.02])
        cbar=plt.colorbar(max, cax = cax)
        cbar.ax.tick_params(labelsize=7)
        cbar.ax.set_ylabel('mm/day', rotation=270, labelpad=10)

        #plt.colorbar(cax).ax.set_title('mm')
        plt.subplots_adjust(right=.8)
        plt.subplots_adjust(bottom=.3)
        plt.subplots_adjust(hspace=.2,wspace=.05)
        #plt.tight_layout()
        #plt.title(file_name, y=0, x=0)
      
        #plt.draw() nope
        plt.show()
        fig.savefig(workdir+'mean_'+reg+musim, dpi=300, bbox_inches='tight') 

def mean_rainfall_jambi(seasons, obs_dataset, obs_name, model_datasets, model_names, workdir):
    print(seasons)
    for musim in seasons: 
       
        fig, axes = plt.subplots(nrows=1, ncols=6, figsize=(10,6))
        Tmax = 15; Tmin =0 ; delT = 3
        clevs = np.arange(Tmin,Tmax+delT,delT)
       
        cmap='rainbow'
        x_tick=['J','F','M','A','M','J','J','A','S','O','N','D']
        
        ds = xr.DataArray(obs_dataset.values,
        coords={'time': obs_dataset.times,
        'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
        dims=["time", "lat", "lon"])
        
        #01°30'2,98" - 01°7'1,07" LS
        #103°40'1,67" - 103°40'0,23" BT
        ds = ds.where(
                    (ds.lat > -3) & (ds.lat < -0.8) & 
                    (ds.lon > 99) & (ds.lon < 106), drop=True)
            
        #season
        dss = ds.groupby('time.season')
        ds0= dss[musim].mean(dim='time')
      
        plot='sum'
     
        if plot=='sum':
            lat_min1 = 0.22*1
            lat_max1 = 0.22*3
            lon_min1 = 0.22*3
            lon_max1 = 0.22*1
      
        lat_min = ds.lat.min()
        lat_max = ds.lat.max()
        lon_min = ds.lon.min()
        lon_max = ds.lon.max()
        x,y = np.meshgrid(ds.lon, ds.lat)
        
        m = Basemap(ax=axes[0], projection ='cyl', 
                    llcrnrlat = lat_min+lat_min1, urcrnrlat = lat_max-lat_max1,
                    llcrnrlon = lon_min+lon_min1, urcrnrlon = lon_max-lon_max1, 
                    resolution = 'l', fix_aspect=False)
        m.drawcoastlines(linewidth=1)
        m.drawcountries(linewidth=1)
        m.drawstates(linewidth=0.5, color='w')

        
        max1 = m.contourf(x, y, 
                          ds0, 
                          #levels=clevs, 
                          extend='both',
                          cmap=cmap)
        axes[0].set_title(obs_name, fontsize=9)
        '''  
        axes[0].set_yticks([-3,-1])  
        axes[0].set_yticklabels(['3S','0','1S'])
        axes[0].tick_params(axis='y', pad=1,labelsize=9)
        
        axes[0].set_xticks([95,100,105])  
        axes[0].set_xticklabels(['95E','100E','105E'])
        axes[0].tick_params(axis='x', pad=1, labelsize=8)
        '''
        axes[0].set_ylabel(musim,fontsize=9, labelpad=0)
      
        Tmax = 15; Tmin =-15 ; delT = 3
        clevs = np.arange(Tmin,Tmax+delT,delT)
        for i in np.arange(len(model_datasets)):
            print(i)
            #if i<6: axes[0,1+i].axis('off')
            ds = xr.DataArray(model_datasets[i].values,
            coords={'time': obs_dataset.times,
            'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
            dims=["time", "lat", "lon"])
            
            ds = ds.where(
                    (ds.lat > -3) & (ds.lat < -0.8) & 
                    (ds.lon > 99) & (ds.lon < 106), drop=True)
            
            dss= ds.groupby('time.season')
            dsi= dss[musim].mean(dim='time')
           
            #akibat axes[i+1] maka harus diatur i<4 agar tidak out of ...
            if i<5:
                m = Basemap(ax=axes[i+1], projection ='cyl', 
                    llcrnrlat = lat_min+lat_min1, urcrnrlat = lat_max-lat_max1,
                    llcrnrlon = lon_min+lon_min1, urcrnrlon = lon_max-lon_max1, 
                    resolution = 'l', fix_aspect=False)
                m.drawcoastlines(linewidth=1)
                m.drawcountries(linewidth=1)
                m.drawstates(linewidth=0.5, color='w')
            
                max = m.contourf(x, y, 
                          dsi, 
                          #levels=clevs, 
                          extend='both',
                          cmap=cmap)
              
                axes[i+1].set_title(model_names[i],fontsize=9)
                
                axes[i+1].set_yticks([])
                 
                #axes[i+1].set_xticks([95,100,105])  
                #axes[i+1].set_xticklabels(['95E','100E','105E'])
                #axes[i+1].tick_params(axis='x', pad=1,labelsize=8)
       
        cax = fig.add_axes([0.81, 0.3, 0.01, 0.6])
        #cax = fig.add_axes([0.5, 0.97, 0.6, 0.02])
        cbar=plt.colorbar(max, cax = cax)
        cbar.ax.tick_params(labelsize=7)
        cbar.ax.set_ylabel('mm/day', rotation=270, labelpad=10)

        #plt.colorbar(cax).ax.set_title('mm')
        plt.subplots_adjust(right=.8)
        plt.subplots_adjust(bottom=.3)
        plt.subplots_adjust(hspace=.2,wspace=.05)
        #plt.tight_layout()
        #plt.title(file_name, y=0, x=0)
      
        #plt.draw() nope
        plt.show()
        fig.savefig(workdir+'mean_'+reg+musim, dpi=300, bbox_inches='tight') 

def obs5_cross_corr(obs_dataset, obs_name, model_datasets, model_names, workdir):
    import pandas as pd
    
    musim='JJA'
    file_name='mean_rainfall_Taylor_'+musim
        
    d=[]
    for i in np.arange(len(model_datasets)):
        print(i)
      
        ds = xr.DataArray(model_datasets[i].values,
        coords={'time': obs_dataset.times,
        'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
        dims=["time", "lat", "lon"])
        
        ds= ds.groupby('time.season')
        ds= ds[musim].mean(dim=('lat','lon'))
        
        #ds.rename('ts'+str(i))
        #print(ds)
        d.append(ds)
    #print(len(d))
    #print(d[0])
    
    
    #d=np.array(d
    dd= pd.concat([d[0].to_series(),d[1].to_series(),
        d[2].to_series(),d[3].to_series(),
        d[4].to_series(),d[5].to_series()],axis=1)
    #dd= pd.concat([d[0],d[1]])
    #dd= pd.concat([d])
    
    dd.columns =  model_names
    #print(dd) 
    
    res=dd.corr()
    #print(res) 
    # Exclude values of 1 from the correlation matrix
    res = res[res != 1.0]

    res.plot(kind='bar', rot=0)
    #plt.title('Cross-Correlation Bar Plot')
    #plt.xlabel('Time Series')
    plt.ylabel('Correlation')
    #plt.legend(title='Variables')
    #plt.legend(bbox_to_anchor=(1, .5), loc='best', prop={'size':8.5}, frameon=False)
    plt.legend(loc='upper left', ncol=6)
    plt.ylim(top=1.1)

    plt.show()
  
    #fig.savefig(workdir+file_name+reg+musim,dpi=300,bbox_inches='tight') 

def random_weigths(obs_dataset, obs_name, model_datasets, model_names, workdir):
    import pandas as pd
    from scipy import stats
    
    # Number of climate models
    # 2 obs and MME => all models - 2
    # 1 obs dan MME => -1
    num_models = len(model_datasets)-2

    # Number of random weight sets
    num_weight_sets = 100

    # Initialize arrays to store the random weight sets, data for each model, ensemble means, and observational data
    random_weights = np.zeros((num_weight_sets, num_models))
    #model_data = np.zeros((num_models, 365))  # Assuming 365 days of data for each model
    weighted_means = np.zeros((num_weight_sets, 365))  # Ensemble means for each weight set
    #observational_data = np.random.normal(10, 2, size=365)  # Simulated observational data

    # Initialize arrays to store RMSE, correlation, and SDR
    rmse_models = np.zeros(num_models)
    correlation_models = np.zeros(num_models)
    sdr_models = np.zeros(num_models)

    rmse_we = np.zeros(num_weight_sets)
    correlation_we = np.zeros(num_weight_sets)
    sdr_we = np.zeros(num_weight_sets)
    
    
    n=config['nobs']-1  #-1 ?? => era5 tidak ikut 
    reg=config['region']
    tipe= config['Metric_type']
    
    print('')
    print('Generating random weighted ensemble ...')
    
  
    lastElementIndex = len(model_datasets)-1
   
    # second obs dan MME tidak ikut mmew
    w_datasets=model_datasets[n:lastElementIndex]
    w_datasets_names=model_names[n:lastElementIndex]
    print('#target_names[n:]=',w_datasets_names)
    
    
    #w_random_names=[]
    
    rmse_mmew=[]
    w_random=[]
    #gab=np.zeros((num_weight_sets, num_models))
    gab=[]
    
    #rmse mme
    mme=dsp.ensemble_weighted(w_datasets,np.repeat(1, 9))
    
    #DJF
    mme= dsp.temporal_subset(mme, 12, 2)
    obs= dsp.temporal_subset(obs_dataset, 12, 2)
    
    mme = ds.Dataset(obs_dataset.lats, obs_dataset.lons, 
                        obs_dataset.times, 
                        utils.calc_temporal_mean(mme))
    obs = ds.Dataset(obs_dataset.lats, obs_dataset.lons, 
                        obs_dataset.times, 
                        utils.calc_temporal_mean(obs))
    
    predicted = np.array(mme.values)
    observed = np.array(obs.values)
    
    predicted[ predicted >999.0] = np.nan
    observed[observed >999.0] = np.nan
   
    differences = predicted - observed
    squared_differences = differences**2
    '''
    # Calculate RMSE
    rmse = np.sqrt(np.nanmean(squared_differences))
    print('rmse_mme', np.round(rmse,2))
    
    #std
    stdo=np.nanstd(observed, ddof=1)
    stdmme=np.nanstd(predicted, ddof=1)
    stdr=stdmme/stdo
    print('stdr', np.round(stdr,2))
    #exit()
    '''
    
    x0=observed.flatten()
    y0=predicted.flatten()
    #print(x0, y0)
                
    bad = ~np.logical_or(np.isnan(x0), np.isnan(y0))
    x1=np.compress(bad, x0) 
    y1=np.compress(bad, y0)
    r, p_value = stats.pearsonr(x1, y1)
    print('r_mme', np.round(r,3))
        
    #rmse_mme 2.4 stdr 1.21 r_mme 0.89
    
    
    # Generate random weights for each weight set
    for i in range(num_weight_sets):
        print(i)
        # Generate uniformly distributed random numbers between 0 and 1 for weights
        weights = np.random.rand(num_models)
        
        # Normalize the weights to sum to 1
        normalized_weights = weights #/ np.sum(weights)
        #print('normalized_weights', normalized_weights)
  
        w_random=dsp.ensemble_weighted(w_datasets,normalized_weights)
        
        #DJF
        w_random= dsp.temporal_subset(w_random, 12, 2)
        w_random = ds.Dataset(obs_dataset.lats, obs_dataset.lons, 
                        obs_dataset.times, 
                        utils.calc_temporal_mean(w_random))
        
        predicted = np.array(w_random.values)
        predicted[ predicted >999.0] = np.nan
       
        differences = predicted - observed
        squared_differences = differences**2
        
        '''
        #std
        stdmw=np.nanstd(predicted, ddof=1)
        stdr1=stdmw/stdo
        
        if 0.95 <stdr1<1.05:
            gab.append(np.insert(normalized_weights, 0, stdr1))
            print('stdr1', np.round(stdr1,2))
        
        #rmse
        rmse1 = np.sqrt(np.nanmean(squared_differences))
        rmse11=np.array((rmse1))
        #print(np.round(rmse1,2))
        
        #hasil rmse pada mmew cukup menurun
        #simpan rmse_min dan weightnys
        if rmse1<rmse: 
            print('mmew > mme')
            print(np.round(rmse1,2),' < ', np.round(rmse,2))
            print('normalized_weights', normalized_weights)
            #gab.append(np.array((rmse11,normalized_weights)))
            gab.append(np.insert(normalized_weights, 0, rmse1))
        
        
        '''
        #korelasi
        x0=observed.flatten()
        y0=predicted.flatten()
        #print(x0, y0)
                    
        bad = ~np.logical_or(np.isnan(x0), np.isnan(y0))
        x1=np.compress(bad, x0) 
        y1=np.compress(bad, y0)
        r1, p_value = stats.pearsonr(x1, y1)
        #print('r_mme', np.round(rr,2))
        
        #hasil r pada mmew kecil sekali
        if r1>r: 
            print('R_mmew > mme')
            print(np.round(r1,3),' > ', np.round(r,2))
            print('normalized_weights', normalized_weights)
            gab.append(np.insert(normalized_weights, 0, r1))
    
    '''    
    # rmse models
    for i in range(num_models):
        print(i) 
        predicted = np.array(w_datasets[i].values)
        observed = np.array(obs_dataset.values)
        
        predicted[ predicted >999.0] = np.nan
        observed[observed >999.0] = np.nan
                     
        # Calculate differences between observed and predicted
        # xr jika ada nan ini hasil kosong =[]
        # maka pakai np array diatas ######
        differences = predicted - observed
        squared_differences = differences**2
        
        #print(observed)
       
        # Calculate RMSE
        rmse = np.sqrt(np.nanmean(squared_differences))
        rmse_models.append(np.round(rmse,2))
    print(rmse_models)
    exit()
    '''
    gab2=np.array(gab)
    print(gab2)
    import pandas as pd
    df = pd.DataFrame(gab2)
    df.to_excel('D:/tes.xlsx', index=False, header=False)
    #gab2.tofile('D:/tes.csv', sep = ',')

def rmse_5obs(obs_dataset, obs_name, model_datasets, model_names, workdir):
    #import pandas as pd
    #from scipy import stats
    
    #structural data handling
    '''
    print(model_datasets[0].values[0,30,30])
    print(model_datasets[0].values.shape) #3D
    print(model_datasets[0].times.shape) #1D
    print(model_datasets[0].lats.shape)
    print(model_datasets[0].lons.shape)
    print(model_datasets[0].values[:,30,30].max())
    exit()
    '''
    nx=len(obs_dataset.lons)
    ny=len(obs_dataset.lats)
    
    for i in np.arange(nx):
        for j in np.arange(ny):
            x=1
    
    
    
    
    # Create an empty list to store RMSD values
    rmsd_values = []

    # Nested loops to calculate RMSD for all pairs of datasets
    for i in range(4):
        for j in range(i + 1, 4):
            # Calculate the squared differences between corresponding elements of the two datasets
            squared_diff = (model_datasets[i] - model_datasets[j])**2
            
            # Calculate the mean of squared differences
            mean_squared_diff = squared_diff.mean()
            
            # Calculate the square root to get RMSD
            rmsd = np.sqrt(mean_squared_diff)
            
            # Append the RMSD value to the list
            rmsd_values.append(rmsd)

    # Print or use the RMSD values as needed
    print(rmsd_values)
    
    
    
       
def rw_Taylor_D(obs_dataset, obs_name, model_datasets, model_names, workdir):
    # random weight pada Taylor diagram
    import pandas as pd
    from scipy import stats
    
        
    n=config['nobs']-1  #-1 ?? =>obs 2 tidak ikut 
    reg=config['region']
    tipe= config['Metric_type']
    
    print('')
    print('Processing random weighted ensemble ...')
    
    lastElementIndex = len(model_datasets)-1
   
    # second obs dan MME tidak ikut mmew
    w_datasets=model_datasets[n:lastElementIndex]
    w_datasets_names=model_names[n:lastElementIndex]
    print('#target_names[n:]=',w_datasets_names)
    
    #normalized_weights=np.array([0.97901605 0.19791322 0.80776814 0.08513868 0.40922281 0.310949960.21551825 0.10226996 0.61699519])
    w_random=dsp.ensemble_weighted(w_datasets,normalized_weights)      

    Taylor_diagram_spatial_pattern_of_multiyear_climatology(
                reference_dataset, reference_name, target_datasets, target_names,
                file_name)




def ori_resolution():
    file_path=[
    'D:/data1/APHRO_MA_025deg_V1101.1976-2005_mon.nc',
	'D:/data1/pr_chirps_monthly2_indo_1981-2021.nc', 
	'D:/data1/rr_0.25_saobs_mon_1981-2017.nc',
	'D:/data1/574766.p.gpcc_full_data_v7_05_1975-12-01_2006-01-01.nc',
	'D:/data1/pr_GPCP-SG_L3_v2.3_197901-201710.nc',
    ]


def station_cek(obs_dataset, obs_name, model_datasets, model_names, workdir):
    #14 sts Sumatera -- harian
    #ERA5 hourly data on single levels from 1940 to present
    #Daily nya gak ada, APHRO ada
    
    #Bulanan
    
    import xarray as xr
    from scipy import stats
    
    ds = xr.DataArray(obs_dataset.values,
    coords={'time': obs_dataset.times,
            'lat': obs_dataset.lats, 
            'lon': obs_dataset.lons},
    dims=["time", "lat", "lon"])
    
    print('obs_name: ', obs_name )
    
    fig, ax = plt.subplots(nrows=1, ncols=1 ,figsize=(8,6))
    
    lat_min = ds.lat.min()
    lat_max = ds.lat.max()
    lon_min = ds.lon.min()
    lon_max = ds.lon.max()
   
    x,y = np.meshgrid(ds.lon, ds.lat)
    
    m = Basemap(ax=ax, projection ='cyl', llcrnrlat = lat_min, urcrnrlat = lat_max,
                llcrnrlon = lon_min, urcrnrlon = lon_max, resolution = 'l', fix_aspect=False)
    m.drawcoastlines(linewidth=1)
    m.drawcountries(linewidth=1)
    m.drawstates(linewidth=0.5, color='w')
    
    dss = xr.open_dataset('D:/bmkg_csv/sum17/Pr_bmkg_1981-2005.nc')
    #karna model nya monthly maka
    #dss=dss.sortby('lat','lon')
    #lons=dss.lon
    
    lats=dss.lat.values    
    for i in np.argsort(lats)[::-1]:
        #plot Sumatera
        print(i)
        n=n+1
        ax.text(dss.lon[i], dss.lat[i], s=str(n), color='red')  #s='max='+'%.2f'%mak, fontsize=6)
    plt.show()
        
  
def station(obs_dataset, obs_name, model_datasets, model_names, workdir):
    #14 sts Sumatera -- harian
    #ERA5 hourly data on single levels from 1940 to present
    #Daily nya gak ada, APHRO ada
    
    #Bulanan
    
    import xarray as xr
    from scipy import stats
    
    ds = xr.DataArray(obs_dataset.values,
    coords={'time': obs_dataset.times,
            'lat': obs_dataset.lats, 
            'lon': obs_dataset.lons},
    dims=["time", "lat", "lon"])
    
    print('obs_name: ', obs_name )
    
    fig, ax = plt.subplots(nrows=1, ncols=1 ,figsize=(8,6))
    
    lat_min = ds.lat.min()
    lat_max = ds.lat.max()-1
    lon_min = ds.lon.min()+2
    lon_max = ds.lon.max()
   
    x,y = np.meshgrid(ds.lon, ds.lat)
    
    m = Basemap(ax=ax, projection ='cyl', llcrnrlat = lat_min, urcrnrlat = lat_max,
                llcrnrlon = lon_min, urcrnrlon = lon_max, resolution = 'l', fix_aspect=False)
    m.drawcoastlines(linewidth=1)
    m.drawcountries(linewidth=1)
    m.drawstates(linewidth=0.5, color='w')
    
    dss = xr.open_dataset('D:/bmkg_csv/sum17/Pr_bmkg_1981-2005.nc')
    #karna model nya monthly maka
    #dss=dss.sortby('lat','lon')
    
    dss = dss.resample(time='1M').mean()
    rmse=[]
    mae=[]
    pb=[]
    r=[]
    pv=[]
    
    lats=dss.lat.values
    #n=np.argsort(lats)[::-1]
    n=0
    # i dimulai dari lat rendah agar teratur
    for i in np.argsort(lats)[::-1]:
    #for i in np.arange(len(dss.lat)):
        n=n+1
        print('i=', i)
        
        '''
        #Count 4 neighbours, off kan jika tidak perlu
        #------------------
        d=ds
        # Define the central latitude and longitude
        central_lat = dss.lat[i].data
        central_lon = dss.lon[i].data

        data = []
        k=d.sel(lat=central_lat, lon=central_lon, method='nearest').lat.data
        j=d.sel(lat=central_lat, lon=central_lon, method='nearest').lon.data
        data1=d.sel(lat=k, lon=j, method='nearest')
        data.append(np.array(data1))
        
        #print(data1)
        #lt=abs(ds.lat[1])-abs(ds.lat[0])
        lt=0.22
        print(lt)
        # Define offsets for creating a box around the central point
        lat_offsets = np.array([-lt, lt])  # Adjust the range as needed
        lon_offsets = np.array([-lt, lt])  # Adjust the range as needed

        # Calculate the latitudes and longitudes for the box around the central point
        latitudes = central_lat + lat_offsets
        longitudes = central_lon + lon_offsets
        zz=0
        # Loop through latitudes and longitudes, extract data, and calculate averages
        for lat in latitudes:
            for lon in longitudes:
                selected_data = d.sel(lat=lat, lon=lon, method='nearest')
                #print(selected_data)
                if np.isnan(np.mean(selected_data)): 
                    selected_data=np.array([0])
                    zz=zz+1
                data.append(np.array(selected_data))

        print('len(data)', len(data))        
        h=(data[0]+data[1]+data[2]+data[3]+data[4])/(5-zz)
        
        
        print(zz)
        predicted = np.array(h)
        #----------------------------------------------------
        '''
        
        #-----Count only 1 center------- 
        ed = ds.sel(lat=dss.lat[i], lon=dss.lon[i], method='nearest')
        predicted = np.array(ed.values)
        #-------------------------------
        observed = np.array(dss.pr[:,i,i].values)
                     
        # Calculate differences between observed and predicted
        # xr jika ada nan ini hasil kosong =[]
        # maka pakai np array diatas ######
        differences = predicted - observed
        print('')
        #print('differences',  differences)

        # Calculate squared differences for RMSE
        squared_differences = differences**2
        #print('squared_differences', squared_differences)

        # Calculate RMSE
        rmse1 = np.sqrt(np.nanmean(squared_differences))
        rmse.append(round(rmse1,2))
        
        # Calculate MAE
        mae1 = np.nanmean(np.abs(differences))
        mae.append(round(mae1,2))

        # Calculate Percent Bias
        percent_bias = (np.nansum(differences) / np.nansum(observed)) * 100
        pb.append(round(percent_bias,2))
        
        # Calculate Correlation and p-value
        # ini sensitif pada panjang data dan NaNs
        x0=observed
        y0=predicted
        #print(x0, y0)
                    
        bad = ~np.logical_or(np.isnan(x0), np.isnan(y0))
        x1=np.compress(bad, x0) 
        y1=np.compress(bad, y0)
        rr, p_value = stats.pearsonr(x1, y1)
        r.append(round(rr,2))
        pv.append(round(p_value,3))
        
        #plot RMSE-station
        #ax.scatter(i+1,rmse1, label = 'RMSE',marker='o',color='black')
        #ax.scatter(i+1,mae1, label = 'MAE',marker='o',color='blue')
    
    
   
    '''
        #plot Sumatera
        if n in [2,7,11, 13]:
            ax.text(dss.lon[i], dss.lat[i], s='    '+str(n), color='red') 
            #ax.text(dss.lon[i], dss.lat[i]+, s=str(n), color='red') 
        else:
            ax.text(dss.lon[i], dss.lat[i], s=str(n), color='red')  
            #s='max='+'%.2f'%mak, fontsize=6)
    
    ax.set_yticks([-5,0,5])          
    ax.set_xticks([96,100,104]) 
    ax.set_yticklabels(['5S','0','5N'])
    ax.set_xticklabels(['96E','100E','104E']) 
    ax.xaxis.set_tick_params(labelsize=9)
    ax.yaxis.set_tick_params(labelsize=9)
    plt.show()
    exit()
    '''
    
    fig, ax = plt.subplots(nrows=1, ncols=2 ,figsize=(8,6))
    sts= ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14']
    
    ax[0].plot(np.arange(14)+1,rmse, label = 'RMSE',marker='o',color='black')
    ax[0].legend(bbox_to_anchor=(0.2, .99), loc='best', prop={'size':10}, frameon=False) 
    ax[0].set_xticks(np.arange(14)+1)
    ax[0].set_xticklabels(sts)
    #ax[0].set_title(obs_name+' rainfall dataset vs 14 Stations')
    
    ax2 = ax[0].twinx()
    ax2.plot(np.arange(14)+1,pb, label = '% Bias',marker='o',color='blue')
    ax2.axhline(y=0, ls='--', color='blue') 
    ax2.tick_params(axis='y', labelcolor='blue')
    #ax2.set_ylabel('Percent bias')
    ax2.legend(bbox_to_anchor=(.5, .95), loc='best', prop={'size':10}, frameon=False) 
    #ax2.set_yticklabels(ax2.get_yticks(), weight='bold')
    fig.canvas.draw()
    
    ax[1].plot(np.arange(14)+1, r, label = 'Correlation',marker='o',color='black')
    ax[1].plot(np.arange(14)+1, pv, label = 'p-value',marker='o',color='blue')
    
    ax[1].set_xticks(np.arange(14)+1)
    ax[1].set_xticklabels(sts)
    
    ax[1].legend(bbox_to_anchor=(.4, .99), loc='best', prop={'size':10}, frameon=False) 
    
    ax[0].text(10.5, 7.3, s='mean1='+str(round(np.mean(rmse),2)), color='black') 
    ax[0].text(10.5, 7, s='mean2='+str(round(np.mean(pb),2)), color='blue') 
    ax[1].text(11, 0.5, s='mean='+str(round(np.mean(r),2)), color='black') 
    
    ax[0].set_xlabel('Station ID')
    ax[1].set_xlabel('Station ID')
    
    plt.title(obs_name+' rainfall dataset vs 14 Stations')
    plt.subplots_adjust(hspace=.23,wspace=.3)
   
    print("RMSE:", rmse)
    print("MAE:", mae)
    print("Percent Bias:", pb)
    print("Correlation:", r)
    print("P-value:", pv)
    
    print("rRMSE:", np.mean(rmse))
    print("rMAE:", np.mean(mae))
    print("rPercent Bias:", np.mean(pb))
    print("rCorrelation:", np.mean(r))
    plt.show()
    
def station2(obs_dataset, obs_name, model_datasets, model_names, workdir):
    #rmse for 5obs vs 14 station
    
    import xarray as xr
    from scipy import stats
    
    fig, ax = plt.subplots(nrows=1, ncols=2 ,figsize=(8,6))
    sts= ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14']
              
    dss = xr.open_dataset('D:/bmkg_csv/sum17/Pr_bmkg_1981-2005.nc')
        #karna model nya monthly maka
        #dss=dss.sortby('lat','lon')
       
    dss = dss.resample(time='1M').mean()
    
    lats=dss.lat.values
    
    for i2 in np.arange(len(model_datasets)):
        ds = xr.DataArray(model_datasets[i2].values,
        coords={'time': model_datasets[i2].times,
                'lat': model_datasets[i2].lats, 
                'lon': model_datasets[i2].lons},
        dims=["time", "lat", "lon"])
        
        print('model:',model_names[i2])
      
        #n=np.argsort(lats)[::-1]
        n=0
        rmse=[]
        r=[]
        mm=['>','s','x','*','+', 'o']
        mm2=['-','-.',':','--','--']
        # i dimulai dari lat rendah agar teratur
        for i in np.argsort(lats)[::-1]:
        #for i in np.arange(len(dss.lat)):
            n=n+1
            print('i=', i)
            
          
            #-----Count only 1 center------- 
            ed = ds.sel(lat=dss.lat[i], lon=dss.lon[i], method='nearest')
            predicted = np.array(ed.values)
            #-------------------------------
            observed = np.array(dss.pr[:,i,i].values)
                         
            #RSME
            differences = predicted - observed
            print('')
           
            squared_differences = differences**2
      
            rmse1 = np.sqrt(np.nanmean(squared_differences))
            rmse.append(round(rmse1,2))
            
            #R
            x0=observed
            y0=predicted
            #print(x0, y0)
                        
            bad = ~np.logical_or(np.isnan(x0), np.isnan(y0))
            x1=np.compress(bad, x0) 
            y1=np.compress(bad, y0)
            rr, p_value = stats.pearsonr(x1, y1)
            r.append(round(rr,2))
            '''
            ax[0].plot(n,rmse1, color='k', marker=mm[i2])#,label = model_names[i2])
            #ax[0].legend(bbox_to_anchor=(0.2, .99), loc='best', prop={'size':10}, frameon=False) 
            ax[0].set_xticks(np.arange(14)+1)
            ax[0].set_xticklabels(sts)
            ax[0].set_xlabel('Station ID')
            
            ax[1].plot(n,rr, color='k', marker=mm[i2])#,label = model_names[i2])
            ax[1].legend(bbox_to_anchor=(0.99, .3), loc='best', prop={'size':10}, frameon=False) 
            ax[1].set_xticks(np.arange(14)+1)
            ax[1].set_xticklabels(sts)
            ax[1].set_xlabel('Station ID')
            '''
        '''
        ax[0].plot(np.arange(14)+1,rmse,marker='o',label=model_names[i2])
        #ax[0].legend(bbox_to_anchor=(0.2, .99), loc='best', prop={'size':10}, frameon=False) 
        ax[0].set_xticks(np.arange(14)+1)
        ax[0].set_xticklabels(sts)
        ax[0].set_xlabel('Station ID')
        
        ax[1].plot(np.arange(14)+1,r,marker='o',label=model_names[i2])
        ax[1].legend(bbox_to_anchor=(0.99, .3), loc='best', prop={'size':10}, frameon=False) 
        ax[1].set_xticks(np.arange(14)+1)
        ax[1].set_xticklabels(sts)
        ax[1].set_xlabel('Station ID')
        '''
        
        ax[0].scatter(np.arange(14)+1,rmse, color='k', marker=mm[i2],label=model_names[i2])
        #ax[0].legend(bbox_to_anchor=(0.2, .99), loc='best', prop={'size':10}, frameon=False) 
        ax[0].set_xticks(np.arange(14)+1)
        ax[0].set_xticklabels(sts)
        ax[0].set_xlabel('Station ID')
        
        ax[1].scatter(np.arange(14)+1,r, color='k', marker=mm[i2],label=model_names[i2])
        ax[1].legend(bbox_to_anchor=(0.99, .3), loc='best', prop={'size':10}, frameon=False) 
        ax[1].set_xticks(np.arange(14)+1)
        ax[1].set_xticklabels(sts)
        ax[1].set_xlabel('Station ID')
    
        #plt.title('RMSE for 5 obs vs 14 Stations')
        ax[0].set_ylabel('RMSE (mm/month)')
        ax[1].set_ylabel('Correlation coefficient')
    plt.show()
    
def station_5obs(obs_dataset, obs_name, model_datasets, model_names, workdir):
    #5obs vs 14 stations in Sumatera
    #hitung rata rmse dkk di 14 titik sts
   
    import xarray as xr
    from scipy import stats
    fig, ax = plt.subplots(nrows=1, ncols=1 ,figsize=(8,6))
    mm=['','>','s','x','o','+', '*']
    
    rmse_=[]
    pb_=[]
    r_=[]
    pv_=[]
    
    dss = xr.open_dataset('D:/bmkg_csv/sum17/Pr_bmkg_1981-2005.nc')
        #karna model nya monthly maka
        #dss=dss.sortby('lat','lon')   
    dss = dss.resample(time='1M').mean()
    for i in np.arange(len(model_datasets)):
    
        ds = xr.DataArray(model_datasets[i].values,
        coords={'time': model_datasets[i].times,
                'lat': model_datasets[i].lats, 
                'lon': model_datasets[i].lons},
        dims=["time", "lat", "lon"])
    
        print('mod_name: ', model_names[i] )
       
       
        rmse=[]
        mae=[]
        pb=[]
        r=[]
        pv=[]
    
        lats=dss.lat.values
        #n=np.argsort(lats)[::-1]
        n=0
        # i dimulai dari lat rendah agar teratur
        for i in np.argsort(lats)[::-1]:
        #for i in np.arange(len(dss.lat)):
            n=n+1
            print('i=', i)
           
            #-----Count only 1 center------- 
            ed = ds.sel(lat=dss.lat[i], lon=dss.lon[i], method='nearest')
            #print(ed)
            predicted = np.array(ed.values)
            #-------------------------------
            observed = np.array(dss.pr[:,i,i].values)
                         
            # Calculate differences between observed and predicted
            # xr jika ada nan ini hasil kosong =[]
            # maka pakai np array diatas ######
            differences = predicted - observed
            print('')
            #print('pre, obs, diff',predicted, observed, differences)

            # Calculate squared differences for RMSE
            squared_differences = differences**2
            #print('squared_differences', squared_differences)

            # Calculate RMSE
            rmse1 = np.sqrt(np.nanmean(squared_differences))
            rmse.append(round(rmse1,2))
            
            # Calculate MAE
            mae1 = np.nanmean(np.abs(differences))
            mae.append(round(mae1,2))

            # Calculate Percent Bias
            percent_bias = (np.nansum(differences) / np.nansum(observed)) * 100
            pb.append(round(percent_bias,2))
            
            # Calculate Correlation and p-value
            # ini sensitif pada panjang data dan NaNs
            x0=observed
            y0=predicted
            #print(x0, y0)
                        
            bad = ~np.logical_or(np.isnan(x0), np.isnan(y0))
            x1=np.compress(bad, x0) 
            y1=np.compress(bad, y0)
            rr, p_value = stats.pearsonr(x1, y1)
            r.append(round(rr,2))
            pv.append(round(p_value,3))
        rmse_.append(np.mean(rmse))
        pb_.append(np.mean(pb))
        r_.append(np.mean(r))
        pv_.append(np.mean(pv))
        
    ax.plot(model_names, rmse_, color='k', marker=mm[1],label = 'RMSE')
    ax.plot(model_names, pb_, color='k', marker=mm[2],label = 'PB')
    ax.plot(model_names, r_, color='k', marker=mm[3],label = 'R')
    ax.plot(model_names, pv_, color='k', marker=mm[4],label = 'p-value')
            
    plt.legend(bbox_to_anchor=(.2, .95), loc='best', prop={'size':8.5}, frameon=False) 
    
    print("rRMSE:", rmse_)
    print("rMAE:", np.mean(mae))
    print("rPercent Bias:", np.mean(pb))
    print("rCorrelation:", np.mean(r))
    plt.show()
    
def station_5obs2(obs_dataset, obs_name, model_datasets, model_names, workdir):
    #monthly 5obs vs 14 stations in Sumatera
    #hitung rata rmse dkk di 14 titik sts
    #ini error tidak jalan pada 5obs2 p25 ??? pada i=0 error
   
    import xarray as xr
    from scipy import stats
    fig, ax = plt.subplots(nrows=1, ncols=1 ,figsize=(8,6))
    mm=['','>','s','x','o','+', '*']
    
    rmse_=[]
    pb_=[]
    r_=[]
    pv_=[]
    
    dss = xr.open_dataset('D:/bmkg_csv/sum17/Pr_bmkg_1981-2005.nc')
        #karna model nya monthly maka
        #dss=dss.sortby('lat','lon')   
    dss = dss.resample(time='1M').mean()
    for i in np.arange(len(model_datasets)):
    
        ds = xr.DataArray(model_datasets[i].values,
        coords={'time': model_datasets[i].times,
                'lat': model_datasets[i].lats, 
                'lon': model_datasets[i].lons},
        dims=["time", "lat", "lon"])
    
        print('mod_name: ', model_names[i] )
       
       
        rmse=[]
        mae=[]
        pb=[]
        r=[]
        pv=[]
    
        lats=dss.lat.values
        #n=np.argsort(lats)[::-1]
        n=0
        # i dimulai dari lat rendah agar teratur
        #for i in np.argsort(lats)[::-1]:
        for i in np.arange(len(dss.lat)):
            n=n+1
            print('i=', i)
            #pada 3 ini error maka dilompat ke i+1
            #untuk eror terakhir di break keluar loop
            if i==1: i=i+1
            if i==3: i=i+1
            if i==11: break 
            observed = np.array(dss.pr[:,i,i].values)
           
           
            ed = ds.sel(lat=dss.lat[i], lon=dss.lon[i], method='nearest')
            print(np.nanmean(ed))
            
            #if np.nanmean(ed)==np.nan: break
            
            predicted = np.array(ed.values)
         
            # Calculate differences between observed and predicted
            # xr jika ada nan ini hasil kosong =[]
            # maka pakai np array diatas ######
            differences = predicted - observed
            print('')
            #print('pre, obs, diff',predicted, observed, differences)

            # Calculate squared differences for RMSE
            squared_differences = differences**2
            #print('squared_differences', squared_differences)

            # Calculate RMSE
            rmse1 = np.sqrt(np.nanmean(squared_differences))
            rmse.append(round(rmse1,2))
            
            # Calculate MAE
            mae1 = np.nanmean(np.abs(differences))
            mae.append(round(mae1,2))

            # Calculate Percent Bias
            percent_bias = (np.nansum(differences) / np.nansum(observed)) * 100
            pb.append(round(percent_bias,2))
            
            # Calculate Correlation and p-value
            # ini sensitif pada panjang data dan NaNs
            x0=observed
            y0=predicted
            #print(x0, y0)
                        
            bad = ~np.logical_or(np.isnan(x0), np.isnan(y0))
            x1=np.compress(bad, x0) 
            y1=np.compress(bad, y0)
            rr, p_value = stats.pearsonr(x1, y1)
            r.append(round(rr,2))
            pv.append(round(p_value,3))
        rmse_.append(np.mean(rmse))
        pb_.append(np.mean(pb))
        r_.append(np.mean(r))
        pv_.append(np.mean(pv))
        
    ax.plot(model_names, rmse_, color='k', marker=mm[1],label = 'RMSE')
    ax.plot(model_names, pb_, color='k', marker=mm[2],label = 'PB')
    ax2 = ax.twinx()
    ax2.plot(model_names, r_, color='blue', marker=mm[3],label = 'R')
    ax2.plot(model_names, pv_, color='blue', marker=mm[4],label = 'p-value')
    ax.axhline(0, linestyle = 'dashed', color='red')
    
    ax2.tick_params(axis='y', labelcolor='blue')
    #ax2.set_ylabel('Percent bias')
    
    ax.legend(bbox_to_anchor=(.17, .62), loc='best', prop={'size':10}, frameon=False)
    ax2.legend(bbox_to_anchor=(.97, .62), loc=None, prop={'size':10}, frameon=False)     
            
    #plt.legend(bbox_to_anchor=(.2, .95), loc='best', prop={'size':8.5}, frameon=False) 
    #plt.legend(loc='best', prop={'size':8.5}, frameon=False) 
    
    print("rRMSE:", rmse_)
    print("rMAE:", np.mean(mae))
    print("rPercent Bias:", np.mean(pb))
    print("rCorrelation:", np.mean(r))
    plt.show()
 
def obs5_rmsd(obs_dataset, obs_name, model_datasets, model_names, workdir):
    #Juneng 2016
    #RMSD between the 4 products ?? over land
    #how ?
    #rmse ...m-o, pilih o as r, hitung 4obs vs r, hasil rata2kan ?
    
   
    import xarray as xr
    from scipy import stats
    fig, ax = plt.subplots(nrows=1, ncols=1 ,figsize=(8,6))
    mm=['','>','s','x','o','+', '*']
    
    rmse_=[]
    pb_=[]
    r_=[]
    pv_=[]
    for i in np.arange(len(model_datasets)):
    
        ds = xr.DataArray(model_datasets[i].values,
        coords={'time': model_datasets[i].times,
                'lat': model_datasets[i].lats, 
                'lon': model_datasets[i].lons},
        dims=["time", "lat", "lon"]) 
  
  
def ac_station(obs_dataset, obs_name, model_datasets, model_names, workdir):
    #ts
    import xarray as xr
    from scipy import stats
    
    print('obs_name: ', obs_name )
    
    ds = xr.DataArray(obs_dataset.values,
    coords={'time': obs_dataset.times,
            'lat': obs_dataset.lats, 
            'lon': obs_dataset.lons},
    dims=["time", "lat", "lon"])
       
        
    
    fig, ax = plt.subplots(nrows=1, ncols=2 ,figsize=(8,6))
    
       
    dss = xr.open_dataset('D:/bmkg_csv/sum17/Pr_bmkg_1981-2005.nc')
    #karna model nya monthly maka
    #dss=dss.sortby('lat','lon')
    
    dss = dss.resample(time='1M').mean()
    
        
    x_tick=['J','F','M','A','M','J','J','A','S','O','N','D']
    mm=['*','+','x','>','o', 's']
    mm2=[':','-.','--']
    
    lats=dss.lat.values
    #n=np.argsort(lats)[::-1]
    n=0
    # i dimulai dari lat rendah agar teratur
    for i in np.argsort(lats)[::-1]:
    #for i in np.arange(len(dss.lat)):
        n=n+1
        print('i=', n)
        #print('dss.lat[i]',dss.lat[i])
        #print(ds)
          
        #-----Count only 1 center------- 
        ed = ds.sel(lat=dss.lat[i].data, lon=dss.lon[i].data, method='nearest')
        #print(ed)
        ds1 = ed.groupby('time.month').mean()
        #ds1 = ds1.mean(dim=("lat", "lon"))
        
       
        #-------------------------------
        obs = dss.pr[:,i,i]
        dss1 = obs.groupby('time.month').mean()
        #dss1 = dss1.mean(dim=("lat", "lon"))
        if n<4: 
            ls=mm2[n-1] 
            cc='black'
            mr=None
        elif n<10:
            ls=None
            cc='black'
            mr=mm[n-4]
        else:
            ls=None
            cc=None
            mr=None
        
        ax[0].plot(ds1.month, ds1.values, marker=mr, linestyle=ls, color=cc, lw=2, label = obs_name)
        ax[1].plot(ds1.month, dss1.values, marker=mr, linestyle=ls, color=cc, lw=2, label = 'Sts_'+str(n))
        if n==1:
           ax[0].plot(ds1.month, ds1.values, marker=mr, linestyle=':', color=cc, lw=2, label = obs_name)
    ax[0].set_xticks(np.arange(12)+1)
    ax[0].set_xticklabels(x_tick, fontsize=8)
    ax[0].set_ylabel('Mean rainfall (mm/month)')
    ax[0].set_title(obs_name+' rainfall in Sumatera')
    ax[1].set_xticks(np.arange(12)+1)
    ax[1].set_xticklabels(x_tick, fontsize=8)
    plt.legend(bbox_to_anchor=(1, .6), loc='best', prop={'size':8.5}, frameon=False) 
    plt.subplots_adjust(hspace=.23,wspace=.2)
    ax[1].set_title('14 Stations rainfall in Sumatera')
   

    plt.show()
  
def eofs_pca(obs_dataset, obs_name, model_datasets, model_names, workdir):
    from eofs2.xarray import Eof
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    
    #choose
    annual_cycle=False
    season=False
    musim='JJA'
    monthly=False
    yearly=True
    
    ds = xr.DataArray(obs_dataset.values,
    coords={'time': obs_dataset.times,
            'lat': obs_dataset.lats, 
            'lon': obs_dataset.lons},
    dims=["time", "lat", "lon"])
    
    print('1 ds.min_max=',ds.min().data,ds.max().data)
    
    #Calculate monthly anomalies: 
    #https://docs.xarray.dev/en/stable/examples/weather-data.html
    #In climatology, “anomalies” refer to the difference between 
    #observations and typical weather for a particular season. 
    #Unlike observations, anomalies should not show any seasonal cycle.
    
    #climatology = ds.groupby("time.month").mean("time")
    #ds = ds.groupby("time.month") - climatology
    
    climatology_mean = ds.groupby("time.month").mean("time")
    climatology_std = ds.groupby("time.month").std("time")
    ds = xr.apply_ufunc(
        lambda x, m, s: 
        (x - m) / s,
        ds.groupby("time.month"),
        climatology_mean,
        climatology_std,
    )
    
    print('ds=',ds)
    print('2 ds.min_max=',ds.min().data,ds.max().data)
    
    #print('obs_name')
    #ds= ds.stack(z=("lat", "lon"))
    
    if annual_cycle:
        dm = ds.groupby('time.month').mean()
        dm=dm.rename({'month':'time'})
        dm=dm.rename('SEAR')
        solver = Eof(dm)
        fn='annual_cycle'
    if monthly:
        
        #da.assign_coords(lon=(((da.lon + 180) % 360) - 180))
        #ds=ds.assign_coords(time=np.arange(300))
        ds=ds.drop_vars('time')
        print('3 ds.min_max=',ds.min().data,ds.max().data)
        ds=ds.rename({'month':'time'}) #if climatology anomalies used
        ds=ds.assign_coords(time=np.arange(300)) #1981,2006))
        ds=ds.rename('SEAR')
        solver = Eof(ds)
        fn='monthly'
        print('ds=',ds)
        #dm=ds
    
    if season:
        ds = ds.groupby('time.season')#.mean() 
        ds=ds[musim]
        ds=ds.drop_vars('time')
        print('3 ds.min_max=',ds.min().data,ds.max().data)
        ds=ds.rename({'month':'time'})
        ds=ds.assign_coords(time=np.arange(75))
        
        fn='season'
        
        #ds=ds[2]
        print('ds=',ds)
        ds=ds.rename('SEA-RA')
        solver = Eof(ds)
    
    if yearly:
        ds = ds.groupby('time.year').sum()
        ds=ds.rename({'year':'time'})
        print('ds=',ds)
        ds=ds.rename('SEA-RA')
        solver = Eof(ds)
        fn='yearly'
    
    
    #print('dm.shape', dm)
    #print('dm.values.shape', dm.values)
    
      
    eof = solver.eofsAsCorrelation(neofs=1)
    pc = solver.pcs(npcs=1, pcscaling=0) #1 with scaling
    print('eof=',eof)
    print('solver.varianceFraction=', solver.varianceFraction())
    
    print('pc=',pc)
    
      
    clevs = np.linspace(0.5, 1, 5)
    
    for i in [0]: #len(eof):
        plt.figure()
        ax = plt.axes(projection=ccrs.PlateCarree())
        fill = eof[i].plot.contourf(ax=ax, 
                             #levels=clevs, 
                             #title=musim,
                             #add_colorbar=False, 
                             transform=ccrs.PlateCarree(),
                             #colors='brown'
                             )
        
        #plt.clabel(fill, inline=True, fontsize=8)
        #ax.set_title(musim)
        
        ax.add_feature(cfeature.LAND, facecolor='w', edgecolor='k')
        #cb = plt.colorbar(fill, orientation='horizontal')
        #cb.set_label('correlation coefficient', fontsize=12)
        ax.set_title('EOF'+str(i)+' expressed as correlation_'+fn, pad=20, fontsize=16)
        #ax.set_xlabel(fn)
        plt.savefig(workdir+reg+'_xEOF_'+str(i)+'_'+obs_name+'_'+fn,dpi=300,bbox_inches='tight')
    
    for i in [0]: #np.arange(len(pc)-1):
        print('i=', i, len(pc))
        plt.figure()
        pc[:, i].plot(color='b', linewidth=2)
        ax = plt.gca()
        ax.axhline(0, color='k')
        #ax.set_ylim(-3, 3)
        ax.set_xlabel('month')
        ax.set_ylabel('Normalized Units')
        ax.set_title('PC'+str(i)+' Time Series_'+fn, pad=20, fontsize=16)
       
        plt.savefig(workdir+reg+'_xPCA_ts_'+str(i)+'_'+obs_name+'_'+fn,dpi=300,bbox_inches='tight')
    
    '''
    eof.plot()
    plt.show()
    
    pc.plot()
    plt.show()
    
    clevs = np.linspace(-1, 1, 11)
    for i in len(eof):
        plt.figure()
        ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=100))
        fill = eof[i].plot.contour(ax=ax, levels=clevs, cmap=plt.cm.RdBu_r,
                                     add_colorbar=False, transform=ccrs.PlateCarree())
        
        ax.add_feature(cfeature.LAND, facecolor='black', edgecolor='k')
       
        cb = plt.colorbar(fill, orientation='horizontal')
        cb.set_label('correlation coefficient', fontsize=12)
        ax.set_title('EOF'+str(i)+' expressed as correlation', fontsize=16)
        
        plt.savefig(workdir+reg+'_EOF_'+str(i)+'_'+obs_name+'_'+fn,dpi=300,bbox_inches='tight')
    
    
    
    # Plot the leading PC time series.
       
    for i in [0,1,2]: # len(pc):
        plt.figure()
        pc[:, i].plot(color='b', linewidth=2)
        ax = plt.gca()
        ax.axhline(0, color='k')
        #ax.set_ylim(-3, 3)
        ax.set_xlabel('month')
        ax.set_ylabel('Normalized Units')
        ax.set_title('PC'+str(i)+' Time Series', fontsize=16)
        
        plt.savefig(workdir+reg+'_PCA_ts_'+str(i)+'_'+obs_name+'_'+fn,dpi=300,bbox_inches='tight')
    
    
    eof = solver.eofsAsCovariance(neofs=3)
    for i in [0,1,2]: #len(eof):
        plt.figure()
        ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=100))
        fill = eof[i].plot.contourf(ax=ax, levels=clevs, cmap=plt.cm.RdBu_r,
                                     add_colorbar=False, transform=ccrs.PlateCarree())
        
        ax.add_feature(cfeature.LAND, facecolor='w', edgecolor='k')
        cb = plt.colorbar(fill, orientation='horizontal')
        cb.set_label('covariance coefficient', fontsize=12)
        ax.set_title('EOF'+str(i)+' expressed as covariance', fontsize=16)
        
        plt.savefig(workdir+reg+'_EOF-C_'+str(i)+'_'+obs_name+'_'+fn,dpi=300,bbox_inches='tight')
    '''
    plt.show()
    
def eofs_multi(obs_dataset, obs_name, model_datasets, model_names, workdir):
    from eofs2.xarray import Eof
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    
    #choose
    annual_cycle=False
    season=False
    musim='JJA'
    monthly=False
    
    yearly=True #
    
    anom=1
    
    n_eofs=3 #1,2,3..
    n_ke=0 #0,1,2..
    
    fig, ax = plt.subplots(2,len(model_datasets)) #, figsize=(8,8))
    
    for i in np.arange(len(model_datasets)):
        print(model_names[i])
     
        ds = xr.DataArray(model_datasets[i].values,
        coords={'time': model_datasets[i].times,
                'lat': model_datasets[i].lats, 
                'lon': model_datasets[i].lons},
        dims=["time", "lat", "lon"])
        
        lat_min = ds.lat.min()
        lat_max = ds.lat.max()
        lon_min = ds.lon.min()
        lon_max = ds.lon.max()
        x,y = np.meshgrid(ds.lon, ds.lat)
        
        m = Basemap(ax=ax[1,i], projection ='cyl', 
                llcrnrlat = lat_min, #+1*0.22, 
                urcrnrlat = lat_max, #-6*0.22,
                llcrnrlon = lon_min, #+4*0.22, 
                urcrnrlon = lon_max, #-3*0.22, 
                resolution = 'l', fix_aspect=False)
        m.drawcoastlines(linewidth=1)
        m.drawcountries(linewidth=1)
        m.drawstates(linewidth=0.5, color='w')
    
        #print('1 ds.min_max=',ds.min().data,ds.max().data)
        
        #Calculate monthly anomalies: 
        #https://docs.xarray.dev/en/stable/examples/weather-data.html
        #In climatology, “anomalies” refer to the difference between 
        #observations and typical weather for a particular season. 
        #Unlike observations, anomalies should not show any seasonal cycle.
        
        #climatology = ds.groupby("time.month").mean("time")
        #ds = ds.groupby("time.month") - climatology
        fnn=''
        if anom==1:
            climatology_mean = ds.groupby("time.month").mean("time")
            climatology_std = ds.groupby("time.month").std("time")
            ds = xr.apply_ufunc(
                lambda x, m, s: 
                (x - m) / s,
                ds.groupby("time.month"),
                climatology_mean,
                climatology_std,
            )
            fnn='anom'
        
        #print('ds=',ds)
        #print('2 ds.min_max=',ds.min().data,ds.max().data)
        
        #print('obs_name')
        #ds= ds.stack(z=("lat", "lon"))
        
        if annual_cycle:
            dm = ds.groupby('time.month').mean()
            dm=dm.rename({'month':'time'})
            dm=dm.rename('SEAR')
            solver = Eof(dm)
            fn='annual_cycle'
        if monthly:
            
            #da.assign_coords(lon=(((da.lon + 180) % 360) - 180))
            #ds=ds.assign_coords(time=np.arange(300))
            ds=ds.drop_vars('time')
            #print('3 ds.min_max=',ds.min().data,ds.max().data)
            if anom==1: ds=ds.rename({'month':'time'}) #if climatology anomalies used
            ds=ds.assign_coords(time=np.arange(300)) #1981,2006))
            ds=ds.rename('SEAR')
            solver = Eof(ds)
            fn='monthly'
            #print('ds=',ds)
            #dm=ds
        
        if season:
            ds = ds.groupby('time.season')#.mean() 
            ds=ds[musim]
            ds=ds.drop_vars('time')
            print('3 ds.min_max=',ds.min().data,ds.max().data)
            ds=ds.rename({'month':'time'})
            ds=ds.assign_coords(time=np.arange(75))
            
            fn='season'
            
            #ds=ds[2]
            print('ds=',ds)
            ds=ds.rename('SEA-RA')
            solver = Eof(ds)
        
        if yearly:
            ds = ds.groupby('time.year').sum()
            ds=ds.rename({'year':'time'})
            #print('ds=',ds)
            ds=ds.rename('SEA-RA')
            solver = Eof(ds)
            fn='yearly'
        
        #print('dm.shape', dm)
        #print('dm.values.shape', dm.values)
        
        eof = solver.eofsAsCorrelation(neofs=n_eofs)
        pc = solver.pcs(npcs=n_eofs, pcscaling=0) #1 with scaling
        
        #print('eof=',eof)
        print('solver.varianceFraction=', solver.varianceFraction(3))
        
        #print('pc=',pc)
        
        p = eof[n_ke].plot.contourf(ax=ax[1,i], add_colorbar=False)
       
        #ax.add_feature(cfeature.LAND, facecolor='w', edgecolor='k')
        #cb = plt.colorbar(fill, orientation='horizontal')
        #cb.set_label('correlation coefficient', fontsize=12)
        
        #ax.set_xlabel(fn)
        
    
       
        pc[:, n_ke].plot(ax=ax[0,i], color='b', linewidth=2)
        #ax[0,i] = plt.gca()
        ax[0,i].axhline(0, color='k')
        #ax.set_ylim(-3, 3)
        #ax.set_xlabel('month')
        if anom==1: 
            ax[0,0].set_ylabel('Normalized Units')
        else:
            ax[0,0].set_ylabel('Un-normalized Units')
        #ax.set_title('PC'+str(i)+' Time Series_'+fn, pad=20, fontsize=16)
        
        ax[0,i].tick_params(axis='x', pad=1,labelsize=8)
        ax[0,i].tick_params(axis='y', pad=1,labelsize=8)
        
        #tt=' ['+str(n_ke+1)+']'+' ['+str(np.round(solver.varianceFraction()[n_ke].data,2)*100)+'%]'   
        
        tt='  Mode='+str(n_ke+1)+'  vf='+str(np.round(solver.varianceFraction()[n_ke].data*100,2))+'%'
        
        ax[0,i].set_title(model_names[i]+tt, fontsize=9)
        
        ax[1,i].set_xlabel('')
        ax[1,i].set_ylabel('')
        ax[0,i].set_ylabel('')
        ax[0,i].set_xlabel('')
        
        '''
        ax[1,i].set_xticks([100,120,140])
        ax[1,i].set_xticklabels(['100E','120E','140E'])
        ax[1,i].tick_params(axis='x', pad=1,labelsize=8)
        ax[1,i].set_title('')
        
        ax[1,0].set_yticks([-10,0,10,20])
        ax[1,0].set_yticklabels(['10S','0','10N','20N'])
        ax[1,0].tick_params(axis='y', pad=1,labelsize=8)
        '''
        
        ax[1,i].set_xticks([95,100,105])
        ax[1,i].set_xticklabels(['95$^\circ$E','100$^\circ$E','105$^\circ$E'])
        ax[1,i].tick_params(axis='x', pad=1,labelsize=8)
        ax[1,i].set_title('')
        
        ax[1,0].set_yticks([-5,0,5])
        ax[1,0].set_yticklabels(['5$^\circ$S','0','5$^\circ$N'])
        ax[1,0].tick_params(axis='y', pad=1,labelsize=8)
        
        if i>0: ax[0,i].set_yticks([])
    
    
    #plt.subplots_adjust(right=.5)
    plt.subplots_adjust(hspace=.2,wspace=.1)
    
    cax = fig.add_axes([0.91, 0.11, 0.015, 0.345])
   
    cbar = plt.colorbar(p,cax=cax)

    #cbar.ax.get_yaxis().set_ticks([])
        
    #ax[0].set_title('EOF'+str(1)+' expressed as correlation_'+fn, pad=20, fontsize=16)
    plt.savefig(workdir+reg+'_EOF_'+str(n_ke)+'_3tos_'+fn+'_'+fnn, dpi=300,bbox_inches='tight')
        
    plt.show()
    
def eofs_multi_tos_nino(obs_dataset, obs_name, model_datasets, model_names, workdir):
    from eofs2.xarray import Eof
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    
    #choose
    annual_cycle=False
    season=False
    musim='JJA'
    monthly=False
    yearly=True
    
    anom=1
    
    n_eofs=3 #1,2,3..
    n_ke=0 #0,1,2..
    
    fig, ax = plt.subplots(2,3) #, figsize=(8,8))
    
    for i in np.arange(len(model_datasets)):
        print(model_names[i])
     
        ds = xr.DataArray(model_datasets[i].values,
        coords={'time': model_datasets[i].times,
                'lat': model_datasets[i].lats, 
                'lon': model_datasets[i].lons},
        dims=["time", "lat", "lon"])
        
        lat_min = ds.lat.min()
        lat_max = ds.lat.max()
        lon_min = ds.lon.min()
        lon_max = ds.lon.max()
        x,y = np.meshgrid(ds.lon, ds.lat)
        
        m = Basemap(ax=ax[1,i], projection ='cyl', 
                llcrnrlat = lat_min, #+1*0.22, 
                urcrnrlat = lat_max, #-6*0.22,
                llcrnrlon = lon_min, #+4*0.22, 
                urcrnrlon = lon_max, #-3*0.22, 
                resolution = 'l', fix_aspect=False)
        m.drawcoastlines(linewidth=1)
        m.drawcountries(linewidth=1)
        m.drawstates(linewidth=0.5, color='w')
    
        #print('1 ds.min_max=',ds.min().data,ds.max().data)
        
        #Calculate monthly anomalies: 
        #https://docs.xarray.dev/en/stable/examples/weather-data.html
        #In climatology, “anomalies” refer to the difference between 
        #observations and typical weather for a particular season. 
        #Unlike observations, anomalies should not show any seasonal cycle.
        
        #climatology = ds.groupby("time.month").mean("time")
        #ds = ds.groupby("time.month") - climatology
        fnn=''
        if anom==1:
            climatology_mean = ds.groupby("time.month").mean("time")
            climatology_std = ds.groupby("time.month").std("time")
            ds = xr.apply_ufunc(
                lambda x, m, s: 
                (x - m) / s,
                ds.groupby("time.month"),
                climatology_mean,
                climatology_std,
            )
            fnn='anom'
        
        #print('ds=',ds)
        #print('2 ds.min_max=',ds.min().data,ds.max().data)
        
        #print('obs_name')
        #ds= ds.stack(z=("lat", "lon"))
        
        if annual_cycle:
            dm = ds.groupby('time.month').mean()
            dm=dm.rename({'month':'time'})
            dm=dm.rename('SEAR')
            solver = Eof(dm)
            fn='annual_cycle'
        if monthly:
            
            #da.assign_coords(lon=(((da.lon + 180) % 360) - 180))
            #ds=ds.assign_coords(time=np.arange(300))
            ds=ds.drop_vars('time')
            #print('3 ds.min_max=',ds.min().data,ds.max().data)
            if anom==1: ds=ds.rename({'month':'time'}) #if climatology anomalies used
            ds=ds.assign_coords(time=np.arange(300)) #1981,2006))
            ds=ds.rename('SEAR')
            solver = Eof(ds)
            fn='monthly'
            #print('ds=',ds)
            #dm=ds
        
        if season:
            ds = ds.groupby('time.season')#.mean() 
            ds=ds[musim]
            ds=ds.drop_vars('time')
            print('3 ds.min_max=',ds.min().data,ds.max().data)
            ds=ds.rename({'month':'time'})
            ds=ds.assign_coords(time=np.arange(75))
            
            fn='season'
            
            #ds=ds[2]
            print('ds=',ds)
            ds=ds.rename('SEA-RA')
            solver = Eof(ds)
        
        if yearly:
            ds = ds.groupby('time.year').sum()
            ds=ds.rename({'year':'time'})
            #print('ds=',ds)
            ds=ds.rename('SEA-RA')
            solver = Eof(ds)
            fn='yearly'
        print('ds.lon.min(),ds.lon.max()=', ds.lon.min(),ds.lon.max())
        #print('dm.shape', dm)
        #print('dm.values.shape', dm.values)
        
        eof = solver.eofsAsCorrelation(neofs=n_eofs)
        pc = solver.pcs(npcs=n_eofs, pcscaling=0) #1 with scaling
        
        #print('eof=',eof)
        print('solver.varianceFraction=', solver.varianceFraction(3))
        
        #print('pc=',pc)
        levels =np.arange(0.8,1,.02)
        #cmap='rainbow'
        cmap='viridis'
        p = eof[n_ke].plot.contourf(ax=ax[1,i], add_colorbar=False)
        #p = eof[n_ke].plot.contourf(ax=ax[1,i], levels =levels , vmin=-1, vmax=1,cmap=cmap, add_colorbar=False)
       
        #ax.add_feature(cfeature.LAND, facecolor='w', edgecolor='k')
        #cb = plt.colorbar(fill, orientation='horizontal')
        #cb.set_label('correlation coefficient', fontsize=12)
        
        #ax.set_xlabel(fn)
        
    
       
        pc[:, n_ke].plot(ax=ax[0,i], color='b', linewidth=2)
        #ax[0,i] = plt.gca()
        ax[0,i].axhline(0, color='k')
        #ax.set_ylim(-3, 3)
        #ax.set_xlabel('month')
        if anom==1: 
            ax[0,0].set_ylabel('Normalized Units')
        else:
            ax[0,0].set_ylabel('Un-normalized Units')
        #ax.set_title('PC'+str(i)+' Time Series_'+fn, pad=20, fontsize=16)
        
        ax[0,i].tick_params(axis='x', pad=1,labelsize=8)
        ax[0,i].tick_params(axis='y', pad=1,labelsize=8)
        
        #tt=' ['+str(n_ke+1)+']'+' ['+str(np.round(solver.varianceFraction()[n_ke].data,2)*100)+'%]'   
        
        tt='  Mode='+str(n_ke+1)+'  EV='+str(np.round(solver.varianceFraction()[n_ke].data*100,2))+'%'
        
        ax[0,i].set_title(model_names[i]+tt, fontsize=9)
        
        ax[1,i].set_xlabel('')
        ax[1,i].set_ylabel('')
        ax[0,i].set_ylabel('')
        ax[0,i].set_xlabel('')
        
        #190-240 => -170 to -120
        ax[1,i].set_xticks([-160,-140,-120])
        ax[1,i].set_xticklabels(['160W','140W','120W'])
        ax[1,i].tick_params(axis='x', pad=1,labelsize=8)
        ax[1,i].set_title('')
        if i>0: ax[0,i].set_yticks([])
    #-5 5    
    ax[1,0].set_yticks([-5,0,5])
    ax[1,0].set_yticklabels(['5S','0','5N'])
    ax[1,0].tick_params(axis='y', pad=1,labelsize=8)
    
    
    
    #plt.subplots_adjust(right=.5)
    plt.subplots_adjust(hspace=.2,wspace=.1)
    
    cax = fig.add_axes([0.91, 0.11, 0.015, 0.345])
   
    cbar = plt.colorbar(p,cax=cax)
    
    
    plt.draw()
    #cbar.ax.get_yaxis().set_ticks([])
        
    #ax[0].set_title('EOF'+str(1)+' expressed as correlation_'+fn, pad=20, fontsize=16)
    plt.savefig(workdir+reg+'_EOF_'+str(n_ke)+'_3tos_new_'+fn+'_'+fnn, dpi=300,bbox_inches='tight')
    plt.show()

def corr_pc1(obs_dataset, obs_name, model_datasets, model_names, workdir):
    '''
    Correlation of the first leading principal component of 
    Nino3.4 and rainfall time-series.
    PCA SEA vs Nino34 for 1981-2005 data
    
    sebelum eof untuk melihar inter-annual maka remove seasonal, sudah??  
    '''
    from eofs2.xarray import Eof
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from scipy import stats
    import numpy.polynomial.polynomial as poly
    
    #choose
    annual_cycle=False
    season=False
    musim='JJA'
    monthly=False
    yearly=True
    
    anom=1
    
    n_eofs=2 #1,2,3..
    n_ke=1 #0,1,2..
    
    fig, ax = plt.subplots(2,4) #, figsize=(8,8))
    n=[0,2]
    nn=[1,3]
    for i in [0,1]: #np.arange(len(model_datasets)):
        print('')
        print('')
        print(model_names[n[i]])
        print(model_names[nn[i]])
     
        ds = xr.DataArray(model_datasets[n[i]].values,
        coords={'time': model_datasets[n[i]].times,
                'lat': model_datasets[n[i]].lats, 
                'lon': model_datasets[n[i]].lons},
        dims=["time", "lat", "lon"])
        
        lat_min = ds.lat.min()
        lat_max = ds.lat.max()
        lon_min = ds.lon.min()
        lon_max = ds.lon.max()
        x,y = np.meshgrid(ds.lon, ds.lat)
        
        m = Basemap(ax=ax[1,n[i]], projection ='cyl', 
                llcrnrlat = lat_min, #+1*0.22, 
                urcrnrlat = lat_max, #-6*0.22,
                llcrnrlon = lon_min, #+4*0.22, 
                urcrnrlon = lon_max, #-3*0.22, 
                resolution = 'l', fix_aspect=False)
        m.drawcoastlines(linewidth=1)
        m.drawcountries(linewidth=1)
        m.drawstates(linewidth=0.5, color='w')
        
       
        fnn=''
        if anom==1:
            climatology_mean = ds.groupby("time.month").mean("time")
            climatology_std = ds.groupby("time.month").std("time")
            ds = xr.apply_ufunc(
                lambda x, m, s: 
                (x - m) / s,
                ds.groupby("time.month"),
                climatology_mean,
                climatology_std,
            )
            fnn='anom'
    
        
        if annual_cycle:
            dm = ds.groupby('time.month').mean()
            dm=dm.rename({'month':'time'})
            dm=dm.rename('SEAR')
            solver = Eof(dm)
            fn='annual_cycle'
        if monthly:
   
            ds=ds.drop_vars('time')
            #print('3 ds.min_max=',ds.min().data,ds.max().data)
            if anom==1: ds=ds.rename({'month':'time'}) #if climatology anomalies used
            ds=ds.assign_coords(time=np.arange(300)) #1981,2006))
            ds=ds.rename('SEAR')
            solver = Eof(ds)
            fn='monthly'
     
        if season:
            ds = ds.groupby('time.season')#.mean() 
            ds=ds[musim]
            ds=ds.drop_vars('time')
            print('3 ds.min_max=',ds.min().data,ds.max().data)
            ds=ds.rename({'month':'time'})
            ds=ds.assign_coords(time=np.arange(75))
            
            fn='season'
            
            #ds=ds[2]
            print('ds=',ds)
            ds=ds.rename('SEA-RA')
            solver = Eof(ds)
        
        if yearly:
            ds = ds.groupby('time.year').sum()
            ds=ds.rename({'year':'time'})
            #print('ds=',ds)
            ds=ds.rename('SEA-RA')
            solver = Eof(ds)
            fn='yearly'
        #print('ds.lon.min(),ds.lon.max()=', ds.lon.min(),ds.lon.max())
     
        eof = solver.eofsAsCorrelation(neofs=n_eofs)
        pc = solver.pcs(npcs=n_eofs, pcscaling=0) #1 with scaling
        
        #print('eof=',eof)
        print('nino_varianceFraction=', solver.varianceFraction(3).data)
        
        #print('pc=',pc)
        levels =np.arange(0.8,1,.02)
        #cmap='rainbow'
        cmap='viridis'
        p = eof[n_ke].plot.contourf(ax=ax[1,n[i]], add_colorbar=False)
    
        pc[:, n_ke].plot(ax=ax[0,n[i]], color='b', linewidth=2)
        print(pc.shape)
        
        #ax[0,i] = plt.gca()
        ax[0,n[i]].axhline(0, color='k')
        #ax.set_ylim(-3, 3)
        #ax.set_xlabel('month')
        if anom==1: 
            ax[0,0].set_ylabel('Normalized Units')
        else:
            ax[0,0].set_ylabel('Un-normalized Units')
        #ax.set_title('PC'+str(i)+' Time Series_'+fn, pad=20, fontsize=16)
        
        ax[0,n[i]].tick_params(axis='x', pad=1,labelsize=8)
        ax[0,n[i]].tick_params(axis='y', pad=1,labelsize=8)
        ax[0,nn[i]].tick_params(axis='x', pad=1,labelsize=8)
        ax[0,nn[i]].tick_params(axis='y', pad=1,labelsize=8)
        
        ax[1,n[i]].tick_params(axis='x', pad=1,labelsize=8)
        ax[1,n[i]].tick_params(axis='y', pad=1,labelsize=8)
        ax[1,nn[i]].tick_params(axis='x', pad=1,labelsize=8)
        ax[1,nn[i]].tick_params(axis='y', pad=1,labelsize=8)
        
        #tt=' ['+str(n_ke+1)+']'+' ['+str(np.round(solver.varianceFraction()[n_ke].data,2)*100)+'%]'   
        
        tt='  Mode='+str(n_ke+1)+'  EV='+str(np.round(solver.varianceFraction()[n_ke].data*100,2))+'%'
        
        ax[0,n[i]].set_title(model_names[n[i]]+tt, fontsize=9)
        
        
        ax[0,n[i]].set_ylabel('')
        ax[0,n[i]].set_xlabel('')
        ax[1,n[i]].set_xlabel('')
        ax[1,n[i]].set_ylabel('')
        
        ax[1,n[i]].set_title('')
        ax[1,n[i]].set_title('')
    
        #----------------------- pr
        
        ds = xr.DataArray(model_datasets[nn[i]].values,
        coords={'time': model_datasets[nn[i]].times,
                'lat': model_datasets[nn[i]].lats, 
                'lon': model_datasets[nn[i]].lons},
        dims=["time", "lat", "lon"])
        
        lat_min = ds.lat.min()
        lat_max = ds.lat.max()
        lon_min = ds.lon.min()
        lon_max = ds.lon.max()
        x,y = np.meshgrid(ds.lon, ds.lat)
        
        m = Basemap(ax=ax[1,nn[i]], projection ='cyl', 
                llcrnrlat = lat_min, #+1*0.22, 
                urcrnrlat = lat_max, #-6*0.22,
                llcrnrlon = lon_min, #+4*0.22, 
                urcrnrlon = lon_max, #-3*0.22, 
                resolution = 'l', fix_aspect=False)
        m.drawcoastlines(linewidth=1)
        m.drawcountries(linewidth=1)
        m.drawstates(linewidth=0.5, color='w')
        
        if anom==1:
            climatology_mean = ds.groupby("time.month").mean("time")
            climatology_std = ds.groupby("time.month").std("time")
            ds = xr.apply_ufunc(
                lambda x, m, s: 
                (x - m) / s,
                ds.groupby("time.month"),
                climatology_mean,
                climatology_std,
            )
        
        
        if yearly:
            ds = ds.groupby('time.year').sum()
            ds=ds.rename({'year':'time'})
            #print('ds=',ds)
            ds=ds.rename('SEA-RA')
            solver = Eof(ds)
            fn='yearly'
       
        
        
        eof2 = solver.eofsAsCorrelation(neofs=n_eofs)
        pc2 = solver.pcs(npcs=n_eofs, pcscaling=0) #1 with scaling
        
        #print('eof=',eof)
        print('pr_varianceFraction=', solver.varianceFraction(3).data)
        
        p2 = eof2[n_ke].plot.contourf(ax=ax[1,nn[i]], add_colorbar=False)
        
        pc2[:, n_ke].plot(ax=ax[0,nn[i]], color='b', linewidth=2)
        print(pc2.shape)
        
        tt='  Mode='+str(n_ke+1)+'  EV='+str(np.round(solver.varianceFraction()[n_ke].data*100,2))+'%'
        
        ax[0,nn[i]].set_title(model_names[nn[i]]+tt, fontsize=9)
        
        r, p_value = stats.pearsonr(pc[:,n_ke], pc2[:,n_ke])
        
        
        
    
        print("Correlation:", r)
        print("P-value:", p_value)
        
        ax[0,nn[i]].axhline(0, color='k')
        ax[1,nn[i]].set_xlabel('')
        ax[1,nn[i]].set_ylabel('')
        if nn[i]!=0:
            ax[0,nn[i]].set_ylabel('')
        ax[0,nn[i]].set_xlabel('')
        
        
        ax[1,nn[i]].set_title('')
        
        
        ax[0,n[i]].set_yticks([])
        ax[0,nn[i]].set_yticks([])
        
        tt='r='+str(np.round(r,2))+'   pv='+str(np.round(p_value,4))
        
        ax[1,nn[i]].set_title(tt, fontsize=9)
        #ax[0,n[i]].text(1980, lon_min, s=tt, fontsize=9)
        
        ax[1,0].set_xlabel('Nino3.4 region')
        ax[1,1].set_xlabel('SEA region')
        ax[1,2].set_xlabel('Nino3.4 region')
        ax[1,3].set_xlabel('SEA region')
    
    
    
    #plt.subplots_adjust(right=.5)
    plt.subplots_adjust(hspace=.2,wspace=.15)
    
    cax = fig.add_axes([0.91, 0.4, 0.015, 0.25])
    cbar = plt.colorbar(p,cax=cax)
    cbar.ax.tick_params(labelsize=8)
    
    cax = fig.add_axes([0.91, 0.11, 0.015, 0.25])
    cbar = plt.colorbar(p2,cax=cax)
    cbar.ax.tick_params(labelsize=8)
    
    #cbar1.ax.xaxis.set_ticks_position('top')
    #cbar1.ax.xaxis.set_label_position('top')   
    
    
    plt.draw()
    #cbar.ax.get_yaxis().set_ticks([])
        
    #ax[0].set_title('EOF'+str(1)+' expressed as correlation_'+fn, pad=20, fontsize=16)
    plt.savefig(workdir+reg+'_EOF_'+str(n_ke)+'_3tos_new_'+fn+'_'+fnn, dpi=300,bbox_inches='tight')
    plt.show()
    
def corr_pc1_1(workdir):
    #lihat tele_pr
    '''
    Correlation of the first leading principal component of 
    Nino3.4 and rainfall time-series.
    PCA SEA vs Nino34 for 1981-2005 data
    
    sebelum eof untuk melihar inter-annual maka remove seasonal, sudah??  
    '''
    import numpy as np
    from scipy import signal
    import numpy.polynomial.polynomial as poly
    from netCDF4 import Dataset

    import matplotlib.pyplot as plt
    from mpl_toolkits.basemap import Basemap
    from eofs.standard import Eof
    from scipy import stats
    
    n_eofs=1 #1,2,3..
    n_ke=0 #0,1,2..
    
    infile2 = [
    'D:/data1/nino_1deg_1981-2005.nc',
    'D:/Cordex/RCMES/pr_eva_ds/pr_SEA_LO_2obs_moe_9model_mme_1981_2005.nc',
    'D:/data1/nino_1deg_1981-2005.nc',
    'D:/Cordex/RCMES/pr_eva_ds/pr_SEA_LO_2obs_moe_9model_mme_1981_2005.nc',
    #'D:/data1/pr_SEA_CNRM_CMIP5_1981_2005.nc',
    ]

    #IOD
    infile = [
    
    'D:/Cordex/RCMES/pr_eva_ds/pr_SEA_LO_2obs_moe_9model_mme_1981_2005.nc',
    'D:/data1/iod_1deg_1981-2005.nc',
    'D:/Cordex/RCMES/pr_eva_ds/pr_SEA_LO_2obs_moe_9model_mme_1981_2005.nc',
    'D:/data1/iod_1deg_1981-2005.nc',
    ]

    
    model_names2=[ 'ERA5', 'COBE_SST',  'CNRM_a','CNRM']
    #model_names2=[ 'COBE_SST', 'ERA5', 'CNRM', 'CNRM']
    
    fig, ax = plt.subplots(2,4) #, figsize=(8,8))
    n=[0,2]
    nn=[1,3]
    for i in [0]: #np.arange(len(model_datasets)):
        print('')
        print('')
        print(model_names2[n[i]])
        
        
        ncin = Dataset(infile[n[i]], 'r')
        sst  = ncin.variables[model_names2[n[i]]][:]
        lat  = ncin.variables['lat'][:]
        lon  = ncin.variables['lon'][:]
        ncin.close()
        nt,nlat,nlon = sst.shape    
        print('nt,nlat,nlon.shape',nt,nlat,nlon)
        
        lat_min = lat.min()
        lat_max = lat.max()
        lon_min = lon.min()
        lon_max = lon.max()
        m = Basemap(ax=ax[1,n[i]], projection ='cyl', 
                llcrnrlat = lat_min+1.5, #+1*0.22, 
                urcrnrlat = lat_max-1.5, #-6*0.22,
                llcrnrlon = lon_min+1, #+4*0.22, 
                urcrnrlon = lon_max-1.5, #-3*0.22, 
                resolution = 'l', fix_aspect=False)
        m.drawcoastlines(linewidth=1)
        m.drawcountries(linewidth=1)
        m.drawstates(linewidth=0.5, color='w')
        
        x = np.empty((nt)) 
        sst_coeffs = np.empty((2, nlat, nlon)) 
        sst_detrend = np.empty((nt, nlat, nlon))

        for ii in range(0, nt): 
            x[ii] = ii

        for ii in range(0,nlat): 
            for j in range(0,nlon): 
                ytemp = np.copy(sst[:,ii,j]) 
                y = sst[:,ii,j] 
                b = ~np.isnan(y) 
                coefs = poly.polyfit(x[b], y[b], 1) 
                sst_coeffs[0,ii,j] = coefs[0] 
                sst_coeffs[1,ii,j] = coefs[1] 
                ffit = poly.polyval(x[b], coefs) 
                sst_detrend[b,ii,j] = y[b] - ffit

        sst_detrend = sst_detrend.reshape((12,int(nt/12), nlat,nlon), order='F').transpose((1,0,2,3))
    
        #Calculate seasonal cycle
        sst_season = np.mean(sst_detrend, axis=0)
        #Remove seasonal cycle
        sst_diff = sst_detrend - sst_season
        sst_diff = sst_diff.transpose((1,0,2,3)).reshape((nt, nlat,nlon), order='F')
        
        #Cosine of latitude weights are applied before 
        #the computation of EOFs
        wgts   = np.cos(np.deg2rad(lat))
        wgts   = wgts.reshape(len(wgts), 1)

        solver = Eof(sst_diff, weights=wgts)
        #Retrieve the leading EOFs
        eof1 = solver.eofs(neofs=n_eofs)
        pc1  = solver.pcs(npcs=n_eofs, pcscaling=0)
        varfrac = solver.varianceFraction()
        lambdas = solver.eigenvalues()
        
        x1,y1 = np.meshgrid(lon, lat)
        #print('x1,y1,eof1.shape', x1.shape,y1.shape,eof1.shape)        
        clevs = np.linspace(np.min(eof1[n_ke,:,:].squeeze()), np.max(eof1[n_ke,:,:].squeeze()), 10)
        p = ax[1,n[i]].contourf(x1, y1, 
                                eof1[n_ke,:,:].squeeze(), 
                                clevs, 
                                cmap=plt.cm.RdBu_r,
                                )
        
        days = np.linspace(1981,2005,nt)
        ax[0,n[i]].plot(days, pc1[:,n_ke], linewidth=2)
        
        tt='  Mode='+str(n_ke+1)+'  vf='+str(np.round(varfrac[n_ke]*100,2))+'%'
        
        ax[0,n[i]].set_title(model_names2[n[i]]+tt, fontsize=9)
        ax[0,0].set_ylabel('Normalized Units')
        ax[0,0].set_yticks([0])
        
        #--------------pr
        print(model_names2[nn[i]])
        
        ncin = Dataset(infile[nn[i]], 'r')
        sst  = ncin.variables[model_names2[nn[i]]][:]
        lat  = ncin.variables['lat'][:]
        lon  = ncin.variables['lon'][:]
        
        ncin.close()
        nt,nlat,nlon = sst.shape
        print('nt,nlat,nlon.shape',nt,nlat,nlon)        
     
        lat_min = lat.min()
        lat_max = lat.max()
        lon_min = lon.min()
        lon_max = lon.max()
        x1,y1 = np.meshgrid(lon, lat)
        
        m = Basemap(ax=ax[1,nn[i]], projection ='cyl', 
                llcrnrlat = lat_min+1, #+1*0.22, 
                urcrnrlat = lat_max-1.5, #-6*0.22,
                llcrnrlon = lon_min+1.5, #+4*0.22, 
                urcrnrlon = lon_max-1.5, #-3*0.22, 
                resolution = 'l', fix_aspect=False)
        m.drawcoastlines(linewidth=1)
        m.drawcountries(linewidth=1)
        m.drawstates(linewidth=0.5, color='w')
        
        x = np.empty((nt)) 
        sst_coeffs = np.empty((2, nlat, nlon)) 
        sst_detrend = np.empty((nt, nlat, nlon))

        for ii in range(0, nt): 
            x[ii] = ii

        for ii in range(0,nlat): 
            for j in range(0,nlon): 
                ytemp = np.copy(sst[:,ii,j]) 
                y = sst[:,ii,j] 
                b = ~np.isnan(y) 
                coefs = poly.polyfit(x[b], y[b], 1) 
                sst_coeffs[0,ii,j] = coefs[0] 
                sst_coeffs[1,ii,j] = coefs[1] 
                ffit = poly.polyval(x[b], coefs) 
                sst_detrend[b,ii,j] = y[b] - ffit

        sst_detrend = sst_detrend.reshape((12,int(nt/12), nlat,nlon), order='F').transpose((1,0,2,3))
        
        #Calculate seasonal cycle
        sst_season = np.mean(sst_detrend, axis=0)
        #Remove seasonal cycle
        sst_diff = sst_detrend - sst_season
        sst_diff = sst_diff.transpose((1,0,2,3)).reshape((nt, nlat,nlon), order='F')
        
        #Cosine of latitude weights are applied before 
        #the computation of EOFs
        wgts   = np.cos(np.deg2rad(lat))
        wgts   = wgts.reshape(len(wgts), 1)

        solver = Eof(sst_diff, weights=wgts)
        #Retrieve the leading EOFs
        eof1 = solver.eofs(neofs=n_eofs)
        pc2  = solver.pcs(npcs=n_eofs, pcscaling=0)
        varfrac = solver.varianceFraction()
        lambdas = solver.eigenvalues()
        
        
        clevs = np.linspace(np.min(eof1[n_ke,:,:].squeeze()), np.max(eof1[n_ke,:,:].squeeze()), 10)
        p2 = m.contourf(x1, y1, 
                        eof1[n_ke,:,:].squeeze(), 
                        clevs, 
                        cmap=plt.cm.RdBu_r,
                        )
        
        days = np.linspace(1981,2005,nt)
        ax[0,nn[i]].plot(days, pc2[:,n_ke], linewidth=2)
        
        tt='  Mode='+str(n_ke+1)+'  vf='+str(np.round(varfrac[n_ke]*100,2))+'%'
        
        ax[0,nn[i]].set_title(model_names2[nn[i]]+tt, fontsize=9)
        
        
        ax[0,n[i]].axhline(0, color='k')
        ax[0,nn[i]].axhline(0, color='k')
        ax[1,nn[i]].set_xlabel('')
        ax[1,nn[i]].set_ylabel('')
        if nn[i]!=0:
            ax[0,nn[i]].set_ylabel('')
        ax[0,nn[i]].set_xlabel('')
        
        
        ax[1,nn[i]].set_title('')
        
        
        ax[0,n[i]].set_yticks([])
        ax[0,nn[i]].set_yticks([])
        
        r, p_value = stats.pearsonr(pc1[:,0], pc2[:,0])
        tt='r='+str(np.round(r,2))+'   pv='+str(np.round(p_value,4))
        
        ax[1,nn[i]].set_title(tt, fontsize=9)
        #ax[0,n[i]].text(1980, lon_min, s=tt, fontsize=9)
        
        ax[1,0].set_xlabel('Nino3.4 region')
        ax[1,1].set_xlabel('SEA region')
        ax[1,2].set_xlabel('Nino3.4 region')
        ax[1,3].set_xlabel('SEA region')
    
    
    
    #plt.subplots_adjust(right=.5)
    plt.subplots_adjust(hspace=.23,wspace=.15)
    '''
    cax = fig.add_axes([0.91, 0.4, 0.015, 0.25])
    cbar = plt.colorbar(p,cax=cax)
    cbar.ax.tick_params(labelsize=8)
    '''
    cax = fig.add_axes([0.91, 0.11, 0.015, 0.34])
    cbar = plt.colorbar(p2,cax=cax)
    cbar.ax.tick_params(labelsize=8)
    
    #cbar1.ax.xaxis.set_ticks_position('top')
    #cbar1.ax.xaxis.set_label_position('top')   
    
    
    #plt.draw()
    #cbar.ax.get_yaxis().set_ticks([])
        
    #ax[0].set_title('EOF'+str(1)+' expressed as correlation_'+fn, pad=20, fontsize=16)
    #plt.savefig(workdir+reg+'_EOF_'+str(n_ke)+'_3tos_new_'+fn+'_'+fnn, dpi=300,bbox_inches='tight')
    plt.show()

def tele_nino_iod_pr(workdir):
    '''
    Correlation of the first leading principal component of 
    Nino3.4 and rainfall time-series.
    PCA SEA vs Nino34 for 1981-2005 data
    
    sebelum eof untuk melihar inter-annual maka remove seasonal, sudah??  
    '''
    import numpy as np
    from scipy import signal
    import numpy.polynomial.polynomial as poly
    from netCDF4 import Dataset

    import matplotlib.pyplot as plt
    from mpl_toolkits.basemap import Basemap
    from eofs.standard import Eof
    from scipy import stats
    
    n_eofs=1 #1,2,3..
    n_ke=0 #0,1,2..
    
    #ref
    infile = [
    'D:/Cordex/RCMES/pr_eva_ds/pr_SEA_LO_2obs_moe_9model_mme_1981_2005.nc',
    'D:/data1/iod_1deg_1981-2005.nc',
    'D:/data1/nino_1deg_1981-2005.nc',
    ]

    model_names2=[ 'ERA5', 'COBE_SST', 'COBE_SST']
    model_names=[ 'ERA5_rainfall', 'COBE_sst', 'COBE_sst']
    
    model_names2=[ 'CNRM_a', 'CNRM', 'CNRM']
    model_names=[ 'CNRM_a_rainfall', 'CNRM_sst', 'CNRM_sst']
    
    model_names2=[ 'IPSL_b', 'IPSL', 'IPSL']
    model_names=[ 'IPSL_b_rainfall', 'IPSL_sst', 'IPSL_sst']
    
    model_names2=[ 'NorESM1_d', 'NorESM1', 'NorESM1']
    model_names=[ 'NorESM1_d_rainfall', 'NorESM1_sst', 'NorESM1_sst']
    
    model_names2=[ 'GFDL_b', 'GFDL', 'GFDL']
    model_names=[ 'GFDL_b_rainfall', 'GFDL_sst', 'GFDL_sst']
    
    model_names2 = ['MME','MME','MME']
    model_names = ['MME_rainfall', 'MME_sst', 'MME_sst']

    model_names2=[ 'GPCP', 'COBE_SST', 'COBE_SST']
    model_names=[ 'GPCP_rainfall', 'COBE_sst', 'COBE_sst']
    
    
    fig, ax = plt.subplots(2,3) #, figsize=(8,8))
    n=[0,1,2]
    for i in n: #np.arange(len(model_datasets)):
        print('')
        print('')
        print(model_names2[n[i]])
        
        
        ncin = Dataset(infile[n[i]], 'r')
        sst  = ncin.variables[model_names2[n[i]]][:]
        lat  = ncin.variables['lat'][:]
        lon  = ncin.variables['lon'][:]
        ncin.close()
        nt,nlat,nlon = sst.shape    
        print('nt,nlat,nlon.shape',nt,nlat,nlon)
        
        lat_min = lat.min()
        lat_max = lat.max()
        lon_min = lon.min()
        lon_max = lon.max()
        m = Basemap(ax=ax[1,n[i]], projection ='cyl', 
                llcrnrlat = lat_min+1.5, #+1*0.22, 
                urcrnrlat = lat_max-1.5, #-6*0.22,
                llcrnrlon = lon_min+1, #+4*0.22, 
                urcrnrlon = lon_max-1.5, #-3*0.22, 
                resolution = 'l', fix_aspect=False)
        m.drawcoastlines(linewidth=1)
        m.drawcountries(linewidth=1)
        m.drawstates(linewidth=0.5, color='w')
        
        x = np.empty((nt)) 
        sst_coeffs = np.empty((2, nlat, nlon)) 
        sst_detrend = np.empty((nt, nlat, nlon))

        for ii in range(0, nt): 
            x[ii] = ii

        for ii in range(0,nlat): 
            for j in range(0,nlon): 
                ytemp = np.copy(sst[:,ii,j]) 
                y = sst[:,ii,j] 
                b = ~np.isnan(y) 
                coefs = poly.polyfit(x[b], y[b], 1) 
                sst_coeffs[0,ii,j] = coefs[0] 
                sst_coeffs[1,ii,j] = coefs[1] 
                ffit = poly.polyval(x[b], coefs) 
                sst_detrend[b,ii,j] = y[b] - ffit

        sst_detrend = sst_detrend.reshape((12,int(nt/12), nlat,nlon), order='F').transpose((1,0,2,3))
    
        #Calculate seasonal cycle
        sst_season = np.mean(sst_detrend, axis=0)
        #Remove seasonal cycle
        sst_diff = sst_detrend - sst_season
        sst_diff = sst_diff.transpose((1,0,2,3)).reshape((nt, nlat,nlon), order='F')
        
        #Cosine of latitude weights are applied before 
        #the computation of EOFs
        wgts   = np.cos(np.deg2rad(lat))
        wgts   = wgts.reshape(len(wgts), 1)

        solver = Eof(sst_diff, weights=wgts)
        #Retrieve the leading EOFs
        eof1 = solver.eofs(neofs=n_eofs)
        pc1  = solver.pcs(npcs=n_eofs, pcscaling=0)
        if n[i]==0: pc=pc1
        varfrac = solver.varianceFraction()
        lambdas = solver.eigenvalues()
        
        x1,y1 = np.meshgrid(lon, lat)
        #print('x1,y1,eof1.shape', x1.shape,y1.shape,eof1.shape)        
        clevs = np.linspace(np.min(eof1[n_ke,:,:].squeeze()), np.max(eof1[n_ke,:,:].squeeze()), 10)
        p = ax[1,n[i]].contourf(x1, y1, 
                                eof1[n_ke,:,:].squeeze(), 
                                clevs, 
                                cmap=plt.cm.RdBu_r,
                                )
        
        days = np.linspace(1981,2005,nt)
        ax[0,n[i]].plot(days, pc1[:,n_ke], linewidth=2)
        
        
        ax[0,0].set_ylabel('Normalized Units')
        ax[0,0].set_yticks([0])
        
        ax[0,n[i]].axhline(0, color='k')
        ax[0,n[i]].axhline(0, color='k')
        ax[1,n[i]].set_xlabel('')
        ax[1,n[i]].set_ylabel('')
        if n[i]!=0:
            ax[0,n[i]].set_ylabel('')
        ax[0,n[i]].set_xlabel('')
        
        
        ax[1,n[i]].set_title('')
        
        
        ax[0,n[i]].set_yticks([])
        ax[0,n[i]].set_yticks([])
        
        if n[i]==0:
            tt=' [vf='+str(np.round(varfrac[n_ke]*100,2))+'%]'
        else:
            r, p_value = stats.pearsonr(pc[:,0], pc1[:,0])
            tt2=' r='+str(np.round(r,2))+' pv='+str(np.round(p_value,4))+']'
        
            tt=' [vf='+str(np.round(varfrac[n_ke]*100,2))+'%'+tt2
        
        ax[0,n[i]].set_title(model_names[n[i]]+tt, fontsize=9)
      
    ax[1,0].set_xticks([100,120,140])
    ax[1,0].set_xticklabels(['100E','120E','140E'], fontsize=7)
    #ax[1,0].xaxis.set_tick_params(labelsize=6)
    ax[1,0].set_yticks([-10,0,10,20])
    ax[1,0].set_yticklabels(['10S','0','10N','20N'], fontsize=7)
    #ax[1,0].yaxis.set_tick_params(labelsize=6)
    
    #IOD -15 15, 50-110
    ax[1,1].set_xticks([50,60,70])
    ax[1,1].set_xticklabels(['50E','60E','70E'], fontsize=7)
    #ax[1,1].xaxis.set_tick_params(labelsize=6)
    ax[1,1].set_yticks([-10,0,10])
    ax[1,1].set_yticklabels(['10S','0','10N'], fontsize=7)
    #ax[1,1].yaxis.set_tick_params(labelsize=6)
    
    #Nino -5 5, 190-240
    ax[1,2].set_xticks([-160,-140])
    ax[1,2].set_xticklabels(['200E','220E'], fontsize=7)
    #ax[1,1].xaxis.set_tick_params(labelsize=6)
    ax[1,2].set_yticks([-3,0,3])
    ax[1,2].set_yticklabels(['3S','0','3N'], fontsize=7)
    #ax[1,1].yaxis.set_tick_params(labelsize=6)
    
    
    ax[1,0].set_xlabel('EOF-1 Rainfall in SEA region')
    ax[1,1].set_xlabel('EOF-1 SST in IOD region')
    ax[1,2].set_xlabel('EOF-1 SST Nino3.4 region')
       
    
    
    
    #plt.subplots_adjust(right=.5)
    plt.subplots_adjust(hspace=.23,wspace=.13)
    '''
    cax = fig.add_axes([0.91, 0.4, 0.015, 0.25])
    cbar = plt.colorbar(p,cax=cax)
    cbar.ax.tick_params(labelsize=8)
    '''
    cax = fig.add_axes([0.91, 0.11, 0.015, 0.34])
    cbar = plt.colorbar(p,cax=cax)
    cbar.ax.tick_params(labelsize=8)
    
    #cbar1.ax.xaxis.set_ticks_position('top')
    #cbar1.ax.xaxis.set_label_position('top')   
    
    
    #plt.draw()
    #cbar.ax.get_yaxis().set_ticks([])
        
    #ax[0].set_title('EOF'+str(1)+' expressed as correlation_'+fn, pad=20, fontsize=16)
    #plt.savefig(workdir+reg+'_EOF_'+str(n_ke)+'_3tos_new_'+fn+'_'+fnn, dpi=300,bbox_inches='tight')
    plt.show()


def corr_pc1_xr(workdir):
    '''
    Correlation of the first leading principal component of 
    Nino3.4 and rainfall time-series.
    PCA SEA vs Nino34 for 1981-2005 data
    
    sebelum eof untuk melihaT inter-annual maka remove seasonal, sudah??  
    '''
    import numpy as np
    from scipy import signal
    import numpy.polynomial.polynomial as poly
    from netCDF4 import Dataset

    import matplotlib.pyplot as plt
    from mpl_toolkits.basemap import Basemap
    #from eofs.standard import Eof
    from scipy import stats
    
    from eofs2.xarray import Eof
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    
    n_eofs=1 #1,2,3..
    n_ke=0 #0,1,2..
    anom=1
    
    infile = [
    'D:/data1/nino_1deg_1981-2005.nc',
    'D:/Cordex/RCMES/pr_eva_ds/pr_SEA_LO_2obs_moe_9model_mme_1981_2005.nc',
    'D:/data1/nino_1deg_1981-2005.nc',
    'D:/Cordex/RCMES/pr_eva_ds/pr_SEA_LO_2obs_moe_9model_mme_1981_2005.nc',
    ]

    
   
    #model_names2=[ 'COBE_SST', 'ERA5', 'CNRM', 'CNRM_a']
    model_names2=[ 'COBE_SST', 'GPCP', 'CNRM', 'CNRM_a']
    
    fig, ax = plt.subplots(2,4) #, figsize=(8,8))
    n=[0,2]
    nn=[1,3]
    for i in [0]: #np.arange(len(model_datasets)):
        print('')
        print('')
        print(model_names2[n[i]])
        
        
        ncin = Dataset(infile[n[i]], 'r')
        sst  = ncin.variables[model_names2[n[i]]][:]
        lat  = ncin.variables['lat'][:]
        lon  = ncin.variables['lon'][:]
        time = ncin.variables['time'][:]
        ncin.close()
        nt,nlat,nlon = sst.shape    
        print('nt,nlat,nlon.shape',nt,nlat,nlon)
        
        lat_min = lat.min()
        lat_max = lat.max()
        lon_min = lon.min()
        lon_max = lon.max()
        x1,y1 = np.meshgrid(lon, lat)
        
        m = Basemap(ax=ax[1,n[i]], projection ='cyl', 
                llcrnrlat = lat_min, #+1*0.22, 
                urcrnrlat = lat_max, #-6*0.22,
                llcrnrlon = lon_min, #+4*0.22, 
                urcrnrlon = lon_max, #-3*0.22, 
                resolution = 'l', fix_aspect=False)
        m.drawcoastlines(linewidth=1)
        m.drawcountries(linewidth=1)
        m.drawstates(linewidth=0.5, color='w')
        
        x = np.empty((nt)) 
        sst_coeffs = np.empty((2, nlat, nlon)) 
        sst_detrend = np.empty((nt, nlat, nlon))

        for ii in range(0, nt): 
            x[ii] = ii

        for ii in range(0,nlat): 
            for j in range(0,nlon): 
                ytemp = np.copy(sst[:,ii,j]) 
                y = sst[:,ii,j] 
                b = ~np.isnan(y) 
                coefs = poly.polyfit(x[b], y[b], 1) 
                sst_coeffs[0,ii,j] = coefs[0] 
                sst_coeffs[1,ii,j] = coefs[1] 
                ffit = poly.polyval(x[b], coefs) 
                sst_detrend[b,ii,j] = y[b] - ffit

        sst_detrend = sst_detrend.reshape((12,int(nt/12), nlat,nlon), order='F').transpose((1,0,2,3))
    
        #Calculate seasonal cycle
        sst_season = np.mean(sst_detrend, axis=0)
        #Remove seasonal cycle
        sst_diff = sst_detrend - sst_season
        sst_diff = sst_diff.transpose((1,0,2,3)).reshape((nt, nlat,nlon), order='F')
        
        ds = xr.DataArray(sst_diff,
        coords={'time': time,
                'lat': lat, 
                'lon': lon},
        dims=["time", "lat", "lon"])
        
        #Cosine of latitude weights are applied before 
        #the computation of EOFs
        wgts   = np.cos(np.deg2rad(lat))
        wgts   = wgts.reshape(len(wgts), 1)

        #solver = Eof(ds, weights=wgts)
        solver = Eof(ds)
        
        #Retrieve the leading EOFs
        eof = solver.eofsAsCorrelation(neofs=n_eofs)
        pc = solver.pcs(npcs=n_eofs, pcscaling=0) #1 with scaling
        
        #print('eof=',eof)
        print('nino_varianceFraction=', solver.varianceFraction(3).data)
        
        #print('pc=',pc)
        #clevs = np.linspace(np.min(eof1[n_ke,:,:].squeeze()), np.max(eof1[n_ke,:,:].squeeze()), 21)
        #eof1.min(), eof1.max()
        #clevs = np.linspace(eof.min(), eof.max(),10)
        #levels =np.arange(0.8,1,.02)
        #cmap='rainbow'
        cmap='viridis'
        p = eof[n_ke].plot.contourf(ax=ax[1,n[i]], 
                            #levels=clevs,
                            add_colorbar=False)
    
        pc[:, n_ke].plot(ax=ax[0,n[i]], color='b', linewidth=2)
        print(pc.shape)
        
        #ax[0,i] = plt.gca()
        ax[0,n[i]].axhline(0, color='k')
        #ax.set_ylim(-3, 3)
        #ax.set_xlabel('month')
        if anom==1: 
            ax[0,0].set_ylabel('Normalized Units')
        else:
            ax[0,0].set_ylabel('Un-normalized Units')
        #ax.set_title('PC'+str(i)+' Time Series_'+fn, pad=20, fontsize=16)
        
        ax[0,n[i]].tick_params(axis='x', pad=1,labelsize=8)
        ax[0,n[i]].tick_params(axis='y', pad=1,labelsize=8)
        ax[0,nn[i]].tick_params(axis='x', pad=1,labelsize=8)
        ax[0,nn[i]].tick_params(axis='y', pad=1,labelsize=8)
        
        ax[1,n[i]].tick_params(axis='x', pad=1,labelsize=8)
        ax[1,n[i]].tick_params(axis='y', pad=1,labelsize=8)
        ax[1,nn[i]].tick_params(axis='x', pad=1,labelsize=8)
        ax[1,nn[i]].tick_params(axis='y', pad=1,labelsize=8)
        
        #tt=' ['+str(n_ke+1)+']'+' ['+str(np.round(solver.varianceFraction()[n_ke].data,2)*100)+'%]'   
        
        tt='  Mode='+str(n_ke+1)+'  EV='+str(np.round(solver.varianceFraction()[n_ke].data*100,2))+'%'
        
        ax[0,n[i]].set_title(model_names2[n[i]]+tt, fontsize=9)
        
        
        ax[0,n[i]].set_ylabel('')
        ax[0,n[i]].set_xlabel('')
        ax[1,n[i]].set_xlabel('')
        ax[1,n[i]].set_ylabel('')
        
        ax[1,n[i]].set_title('')
        ax[1,n[i]].set_title('')        
        
        #--------------pr
        print(model_names2[nn[i]])
        
        ncin = Dataset(infile[nn[i]], 'r')
        sst  = ncin.variables[model_names2[nn[i]]][:]
        lat  = ncin.variables['lat'][:]
        lon  = ncin.variables['lon'][:]
        time = ncin.variables['time'][:]
      
        ncin.close()
        nt,nlat,nlon = sst.shape
        print('nt,nlat,nlon.shape',nt,nlat,nlon)        
     
        lat_min = lat.min()
        lat_max = lat.max()
        lon_min = lon.min()
        lon_max = lon.max()
        x1,y1 = np.meshgrid(lon, lat)
        
        m = Basemap(ax=ax[1,nn[i]], projection ='cyl', 
                llcrnrlat = lat_min, #+1*0.22, 
                urcrnrlat = lat_max, #-6*0.22,
                llcrnrlon = lon_min, #+4*0.22, 
                urcrnrlon = lon_max, #-3*0.22, 
                resolution = 'l', fix_aspect=False)
        m.drawcoastlines(linewidth=1)
        m.drawcountries(linewidth=1)
        m.drawstates(linewidth=0.5, color='w')
        
        x = np.empty((nt)) 
        sst_coeffs = np.empty((2, nlat, nlon)) 
        sst_detrend = np.empty((nt, nlat, nlon))

        for ii in range(0, nt): 
            x[ii] = ii

        for ii in range(0,nlat): 
            for j in range(0,nlon): 
                ytemp = np.copy(sst[:,ii,j]) 
                y = sst[:,ii,j] 
                b = ~np.isnan(y) 
                coefs = poly.polyfit(x[b], y[b], 1) 
                sst_coeffs[0,ii,j] = coefs[0] 
                sst_coeffs[1,ii,j] = coefs[1] 
                ffit = poly.polyval(x[b], coefs) 
                sst_detrend[b,ii,j] = y[b] - ffit

        sst_detrend = sst_detrend.reshape((12,int(nt/12), nlat,nlon), order='F').transpose((1,0,2,3))
        
        #Calculate seasonal cycle
        sst_season = np.mean(sst_detrend, axis=0)
        #Remove seasonal cycle
        sst_diff = sst_detrend - sst_season
        sst_diff = sst_diff.transpose((1,0,2,3)).reshape((nt, nlat,nlon), order='F')
        
        ds = xr.DataArray(sst_diff,
        coords={'time': time,
                'lat': lat, 
                'lon': lon},
        dims=["time", "lat", "lon"])
        
        #Cosine of latitude weights are applied before 
        #the computation of EOFs
        wgts   = np.cos(np.deg2rad(lat))
        wgts   = wgts.reshape(len(wgts), 1)

        #solver = Eof(ds, weights=wgts)
        solver = Eof(ds)
        
        #Retrieve the leading EOFs
        eof2 = solver.eofsAsCorrelation(neofs=n_eofs)
        pc2 = solver.pcs(npcs=n_eofs, pcscaling=0) #1 with scaling
        
        #print('eof=',eof)
        print('pr_varianceFraction=', solver.varianceFraction(3).data)
        
        p2 = eof2[n_ke].plot.contourf(ax=ax[1,nn[i]], 
                                        #levels=clevs,
                                        add_colorbar=False)
                                        
        pc2[:, n_ke].plot(ax=ax[0,nn[i]], color='b', linewidth=2)
        print(pc2.shape)
        
        tt='  Mode='+str(n_ke+1)+'  EV='+str(np.round(solver.varianceFraction()[n_ke].data*100,2))+'%'
        
        ax[0,nn[i]].set_title(model_names2[nn[i]]+tt, fontsize=9)
        
        r, p_value = stats.pearsonr(pc[:,n_ke], pc2[:,n_ke])
        
        print("Correlation:", r)
        print("P-value:", p_value)
        
        ax[0,nn[i]].axhline(0, color='k')
        ax[1,nn[i]].set_xlabel('')
        ax[1,nn[i]].set_ylabel('')
        if nn[i]!=0:
            ax[0,nn[i]].set_ylabel('')
        ax[0,nn[i]].set_xlabel('')
        
        
        ax[1,nn[i]].set_title('')
        
        
        ax[0,n[i]].set_yticks([])
        ax[0,nn[i]].set_yticks([])
        
        tt='r='+str(np.round(r,2))+'   pv='+str(np.round(p_value,4))
        
        ax[1,nn[i]].set_title(tt, fontsize=9)
        #ax[0,n[i]].text(1980, lon_min, s=tt, fontsize=9)
        
        ax[1,0].set_xlabel('Nino3.4 region')
        ax[1,1].set_xlabel('SEA region')
        ax[1,2].set_xlabel('Nino3.4 region')
        ax[1,3].set_xlabel('SEA region')
    
    
    
    #plt.subplots_adjust(right=.5)
    plt.subplots_adjust(hspace=.2,wspace=.15)
    
    cax = fig.add_axes([0.91, 0.4, 0.015, 0.25])
    cbar = plt.colorbar(p,cax=cax)
    cbar.ax.tick_params(labelsize=8)
    
    cax = fig.add_axes([0.91, 0.11, 0.015, 0.25])
    cbar = plt.colorbar(p2,cax=cax)
    cbar.ax.tick_params(labelsize=8)
    
    #cbar1.ax.xaxis.set_ticks_position('top')
    #cbar1.ax.xaxis.set_label_position('top')   
    
    
    plt.draw()
    #cbar.ax.get_yaxis().set_ticks([])
        
    #ax[0].set_title('EOF'+str(1)+' expressed as correlation_'+fn, pad=20, fontsize=16)
    plt.savefig(workdir+reg+'_EOF_'+str(n_ke), dpi=300,bbox_inches='tight')
    plt.show()

def eofs_multi_tos_iod(obs_dataset, obs_name, model_datasets, model_names, workdir):
    from eofs2.xarray import Eof
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import matplotlib.colors as colors
    
    #choose
    annual_cycle=False
    season=False
    musim='JJA'
    monthly=False
    yearly=True
    
    anom=1
    
    n_eofs=3 #1,2,3..
    n_ke=0 #0,1,2..
    
    fig, ax = plt.subplots(2,3) #, figsize=(8,8))
    
    for i in np.arange(len(model_datasets)):
        print(model_names[i])
     
        ds = xr.DataArray(model_datasets[i].values,
        coords={'time': model_datasets[i].times,
                'lat': model_datasets[i].lats, 
                'lon': model_datasets[i].lons},
        dims=["time", "lat", "lon"])
        
        lat_min = ds.lat.min()
        lat_max = ds.lat.max()
        lon_min = ds.lon.min()
        lon_max = ds.lon.max()
        x,y = np.meshgrid(ds.lon, ds.lat)
        
        m = Basemap(ax=ax[1,i], projection ='cyl', 
                llcrnrlat = lat_min, #+1*0.22, 
                urcrnrlat = lat_max, #-6*0.22,
                llcrnrlon = lon_min, #+4*0.22, 
                urcrnrlon = lon_max, #-3*0.22, 
                resolution = 'l', fix_aspect=False)
        m.drawcoastlines(linewidth=1)
        m.drawcountries(linewidth=1)
        m.drawstates(linewidth=0.5, color='w')
    
        #print('1 ds.min_max=',ds.min().data,ds.max().data)
        
        #Calculate monthly anomalies: 
        #https://docs.xarray.dev/en/stable/examples/weather-data.html
        #In climatology, “anomalies” refer to the difference between 
        #observations and typical weather for a particular season. 
        #Unlike observations, anomalies should not show any seasonal cycle.
        
        #climatology = ds.groupby("time.month").mean("time")
        #ds = ds.groupby("time.month") - climatology
        fnn=''
        if anom==1:
            climatology_mean = ds.groupby("time.month").mean("time")
            climatology_std = ds.groupby("time.month").std("time")
            ds = xr.apply_ufunc(
                lambda x, m, s: 
                (x - m) / s,
                ds.groupby("time.month"),
                climatology_mean,
                climatology_std,
            )
            fnn='anom'
        
        #print('ds=',ds)
        #print('2 ds.min_max=',ds.min().data,ds.max().data)
        
        #print('obs_name')
        #ds= ds.stack(z=("lat", "lon"))
        
        if annual_cycle:
            dm = ds.groupby('time.month').mean()
            dm=dm.rename({'month':'time'})
            dm=dm.rename('SEAR')
            solver = Eof(dm)
            fn='annual_cycle'
        if monthly:
            
            #da.assign_coords(lon=(((da.lon + 180) % 360) - 180))
            #ds=ds.assign_coords(time=np.arange(300))
            ds=ds.drop_vars('time')
            #print('3 ds.min_max=',ds.min().data,ds.max().data)
            if anom==1: ds=ds.rename({'month':'time'}) #if climatology anomalies used
            ds=ds.assign_coords(time=np.arange(300)) #1981,2006))
            ds=ds.rename('SEAR')
            solver = Eof(ds)
            fn='monthly'
            #print('ds=',ds)
            #dm=ds
        
        if season:
            ds = ds.groupby('time.season')#.mean() 
            ds=ds[musim]
            ds=ds.drop_vars('time')
            print('3 ds.min_max=',ds.min().data,ds.max().data)
            ds=ds.rename({'month':'time'})
            ds=ds.assign_coords(time=np.arange(75))
            
            fn='season'
            
            #ds=ds[2]
            print('ds=',ds)
            ds=ds.rename('SEA-RA')
            solver = Eof(ds)
        
        if yearly:
            ds = ds.groupby('time.year').sum()
            ds=ds.rename({'year':'time'})
            #print('ds=',ds)
            ds=ds.rename('SEA-RA')
            solver = Eof(ds)
            fn='yearly'
        print('ds.lon.min(),ds.lon.max()=', ds.lon.min(),ds.lon.max())
        #print('dm.shape', dm)
        #print('dm.values.shape', dm.values)
        
        eof = solver.eofsAsCorrelation(neofs=n_eofs)
        pc = solver.pcs(npcs=n_eofs, pcscaling=0) #1 with scaling
        
        #print('eof=',eof)
        print('solver.varianceFraction=', solver.varianceFraction(3))
        
        #print('pc=',pc)
        
        #norm = plt.Normalize(vmin=-1, vmax=1)
        #norm = colors.Normalize(vmin=-1, vmax=1)
        levels =np.arange(0,1.2,.2)
        print(levels)
        
        #p = eof[n_ke].plot.contourf(ax=ax[1,i], vmin=-1, vmax=1, add_colorbar=False)
        p = eof[n_ke].plot.contourf(ax=ax[1,i], levels =levels , vmin=-1, vmax=1,cmap='rainbow', add_colorbar=False)
       
        #ax.add_feature(cfeature.LAND, facecolor='w', edgecolor='k')
        #cb = plt.colorbar(fill, orientation='horizontal')
        #cb.set_label('correlation coefficient', fontsize=12)
        
        #ax.set_xlabel(fn)
        
    
       
        pc[:, n_ke].plot(ax=ax[0,i], color='b', linewidth=2)
        #ax[0,i] = plt.gca()
        ax[0,i].axhline(0, color='k')
        #ax.set_ylim(-3, 3)
        #ax.set_xlabel('month')
        if anom==1: 
            ax[0,0].set_ylabel('Normalized Units')
        else:
            ax[0,0].set_ylabel('Un-normalized Units')
        #ax.set_title('PC'+str(i)+' Time Series_'+fn, pad=20, fontsize=16)
        
        ax[0,i].tick_params(axis='x', pad=1,labelsize=8)
        ax[0,i].tick_params(axis='y', pad=1,labelsize=8)
        
        #tt=' ['+str(n_ke+1)+']'+' ['+str(np.round(solver.varianceFraction()[n_ke].data,2)*100)+'%]'   
        
        tt='  Mode='+str(n_ke+1)+'  EV='+str(np.round(solver.varianceFraction()[n_ke].data,2)*100)+'%'
        #if i==0: tt='  Mode='+str(n_ke+1)+'  EV=56.0%'
        ax[0,i].set_title(model_names[i]+tt, fontsize=9)
        
        ax[1,i].set_xlabel('')
        ax[1,i].set_ylabel('')
        ax[0,i].set_ylabel('')
        ax[0,i].set_xlabel('')
        
        #45-115
        ax[1,i].set_xticks([50,80,110])
        ax[1,i].set_xticklabels(['50E','80E','110E'])
        ax[1,i].tick_params(axis='x', pad=1,labelsize=8)
        ax[1,i].set_title('')
        if i>0: ax[0,i].set_yticks([])
    #-15 15    
    ax[1,0].set_yticks([-10,0,10])
    ax[1,0].set_yticklabels(['10S','0','10N'])
    ax[1,0].tick_params(axis='y', pad=1,labelsize=8)
    
    
    
    #plt.subplots_adjust(right=.5)
    plt.subplots_adjust(hspace=.2,wspace=.1)
    
    cax = fig.add_axes([0.91, 0.11, 0.015, 0.345])
   
    cbar = plt.colorbar(p,cax=cax)
    
    
    plt.draw()
    #cbar.ax.get_yaxis().set_ticks([])
        
    #ax[0].set_title('EOF'+str(1)+' expressed as correlation_'+fn, pad=20, fontsize=16)
    plt.savefig(workdir+reg+'_EOF_'+str(n_ke)+'_3-2tos-iod-2_'+fn+'_'+fnn, dpi=300,bbox_inches='tight') 
    plt.show()

    
def eofs_multi_tos2(obs_dataset, obs_name, model_datasets, model_names, workdir):
    # ini aneh pc error di fig [1,0] tidak keluar 
    #namun di 0 bisa keluar
    
    from eofs2.xarray import Eof
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    
    #choose
    annual_cycle=False
    season=False
    musim='JJA'
    monthly=False
    yearly=True
    
    anom=1
    
    n_eofs=3 #1,2,3..
    n_ke=0 #0,1,2..
    
    fig, ax = plt.subplots(2,3) #, figsize=(8,8))
    
    for i in np.arange(len(model_datasets)):
        print(model_names[i])
     
        ds = xr.DataArray(model_datasets[i].values,
        coords={'time': model_datasets[i].times,
                'lat': model_datasets[i].lats, 
                'lon': model_datasets[i].lons},
        dims=["time", "lat", "lon"])
        
        lat_min = ds.lat.min()
        lat_max = ds.lat.max()
        lon_min = ds.lon.min()
        lon_max = ds.lon.max()
        x,y = np.meshgrid(ds.lon, ds.lat)
        
        m = Basemap(ax=ax[1,i], projection ='cyl', 
                llcrnrlat = lat_min, #+1*0.22, 
                urcrnrlat = lat_max, #-6*0.22,
                llcrnrlon = lon_min, #+4*0.22, 
                urcrnrlon = lon_max, #-3*0.22, 
                resolution = 'l', fix_aspect=False)
        m.drawcoastlines(linewidth=1)
        m.drawcountries(linewidth=1)
        m.drawstates(linewidth=0.5, color='w')
    
        #print('1 ds.min_max=',ds.min().data,ds.max().data)
        
        #Calculate monthly anomalies: 
        #https://docs.xarray.dev/en/stable/examples/weather-data.html
        #In climatology, “anomalies” refer to the difference between 
        #observations and typical weather for a particular season. 
        #Unlike observations, anomalies should not show any seasonal cycle.
        
        #climatology = ds.groupby("time.month").mean("time")
        #ds = ds.groupby("time.month") - climatology
        fnn=''
        if anom==1:
            climatology_mean = ds.groupby("time.month").mean("time")
            climatology_std = ds.groupby("time.month").std("time")
            ds = xr.apply_ufunc(
                lambda x, m, s: 
                (x - m) / s,
                ds.groupby("time.month"),
                climatology_mean,
                climatology_std,
            )
            fnn='anom'
        
        #print('ds=',ds)
        #print('2 ds.min_max=',ds.min().data,ds.max().data)
        
        #print('obs_name')
        #ds= ds.stack(z=("lat", "lon"))
        
        if annual_cycle:
            dm = ds.groupby('time.month').mean()
            dm=dm.rename({'month':'time'})
            dm=dm.rename('SEAR')
            solver = Eof(dm)
            fn='annual_cycle'
        if monthly:
            
            #da.assign_coords(lon=(((da.lon + 180) % 360) - 180))
            #ds=ds.assign_coords(time=np.arange(300))
            ds=ds.drop_vars('time')
            #print('3 ds.min_max=',ds.min().data,ds.max().data)
            if anom==1: ds=ds.rename({'month':'time'}) #if climatology anomalies used
            ds=ds.assign_coords(time=np.arange(300)) #1981,2006))
            ds=ds.rename('SEAR')
            solver = Eof(ds)
            fn='monthly'
            #print('ds=',ds)
            #dm=ds
        
        if season:
            ds = ds.groupby('time.season')#.mean() 
            ds=ds[musim]
            ds=ds.drop_vars('time')
            print('3 ds.min_max=',ds.min().data,ds.max().data)
            ds=ds.rename({'month':'time'})
            ds=ds.assign_coords(time=np.arange(75))
            
            fn='season'
            
            #ds=ds[2]
            print('ds=',ds)
            ds=ds.rename('SEA-RA')
            solver = Eof(ds)
        
        if yearly:
            ds = ds.groupby('time.year').sum()
            ds=ds.rename({'year':'time'})
            #print('ds=',ds)
            ds=ds.rename('SEA-RA')
            solver = Eof(ds)
            fn='yearly'
        print('ds.lon.min(),ds.lon.max()=', ds.lon.min(),ds.lon.max())
        #print('dm.shape', dm)
        #print('dm.values.shape', dm.values)
        
        eof = solver.eofsAsCorrelation(neofs=n_eofs)
        pc = solver.pcs(npcs=n_eofs, pcscaling=0) #1 with scaling
        
        #print('eof=',eof)
        print('solver.varianceFraction=', solver.varianceFraction(3))
        
        print('pc=',pc)
        
        p = eof[n_ke].plot.contourf(ax=ax[0,i], add_colorbar=False)
           
        pc[:, n_ke].plot(ax=ax[1,i], color='b', linewidth=2)
        
        '''
        #ax[1,i].axhline(0, color='k')
        #ax.set_ylim(-3, 3)
        #ax.set_xlabel('month')
        if anom==1: 
            ax[1,0].set_ylabel('Normalized Units')
        else:
            ax[1,0].set_ylabel('Un-normalized Units')
        
        
        #ax[1,i].tick_params(axis='x', pad=1,labelsize=8)
        #ax[1,i].tick_params(axis='y', pad=1,labelsize=8)
        
        #tt=' ['+str(n_ke+1)+']'+' ['+str(np.round(solver.varianceFraction()[n_ke].data,2)*100)+'%]'   
        
        tt='  Mode='+str(n_ke+1)+'  EV='+str(np.round(solver.varianceFraction()[n_ke].data,2)*100)+'%'
        
        ax[0,i].set_title(model_names[i]+tt, fontsize=9)
        
        
        ax[0,i].set_ylabel('')
        ax[0,i].set_xlabel('')
        
        #190-240 => -170 to -120
        ax[0,i].set_xticks([-160,-140,-120])
        ax[0,i].set_xticklabels(['160W','140W','120W'])
        ax[0,i].tick_params(axis='x', pad=1,labelsize=8)
        
        if i>0: 
            ax[0,i].set_yticks([])
            #ax[1,i].set_yticks([])
        ax[1,i].set_title('')
        ax[1,i].set_xlabel('')
        ax[1,i].set_ylabel('')
        '''
    #-5 5    
    ax[0,0].set_yticks([-5,0,5])
    ax[0,0].set_yticklabels(['5S','0','5N'])
    ax[0,0].tick_params(axis='y', pad=1,labelsize=8)
    
    
    
    #plt.subplots_adjust(right=.5)
    plt.subplots_adjust(hspace=.2,wspace=.1)
    
    cax = fig.add_axes([0.91, 0.54, 0.015, 0.345])
   
    cbar = plt.colorbar(p,cax=cax)

    #cbar.ax.get_yaxis().set_ticks([])
        
    #ax[0].set_title('EOF'+str(1)+' expressed as correlation_'+fn, pad=20, fontsize=16)
    plt.savefig(workdir+reg+'_EOF_'+str(n_ke)+'_3tos_'+fn+'_'+fnn, dpi=300,bbox_inches='tight')
        
    plt.show()

def eofs_pca2(obs_dataset, obs_name, model_datasets, model_names, workdir):
    from eofs2.xarray import Eof
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    
    #choose
    annual_cycle=True
    
    ds = xr.DataArray(obs_dataset.values,
    coords={'time': obs_dataset.times,
            'lat': obs_dataset.lats, 
            'lon': obs_dataset.lons},
    dims=["time", "lat", "lon"])
    
    print(obs_name)
    
    #ds= ds.stack(z=("lat", "lon"))
    
    if annual_cycle:
        dm = ds.groupby('time.month').mean()
        dm=dm.rename({'month':'time'})
        #dm=dm.T # error: time must be the first dimension
        solver = Eof(dm)
        fn='annual_cycle'
    else:
        solver = Eof(ds)
        fn='raw_monthly'
    
    eof = solver.eofsAsCorrelation(neofs=1)
    pc = solver.pcs(npcs=1, pcscaling=0) #1 with scaling
    print(eof)
    print(pc)
    
  
    eof.plot()
    plt.show()
    pc.plot()
    plt.show()
    
    #exit()
    
    clevs = np.linspace(-1, 1, 11)
    '''
    for i in [0,1,2]: #len(eof):
        plt.figure()
        ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=100))
        fill = eof[i].plot.contourf(ax=ax, levels=clevs, cmap=plt.cm.RdBu_r,
                                     add_colorbar=False, transform=ccrs.PlateCarree())
        
        ax.add_feature(cfeature.LAND, facecolor='w', edgecolor='k')
        cb = plt.colorbar(fill, orientation='horizontal')
        cb.set_label('correlation coefficient', fontsize=12)
        ax.set_title('EOF'+str(i)+' expressed as correlation', fontsize=16)
        
        plt.savefig(workdir+reg+'_EOF_'+str(i)+'_'+obs_name+'_'+fn,dpi=300,bbox_inches='tight')
    
    # Plot the leading PC time series.
    '''
    
    '''   
    for i in [0,1,2]: # len(pc):
        plt.figure()
        pc[:, i].plot(color='b', linewidth=2)
        ax = plt.gca()
        ax.axhline(0, color='k')
        #ax.set_ylim(-3, 3)
        ax.set_xlabel('month')
        ax.set_ylabel('Un-Normalized Units')
        ax.set_title('PC'+str(i)+' Time Series', fontsize=16)
        
        plt.savefig(workdir+reg+'_PCA_ts_'+str(i)+'_'+obs_name+'_'+fn,dpi=300,bbox_inches='tight')
    '''
    
    plt.show()
    
def kmeans(obs_dataset, obs_name, model_datasets, model_names, workdir):
    from xlearn22.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    #import geoxarray
 
    #choose
    annual_cycle=True
    
    ds = xr.DataArray(obs_dataset.values,
    coords={'time': obs_dataset.times,
            'lat': obs_dataset.lats, 
            'lon': obs_dataset.lons},
    dims=["time", "lat", "lon"])
    
    ds= ds.stack(z=("lat", "lon")) #s= s.unstack()
    
    if annual_cycle:
        dm = ds.groupby('time.month').mean()
       
        fn='annual_cycle'
    else:
        
        fn='raw_monthly'
        dm=ds
    
    #print('dm.shape', dm)
    #print('dm.values.shape', dm.values)
    
    #scaler =StandardScaler()
    #features =scaler.fit(dm)
    #features =features.transform(dm)
    #from sklearn_xarray import wrap
    #Xt = wrap(StandardScaler()).fit_transform(dm.data)
    
    
    #da = xr.DataArray(np.random.randn(100, 2, 3))
   
    m = KMeans(n_clusters=3, random_state=0).fit2(dm)
    print(62*64)
    #m= m.unstack() #error
    #kmeans.labels_
    
    #print('m.cluster_centers_.shape',m.cluster_centers_.shape)
    #print('m.labels_', m.labels_)
    #print('m.labels_.shape', m.labels_.shape)
    #print(m.n_clusters)
    fig, axes = plt.subplots(2,6, figsize=(8,8))
    regimes = ['NAO$^-$', 'NAO$^+$', 'Blocking', 'Atlantic Ridge']
    tags = list('abcd')
    
    
    
    for i in range (12): #(m.n_clusters):
        
        m.cluster_centers_da.sel(cluster=i).plot(ax=axes.flat[i])
    
    #dm.mean(dim='month').plot.contourf(ax=axes[1,1])
    plt.show()
    
    
    exit()
   
    
    for i in range(m.n_clusters):
        m.plot_cluster_centers(label=i, 
            #proj='ortho', 
            #plot_type='contourf+', 
            #levels=np.arange(-110, 111, 20),
            #units='m',
            ax=axes.flat[i])
        title = '{}, {}'.format(regimes[i],
                                axes.flat[i].title.get_text())
        plt.title(title)
        plt.text(0, 1, tags[i], 
                 transform=axes.flat[i].transAxes, 
                 va='bottom', 
                 fontsize=plt.rcParams['font.size']*2, 
                 fontweight='bold')
    
    plt.show()
    
def kmeans2(obs_dataset, obs_name, model_datasets, model_names, workdir):
    from xlearn22.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from kneed import KneeLocator
    #from sklearn.preprocessing import StandardScaler
    #import geoxarray
    print('*------------------')
    
    #choose
    annual_cycle=True
    with_pca=False #default n=3 
    
    fig, ax = plt.subplots(3,len(model_datasets), figsize=(8,8))
    
    for i in np.arange(len(model_datasets)):
        print(model_names[i])
     
        ds = xr.DataArray(model_datasets[i].values,
        coords={'time': model_datasets[i].times,
                'lat': model_datasets[i].lats, 
                'lon': model_datasets[i].lons},
        dims=["time", "lat", "lon"])
        
        lat_min = ds.lat.min()
        lat_max = ds.lat.max()
        lon_min = ds.lon.min()
        lon_max = ds.lon.max()
        x,y = np.meshgrid(ds.lon, ds.lat)
        
        m = Basemap(ax=ax[0,i], projection ='cyl', 
                llcrnrlat = lat_min, #+1*0.22, 
                urcrnrlat = lat_max, #-6*0.22,
                llcrnrlon = lon_min, #+4*0.22, 
                urcrnrlon = lon_max, #-3*0.22, 
                resolution = 'l', fix_aspect=False)
        m.drawcoastlines(linewidth=1)
        m.drawcountries(linewidth=1)
        m.drawstates(linewidth=0.5, color='w')
        
        ds= ds.stack(z=("lat", "lon")) #s= s.unstack()
        
        if annual_cycle:
            dm = ds.groupby('time.month').mean()
           
            fn='annual_cycle'
        else:
            
            fn='raw_monthly'
            dm=ds
       
        m = KMeans(n_clusters=3, random_state=0).fit2(dm, annual_cycle, with_pca)
                   
        #print(62*64)
        #m= m.unstack() #error
        #kmeans.labels_
        #print('m.cluster_centers_',m.cluster_centers_)
        print('m.cluster_centers_.shape',m.cluster_centers_.shape)
        #print('m.labels_', m.labels_)
        print('m.labels_.shape', m.labels_.shape)
        #print(m.n_clusters)
       
        
        p=m.cluster_centers_da.sel(cluster=0).plot(ax=ax[0,i], 
          levels=np.arange(4),
          #cmap='rainbow',
          colors=['blue','red','pink'],
          add_colorbar = False)
        ax[0,i].set_title(model_names[i], fontsize=9)
        ax[0,i].set_xticks([95,100,105])
        ax[0,i].set_xticklabels(['95E','100E','105E'])
        ax[0,i].tick_params(axis='x', pad=1,labelsize=8)
        ax[0,i].set_xlabel('')
        ax[0,2].set_ylabel('')
        ax[0,0].set_yticks([-5,0,5])
        ax[0,0].set_yticklabels(['5S','0','5N'])
        '''
        z=[-1,-3,-5] #Sumatera
        co=['blue','red','pink']
        for ii in range (m.n_clusters):
            frac = '{:4.1f}%'.format(KMeans.get_cluster_fraction(m, label=ii)*100)
            #print('title1', title1)
            
            f=str(co[ii])+'='+str(frac)
           
            ax[0,i].text(lon_min,z[ii],f,color='black')
        '''
        
        #SEA
        z=[20,16,12]
        co=['blue','red','pink']
        for ii in range (m.n_clusters):
            frac = '{:4.1f}%'.format(KMeans.get_cluster_fraction(m, label=ii)*100)
            #print('title1', title1)
            
            f=str(co[ii])+'='+str(frac)
           
            ax[0,i].text(lon_max-17,z[ii],f,color='black')
       
        
        if i<len(model_datasets)-1:
            ax[0,i+1].set_yticks([])
            ax[0,i+1].set_yticklabels([])
            ax[0,i+1].set_ylabel('')
        
        x_tick=['J','F','M','A','M','J','J','A','S','O','N','D']
        x=np.arange(m.cluster_centers_.shape[1])
        
        #c=['black','green','yellow']
        c=['blue','red', 'pink']
        #c=['purple','green','red']
        lw=['3','2','1']
        for ii in range (m.n_clusters):
            
            ax[1,i].plot(x, m.cluster_centers_[ii], color=c[ii],lw=lw[ii]) #, alpha=.7) #label = ii+1)
            #ax[0,i].plot(x, m.cluster_centers_[ii],lw=2, label = ii+1)
            #ax[1,i].set_ylim(1,12)
            #title1 = '{:4.1f}%'.format(KMeans.get_cluster_fraction(m, label=ii)*100)
            #print('title1', title1)
            #ax[1,i].text(0,5,title1,color='black')
        #f.append(m.cluster_centers_[m.n_clusters-1].max()*1.05)
        if i<len(model_datasets)-1:
            ax[1,i+1].set_yticks([])
            ax[1,i+1].set_yticklabels([])
        #dm.mean(dim='month').plot.contourf(ax=axes[1,1])
        #ax[1,i].set_title(model_names[i], fontsize=10)
        ax[1,i].set_xticks(x)
        if annual_cycle and not with_pca:
            ax[1,i].set_xticklabels(x_tick, fontsize=8)
        #ax[0,i].text(3,f[i],model_names[i])
        ax[1,0].set_ylabel('Mean pr (mm/day)')
        #plt.legend(bbox_to_anchor=(1, .4), loc='best', prop={'size':8.5}, frameon=False) 
        
        '''
        sse = []
        silhouette_coef = []
        #fit2 dicoba kini fit3
        for k in range(1, 7):
            print('k=',k)
            kmeans, features = KMeans(n_clusters=k,  random_state=42).fit3(dm, annual_cycle, with_pca)
            sse.append(kmeans.inertia_) 
            #print('kmeans.labels.shape', kmeans.labels_.shape)
            #print('features.shape', features.shape)
            if k>1:
                label2=kmeans.cluster_centers_da.sel(cluster=0).stack(z=("lat", "lon"))
                label2=label2.dropna(dim=('z'))
                #print('label2', label2)
           
                score = silhouette_score(features, label2)
            else:
                score=0
            silhouette_coef.append(score)
        
        kl = KneeLocator(
            range(1, 7), sse, curve="convex", direction="decreasing"
            )
        c=kl.elbow
        print('Optimum cluster1=',c)
        
        kl = KneeLocator(
            range(1, 7), silhouette_coef, curve="convex", direction="decreasing"
            )
        c2=kl.elbow
        
        print('Optimum cluster2=',c2)
        print('sse[c-1]',sse)   
        #plt.style.use("fivethirtyeight")
        ax[2,i].plot(range(1, 7), sse, color='black')
        ax[2,i].axvline(x=c,ymin=0, ymax=sse[c-1]/100000, ls=':', color='black') 
        ax[2,0].ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        ax[2,i].set_xticks(range(1, 7))
        ax[2,0].set_xlabel("N optimum of Clusters", fontsize=7)
        #ax[2,i].set_yticks([])
        #ax[2,i].set_yticklabels([])
        ax[2,0].set_ylabel("SSE")
        if i<len(model_datasets)-1:
            ax[2,i+1].set_yticks([])
            ax[2,i+1].set_yticklabels([])
        
        s=np.array(silhouette_coef)
        c2=np.where(s==s.max())[0][0] # [0][0] first only
        print('Optimum cluster3=',c2)
        
        ax2 = ax[2,i].twinx()  # instantiate a second axes that shares the same x-axis

        color = 'tab:red'
        
        ax2.plot(range(1, 7), silhouette_coef, color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.axvline(x=c2+1, ls=':', color='red') 
        if i<len(model_datasets)-1:
            ax2.set_yticks([])
            ax2.set_yticklabels([])        
        if i==len(model_datasets)-1:
            ax2.set_ylabel('silhouette_coef', color=color)
        
        '''   
        
        '''   
        color = 'tab:red'
        ax[3,0].set_ylabel('silhouette_coef', color=color)  
        ax[3,i].plot(range(1, 7), silhouette_coef, color=color)
        ax[3,0].tick_params(axis='y', labelcolor=color)
        ''' 
           
        #untuk model<2
        ax[0,1].set_ylabel('')
        
        # #untuk model<6
        # ax[0,1].set_ylabel('')
        # ax[0,2].set_ylabel('')
        # ax[0,3].set_ylabel('')
        # ax[0,4].set_ylabel('')
        # ax[0,5].set_ylabel('')
        # ax[2,1].set_yticklabels([])
        # ax[2,2].set_yticklabels([])
        # ax[2,3].set_yticklabels([])
        # ax[2,4].set_yticklabels([])
        # ax[2,5].set_yticklabels([])
       
    
    plt.subplots_adjust(hspace=.35,wspace=.05)
    
    cax = fig.add_axes([0.91, 0.38, 0.02, 0.5])
   
    cbar = plt.colorbar(p,cax=cax)

    cbar.ax.get_yaxis().set_ticks([])

    for j in range(1, 4, 1):
        cbar.ax.text(1.5, (j-1+.5), j, ha='center', va='center', color='black')
    cbar.ax.get_yaxis().labelpad = 15
    #plt.title(model_names[i])
    
    plt.draw()
    plt.savefig(workdir+reg+'_cluster_xxx'+fn,dpi=300,bbox_inches='tight')
        
        
    plt.show()

def kmeans2a(obs_dataset, obs_name, model_datasets, model_names, workdir):
    
    #tidak pakai pengaturan warna
    from xlearn22.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from kneed import KneeLocator
 
    print('*------------------')
    
    #choose
    n_clusters=4
    annual_cycle=False
    with_pca=False #default n=3 
    
    #domain
    sea=False
    indo=False
    sumatera=True
    jambi=False
    
    fig, ax = plt.subplots(3,len(model_datasets), figsize=(8,8))
    
    for i in np.arange(len(model_datasets)):
        print(model_names[i])
     
        ds = xr.DataArray(model_datasets[i].values,
        coords={'time': model_datasets[i].times,
                'lat': model_datasets[i].lats, 
                'lon': model_datasets[i].lons},
        dims=["time", "lat", "lon"])
        
        
        
        if indo:
            ds = ds.where(
                    (ds.lat > -11) & (ds.lat < 6.5) & 
                    (ds.lon > 95) & (ds.lon < 141), drop=True)
        #-6.5 7 93 107
        
        if sumatera:
            ds = ds.where(
                    (ds.lat > -6.5) & (ds.lat < 6.5) & 
                    (ds.lon > 94) & (ds.lon < 107), drop=True)
               
        if jambi:
            ds = ds.where(
                    (ds.lat > -3) & (ds.lat < -0.5) & 
                    (ds.lon > 99) & (ds.lon < 106), drop=True)
                   
        lat_min = ds.lat.min()
        lat_max = ds.lat.max()
        lon_min = ds.lon.min()
        lon_max = ds.lon.max()
        x,y = np.meshgrid(ds.lon, ds.lat)
        
        m = Basemap(ax=ax[0,i], projection ='cyl', 
                llcrnrlat = lat_min, #+1*0.22, 
                urcrnrlat = lat_max, #-6*0.22,
                llcrnrlon = lon_min, #+4*0.22, 
                urcrnrlon = lon_max, #-3*0.22, 
                resolution = 'l', fix_aspect=False)
        m.drawcoastlines(linewidth=1)
        m.drawcountries(linewidth=1)
        m.drawstates(linewidth=0.5, color='w')
        
        ds= ds.stack(z=("lat", "lon")) #s= s.unstack()
        
        if annual_cycle:
            dm = ds.groupby('time.month').mean()
           
            fn='annual_cycle'
        else:
            
            fn='raw_monthly'
            dm=ds
       
        km = KMeans(n_clusters=n_clusters, random_state=42).fit2(dm, annual_cycle, with_pca)
        '''
        cek=km.cluster_centers_da.sel(cluster=0)
        print('cek=',cek)
        exit()
        '''
        p=km.cluster_centers_da.sel(cluster=0).plot(ax=ax[0,i], 
          levels=np.arange(n_clusters+1),
          #cmap='rainbow',
          colors=['blue','red','pink','green', 'orange'],
          add_colorbar = False)
        ax[0,i].set_title(model_names[i], fontsize=9)
        
        if sea:
            ax[0,i].set_xticks([100,120,140])
            ax[0,i].set_xticklabels(['100E','120E','140E'])
            ax[0,i].tick_params(axis='x', pad=1,labelsize=8)
            ax[0,i].set_xlabel('')
            ax[0,i].set_ylabel('')
            ax[0,0].set_yticks([-10,0,10,20])
            ax[0,0].set_yticklabels(['10S','0','10N','20N'])
            ax[0,0].tick_params(axis='y', pad=1,labelsize=8)
        
        if indo:
            ax[0,i].set_xticks([105,120,135])
            ax[0,i].set_xticklabels(['105E','120E','135E'])
            ax[0,i].tick_params(axis='x', pad=1,labelsize=8)
            ax[0,i].set_xlabel('')
            if i>0:
                ax[0,i].set_ylabel('')
            ax[0,0].set_yticks([-11,-5,0,5])
            ax[0,0].set_yticklabels(['10S','5S','0','5N'])
            ax[0,0].tick_params(axis='y', pad=1,labelsize=8)
            
        if sumatera:
            ax[0,i].set_xticks([95,100,105])
            ax[0,i].set_xticklabels(['95E','100E','105E'])
            ax[0,i].tick_params(axis='x', pad=1,labelsize=8)
            ax[0,i].set_xlabel('')
            ax[0,i].set_ylabel('')
            ax[0,0].set_yticks([-5,0,5])
            ax[0,0].set_yticklabels(['5S','0','5N'])
            ax[0,0].tick_params(axis='y', pad=1,labelsize=8)
            
        if jambi:
            ax[0,i].set_xticks([100,104])
            ax[0,i].set_xticklabels(['100$^\circ$E','104$^\circ$E'])
            ax[0,i].tick_params(axis='x', pad=1,labelsize=8)
            ax[0,i].set_xlabel('')
            ax[0,i].set_ylabel('')
            ax[0,0].set_yticks([-3,-2,-1])
            ax[0,0].set_yticklabels(['3$^\circ$S','2$^\circ$S','1$^\circ$S'])
            ax[0,0].tick_params(axis='y', pad=1,labelsize=8)
            
      
        
        if i<len(model_datasets)-1:
            ax[0,i+1].set_yticks([])
            ax[0,i+1].set_yticklabels([])
            ax[0,i+1].set_ylabel('')
        
        x_tick=['J','F','M','A','M','J','J','A','S','O','N','D']
        
        x=np.arange(km.cluster_centers_.shape[1])
        
        c=['blue','red', 'pink','green', 'orange']
      
        lw=['2','2','2','2','2']
        for ii in range (km.n_clusters):
            
            ax[1,i].plot(x, km.cluster_centers_[ii], color=c[ii],lw=lw[ii]) #, alpha=.7) #label = ii+1)
       
        ax[1,i].set_xticks(x)
        if annual_cycle and not with_pca:
            ax[1,i].set_xticklabels(x_tick, fontsize=8)
    
        ax[1,0].set_ylabel('Mean pr (mm/month)')
        ax[1,i].tick_params(axis='y', pad=1,labelsize=8)
        
        #jumlah cluster optimum
        #The Sum of Squared Error (SSE) is a measure used in k-means clustering 
        #to evaluate the quality of the cluster assignments. 
        sse = [] 
        silhouette_coef = []
        #fit3 = fit2 hanya beda ini: return self, X_valid vs return self 
        for k in range(1, 7):
            print('k=',k)
            kmeans, features = KMeans(n_clusters=k,  random_state=42).fit3(dm, annual_cycle, with_pca)
            sse.append(kmeans.inertia_) 
        
            if k>1:
                label2=kmeans.cluster_centers_da.sel(cluster=0).stack(z=("lat", "lon"))
                label2=label2.dropna(dim=('z'))
                score = silhouette_score(features, label2)
            else:
                score=0
            silhouette_coef.append(score)
        
        kl = KneeLocator(
            range(1, 7), sse, curve="convex", direction="decreasing"
            )
        c=kl.elbow
        print('Optimum cluster1=',c)
        
        kl = KneeLocator(
            range(1, 7), silhouette_coef, curve="convex", direction="decreasing"
            )
        c2=kl.elbow
        
        print('Optimum cluster2=',c2)
        print('sse[c-1]',sse)  
 
        #plt.style.use("fivethirtyeight")
        ax[2,i].plot(range(1, 7), sse, color='black')
        ax[2,i].axvline(x=c,ymin=0, ymax=sse[c-1]/100000, ls=':', color='black') 
        ax[2,0].ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        ax[2,i].set_xticks(range(1, 7))
        ax[2,0].set_xlabel("N optimum of Clusters", fontsize=7)

        ax[2,0].set_ylabel("SSE")
        if i<len(model_datasets)-1:
            ax[2,i+1].set_yticks([])
            ax[2,i+1].set_yticklabels([])
        
        s=np.array(silhouette_coef)
        c2=np.where(s==s.max())[0][0] # [0][0] first only
        print('Optimum cluster3=',c2)
        
        ax2 = ax[2,i].twinx()  # instantiate a second axes that shares the same x-axis

        color = 'tab:red'
        
        ax2.plot(range(1, 7), silhouette_coef, color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.axvline(x=c2+1, ls=':', color='red') 
        if i<len(model_datasets)-1:
            ax2.set_yticks([])
            ax2.set_yticklabels([])        
        if i==len(model_datasets)-1:
            ax2.set_ylabel('silhouette_coef', color=color)
        
        '''   
        color = 'tab:red'
        ax[3,0].set_ylabel('silhouette_coef', color=color)  
        ax[3,i].plot(range(1, 7), silhouette_coef, color=color)
        ax[3,0].tick_params(axis='y', labelcolor=color)
        ''' 
    
    plt.subplots_adjust(hspace=.3,wspace=.17)
    
    cax = fig.add_axes([0.91, 0.38, 0.02, 0.5])
   
    cbar = plt.colorbar(p,cax=cax)

    cbar.ax.get_yaxis().set_ticks([])

    for j in range(1, n_clusters+1, 1):
        cbar.ax.text(1.5, (j-1+.5), j, ha='center', va='center', color='black')
    cbar.ax.get_yaxis().labelpad = 15
    #plt.title(model_names[i])
    
    plt.draw()
    plt.savefig(workdir+reg+'_cluster_0'+fn,dpi=300,bbox_inches='tight')   
    plt.show()

def kmeans_sea(obs_dataset, obs_name, model_datasets, model_names, workdir):
    #pakai pengaturan warna u 3 dataset ERA5 CNRM dan MME
    from xlearn22.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from kneed import KneeLocator
    #from sklearn.preprocessing import StandardScaler
    #import geoxarray
    print('*------------------')
    
    #choose
    n_clusters=3
    annual_cycle=True
    with_pca=False #default n=3 
    
    #domain
    sea=False
    indo=False
    sumatera=True
    jambi=False
    
    fig, ax = plt.subplots(3,len(model_datasets), figsize=(8,8))
    
    for i in np.arange(len(model_datasets)):
        print(model_names[i])
     
        ds = xr.DataArray(model_datasets[i].values,
        coords={'time': model_datasets[i].times,
                'lat': model_datasets[i].lats, 
                'lon': model_datasets[i].lons},
        dims=["time", "lat", "lon"])
        
        
        
        if indo:
            ds = ds.where(
                    (ds.lat > -11) & (ds.lat < 6.5) & 
                    (ds.lon > 95) & (ds.lon < 141), drop=True)
        #-6.5 7 93 107
        '''
        if sumatera:
            ds = ds.where(
                    (ds.lat > -6.5) & (ds.lat < 6.5) & 
                    (ds.lon > 94) & (ds.lon < 107), drop=True)
        '''          
        if jambi:
            ds = ds.where(
                    (ds.lat > -3) & (ds.lat < -0.5) & 
                    (ds.lon > 99) & (ds.lon < 106), drop=True)
                   
        lat_min = ds.lat.min()
        lat_max = ds.lat.max()
        lon_min = ds.lon.min()
        lon_max = ds.lon.max()
        x,y = np.meshgrid(ds.lon, ds.lat)
        
        m = Basemap(ax=ax[0,i], projection ='cyl', 
                llcrnrlat = lat_min, #+1*0.22, 
                urcrnrlat = lat_max, #-6*0.22,
                llcrnrlon = lon_min, #+4*0.22, 
                urcrnrlon = lon_max, #-3*0.22, 
                resolution = 'l', fix_aspect=False)
        m.drawcoastlines(linewidth=1)
        m.drawcountries(linewidth=1)
        m.drawstates(linewidth=0.5, color='w')
        
        ds= ds.stack(z=("lat", "lon")) #s= s.unstack()
        
        if annual_cycle:
            dm = ds.groupby('time.month').mean()
           
            fn='annual_cycle'
        else:
            
            fn='raw_monthly'
            dm=ds
       
        km = KMeans(n_clusters=n_clusters, random_state=0).fit2(dm, annual_cycle, with_pca)
                   
        #print(62*64)
        #m= m.unstack() #error
        #kmeans.labels_
        #print('m.cluster_centers_',m.cluster_centers_)
        print('m.cluster_centers_.shape',km.cluster_centers_.shape)
        #print('m.labels_', m.labels_)
        print('m.labels_.shape', km.labels_.shape)
        #print(m.n_clusters)
        '''
        print(m.cluster_centers_da)
        import pandas as pd
        result = pd.DataFrame(m.cluster_centers_da.sel(cluster=0))
        result.to_excel(workdir+'cluster'+str(i+1)+'.xlsx')
        '''
        #plot kode per cluster
        ds0 = xr.DataArray(model_datasets[i].values,
        coords={'time': model_datasets[i].times,
                'lat': model_datasets[i].lats, 
                'lon': model_datasets[i].lons},
        dims=["time", "lat", "lon"])
        n=km.cluster_centers_da.sel(cluster=0)
        #print kode cluster pada map
        '''
        print(n.shape)
        print(n[0,0].values)
        print(ds0.lon)
        print(ds0.lon[0])
        print('np.arange(len(ds0.lat))',np.arange(len(ds0.lat)))
        
        for ii in np.arange(len(ds0.lon)-2):
            for j in np.arange(len(ds0.lat)-2):
                #print(i,j)
                #print(ds0.lon[i].values, ds0.lat[j].values)
                #print(n[i,j].values)
                if str(n[ii,j].values) != 'nan' and j%2==0 and ii%2==0:
                    #print('cek')
                    #print(ds0.lon[i].values, ds0.lat[j].values)
                    #print(int(n[ii,j].values))
                    #ax[0,i].text(ds0.lon[j], ds0.lat[ii], s=str(int(n[j,ii].values)), size=7, color='red') 
                    #ax[0,i].text(ds0.lon[::-1][ii], -1*ds0.lat[j], s=str(int(n[ii,j].values)), size=7, color='red') 
                    
                    #ax[0,i].text(ds0.lon[::-1][ii], ds0.lat[::-1][j], s=str(int(n[ii,j].values)), size=7, color='red') 
                    #x,y=m(ds0.lon[ii],ds0.lat[j])
                    x,y=m(ds0.lon[j],ds0.lat[ii])
                    ax[0,i].text(x, y, s=str(int(n[ii,j].values)), size=6, color='black') 
        plt.show()
        #print(m.cluster_centers_da.sel(cluster=0).shape)
        '''
        '''
        nn = n.where(n != 1) 
        nn = nn.where(nn != 2) 
              
        
        for ii in np.arange(len(ds0.lon)-2):
            for j in np.arange(len(ds0.lat)-2):
                #print(i,j)
                #print(ds0.lon[i].values, ds0.lat[j].values)
                #print(n[i,j].values)
                if str(nn[ii,j].values) != 'nan' and j%2==0 and ii%2==0:
               
                    x,y=m(ds0.lon[j],ds0.lat[ii])
                    ax[0,i].text(x, y, s=str(int(nn[ii,j].values)), size=6, color='black') 
        plt.show()
        exit()
        
        
        mask_array = np.zeros(ds0.values.shape)
        index=np.where(n==1)
        mask_array[index]=1
       
        dd=ma.array(ds0.values, mask= mask_array)
        
        index=np.where(n==2)
        mask_array[index]=1
       
        dd=ma.array(ds0.values, mask= mask_array)
        #ini ke xarray nya bgm?
        '''    
                
        #xarray masking ==> kode hasil kmeans clustering sbg mask 
        #buat coord mask isi dengan kode cluster
        ds0.coords['mask'] = (('lat', 'lon'), n.values)
        da=ds0
        #contoh penggunaan ==> hanya cluster 1 yg diproses
        #ds0.mean('time').where(ds.mask == 1).plot()s
        #tes=ds0.mean('time').where(ds0.mask == 0)
        #tes.plot(ax=ax[0,i])
        
        p=km.cluster_centers_da.sel(cluster=0).plot(ax=ax[0,i], 
          levels=np.arange(n_clusters+1),
          #cmap='rainbow',
          colors=['blue','red','pink','green', 'orange'],
          add_colorbar = False)
        ax[0,i].set_title(model_names[i], fontsize=9)
        
        if sea:
            ax[0,i].set_xticks([100,120,140])
            ax[0,i].set_xticklabels(['100E','120E','140E'])
            ax[0,i].tick_params(axis='x', pad=1,labelsize=8)
            ax[0,i].set_xlabel('')
            ax[0,i].set_ylabel('')
            ax[0,0].set_yticks([-10,0,10,20])
            ax[0,0].set_yticklabels(['10S','0','10N','20N'])
            ax[0,0].tick_params(axis='y', pad=1,labelsize=8)
        
        if indo:
            ax[0,i].set_xticks([105,120,135])
            ax[0,i].set_xticklabels(['105E','120E','135E'])
            ax[0,i].tick_params(axis='x', pad=1,labelsize=8)
            ax[0,i].set_xlabel('')
            if i>0:
                ax[0,i].set_ylabel('')
            ax[0,0].set_yticks([-11,-5,0,5])
            ax[0,0].set_yticklabels(['10S','5S','0','5N'])
            ax[0,0].tick_params(axis='y', pad=1,labelsize=8)
            
        if sumatera:
            ax[0,i].set_xticks([95,100,105])
            ax[0,i].set_xticklabels(['95E','100E','105E'])
            ax[0,i].tick_params(axis='x', pad=1,labelsize=8)
            ax[0,i].set_xlabel('')
            ax[0,i].set_ylabel('')
            ax[0,0].set_yticks([-5,0,5])
            ax[0,0].set_yticklabels(['5S','0','5N'])
            ax[0,0].tick_params(axis='y', pad=1,labelsize=8)
            
        if jambi:
            ax[0,i].set_xticks([100,104])
            ax[0,i].set_xticklabels(['100$^\circ$E','104$^\circ$E'])
            ax[0,i].tick_params(axis='x', pad=1,labelsize=8)
            ax[0,i].set_xlabel('')
            ax[0,i].set_ylabel('')
            ax[0,0].set_yticks([-3,-2,-1])
            ax[0,0].set_yticklabels(['3$^\circ$S','2$^\circ$S','1$^\circ$S'])
            ax[0,0].tick_params(axis='y', pad=1,labelsize=8)
            
        if i==1:
            #CNRM sea hijau ke merah 4ke2, biru ke hijau 1ke4
          
            #bi me pi hi
            #0  1  2  3
            #hi-me, me-hi, biru-pink, pink-biru
            #31,13,02,20
           
            #14,31,43
            #24,02,40
                 
            #print(da.where(da.mask != 3))
            dd=da.mask.where(da.mask != 1, other=4)
            #print(dd)
            dd=dd.where(dd.mask != 3, other=1)
            #print(dd)
            dd=dd.where(dd != 4, other=3)
            
            dd=dd.where(dd != 2, other=4)
            dd=dd.where(dd != 0, other=2)
            dd=dd.where(dd != 4, other=0)
            p=dd.plot(ax=ax[0,1], 
              levels=np.arange(n_clusters+1),
              #cmap='rainbow',
              colors=['blue','red','pink','green', 'orange'],
              add_colorbar = False)
            ax[0,1].set_title(model_names[i], fontsize=9)
            
        if i==2:
            #MME sea pink ke merah dsn
          
            #bi me pi hi
            #0  1  2  3
            #me-pi,pink-merah
            #12, 21
            #24,12,41
                             
            #print(da.where(da.mask != 3))
            dd=da.mask.where(da.mask != 2, other=4)
            
            dd=dd.where(dd.mask != 1, other=2)
           
            dd=dd.where(dd != 4, other=1)
            
            p=dd.plot(ax=ax[0,2], 
              levels=np.arange(n_clusters+1),
              #cmap='rainbow',
              colors=['blue','red','pink','green', 'orange'],
              add_colorbar = False)
            ax[0,2].set_title(model_names[i], fontsize=9)
            ax[0,2].set_ylabel('')
                   
        #SEA 4 color
        z=[20,16,12,8]
        co=['blue','red','pink','green', 'orange']
        '''
        #tampilkan fraksi cluster
        for ii in range (m.n_clusters):
            frac = '{:4.1f}%'.format(KMeans.get_cluster_fraction(m, label=ii)*100)
            #print('title1', title1)
            
            f=str(co[ii])+'='+str(frac)
           
            ax[0,i].text(lon_max-17,z[ii],f,color='black')
        '''
        
        if i<len(model_datasets)-1:
            ax[0,i+1].set_yticks([])
            ax[0,i+1].set_yticklabels([])
            ax[0,i+1].set_ylabel('')
        
        x_tick=['J','F','M','A','M','J','J','A','S','O','N','D']
        x=np.arange(km.cluster_centers_.shape[1])
        
        if i==1: #khusus CNRM
             c=['pink','green','blue','red','orange',]
        elif i==2: #khusus MME
             c=['blue','pink', 'red','green', 'orange']
        else:
            c=['blue','red', 'pink','green', 'orange']
      
        lw=['2','2','2','2','2']
        for ii in range (km.n_clusters):
            
            ax[1,i].plot(x, km.cluster_centers_[ii], color=c[ii],lw=lw[ii]) #, alpha=.7) #label = ii+1)
            #ax[0,i].plot(x, m.cluster_centers_[ii],lw=2, label = ii+1)
            #ax[1,i].set_ylim(1,12)
            #title1 = '{:4.1f}%'.format(KMeans.get_cluster_fraction(m, label=ii)*100)
            #print('title1', title1)
            #ax[1,i].text(0,5,title1,color='black')
        #f.append(m.cluster_centers_[m.n_clusters-1].max()*1.05)
        #if i<len(model_datasets)-1:
            #ax[1,i+1].set_yticks([])
            #ax[1,i+1].set_yticklabels([])
        #dm.mean(dim='month').plot.contourf(ax=axes[1,1])
        #ax[1,i].set_title(model_names[i], fontsize=10)
        ax[1,i].set_xticks(x)
        if annual_cycle and not with_pca:
            ax[1,i].set_xticklabels(x_tick, fontsize=8)
        #ax[0,i].text(3,f[i],model_names[i])
        ax[1,0].set_ylabel('Mean pr (mm/month)')
        ax[1,i].tick_params(axis='y', pad=1,labelsize=8)
        #plt.legend(bbox_to_anchor=(1, .4), loc='best', prop={'size':8.5}, frameon=False) 
        
        '''
        sse = []
        silhouette_coef = []
        #fit2 dicoba kini fit3
        for k in range(1, 7):
            print('k=',k)
            kmeans, features = KMeans(n_clusters=k,  random_state=42).fit3(dm, annual_cycle, with_pca)
            sse.append(kmeans.inertia_) 
            #print('kmeans.labels.shape', kmeans.labels_.shape)
            #print('features.shape', features.shape)
            if k>1:
                label2=kmeans.cluster_centers_da.sel(cluster=0).stack(z=("lat", "lon"))
                label2=label2.dropna(dim=('z'))
                #print('label2', label2)
           
                score = silhouette_score(features, label2)
            else:
                score=0
            silhouette_coef.append(score)
        
        kl = KneeLocator(
            range(1, 7), sse, curve="convex", direction="decreasing"
            )
        c=kl.elbow
        print('Optimum cluster1=',c)
        
        kl = KneeLocator(
            range(1, 7), silhouette_coef, curve="convex", direction="decreasing"
            )
        c2=kl.elbow
        
        print('Optimum cluster2=',c2)
        print('sse[c-1]',sse)  
        
        
        #plt.style.use("fivethirtyeight")
        ax[2,i].plot(range(1, 7), sse, color='black')
        ax[2,i].axvline(x=c,ymin=0, ymax=sse[c-1]/100000, ls=':', color='black') 
        ax[2,0].ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        ax[2,i].set_xticks(range(1, 7))
        ax[2,0].set_xlabel("N optimum of Clusters", fontsize=7)
        #ax[2,i].set_yticks([])
        #ax[2,i].set_yticklabels([])
        ax[2,0].set_ylabel("SSE")
        if i<len(model_datasets)-1:
            ax[2,i+1].set_yticks([])
            ax[2,i+1].set_yticklabels([])
        
        s=np.array(silhouette_coef)
        c2=np.where(s==s.max())[0][0] # [0][0] first only
        print('Optimum cluster3=',c2)
        
        ax2 = ax[2,i].twinx()  # instantiate a second axes that shares the same x-axis

        color = 'tab:red'
        
        ax2.plot(range(1, 7), silhouette_coef, color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.axvline(x=c2+1, ls=':', color='red') 
        if i<len(model_datasets)-1:
            ax2.set_yticks([])
            ax2.set_yticklabels([])        
        if i==len(model_datasets)-1:
            ax2.set_ylabel('silhouette_coef', color=color)
        
        '''
        
        '''   
        color = 'tab:red'
        ax[3,0].set_ylabel('silhouette_coef', color=color)  
        ax[3,i].plot(range(1, 7), silhouette_coef, color=color)
        ax[3,0].tick_params(axis='y', labelcolor=color)
        ''' 
           
        #untuk model<2
        ax[0,1].set_ylabel('')
        
        # #untuk model<6
        # ax[0,1].set_ylabel('')
        # ax[0,2].set_ylabel('')
        # ax[0,3].set_ylabel('')
        # ax[0,4].set_ylabel('')
        # ax[0,5].set_ylabel('')
        # ax[2,1].set_yticklabels([])
        # ax[2,2].set_yticklabels([])
        # ax[2,3].set_yticklabels([])
        # ax[2,4].set_yticklabels([])
        # ax[2,5].set_yticklabels([])
       
    
    plt.subplots_adjust(hspace=.3,wspace=.17)
    
    cax = fig.add_axes([0.91, 0.38, 0.02, 0.5])
   
    cbar = plt.colorbar(p,cax=cax)

    cbar.ax.get_yaxis().set_ticks([])

    for j in range(1, n_clusters+1, 1):
        cbar.ax.text(1.5, (j-1+.5), j, ha='center', va='center', color='black')
    cbar.ax.get_yaxis().labelpad = 15
    #plt.title(model_names[i])
    
    plt.draw()
    plt.savefig(workdir+reg+'_cluster_xxx'+fn,dpi=300,bbox_inches='tight')
        
        
    plt.show()
    
def kmeans_sea2(obs_dataset, obs_name, model_datasets, model_names, workdir):
    #cluster > 5 no SSE
    from xlearn22.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from kneed import KneeLocator
    #from sklearn.preprocessing import StandardScaler
    #import geoxarray
    print('*------------------')
    
    #choose
    n_clusters=3
    annual_cycle=True
    with_pca=False #default n=3 
    
    #domain
    sea=True
    indo=False
    sumatera=False
    jambi=False
    
    fig, ax = plt.subplots(3,len(model_datasets), figsize=(8,8))
    
    for i in np.arange(len(model_datasets)):
        print(model_names[i])
     
        ds = xr.DataArray(model_datasets[i].values,
        coords={'time': model_datasets[i].times,
                'lat': model_datasets[i].lats, 
                'lon': model_datasets[i].lons},
        dims=["time", "lat", "lon"])
        
        
        
        if indo:
            ds = ds.where(
                    (ds.lat > -11) & (ds.lat < 6.5) & 
                    (ds.lon > 95) & (ds.lon < 141), drop=True)
        #-6.5 7 93 107
        '''
        if sumatera:
            ds = ds.where(
                    (ds.lat > -6.5) & (ds.lat < 6.5) & 
                    (ds.lon > 94) & (ds.lon < 107), drop=True)
        '''          
        if jambi:
            ds = ds.where(
                    (ds.lat > -3) & (ds.lat < -0.5) & 
                    (ds.lon > 99) & (ds.lon < 106), drop=True)
                   
        lat_min = ds.lat.min()
        lat_max = ds.lat.max()
        lon_min = ds.lon.min()
        lon_max = ds.lon.max()
        x,y = np.meshgrid(ds.lon, ds.lat)
        
        m = Basemap(ax=ax[0,i], projection ='cyl', 
                llcrnrlat = lat_min, #+1*0.22, 
                urcrnrlat = lat_max, #-6*0.22,
                llcrnrlon = lon_min, #+4*0.22, 
                urcrnrlon = lon_max, #-3*0.22, 
                resolution = 'l', fix_aspect=False)
        m.drawcoastlines(linewidth=1)
        m.drawcountries(linewidth=1)
        m.drawstates(linewidth=0.5, color='w')
        
        ds= ds.stack(z=("lat", "lon")) #s= s.unstack()
        
        if annual_cycle:
            dm = ds.groupby('time.month').mean()
           
            fn='annual_cycle'
        else:
            
            fn='raw_monthly'
            dm=ds
       
        km = KMeans(n_clusters=n_clusters, random_state=0).fit2(dm, annual_cycle, with_pca)
                   
       
        p=km.cluster_centers_da.sel(cluster=0).plot(ax=ax[0,i], 
          levels=np.arange(n_clusters+1),
          #cmap='rainbow',
          #colors=['blue','red','pink','green', 'orange'],
          colors = ['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet', 'darkred', 'darkorange', 'darkgreen'],

          add_colorbar = False)
        ax[0,i].set_title(model_names[i], fontsize=9)
        
        if sea:
            ax[0,i].set_xticks([100,120,140])
            ax[0,i].set_xticklabels(['100E','120E','140E'])
            ax[0,i].tick_params(axis='x', pad=1,labelsize=8)
            ax[0,i].set_xlabel('')
            ax[0,2].set_ylabel('')
            ax[0,0].set_yticks([-10,0,10,20])
            ax[0,0].set_yticklabels(['10S','0','10N','20N'])
            ax[0,0].tick_params(axis='y', pad=1,labelsize=8)
        
        if indo:
            ax[0,i].set_xticks([105,120,135])
            ax[0,i].set_xticklabels(['105E','120E','135E'])
            ax[0,i].tick_params(axis='x', pad=1,labelsize=8)
            ax[0,i].set_xlabel('')
            if i>0:
                ax[0,i].set_ylabel('')
            ax[0,0].set_yticks([-11,-5,0,5])
            ax[0,0].set_yticklabels(['10S','5S','0','5N'])
            ax[0,0].tick_params(axis='y', pad=1,labelsize=8)
            
        if sumatera:
            ax[0,i].set_xticks([95,100,105])
            ax[0,i].set_xticklabels(['95E','100E','105E'])
            ax[0,i].tick_params(axis='x', pad=1,labelsize=8)
            ax[0,i].set_xlabel('')
            ax[0,i].set_ylabel('')
            ax[0,0].set_yticks([-5,0,5])
            ax[0,0].set_yticklabels(['5S','0','5N'])
            ax[0,0].tick_params(axis='y', pad=1,labelsize=8)
            
        if jambi:
            ax[0,i].set_xticks([100,104])
            ax[0,i].set_xticklabels(['100$^\circ$E','104$^\circ$E'])
            ax[0,i].tick_params(axis='x', pad=1,labelsize=8)
            ax[0,i].set_xlabel('')
            ax[0,i].set_ylabel('')
            ax[0,0].set_yticks([-3,-2,-1])
            ax[0,0].set_yticklabels(['3$^\circ$S','2$^\circ$S','1$^\circ$S'])
            ax[0,0].tick_params(axis='y', pad=1,labelsize=8)
            
        
        '''           
        #SEA 4 color
        z=[20,16,12,8]
        co=['blue','red','pink','green', 'orange']
        
        #tampilkan fraksi cluster
        for ii in range (m.n_clusters):
            frac = '{:4.1f}%'.format(KMeans.get_cluster_fraction(m, label=ii)*100)
            #print('title1', title1)
            
            f=str(co[ii])+'='+str(frac)
           
            ax[0,i].text(lon_max-17,z[ii],f,color='black')
        '''
        
        if i<len(model_datasets)-1:
            ax[0,i+1].set_yticks([])
            ax[0,i+1].set_yticklabels([])
            ax[0,i+1].set_ylabel('')
        
        x_tick=['J','F','M','A','M','J','J','A','S','O','N','D']
        x=np.arange(km.cluster_centers_.shape[1])
        
        
        #c=['blue','red', 'pink','green', 'orange']
        c=colors = ['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet', 'darkred', 'darkorange', 'darkgreen']
      
        lw=['2','2','2','2','2']
        for ii in range (km.n_clusters):
            
            ax[1,i].plot(x, km.cluster_centers_[ii], 
            color=c[ii],
            #lw=lw[ii]
            ) 
     
        ax[1,i].set_xticks(x)
        if annual_cycle and not with_pca:
            ax[1,i].set_xticklabels(x_tick, fontsize=8)
        #ax[0,i].text(3,f[i],model_names[i])
        ax[1,0].set_ylabel('Mean pr (mm/month)')
        ax[1,i].tick_params(axis='y', pad=1,labelsize=8)
        #plt.legend(bbox_to_anchor=(1, .4), loc='best', prop={'size':8.5}, frameon=False) 
        
        #untuk model<2
        ax[0,1].set_ylabel('')
        
        '''
        sse = []
        silhouette_coef = []
        #fit2 dicoba kini fit3
        for k in range(1, 7):
            print('k=',k)
            kmeans, features = KMeans(n_clusters=k,  random_state=42).fit3(dm, annual_cycle, with_pca)
            sse.append(kmeans.inertia_) 
            #print('kmeans.labels.shape', kmeans.labels_.shape)
            #print('features.shape', features.shape)
            if k>1:
                label2=kmeans.cluster_centers_da.sel(cluster=0).stack(z=("lat", "lon"))
                label2=label2.dropna(dim=('z'))
                #print('label2', label2)
                score = silhouette_score(features, label2)
            else:
                score=0
            silhouette_coef.append(score)
        
        kl = KneeLocator(
            range(1, 7), sse, curve="convex", direction="decreasing"
            )
        c=kl.elbow
        print('Optimum cluster1=',c)
        
        kl = KneeLocator(
            range(1, 7), silhouette_coef, curve="convex", direction="decreasing"
            )
        c2=kl.elbow
        
        print('Optimum cluster2=',c2)
        print('sse[c-1]',sse)  
        
        
        #plt.style.use("fivethirtyeight")
        ax[2,i].plot(range(1, 7), sse, color='black')
        ax[2,i].axvline(x=c,ymin=0, ymax=sse[c-1]/100000, ls=':', color='black') 
        ax[2,0].ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        ax[2,i].set_xticks(range(1, 7))
        ax[2,0].set_xlabel("N optimum of Clusters", fontsize=7)
        #ax[2,i].set_yticks([])
        #ax[2,i].set_yticklabels([])
        ax[2,0].set_ylabel("SSE")
        if i<len(model_datasets)-1:
            ax[2,i+1].set_yticks([])
            ax[2,i+1].set_yticklabels([])
        
        s=np.array(silhouette_coef)
        c2=np.where(s==s.max())[0][0] # [0][0] first only
        print('Optimum cluster3=',c2)
        
        ax2 = ax[2,i].twinx()  # instantiate a second axes that shares the same x-axis

        color = 'tab:red'
        
        ax2.plot(range(1, 7), silhouette_coef, color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.axvline(x=c2+1, ls=':', color='red') 
        if i<len(model_datasets)-1:
            ax2.set_yticks([])
            ax2.set_yticklabels([])        
        if i==len(model_datasets)-1:
            ax2.set_ylabel('silhouette_coef', color=color)
        
        '''
    
    plt.subplots_adjust(hspace=.3,wspace=.17)
    
    cax = fig.add_axes([0.91, 0.38, 0.02, 0.5])
   
    cbar = plt.colorbar(p,cax=cax)

    cbar.ax.get_yaxis().set_ticks([])

    for j in range(1, n_clusters+1, 1):
        cbar.ax.text(1.5, (j-1+.5), j, ha='center', va='center', color='black')
    cbar.ax.get_yaxis().labelpad = 15
    #plt.title(model_names[i])
    
    plt.draw()
    plt.savefig(workdir+reg+'_cluster_yyy'+fn,dpi=300,bbox_inches='tight')
        
        
    plt.show()

def kmeans_hirarki(obs_dataset, obs_name, model_datasets, model_names, workdir):

    annual_cycle=True
    #domain
    sea=True
    indo=True
    sumatera=True
   
    plt.figure(figsize=(15, 5))
    for i in [0]: #np.arange(len(model_datasets)):
        print(model_names[i])
             
        ds = xr.DataArray(model_datasets[i].values,
        coords={'time': model_datasets[i].times,
                'lat': model_datasets[i].lats, 
                'lon': model_datasets[i].lons},
        dims=["time", "lat", "lon"])
        
        if annual_cycle:
            ds = ds.groupby('time.month').mean()
        
        #for SEA
        ds1= ds.stack(z=("lat", "lon")) 
        y1=350        
        
        if indo:
            ds2 = ds.where(
                    (ds.lat > -11) & (ds.lat < 6.5) & 
                    (ds.lon > 95) & (ds.lon < 141), drop=True)
            ds2= ds2.stack(z=("lat", "lon")) 
           
        #-6.5 7 93 107
        if sumatera:
            ds3 = ds.where(
                    (ds.lat > -6.5) & (ds.lat < 6.5) & 
                    (ds.lon > 94) & (ds.lon < 107), drop=True)
            ds3= ds3.stack(z=("lat", "lon")) 
                  
        n=0
        #run 3 domains 
        #pakai namelist seaL, 5sum obs dan 5indoL_obs beda hasil SHS pada Sumatera
        #beda resolusi?
        for dat in [ds3, ds2, ds1]:     
            
            #ini untuk metrik SEE,SHS
            dm=dat #.T dan X_valid dilakukan di KMeans22
            
            #ini untuk CHI, DHI, GST
            X=dat.T
            #if annual_cycle: 
            #    X = X.rename({'month':'time'})
            
            # Membuang np.NaN (for land only)
            valid_features_index = ~np.isnan(X[:,0])
            X_valid = X[valid_features_index.data,:]
                       
          
            #for PCA, error NA pada monthly data 
            #sementara diisi 0, cara lebih valid ada?
            if not annual_cycle:
                X_valid=X_valid.fillna(0)
                
            '''
            #------------PCA
            #import numpy as np
            from sklearn.decomposition import PCA
            #import matplotlib.pyplot as plt
          
            # Perform PCA
            pca = PCA()
            pca.fit(X_valid)

            # Variance explained by each principal component
            explained_variance_ratio = pca.explained_variance_ratio_

            # Cumulative variance explained
            cumulative_variance = np.cumsum(explained_variance_ratio)

            # Find the number of components that explain ~98% of the cumulative variance
            cut_off_point = np.argmax(cumulative_variance >= 0.90) + 1
            
            print(f"Number of principal components explaining ~90% of cumulative variance: {cut_off_point}")
            
            # Plot explained variance
            plt.plot(range(1, len(explained_variance_ratio) + 1), cumulative_variance, marker='o')
            plt.xlabel('Number of Principal Components')
            plt.ylabel('Cumulative Variance Explained')
            plt.title('Scree Plot')
            plt.show()
            
            # Using Monte-Carlo randomizations
            data_matrix=X_valid
            num_randomizations = 100

            # Perform PCA on the original data
            original_pca = PCA()
            original_pca.fit(data_matrix)

            # Cumulative variance explained by original data
            original_cumulative_variance = np.cumsum(original_pca.explained_variance_ratio_)

            # Perform Monte-Carlo randomization test
            cut_off_points = []

            for _ in range(num_randomizations):
                # Randomly shuffle the data
                shuffled_data_matrix = np.random.permutation(data_matrix)

                # Perform PCA on the shuffled data
                pca_shuffle = PCA()
                pca_shuffle.fit(shuffled_data_matrix)

                # Cumulative variance explained by shuffled data
                cumulative_variance_shuffle = np.cumsum(pca_shuffle.explained_variance_ratio_)

                # Find the number of components that explain ~98% of cumulative variance in shuffled data
                cut_off_point_shuffle = np.argmax(cumulative_variance_shuffle >= 0.98) + 1
                cut_off_points.append(cut_off_point_shuffle)

            # Plot histogram of cut-off points from randomizations
            plt.hist(cut_off_points, bins=np.arange(1, max(cut_off_points) + 2) - 0.5, edgecolor='black')
            plt.xlabel('Number of Principal Components')
            plt.ylabel('Frequency')
            plt.title('Histogram of Cut-off Points from Monte-Carlo Test')
            plt.show()

            # Find the critical value (e.g., 95th percentile) from the distribution of cut-off points
            critical_value = np.percentile(cut_off_points, 95)

            print(f"Critical value from Monte-Carlo test: {critical_value}")

            exit()
                        
            #######
            '''
            
            
            # --- Perform hierarchical clustering
            # hasil bisa untuk estimate n_clusters but subjective
                        
            from scipy.cluster.hierarchy import dendrogram, linkage
            linkage_matrix = linkage(X_valid, method='ward', metric='euclidean')  
            # You can choose a different linkage method
            #linkage_methods = ['single', 'complete', 'average', 'ward','centroid']
            
            
            plt.subplot(1, 3, n+1)
            dendrogram(linkage_matrix, truncate_mode='level', p=5)  # Adjust 'p' based on your preference
            # Draw a horizontal line at distance/2 or more
            if n==2:
                plt.axhline(y=350, color='red', linestyle='--')
                plt.title('Southeast Asia')
                plt.xlabel('Data points')
            if n==1:
                plt.axhline(y=280, color='red', linestyle='--')
                plt.title('Indonesia')
                plt.xlabel('Data points')
            if n==0:
                plt.axhline(y=120, color='red', linestyle='--')
                plt.title('Sumatera')
                plt.xlabel('Data points')
                plt.ylabel('Euclidean distance')
            plt.xticks([])                
            n=n+1
    plt.show()

def kmeans_metrics(obs_dataset, obs_name, model_datasets, model_names, workdir):
    #pakai gap_stat
    from xlearn22.cluster import KMeans as KMeans22
    from sklearn.metrics import silhouette_score
    from kneed import KneeLocator
    from gap_statistic import OptimalK
    print('*------------------')
    
    #choose
    n_clusters=7
    
    annual_cycle=True
     
    #domain
    sea=True
    indo=True
    sumatera=True
    jambi=False
    
    malaysia=False
    
    plt.figure(figsize=(15, 5))
    for i in [0]: #np.arange(len(model_datasets)):
        print(model_names[i])
             
        ds = xr.DataArray(model_datasets[i].values,
        coords={'time': model_datasets[i].times,
                'lat': model_datasets[i].lats, 
                'lon': model_datasets[i].lons},
        dims=["time", "lat", "lon"])
        
        if annual_cycle:
            ds = ds.groupby('time.month').mean()
        
        #for SEA
        ds1= ds.stack(z=("lat", "lon")) 
        y1=350        
        
        if indo:
            ds2 = ds.where(
                    (ds.lat > -11) & (ds.lat < 6.5) & 
                    (ds.lon > 95) & (ds.lon < 141), drop=True)
            ds2= ds2.stack(z=("lat", "lon")) 
           
        #-6.5 7 93 107
        if sumatera:
            ds3 = ds.where(
                    (ds.lat > -6.5) & (ds.lat < 6.5) & 
                    (ds.lon > 94) & (ds.lon < 107), drop=True)
            ds3= ds3.stack(z=("lat", "lon")) 
            
        if jambi:
            ds = ds.where(
                    (ds.lat > -3) & (ds.lat < -0.5) & 
                    (ds.lon > 99) & (ds.lon < 106), drop=True)
                   
        if malaysia:
            ds4 = ds.where(
                    (ds.lat > 0.6) & (ds.lat < 6.6) & 
                    (ds.lon > 99) & (ds.lon < 104.6), drop=True)
            ds4= ds4.stack(z=("lat", "lon")) 
               
        
        n=0
        #run 3 domains 
        #pakai namelist seaL, 5sum obs dan 5indoL_obs beda hasil SHS pada Sumatera
        #beda resolusi?
        for dat in [ds3, ds2, ds1]:     
            n=n+1
            #ini untuk metrik SEE,SHS
            dm=dat #.T dan X_valid dilakukan di KMeans22
            
            #ini untuk CHI, DHI, GST
            X=dat.T
            #if annual_cycle: 
            #    X = X.rename({'month':'time'})
            
            # Membuang np.NaN (for land only)
            valid_features_index = ~np.isnan(X[:,0])
            X_valid = X[valid_features_index.data,:]
                       
          
            #for PCA, error NA pada monthly data 
            #sementara diisi 0, cara lebih valid ada?
            if not annual_cycle:
                X_valid=X_valid.fillna(0)
                
            '''
            #------------PCA
            #import numpy as np
            from sklearn.decomposition import PCA
            #import matplotlib.pyplot as plt
          
            # Perform PCA
            pca = PCA()
            pca.fit(X_valid)

            # Variance explained by each principal component
            explained_variance_ratio = pca.explained_variance_ratio_

            # Cumulative variance explained
            cumulative_variance = np.cumsum(explained_variance_ratio)

            # Find the number of components that explain ~98% of the cumulative variance
            cut_off_point = np.argmax(cumulative_variance >= 0.90) + 1
            
            print(f"Number of principal components explaining ~90% of cumulative variance: {cut_off_point}")
            
            # Plot explained variance
            plt.plot(range(1, len(explained_variance_ratio) + 1), cumulative_variance, marker='o')
            plt.xlabel('Number of Principal Components')
            plt.ylabel('Cumulative Variance Explained')
            plt.title('Scree Plot')
            plt.show()
            
            # Using Monte-Carlo randomizations
            data_matrix=X_valid
            num_randomizations = 100

            # Perform PCA on the original data
            original_pca = PCA()
            original_pca.fit(data_matrix)

            # Cumulative variance explained by original data
            original_cumulative_variance = np.cumsum(original_pca.explained_variance_ratio_)

            # Perform Monte-Carlo randomization test
            cut_off_points = []

            for _ in range(num_randomizations):
                # Randomly shuffle the data
                shuffled_data_matrix = np.random.permutation(data_matrix)

                # Perform PCA on the shuffled data
                pca_shuffle = PCA()
                pca_shuffle.fit(shuffled_data_matrix)

                # Cumulative variance explained by shuffled data
                cumulative_variance_shuffle = np.cumsum(pca_shuffle.explained_variance_ratio_)

                # Find the number of components that explain ~98% of cumulative variance in shuffled data
                cut_off_point_shuffle = np.argmax(cumulative_variance_shuffle >= 0.98) + 1
                cut_off_points.append(cut_off_point_shuffle)

            # Plot histogram of cut-off points from randomizations
            plt.hist(cut_off_points, bins=np.arange(1, max(cut_off_points) + 2) - 0.5, edgecolor='black')
            plt.xlabel('Number of Principal Components')
            plt.ylabel('Frequency')
            plt.title('Histogram of Cut-off Points from Monte-Carlo Test')
            plt.show()

            # Find the critical value (e.g., 95th percentile) from the distribution of cut-off points
            critical_value = np.percentile(cut_off_points, 95)

            print(f"Critical value from Monte-Carlo test: {critical_value}")

            exit()
                        
            #######
            '''
   
            metrik=[]       
            sse = []
            silhouette_coef = []
            #fit2 dicoba kini fit3
            for k in range(1, n_clusters):
                print('k=',k)
                kmeans, features = KMeans22(n_clusters=k,  random_state=42).\
                                    fit3(dm, annual_cycle, with_pca=False)
                sse.append(kmeans.inertia_) 
               
                if k>1:
                    label2=kmeans.cluster_centers_da.sel(cluster=0).stack(z=("lat", "lon"))
                    label2=label2.dropna(dim=('z'))
                    #print('label2', label2)
                    score = silhouette_score(features, label2)
                else:
                    score=0
                silhouette_coef.append(score)
            
            kl = KneeLocator(
                range(1, n_clusters), sse, curve="convex", direction="decreasing"
                )
            c=kl.elbow
            print('Optimum cluster SSE=',c)
            #1
            metrik.append(c)
            
            #----- ini??
            '''
            kl = KneeLocator(
                range(1, 7), silhouette_coef, curve="convex", direction="decreasing"
                )
            c2=kl.elbow
            '''
            
            s=np.array(silhouette_coef)
            c2=np.where(s==s.max())[0][0]+1
            
            print('Optimum cluster SHS=',c2)
            #2
            metrik.append(c2)
            
            #--- Davies-Bouldin index
            # hasil tidak jelas, selalu 2
            X=X_valid
            from sklearn.cluster import KMeans
            from sklearn.metrics import pairwise_distances_argmin_min
            from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score

            #--- calinski_harabasz_score
            def find_optimal_clusters(X, max_clusters=n_clusters):
               
                #Find the optimal number of clusters that maximizes the Calinski-Harabasz index.

                calinski_harabasz_scores = []

                for k in range(2, max_clusters + 1):
                    kmeans = KMeans(n_clusters=k, random_state=42)
                    labels = kmeans.fit_predict(X)
                    ch_score = calinski_harabasz_score(X, labels)
                    calinski_harabasz_scores.append(ch_score)

                # Find the number of clusters with the maximum Calinski-Harabasz index
                optimal_clusters = np.argmax(calinski_harabasz_scores) + 2  # Add 2 because we started from k=2
                return optimal_clusters
                
            # Call
            optimal_clusters = find_optimal_clusters(X)
            print(f"Optimal number of clusters 2: {optimal_clusters}")
            #3
            metrik.append(optimal_clusters)
            
            #DBI
            def davies_bouldin_index(X, labels):
                
                #Calculate the Davies-Bouldin index for a given clustering.

                k = len(np.unique(labels))
                #print('len(np.unique(labels))=', k)
                cluster_centers = [np.mean(X[labels == i], axis=0) for i in range(k)]

                # Calculate pairwise distances between cluster centers
                center_distances = pairwise_distances_argmin_min(cluster_centers, cluster_centers)

                # Calculate cluster-wise Davies-Bouldin index
                db_index = 0.0
                for i in range(k):
                    max_ratio = 0.1
                    for j in range(k):
                        if i != j:
                            ratio = (np.sum(pairwise_distances_argmin_min(X[labels == i], X[labels == j])[1]) +
                                     np.sum(pairwise_distances_argmin_min(X[labels == j], X[labels == i])[1])) \
                                     / center_distances[1][i]
                            if ratio > max_ratio:
                                max_ratio = ratio
                    db_index += max_ratio

                return db_index / k
                
            #ini aneh jika max_clusters>9 ..hasil selalu 10 
            #apakah hanya terbatas pada <10 ???
            #tapi notebook gk masalah
            def find_optimal_clusters(X, max_clusters=n_clusters):
              
                #Find the optimal number of clusters that minimizes the Davies-Bouldin index.
                
                davies_bouldin_scores = []

                for k in range(2, max_clusters + 1):
                    print(k)
                    kmeans = KMeans(n_clusters=k, random_state=42)
                    labels = kmeans.fit_predict(X)
                    #db_score = davies_bouldin_index(X, labels)
                    db_score = davies_bouldin_score(X, labels)
                    davies_bouldin_scores.append(db_score)

                # Find the number of clusters with the minimum Davies-Bouldin index
                optimal_clusters = np.argmin(davies_bouldin_scores) + 2  
                # Add 2 because we started from k=2
                return optimal_clusters

            # Call
            optimal_clusters = find_optimal_clusters(X)
            print(f"Optimal number of clusters 1: {optimal_clusters}")
            #4
            metrik.append(optimal_clusters)
            
            #GAP
            #----- gap_stat tidak konsisten tiap running beda hasil
            # Determine optimal number of clusters using Gap Statistics
            
            optimal_k = OptimalK(parallel_backend='rust')
            #optimal_k = OptimalK(parallel_backend='joblib') #ini hasilnya aneh
            #optimal_k = OptimalK() #ini hasilnya aneh
            optimal_k_clusters = optimal_k(X_valid, cluster_array=np.arange(1, n_clusters), n_refs=10)
            
            self=optimal_k
            
            '''
            # Print the optimal number of clusters
            print("Optimal number of clusters:", optimal_k_clusters)

            # Plot Gap Statistics
            #optimal_k.plot_results()

            # Gap values plot
            
            ax[0,i].plot(self.gap_df.n_clusters, self.gap_df.gap_value, linewidth=3)
            ax[0,i].scatter(
            self.gap_df[self.gap_df.n_clusters == self.n_clusters].n_clusters,
            self.gap_df[self.gap_df.n_clusters == self.n_clusters].gap_value,
            s=250,
            c="r",
            )
            
            
            # Gap* plot
            max_ix = self.gap_df[self.gap_df["gap*"] == self.gap_df["gap*"].max()].index[0]
            ax[0,i+1].plot(self.gap_df.n_clusters, self.gap_df["gap*"], linewidth=3)
            ax[0,i+1].scatter(
                self.gap_df.loc[max_ix]["n_clusters"],
                self.gap_df.loc[max_ix]["gap*"],
                s=250,
                c="r",
            )
            plt.show()
            '''
            max_ix = self.gap_df[self.gap_df["gap*"] == self.gap_df["gap*"].max()].index[0]
            print("Optimal number of clusters:", self.gap_df.loc[max_ix]["n_clusters"])
            #5
            metrik.append(self.gap_df.loc[max_ix]["n_clusters"])
            
            print('metrik=', metrik)
      
        #plt.show()
        
def kmeans_metrics2(obs_dataset, obs_name, model_datasets, model_names, workdir):
    #1 region many metric many dataset
    
    
    from xlearn22.cluster import KMeans as KMeans22
    from sklearn.metrics import silhouette_score
    from kneed import KneeLocator
    from gap_statistic import OptimalK
    print('*------------------')
    
    #choose
    n_clusters=7 #variasi jumlah klaster pada sumbu x
    
    annual_cycle=True
     
    #domain
    sea=True #khusus datainput SEA
    indo=False
    sumatera=False
    jambi=False
    
    malaysia=False
    
    fig,ax=plt.subplots(figsize=[8,6])
    plt.subplots_adjust(bottom=.2)
    mm=['','>','s','x','o','+', '*']
    
    metrik2=[]            
    for i in [0]: 
    #for i in np.arange(len(model_datasets)):
        print(model_names[i])
             
        ds = xr.DataArray(model_datasets[i].values,
        coords={'time': model_datasets[i].times,
                'lat': model_datasets[i].lats, 
                'lon': model_datasets[i].lons},
        dims=["time", "lat", "lon"])
        
        if annual_cycle:
            ds = ds.groupby('time.month').mean()
        #----------------------------------------------------------
        if sea:
            dss= ds.stack(z=("lat", "lon")) 
            dd=ds
        if indo:
            dd = ds.where(
                    (ds.lat > -11) & (ds.lat < 6.5) & 
                    (ds.lon > 95) & (ds.lon < 141), drop=True)
            dss= dd.stack(z=("lat", "lon")) 
           
        #-6.5 7 93 107
        if sumatera:
            dd = ds.where(
                    (ds.lat > -6.5) & (ds.lat < 6.5) & 
                    (ds.lon > 94) & (ds.lon < 107), drop=True)
            dss= dd.stack(z=("lat", "lon")) 
            
        if jambi:
            ds = ds.where(
                    (ds.lat > -3) & (ds.lat < -0.5) & 
                    (ds.lon > 99) & (ds.lon < 106), drop=True)
                   
        if malaysia:
            dd = ds.where(
                    (ds.lat > 0.6) & (ds.lat < 6.6) & 
                    (ds.lon > 99) & (ds.lon < 104.6), drop=True)
            dss= dd.stack(z=("lat", "lon")) 
        
        #cek plot map
        dd.mean(dim='month').plot()
        #plt.show()
            
        #----------------------------------------------------------
        for dat in [dss]:     
        
            #ini untuk metrik SEE,SHS
            dm=dat #.T dan X_valid dilakukan di KMeans22
            
            #ini untuk CHI, DHI, GST
            X=dat.T
            #if annual_cycle: 
            #    X = X.rename({'month':'time'})
            
            # Membuang np.NaN (for land only)
            valid_features_index = ~np.isnan(X[:,0])
            X_valid = X[valid_features_index.data,:]
                       
          
            #for PCA, error NA pada monthly data 
            #sementara diisi 0, cara lebih valid ada?
            if not annual_cycle:
                X_valid=X_valid.fillna(0)
                

            metrik=[]
            sse = []
            silhouette_coef = []
            #fit2 dicoba kini fit3
            for k in range(1, n_clusters):
                print('k=',k)
                kmeans, features = KMeans22(n_clusters=k,  random_state=42).\
                                    fit3(dm, annual_cycle, with_pca=False)
                sse.append(kmeans.inertia_) 
               
                if k>1:
                    label2=kmeans.cluster_centers_da.sel(cluster=0).stack(z=("lat", "lon"))
                    label2=label2.dropna(dim=('z'))
                    #print('label2', label2)
                    score = silhouette_score(features, label2)
                else:
                    score=0
                silhouette_coef.append(score)
            
            kl = KneeLocator(
                range(1, n_clusters), sse, curve="convex", direction="decreasing"
                )
            c=kl.elbow
            print('Optimum cluster SSE=',c)
            #1
            metrik.append(c)
          
            s=np.array(silhouette_coef)
            c2=np.where(s==s.max())[0][0]+1
            
            print('Optimum cluster SHS=',c2)
            #2
            metrik.append(c2)
            
            #--- Davies-Bouldin index
            # hasil tidak jelas, selalu 2
            X=X_valid
            from sklearn.cluster import KMeans
            from sklearn.metrics import pairwise_distances_argmin_min
            from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score

            #--- calinski_harabasz_score
            def find_optimal_clusters(X, max_clusters=n_clusters):
               
                #Find the optimal number of clusters that maximizes the Calinski-Harabasz index.

                calinski_harabasz_scores = []

                for k in range(2, max_clusters + 1):
                    kmeans = KMeans(n_clusters=k, random_state=42)
                    labels = kmeans.fit_predict(X)
                    ch_score = calinski_harabasz_score(X, labels)
                    calinski_harabasz_scores.append(ch_score)

                # Find the number of clusters with the maximum Calinski-Harabasz index
                optimal_clusters = np.argmax(calinski_harabasz_scores) + 2  # Add 2 because we started from k=2
                return optimal_clusters
                
            # Call
            optimal_clusters = find_optimal_clusters(X)
            print(f"Optimal number of clusters 2: {optimal_clusters}")
            #3
            metrik.append(optimal_clusters)
            
            #DBI
            def davies_bouldin_index(X, labels):
                
                #Calculate the Davies-Bouldin index for a given clustering.

                k = len(np.unique(labels))
                #print('len(np.unique(labels))=', k)
                cluster_centers = [np.mean(X[labels == i], axis=0) for i in range(k)]

                # Calculate pairwise distances between cluster centers
                center_distances = pairwise_distances_argmin_min(cluster_centers, cluster_centers)

                # Calculate cluster-wise Davies-Bouldin index
                db_index = 0.0
                for i in range(k):
                    max_ratio = 0.1
                    for j in range(k):
                        if i != j:
                            ratio = (np.sum(pairwise_distances_argmin_min(X[labels == i], X[labels == j])[1]) +
                                     np.sum(pairwise_distances_argmin_min(X[labels == j], X[labels == i])[1])) \
                                     / center_distances[1][i]
                            if ratio > max_ratio:
                                max_ratio = ratio
                    db_index += max_ratio

                return db_index / k
                
            #ini aneh jika max_clusters>9 ..hasil selalu 10 
            #apakah hanya terbatas pada <10 ???
            #tapi notebook gk masalah
            def find_optimal_clusters(X, max_clusters=n_clusters):
              
                #Find the optimal number of clusters that minimizes the Davies-Bouldin index.
                
                davies_bouldin_scores = []

                for k in range(2, max_clusters + 1):
                    print(k)
                    kmeans = KMeans(n_clusters=k, random_state=42)
                    labels = kmeans.fit_predict(X)
                    #db_score = davies_bouldin_index(X, labels)
                    db_score = davies_bouldin_score(X, labels)
                    davies_bouldin_scores.append(db_score)

                # Find the number of clusters with the minimum Davies-Bouldin index
                optimal_clusters = np.argmin(davies_bouldin_scores) + 2  
                # Add 2 because we started from k=2
                return optimal_clusters

            # Call
            optimal_clusters = find_optimal_clusters(X)
            print(f"Optimal number of clusters 1: {optimal_clusters}")
            #4
            metrik.append(optimal_clusters)
            
            #GAP
            #----- gap_stat tidak konsisten tiap running beda hasil
            # Determine optimal number of clusters using Gap Statistics
            
            optimal_k = OptimalK(parallel_backend='rust')
            #optimal_k = OptimalK(parallel_backend='joblib') #ini hasilnya aneh
            #optimal_k = OptimalK() #ini hasilnya aneh
            optimal_k_clusters = optimal_k(X_valid, cluster_array=np.arange(1, n_clusters), n_refs=10)
            
            self=optimal_k
          
            max_ix = self.gap_df[self.gap_df["gap*"] == self.gap_df["gap*"].max()].index[0]
            print("Optimal number of clusters:", self.gap_df.loc[max_ix]["n_clusters"])
            #5
            metrik.append(self.gap_df.loc[max_ix]["n_clusters"])
        
        print(model_names[i])
        print('metrik SEE, SHS, CHI,DHI, GST=', metrik)
        #plot
        metrik_name = ['SEE', 'SHS', 'CHI','DHI', 'GST']
        ax.plot(metrik_name,metrik, label =model_names[i], marker=mm[i], color='black')
        plt.ylabel('Optimal number of clusters')
        plt.legend(loc='best', prop={'size':8.5}, frameon=False) 
    
        #bar
        
        metrik2.append(metrik)  
    import pandas as pd
    print('metrik2=',metrik2)
    c = pd.DataFrame(metrik2,model_names,
            columns=metrik_name)
        
    ax=c.plot(kind='bar')
    xlabel=metrik_name
   
    ax.set_xticklabels(model_names,rotation=0)
    #out
    #plt.legend(loc='center left', bbox_to_anchor=(1, 0.15))
    #inside
    #plt.legend(loc='center left', bbox_to_anchor=(.15, 0.85))
    plt.legend(loc='best')
    plt.ylabel('Optimal clusters number')
    plt.subplots_adjust(right=.7)
    
    
    
    plt.show()
    
def kmeans_dcm(obs_dataset, obs_name, model_datasets, model_names, workdir):
   
    annual_cycle=True
   
    #domain
    sea=False
    indo=True
    sumatera=False
    
    fig, ax = plt.subplots(2,1, figsize=(6,10)) 
    
    for i in [0]:
        print(model_names[i])
     
        ds = xr.DataArray(model_datasets[i].values,
        coords={'time': model_datasets[i].times,
                'lat': model_datasets[i].lats, 
                'lon': model_datasets[i].lons},
        dims=["time", "lat", "lon"])
        
        #-14.5, 25.5, 90.5, 145
        if sea: 
            ds = ds.where(
                    (ds.lat > -14.5) & (ds.lat < 25.5) & 
                    (ds.lon > 90.5) & (ds.lon < 145), drop=True)
        
        if indo:
            ds = ds.where(
                    (ds.lat > -11) & (ds.lat < 6.5) & 
                    (ds.lon > 95) & (ds.lon < 141), drop=True)
        #-6.5 7 93 107
        
        if sumatera:
            ds = ds.where(
                    (ds.lat > -6.5) & (ds.lat < 6.5) & 
                    (ds.lon > 94) & (ds.lon < 107), drop=True)
                           
        lat_min = ds.lat.min()
        lat_max = ds.lat.max()
        lon_min = ds.lon.min()
        lon_max = ds.lon.max()
        x,y = np.meshgrid(ds.lon, ds.lat)
        
        m = Basemap(ax=ax[0], projection ='cyl', 
                llcrnrlat = lat_min, #+1*0.22, 
                urcrnrlat = lat_max, #-6*0.22,
                llcrnrlon = lon_min, #+4*0.22, 
                urcrnrlon = lon_max, #-3*0.22, 
                resolution = 'l', fix_aspect=False)
        m.drawcoastlines(linewidth=1)
        m.drawcountries(linewidth=1)
        m.drawstates(linewidth=0.5, color='w')
              
        if annual_cycle:
            dm = ds.groupby('time.month').mean()
           
            fn='annual_cycle'
        else:
            fn='raw_monthly'
            dm=ds
        '''   
        if model_names[i]=='TRMM':
            rainfall_data=dm*100000
        else:
            rainfall_data=dm*1000
        '''
        rainfall_data=dm
        num_lat, num_lon =len(ds.lat), len(ds.lon)
        latitudes=ds.lat
        longitudes=ds.lon
        -3.10, 129.20
        # Step 2: Select a reference cell
        #set  lampung, jambi, Ceram, laut maluku, brunei, NH
        #rlat=[-5.125,  -0.125,   -3.10, ] #-4.14, 5.5,  ] # 19.0] # 4.375]
        #rlon=[105.125, 102.125, 129.20, ] #127.26, 116.5,] # 100.0] # 97.375 ]
        
        #set  lampung, jambi, laut maluku, brunei, NH
        rlat=[-5.125,  -0.125,   -4.14, ] # 5.5,  ] # 19.0] # 4.375]
        rlon=[105.125, 102.125, 127.26 ] #127.26, 116.5,] # 100.0] # 97.375 ]
        
        # Find the nearest latitude and longitude points
        rlat = ds['lat'].sel(lat=rlat, method='nearest')
        rlon = ds['lon'].sel(lon=rlon, method='nearest')


        # Step 3: Correlate all other cells to the reference cell
       
        threshold_pos = 0.8
        mask_pos1=np.zeros((num_lat, num_lon))
        mask_pos2=np.zeros((num_lat, num_lon))
        mask_pos3=np.zeros((num_lat, num_lon))
        mask_pos4=np.zeros((num_lat, num_lon))
        mask_pos5=np.zeros((num_lat, num_lon))
            
        n=0
        #corr 1st
        for (lat,lon) in zip(rlat,rlon):
            n=n+1
            print('n=',n)
            correlation_matrix = np.zeros((num_lat, num_lon))
            
            for i in range(num_lat):
                for j in range(num_lon):
                    if (latitudes[i] != lat) or (longitudes[j] != lon):
                        # Calculate correlation with reference cell
                        correlation_matrix[i, j] = np.corrcoef(rainfall_data[:, i, j], rainfall_data.sel(lat=lat, lon=lon))[0, 1]
        
            # Create masks for each threshold
            if n==1:
                mask_pos1 = np.where(correlation_matrix > threshold_pos, 1, 0)   # Label as 1 for values above threshold_pos
                      
            if n==2:
                mask_pos2 = np.where(correlation_matrix > threshold_pos, 2, 0)   # Label as 1 for values above threshold_pos
                '''
                rainfall_data.coords['mask2'] = (('lat', 'lon'), mask_pos2)
                #corr 2nd
                #korelasi all cells dengan rata-rata region 1 
                dd=rainfall_data.where(rainfall_data.mask2==2).mean(dim=('lat', 'lon'))

                mask_pos2=np.zeros((num_lat, num_lon))
                correlation_matrix = np.zeros((num_lat, num_lon))
                for i in range(num_lat):
                    for j in range(num_lon):
                        if (latitudes[i] != lat) or (longitudes[j] != lon):
                            # Calculate correlation with reference cell
                            correlation_matrix[i, j] = np.corrcoef(rainfall_data[:, i, j], dd)[0, 1]

                mask_pos2 = np.where(correlation_matrix > threshold_pos, 2, 0)   
                '''
            if n==3:
                mask_pos3 = np.where(correlation_matrix > threshold_pos, 3, 0)   # Label as 1 for valu
                # Combine masks
            if n==4:
                mask_pos4 = np.where(correlation_matrix > threshold_pos, 4, 0) 
            if n==5:
                mask_pos5 = np.where(correlation_matrix > threshold_pos, 5, 0) 
        
    region_label = mask_pos1 + mask_pos2 + mask_pos3 + mask_pos4 + mask_pos5
    rainfall_data.coords['mask'] = (('lat', 'lon'), region_label)
            
    nn=len(rlat)      
    ax[0].axhline(y=0)
    c=['white','blue','pink', 'yellow', 'green', 'darkviolet']
    rainfall_data.mask.plot(ax=ax[0],levels=np.arange(nn+1), colors=c[0:nn+1], add_colorbar = False)
    c=['blue','pink', 'yellow', 'green', 'darkviolet']
    for n in range (1,nn+1):
        
        rainfall_data.where(rainfall_data.mask==n).mean(dim=('lat', 'lon')).plot(ax=ax[1], color=c[n-1],lw=2)

    ax[0].scatter(rlon, rlat, color='red', s=30)
 
    
    if sea:
        ax[0].set_xticks([100,120,140])
        ax[0].set_xticklabels(['100E','120E','140E'])
        ax[0].tick_params(axis='x', pad=1,labelsize=8)
        ax[0].set_yticks([-10,0,10,20])
        ax[0].set_yticklabels(['10S','0','10N','20N'])
        ax[0].tick_params(axis='y', pad=1,labelsize=8)
    
    if indo:
        ax[0].set_xticks([105,120,135])
        ax[0].set_xticklabels(['105E','120E','135E'])
        ax[0].tick_params(axis='x', pad=1,labelsize=8)
        ax[0].set_yticks([-11,-5,0,5])
        ax[0].set_yticklabels(['10S','5S','0','5N'])
        ax[0].tick_params(axis='y', pad=1,labelsize=8)
        
    if sumatera:
        ax[0].set_xticks([95,100,105])
        ax[0].set_xticklabels(['95E','100E','105E'])
        ax[0].tick_params(axis='x', pad=1,labelsize=8)
        ax[0].set_yticks([-5,0,5])
        ax[0].set_yticklabels(['5S','0','5N'])
        ax[0].tick_params(axis='y', pad=1,labelsize=8)
    
    ax[0].set_ylabel('Latitude')
    ax[0].set_xlabel('')
    ax[1].set_ylabel('Mean rainfall (mm/month)')
    ax[1].set_xlabel('Month')
    ax[1].set_xticks(np.arange(12)+1)
    x_tick=['J','F','M','A','M','J','J','A','S','O','N','D']
    ax[1].set_xticklabels(x_tick, fontsize=8)
    plt.show()
    
def kmeans_sea22(obs_dataset, obs_name, model_datasets, model_names, workdir):
    #cluster > 5 no SSE, for plot SSE diatas
    from xlearn22.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from kneed import KneeLocator
    #from sklearn.preprocessing import StandardScaler
    #import geoxarray
    print('*------------------')
    
    #choose
    n_clusters=4
    
    annual_cycle=True
    with_pca=False#default diset n=3 ubah di cluster.py fit2 
    
    sse=False #khusus malaysia error di ...2336
    gap_stat=False
    
    #domain
    sea=False
    indo=True
    sumatera=False
    jambi=False
    malaysia=False
    maluku=False
    
    fig, ax = plt.subplots(3,len(model_datasets), figsize=(8,8))
    
    for i in np.arange(len(model_datasets)):
    #for i in [0]:
        print(model_names[i])
     
        ds = xr.DataArray(model_datasets[i].values,
        coords={'time': model_datasets[i].times,
                'lat': model_datasets[i].lats, 
                'lon': model_datasets[i].lons},
        dims=["time", "lat", "lon"])
        
        if sea: #-14.5, 25.5, 90.5, 145
            #ds = ds.where(
            #        (ds.lat > -14.5) & (ds.lat < 25.5) & 
            #        (ds.lon > 90.5) & (ds.lon < 145), drop=True)
            
            ds = ds.where(
                    (ds.lat > -30) & (ds.lat < 30) & 
                    (ds.lon > 40) & (ds.lon < 240), drop=True)

        
        if indo:
            ds = ds.where(
                    (ds.lat > -11) & (ds.lat < 6.5) & 
                    (ds.lon > 95) & (ds.lon < 150), drop=True)
        #-6.5 7 93 107
        
        if sumatera:
            ds = ds.where(
                    (ds.lat > -6.5) & (ds.lat < 6.5) & 
                    (ds.lon > 94) & (ds.lon < 107), drop=True)
                
        if jambi:
            ds = ds.where(
                    (ds.lat > -3) & (ds.lat < -0.5) & 
                    (ds.lon > 99) & (ds.lon < 106), drop=True)
                    
        if malaysia:
            ds = ds.where(
                    (ds.lat > 1) & (ds.lat < 7) & 
                    (ds.lon > 100) & (ds.lon < 104.5), drop=True)
        
        if maluku:
            ds = ds.where(
                    (ds.lat > -5) & (ds.lat < 5) & 
                    (ds.lon > 125) & (ds.lon < 130), drop=True)
                   
        lat_min = ds.lat.min()
        lat_max = ds.lat.max()
        lon_min = ds.lon.min()
        lon_max = ds.lon.max()
        x,y = np.meshgrid(ds.lon, ds.lat)
        
        m = Basemap(ax=ax[1,i], projection ='cyl', 
                llcrnrlat = lat_min, #+1*0.22, 
                urcrnrlat = lat_max, #-6*0.22,
                llcrnrlon = lon_min, #+4*0.22, 
                urcrnrlon = lon_max, #-3*0.22, 
                resolution = 'l', fix_aspect=False)
        m.drawcoastlines(linewidth=1)
        m.drawcountries(linewidth=1)
        m.drawstates(linewidth=0.5, color='w')
        #m.etopo()
        '''
        # Define the months for pre-monsoon (replace with your actual months)
        pre_monsoon_months = [3, 4, 5]  # March, April, May

        # Select the pre-monsoon data
        ds = ds.sel(time=ds['time.month'].isin(pre_monsoon_months))

        # Compute the mean over the selected months
        #pre_monsoon_data = pre_monsoon_data.mean(dim='time')
        '''
        
        
        #3D ubah ke 2D (time, z(x,y))
        ds= ds.stack(z=("lat", "lon")) #s= s.unstack()
        
        if annual_cycle:
            dm = ds.groupby('time.month').mean()
           
            fn='annual_cycle'
        else:
            
            fn='raw_monthly'
            dm=ds
       
        km = KMeans(n_clusters=n_clusters, random_state=0).fit2(dm, annual_cycle, with_pca)
                   
       
        p=km.cluster_centers_da.sel(cluster=0).plot(ax=ax[1,i], 
          levels=np.arange(n_clusters+1),
          #cmap='rainbow',
          #colors=['blue','red','pink','green', 'orange'],
          #colors = ['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet', 'darkred', 'darkorange', 'darkgreen'],
          colors=['blue','red','pink','green', 'orange', 'indigo', 'violet', 'darkred', 'darkorange', 'darkgreen'],
          add_colorbar = False)
        
        if not sse:
            ax[1,i].set_title(model_names[i], fontsize=9)
            
            ax[0,i].set_xticks([])
            
        if sse:
            ax[0,i].set_title(model_names[i], fontsize=9)
            ax[1,i].set_title('', fontsize=9)
        
        if sea:
            ax[1,i].set_xticks([100,120,140])
            ax[1,i].set_xticklabels(['100E','120E','140E'])
            ax[1,i].tick_params(axis='x', pad=1,labelsize=8)
            ax[1,i].set_xlabel('')
            ax[1,2].set_ylabel('')
            ax[1,0].set_yticks([-10,0,10,20])
            ax[1,0].set_yticklabels(['10S','0','10N','20N'])
            ax[1,0].tick_params(axis='y', pad=1,labelsize=8)
        
        if indo:
            ax[1,i].set_xticks([105,120,135])
            ax[1,i].set_xticklabels(['105E','120E','135E'])
            ax[1,i].tick_params(axis='x', pad=1,labelsize=8)
            ax[1,i].set_xlabel('')
            if i>0:
                ax[1,i].set_ylabel('')
            ax[1,0].set_yticks([-11,-5,0,5])
            ax[1,0].set_yticklabels(['10S','5S','0','5N'])
            ax[1,0].tick_params(axis='y', pad=1,labelsize=8)
            
        if sumatera:
            ax[1,i].set_xticks([95,100,105])
            ax[1,i].set_xticklabels(['95E','100E','105E'])
            ax[1,i].tick_params(axis='x', pad=1,labelsize=8)
            ax[1,i].set_xlabel('')
            ax[1,i].set_ylabel('')
            ax[1,0].set_yticks([-5,0,5])
            ax[1,0].set_yticklabels(['5S','0','5N'])
            ax[1,0].tick_params(axis='y', pad=1,labelsize=8)
            
        if jambi:
            ax[0,i].set_xticks([100,104])
            ax[0,i].set_xticklabels(['100$^\circ$E','104$^\circ$E'])
            ax[0,i].tick_params(axis='x', pad=1,labelsize=8)
            ax[0,i].set_xlabel('')
            ax[0,i].set_ylabel('')
            ax[0,0].set_yticks([-3,-2,-1])
            ax[0,0].set_yticklabels(['3$^\circ$S','2$^\circ$S','1$^\circ$S'])
            ax[0,0].tick_params(axis='y', pad=1,labelsize=8)
        
        if malaysia:
            ax[1,i].set_xticks([101,103])
            ax[1,i].set_xticklabels(['101$^\circ$E','103$^\circ$E'])
            ax[1,i].tick_params(axis='x', pad=1,labelsize=8)
            ax[1,i].set_xlabel('')
            ax[1,i].set_ylabel('')
            ax[1,0].set_yticks([2,4,6])
            ax[1,0].set_yticklabels(['2$^\circ$N','4$^\circ$N','6$^\circ$N'])
            ax[1,0].tick_params(axis='y', pad=1,labelsize=8)
       
        #SEA 4 color
        z=[20,16,12,8]
        co=['blue','red','pink','green', 'orange']
        '''
        #tampilkan fraksi cluster
        for ii in range (m.n_clusters):
            frac = '{:4.1f}%'.format(KMeans.get_cluster_fraction(m, label=ii)*100)
            #print('title1', title1)
            
            f=str(co[ii])+'='+str(frac)
           
            ax[0,i].text(lon_max-17,z[ii],f,color='black')
        '''
        
        if i<len(model_datasets)-1:
            ax[1,i+1].set_yticks([])
            ax[1,i+1].set_yticklabels([])
            ax[1,i+1].set_ylabel('')
        
        x_tick=['J','F','M','A','M','J','J','A','S','O','N','D']
        x=np.arange(km.cluster_centers_.shape[1])
        
        '''
        #for ori resolution and ocean included
        if i==0:
            km.cluster_centers_=km.cluster_centers_*1000
        if i==1:
            km.cluster_centers_=km.cluster_centers_*1000
        '''
        
        #c=['blue','red', 'pink','green', 'orange']
        c= ['blue','red','pink','green', 'orange', 'indigo', 'violet', 'darkred', 'darkorange', 'darkgreen']
      
        lw=['2','2','2','2','2']
        for ii in range (km.n_clusters):
            
            ax[2,i].plot(x, km.cluster_centers_[ii], 
            color=c[ii],
            lw=2
            ) 
     
        ax[2,i].set_xticks(x)
        if annual_cycle and not with_pca:
            ax[2,i].set_xticklabels(x_tick, fontsize=8)
        #ax[0,i].text(3,f[i],model_names[i])
        ax[2,0].set_ylabel('Mean pr (mm/month)')
        ax[2,i].tick_params(axis='y', pad=1,labelsize=8)
        #plt.legend(bbox_to_anchor=(1, .4), loc='best', prop={'size':8.5}, frameon=False) 
        
        #untuk model<2
        ax[1,1].set_ylabel('')
        
        if sse:
            sse = []
            silhouette_coef = []
            #fit2 dicoba kini fit3
            for k in range(1, 7):
                print('k=',k)
                kmeans, features = KMeans(n_clusters=k,  random_state=42).fit3(dm, annual_cycle, with_pca)
                sse.append(kmeans.inertia_) 
                #print('kmeans.labels.shape', kmeans.labels_.shape)
                #print('features.shape', features.shape)
                if k>1:
                    label2=kmeans.cluster_centers_da.sel(cluster=0).stack(z=("lat", "lon"))
                    label2=label2.dropna(dim=('z'))
                    #print('label2', label2)
                    score = silhouette_score(features, label2)
                else:
                    score=0
                silhouette_coef.append(score)
            
            kl = KneeLocator(
                range(1, 7), sse, curve="convex", direction="decreasing"
                )
            c=kl.elbow
            print('Optimum cluster1=',c)
            
            #---ini ??
            '''
            kl = KneeLocator(
                range(1, 7), silhouette_coef, curve="convex", direction="decreasing"
                )
            c2=kl.elbow
            
            print('Optimum cluster2=',c2)
            '''
            print('sse[c-1]',sse)  
            
            
            #plt.style.use("fivethirtyeight")
            ax[0,i].plot(range(1, 7), sse, color='black')
            ax[0,i].axvline(x=c,ymin=0, ymax=sse[c-1]/100000, ls=':', color='black') 
            ax[0,0].ticklabel_format(axis='y', style='sci', scilimits=(0,0))
            ax[0,i].set_xticks(range(1, 7))
            #ax[0,0].set_xlabel("N optimum of Clusters", fontsize=7)
            #ax[2,i].set_yticks([])
            #ax[2,i].set_yticklabels([])
            ax[0,0].set_ylabel("SSE")
            if i<len(model_datasets)-1:
                ax[0,i+1].set_yticks([])
                ax[0,i+1].set_yticklabels([])
            
            s=np.array(silhouette_coef)
            c2=np.where(s==s.max())[0][0] # [0][0] first only
            print('Optimum cluster3=',c2)
            
            ax2 = ax[0,i].twinx()  # instantiate a second axes that shares the same x-axis

            color = 'tab:red'
            
            ax2.plot(range(1, 7), silhouette_coef, color=color)
            ax2.tick_params(axis='y', labelcolor=color)
            ax2.axvline(x=c2+1, ls=':', color='red') 
            if i<len(model_datasets)-1:
                ax2.set_yticks([])
                ax2.set_yticklabels([])        
            if i==len(model_datasets)-1:
                ax2.set_ylabel('silhouette_coef', color=color)
               
            if not annual_cycle and not with_pca:
                ax[2,i].set_xticks([])
                ax[2,i].set_xticklabels([])
                ax[2,i].set_xlabel('Monthly time scale')
            
            
    plt.subplots_adjust(hspace=.3,wspace=.17)
    
    cax = fig.add_axes([0.91, 0.1, 0.02, 0.5])
   
    cbar = plt.colorbar(p,cax=cax)

    cbar.ax.get_yaxis().set_ticks([])

    for j in range(1, n_clusters+1, 1):
        cbar.ax.text(1.5, (j-1+.5), j, ha='center', va='center', color='black')
    cbar.ax.get_yaxis().labelpad = 15
    #plt.title(model_names[i])
    
    plt.draw()
    plt.savefig(workdir+reg+'_cluster_yyy'+fn,dpi=300,bbox_inches='tight')
    
    if indo and config['use_subregions']:
        import operator
        subregions= sorted(config['subregions'].items(),key=operator.itemgetter(0))
        lons, lats = np.meshgrid(ds.lon, ds.lat) 
        '''
        fig = plt.figure()
        ax = fig.add_subplot(111)
        m = Basemap(ax=ax, projection='cyl',llcrnrlat = lats.min(), urcrnrlat = lats.max(),
                    llcrnrlon = lons.min(), urcrnrlon = lons.max(), resolution = 'h')
        m.drawcoastlines(linewidth=0.75)
        m.drawcountries(linewidth=0.75)
        m.etopo()
        '''
        x, y = m(lons, lats) 
        #subregion_array = ma.masked_equal(subregion_array, 0)
        #max=m.contourf(x, y, subregion_array, alpha=0.7, cmap='Accent')
        for subregion in subregions:
            draw_screen_poly(subregion[1], m, 'w') 
            plt.annotate(subregion[0],xy=(0.5*(subregion[1][2]+subregion[1][3]), 0.5*(subregion[1][0]+subregion[1][1])), 
                          ha='center',va='center', fontsize=5,
                          backgroundcolor='0.90',alpha=1) 

        
        
    plt.show()

def kmeans_sr(obs_dataset, obs_name, model_datasets, model_names, workdir):
    #plot sub region
    from xlearn22.cluster import KMeans
 
    print('*------------------')
    
    #choose
    n_clusters=4
    n_clusters= n_clusters +1
    
    annual_cycle=True
    with_pca=False#default diset n=3 ubah di cluster.py fit2 
    
    sse=False #khusus malaysia error di ...2336
    gap_stat=False
    
    #domain
    sea=False
    indo=True

    fig = plt.figure()
    ax = fig.add_subplot(111)

    #for i in np.arange(len(model_datasets)):
    for i in [0]:
        print(model_names[i])
     
        ds = xr.DataArray(model_datasets[i].values,
        coords={'time': model_datasets[i].times,
                'lat': model_datasets[i].lats, 
                'lon': model_datasets[i].lons},
        dims=["time", "lat", "lon"])
        
        if sea: #-14.5, 25.5, 90.5, 145
            ds = ds.where(
                    (ds.lat > -14.5) & (ds.lat < 25.5) & 
                    (ds.lon > 90.5) & (ds.lon < 145), drop=True)
        
        if indo:
            ds = ds.where(
                    (ds.lat > -11) & (ds.lat < 6.5) & 
                    (ds.lon > 95) & (ds.lon < 150), drop=True)
                        
        lat_min = ds.lat.min()
        lat_max = ds.lat.max()
        lon_min = ds.lon.min()
        lon_max = ds.lon.max()
        x1,y1 = np.meshgrid(ds.lon, ds.lat)
        
        m = Basemap(ax=ax, projection ='cyl', 
                llcrnrlat = lat_min-1*0.22, 
                urcrnrlat = lat_max+2*0.22,
                llcrnrlon = lon_min-4*0.22, 
                urcrnrlon = lon_max-10*0.22, 
                resolution = 'l', fix_aspect=False)
        m.drawcoastlines(linewidth=1)
        m.drawcountries(linewidth=1)
        m.drawstates(linewidth=0.5, color='w')
                       
        #3D ubah ke 2D (time, z(x,y))
        ds= ds.stack(z=("lat", "lon")) #s= s.unstack()
        
        if annual_cycle:
            dm = ds.groupby('time.month').mean()
           
            fn='annual_cycle'
        else:
            
            fn='raw_monthly'
            dm=ds
       
        km = KMeans(n_clusters=n_clusters, random_state=0).fit2(dm, annual_cycle, with_pca)
        cc=['blue','red','pink','green', 'orange',]         
        ds0=km.cluster_centers_da.sel(cluster=0).values
        levels=np.arange(n_clusters)
        p=ax.contourf(x1, y1, ds0, levels=levels, 
                #cmap=cmap2,
                colors=cc,
                )

        if sea:
            ax.set_xticks([100,120,140])
            ax.set_xticklabels(['100E','120E','140E'])
            ax[1,i].tick_params(axis='x', pad=1,labelsize=8)
            ax[1,i].set_xlabel('')
            ax[1,2].set_ylabel('')
            ax[1,0].set_yticks([-10,0,10,20])
            ax[1,0].set_yticklabels(['10S','0','10N','20N'])
            ax[1,0].tick_params(axis='y', pad=1,labelsize=8)
        
        if indo:
            ax.set_xticks([105,120,135])
            ax.set_xticklabels(['105E','120E','135E'])
            ax.tick_params(axis='x', pad=1,labelsize=8)
            ax.set_xlabel('')
            if i>0:
                ax.set_ylabel('')
            ax.set_yticks([-11,-5,0,5])
            ax.set_yticklabels(['10S','5S','0','5N'])
            ax.tick_params(axis='y', pad=1,labelsize=8)
            
    if indo and config['use_subregions']:
        import operator
        subregions= sorted(config['subregions'].items(),key=operator.itemgetter(0))
        lons, lats = np.meshgrid(ds.lon, ds.lat) 
     
        #x, y = m(lons, lats) 
        #subregion_array = ma.masked_equal(subregion_array, 0)
        #max=m.contourf(x, y, subregion_array, alpha=0.7, cmap='Accent')
        for subregion in subregions:
            draw_screen_poly(subregion[1], m, 'r') 
            #ax.annotate(subregion[0],xy=(0.5*(subregion[1][2]+subregion[1][3]), 0.5*(subregion[1][0]+subregion[1][1])), 
            #[-6.5, 6, 95, 107]
            ax.annotate(subregion[0],xy=(1.75+subregion[1][2], .75+subregion[1][0]), 
                          ha='center',va='center', fontsize=10,
                          backgroundcolor='1',
                          alpha=1) 
    
    plt.subplots_adjust(hspace=.3,wspace=.17)
    cax = fig.add_axes([0.91, 0.1, 0.02, 0.5])
    cbar = plt.colorbar(p,cax=cax)
    cbar.ax.get_yaxis().set_ticks([])

    for j in range(1, n_clusters+1, 1):
        cbar.ax.text(1.5, (j-1+.5), j, ha='center', va='center', color='black')
    cbar.ax.get_yaxis().labelpad = 15   
    #plt.draw()
    #plt.savefig(workdir+reg+'_cluster_yyy'+fn,dpi=300,bbox_inches='tight')  
    plt.show()

def kmeans_season(obs_dataset, obs_name, model_datasets, model_names, workdir):
    #tidak bisa dilakukan
    #DJF hanya 3 bulan sementara pola hujan kan 12 bulan
    
    from xlearn22.cluster import KMeans
 
    print('*------------------')
    
    musim=['DJF']
    
    #choose
    n_clusters=4
    n_clusters= n_clusters +1
    
    annual_cycle=True
    with_pca=False#default diset n=3 ubah di cluster.py fit2 
    
    sse=False #khusus malaysia error di ...2336
    gap_stat=False
    
    #domain
    sea=False
    indo=True

    fig = plt.figure()
    ax = fig.add_subplot(111)

    #for i in np.arange(len(model_datasets)):
    for i in [0]:
        print(model_names[i])
     
        ds = xr.DataArray(model_datasets[i].values,
        coords={'time': model_datasets[i].times,
                'lat': model_datasets[i].lats, 
                'lon': model_datasets[i].lons},
        dims=["time", "lat", "lon"])
        
        if sea: #-14.5, 25.5, 90.5, 145
            ds = ds.where(
                    (ds.lat > -14.5) & (ds.lat < 25.5) & 
                    (ds.lon > 90.5) & (ds.lon < 145), drop=True)
        
        if indo:
            ds = ds.where(
                    (ds.lat > -11) & (ds.lat < 6.5) & 
                    (ds.lon > 95) & (ds.lon < 150), drop=True)
                        
        lat_min = ds.lat.min()
        lat_max = ds.lat.max()
        lon_min = ds.lon.min()
        lon_max = ds.lon.max()
        x1,y1 = np.meshgrid(ds.lon, ds.lat)
        
        m = Basemap(ax=ax, projection ='cyl', 
                llcrnrlat = lat_min-1*0.22, 
                urcrnrlat = lat_max+2*0.22,
                llcrnrlon = lon_min-4*0.22, 
                urcrnrlon = lon_max-10*0.22, 
                resolution = 'l', fix_aspect=False)
        m.drawcoastlines(linewidth=1)
        m.drawcountries(linewidth=1)
        m.drawstates(linewidth=0.5, color='w')
                       
        #3D ubah ke 2D (time, z(x,y))
        ds= ds.stack(z=("lat", "lon")) #s= s.unstack()
        
        if annual_cycle:
            dm1 = ds.groupby('time.season')
            dm=dm1['DJF'].mean(dim='time')
           
            fn='annual_cycle'
        else:
            
            fn='raw_monthly'
            dm=ds
       
        km = KMeans(n_clusters=n_clusters, random_state=0).fit4(dm, annual_cycle, with_pca)
        cc=['blue','red','pink','green', 'orange',]         
        ds0=km.cluster_centers_da.sel(cluster=0).values
        levels=np.arange(n_clusters)
        p=ax.contourf(x1, y1, ds0, levels=levels, 
                #cmap=cmap2,
                colors=cc,
                )

        if sea:
            ax.set_xticks([100,120,140])
            ax.set_xticklabels(['100E','120E','140E'])
            ax[1,i].tick_params(axis='x', pad=1,labelsize=8)
            ax[1,i].set_xlabel('')
            ax[1,2].set_ylabel('')
            ax[1,0].set_yticks([-10,0,10,20])
            ax[1,0].set_yticklabels(['10S','0','10N','20N'])
            ax[1,0].tick_params(axis='y', pad=1,labelsize=8)
        
        if indo:
            ax.set_xticks([105,120,135])
            ax.set_xticklabels(['105E','120E','135E'])
            ax.tick_params(axis='x', pad=1,labelsize=8)
            ax.set_xlabel('')
            if i>0:
                ax.set_ylabel('')
            ax.set_yticks([-11,-5,0,5])
            ax.set_yticklabels(['10S','5S','0','5N'])
            ax.tick_params(axis='y', pad=1,labelsize=8)
    '''        
    if indo and config['use_subregions']:
        import operator
        subregions= sorted(config['subregions'].items(),key=operator.itemgetter(0))
        lons, lats = np.meshgrid(ds.lon, ds.lat) 
     
        #x, y = m(lons, lats) 
        #subregion_array = ma.masked_equal(subregion_array, 0)
        #max=m.contourf(x, y, subregion_array, alpha=0.7, cmap='Accent')
        for subregion in subregions:
            draw_screen_poly(subregion[1], m, 'r') 
            #ax.annotate(subregion[0],xy=(0.5*(subregion[1][2]+subregion[1][3]), 0.5*(subregion[1][0]+subregion[1][1])), 
            #[-6.5, 6, 95, 107]
            ax.annotate(subregion[0],xy=(1.75+subregion[1][2], .75+subregion[1][0]), 
                          ha='center',va='center', fontsize=10,
                          backgroundcolor='1',
                          alpha=1) 
    '''
    plt.subplots_adjust(hspace=.3,wspace=.17)
    cax = fig.add_axes([0.91, 0.1, 0.02, 0.5])
    cbar = plt.colorbar(p,cax=cax)
    cbar.ax.get_yaxis().set_ticks([])

    for j in range(1, n_clusters+1, 1):
        cbar.ax.text(1.5, (j-1+.5), j, ha='center', va='center', color='black')
    cbar.ax.get_yaxis().labelpad = 15   
    #plt.draw()
    #plt.savefig(workdir+reg+'_cluster_yyy'+fn,dpi=300,bbox_inches='tight')  
    plt.show()


def dbscan_sea22(obs_dataset, obs_name, model_datasets, model_names, workdir):
    #hasil tidak sejelas Kmeans
   
    from xlearn_dbscan.cluster import DBSCAN
   
    print('*------------------')
    
    #choose
    n_clusters=3
    
    annual_cycle=True
    with_pca=False #default diset n=3 ubah di cluster.py fit2 
    
    sse=False
    gap_stat=False
    
    #domain
    sea=False
    indo=False
    sumatera=True
    jambi=False
    malaysia=False
    maluku=False
    
    fig, ax = plt.subplots(3,len(model_datasets), figsize=(8,8))
    
    for i in [0]: #np.arange(len(model_datasets)):
        print(model_names[i])
     
        ds = xr.DataArray(model_datasets[i].values,
        coords={'time': model_datasets[i].times,
                'lat': model_datasets[i].lats, 
                'lon': model_datasets[i].lons},
        dims=["time", "lat", "lon"])
        
        
        
        if indo:
            ds = ds.where(
                    (ds.lat > -11) & (ds.lat < 6.5) & 
                    (ds.lon > 95) & (ds.lon < 141), drop=True)
        #-6.5 7 93 107
        
        if sumatera:
            ds = ds.where(
                    (ds.lat > -6.5) & (ds.lat < 6.5) & 
                    (ds.lon > 94) & (ds.lon < 107), drop=True)
                
        if jambi:
            ds = ds.where(
                    (ds.lat > -3) & (ds.lat < -0.5) & 
                    (ds.lon > 99) & (ds.lon < 106), drop=True)
                    
        if malaysia:
            ds = ds.where(
                    (ds.lat > 0.6) & (ds.lat < 6.6) & 
                    (ds.lon > 99) & (ds.lon < 104.6), drop=True)
        
        if maluku:
            ds = ds.where(
                    (ds.lat > -5) & (ds.lat < 5) & 
                    (ds.lon > 125) & (ds.lon < 130), drop=True)
                   
        lat_min = ds.lat.min()
        lat_max = ds.lat.max()
        lon_min = ds.lon.min()
        lon_max = ds.lon.max()
        x,y = np.meshgrid(ds.lon, ds.lat)
        
        m = Basemap(ax=ax[1,i], projection ='cyl', 
                llcrnrlat = lat_min, #+1*0.22, 
                urcrnrlat = lat_max, #-6*0.22,
                llcrnrlon = lon_min, #+4*0.22, 
                urcrnrlon = lon_max, #-3*0.22, 
                resolution = 'l', fix_aspect=False)
        m.drawcoastlines(linewidth=1)
        m.drawcountries(linewidth=1)
        m.drawstates(linewidth=0.5, color='w')
        
   
        
        #3D ubah ke 2D (time, z(x,y))
        ds= ds.stack(z=("lat", "lon")) #s= s.unstack()
        
        if annual_cycle:
            dm = ds.groupby('time.month').mean()
           
            fn='annual_cycle'
        else:
            
            fn='raw_monthly'
            dm=ds
       
        km = DBSCAN(eps=0.7, min_samples=3).fit(dm, annual_cycle, with_pca)
                   
              
        print('core_sample_indices_',km.core_sample_indices_)
        print('core_sample_indices_',km.core_sample_indices_.shape)
        
        print('components_ ',km.components_ )
        print('components_ ',km.components_.shape )
        
        print('labels_ ',km.labels_ )
        print('labels_ ',km.labels_.shape )
        
        print('km.cluster_centers_da.sel(cluster=0)', km.cluster_centers_da.sel(cluster=0))
        print('dbscan.cluster_centers_ shape max and min:')
        print(km.cluster_centers_da.sel(cluster=0).shape,
            km.cluster_centers_da.sel(cluster=0).max().data,
            km.cluster_centers_da.sel(cluster=0).min().data)
        
        print('km.labels_.max()',km.labels_.max())
        #plot map n clusters
        p=km.cluster_centers_da.sel(cluster=0).plot(ax=ax[1,i], 
          levels=np.arange(km.labels_.max()),
    
          colors=['blue','red','pink','green', 'orange', 'indigo', 
                 'violet', 'darkred', 'darkorange', 'darkgreen'],
          add_colorbar = False)
        
        #plot ts
        x_tick=['J','F','M','A','M','J','J','A','S','O','N','D']
        x=np.arange(km.components_.shape[1]) #12
                
       
        c= ['blue','red','pink','green', 'orange', 'indigo', 'violet', 'darkred', 'darkorange', 'darkgreen']
      
        lw=['2','2','2','2','2']
        #jumlah samples valid
        for ii in range (km.components_.shape[0]):
            print(ii)
            ax[2,i].plot(x, km.components_[ii], 
            #color=c[ii],
            lw=2
            ) 
     
        ax[2,i].set_xticks(x)
        if annual_cycle and not with_pca:
            ax[2,i].set_xticklabels(x_tick, fontsize=8)
        #ax[0,i].text(3,f[i],model_names[i])
        ax[2,0].set_ylabel('Mean pr (mm/month)')
        ax[2,i].tick_params(axis='y', pad=1,labelsize=8)
        #plt.legend(bbox_to_anchor=(1, .4), loc='best', prop={'size':8.5}, frameon=False) 
        
        #untuk model<2
        ax[1,1].set_ylabel('')
        
        
        if not sse:
            ax[1,i].set_title(model_names[i], fontsize=9)
            
            ax[0,i].set_xticks([])
            
        if sse:
            ax[0,i].set_title(model_names[i], fontsize=9)
            ax[1,i].set_title('', fontsize=9)
        
        if sea:
            ax[1,i].set_xticks([100,120,140])
            ax[1,i].set_xticklabels(['100E','120E','140E'])
            ax[1,i].tick_params(axis='x', pad=1,labelsize=8)
            ax[1,i].set_xlabel('')
            ax[1,2].set_ylabel('')
            ax[1,0].set_yticks([-10,0,10,20])
            ax[1,0].set_yticklabels(['10S','0','10N','20N'])
            ax[1,0].tick_params(axis='y', pad=1,labelsize=8)
        
        if indo:
            ax[1,i].set_xticks([105,120,135])
            ax[1,i].set_xticklabels(['105E','120E','135E'])
            ax[1,i].tick_params(axis='x', pad=1,labelsize=8)
            ax[1,i].set_xlabel('')
            if i>0:
                ax[1,i].set_ylabel('')
            ax[1,0].set_yticks([-11,-5,0,5])
            ax[1,0].set_yticklabels(['10S','5S','0','5N'])
            ax[1,0].tick_params(axis='y', pad=1,labelsize=8)
            
        if sumatera:
            ax[1,i].set_xticks([95,100,105])
            ax[1,i].set_xticklabels(['95E','100E','105E'])
            ax[1,i].tick_params(axis='x', pad=1,labelsize=8)
            ax[1,i].set_xlabel('')
            ax[1,i].set_ylabel('')
            ax[1,0].set_yticks([-5,0,5])
            ax[1,0].set_yticklabels(['5S','0','5N'])
            ax[1,0].tick_params(axis='y', pad=1,labelsize=8)
            
        if jambi:
            ax[0,i].set_xticks([100,104])
            ax[0,i].set_xticklabels(['100$^\circ$E','104$^\circ$E'])
            ax[0,i].tick_params(axis='x', pad=1,labelsize=8)
            ax[0,i].set_xlabel('')
            ax[0,i].set_ylabel('')
            ax[0,0].set_yticks([-3,-2,-1])
            ax[0,0].set_yticklabels(['3$^\circ$S','2$^\circ$S','1$^\circ$S'])
            ax[0,0].tick_params(axis='y', pad=1,labelsize=8)
        
        if i<len(model_datasets)-1:
            ax[1,i+1].set_yticks([])
            ax[1,i+1].set_yticklabels([])
            ax[1,i+1].set_ylabel('')
  
    plt.subplots_adjust(hspace=.3,wspace=.17)
    
    cax = fig.add_axes([0.91, 0.1, 0.02, 0.5])
   
    cbar = plt.colorbar(p,cax=cax)

    cbar.ax.get_yaxis().set_ticks([])

    for j in range(1, km.labels_.max(), 1):
        cbar.ax.text(1.5, (j-1+.5), j, ha='center', va='center', color='black')
    cbar.ax.get_yaxis().labelpad = 15
    #plt.title(model_names[i])
    
    plt.draw()
    plt.savefig(workdir+reg+'_cluster_yyy'+fn,dpi=300,bbox_inches='tight')
        
        
    plt.show()


def kmeans_masking(obs_dataset, obs_name, model_datasets, model_names, workdir):
    from xlearn22.cluster import KMeans
    from scipy.spatial.distance import euclidean
   
    print('*------------------')
    
    #choose
    n_clusters=3
    annual_cycle=True
    with_pca=False #default n=3 
    
    #domain
    sea=False
    indo=False
    sumatera=True
    jambi=False
    
    
    clusters=[]
    for i in [0]: #np.arange(len(model_datasets)):
        print(model_names[i])
     
        ds = xr.DataArray(model_datasets[i].values,
        coords={'time': model_datasets[i].times,
                'lat': model_datasets[i].lats, 
                'lon': model_datasets[i].lons},
        dims=["time", "lat", "lon"])
        
        print('ds.time=',ds.time)
        
        if indo:
            ds = ds.where(
                    (ds.lat > -11) & (ds.lat < 6.5) & 
                    (ds.lon > 95) & (ds.lon < 141), drop=True)
        #-6.5 7 93 107
        if sumatera:
            ds = ds.where(
                    (ds.lat > -6.5) & (ds.lat < 6.5) & 
                    (ds.lon > 94) & (ds.lon < 107), drop=True)
                 
        if jambi:
            ds = ds.where(
                    (ds.lat > -3) & (ds.lat < -0.5) & 
                    (ds.lon > 99) & (ds.lon < 106), drop=True)
                    
        ds0=ds
      
        
        ds= ds.stack(z=("lat", "lon")) #s= s.unstack()
        
        if annual_cycle:
            dm = ds.groupby('time.month').mean()
           
            fn='annual_cycle'
        else:
            
            fn='raw_monthly'
            dm=ds
       
        km = KMeans(n_clusters=n_clusters, random_state=0).fit2(dm, annual_cycle, with_pca)
           
        n=km.cluster_centers_da.sel(cluster=0)
        
        #labeling cluster code to ds0 (monthly data) 
        #to wavelet, fft, ts,  dll
        ds0.coords['mask'] = (('lat', 'lon'), n.values)
        for n in np.arange(n_clusters):
            tes=ds0.where(ds0.mask == n).mean(dim=("lat", "lon"))
            clusters.append(tes)
        
        cluster_names=['cluster 1 (blue)', 'cluster 2 (red)','cluster 3 (pink)']
        model_name=model_names[i]
        if model_name == 'TRMM':
           print('ds.time=',ds.time)
           #print('ds0.time',ds0.time)
        
        '''
        # Calculate pairwise distances between centroids
        centroids = km.cluster_centers_
        euclidean_distances = np.zeros((len(centroids), len(centroids)))
        euclidean_distances2=[]
        for i in range(len(centroids)):
            for j in range(i + 1, len(centroids)):
                euclidean_distances[i, j] = euclidean(centroids[i], centroids[j])
                euclidean_distances[j, i] = euclidean_distances[i, j]

        for i in range(len(centroids)):
            for j in range(i + 1, len(centroids)):
                euclidean_distances2.append(euclidean_distances[i,j])
        
        euclidean_distances2=[round(x,2) for x in euclidean_distances2]
        average_euclidean_distance = np.mean(euclidean_distances)

        print("Pairwise Euclidean Distances 1-2, 1-3, 2-3:")
        print(euclidean_distances)
        print(euclidean_distances2)
        print("\nAverage Euclidean Distance:", average_euclidean_distance)
        print("\nAverage Euclidean Distance:", np.mean(euclidean_distances2))

        exit()
        '''
    #print(model)
    #cluster_wavelet(model_name, clusters, cluster_names, workdir)
    cluster_trend(model_name, clusters, cluster_names, workdir)  
    #cluster_spectral(model_name, clusters, cluster_names, workdir)  
    #plt.plot(tes.time, tes.values)
    #plt.show()  
        
def cluster_spectral(model_name, clusters, cluster_names, workdir):
    from scipy.signal import periodogram
    fig, ax = plt.subplots(3,1, figsize=(3,2))
   
    for i in [0,1,2]:
        
        dsi=clusters[i]
        
        # Perform spectral analysis
        frequencies, psd = periodogram(dsi, fs=12)  
        # Sampling frequency assumed to be 12 month
        
        ax[i].plot(1/frequencies, psd)
        ax[i].set_xscale('log')
        #plt.yscale('log')
        ax[2].set_xlabel('Period (years)')
        ax[1].set_ylabel('Power spectral density')
        ax[i].set_title(cluster_names[i])
        #plt.grid(True)
        
        # Add important time scales to the plot
        important_periods = [0.25, 0.5, 1, 2, 4, 7, 10]
        #for period in important_periods:
        #    plt.axvline(x=period, color='r', linestyle='--', linewidth=0.8)

        ax[i].set_xticks(important_periods, labels=[f'{period}' for period in important_periods])
        ax[i].set_xlim(left=0.2, right=8)
    plt.subplots_adjust(hspace=.7,wspace=.1)
    plt.subplots_adjust(right=.45)
    plt.subplots_adjust(bottom=.3)
    #plt.tight_layout()
    plt.show()    
                 
def cluster_trend(model_name, clusters, cluster_names, workdir):
    import pymannkendall as mk
    from scipy.stats import t
    from scipy.stats import linregress
    '''
    #cluster 1
    ds=model[0]
    
    k=mk.original_test(ds)
               
    print('k,k.p,k.z=',k,k.p,k.z)          
    if k.p<=0.05: 
        
        #sig=k.p
        s=k.z
        print('s',s)
    '''   
    #fig, ax = plt.subplots(1,len(model_datasets), figsize=(8,8))
    fig=plt.figure(figsize=[8,6])
    fig.subplots_adjust(right=.7)
    mm=['','>','s','x','*','+', 'o']
    c=['blue', 'red', 'pink']
    for i in [0,1,2]:
        k=mk.original_test(clusters[i])
        print('k,k.p,k.z=',k,k.p,k.z)
        # Calculate the trend line (linear regression)
        dsi=clusters[i]
        
        #for annual
        dsi = dsi.groupby('time.year').sum() 
        time = dsi.year
        
        #for monthly time scale ==> error at i=1 in line 749 why?
        #time = dsi.time.data
        #this ok
        #time = np.arange(len(dsi.time.data))
        #print(time)
        
        rainfall_values = dsi.values
        trend = np.polyfit(time, rainfall_values, 1)
        trend_line = np.polyval(trend, time)
        
        #--- tes significance_level
        # Fit the trend line (linear regression)
        slope, intercept, r_value, p_value, std_err = linregress(time, rainfall_values)

        # Calculate the t-statistic
        t_statistic = slope / std_err

        # Degrees of freedom
        df = len(time) - 2  # degrees of freedom

        # Calculate the p-value
        p_value = 2 * (1 - t.cdf(abs(t_statistic), df))
        
        print('p_value',p_value)
        # Compare p-value to significance level (e.g., 0.05)
        if p_value < 0.05:
            print("The trend is significant.")
        else:
            print("The trend is not significant.")
   
        plt.plot(time, dsi.values, color='k', 
            marker=mm[i],
            label = cluster_names[i])
        plt.plot(time, trend_line, color=c[i], ls='--', label = '   "'+'      ('+str(k.trend)+')') #model_names[i])
    #saobs
    #plt.legend(bbox_to_anchor=(0.35, .7), prop={'size':8.5}, frameon=False, handlelength=3.5) 
    #ERA5
    plt.legend(bbox_to_anchor=(.3, .26), prop={'size':8.5}, frameon=False, handlelength=3.5) 
    plt.ylabel('Mean precipitation (mm/year)')
    plt.title (model_name)
    #plt.title ('SAOBS')
    plt.show()

def cluster_trend2(model_name, clusters, cluster_names, workdir):
    #trend with auto=correlation_
    from statsmodels.tsa.ar_model import AutoReg
    from pymannkendall import pre_whitening_modification_test as pmk
    import pandas as pd
    from statsmodels.stats.stattools import durbin_watson
    
    fig=plt.figure(figsize=[8,6])
    fig.subplots_adjust(right=.7)
    mm=['','>','s','x','*','+', 'o']
    c=['blue', 'red', 'pink']
    for i in [0,1,2]:
        #k=mk.original_test(clusters[i])
        
        time_series=clusters[i].values
        # Fit AR model to estimate autocorrelation structure
        lag_order = 1
        ar_model = AutoReg(time_series, lags=lag_order)
        ar_model_fit = ar_model.fit()
        
        # Plot time series data and AR model fit results
        plt.figure(figsize=(10, 6))
        plt.plot(time_series, label='Time Series Data', color='blue')
        plt.plot(ar_model_fit.fittedvalues, label='AR Model Fit', color='red', linestyle='--')
        plt.title('Time Series Data and AR Model Fit')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        plt.show()
        
        # Estimated autocorrelation coefficient
        autocorr_estimated = ar_model_fit.params[1]  
        
        # Print estimated autocorrelation
        print(" ")
        print('Estimated autocorrelation coefficient:', autocorr_estimated)
        # Calculate autocorrelation
        acf_values = pd.plotting.autocorrelation_plot(time_series) 
        plt.title('Autocorrelation Function (ACF)')
        plt.xlabel('Lag')
        plt.ylabel('Autocorrelation')
        plt.grid(True)
        plt.show()
        
        # Extract residuals (deseasonalized and detrended data)
        residuals = ar_model_fit.resid
        #residuals = time_series
        
        # Calculate autocorrelation 2
        acf_values = pd.plotting.autocorrelation_plot(residuals) 
        plt.title('Autocorrelation Function (ACF) for residuals')
        plt.xlabel('Lag')
        plt.ylabel('Autocorrelation')
        plt.grid(True)
        plt.show()
        
        # Assess the model using residuals
        # 1. Mean of Residuals
        mean_residuals = np.mean(residuals)
        print("Mean of Residuals:", mean_residuals)
       
        print("Mean of residuals is close to zero, indicating that the model's predictions are unbiased.")
        
        # Durbin-Watson test for autocorrelation in residuals
        dw_statistic = durbin_watson(residuals)
        print('DW_statistic_residuals:', dw_statistic)
        print("Durbin-Watson statistic is close to 2, suggesting no significant autocorrelation in residuals.")
        
        dw_statistic = durbin_watson(time_series)
        print('DW_statistic_time_series:', dw_statistic)
        
        # Perform Pre-whitened Mann-Kendall test
        trend, h, p, z, Tau, s, var_s, slope, intercept = pmk(residuals)
        
        # Print results
        print(" ")
        print("Trend:", "increasing" if trend == "up" else "decreasing" if trend == "down" else "no trend")
        print("autocorrelation:", h)
        print("p-value:", p)
        print("Kendall's Tau:", Tau)
        print("Slope:", slope)
        print("Intercept:", intercept)
        
         #non residual
        # Perform Pre-whitened Mann-Kendall test
        trend, h, p, z, Tau, s, var_s, slope, intercept = pmk(time_series)
        # Print results
        print(" ")
        print("Trend:", "increasing" if trend == "up" else "decreasing" if trend == "down" else "no trend")
        print("autocorrelation_non_residual:", h)
        print("p-value:", p)
        print("Kendall's Tau:", Tau)
        print("Slope:", slope)
        print("Intercept:", intercept)

        exit()
        
def cluster_trend3(obs_dataset, obs_name, model_datasets, model_names, workdir):
    #trend with auto=correlation_
    from statsmodels.tsa.ar_model import AutoReg
    import pymannkendall as mk
    from pymannkendall import pre_whitening_modification_test as pmk
    import pandas as pd
    from statsmodels.stats.stattools import durbin_watson
    from scipy.stats import t
    
    from xlearn22.cluster import KMeans
    from scipy.spatial.distance import euclidean
   
    print('*------------------')
    
    #choose
    n_clusters=3
    annual_cycle=True
    with_pca=False #default n=3 
    
    #domain
    sea=False
    indo=False
    sumatera=True
    jambi=False
    
    
    clusters=[]
    for i in np.arange(len(model_datasets)):
        print(model_names[i])
     
        ds = xr.DataArray(model_datasets[i].values,
        coords={'time': model_datasets[i].times,
                'lat': model_datasets[i].lats, 
                'lon': model_datasets[i].lons},
        dims=["time", "lat", "lon"])
        
        #print('ds.time=',ds.time)
        
        if indo:
            ds = ds.where(
                    (ds.lat > -11) & (ds.lat < 6.5) & 
                    (ds.lon > 95) & (ds.lon < 141), drop=True)
        #-6.5 7 93 107
        if sumatera:
            ds = ds.where(
                    (ds.lat > -6.5) & (ds.lat < 6.5) & 
                    (ds.lon > 94) & (ds.lon < 107), drop=True)
                 
        if jambi:
            ds = ds.where(
                    (ds.lat > -3) & (ds.lat < -0.5) & 
                    (ds.lon > 99) & (ds.lon < 106), drop=True)
                    
        ds0=ds
      
        
        ds= ds.stack(z=("lat", "lon")) #s= s.unstack()
        
        if annual_cycle:
            dm = ds.groupby('time.month').mean()
           
            fn='annual_cycle'
        else:
            
            fn='raw_monthly'
            dm=ds
       
        km = KMeans(n_clusters=n_clusters, random_state=0).fit2(dm, annual_cycle, with_pca)
           
        n=km.cluster_centers_da.sel(cluster=0)
        
        #labeling cluster code to ds0 (monthly data) 
        #to wavelet, fft, ts,  dll
        ds0.coords['mask'] = (('lat', 'lon'), n.values)
        print(model_names[i])
        t_stat=[]
        tmk=[]
        tmkp=[]
        tpmk=[]
        tpmkp=[]
        for n in np.arange(n_clusters):
            dsi=ds0.where(ds0.mask == n).mean(dim=("lat", "lon"))
            
            dsi = dsi.groupby('time.year').sum() 
            
            # Perform mk------------------------
            k=mk.original_test(dsi.values)
            #print('k,k.p,k.z=',k)
            #print('k.p,k.z=',k.p,k.z)
            tmk.append(k.trend)
            tmkp.append(round(k.p,2))
            
            # Perform Pre-whitened Mann-Kendall test
            trend, h, p, z, Tau, s, var_s, slope, intercept = pmk(dsi.values)
            tpmk.append(trend)
            tpmkp.append(round(p,2))
            # Print results
            print(" ")
            #print("Trend:", "increasing" if trend == "up" else "decreasing" if trend == "down" else "no trend")
            #print("autocorrelation:", h)
            #print("p-value:", p)
            #print(" ")
            #-----------------------------
            
            df = pd.DataFrame({'time_series': dsi})

            # Fit an autoregressive model with lag order 1
            lag_order = 1
            model = AutoReg(df['time_series'], lags=lag_order)
            result = model.fit()

            # Get residuals
            residuals = result.resid
            
            # Durbin-Watson test for autocorrelation in residuals
            dw_statistic = durbin_watson(residuals)
            #print('Durbin-Watson statistic:', dw_statistic)
            #print("Durbin-Watson statistic is close to 2, suggesting no significant autocorrelation in residuals.")
            #print("")
            
            # Calculate autocorrelation coefficient at lag 1
            autocorr_lag1 = df['time_series'].autocorr(lag=1)

            alpha = 0.05
            # Calculate t-statistic
            n = len(df)
            t_statistic = autocorr_lag1 * (((n - 2)**0.5) / (1 - autocorr_lag1**2)**0.5)
            #print('t_statistic =',t_statistic)
             
            t_stat.append(round(t_statistic,2))
            
           
            #critical_value = 2 / (n ** 0.5) * (1 - alpha/2)
            #print('critical_value =', critical_value)
            
            # Calculate critical value for significance level (alpha)
            # where the null hypothesis of no autocorrelation is
            # rejected if t >= t_alfa/2
            
            df = n-1 # Example degrees of freedom

            # Find t_{alpha/2} (two-tailed critical value) using the ppf (percent point function) method
            critical_value = t.ppf(1 - alpha/2, df)
            #print('critical_value=', critical_value)
            
            # Perform t-test
            #if abs(t_statistic) > critical_value:
            #    print("There is significant autocorrelation in the series (reject the null hypothesis).")
            #else:
            #    print("There is no significant autocorrelation in the series (fail to reject the null hypothesis).")
            #print("")
        print('alpha=', alpha)
        print('critical_value=', round(critical_value,2))
        print('t_stat=',t_stat)
        print('tmk=',tmk)
        print('tmkp=',tmkp)
        print('tpmk=',tpmk)
        print('tpmkp=',tpmkp)
        print("")

def kmeans3(obs_dataset, obs_name, model_datasets, model_names, workdir):
    from xlearn22.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from kneed import KneeLocator
    #from sklearn.preprocessing import StandardScaler
    #import geoxarray
    print('*--------clustering----------')
    
    #choose
    annual_cycle=True #jika False alias raw monthly => error contain NaN
    monthly=True
    with_pca=False #default n=3 
    
    
    fig, ax = plt.subplots(3,len(model_datasets), figsize=(8,8))
    
    for i in np.arange(len(model_datasets)):
        print(model_names[i])
     
        ds = xr.DataArray(model_datasets[i].values,
        coords={'time': obs_dataset.times,
                'lat': obs_dataset.lats, 
                'lon': obs_dataset.lons},
        dims=["time", "lat", "lon"])
        
        lat_min = ds.lat.min()
        lat_max = ds.lat.max()
        lon_min = ds.lon.min()
        lon_max = ds.lon.max()
        x,y = np.meshgrid(ds.lon, ds.lat)
       
        ds= ds.stack(z=("lat", "lon")) #s= s.unstack()
        
        if annual_cycle:
            dm = ds.groupby('time.month').mean()
           
            fn='annual_cycle'
        elif monthly:
            print('warning: data raw monthly tanpa PCA hasil')
            print('error akibat masih ada data NaN yg rumit jika')
            print('akan dihilangkan')
            fn='raw_monthly'
            dm=ds
        
        print('dimensi input=', dm.shape)
        
         #The data needs to be normalized, 
        #if there is a high degree of heterogeneity among features (CV > 35%).
        #coefficient of variation CV=std/mean *100% (for @feature 1 2 3 ...)
        if annual_cycle:
            cv=[]
            for ix in np.arange(dm.shape[0]):
                cvv=dm[ix].std()/dm[ix].mean()*100
                cv.append(np.round(cvv.data,2))
            #print('CV=', cv)
            
        # apakah pada Kmeans ini dm perlu di scaling 0-1 ??
        # disini kayaknya tidak pakai
        
        sse = []
        silhouette_coef = []
        #fit2 dicoba kini fit3
        for k in range(1, 7):
            print('k=',k)
            kmeans, features = KMeans(n_clusters=k,  random_state=0).fit3(dm, annual_cycle, with_pca)
            sse.append(kmeans.inertia_) 
            #print('kmeans.labels.shape', kmeans.labels_.shape)
            #print('features.shape', features.shape)
            if k>1:
                label2=kmeans.cluster_centers_da.sel(cluster=0).stack(z=("lat", "lon"))
                label2=label2.dropna(dim=('z'))
                #print('label2', label2)
           
                score = silhouette_score(features, label2)
            else:
                score=0
            silhouette_coef.append(score)
        
        kl = KneeLocator(
            range(1, 7), sse, curve="convex", direction="decreasing"
            )
        c=kl.elbow
        print('Optimum cluster1=',c)
        
       
        #plt.style.use("fivethirtyeight")
        ax[0,i].plot(range(1, 7), sse, color='black')
        #ax[0,i].axvline(x=c,ymin=0, ymax=sse[c-1]/100000, ls=':', color='black') 
        ax[0,i].axvline(x=c,ls=':', color='black') 
        
        ax[0,0].ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        ax[0,i].set_xticks(range(1, 7))
        ax[0,i].set_title("N optimum of Clusters", fontsize=8)
        #ax[2,i].set_yticks([])
        #ax[2,i].set_yticklabels([])
        ax[0,0].set_ylabel("SSE")
        if i<len(model_datasets)-1:
            ax[0,i+1].set_yticks([])
            ax[0,i+1].set_yticklabels([])
        
        s=np.array(silhouette_coef)
        c2=np.where(s==s.max())[0][0] # [0][0] first only
        #print('silhouette_coef',silhouette_coef)
        print('Optimum cluster2=',c2+1)
        
        ax2 = ax[0,i].twinx()  # instantiate a second axes that shares the same x-axis

        color = 'tab:red'
        
        ax2.plot(range(1, 7), silhouette_coef, color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        ax2.axvline(x=c2+1, ls=':', color='red') 
        if i<len(model_datasets)-1:
            ax2.set_yticks([])
            ax2.set_yticklabels([])        
        if i==len(model_datasets)-1:
            ax2.set_ylabel('silhouette_coef', color=color)
        
        if c<3: c=3
        
        km = KMeans(n_clusters=c, random_state=0).fit2(dm, annual_cycle, with_pca)
                   
        
        #print('km.cluster_centers_',km.cluster_centers_)
        #print('km.cluster_centers_.shape',km.cluster_centers_.shape)
        #print('m.labels_', m.labels_)
        #print('km.labels_.shape', km.labels_.shape)
        #print(km.n_clusters)
        
        m = Basemap(ax=ax[1,i], projection ='cyl', 
                llcrnrlat = lat_min, #+1*0.22, 
                urcrnrlat = lat_max, #-6*0.22,
                llcrnrlon = lon_min, #+4*0.22, 
                urcrnrlon = lon_max, #-3*0.22, 
                resolution = 'l', fix_aspect=False)
        m.drawcoastlines(linewidth=1)
        m.drawcountries(linewidth=1)
        m.drawstates(linewidth=0.5, color='w')
        
        p=km.cluster_centers_da.sel(cluster=0).plot(ax=ax[1,i], 
          levels=np.arange(c+1),
          cmap='rainbow',
          add_colorbar = False)
        ax[1,i].set_title(model_names[i], fontsize=9)
        ax[1,i].set_xticks([95,100,105])
        ax[1,i].set_xticklabels(['95E','100E','105E'])
        ax[1,i].tick_params(axis='x', pad=1,labelsize=8)
        ax[1,i].set_xlabel('')
        ax[1,0].set_yticks([-5,0,5])
        ax[1,0].set_yticklabels(['5S','0','5N'])
        
        if i<len(model_datasets)-1:
            ax[1,i+1].set_yticks([])
            ax[1,i+1].set_yticklabels([])
            ax[1,i+1].set_ylabel('')
        
        x_tick=['J','F','M','A','M','J','J','A','S','O','N','D']
        x=np.arange(km.cluster_centers_.shape[1])+1
        
        #c=['black','green','yellow']
        c=['blue','green','red']
        #c=['purple','green','red']
        lw=['3','2','1']
        for ii in range (km.n_clusters):
            
            ax[2,i].plot(x, km.cluster_centers_[ii], color=c[ii],lw=lw[ii]) #, alpha=.7) #label = ii+1)
            #ax[0,i].plot(x, m.cluster_centers_[ii],lw=2, label = ii+1)
            #ax[1,i].set_ylim(1,12)
        #f.append(m.cluster_centers_[m.n_clusters-1].max()*1.05)
        if i<len(model_datasets)-1:
            ax[2,i+1].set_yticks([])
            ax[2,i+1].set_yticklabels([])
        #dm.mean(dim='month').plot.contourf(ax=axes[1,1])
        #ax[1,i].set_title(model_names[i], fontsize=10)
        ax[2,i].set_xticks(x)
        if annual_cycle and not with_pca:
            ax[2,i].set_xticklabels(x_tick, fontsize=8)
        #if with_pca:
            #ax[2,i].set_xticks(x)
            #ax[2,i].set_xticklabels(x_tick2, fontsize=8)
         
        #ax[0,i].text(3,f[i],model_names[i])
        ax[2,0].set_ylabel('Mean pr (mm/month)')
        #plt.legend(bbox_to_anchor=(1, .4), loc='best', prop={'size':8.5}, frameon=False) 
        
        '''      
        color = 'tab:red'
        ax[3,0].set_ylabel('silhouette_coef', color=color)  
        ax[3,i].plot(range(1, 7), silhouette_coef, color=color)
        ax[3,0].tick_params(axis='y', labelcolor=color)
        ''' 
           
        #untuk model<2
        ax[0,1].set_ylabel('')
        
        # #untuk model<6
        # ax[0,1].set_ylabel('')
        # ax[0,2].set_ylabel('')
        # ax[0,3].set_ylabel('')
        # ax[0,4].set_ylabel('')
        # ax[0,5].set_ylabel('')
        # ax[2,1].set_yticklabels([])
        # ax[2,2].set_yticklabels([])
        # ax[2,3].set_yticklabels([])
        # ax[2,4].set_yticklabels([])
        # ax[2,5].set_yticklabels([])
        
    
    plt.subplots_adjust(hspace=.35,wspace=.05)
    cax = fig.add_axes([0.91, 0.1, 0.02, 0.5])
   
    cbar = plt.colorbar(p,cax=cax)

    cbar.ax.get_yaxis().set_ticks([])

    for j in range(1, 4, 1):
        cbar.ax.text(1.5, (j-1+.5), j, ha='center', va='center', color='black')
    cbar.ax.get_yaxis().labelpad = 15
    #plt.title(model_names[i])
    
    plt.draw()
    plt.savefig(workdir+reg+'_cluster_'+fn,dpi=300,bbox_inches='tight')
        
        
    plt.show()   

def hirarki_clustering(obs_dataset, obs_name, model_datasets, model_names, workdir):
    from scipy.cluster import hierarchy
    print('*------------------')
    
       
    for i in np.arange(len(model_datasets)):
        print(model_names[i])
     
        ds = xr.DataArray(model_datasets[i].values,
        coords={'time': obs_dataset.times,
                'lat': obs_dataset.lats, 
                'lon': obs_dataset.lons},
        dims=["time", "lat", "lon"])
        
             
        ds= ds.stack(z=("lat", "lon")) #s= s.unstack()
        
       
        dm = ds.groupby('time.month').mean()
        
        print(dm)
        '''       
        # any feature contains np.NaN
        valid_features_index = ~np.isnan(dm[0,:])
        dm = dm[:, valid_features_index.data]
        '''
        print('dm=',dm.shape)
        dm=dm.T
        print('dm=',dm.shape)
        valid_features_index = ~np.isnan(dm[:,0])
        dm = dm[valid_features_index.data, :]
        print('dm=',dm.shape)
        print(dm[0:200,:])
        
        # Perform hierarchical clustering
        Z = hierarchy.linkage(dm[0:200,:], method='single', metric='euclidean')

        # Plot the dendrogram
        plt.figure(figsize=(10, 5))
        dn = hierarchy.dendrogram(Z)
        plt.title('Dendrogram')
        plt.xlabel('Data points')
        plt.ylabel('Distance')

        # Set a cutoff distance to determine the number of clusters
        cutoff_distance = 1.0
        clusters = hierarchy.fcluster(Z, cutoff_distance, criterion='distance')

        print("Cluster assignments:", clusters)
        print("Cluster assignments:", clusters.shape)

        plt.show()

        
        
        
        
        

def temporal_corr(obs_dataset, obs_name, model_datasets, model_names, file_name):
    print('temporal_corr')
    #ds = xr.DataArray(obs_dataset) #tidak bisa langsung
    #print('obs_dataset.times=', obs_dataset.times)
    
    ds = xr.DataArray(obs_dataset.values,
    coords={'time': obs_dataset.times,
    'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
    dims=["time", "lat", "lon"])
        
    #ds = ds.groupby('time.month').mean() 
    ds = ds.groupby('time.year').sum() 
    #ds = ds.groupby('time.dayofyear').mean() # datanya bulanan jadi error
    print('xx=',ds.shape, ds.lat.shape) 
    
    ##temporal_corr
    lon=103
    lat=-1
    x=extract_data_at_nearest_grid_point2(ds, lon, lat)
    print('x=',x)
    
    
    ###cal corr from monthly cycle => from annual ? and from raw_input
    from scipy.stats import pearsonr
    corr=ma.zeros(len(model_datasets)) 
    for i in np.arange(len(model_datasets)):
        ds = xr.DataArray(model_datasets[i].values,
        coords={'time': obs_dataset.times,
        'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
        dims=["time", "lat", "lon"])
        
        #ds = ds.groupby('time.month').mean()
        ds = ds.groupby('time.year').sum()
        dsi = ds.groupby('time.year').sum()
        #ds = ds.groupby('time.dayofyear').mean()
        
        #print(ds.shape,ds.lat.shape, ds.lon.shape)
        y=extract_data_at_nearest_grid_point2(ds, lon, lat)
        #print('y=',y)
        corr[i]=pearsonr(y, x)[0]
    print('corr_time.year=',corr)        
    #########

def spatial_taylor(obs_dataset, obs_name, model_datasets, model_names, workdir):
    print('spatial_taylor')
    
    ds = xr.DataArray(obs_dataset.values,
    coords={'time': obs_dataset.times,
    'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
    dims=["time", "lat", "lon"])
    
    dsm=ds.mean(dim=("time"))
    sd1m=dsm.std(skipna=None)
    
   
    T=[]
    r0=1
    for i in np.arange(len(model_datasets)):
        #i=10
        #print(model_names[i])
        dsi = xr.DataArray(model_datasets[i].values,
        coords={'time': obs_dataset.times,
        'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
        dims=["time", "lat", "lon"])  
        
        dsim=dsi.mean(dim=("time")) #year, value (1 y maka 1 v)
    
        #dsim.plot()
        #plt.show()
        
        c=xr.corr(dsim,dsm)
        sd2=dsim.std(skipna=True)
        s=sd2/sd1m
      
        #T= (4*(1+c)**4)/(((s+1/s)**2)*((1+r0)**4))
        # Tanpa pakai **4 T naik dikit
        tt= (4*(1+c))/(((s+1/s)**2)*((1+r0)))
        #print(tt.data)
        T.append(np.round(tt.data,2))
    print('Metrik spatial',T)
    return T   

def spatial_taylor2(obs_dataset, obs_name, model_datasets, model_names, workdir):
    #tidak pakai xr > hasil sama
    # calculate climatological mean fields
    obs_clim_dataset = ds.Dataset(obs_dataset.lats, obs_dataset.lons, obs_dataset.times, utils.calc_temporal_mean(obs_dataset))
    model_clim_datasets = []
    for dataset in model_datasets:
        model_clim_datasets.append(ds.Dataset(dataset.lats, dataset.lons, dataset.times, utils.calc_temporal_mean(dataset)))
    
    print('All_metrics subsetting =>', obs_clim_dataset.values.shape)
    
    metrics1=metrics.Metric_s_taylor()
    metrics1_evaluation = Evaluation(obs_clim_dataset, # Climatological mean of reference dataset for the evaluation
                                 model_clim_datasets, # list of climatological means from model datasets for the evaluation
                                 [metrics1])
    metrics1_evaluation.run() 
    xmetrics1 = metrics1_evaluation.results[0]  
    #print('xmetrics=',xmetrics1)
    
    
def xskillscore(obs_dataset, obs_name, model_datasets, model_names, workdir):
    from scipy.stats import pearsonr, shapiro, skew
    from statsmodels.graphics.gofplots import qqplot
    from scipy import stats
    import xskillscore as xs
    
    fig, ax = plt.subplots(1,len(model_datasets), figsize=(8,8))
       
    ds = xr.DataArray(obs_dataset.values,
    coords={'time': obs_dataset.times,
    'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
    dims=["time", "lat", "lon"])
    
    ds1=ds.mean(dim=("lat", 'lon'))
    
    print('obs=', obs_name)
    '''
    # Histogram untuk melihat distribusi data
    plt.hist(ds.data, bins='auto')
    plt.title('Histogram Data Hujan')
    plt.xlabel('Curah Hujan')
    plt.ylabel('Frekuensi')
    plt.show()

    # Q-Q plot untuk melihat kesesuaian dengan distribusi normal
    qqplot(ds.data, line='s')
    plt.title('Q-Q Plot Data Hujan')
    plt.show()
    '''
    
    # Uji normalitas menggunakan Shapiro-Wilk test
    stat, p = shapiro(ds1.data)
    alpha = 0.05
    if p > alpha:
        print("Data terdistribusi normal (tidak dapat menolak hipotesis nol)")
    else:
        print("Data tidak terdistribusi normal (menolak hipotesis nol)")

    skewness = skew(ds1.data)
    print("Skewness:", skewness)
    
   
    for i in np.arange(len(model_datasets)):
        #i=10
        print('')
        print(model_names[i])
        dsi = xr.DataArray(model_datasets[i].values,
        coords={'time': obs_dataset.times,
        'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
        dims=["time", "lat", "lon"]) 
        
        dsi1=dsi.mean(dim=("lat", 'lon'))
        
        # Uji normalitas menggunakan Shapiro-Wilk test
        stat, p = shapiro(dsi1.data)
        alpha = 0.05
        if p > alpha:
            print("Data terdistribusi normal (tidak dapat menolak hipotesis nol)")
        else:
            print("Data tidak terdistribusi normal (menolak hipotesis nol)")
        
        skewness = skew(dsi1.data)
        print("Skewness:", skewness)
        
        pr = xs.pearson_r(ds1, dsi1) #temporal
        prp = xs.pearson_r_p_value(ds1, dsi1) 
        print('pearson_r, pv=', pr.data, prp.data)
        
        sp= xs.spearman_r(ds1, dsi1) #temporal
        spp= xs.spearman_r_p_value(ds1, dsi1) 
        print('spearman_r, pv=', sp.data, spp.data)
        
        pr2 = xs.pearson_r(ds, dsi, dim='time', skipna=True) #spatial
        prp2 = xs.pearson_r_p_value(ds, dsi, dim='time', skipna=True)
        
        print('spatial_pearson_r, pv=', pr2.mean().data, prp2.mean().data)
        
        pr2.plot(ax=ax[i]) #, add_colorbar=False)
        ax[i].set_title(model_names[i])
        if i>0: 
            ax[i].set_xticks([])
            ax[i].set_xlabel('')
            ax[i].set_yticks([])
            ax[i].set_ylabel('')
           
        
        sp2= xs.spearman_r(ds, dsi, dim=['lat', 'lon'], skipna=True) #temporal
        spp2= xs.spearman_r_p_value(ds, dsi, dim=['lat', 'lon'], skipna=True) 
        print('temporal spearman_r, pv=', sp2.mean().data, spp2.mean().data)
        
        #sp2.plot()
        #plt.show()
        
        # 1. Uji t berpasangan (paired t-test)
        t_statistic, p_value_t = stats.ttest_rel(ds1.data, dsi1.data)

        print("Hasil uji t berpasangan:")
        print("T-statistic:", t_statistic)
        print("P-value:", p_value_t)

        # 2. Uji Wilcoxon berpasangan
        wilcoxon_stat, p_value_wilcoxon = stats.wilcoxon(ds1.data, dsi1.data)

        print("\nHasil uji Wilcoxon berpasangan:")
        print("Wilcoxon statistic:", wilcoxon_stat)
        print("P-value:", p_value_wilcoxon)
        
        # Kesimpulan berdasarkan hasil uji hipotesis
        alpha = 0.05
        print("\nKesimpulan:")
        if p_value_t < alpha:
            print("Terdapat perbedaan yang signifikan")
        else:
            print("Tidak terdapat perbedaan yang signifikan")
            
        if p_value_wilcoxon < alpha:
            print("Terdapat perbedaan yang signifikan")
        else:
            print("Tidak terdapat perbedaan yang signifikan")

    plt.subplots_adjust(hspace=.12,wspace=.5)
    file_name=reg+'_Corr spatial via xskillscore'
    #plt.show() 
    #plt.draw()
    #fig.savefig(workdir+file_name,dpi=300,bbox_inches='tight')
    
def xskillscore_season(obs_dataset, obs_name, model_datasets, model_names, workdir):
    from scipy.stats import pearsonr, shapiro, skew
    from statsmodels.graphics.gofplots import qqplot
    from scipy import stats
    import xskillscore as xs
    
    fig, ax = plt.subplots(1,len(model_datasets)-5, figsize=(8,8))
       
    ds = xr.DataArray(obs_dataset.values,
    coords={'time': obs_dataset.times,
    'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
    dims=["time", "lat", "lon"])
    
    ds=ds.groupby('time.season')
    
    ds1=ds.mean(dim=("lat", 'lon'))
    
    print('obs=', obs_name)
    '''
    # Histogram untuk melihat distribusi data
    plt.hist(ds.data, bins='auto')
    plt.title('Histogram Data Hujan')
    plt.xlabel('Curah Hujan')
    plt.ylabel('Frekuensi')
    plt.show()

    # Q-Q plot untuk melihat kesesuaian dengan distribusi normal
    qqplot(ds.data, line='s')
    plt.title('Q-Q Plot Data Hujan')
    plt.show()
    '''
    
    # Uji normalitas menggunakan Shapiro-Wilk test
    stat, p = shapiro(ds1.data)
    alpha = 0.05
    if p > alpha:
        print("Data terdistribusi normal (tidak dapat menolak hipotesis nol)")
    else:
        print("Data tidak terdistribusi normal (menolak hipotesis nol)")

    skewness = skew(ds1.data)
    print("Skewness:", skewness)
    
   
    for i in np.arange(len(model_datasets)-5):
        #i=10
        print('')
        print(model_names[i])
        dsi = xr.DataArray(model_datasets[i].values,
        coords={'time': obs_dataset.times,
        'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
        dims=["time", "lat", "lon"]) 
        
        dsi=dsi.groupby('time.season')
        
        dsi1=dsi.mean(dim=("lat", 'lon'))
        
        # Uji normalitas menggunakan Shapiro-Wilk test
        stat, p = shapiro(dsi1.data)
        alpha = 0.05
        if p > alpha:
            print("Data terdistribusi normal (tidak dapat menolak hipotesis nol)")
        else:
            print("Data tidak terdistribusi normal (menolak hipotesis nol)")
        
        skewness = skew(dsi1.data)
        print("Skewness:", skewness)
        
        pr = xs.pearson_r(ds1, dsi1) #temporal
        prp = xs.pearson_r_p_value(ds1, dsi1) 
        print('pearson_r, pv=', pr.data, prp.data)
        
        sp= xs.spearman_r(ds1, dsi1) #temporal
        spp= xs.spearman_r_p_value(ds1, dsi1) 
        print('spearman_r, pv=', sp.data, spp.data)
        
        #########
        print(ds['DJF'])
        
        pr2 = xs.pearson_r(ds['DJF'], dsi['DJF'], dim='time',  skipna=True) #spatial
        prp2 = xs.pearson_r_p_value(ds['DJF'], dsi['DJF'], dim='time', skipna=True)
        
        print('spatial_pearson_r, pv=', pr2.mean().data, prp2.mean().data)
        '''
        pr2.plot(ax=ax[i]) #, add_colorbar=False)
        ax[i].set_title(model_names[i])
        if i>0: 
            ax[i].set_xticks([])
            ax[i].set_xlabel('')
            ax[i].set_yticks([])
            ax[i].set_ylabel('')
        '''  
        
        sp2= xs.spearman_r(ds['DJF'], dsi['DJF'], dim=['lat', 'lon'], skipna=True) #temporal
        spp2= xs.spearman_r_p_value(ds['DJF'], dsi['DJF'], dim=['lat', 'lon'], skipna=True) 
        print('temporal spearman_r, pv=', sp2.mean().data, spp2.mean().data)
        
        fig=plt.figure()
        sp2.plot()
        plt.show()
    plt.subplots_adjust(hspace=.12,wspace=.5)
    file_name=reg+'_Corr spatial via xskillscore2'
    plt.show() 
    plt.draw()
    fig.savefig(workdir+file_name,dpi=300,bbox_inches='tight') 
        
def ts_taylor(obs_dataset, obs_name, model_datasets, model_names, workdir):
    #Plot time series obs and models for domain in the input
    
    fig,ax=plt.subplots(figsize=[8,6])
    #plt.subplots_adjust(bottom=.2)
    
    ds = xr.DataArray(obs_dataset.values,
    coords={'time': obs_dataset.times,
    'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
    dims=["time", "lat", "lon"])
    
    dsm=ds.mean(dim=("lat", "lon"))
    sd1m=dsm.std(skipna=None)
    
    #ds = ds.groupby('time.month').mean() 
    dsy = ds.groupby('time.year').sum()  
    dsy=dsy.mean(dim=("lat", "lon"))
    sd1y=dsy.std(skipna=None)
   
    Tm=[]
    T=[]
    r0=1
    for i in np.arange(len(model_datasets)):
        #i=10
        #print(model_names[i])
        dsi = xr.DataArray(model_datasets[i].values,
        coords={'time': obs_dataset.times,
        'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
        dims=["time", "lat", "lon"])  
        
        dsim=dsi.mean(dim=("lat", "lon")) #year, value (1 y maka 1 v)
    
        c=xr.corr(dsim,dsm)
        sd2=dsim.std(skipna=True)
        s=sd2/sd1m
        
        #T= (4*(1+c)**4)/(((s+1/s)**2)*((1+r0)**4))
        # Tanpa pakai **4 T naik dikit
        ttm= (4*(1+c))/(((s+1/s)**2)*((1+r0)))
        #print(tt.data)
        Tm.append(np.round(ttm.data,2))

        #annual
        dsiy = dsi.groupby('time.year').sum() #year,lat,lon,value
        dsiy=dsiy.mean(dim=("lat", "lon")) #year, value (1 y maka 1 v)
    
        c=xr.corr(dsiy,dsy)
        sd2=dsiy.std(skipna=True)
        s=sd2/sd1y
        
        #T= (4*(1+c)**4)/(((s+1/s)**2)*((1+r0)**4))
        # Tanpa pakai **4 T naik dikit
        tt= (4*(1+c))/(((s+1/s)**2)*((1+r0)))
        #print(tt.data)
        T.append(np.round(tt.data,2))
        ax.scatter(i,tt, label = model_names[i],marker='o',color='black')
    print('ts_temporal monthly',Tm)
    ax.set_xticks(np.arange(len(model_datasets)))
    #names[0]=''
    ax.set_xticklabels(model_names,fontsize=8.5)
    plt.xticks(rotation=45)
    plt.ylabel('Taylor metric score')
    
    
    file_name='_ts_monthly_taylor'
    plt.savefig(workdir+reg+file_name,dpi=300,bbox_inches='tight')
    plt.show()
    
    
    return Tm, T   
    
    
    
def ac_taylor(obs_dataset, obs_name, model_datasets, model_names, workdir):
    #make time series obs and models: monthly, annual, annual_cycle
    #calculate models' performance by each metrics
    #bisa juga spatially setelah rata2 climatology
    
    from scipy.stats import mstats
    import math
    
    sort=False
    
    fig,ax=plt.subplots(figsize=[8,6])
    plt.subplots_adjust(bottom=.2)
    
    ds = xr.DataArray(obs_dataset.values,
    coords={'time': obs_dataset.times,
    'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
    dims=["time", "lat", "lon"])
    
    #ds=ds.groupby('time.year').sum()
    ds=ds.mean(dim=("lat", "lon"))
    #it is time series data of obs
    
    sd1=ds.std(skipna=None)
   
    tay=[]
    T=[]
    cpi=[]
    ss=[]
    cek=[]
    d=[]
    r0=1
    for i in np.arange(len(model_datasets)):
        #i=10
        print(model_names[i])
        dsi = xr.DataArray(model_datasets[i].values,
        coords={'time': obs_dataset.times,
        'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
        dims=["time", "lat", "lon"])  
        
        #dsi = dsi.groupby('time.year').sum()
        dsi=dsi.mean(dim=("lat", "lon")) 
        #it is time series data of model
        
       
    
        c=xr.corr(dsi,ds)
        #if i==0:
        #    print(ds.data)
        #    print(dsi.data)
        
        sd2=dsi.std(skipna=True)
        s=sd2/sd1
     
        Tay= (4*(1+c))/(((s+1/s)**2)*((1+r0)))
        #print(s.data, c.data,ttm.data)
        tay.append(np.round(Tay.data,4))
        
        
        vr= ds.var() #mstats.tvar(ds.data)
        vf= dsi.var() #vf=mstats.tvar(dsi.data)
        #(ma.mean((calc_bias(target_array, reference_array))**2))**0.5
        #rmse=sqrt(mean(m-o)^2)
        rmse=(((dsi-ds)**2).mean())**0.5
        b=dsi.mean()-ds.mean()
        mse=rmse.data**2
        b=b.data
        #print(r,b) #,vr,vf)
        #tambahkan Tian
        #Tian=(1+c)/2*(1-(r**2/(b**2+vr+vf)))
        Tian=(1+c)/2*(1-(mse/(b**2+vr+vf)))
        T.append(np.round(Tian.data,4))
        
        CPI=math.exp(-1*0.5*(b)**2/vf)
        cpi.append(np.round(CPI,4))
        
        so=sd2
        mm=dsi.mean()
        mo=ds.mean()
        SS=c**2 -(c-s)**2 -((mm-mo)/so)**2
        cek.append(np.round(SS.data,4))
        
        #SS=1/(SS+11)
        ss.append(np.round(SS.data,4))
        
        D=1/math.sqrt(s**2+1-2*s*c)
        d.append(np.round(D,4))
        
        
        #ax.scatter(i,Tian, label = model_names[i],marker='o',color='blue')
        #ax.scatter(i,Tay, label = model_names[i],marker='o',color='black')
    
    tay=np.array(tay)
    
    #normalize 0 to 1   
    #for mak <1
    T=np.array(T)
    
    mak=abs(T.min())
    T=T+mak
    mak=abs(T.max())
    T=T/mak
    
    cpi=np.array(cpi)
    
    #for mak>1
    ss=np.array(ss)
    
    mak=abs(ss.min())
    ss=(ss+mak)/mak
    mak=abs(ss.max())
    ss=ss/mak
    
    
    d=np.array(d)
    mak=abs(d.max())
    d=d/mak
    
    model_names=np.array(model_names)
    
    if sort:
        sorter= np.argsort(tay)[::-1]
        tay=(tay)[sorter]
        T=(T)[sorter]
        cpi=(cpi)[sorter]
        ss=(ss)[sorter]
        d=(d)[sorter]
        model_names=(model_names)[sorter]
        
    #import xarray as xr
    dpi = xr.open_dataset("d:/Cordex/ClimWIP-Brunner_etal_2020_ESD/data/DJF_gpcp.nc")
      
    pi=dpi.weights_q.data 
    
    ax.plot(model_names,tay, color='k', marker='o', lw=1, label = 'Taylor')
    ax.plot(model_names,T, marker='o',lw=1, label = 'Tian')
    ax.plot(model_names,cpi, marker='o',lw=1, label = 'CPI')
    ax.plot(model_names,ss, marker='o',lw=1, label = 'SS')
    ax.plot(model_names,pi, marker='o',lw=1, label = 'PI')
    #ax.plot(model_names,d, marker='o',lw=1, label = 'D')
    
    print('Taylor=', tay)
    print('Tian=', T)
    print('CPI=', cpi)
    print('SS=', ss)
    print('PI=', pi)

    plt.legend(bbox_to_anchor=(1, .4), loc='best', prop={'size':10}, frameon=False) 
    #print('ts_temporal monthly',Tm)
    #ax.set_xticks(np.arange(len(model_datasets)))
    #names[0]=''
    #ax.set_xticklabels(model_names,fontsize=8.5)
    plt.xticks(rotation=45)
    plt.ylabel("Model's performance score")
    
    
    file_name='_ac_taylor'
    plt.savefig(workdir+reg+file_name,dpi=300,bbox_inches='tight')
    plt.show()
    
    
    return #Tm, T   
 
 
 
 
def ts(obs_dataset, obs_name, model_datasets, model_names, workdir):
    #Plot time series obs and models for domain in the input
    
    ds = xr.DataArray(obs_dataset.values,
    coords={'time': obs_dataset.times,
    'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
    dims=["time", "lat", "lon"])
    
    #ds = ds.groupby('time.month').mean() 
    #ds = ds.groupby('time.year').sum() 
    #ds = ds.groupby('time.dayofyear').mean() # datanya bulanan jadi error
    #print('xx=',ds.shape, ds.lat.shape) 
    #print(ds.max())
    ds=ds.mean(dim=("lat", "lon"))
    #ds1=ds.stack(z=("year"))
    #sd1=ma.std(ds)
    
    #plot ts hujan sum() SEA
    fig=plt.figure(figsize=[8,6])
    fig.subplots_adjust(right=.7)
    
    plt.plot(ds.year, ds.values, color='black', lw=2, label = obs_name)
    #plt.plot(ds.time, ds.values, color='black', lw=2, label = 'GPCP')
  
    for i in np.arange(len(model_datasets)):
        #i=10
        #print(model_names[i])
        dsi = xr.DataArray(model_datasets[i].values,
        coords={'time': obs_dataset.times,
        'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
        dims=["time", "lat", "lon"])  
        #print(dsi.max())
        #print(model_datasets[i].values.max())
        #print(dsi.mean())
        
        
        #by year
        #dsi = dsi.groupby('time.year').sum() #year,lat,lon,value
        #print(dsi)
        #mean kan by year tadi = rata2 tahunan 
        #jika 
        dsi=dsi.mean(dim=("lat", "lon")) #year, value (1 y maka 1 v)
        #print(dsi)
        #exit()
        #result2 = pd.DataFrame([dsi.values],index=[model_names[i]]).T
        
        if i==0 or i==10: lws=2
        else: lws=1
        plt.plot(ds.year, dsi.values, lw=lws, label = model_names[i])
    #exit()
    plt.legend(bbox_to_anchor=(1, .6), loc='best', prop={'size':8.5}, frameon=False) 
    plt.ylabel('Mean precipitation (mm/year)')
    #plt.ylabel('Mean precipitation (mm/month)')
    plt.show()
    plt.savefig(workdir+reg+'_ts')
    
def ts_c(obs_dataset, obs_name, model_datasets, model_names, workdir):
    #Plot time series obs and models for domain in the input
    print('ts_c...')
    ds = xr.DataArray(obs_dataset.values,
    coords={'time': obs_dataset.times,
    'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
    dims=["time", "lat", "lon"])
    ds = ds.groupby('time.year').mean()
    ds=ds.mean(dim=("lat", "lon"))
   
    fig=plt.figure(figsize=[8,6])
    fig.subplots_adjust(right=.8)
    
    plt.plot(ds.year, ds.values, label = obs_name)
    #plt.plot(np.arange(len(ds.time)), ds.values, label = obs_name)
    
    for i in np.arange(len(model_datasets)):
        #i=10
        print(model_names[i])
        dsi = xr.DataArray(model_datasets[i].values,
        coords={'time': obs_dataset.times,
        'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
        dims=["time", "lat", "lon"])  
        
        dsi = dsi.groupby('time.year').mean()
        dsi=dsi.mean(dim=("lat", "lon")) #year, value (1 y maka 1 v)
        if model_names[i]=='MME': lw=4; 
        else: 
            lw=1
        #plt.plot(np.arange(len(ds.time)), dsi.values, label = model_names[i])
        plt.plot(dsi.year, dsi.values, lw=lw, label = model_names[i])

    plt.legend(bbox_to_anchor=(1, .6), loc='best', prop={'size':8.5}, frameon=False) 
    plt.ylabel('Mean precipitation (mm/month)')
    plt.show()
    #plt.savefig(workdir+reg+'_ts')

    
def ts_black(obs_dataset, obs_name, model_datasets, model_names, workdir):
    #Plot time series obs and models for domain in the input
    
    ds = xr.DataArray(obs_dataset.values,
    coords={'time': obs_dataset.times,
    'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
    dims=["time", "lat", "lon"])
    
    #ds = ds.groupby('time.month').mean() 
    ds = ds.groupby('time.year').sum() 
    #ds = ds.groupby('time.dayofyear').mean() # datanya bulanan jadi error
    #print('xx=',ds.shape, ds.lat.shape) 
    print(ds.max())
    ds=ds.mean(dim=("lat", "lon"))
    #ds1=ds.stack(z=("year"))
    #sd1=ma.std(ds)
    
    #plot ts hujan sum() SEA
    fig=plt.figure(figsize=[8,6])
    fig.subplots_adjust(right=.7)
    
    plt.plot(ds.year, ds.values, color='black', lw=3, label = obs_name)
    #plt.plot(ds.time, ds.values, color='black', lw=2, label = 'GPCP')
    mm=['*','+','x','>','o', 's']
    mm2=['-.',':','--']
    
    cc='black'
    for i in np.arange(len(model_datasets)):
        #i=10
        #print(model_names[i])
        dsi = xr.DataArray(model_datasets[i].values,
        coords={'time': obs_dataset.times,
        'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
        dims=["time", "lat", "lon"])  
        #print(dsi.max())
        #print(model_datasets[i].values.max())
        #print(dsi.mean())
        
        
        #by year
        dsi = dsi.groupby('time.year').sum() #year,lat,lon,value
        #print(dsi)
        #mean kan by year tadi = rata2 tahunan 
        #jika 
        dsi=dsi.mean(dim=("lat", "lon")) #year, value (1 y maka 1 v)
        #print(dsi)
        #exit()
        #result2 = pd.DataFrame([dsi.values],index=[model_names[i]]).T
        lws=1.5
        if i==0: 
           lws=3
           cc='blue'
        if i==10: 
           lws=3 
           cc='red'
        
        if 0<i<7:
            plt.plot(ds.year, dsi.values, color='k', marker=mm[i-1],lw=lws, label = model_names[i])
        elif 6<i<10:
            plt.plot(ds.year, dsi.values, color='k',lw=lws, linestyle=mm2[i-7],label = model_names[i])
        else:
            plt.plot(ds.year, dsi.values, color=cc,lw=lws, label = model_names[i])
    #exit()
    plt.legend(bbox_to_anchor=(1, .6), loc='best', prop={'size':8.5}, frameon=False, handlelength=3.5) 
    plt.ylabel('Mean precipitation (mm/year)')
    #plt.ylabel('Mean precipitation (mm/month)')
    # Menghilangkan garis sumbu
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    plt.show()
    plt.savefig(workdir+reg+'_ts')
    
    
def ts_black_5obs(obs_dataset, obs_name, model_datasets, model_names, workdir):
    #Plot time series obs and models for domain in the input
    
    #plot ts hujan sum() SEA
    fig=plt.figure(figsize=[8,6])
    fig.subplots_adjust(right=.7)
    
    '''
    ds = xr.DataArray(obs_dataset.values,
    coords={'time': obs_dataset.times,
    'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
    dims=["time", "lat", "lon"])
    
    #ds = ds.groupby('time.month').mean() 
    ds = ds.groupby('time.year').sum() 
 
    print(ds.max())
    ds=ds.mean(dim=("lat", "lon"))
  
    # Calculate the trend line (linear regression)
    rainfall_values = ds.values
    trend = np.polyfit(ds.year, rainfall_values, 1)
    trend_line = np.polyval(trend, ds.year)
    
    
    plt.plot(ds.year, ds.values, color='black', lw=2, label = obs_name)
    #plt.plot(ds.time, ds.values, color='black', lw=2, label = 'GPCP')
    
    plt.plot(ds.year, trend_line, ls='--', color='red') #, label='Trend Line')
    '''
    mm=['','>','s','x','*','+', 'o']
    mm2=['-','-.',':','--','--']
    
   
    for i in np.arange(len(model_datasets)):
        #i=10
        #print(model_names[i])
        dsi = xr.DataArray(model_datasets[i].values,
        coords={'time': obs_dataset.times,
        'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
        dims=["time", "lat", "lon"])  
       
        
        #by year 
        #jika kedua nya mean skala 5-10 
        #jika atas=sum bawah mean skala 14-20 
        #jika keduanya sum ==> hasil skalanya ribuan mm/year
        dsi = dsi.groupby('time.year').sum() #year,lat,lon,value
      
        dsi=dsi.sum(dim=("lat", "lon")) #year, value (1 y maka 1 v)
        
        # Calculate the trend line (linear regression)
        rainfall_values = dsi.values
        trend = np.polyfit(time, rainfall_values, 1)
        trend_line = np.polyval(trend, time)
   
        plt.plot(time, dsi.values, color='k', marker=mm[i],label = model_names[i])
        plt.plot(time, trend_line, ls='--',label = '   "') #model_names[i])
    
    plt.legend(bbox_to_anchor=(1, .5), loc='best', prop={'size':8.5}, frameon=False, handlelength=3.5) 
    plt.ylabel('Mean precipitation (mm/year)')
    #plt.ylabel('Mean precipitation (mm/month)')
    plt.show()
    plt.savefig(workdir+reg+'_ts2')
    
    
def ts2(obs_dataset, obs_name, model_datasets, model_names, workdir):
    #Plot time series obs and models for domain in the input
    
    ds = xr.DataArray(obs_dataset.values,
    coords={'time': obs_dataset.times,
    'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
    dims=["time", "lat", "lon"])
    
    #ds = ds.groupby('time.month').mean() 
    #ds = ds.groupby('time.year').sum() 
    #ds = ds.groupby('time.dayofyear').mean() # datanya bulanan jadi error
    #print('xx=',ds.shape, ds.lat.shape) 
    ds = ds.groupby('time.season')
    ds=ds['DJF'].sum(dim=("lat", "lon"))
    #print(ds)
    
    #ds=ds.mean(dim=("lat", "lon"))
    #ds1=ds.stack(z=("year"))
    #sd1=ma.std(ds)
    
    #plot ts hujan sum() SEA
    fig=plt.figure(figsize=[8,6])
    fig.subplots_adjust(right=.7)
    
    plt.plot(ds.time, ds.values, color='black', lw=2, label = obs_name)
    #plt.plot(ds.time, ds.values, color='black', lw=2, label = 'GPCP')
  
    for i in np.arange(len(model_datasets)):
        #i=10
        #print(model_names[i])
        dsi = xr.DataArray(model_datasets[i].values,
        coords={'time': obs_dataset.times,
        'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
        dims=["time", "lat", "lon"])  
   
     
        dsi = dsi.groupby('time.season')
      
        dsi=dsi['DJF'].sum(dim=("lat", "lon"))
    
        
        if i==0 or i==10: lws=2
        else: lws=1
        
        plt.plot(ds.time, dsi.values, lw=lws, label = model_names[i])
    #exit()
    plt.legend(bbox_to_anchor=(1, .6), loc='best', prop={'size':8.5}, frameon=False) 
    plt.ylabel('Precipitation (mm/season)')
    #plt.ylabel('Mean precipitation (mm/month)')
    plt.show()
    plt.savefig(workdir+reg+'_ts')
    

def annual_cycle(obs_dataset, obs_name, model_datasets, model_names, workdir):
    #Plot time series obs and models for domain in the input
    #annual tepat jika domain tidak luas misal SEA
    #cocok untuk sub domain yang karakteristik hujannya sama/hampir sama
    #ini bisa diperoleh dari k-means clustering
    
    ds = xr.DataArray(obs_dataset.values,
    coords={'time': obs_dataset.times,
    'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
    dims=["time", "lat", "lon"])
    
    ds = ds.groupby('time.month').mean()
    ds=ds.mean(dim=("lat", "lon"))
    
    x_tick=['J','F','M','A','M','J','J','A','S','O','N','D']
    
    fig, ax = plt.subplots(nrows=1, ncols=1,figsize=(16,6))
    fig.subplots_adjust(right=.7)
    
    plt.plot(ds.month, ds.values, color='black', lw=2, label = obs_name)
    #plt.plot(ds.time, ds.values, color='black', lw=2, label = 'GPCP')
    
    #supported values for ls are '-', '--', '-.', ':', 'None', ' ', '', 
    #'solid', 'dashed', 'dashdot', 'dotted'
    ls = ['-', '--','-.',':']
    #dashList = [(5,2),(0,5),(2,5),(5,5),(5,5),(5,10),(5,10)]
    for i in np.arange(len(model_datasets)):
        #i=10
        #print(model_names[i])
        dsi = xr.DataArray(model_datasets[i].values,
        coords={'time': obs_dataset.times,
        'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
        dims=["time", "lat", "lon"])  
       
        dsi = dsi.groupby('time.month').mean()
        dsi=dsi.mean(dim=("lat", "lon")) 
        
        
        if i==0: 
            lws=2
            plt.plot(ds.month, dsi.values,lw=lws, label = model_names[i])
        if 2<=i<=5: 
            lws=1.5
            #print (ls[i])
            plt.plot(ds.month, dsi.values, ls=ls[i-2],
            #dashes=dashList[i], 
            lw=lws, label = model_names[i])
        if i>5:
            lws=1.5
            plt.plot(ds.month, dsi.values,lw=lws, label = model_names[i])
    
        #plt.plot(ds.month, dsi.values,lw=lws, label = model_names[i])
    ax.set_xticks(np.arange(12)+1)
    ax.set_xticklabels(x_tick, fontsize=8)
    plt.legend(bbox_to_anchor=(1, .4), loc='best', prop={'size':8.5}, frameon=False) 
    plt.ylabel('Precipitation (mm/month)')
  
    plt.savefig(workdir+reg+'_annual_cycle')
    plt.show()
    
def annual_cycle_sum2R(obs_dataset, obs_name, model_datasets, model_names, workdir):
    #Plot time series obs and models for domain in the input
    # 5obs
    x_tick=['J','F','M','A','M','J','J','A','S','O','N','D']
    
    fig, ax = plt.subplots(nrows=1, ncols=1,figsize=(16,6))
    fig.subplots_adjust(right=.7)
    
   
    mm=['','>','s','x','*','+', 'o']
    for i in np.arange(len(model_datasets)):
        #i=10
        print(model_names[i])
        dsi = xr.DataArray(model_datasets[i].values,
        coords={'time': obs_dataset.times,
        'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
        dims=["time", "lat", "lon"])  
        
        #R1:[-6, 0, 99, 106.5]
        #R2:[0, 6, 95, 104]
        
                    
        dr1 = dsi.where(
                    (dsi.lat > 0) & (dsi.lat < 6) & 
                    (dsi.lon > 95) & (dsi.lon < 104), drop=True)
                    
        dr2 = dsi.where(
                    (dsi.lat > -6) & (dsi.lat < 0) & 
                    (dsi.lon > 99) & (dsi.lon < 106.5), drop=True)
       
        dm1 = dr1.groupby('time.month').mean()
        dm1=dm1.mean(dim=("lat", "lon")) 
        
        dm2 = dr2.groupby('time.month').mean()
        dm2=dm2.mean(dim=("lat", "lon")) 
        
        print(dr1.max(), dr2.max())
        
        #ax.plot(dm1.month, dm1.values, color='black', lw=2, label = 'R1_'+model_names[i])
        #ax.plot(dm2.month, dm2.values, color='blue', lw=2, label = 'R2_'+model_names[i])
        
        ax.plot(dm1.month, dm1.values, marker=mm[i], color='black', lw=2, label = model_names[i])
        ax.plot(dm2.month, dm2.values, marker=mm[i], color='blue', lw=2, label = '   "')
        #plt.plot(ds.month, dsi.values,lw=lws, label = model_names[i])
    ax.set_xticks(np.arange(12)+1)
    ax.set_xticklabels(x_tick, fontsize=8)
    plt.legend(bbox_to_anchor=(1, .5), loc='best', prop={'size':8.5}, frameon=False) 
    plt.ylabel('Mean rainfall (mm/month)')
  
    plt.savefig(workdir+reg+'_annual_cycle')
    plt.show()

def metrik_enso(obs_dataset, obs_name, model_datasets, model_names, workdir):
    
    #bobot enso pisahkan dengan iod
    #annual
    #bulanan ? 3bulanan?
    
   
    ei=[-0.08,1.87,-0.53,-0.90,-0.62,0.79,0.83,-1.33,-0.25,0.22,1.32,-0.24,
           0.15,1.05,-0.67,-0.45,2.15,-1.28,-1.15,-0.42,-0.02,0.66,0.34,0.49]
    enso=[0.09,1.55,-0.47,-0.68,-0.33,0.93,0.63,-1.23,0.02,0.38,1.11,0.23,0.21,
          0.57,-0.65,-0.08,1.63,-1.22,-1.18,-0.49,0.00,0.64,0.28,0.53]
    iod=[-0.17,0.32,-0.07,-0.23,-0.29,-0.14,0.20,-0.10,-0.27,-0.16,0.21,-0.47,
         -0.06,0.48,-0.02,-0.36,0.52,-0.07,0.03,0.07,-0.02,0.02,0.06,-0.04]
         
    ds = xr.DataArray(obs_dataset.values,
    coords={'time': obs_dataset.times,
    'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
    dims=["time", "lat", "lon"])
    
            
    ds = ds.groupby('time.year').sum() 
    
    lat_min = ds.lat.min()
    lat_max = ds.lat.max()
    lon_min = ds.lon.min()
    lon_max = ds.lon.max()
   
    
    fig, ax = plt.subplots(nrows=2, ncols=4,figsize=(16,6))
    m = Basemap(ax=ax[0,0], projection ='cyl', llcrnrlat = lat_min, urcrnrlat = lat_max,
            llcrnrlon = lon_min, urcrnrlon = lon_max, resolution = 'l', fix_aspect=True)
    m.drawcoastlines(linewidth=1)
    m.drawcountries(linewidth=1)
    m.drawstates(linewidth=0.5, color='w')
    
    x,y = np.meshgrid(ds.lon, ds.lat)
    
    from scipy.stats import pearsonr
    
    map1=ma.zeros((len(ds.lat),len(ds.lon)))

    for i in np.arange(len(ds.lat)):
        for j in np.arange(len(ds.lon)):
            dd=ds[:,i,j].values
            R=pearsonr(enso, dd)[0]*-1
            map1[i,j]=R
    
    max = ax[0,0].contourf(x,y,map1)
    ax[0,0].set_title('GPCP')
    ax[0,0].set_yticks([-10,0,10,20])
    #ax[0,0].set_xticks([90,100,110,120,130,140])
    
    #fig.savefig(workdir+file_name,dpi=600,bbox_inches='tight')
    #exit()
    
    # ini ?? khusus zonal jika ingin obs=2 dan MMEW not included
    #model_datasets=np.delete(model_datasets,[1, -1])
    #model_names=np.delete(model_names,[1, -1])
    model_datasets=np.delete(model_datasets,[-1])
    model_names=np.delete(model_names,[-1])
    
    
    for i in np.arange(7):
        print (i)
        dsi = xr.DataArray(model_datasets[i].values,
        coords={'time': obs_dataset.times,
        'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
        dims=["time", "lat", "lon"])  
        
        dsi = dsi.groupby('time.year').sum() 
        
        if i<3:
            m = Basemap(ax=ax[0,1+i], projection ='cyl', llcrnrlat = lat_min, urcrnrlat = lat_max,
                llcrnrlon = lon_min, urcrnrlon = lon_max, resolution = 'l', fix_aspect=True)
            m.drawcoastlines(linewidth=1)
            m.drawcountries(linewidth=1)
            m.drawstates(linewidth=0.5, color='w')
        
            x,y = np.meshgrid(ds.lon, ds.lat)
        
            from scipy.stats import pearsonr
        
            map1=ma.zeros((len(ds.lat),len(ds.lon)))

            for ii in np.arange(len(ds.lat)):
                for jj in np.arange(len(ds.lon)):
                    dd=dsi[:,ii,jj].values
                    R=pearsonr(enso, dd)[0]*-1
                    map1[ii,jj]=R
        
            max = ax[0,1+i].contourf(x,y,map1)
            ax[0,1+i].set_title(model_names[i])
            #ax[0,0].set_yticks([-10,0,10,20])
            #ax[0,1+i].set_xticks([90,100,110,120,130,140])
            
        else:
            m = Basemap(ax=ax[1,i-3], projection ='cyl', llcrnrlat = lat_min, urcrnrlat = lat_max,
                llcrnrlon = lon_min, urcrnrlon = lon_max, resolution = 'l', fix_aspect=True)
            m.drawcoastlines(linewidth=1)
            m.drawcountries(linewidth=1)
            m.drawstates(linewidth=0.5, color='w')
        
            x,y = np.meshgrid(ds.lon, ds.lat)
        
            from scipy.stats import pearsonr
        
            map1=ma.zeros((len(ds.lat),len(ds.lon)))

            for ii in np.arange(len(ds.lat)):
                for jj in np.arange(len(ds.lon)):
                    dd=dsi[:,ii,jj].values
                    R=pearsonr(enso, dd)[0]*-1
                    map1[ii,jj]=R
        
            max = ax[1,i-3].contourf(x,y,map1)
            ax[1,i-3].set_title(model_names[i])
            ax[1,0].set_yticks([-10,0,10,20])
            ax[1,i-3].set_xticks([90,100,110,120,130,140])
    
    
    plt.subplots_adjust(hspace=.2,wspace=.01)
    cax = fig.add_axes([0.91, 0.5, 0.015, 0.4])
    plt.colorbar(max, cax = cax) 
    
    file_name='Corr_Enso'
    fig.savefig(workdir+file_name,dpi=600,bbox_inches='tight')
       
def metrik_iod(obs_dataset, obs_name, model_datasets, model_names, workdir):
    
    iod=[-0.17,0.32,-0.07,-0.23,-0.29,-0.14,0.20,-0.10,-0.27,-0.16,0.21,-0.47,
         -0.06,0.48,-0.02,-0.36,0.52,-0.07,0.03,0.07,-0.02,0.02,0.06,-0.04]
         
    ds = xr.DataArray(obs_dataset.values,
    coords={'time': obs_dataset.times,
    'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
    dims=["time", "lat", "lon"])
    
            
    ds = ds.groupby('time.year').sum() 
    
    lat_min = ds.lat.min()
    lat_max = ds.lat.max()
    lon_min = ds.lon.min()
    lon_max = ds.lon.max()
   
    
    fig, ax = plt.subplots(nrows=2, ncols=4,figsize=(16,6))
    m = Basemap(ax=ax[0,0], projection ='cyl', llcrnrlat = lat_min, urcrnrlat = lat_max,
            llcrnrlon = lon_min, urcrnrlon = lon_max, resolution = 'l', fix_aspect=True)
    m.drawcoastlines(linewidth=1)
    m.drawcountries(linewidth=1)
    m.drawstates(linewidth=0.5, color='w')
    
    x,y = np.meshgrid(ds.lon, ds.lat)
    
    from scipy.stats import pearsonr
    
    map1=ma.zeros((len(ds.lat),len(ds.lon)))

    for i in np.arange(len(ds.lat)):
        for j in np.arange(len(ds.lon)):
            dd=ds[:,i,j].values
            R=pearsonr(iod, dd)[0]*-1
            map1[i,j]=R
    
    max = ax[0,0].contourf(x,y,map1)
    ax[0,0].set_title('GPCP')
    ax[0,0].set_yticks([-10,0,10,20])
    #ax[0,0].set_xticks([90,100,110,120,130,140])
    
    #fig.savefig(workdir+file_name,dpi=600,bbox_inches='tight')
    #exit()
    
    # ini ?? khusus zonal jika ingin obs=2 dan MMEW not included
    #model_datasets=np.delete(model_datasets,[1, -1])
    #model_names=np.delete(model_names,[1, -1])
    model_datasets=np.delete(model_datasets,[-1])
    model_names=np.delete(model_names,[-1])
    
    
    for i in np.arange(7):
        print (i)
        dsi = xr.DataArray(model_datasets[i].values,
        coords={'time': obs_dataset.times,
        'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
        dims=["time", "lat", "lon"])  
        
        dsi = dsi.groupby('time.year').sum() 
        
        if i<3:
            m = Basemap(ax=ax[0,1+i], projection ='cyl', llcrnrlat = lat_min, urcrnrlat = lat_max,
                llcrnrlon = lon_min, urcrnrlon = lon_max, resolution = 'l', fix_aspect=True)
            m.drawcoastlines(linewidth=1)
            m.drawcountries(linewidth=1)
            m.drawstates(linewidth=0.5, color='w')
        
            x,y = np.meshgrid(ds.lon, ds.lat)
        
            from scipy.stats import pearsonr
        
            map1=ma.zeros((len(ds.lat),len(ds.lon)))

            for ii in np.arange(len(ds.lat)):
                for jj in np.arange(len(ds.lon)):
                    dd=dsi[:,ii,jj].values
                    R=pearsonr(iod, dd)[0]*-1
                    map1[ii,jj]=R
        
            max = ax[0,1+i].contourf(x,y,map1)
            ax[0,1+i].set_title(model_names[i])
            #ax[0,0].set_yticks([-10,0,10,20])
            #ax[0,1+i].set_xticks([90,100,110,120,130,140])
            
        else:
            m = Basemap(ax=ax[1,i-3], projection ='cyl', llcrnrlat = lat_min, urcrnrlat = lat_max,
                llcrnrlon = lon_min, urcrnrlon = lon_max, resolution = 'l', fix_aspect=True)
            m.drawcoastlines(linewidth=1)
            m.drawcountries(linewidth=1)
            m.drawstates(linewidth=0.5, color='w')
        
            x,y = np.meshgrid(ds.lon, ds.lat)
        
            from scipy.stats import pearsonr
        
            map1=ma.zeros((len(ds.lat),len(ds.lon)))

            for ii in np.arange(len(ds.lat)):
                for jj in np.arange(len(ds.lon)):
                    dd=dsi[:,ii,jj].values
                    R=pearsonr(iod, dd)[0]*-1
                    map1[ii,jj]=R
        
            max = ax[1,i-3].contourf(x,y,map1)
            ax[1,i-3].set_title(model_names[i])
            ax[1,0].set_yticks([-10,0,10,20])
            ax[1,i-3].set_xticks([90,100,110,120,130,140])
    
    
    plt.subplots_adjust(hspace=.2,wspace=.01)
    cax = fig.add_axes([0.91, 0.5, 0.015, 0.4])
    plt.colorbar(max, cax = cax) 
    
    file_name='Corr_IOD'
    fig.savefig(workdir+file_name,dpi=600,bbox_inches='tight')
    
    

def per_metrics(obs_dataset, obs_name, model_datasets, model_names):
  
    
    #ds = xr.DataArray(obs_dataset) #disini tidak bisa langsung
    #print('obs_dataset.times=', obs_dataset.times)
    
    ds = xr.DataArray(obs_dataset.values,
    coords={'time': obs_dataset.times,
    'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
    dims=["time", "lat", "lon"])
    
    ds=ds.mean(dim="time")
    #https://permetrics.readthedocs.io/en/latest/
    # #Tes PerMetrics 1.3.1
    from permetrics.regression import RegressionMetric
    evaluator = RegressionMetric()
    # y_true=np.array(obs_dataset)
    # y_pred=np.array(model_datasets[0])
    # rmse_1 = evaluator.RMSE(y_true, y_pred)
    # print(f"RMSE: {rmse_1}")
    # exit()
    
    for i in np.arange(len(model_datasets)):
        dsi = xr.DataArray(model_datasets[i].values,
        coords={'time': obs_dataset.times,
        'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
        dims=["time", "lat", "lon"])
        
        dsi=dsi.mean(dim="time")
        #Tes PerMetrics 1.3.1
        y_true=np.array(ds.stack(z=("lat", "lon")))
        y_pred=np.array(dsi.stack(z=("lat", "lon")))
        print(y_true.shape)
        print(y_pred.shape)
        rmse_1 = evaluator.RMSE(y_true, y_pred)
        rmse_2 = evaluator.R(y_true, y_pred)
        rmse_3 = evaluator.NNSE(y_true, y_pred)
        print(f"RMSE R NSE: {rmse_1}, {rmse_2}, {rmse_3}")
        #TypeError: y_true and y_pred need to be a list, tuple or np.array.
        
def Map_plot_bias_of_multiyear_climatology2(obs_dataset, obs_name, model_datasets, model_names,
                                      workdir):
    '''Draw maps of observed multi-year climatology and biases of models"'''

    # calculate climatology of observation data
    #temporal mean per grid
    obs_clim = utils.calc_temporal_mean(obs_dataset)
    # determine the metrics
    map_of_bias = metrics.TemporalMeanBias()

    # create the Evaluation object
    bias_evaluation = Evaluation(obs_dataset, # Reference dataset for the evaluation
                                 model_datasets, # list of target datasets for the evaluation
                                 [map_of_bias, map_of_bias])
    # run the evaluation (bias calculation)
    bias_evaluation.run() 

    rcm_bias = bias_evaluation.results[0]
    

    #fig = plt.figure()

    lat_min = obs_dataset.lats.min()
    lat_max = obs_dataset.lats.max()
    lon_min = obs_dataset.lons.min()
    lon_max = obs_dataset.lons.max()

    string_list = list(string.ascii_lowercase) 
 
    
    #############################
    #tambahan title + bias on top 
    ###############################
    fig, axes = plt.subplots(nrows=3, ncols=5,figsize=(14,6))
   
    m = Basemap(ax=axes[0,0], projection ='cyl', llcrnrlat = lat_min, urcrnrlat = lat_max,
            llcrnrlon = lon_min, urcrnrlon = lon_max, resolution = 'l', fix_aspect=False)
    if obs_dataset.lons.ndim == 1 and obs_dataset.lats.ndim == 1:
        lons, lats = np.meshgrid(obs_dataset.lons, obs_dataset.lats)
    if obs_dataset.lons.ndim == 2 and obs_dataset.lats.ndim == 2:
        lons = obs_dataset.lons
        lats = obs_dataset.lats
    x,y = m(lons, lats)
    m.drawcoastlines(linewidth=1)
    m.drawcountries(linewidth=1)
    m.drawstates(linewidth=0.5, color='w')
    #plot obs map
    max1 = m.contourf(x,y,obs_clim,levels = plotter._nice_intervals(obs_dataset.values, 10), extend='both',cmap='rainbow')
    #ax.annotate('(a) \n' + obs_name,xy=(lon_min, lat_min))
    #ax.annotate(obs_name,xy=(lon_min, lat_min))
   
    clevs = plotter._nice_intervals(rcm_bias, 11)
    for imodel in np.arange(len(model_datasets)):
        if imodel<4:
            
            m = Basemap(ax=axes[0,1+imodel], projection ='cyl', llcrnrlat = lat_min, urcrnrlat = lat_max,
                    llcrnrlon = lon_min, urcrnrlon = lon_max, resolution = 'l', fix_aspect=False )
            m.drawcoastlines(linewidth=1)
            m.drawcountries(linewidth=1)
            m.drawstates(linewidth=0.5, color='w')
            #plot model
            #bias_mean = np.mean(rcm_bias[imodel,:])
            #print('bias=',bias_mean)
            max = m.contourf(x,y,rcm_bias[imodel,:],levels = clevs, extend='both', cmap='RdBu_r')
            #ax.annotate('('+string_list[imodel+1]+')  \n '+model_names[imodel],xy=(lon_min, lat_min))
            #ax.annotate(model_names[imodel],xy=(lon_min, lat_min))
            bias_mean = np.mean(rcm_bias[imodel,:])
            axes[0,1+imodel].set_title(model_names[imodel]+' [b='+'%.2f'%bias_mean+']',fontsize=8)
        if 3<imodel<9:
           
            m = Basemap(ax=axes[1,1+imodel-5], projection ='cyl', llcrnrlat = lat_min, urcrnrlat = lat_max,
                    llcrnrlon = lon_min, urcrnrlon = lon_max, resolution = 'l', fix_aspect=False )
            m.drawcoastlines(linewidth=1)
            m.drawcountries(linewidth=1)
            m.drawstates(linewidth=0.5, color='w')
            #plot model
            max = m.contourf(x,y,rcm_bias[imodel,:],levels = clevs, extend='both', cmap='RdBu_r')
            #ax.annotate('('+string_list[imodel+1]+')  \n '+model_names[imodel],xy=(lon_min, lat_min))
            #ax.annotate(model_names[imodel],xy=(lon_min, lat_min))
            bias_mean = np.mean(rcm_bias[imodel,:])
            axes[1,1+imodel-5].set_title(model_names[imodel]+' [b='+'%.2f'%bias_mean+']',fontsize=8)
        if 8<imodel<14:
            
            m = Basemap(ax=axes[2,1+imodel-10], projection ='cyl', llcrnrlat = lat_min, urcrnrlat = lat_max,
                    llcrnrlon = lon_min, urcrnrlon = lon_max, resolution = 'l', fix_aspect=False )
            m.drawcoastlines(linewidth=1)
            m.drawcountries(linewidth=1)
            m.drawstates(linewidth=0.5, color='w')
            #plot model
            max = m.contourf(x,y,rcm_bias[imodel,:],levels = clevs, extend='both', cmap='RdBu_r')
            #ax.annotate('('+string_list[imodel+1]+')  \n '+model_names[imodel],xy=(lon_min, lat_min))
            #ax.annotate(model_names[imodel],xy=(lon_min, lat_min))
            bias_mean = np.mean(rcm_bias[imodel,:])
            axes[2,1+imodel-10].set_title(model_names[imodel]+' [b='+'%.2f'%bias_mean+']',fontsize=8)
            axes[2,1+imodel-10].set_xticks([100,120,140])
            axes[2,1+imodel-10].set_xticklabels(['100E','120E','140E'], fontsize=7)
            axes[2,1+imodel-10].xaxis.set_tick_params(labelsize=7)
     
        
    axes[0,0].set_title(obs_name, fontsize=8)
    for i in [0,1,2]:
        axes[i,0].set_yticks([-10,0,10,20])   
        axes[i,0].set_yticklabels(['10S','EQ','10N','20N'], fontsize=7)
        axes[i,0].yaxis.set_tick_params(labelsize=7)
    
    #The dimensions [left, bottom, width, height] of the new axes
    cax1 = fig.add_axes([0.099, 0.97, 0.3, 0.02])
    plt.colorbar(max1, cax = cax1, orientation='horizontal' )
    
    cax = fig.add_axes([0.91, 0.4, 0.01, 0.5])
    #cax = fig.add_axes([0.5, 0.97, 0.6, 0.02])
    plt.colorbar(max, cax = cax)
    #plt.colorbar(cax).ax.set_title('mm')

    plt.subplots_adjust(hspace=.25,wspace=.05)
    file_name=workdir+reg+'bias'
    plt.show()
    fig.savefig(file_name,dpi=600,bbox_inches='tight')        
        
def Map_plot_bias_of_multiyear_climatology(obs_dataset, obs_name, model_datasets, model_names,
                                      file_name, row, column, map_projection=None):
    '''Draw maps of observed multi-year climatology and biases of models"'''

    # calculate climatology of observation data
    #temporal mean per grid
    obs_clim = utils.calc_temporal_mean(obs_dataset)
    # determine the metrics
    map_of_bias = metrics.TemporalMeanBias()

    # create the Evaluation object
    bias_evaluation = Evaluation(obs_dataset, # Reference dataset for the evaluation
                                 model_datasets, # list of target datasets for the evaluation
                                 [map_of_bias, map_of_bias])
    # run the evaluation (bias calculation)
    bias_evaluation.run() 

    rcm_bias = bias_evaluation.results[0]
    

    fig = plt.figure()

    lat_min = obs_dataset.lats.min()
    lat_max = obs_dataset.lats.max()
    lon_min = obs_dataset.lons.min()
    lon_max = obs_dataset.lons.max()

    string_list = list(string.ascii_lowercase) 
    # #plot map
    # ax = fig.add_subplot(row,column,1)
    # if map_projection == 'npstere':
        # m = Basemap(ax=ax, projection ='npstere', boundinglat=lat_min, lon_0=0,
            # resolution = 'l', fix_aspect=True)
    # else:
        # m = Basemap(ax=ax, projection ='cyl', llcrnrlat = lat_min, urcrnrlat = lat_max,
            # llcrnrlon = lon_min, urcrnrlon = lon_max, resolution = 'l', fix_aspect=True)
    # if obs_dataset.lons.ndim == 1 and obs_dataset.lats.ndim == 1:
        # lons, lats = np.meshgrid(obs_dataset.lons, obs_dataset.lats)
    # if obs_dataset.lons.ndim == 2 and obs_dataset.lats.ndim == 2:
        # lons = obs_dataset.lons
        # lats = obs_dataset.lats
    # x,y = m(lons, lats)
    # m.drawcoastlines(linewidth=1)
    # m.drawcountries(linewidth=1)
    # m.drawstates(linewidth=0.5, color='w')
    # #plot obs map
    # max = m.contourf(x,y,obs_clim,levels = plotter._nice_intervals(obs_dataset.values, 10), extend='both',cmap='rainbow')
    # #ax.annotate('(a) \n' + obs_name,xy=(lon_min, lat_min))
    # ax.annotate(obs_name,xy=(lon_min, lat_min))
    
    # #The dimensions [left, bottom, width, height] of the new axes
    # cax = fig.add_axes([0.02, 1.-float(1./row)+1./row*0.25, 0.01, 1./row*0.5])
    # plt.colorbar(max, cax = cax) 
    
    # clevs = plotter._nice_intervals(rcm_bias, 11)
    # for imodel in np.arange(len(model_datasets)):

        # ax = fig.add_subplot(row, column,2+imodel)
        # if map_projection == 'npstere':
            # m = Basemap(ax=ax, projection ='npstere', boundinglat=lat_min, lon_0=0,
                # resolution = 'l', fix_aspect=True)
        # else:
            # m = Basemap(ax=ax, projection ='cyl', llcrnrlat = lat_min, urcrnrlat = lat_max,
                # llcrnrlon = lon_min, urcrnrlon = lon_max, resolution = 'l', fix_aspect=True)
        # m.drawcoastlines(linewidth=1)
        # m.drawcountries(linewidth=1)
        # m.drawstates(linewidth=0.5, color='w')
        # #plot model
        # max = m.contourf(x,y,rcm_bias[imodel,:],levels = clevs, extend='both', cmap='RdBu_r')
        # #ax.annotate('('+string_list[imodel+1]+')  \n '+model_names[imodel],xy=(lon_min, lat_min))
        # ax.annotate(model_names[imodel],xy=(lon_min, lat_min))
    
    # #The dimensions [left, bottom, width, height] of the new axes
    # cax = fig.add_axes([0.91, 0.5, 0.015, 0.4])
    # plt.colorbar(max, cax = cax) 

    # plt.subplots_adjust(hspace=0.01,wspace=0.05)
    # file_name=file_name+'_'+sn+'_'+reg+'_'+'_'+tipe
    # fig.savefig(file_name,dpi=600,bbox_inches='tight')
    
    #############################
    #tambahan title + bias on top 
    ###############################
    fig, axes = plt.subplots(nrows=3, ncols=5,figsize=(14,6))
    if map_projection == 'npstere':
        m = Basemap(ax=axes[0,0], projection ='npstere', boundinglat=lat_min, lon_0=0,
            resolution = 'l', fix_aspect=False)
    else:
        m = Basemap(ax=axes[0,0], projection ='cyl', llcrnrlat = lat_min, urcrnrlat = lat_max,
            llcrnrlon = lon_min, urcrnrlon = lon_max, resolution = 'l', fix_aspect=False)
    if obs_dataset.lons.ndim == 1 and obs_dataset.lats.ndim == 1:
        lons, lats = np.meshgrid(obs_dataset.lons, obs_dataset.lats)
    if obs_dataset.lons.ndim == 2 and obs_dataset.lats.ndim == 2:
        lons = obs_dataset.lons
        lats = obs_dataset.lats
    x,y = m(lons, lats)
    m.drawcoastlines(linewidth=1)
    m.drawcountries(linewidth=1)
    m.drawstates(linewidth=0.5, color='w')
    #plot obs map
    max = m.contourf(x,y,obs_clim,levels = plotter._nice_intervals(obs_dataset.values, 10), extend='both',cmap='rainbow')
    #ax.annotate('(a) \n' + obs_name,xy=(lon_min, lat_min))
    #ax.annotate(obs_name,xy=(lon_min, lat_min))
    axes[0,0].set_title(obs_name)
    
    #The dimensions [left, bottom, width, height] of the new axes
    #cax = fig.add_axes([0.1, 1.-float(1./row)+1./row*0.25, 0.01, 1./row*0.8])
    cax = fig.add_axes([0.08, 0.65, 0.01, 1./row*0.9])
    plt.colorbar(max, cax = cax) 
    
    clevs = plotter._nice_intervals(rcm_bias, 11)
    for imodel in np.arange(len(model_datasets)):
        if imodel<4:
            if map_projection == 'npstere':
                m = Basemap(ax=axes[0,1+imodel], projection ='npstere', boundinglat=lat_min, lon_0=0,
                    resolution = 'l', fix_aspect=False )
            else:
                m = Basemap(ax=axes[0,1+imodel], projection ='cyl', llcrnrlat = lat_min, urcrnrlat = lat_max,
                    llcrnrlon = lon_min, urcrnrlon = lon_max, resolution = 'l', fix_aspect=False )
            m.drawcoastlines(linewidth=1)
            m.drawcountries(linewidth=1)
            m.drawstates(linewidth=0.5, color='w')
            #plot model
            #bias_mean = np.mean(rcm_bias[imodel,:])
            #print('bias=',bias_mean)
            max = m.contourf(x,y,rcm_bias[imodel,:],levels = clevs, extend='both', cmap='RdBu_r')
            #ax.annotate('('+string_list[imodel+1]+')  \n '+model_names[imodel],xy=(lon_min, lat_min))
            #ax.annotate(model_names[imodel],xy=(lon_min, lat_min))
            bias_mean = np.mean(rcm_bias[imodel,:])
            axes[0,1+imodel].set_title(model_names[imodel]+' [b='+'%.2f'%bias_mean+']')
        if 3<imodel<9:
            if map_projection == 'npstere':
                m = Basemap(ax=axes[1,1+imodel-5], projection ='npstere', boundinglat=lat_min, lon_0=0,
                    resolution = 'l', fix_aspect=False )
            else:
                m = Basemap(ax=axes[1,1+imodel-5], projection ='cyl', llcrnrlat = lat_min, urcrnrlat = lat_max,
                    llcrnrlon = lon_min, urcrnrlon = lon_max, resolution = 'l', fix_aspect=False )
            m.drawcoastlines(linewidth=1)
            m.drawcountries(linewidth=1)
            m.drawstates(linewidth=0.5, color='w')
            #plot model
            max = m.contourf(x,y,rcm_bias[imodel,:],levels = clevs, extend='both', cmap='RdBu_r')
            #ax.annotate('('+string_list[imodel+1]+')  \n '+model_names[imodel],xy=(lon_min, lat_min))
            #ax.annotate(model_names[imodel],xy=(lon_min, lat_min))
            bias_mean = np.mean(rcm_bias[imodel,:])
            axes[1,1+imodel-5].set_title(model_names[imodel]+' [b='+'%.2f'%bias_mean+']')
        if 8<imodel<13:
            if map_projection == 'npstere':
                m = Basemap(ax=axes[2,1+imodel-10], projection ='npstere', boundinglat=lat_min, lon_0=0,
                    resolution = 'l', fix_aspect=False )
            else:
                m = Basemap(ax=axes[2,1+imodel-10], projection ='cyl', llcrnrlat = lat_min, urcrnrlat = lat_max,
                    llcrnrlon = lon_min, urcrnrlon = lon_max, resolution = 'l', fix_aspect=False )
            m.drawcoastlines(linewidth=1)
            m.drawcountries(linewidth=1)
            m.drawstates(linewidth=0.5, color='w')
            #plot model
            max = m.contourf(x,y,rcm_bias[imodel,:],levels = clevs, extend='both', cmap='RdBu_r')
            #ax.annotate('('+string_list[imodel+1]+')  \n '+model_names[imodel],xy=(lon_min, lat_min))
            #ax.annotate(model_names[imodel],xy=(lon_min, lat_min))
            bias_mean = np.mean(rcm_bias[imodel,:])
            axes[2,1+imodel-10].set_title(model_names[imodel]+' [b='+'%.2f'%bias_mean+']')
        #plot empty
        #if imodel==13:
        #else:
        if map_projection == 'npstere':
            m = Basemap(ax=axes[2,4], projection ='npstere', boundinglat=lat_min, lon_0=0,
                    resolution = 'l', fix_aspect=False )
        else:
            m = Basemap(ax=axes[2,4], projection ='cyl', llcrnrlat = lat_min, urcrnrlat = lat_max,
                    llcrnrlon = lon_min, urcrnrlon = lon_max, resolution = 'l', fix_aspect=False )
        m.drawcoastlines(linewidth=1)
        m.drawcountries(linewidth=1)
        m.drawstates(linewidth=0.5, color='w')
        axes[2,4].set_title('No model')
    
    #The dimensions [left, bottom, width, height] of the new axes
    cax = fig.add_axes([0.91, 0.5, 0.015, 0.4])
    plt.colorbar(max, cax = cax) 
    #plt.colorbar(cax).ax.set_title('mm')

    plt.subplots_adjust(hspace=.25,wspace=.05)
    file_name=file_name+'_2'

    fig.savefig(file_name,dpi=600,bbox_inches='tight')
    
    # # #############################
    # # #tambahan2 zonal
    # # ###############################
    # fig, axes = plt.subplots(nrows=3, ncols=5,figsize=(14,6))
    # if map_projection == 'npstere':
        # m = Basemap(ax=axes[0,0], projection ='npstere', boundinglat=lat_min, lon_0=0,
            # resolution = 'l', fix_aspect=False)
    # else:
        # m = Basemap(ax=axes[0,0], projection ='cyl', llcrnrlat = lat_min, urcrnrlat = lat_max,
            # llcrnrlon = lon_min, urcrnrlon = lon_max, resolution = 'l', fix_aspect=False)
    # if obs_dataset.lons.ndim == 1 and obs_dataset.lats.ndim == 1:
        # lons, lats = np.meshgrid(obs_dataset.lons, obs_dataset.lats)
    # if obs_dataset.lons.ndim == 2 and obs_dataset.lats.ndim == 2:
        # lons = obs_dataset.lons
        # lats = obs_dataset.lats
    # x,y = m(lons, lats)
    # m.drawcoastlines(linewidth=1)
    # m.drawcountries(linewidth=1)
    # m.drawstates(linewidth=0.5, color='w')
    
    # #plot ref
    # x=np.arange(12)+.5
    # ds0 = ma.average(obs_dataset.values, axis=2) #lon
    # print('ds0.shape=',ds0.shape)
    # print(y.shape)
    # print(lats.shape)
    # max = axes[0,0].contourf(x,lats,ds0.T,levels = plotter._nice_intervals(obs_dataset.values, 10), extend='both',cmap='rainbow')
    # #ax.annotate('(a) \n' + obs_name,xy=(lon_min, lat_min))
    # #ax.annotate(obs_name,xy=(lon_min, lat_min))
    # axes[0,0].set_title(obs_name)
    
    # #The dimensions [left, bottom, width, height] of the new axes
    # #cax = fig.add_axes([0.1, 1.-float(1./row)+1./row*0.25, 0.01, 1./row*0.8])
    # cax = fig.add_axes([0.08, 0.65, 0.01, 1./row*0.9])
    # plt.colorbar(max, cax = cax) 
    
    # clevs = plotter._nice_intervals(rcm_bias, 11)
    # for imodel in np.arange(len(model_datasets)):
        # if imodel<4:
            # if map_projection == 'npstere':
                # m = Basemap(ax=axes[0,1+imodel], projection ='npstere', boundinglat=lat_min, lon_0=0,
                    # resolution = 'l', fix_aspect=False )
            # else:
                # m = Basemap(ax=axes[0,1+imodel], projection ='cyl', llcrnrlat = lat_min, urcrnrlat = lat_max,
                    # llcrnrlon = lon_min, urcrnrlon = lon_max, resolution = 'l', fix_aspect=False )
            # m.drawcoastlines(linewidth=1)
            # m.drawcountries(linewidth=1)
            # m.drawstates(linewidth=0.5, color='w')
            # #plot model
            # #bias_mean = np.mean(rcm_bias[imodel,:])
            # #print('bias=',bias_mean)
            # max = m.contourf(x,y,rcm_bias[imodel,:],levels = clevs, extend='both', cmap='RdBu_r')
            # #ax.annotate('('+string_list[imodel+1]+')  \n '+model_names[imodel],xy=(lon_min, lat_min))
            # #ax.annotate(model_names[imodel],xy=(lon_min, lat_min))
            # bias_mean = np.mean(rcm_bias[imodel,:])
            # axes[0,1+imodel].set_title(model_names[imodel]+' [b='+'%.2f'%bias_mean+']')
        # if 3<imodel<9:
            # if map_projection == 'npstere':
                # m = Basemap(ax=axes[1,1+imodel-5], projection ='npstere', boundinglat=lat_min, lon_0=0,
                    # resolution = 'l', fix_aspect=False )
            # else:
                # m = Basemap(ax=axes[1,1+imodel-5], projection ='cyl', llcrnrlat = lat_min, urcrnrlat = lat_max,
                    # llcrnrlon = lon_min, urcrnrlon = lon_max, resolution = 'l', fix_aspect=False )
            # m.drawcoastlines(linewidth=1)
            # m.drawcountries(linewidth=1)
            # m.drawstates(linewidth=0.5, color='w')
            # #plot model
            # max = m.contourf(x,y,rcm_bias[imodel,:],levels = clevs, extend='both', cmap='RdBu_r')
            # #ax.annotate('('+string_list[imodel+1]+')  \n '+model_names[imodel],xy=(lon_min, lat_min))
            # #ax.annotate(model_names[imodel],xy=(lon_min, lat_min))
            # bias_mean = np.mean(rcm_bias[imodel,:])
            # axes[1,1+imodel-5].set_title(model_names[imodel]+' [b='+'%.2f'%bias_mean+']')
        # if 8<imodel<13:
            # if map_projection == 'npstere':
                # m = Basemap(ax=axes[2,1+imodel-10], projection ='npstere', boundinglat=lat_min, lon_0=0,
                    # resolution = 'l', fix_aspect=False )
            # else:
                # m = Basemap(ax=axes[2,1+imodel-10], projection ='cyl', llcrnrlat = lat_min, urcrnrlat = lat_max,
                    # llcrnrlon = lon_min, urcrnrlon = lon_max, resolution = 'l', fix_aspect=False )
            # m.drawcoastlines(linewidth=1)
            # m.drawcountries(linewidth=1)
            # m.drawstates(linewidth=0.5, color='w')
            # #plot model
            # max = m.contourf(x,y,rcm_bias[imodel,:],levels = clevs, extend='both', cmap='RdBu_r')
            # #ax.annotate('('+string_list[imodel+1]+')  \n '+model_names[imodel],xy=(lon_min, lat_min))
            # #ax.annotate(model_names[imodel],xy=(lon_min, lat_min))
            # bias_mean = np.mean(rcm_bias[imodel,:])
            # axes[2,1+imodel-10].set_title(model_names[imodel]+' [b='+'%.2f'%bias_mean+']')
        # #plot empty
        # #if imodel==13:
        # #else:
        # if map_projection == 'npstere':
            # m = Basemap(ax=axes[2,4], projection ='npstere', boundinglat=lat_min, lon_0=0,
                    # resolution = 'l', fix_aspect=False )
        # else:
            # m = Basemap(ax=axes[2,4], projection ='cyl', llcrnrlat = lat_min, urcrnrlat = lat_max,
                    # llcrnrlon = lon_min, urcrnrlon = lon_max, resolution = 'l', fix_aspect=False )
        # m.drawcoastlines(linewidth=1)
        # m.drawcountries(linewidth=1)
        # m.drawstates(linewidth=0.5, color='w')
        # axes[2,4].set_title('No model')
    
    # #The dimensions [left, bottom, width, height] of the new axes
    # cax = fig.add_axes([0.91, 0.5, 0.015, 0.4])
    # plt.colorbar(max, cax = cax) 
    # #plt.colorbar(cax).ax.set_title('mm')

    # plt.subplots_adjust(hspace=.25,wspace=.05)
    # file_name=file_name+'_4'

    # fig.savefig(file_name,dpi=600,bbox_inches='tight')
    
    
    # #############################
    # #tambahan3: (for annual) Zonal mean:ITCZ, title + bias on top 
    # ###############################
    # fig, axes = plt.subplots(nrows=3, ncols=10,figsize=(16,6))
    # Tmax = 15; Tmin =0 ; delT = 5
    # clevels = np.arange(Tmin,Tmax+delT,delT)
    
    # cmap2='RdBu_r'
    # x_tick=['J','F','M','A','M','J','J','A','S','O','N','D']
    
    # ds = xr.DataArray(obs_dataset.values,
    # coords={'time': obs_dataset.times,
    # 'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
    # dims=["time", "lat", "lon"])
        
    # #ds = xr.DataArray(obs_dataset)
    # ds = ds.groupby('time.month').mean() 
    # ds0= ds.mean(dim='lon')
    # lat=ds.lat
    # lon=ds.lon
    # lat_min=ds.lat.min()
    # lon_min=ds.lon.min()
    
    # cax = axes[0,0].contourf(np.arange(12)+.5, lat, 
                      # ds0.transpose(), 
                      # levels=clevels, extend='both',
                      # cmap=plt.cm.seismic, vmin=Tmin, vmax=Tmax)
    # axes[0,0].set_title('GPCP')
    # axes[0,0].set_yticks([-10,0,10,20])
    # axes[0,0].set_xticks([])
             
    # model_datasets=np.delete(model_datasets,[1, -1])
    # model_names=np.delete(model_names,[1, -1])
    
    # for i in np.arange(len(model_datasets)):
        # print(i)
        # if i<9: axes[0,1+i].axis('off')
        # ds = xr.DataArray(model_datasets[i].values,
        # coords={'time': obs_dataset.times,
        # 'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
        # dims=["time", "lat", "lon"])
        
        # ds = ds.groupby('time.month').mean() 
        # ds= ds.mean(dim='lon')
        # if i<10:
            # cax = axes[1,i].contourf(np.arange(12)+.5, lat, 
                      # ds.transpose(), 
                      # levels=clevels, extend='both',
                      # cmap=plt.cm.seismic, vmin=Tmin, vmax=Tmax)
            # axes[1,i].set_xticks(np.arange(12)+.5)
            # axes[1,i].set_xticklabels(x_tick)
            # axes[1,i].set_title(model_names[i])
            # axes[1,0].set_yticks([-10,0,10,20])
            # #axes[1,i].set_xticks([])
            # if i<9: axes[1,1+i].set_yticks([])
            # #bias
            # ds=ds-ds0
            # bias=ds.mean()
            # cax = axes[2,i].contourf(np.arange(12)+.5, lat, 
                      # ds.transpose(), 
                      # levels=clevels, extend='both',
                      # cmap=cmap2, vmin=Tmin, vmax=Tmax)
            # #axes[2,i].set_title('bias='+'%.2f'%bias, 'bottom')
            # axes[2,i].set_xticks([])
            # axes[2,i].annotate('b='+'%.2f'%bias,xy=(3, lat_min), backgroundcolor='0.85',alpha=1)
            # axes[2,0].set_yticks([-10,0,10,20])
            # if i<9: axes[2,1+i].set_yticks([])
            
            
        
    # #The dimensions [left, bottom, width, height] of the new axes
    # cax = fig.add_axes([0.91, 0.5, 0.015, 0.4])
    # plt.colorbar(max, cax = cax) 
    # #plt.colorbar(cax).ax.set_title('mm')

    # plt.subplots_adjust(hspace=.25,wspace=.15)
    # file_name=file_name+'_3'

    # fig.savefig(file_name,dpi=600,bbox_inches='tight')
    
    #############################
    #tambahan4: (for annual) Zonal mean:ITCZ, title + bias on top 
    # tanpa Tmin Tmax
    ###############################
    fig, axes = plt.subplots(nrows=3, ncols=10,figsize=(16,6))
    #Tmax = 15; Tmin =0 ; delT = 5
    #clevels = np.arange(Tmin,Tmax+delT,delT)
    clevs = plotter._nice_intervals(rcm_bias, 11)
    
    cmap2='RdBu_r'
    x_tick=['J','F','M','A','M','J','J','A','S','O','N','D']
    
    ds = xr.DataArray(obs_dataset.values,
    coords={'time': obs_dataset.times,
    'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
    dims=["time", "lat", "lon"])
        
    #ds = xr.DataArray(obs_dataset)
    ds = ds.groupby('time.month').mean() 
    #print('xx=',ds.shape, ds.lat.shape)
    
    ##temporal_corr
    lon=103
    lat=-1
    x=extract_data_at_nearest_grid_point2(ds, lon, lat)
    #print('x=',x)
    
    
    ###cal corr from monthly cycle => from annual ? and from raw_input
    from scipy.stats import pearsonr
    corr=ma.zeros(len(model_datasets)) 
    for i in np.arange(len(model_datasets)):
        ds = xr.DataArray(model_datasets[i].values,
        coords={'time': obs_dataset.times,
        'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
        dims=["time", "lat", "lon"])
        
        ds = ds.groupby('time.month').mean()
        
        #print(ds.shape,ds.lat.shape, ds.lon.shape)
        y=extract_data_at_nearest_grid_point2(ds, lon, lat)
        #print('y=',y)
        corr[i]=pearsonr(y, x)[0]
    #print('corr=',corr)        
    #########
   
    ds0= ds.mean(dim='lon')
    lat=ds.lat
    lon=ds.lon
    lat_min=ds.lat.min()
    lon_min=ds.lon.min()
    
    cax = axes[0,0].contourf(np.arange(12)+.5, lat, 
                      ds0.transpose(), 
                      levels=clevs, extend='both',
                      cmap=cmap2)
    axes[0,0].set_title('GPCP')
    axes[0,0].set_yticks([-10,0,10,20])
    axes[0,0].set_xticks([])
    
    # ini ?? khusus zonal jika obs>1 dan MMEW not included
    model_datasets=np.delete(model_datasets,[1, -1])
    model_names=np.delete(model_names,[1, -1])
    
    for i in np.arange(len(model_datasets)):
        #print(i)
        if i<9: axes[0,1+i].axis('off')
        ds = xr.DataArray(model_datasets[i].values,
        coords={'time': obs_dataset.times,
        'lat': obs_dataset.lats, 'lon': obs_dataset.lons},
        dims=["time", "lat", "lon"])
        
        ds = ds.groupby('time.month').mean()
        
        #print(ds.shape,ds.lat.shape, ds.lon.shape)
        #y=extract_data_at_nearest_grid_point2(ds, lon, lat)
          
        #corr[i]=pearsonr(y, x)[0]
        
        ds= ds.mean(dim='lon')
        if i<10:
            cax = axes[1,i].contourf(np.arange(12)+.5, lat, 
                      ds.transpose(), 
                      levels=clevs, extend='both',
                      cmap=cmap2)
            axes[1,i].set_xticks(np.arange(12)+.5)
            axes[1,i].set_xticklabels(x_tick)
            axes[1,i].set_title(model_names[i])
            axes[1,0].set_yticks([-10,0,10,20])
            #axes[1,i].set_xticks([])
            if i<9: axes[1,1+i].set_yticks([])
            #bias
            ds=ds-ds0
            #ds= ds.mean(dim='lon')

            bias=ds.mean()
            cax = axes[2,i].contourf(np.arange(12)+.5, lat, 
                      ds.transpose(), 
                      levels=clevs, extend='both',
                      cmap=cmap2)
            #axes[2,i].set_title('bias='+'%.2f'%bias, 'bottom')
            axes[2,i].set_xticks([])
            axes[2,i].annotate('b='+'%.2f'%bias,xy=(3, lat_min), backgroundcolor='0.85',alpha=1)
            axes[2,0].set_yticks([-10,0,10,20])
            if i<9: axes[2,1+i].set_yticks([])
    #print('corr=',corr)        
            
        
    #The dimensions [left, bottom, width, height] of the new axes
    cax = fig.add_axes([0.91, 0.5, 0.015, 0.4])
    plt.colorbar(max, cax = cax) 
    #plt.colorbar(cax).ax.set_title('mm')

    plt.subplots_adjust(hspace=.25,wspace=.15)
    file_name=file_name+'_3'
    plt.show()
    fig.savefig(file_name,dpi=600,bbox_inches='tight')
    
def Map_plot_wind_bias(obs_dataset, obs_name, model_datasets, model_names,
                                      file_name, row, column, map_projection=None):
    '''Draw maps of observed multi-year climatology and biases of models"'''
    print('cek1=',len(model_datasets))
    # calculate climatology of observation data ==> ref
    #temporal mean per grid
    obs_climu = utils.calc_temporal_mean(obs_dataset)
    obs_climv = utils.calc_temporal_mean(model_datasets[0])
    
    print('obs_dataset.times:',obs_dataset.times[:3])

    #model_datasets2=[]
    for i, dataset in enumerate(model_datasets[1:]):
        print('wind i=',i)
        model_datasets[i] = utils.calc_temporal_mean(dataset)
    
    #ini hapus mme dan wmme ??
    model_datasets=np.delete(model_datasets,[-1])
    print('cek2=',len(model_datasets))
    
    model_datasets=np. array(model_datasets, dtype=object)
    #fig = plt.figure()
    print('cek3=',len(model_datasets))
    
    
    lat_min = obs_dataset.lats.min()
    lat_max = obs_dataset.lats.max()
    lon_min = obs_dataset.lons.min()
    lon_max = obs_dataset.lons.max()
    
    lats=obs_dataset.lats
    lons=obs_dataset.lons
    
    #data=[[U1,V1],[U2,V2],...]

    model_datasets= model_datasets.reshape(int(len(model_datasets)/2), 2)

    cmap2='RdBu_r'
   
    
    #ref
    u0,v0= obs_climu,obs_climv
    
    #ubah obs dan model names
    obs_name='ERA5'
    model_names=['CNRM_a','HadGEM2_a','HadGEM2_c','MPI_c']
    
    print('wind_mod_names:',model_names)
    fig, axes = plt.subplots(nrows=3, ncols=5,figsize=(16,6))
    
    #plot ref
    
    m = Basemap(ax=axes[0,0], projection ='cyl', llcrnrlat = lat_min, urcrnrlat = lat_max,
                    llcrnrlon = lon_min, urcrnrlon = lon_max, resolution = 'l', fix_aspect=False)
    m.drawcoastlines(linewidth=1)
        #m.drawcountries(linewidth=1)
        #m.drawstates(linewidth=0.5, color='w')
    #dikali 100 ==> u0*100,v0*100 tidak nampak efeknya
    varU,varV= u0,v0
    speed=np.sqrt(varU*varU + varV*varV)
    sp0=speed
    std0=np.std(speed.flatten())
    print(speed.shape)
    #print(speed.squeeze().shape)
    print(speed.flatten().shape)
    print(np.std(speed.squeeze()))
    print(np.std(speed.flatten()))
    
    print(np.corrcoef(speed.flatten(),speed.flatten()))
    #exit()
    #jika SEA 25 jika Sumatera 5
    yy=np.arange(0,len(lats),5)
    xx=np.arange(0,len(lons),5)

    points=np.meshgrid(yy,xx)
    X4,Y4=np.meshgrid(lons,lats)
    
    #m.quiver(X4[points],Y4[points],varU[points],varV[points],speed[points],cmap=cmap2,latlon=False)
    m.quiver(X4[points],Y4[points],varU[points],varV[points],speed[points],cmap=cmap2,headwidth=12, headlength=6)
    max=m.quiver(X4,Y4,varU,varV,speed,cmap='jet',latlon=True)
    axes[0,0].set_title(obs_name)
    axes[0,1].axis('off')
    axes[0,2].axis('off')
    axes[0,3].axis('off')
    axes[0,4].axis('off')

    for i in np.arange(len(model_datasets)):
        print('len(model_datasets)=',len(model_datasets),i)
        
        varU,varV =  model_datasets[i]
        #print('varV=',varV) #ada errors:
        #len(model_datasets)= 5 4 # max i 5?? harusnya 4 
        #varU= <Dataset - name: vCNRM_a, lat-range: ...
        
        #T1= (4*(1+c))/(((s+1/s)**2)*((1+r0)))
        
        if i<4:
            axes[1,4].axis('off')
            m = Basemap(ax=axes[1,i], projection ='cyl', llcrnrlat = lat_min, urcrnrlat = lat_max,
                        llcrnrlon = lon_min, urcrnrlon = lon_max, resolution = 'l', fix_aspect=False)
            m.drawcoastlines(linewidth=1)
                #m.drawcountries(linewidth=1)
                #m.drawstates(linewidth=0.5, color='w')

            speed=np.sqrt(varU*varU + varV*varV)
            c=np.corrcoef(sp0.flatten(),speed.flatten())[0,1]
            std1=np.std(speed.flatten())
            s=std1/std0
            r0=1
            t= (4*(1+c))/(((s+1/s)**2)*((1+r0)))
            t=np.round(t,2)
            print('Taylor score=',np.round(t,2))
            
            #print(c[0,1])
            #exit()
            points=np.meshgrid(yy,xx)
            
            m.quiver(X4[points],Y4[points],varU[points],varV[points],speed[points],cmap=cmap2,latlon=False,headwidth=12, headlength=6)
            max=m.quiver(X4,Y4,varU,varV,speed,cmap='jet',latlon=True)
            axes[1,i].set_title(model_names[i]+' ('+str(t)+')')
        
        ###
        #if i<4:
            varU=varU-u0 
            varV=varU-v0 
            bias = varU.mean() + varV.mean()
            #Sprint('b=', bias)
            axes[2,4].axis('off')
            m = Basemap(ax=axes[2,i], projection ='cyl', llcrnrlat = lat_min, urcrnrlat = lat_max,
                        llcrnrlon = lon_min, urcrnrlon = lon_max, resolution = 'l', fix_aspect=False)
            m.drawcoastlines(linewidth=1)
                #m.drawcountries(linewidth=1)
                #m.drawstates(linewidth=0.5, color='w')

            #X4,Y4=np.meshgrid(lon,lat)           
            speed=np.sqrt(varU*varU + varV*varV)
            points=np.meshgrid(yy,xx)
            #print('points=',points)
            
            m.quiver(X4[points],Y4[points],varU[points],varV[points],speed[points],cmap=cmap2,latlon=False,headwidth=12, headlength=6)
            max2=m.quiver(X4,Y4,varU,varV,speed,cmap=cmap2,latlon=True)
            
            axes[2,i].set_title('bias ('+'%.2f'%bias+')')
        
    cax = fig.add_axes([0.28, 0.65, 0.015, 0.25])
    plt.colorbar(max, cax = cax) 
    
    cax2 = fig.add_axes([0.75, 0.1, 0.015, 0.25])
    plt.colorbar(max2, cax = cax2) 

    plt.subplots_adjust(hspace=.3,wspace=0.05)
    
    plt.show()
    fig.savefig('wind_sumatera_DJF_'+reg,dpi=600,bbox_inches='tight')
    
def wind_sum(obs_dataset, obs_name, model_datasets, model_names,
                        workdir):
    # Sumatera DJF JJA
    
    if config['temporal_subset']==1:
        month_start=config['start_month']
        month_end=config['end_month']
    
    obs_dataset = dsp.temporal_subset(obs_dataset, month_start, month_end, average_each_year=False)
    for i, dataset in enumerate(model_datasets):
        model_datasets[i] = dsp.temporal_subset(dataset, month_start, month_end,average_each_year=False)


    #non-aktifkan jika tidak perlu
    #spatial subset Sumatera
    obs_dataset = dsp.spatial_slice(obs_dataset, 
                        lat_min=-6, lat_max=6, 
                        lon_min=94, lon_max=107)
    for i, dataset in enumerate(model_datasets):
        model_datasets[i] = dsp.spatial_slice(dataset, 
                        lat_min=-6, lat_max=6, 
                        lon_min=94, lon_max=107)
    
    
    print('cek1=',len(model_datasets))
    # calculate climatology of observation data ==> ref
    #temporal mean per grid
    obs_climu = utils.calc_temporal_mean(obs_dataset)
    obs_climv = utils.calc_temporal_mean(model_datasets[0])
    
    print('obs_dataset.times:',obs_dataset.times[:3])

    #model_datasets2=[]
    for i, dataset in enumerate(model_datasets[1:]):
        print('wind i=',i)
        model_datasets[i] = utils.calc_temporal_mean(dataset)
    
    #ini hapus mme dan wmme ??
    model_datasets=np.delete(model_datasets,[-1])
    print('cek2=',len(model_datasets))
    
    model_datasets=np. array(model_datasets, dtype=object)
    #fig = plt.figure()
    print('cek3=',len(model_datasets))
    
    
    lat_min = obs_dataset.lats.min()
    lat_max = obs_dataset.lats.max()
    lon_min = obs_dataset.lons.min()
    lon_max = obs_dataset.lons.max()
    
    lats=obs_dataset.lats
    lons=obs_dataset.lons
    
    #data=[[U1,V1],[U2,V2],...]

    model_datasets= model_datasets.reshape(int(len(model_datasets)/2), 2)

    cmap2='RdBu_r'
   
    
    #ref
    u0,v0= obs_climu,obs_climv
    
    #ubah obs dan model names
    obs_name='ERA5'
    model_names=['CNRM_a','HadGEM2_a','HadGEM2_c','MPI_c']
    
    print('wind_mod_names:',model_names)
    #fig, axes = plt.subplots(nrows=3, ncols=5,figsize=(16,6))
    fig, axes = plt.subplots(nrows=1, ncols=1,figsize=(8,6))
    
    #plot ref
    
    m = Basemap(ax=axes, projection ='cyl', llcrnrlat = lat_min, urcrnrlat = lat_max,
                    llcrnrlon = lon_min, urcrnrlon = lon_max, resolution = 'l', fix_aspect=False)
    m.drawcoastlines(linewidth=1)
        #m.drawcountries(linewidth=1)
        #m.drawstates(linewidth=0.5, color='w')
    #dikali 100 ==> u0*100,v0*100 tidak nampak efeknya
    varU,varV= u0,v0
    speed=np.sqrt(varU*varU + varV*varV)
    sp0=speed
    std0=np.std(speed.flatten())
    print(speed.shape)
    #print(speed.squeeze().shape)
    print(speed.flatten().shape)
    print(np.std(speed.squeeze()))
    print(np.std(speed.flatten()))
    
    print(np.corrcoef(speed.flatten(),speed.flatten()))
    #exit()
    #jika SEA 25 jika Sumatera 5
    yy=np.arange(0,len(lats),5)
    xx=np.arange(0,len(lons),5)

    points=np.meshgrid(yy,xx)
    X4,Y4=np.meshgrid(lons,lats)
    
    #m.quiver(X4[points],Y4[points],varU[points],varV[points],speed[points],cmap=cmap2,latlon=False)
    m.quiver(X4[points],Y4[points],varU[points],varV[points],speed[points],cmap=cmap2,headwidth=4, headlength=6)
    max=m.quiver(X4,Y4,varU,varV,speed,cmap='jet',latlon=True)
    axes.set_title(obs_name+' (JJA)')
      
    cax = fig.add_axes([0.907, 0.11, 0.015, 0.77])
    plt.colorbar(max,cax) 
    
   
    plt.show()
    fig.savefig('wind_sumatera_JJA_'+reg,dpi=600,bbox_inches='tight')

def wind_sum2(obs_dataset, obs_name, model_datasets, model_names,
                        workdir):
    #fig, axes = plt.subplots(nrows=3, ncols=5,figsize=(16,6))
    fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(6,6))
    seasons=['DJF','MAM']
    #seasons=['JJA','SON']
    n=0
    for mu in seasons:
        
        if mu=='DJF':
            month_start=12
            month_end=2
            
        if mu=='MAM':
            month_start=3
            month_end=5
            
        if mu=='JJA':
            month_start=6
            month_end=8
            
        if mu=='SON':
            month_start=9
            month_end=11
        
        print('n=',n,mu)
        
        #temporal slice
        model_datasets2=model_datasets
        obs_dataset2 = dsp.temporal_subset(obs_dataset, month_start, month_end, average_each_year=False)
        #for i, dataset in enumerate(model_datasets):
        model_datasets2 = dsp.temporal_subset(model_datasets[0], month_start, month_end,average_each_year=False)
        
        '''
        #sptial slice
        obs_dataset2 = dsp.spatial_slice(obs_dataset2, 
                            lat_min=-6, lat_max=6, 
                            lon_min=94, lon_max=107)
        #for i, dataset in enumerate(model_datasets2):
        model_datasets2 = dsp.spatial_slice(model_datasets2, 
                            lat_min=-6, lat_max=6, 
                            lon_min=94, lon_max=107)
        '''
        #sptial slice jambi
        obs_dataset2 = dsp.spatial_slice(obs_dataset2, 
                            lat_min=-3, lat_max=-0.4, 
                            lon_min=100, lon_max=105)
        #for i, dataset in enumerate(model_datasets2):
        model_datasets2 = dsp.spatial_slice(model_datasets2, 
                            lat_min=-3, lat_max=-0.4, 
                            lon_min=100, lon_max=105)
        
        obs_climu = utils.calc_temporal_mean(obs_dataset2)
        obs_climv = utils.calc_temporal_mean(model_datasets2)
                
        
        lat_min = obs_dataset2.lats.min()
        lat_max = obs_dataset2.lats.max()
        lon_min = obs_dataset2.lons.min()
        lon_max = obs_dataset2.lons.max()
        
        lats=obs_dataset2.lats
        lons=obs_dataset2.lons
       
        cmap2='RdBu_r'
       
        u0,v0= obs_climu,obs_climv
        #print(u0,v0)
        
        #ubah obs dan model names
        obs_name='ERA5'
     
        m = Basemap(ax=axes[n], projection ='cyl', llcrnrlat = lat_min, urcrnrlat = lat_max,
                        llcrnrlon = lon_min, urcrnrlon = lon_max, resolution = 'h', fix_aspect=False)
        m.drawcoastlines(linewidth=1)
      
        varU,varV= u0,v0
        speed=np.sqrt(varU*varU + varV*varV)
       
        yy=np.arange(0,len(lats),5)
        xx=np.arange(0,len(lons),5)

        points=np.meshgrid(yy,xx)
        X4,Y4=np.meshgrid(lons,lats)
        
        #m.quiver(X4[points],Y4[points],varU[points],varV[points],speed[points],cmap=cmap2,latlon=False)
        #for 2 season2
        #m.quiver(X4[points],Y4[points],varU[points],varV[points],speed[points],
        #    cmap=cmap2,headwidth=4, headlength=6)
        #for 4
        m.quiver(X4[points],Y4[points],varU[points],varV[points],speed[points],
            cmap=cmap2,headwidth=8, headlength=12)
        max=m.quiver(X4,Y4,varU,varV,speed,cmap='jet',latlon=True)
        axes[n].set_title(obs_name+' ('+seasons[n]+')')
          
        #colorbar tiap map
        plt.colorbar(max, ax=axes[n], pad=0.02) 
        '''
        axes[0].set_yticks([-5,0,5])  
        axes[0].set_yticklabels(['5S','0','5N'])
        axes[0].tick_params(axis='y', pad=1,labelsize=10)
        
        axes[n].set_xticks([95,100,105])  
        axes[n].set_xticklabels(['95E','100E','105E'])
        axes[n].tick_params(axis='x', pad=1, labelsize=10)
        '''
        n=n+1
    #cax = fig.add_axes([0.907, 0.11, 0.015, 0.77])
    #plt.colorbar(max,cax)  
    plt.subplots_adjust(hspace=0,wspace=0)
   
    plt.show()
    fig.savefig('wind_sumatera_'+reg,dpi=600,bbox_inches='tight')


def Taylor_diagram_spatial(workdir, seasons, obs_dataset, obs_name, model_datasets, model_names):
    
    for musim in seasons:
        print(musim)       
        if musim == 'DJF':
            month_start1= 12
            month_end1= 2
            nmon= 3
           
        if musim == 'JJA':
            month_start1= 6
            month_end1= 8
            nmon= 3
             
        #temporal subset      
        obs_dataset2 = dsp.temporal_subset(obs_dataset, month_start1, month_end1)
        
        model_datasets2=[]
        for i, dataset in enumerate(model_datasets):
            print(i, dataset)
            #model_datasets2[i] = dsp.temporal_subset(dataset, month_start1, month_end1)
            model_datasets2.append(dsp.temporal_subset(dataset, month_start1, month_end1))
            #print('model_datasets2[i]', model_datasets2[i])
        print('spatial subsetting =>', obs_dataset2.values.shape)   
    
        #spatial mean
        obs_clim_dataset = ds.Dataset(obs_dataset2.lats, obs_dataset2.lons, 
                        obs_dataset2.times, 
                        utils.calc_temporal_mean(obs_dataset2))
        
              
        model_clim_datasets = []     
        for dataset2 in model_datasets2:
            print('x', dataset2)
            model_clim_datasets.append(ds.Dataset(dataset2.lats, dataset2.lons, 
                                dataset2.times, utils.calc_temporal_mean(dataset2)))
        
        print('last spatial subsetting =>', obs_clim_dataset.values.shape)
        
        
        
        taylor_diagram = metrics.SpatialPatternTaylorDiagram()
        #create the Evaluation object
        taylor_evaluation = Evaluation(obs_clim_dataset, # Climatological mean of reference dataset for the evaluation
                                     model_clim_datasets, # list of climatological means from model datasets for the evaluation
                                     [taylor_diagram])
        print('cek')
        # run the evaluation (bias calculation)
        taylor_evaluation.run() 
        print('cek2')
        taylor_data = taylor_evaluation.results[0]
        #print('td')
        print('td=',taylor_data)
        
        import pandas as pd
        #obs_names=np.array(np.repeat(obs_name, len(model_names)))
        result = pd.DataFrame(taylor_data, model_names, columns=['stddev','corrcoef',])
        result.to_excel(workdir+'Taylor_diagram_spatial_'+reg+'_'+musim+'.xlsx') #set name DJF etc
        obsN=[obs_name]
        obsN = pd.DataFrame(obsN) 
        obsN.to_excel(workdir+'Taylor_diagram_spatial_ref.xlsx') #set name DJF etc
        
        #plotter.draw_taylor_diagram(taylor_data, model_names, obs_name, file_name, pos='upper right',frameon=False)
        file_name=workdir+'Taylor_diagram_spatial_'+reg+'_'+musim
        
        if config['Taylor_diagram_spatial_type']==1:
            plotter2.draw_taylor_diagram(taylor_data, model_names, obs_name, file_name, pos='upper right',frameon=False)
        else:
            plotter2.draw_taylor_diagram2(taylor_data, model_names, obs_name, file_name, pos='upper right',frameon=False)
    
def Taylor_diagram_temporal(obs_dataset, obs_name, model_datasets, model_names,
                                      file_name,workdir):
    #plot yg sudah jadi
    #plotter2.draw_Sum_taylor_diagram(workdir)
    #exit()
    
    #temporal_annual
    '''
    if config['spatial_season?']==1:
    sn= config['spatial_season']['season_name']
    
    if config['temporal_annual?']==1:
    sn= config['temporal_annual']['season_name']
    if config['temporal_annual_cycle?']==1:
    sn= config['temporal_annual_cycle']['season_name']
    #if not config['spatial_season?'] and  not config['temporal_annual_cycle?']: 
    #    sn ='monthly'  
    '''
    if config['temporal_annual?']==1:
        sn= config['temporal_annual']['season_name']
        obs_dataset = dsp.annual_subset(obs_dataset)
        for i, dataset in enumerate(model_datasets):
            model_datasets[i] = dsp.annual_subset(dataset)
        print('temporal_annual...')
        print('temporal subsetting =>', obs_dataset.values.shape)
        
    if config['temporal_annual_cycle?']==1:
        sn= config['temporal_annual_cycle']['season_name']
        obs_dataset = dsp.annual_cycle_subset(obs_dataset)
        for i, dataset in enumerate(model_datasets):
            model_datasets[i] = dsp.annual_cycle_subset(dataset)
        print('temporal_annual_cycle...')
        print('temporal subsetting =>', obs_dataset.values.shape)
        
    if config['temporal_season?']==1:
        sn= config['spatial_season']['season_name']
        obs_dataset = dsp.season_subset(obs_dataset)
        for i, dataset in enumerate(model_datasets):
            model_datasets[i] = dsp.season_subset(dataset)
        print('temporal_season...')
        print('temporal subsetting =>', obs_dataset.values.shape)
        
    if not config['temporal_annual?']==1 and\
       not config['temporal_annual_cycle?']==1 and\
       not config['temporal_season?']==1:
        #khusus data 1981-2005=25
        if len(obs_dataset.times)== 25:
            sn='yearly'
        else: 
            sn='monthly'
    
    print('sn=',sn)
    print('td temporal subsetting =>', obs_dataset.values.shape)
    
    #TEMPORAL
    taylor_diagram = metrics.TemporalPatternTaylorDiagram()

    # create the Evaluation object
    taylor_evaluation = Evaluation(obs_dataset, # Climatological mean of reference dataset for the evaluation
                                 model_datasets, # list of climatological means from model datasets for the evaluation
                                 [taylor_diagram])

    # run the evaluation (bias calculation)
    taylor_evaluation.run() 

    taylor_data = taylor_evaluation.results[0]
    #print('td')
    print('td=',taylor_data)
    
    import pandas as pd
    #obs_names=np.array(np.repeat(obs_name, len(model_names)))
    result = pd.DataFrame(taylor_data, model_names, columns=['stddev','corrcoef',])
    result.to_excel(file_name+'_'+sn+'_'+reg+'_'+tipe+'.xlsx') #set name DJF etc
    obsN=[obs_name]
    obsN = pd.DataFrame(obsN) 
    obsN.to_excel(file_name+'_ref.xlsx') #set name DJF etc
    
    #plotter.draw_taylor_diagram(taylor_data, model_names, obs_name, file_name, pos='upper right',frameon=False)
    file_name=file_name+'_'+sn+'_'+reg+'_'+tipe
    
    if config['taylor_diagram_type?']==1:
        plotter2.draw_taylor_diagram(taylor_data, model_names, obs_name, file_name, pos='upper right',frameon=False)
    else:
        plotter2.draw_taylor_diagram2(taylor_data, model_names, obs_name, file_name, pos='upper right',frameon=False)

def All_metrics(obs_dataset, obs_name, model_datasets, model_names, workdir):
    filename=reg+'_All_metrics'  
    
    # calculate climatological mean fields
    obs_clim_dataset = ds.Dataset(obs_dataset.lats, obs_dataset.lons, obs_dataset.times, utils.calc_temporal_mean(obs_dataset))
    model_clim_datasets = []
    for dataset in model_datasets:
        model_clim_datasets.append(ds.Dataset(dataset.lats, dataset.lons, dataset.times, utils.calc_temporal_mean(dataset)))
    
    print('All_metrics subsetting =>', obs_clim_dataset.values.shape)
    
    metrics1=metrics.Metric3()
    metrics1_evaluation = Evaluation(obs_clim_dataset, # Climatological mean of reference dataset for the evaluation
                                 model_clim_datasets, # list of climatological means from model datasets for the evaluation
                                 [metrics1])
    metrics1_evaluation.run() 
    xmetrics1 = metrics1_evaluation.results[0]  
    #print('xmetrics=',xmetrics1)
    import pandas as pd
    #result = pd.DataFrame(xmetrics1,model_names,columns=[
    #         '1/s','r','1/rmse','Tian','Taylor1','Taylor2','1/D','cpi','H','M'])
    #        [s, c, r, abs(b),vr,vf,Tian,Taylor1,Taylor2,D,cpi,H,M]
    #result.to_excel(file_name+'_'+sn+'_'+reg+'_'+'_'+tipe+'.xlsx') #set DJF etc
    #sederhana
    #result.to_excel(workdir+filename+'4.xlsx')
    
    #----W_PI----hanya 9 model no ERA5 dan MME
    import xarray as xr
    dpi = xr.open_dataset("d:/Cordex/ClimWIP-Brunner_etal_2020_ESD/data/DJF_gpcp.nc")
    #pi=np.append(dpi.weights_q.data,0)
    
    pi=dpi.weights_q.data #.reshape((11, 1))
    #print(xmetrics1.shape)
    #print(pi)
    #print(pi.shape)
    xmetrics1=np.column_stack((xmetrics1,pi))
    #print(xmetrics1.shape)
    # result = pd.DataFrame(xmetrics1,model_names,columns=[
             # '1/s','r','1/rmse','Tian','Taylor1','Taylor2',
             # '1/D','cpi','H','M','PI'])
    # result.to_excel(workdir+filename+'6.xlsx')
        
    # colnames=('1/s','r','1/rmse','Tian','Taylor1','Taylor2','1/D','cpi','H','M','PI')
    
    #Taylor1,Taylor2,1/D, Tian,CPI,H,M,SS
    result = pd.DataFrame(xmetrics1,model_names,columns=[
             #'Taylor1','Taylor2','D','Tian','CPI','H','M','SS','PI'])
             'Taylor','Tian','CPI','SS','PI'])
    result.to_excel(workdir+filename+'8b.xlsx')
    #result.to_csv(workdir+filename+reg+'.csv')
    
    print(result)
    print('model_names',model_names)
    f = plt.figure()
    #ax=result.plot() #(kind='scatter', x=columns, y=xmetrics1)
    ax=result.plot(marker='o', linestyle='-')
    ax.set_xticks(np.arange(11))
    ax.set_xticklabels(model_names, rotation=45)
    #xticklabels(model_names, rotation=45)
    plt.ylabel('Skill scores')
    #plt.xlabel('Metric type')
    plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.15))
    #f.savefig(workdir+reg+'_multi_metrics_corrb.png',bbox_inches='tight')
    plt.subplots_adjust(bottom=.2, right=.7)
    plt.show()
    
    
        
    #colnames=('Taylor1','Taylor2','D','Tian','CPI','H','M','SS','PI')
    colnames=('Taylor','Tian','CPI','SS','PI')
    from sklearn import preprocessing
    d = preprocessing.normalize(result, axis=0)
    scaled_df = pd.DataFrame(d,model_names, columns=colnames)
    scaled_df.to_excel(workdir+filename+'9b.xlsx')
    
    #plot pd corr multi metrik
    c=result.corr()
    c.to_excel(workdir+filename+'_corrb.xlsx')
    
    '''
    print(c)
    f = plt.figure()
    #f.subplots_adjust(right=.7)
    ax=c.plot(kind='bar')
    xlabel=[ 'Taylor',  'Tian', 'CPI', 'SS', 'PI']
    #ax.set_xticks([])
    ax.set_xticklabels(xlabel,rotation=0)
    #ax.set_xlabel(xlabel,rotation=0, ha='right')
    #ax.legend(loc='upper left')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.15))
    #plt.xlabel(xlabel,rotation=45, ha='right')
    plt.ylabel('Correlation')
    plt.subplots_adjust(right=.7)

    
    plt.show()
    
    
    f = plt.figure()
    c.plot(ax=f.gca())
    plt.ylabel('Correlation')
    plt.xlabel('Metric type')
    plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    f.savefig(workdir+reg+'_multi_metrics_corrb.png',bbox_inches='tight')
    plt.show()
    '''

def extract_data_at_nearest_grid_point(target_dataset, longitude, latitude):
    """ Spatially subset a dataset within the given longitude and latitude boundaryd_lon-grid_space, grid_lon+grid_space
    :param target_dataset: Dataset object that needs spatial subsetting
    :type target_dataset: Open Climate Workbench Dataset Object
    :type longitude: float
    :param longitude: longitude
    :type latitude: float
    :param latitude: latitude 
    :returns: A new spatially subset Dataset
    :rtype: Open Climate Workbench Dataset Object
    """

    if target_dataset.lons.ndim == 1 and target_dataset.lats.ndim == 1:
        new_lon, new_lat = np.meshgrid(target_dataset.lons, target_dataset.lats)
    elif target_dataset.lons.ndim == 2 and target_dataset.lats.ndim == 2:
        new_lon = target_datasets.lons
        new_lat = target_datasets.lats
    distance = (new_lon - longitude)**2. + (new_lat - latitude)**2.
    y_index, x_index = np.where(distance == np.min(distance))[0:2]

    return target_dataset.values[:,y_index[0], x_index[0]]
    
def extract_data_at_nearest_grid_point2(target_dataset, longitude, latitude):
    """ Spatially subset a dataset within the given longitude and latitude boundaryd_lon-grid_space, grid_lon+grid_space
    :param target_dataset: Dataset object that needs spatial subsetting
    :type target_dataset: Open Climate Workbench Dataset Object
    :type longitude: float
    :param longitude: longitude
    :type latitude: float
    :param latitude: latitude 
    :returns: A new spatially subset Dataset
    :rtype: Open Climate Workbench Dataset Object
    """

    if target_dataset.lon.ndim == 1 and target_dataset.lat.ndim == 1:
        #print('qqq')
        new_lon, new_lat = np.meshgrid(target_dataset.lon, target_dataset.lat)
    elif target_dataset.lon.ndim == 2 and target_dataset.lat.ndim == 2:
        #print('tttt')
        new_lon = target_datasets.lon
        new_lat = target_datasets.lat
    distance = (new_lon - longitude)**2. + (new_lat - latitude)**2.
    #distance = (new_lon - longitude)*(new_lon - longitude) + (new_lat - latitude)*(new_lat - latitude)
    y_index, x_index = np.where(distance == np.min(distance))[0:2]

    return target_dataset.values[:,y_index[0], x_index[0]]
    
def Wmetric(obs_dataset, obs_name, model_datasets, model_names,file_name):
    #for MMEW
    #Fungsi ini menghitung bobot based on Tmetric                                  
    # calculate climatological mean fields
    obs_clim_dataset = ds.Dataset(obs_dataset.lats, obs_dataset.lons, obs_dataset.times, utils.calc_temporal_mean(obs_dataset))
    model_clim_datasets = []
    for dataset in model_datasets:
        model_clim_datasets.append(ds.Dataset(dataset.lats, dataset.lons, dataset.times, utils.calc_temporal_mean(dataset)))
    
    
    metrics2 = metrics.Tmetric()
    metrics2_evaluation = Evaluation(obs_clim_dataset, # Climatological mean of reference dataset for the evaluation
                                 model_clim_datasets, # list of climatological means from model datasets for the evaluation
                                 [metrics2])
    metrics2_evaluation.run() 
    metric = metrics2_evaluation.results[0]  
 
    return metric

def Wmetric_temporal(obs_dataset, obs_name, model_datasets, model_names,file_name):
    #for MMEW
    #Fungsi ini menghitung bobot based on Tmetric                                  
      
    metricst = metrics.Tmetric_temporal()
    metricst_evaluation = Evaluation(obs_dataset, # Climatological mean of reference dataset for the evaluation
                                 model_datasets, # list of climatological means from model datasets for the evaluation
                                 [metricst])
    metricst_evaluation.run() 
    metric = metricst_evaluation.results[0]  
 
    return metric

def Time_series_subregion(obs_subregion_mean, obs_name, model_subregion_mean, model_names, seasonal_cycle, 
                          file_name, row, column, x_tick=['']):
    print('Time_series_subregion')
    #khusus untuk annual cycle dengan nmon=12
    
    nmodel, nt, nregion = model_subregion_mean.shape  
    #print(nt,)
    if seasonal_cycle:
        obs_data = ma.mean(obs_subregion_mean.reshape([1,int(nt/12),12,nregion] if not config['season'] else [1,int(nt/nmon),nmon,nregion]), axis=1)
        model_data = ma.mean(model_subregion_mean.reshape([nmodel,int(nt/12),12,nregion] if not config['season'] else [nmodel,int(nt/nmon),nmon,nregion]), axis=1)
        nt = 12 if not config['season'] else nmon
    else:
        obs_data = obs_subregion_mean
        model_data = model_subregion_mean
        
    x_axis = np.arange(nt)
    x_tick_values = x_axis

    fig = plt.figure()
    
    rcParams['xtick.labelsize'] = 6
    rcParams['ytick.labelsize'] = 6
  
    for iregion in np.arange(nregion):
        ax = fig.add_subplot(row, column, iregion+1) 
        # x_tick_labels = ['']
        # if iregion+1  > column*(row-1):
            # x_tick_labels = x_tick 
        # else:
            # x_tick_labels=['']
        # print('x_tick_labels=',x_tick_labels)
        
        x_tick_labels = x_tick
        
        #plot obs tebal 2    
        ax.plot(x_axis, obs_data[0, :, iregion], color='black', lw=2, label=obs_name)   
        
        for imodel in np.arange(nmodel):
            if imodel < nmodel-2:
               ax.plot(x_axis, model_data[imodel, :, iregion], lw=0.5, label = model_names[imodel])
        #plot Ens
        ax.plot(x_axis, model_data[nmodel-2, :, iregion], color='r', lw=2, label = model_names[nmodel-2])
        #plot Ens_w
        ax.plot(x_axis, model_data[nmodel-1, :, iregion], color='b', lw=2, label = model_names[nmodel-1])
        
        ax.set_xlim([-0.5,nt-0.5])
        ax.set_xticks(x_tick_values)
        ax.set_xticklabels(x_tick_labels)
        ax.set_title('Region %02d' % (iregion+1), fontsize=8)
    
    fig.subplots_adjust(hspace=0.4, wspace=0.2)
    ax.legend(bbox_to_anchor=(1.02, 1.2), loc='best' , prop={'size':7}, frameon=False) 
    #ax.legend(bbox_to_anchor=(-0.2, row/2), loc='lower right' , prop={'size':7}, frameon=False)
    #ax.legend(bbox_to_anchor=(-0.2, row/3), loc='lower right' , prop={'size':7}, frameon=False)    
    if len(x_tick) == 12:
        fig.savefig(file_name, dpi=600, bbox_inches='tight')
    else:
        fig.savefig(file_name+'_season', dpi=600, bbox_inches='tight')
    
    # ########################################tambahan fig2 khusus MOE
    # fig2 = plt.figure()
    # for iregion in np.arange(nregion):
        # ax2 = fig2.add_subplot(row, column, iregion+1) 
        # x_tick_labels = ['']
        # if iregion+1  > column*(row-1):
            # x_tick_labels = x_tick 
        # else:
            # x_tick_labels=['']
               
        # #ax2.plot(x_axis, obs_data[0, :, iregion], color='black', lw=1, label=obs_name)
        
        
        # for imodel in np.arange(nmodel):
            # if imodel < nmodel-2:
                # ax2.plot(x_axis, model_data[imodel, :, iregion], lw=1, label = model_names[imodel])
        
        # ax2.plot(x_axis, model_data[nmodel-2, :, iregion], color='r', lw=1, label = model_names[nmodel-2])
        # #ax2.plot(x_axis, model_data[nmodel-1, :, iregion], color='b', lw=1, label = model_names[nmodel-1])
        # ax2.set_xlim([-0.5,nt-0.5])
        # ax2.set_xticks(x_tick_values)
        
        # ax2.set_xticklabels(x_tick_labels)
        # ax2.set_title('Region %02d' % (iregion+1), fontsize=8)
    
    # fig2.subplots_adjust(hspace=0.2, wspace=.2)

    # ax2.legend(bbox_to_anchor=(-0.2, row/2), loc='right' , prop={'size':7}, frameon=False)  
   
    # #fig2.savefig(file_name+'_legend', dpi=600, bbox_inches='tight')
    # if len(x_tick) == 12:
        # fig2.savefig(file_name+'_moe', dpi=600, bbox_inches='tight')
    # else:
        # fig2.savefig(file_name+'_season2', dpi=600, bbox_inches='tight')
    
def Time_series_subregion2(obs_subregion_mean, obs_name, model_subregion_mean, model_names, seasonal_cycle, 
                          file_name, row, column, x_tick=['']):
    
    nmodel, nt, nregion = model_subregion_mean.shape  
    print('Time_series_subregion2')
    if seasonal_cycle:
        obs_data = ma.mean(obs_subregion_mean.reshape([1,int(nt/12),12,nregion] if not config['season'] else [1,int(nt/nmon),nmon,nregion]), axis=1)
        model_data = ma.mean(model_subregion_mean.reshape([nmodel,int(nt/12),12,nregion] if not config['season'] else [nmodel,int(nt/nmon),nmon,nregion]), axis=1)
        nt = 12 if not config['season'] else nmon
    else:
        obs_data = obs_subregion_mean
        model_data = model_subregion_mean
        
    x_axis = np.arange(nt)
    x_tick_values = x_axis

    fig = plt.figure()
    fig2 = plt.figure()
    rcParams['xtick.labelsize'] = 6
    rcParams['ytick.labelsize'] = 6
  
    for iregion in np.arange(nregion):
        ax = fig.add_subplot(row, column, iregion+1) 
        # x_tick_labels = ['']
        # if iregion+1  > column*(row-1):
            # x_tick_labels = x_tick 
        # else:
            # x_tick_labels=['']
            
        x_tick_labels = x_tick 
        #plot obs tebal 2    
        ax.plot(x_axis, obs_data[0, :, iregion], color='black', lw=2, label=obs_name)   
        
        for imodel in np.arange(nmodel):
            if imodel < nmodel-2:
               ax.plot(x_axis, model_data[imodel, :, iregion], lw=0.5, label = model_names[imodel])
        #plot Ens
        ax.plot(x_axis, model_data[nmodel-2, :, iregion], color='r', lw=2, label = model_names[nmodel-2])
        #plot Ens_w
        ax.plot(x_axis, model_data[nmodel-1, :, iregion], color='b', lw=2, label = model_names[nmodel-1])
        
        ax.set_xlim([-0.5,nt-0.5])
        ax.set_xticks(x_tick_values)
        ax.set_xticklabels(x_tick_labels)
        ax.set_title('Region %02d' % (iregion+1), fontsize=8)
    
    fig.subplots_adjust(hspace=0.4, wspace=0.17)
    ax.legend(bbox_to_anchor=(1, 1.2), loc='best' , prop={'size':7}, frameon=False) 
    #if len(x_tick) == 12:
    fig.savefig(file_name+'_no_L', dpi=600, bbox_inches='tight')
    #else:
    #    fig.savefig(file_name+'_season', dpi=600, bbox_inches='tight')

def Portrait_diagram_subregion(obs_subregion_mean, obs_name, model_subregion_mean, model_names, seasonal_cycle,
                               file_name, normalize=True):
    
    file_name=file_name+'_'+reg+'_'+'_'+tipe
    nmodel, nt, nregion = model_subregion_mean.shape
    print('Portrait_diagram')
    if seasonal_cycle:
        #print('seasonal_cycle=',seasonal_cycle)
        obs_data = ma.mean(obs_subregion_mean.reshape([1,int(nt/12),12,nregion] if not config['season'] else [1,int(nt/nmon),nmon,nregion]), axis=1)
        model_data = ma.mean(model_subregion_mean.reshape([nmodel,int(nt/12),12,nregion] if not config['season'] else [nmodel,int(nt/nmon),nmon,nregion]), axis=1)
        nt = 12 if not config['season'] else nmon
    else:
        obs_data = obs_subregion_mean
        model_data = model_subregion_mean
        #print('not seasonal_cycle')

    subregion_metrics = ma.zeros([4, nregion, nmodel])

    for imodel in np.arange(nmodel):
        for iregion in np.arange(nregion):
            # First metric: bias
            subregion_metrics[0, iregion, imodel] = metrics.calc_bias(model_data[imodel, :, iregion], obs_data[0, :, iregion], average_over_time = True)
            # Second metric: standard deviation
            subregion_metrics[1, iregion, imodel] = metrics.calc_stddev_ratio(model_data[imodel, :, iregion], obs_data[0, :, iregion])
            # Third metric: RMSE
            subregion_metrics[2, iregion, imodel] = metrics.calc_rmse(model_data[imodel, :, iregion], obs_data[0, :, iregion])
            # Fourth metric: correlation
            subregion_metrics[3, iregion, imodel] = metrics.calc_correlation(model_data[imodel, :, iregion], obs_data[0, :, iregion])
   
    if normalize:
        print('Portrait_diagram_normalize')
        for iregion in np.arange(nregion):
            subregion_metrics[0, iregion, : ] = subregion_metrics[0, iregion, : ]/ma.std(obs_data[0, :, iregion])*100. 
            subregion_metrics[1, iregion, : ] = subregion_metrics[1, iregion, : ]*100. 
            subregion_metrics[2, iregion, : ] = subregion_metrics[2, iregion, : ]/ma.std(obs_data[0, :, iregion])*100. 

    region_names = ['R%02d' % i for i in np.arange(nregion)+1]
    
    ##tambahan judul: ptitle=pt[imetric]
    pt=['bias','std','RMSE','corr']
    ##
    #for imetric, metric in enumerate(['bias','std','RMSE','corr']):
        
    #    plotter2.draw_portrait_diagram(subregion_metrics[imetric, :, :], region_names, model_names, file_name+'_'+metric, 
    #                                  xlabel='model', ylabel='region', ptitle=pt[imetric])             
    for imetric, metric in enumerate(['bias','std','RMSE','corr']):
            
            plotter2.draw_portrait_diagram2(subregion_metrics[imetric, :, :], region_names, model_names, file_name+'_'+metric, 
                                          xlabel='model', ylabel='region', ptitle=pt[imetric])  
                                          
                                          
def Map_plot_subregion(subregions, ref_dataset, directory):
    #subregions [-6.5, 6, 95, 107]
    lons, lats = np.meshgrid(ref_dataset.lons, ref_dataset.lats) 
    fig = plt.figure()
    ax = fig.add_subplot(111)
    m = Basemap(ax=ax, projection='cyl',llcrnrlat = lats.min(), urcrnrlat = lats.max(),
                llcrnrlon = lons.min(), urcrnrlon = lons.max(), resolution = 'h')
    m.drawcoastlines(linewidth=0.75)
    m.drawcountries(linewidth=0.75)
    m.etopo()  
    x, y = m(lons, lats) 
    #subregion_array = ma.masked_equal(subregion_array, 0)
    #max=m.contourf(x, y, subregion_array, alpha=0.7, cmap='Accent')
    for subregion in subregions:
        draw_screen_poly(subregion[1], m, 'w') 
        plt.annotate(subregion[0],xy=(0.5*(subregion[1][2]+subregion[1][3]), 0.5*(subregion[1][0]+subregion[1][1])), 
                      ha='center',va='center', fontsize=5,
                      backgroundcolor='0.90',alpha=1) 
    fig.savefig(directory+'map_subregion', bbox_inches='tight')

def draw_screen_poly(boundary_array, m, linecolor='red'):

    print('draw_screen_poly...')
    ''' Draw a polygon on a map

    :param boundary_array: [lat_north, lat_south, lon_east, lon_west]
    :param m   : Basemap object
    '''

    lats = [boundary_array[0], boundary_array[0], boundary_array[1], boundary_array[1]]
    lons = [boundary_array[3], boundary_array[2], boundary_array[2], boundary_array[3]]
    
        
    #x, y = m( lons, lats )
    x=np.array(lons)
    y=np.array(lats) 
    xy = np.column_stack((x,y))
        
    #print('x,y=',x,y)
    #xy = zip(x,y)
    #print('xy=',xy)
    #off kan 2 ini jika error not ok yet
    poly = Polygon( xy, 
                    facecolor='none', 
                    #facecolor='red', 
                    edgecolor=linecolor )
    plt.gca().add_patch(poly)
    
