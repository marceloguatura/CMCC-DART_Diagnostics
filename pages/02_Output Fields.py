# from itertools import product
# from re import S
# from tkinter import TRUE, Variable
import streamlit as st
import matplotlib.pyplot as plt
import xarray as xr
import cartopy, cartopy.crs as ccrs
import cartopy.feature as cfeature
# import matplotlib.colors as colors
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from PIL import Image
# import glob
# import warnings
# import os
from PIL import Image
import numpy as np
# import time
# import io
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as mcolors
from cycler import cycler
from streamlit.elements.utils import _shown_default_value_warning
_shown_default_value_warning = False
from matplotlib.gridspec import GridSpec
import json
import pandas as pd
from shapely.geometry import mapping
import rioxarray
# "description": 
import init

init.load()

# homepath = '/work/csp/mg20022/charts/CESM-DART'
img = Image.open(f"{st.session_state['homepath']}/.thumb.jpg")

st.set_page_config(page_title='CMCC-DART', page_icon=img, layout="wide") # , layout="wide"


# Remove menu 
# #MainMenu {visibility: hidden; }
hide_menu = """
        <style>
        footer {visibility: hidden;}
        footer:after{
            visibility: visible;
            content: 'Copyright @ 2022: CMCC';
            color: gray;
            position:relative;
        }
        </style>
        """
st.markdown(hide_menu, unsafe_allow_html=True)

with open(f"{st.session_state['homepath']}/data/config.json", 'r') as f:

    st.session_state['expander'] = json.load(f)

def mask(f):
    f.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
    f.rio.write_crs("epsg:4326", inplace=True)
    f.rio.write_grid_mapping(inplace=True)


    match st.session_state['maskover']:

        case 'Continents':
            f = f.rio.clip(st.session_state['f']['shape'].geometry.apply(mapping), st.session_state['f']['shape'].crs, invert=False, all_touched=False, drop=True)
        case 'Ocean':
            f = f.rio.clip(st.session_state['f']['shape'].geometry.apply(mapping), st.session_state['f']['shape'].crs, invert=True, all_touched=False, drop=True)
        case _:
            pass

    return f.isel(lon=slice(1,-1))

def reduce_domain(f, lineplot=False):

    if lineplot == False:

        match st.session_state['domain']:
            case 'Global':
                pass
            case 'NH':
                f = f.sel(lat=slice(30, 90))#.mean(dim=['lat'])
            case 'SH':
                f = f.sel(lat=slice(-90, -30))#.mean(dim=['lat'])
            case 'Tropics':
                f = f.sel(lat=slice(-30, 30))#.mean(dim=['lat'])
            case _:
                f = f.sel(lat=slice(st.session_state['lat'][0], st.session_state['lat'][1]), 
                        lon=slice(st.session_state['lon'][0], st.session_state['lon'][1]))
        return f
    
    else:
        arr = {}
        for area in ['Global', 'NH', 'SH', 'Tropics', 'Box']:
            
            match area:
                case 'Global':
                    arr[area] = f.mean(dim=['lat', 'lon']).copy()
                case 'NH':
                    arr[area] = f.sel(lat=slice(30, 90)).mean(dim=['lat', 'lon']).copy()
                case 'SH':
                    arr[area] = f.sel(lat=slice(-90, -30)).mean(dim=['lat', 'lon']).copy()
                case 'Tropics':
                    arr[area] = f.sel(lat=slice(-30, 30)).mean(dim=['lat', 'lon']).copy()
                case 'Box':
                    arr[area] = f.sel(lat=slice(st.session_state['lat'][0], st.session_state['lat'][1]), 
                                      lon=slice(st.session_state['lon'][0], st.session_state['lon'][1])).mean(dim=['lat', 'lon']).copy()
        return arr


# Dont remove
# def _update_slider():
#     st.session_state['frame'] = st.session_state['frame2']

# Dont remove
# def animate(**kwargs):

#     if st.session_state['plot'] == 'Shaded':
#         if st.session_state['session_change'] == False:

#             # if st.session_state['product'] == 'OUTPUT':
#             #     f = st.session_state['f'][st.session_state['fstexp']]['default'][st.session_state['menuoutput']]
#             # else:
#             #     if st.session_state['fstexp'] == st.session_state['compare']:
#             #         f = st.session_state['f'][st.session_state['fstexp']]['PP']['Analysis Increments']
#             #     else:
#             #         f = st.session_state['f'][st.session_state['fstexp']]['PP'][st.session_state['compare']]
            
#             match st.session_state['product']:
#                 case 'OUTPUT':
#                     f = st.session_state['f'][st.session_state['fstexp']]['default'][st.session_state['menuoutput']]
#                 case 'MODEL SPACE':
#                     if st.session_state['compare'] == 'Analysis Increments':
#                         f = st.session_state['f'][st.session_state['fstexp']]['PP']['Analysis Increments']
#                     else:
#                         f = st.session_state['f'][st.session_state['fstexp']]['PP'][st.session_state['compare2']]
            
#             f = mask(f)
#             f = reduce_domain(f)
            
#             dt = str(f.coords['time'].values[st.session_state['frame']])[:16]
#             level = f.sel(lev=st.session_state['level'], method='nearest').lev

#             if 'Temperature' in st.session_state['variables']:
#                 pos = st.session_state['variables'].index('Temperature')
#                 st.session_state['figure'][pos].set_array(f.sel(time=dt,lev=level).T.values.flatten())
#                 st.session_state['axes'][pos].set_title(f"Time = {dt}, Lev = {level}", fontsize=11)

#             if 'Surface Pressure' in st.session_state['variables']:
#                 pos = st.session_state['variables'].index('Surface Pressure')
#                 st.session_state['figure'][pos].set_array(f.sel(time=dt).PS.values.flatten())
#                 st.session_state['axes'][pos].set_title(f"Time = {dt}", fontsize=11)

#             if 'Zonal Wind' in st.session_state['variables']:
#                 pos = st.session_state['variables'].index('Zonal Wind')
#                 st.session_state['figure'][pos].set_array(f.sel(time=dt,lev=level).US.values.flatten())
#                 st.session_state['axes'][pos].set_title(f"Time = {dt}, Lev = {level}", fontsize=11)

#             if 'Meridional Wind' in st.session_state['variables']:
#                 pos = st.session_state['variables'].index('Meridional Wind')
#                 st.session_state['figure'][pos].set_array(f.sel(time=dt,lev=level).VS.values.flatten())
#                 st.session_state['axes'][pos].set_title(f"Time = {dt}, Lev = {level}", fontsize=11)

#             if 'Specific Humidity' in st.session_state['variables']:
#                 pos = st.session_state['variables'].index('Specific Humidity')
#                 st.session_state['figure'][pos].set_array(f.sel(time=dt,lev=level).Q.values.flatten())
#                 st.session_state['axes'][pos].set_title(f"Time = {dt}, Lev = {level}", fontsize=11)
            
#             st.session_state['figout'].pyplot(st.session_state['fig'])
          
def get_experiment():

    if st.session_state['load'] == True:

        menuexp = st.sidebar.selectbox(
            "CESM2 Experiments",
               st.session_state['name'] , index=0
        )
        # st.sidebar.markdown(f"## Experiment load {st.session_state['fstexp']}")
        maskover = st.sidebar.radio('Maskout:',['None', 'Ocean', 'Continents'], index=0) # 'OBS SPACE'
        domain = st.sidebar.radio('Domain:', ['Global', 'NH', 'SH', 'Tropics', 'Box'], index=0)
        if domain == 'Box':
            st.session_state['lon'] = st.sidebar.select_slider('Longitude range:', options=np.arange(-180, 180, 0.25), value=[-180,179.75])
            st.session_state['lat'] = st.sidebar.select_slider('Latitude range:', options=np.arange(-90, 91, 1), value=[-90,90])
            st.session_state['session_change'] = True

        st.session_state['product'] = st.sidebar.radio('Section:',['OUTPUT', 'MODEL SPACE'], index=0) #, label_visibility='collapsed' 'OBS SPACE'
        st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)

        match st.session_state['product']:

            case 'OUTPUT':

                menuoutput = st.sidebar.selectbox(
                    "Select Products",
                    [
                        "Background",
                        "Analysis",
                        "Analysis Inflation",
                        "Background Inflation",
                        "Background Standard Deviation",
                        "Analysis Standard Deviation",
                        "Analysis Inflation Standard Deviation",
                        "Background Inflation Standard Deviation"], index=0
                )
                # st.session_state['menuoutput'] = menuoutput
                print(f" -------------------- menu: {menuoutput} session {st.session_state['session']}")
                
                __menuMaps__(menuoutput=menuoutput, compare=menuexp, 
                             maskover=maskover, domain=domain, fstexp=menuexp)

            case 'MODEL SPACE':

                compare = st.sidebar.selectbox(
                    "See the Increments or compare Analysis",
                    ['Analysis Increments', 'Analysis'], index=0
                )

                if compare == 'Analysis Increments':
                    compare2 = None

                    menuoutput = st.sidebar.selectbox(
                        "Select Products",
                        [
                            "Analysis Increments",
                            "Analysis Increments probability density distributions", 
                            "Analysis Increments cross section"], index=0
                    )

                    print(f" -------------------- menu: {menuoutput} session {st.session_state['session']}")

                    if (menuoutput == "Analysis Increments") | (menuoutput == "Difference"): 
                        __menuMaps__(menuoutput=menuoutput, compare=compare, maskover=maskover, domain=domain, fstexp=menuexp, compare2=compare2)
                    elif menuoutput != "":
                        __menuLonlev__(menuoutput=menuoutput, compare=compare, maskover=maskover, domain=domain, fstexp=menuexp, compare2=compare2)
                    else:
                        st.empty()

                else:
                    # val = st.session_state['f'].values()
                    # newmenu = st.session_state['name'].copy()
                    # newmenu.remove(menuexp)
                    newmenu = []
                    for nm in st.session_state['name']:
                        
                        key_list = list(st.session_state['f'][nm]['PP'])
                        if 'ERA5-Reanalysis' in key_list: 
                            if 'ERA5-Reanalysis' not in newmenu:
                                newmenu = newmenu + ['ERA5-Reanalysis']
                        # if 'NCEP-Reanalysis' in key_list: 
                        #     if 'NCEP-Reanalysis' not in newmenu:
                        #         newmenu = newmenu + ['NCEP-Reanalysis']
                    

                    if not newmenu:
                        
                        st.error('Load reanalysis with an experiment to compare it.', icon="🚨")

                    else:

                        compare2 = st.sidebar.selectbox(
                            'Select Reanalysis', newmenu
                        )

                        menuoutput = st.sidebar.selectbox(
                            "Select Products",
                            [
                                "Difference",
                                "Difference probability density distributions", 
                                "Difference cross section"], index=0
                        )       

                        print(f" -------------------- menu: {menuoutput} session {st.session_state['session']}")

                        if (menuoutput == "Analysis Increments") | (menuoutput == "Difference"): 
                            __menuMaps__(menuoutput=menuoutput, compare=compare, maskover=maskover, domain=domain, fstexp=menuexp, compare2=compare2)
                        elif menuoutput != "":
                            __menuLonlev__(menuoutput=menuoutput, compare=compare, maskover=maskover, domain=domain, fstexp=menuexp, compare2=compare2)
                        else:
                            st.empty()


def __menuLonlev__(menuoutput=None, domain=None, compare=None, maskover=None, fstexp=None, compare2=None):

    variables = st.sidebar.multiselect(
                            'Select variables:',
                            ['Temperature', 'Specific Humidity', 'Meridional Wind', 
                            'Zonal Wind'], default=['Temperature'])

    st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)

    if len(variables) > 0:

        cyclerange = st.select_slider('Select start and end cycle:', value=[0,4],
                            options=np.arange(0,len(st.session_state['f'][st.session_state['fstexp']]['default']['Background'].coords['time']),1)) #, on_change=animate()

    # if (variables != st.session_state['variables']) | (st.session_state['domain'] != domain) | (st.session_state['menuoutput'] != menuoutput) | (st.session_state['session_change'] == True):

        st.session_state['session_change'] = False
        st.session_state['varlen'] = len(variables)

        match st.session_state['varlen']:
            case 1: 
                st.session_state['fig'], st.session_state['axes'] = plt.subplots(ncols=1, nrows=1, figsize=(12,5), gridspec_kw = {'wspace':0.05, 'hspace':0.13})
            case 2: 
                st.session_state['fig'], st.session_state['axes'] = plt.subplots(ncols=1, nrows=2, figsize=(12, 10), gridspec_kw = {'wspace':0.05, 'hspace':0.2}, 
                                                                                sharex=True, sharey=True)
            case 3: 
                st.session_state['fig'], st.session_state['axes'] = plt.subplots(ncols=1, nrows=3, figsize=(12, 12), gridspec_kw = {'wspace':0.1, 'hspace':0.25}, 
                                                                                sharex=True, sharey=True)
            case 4: 
                st.session_state['fig'], st.session_state['axes'] = plt.subplots(ncols=1, nrows=4, figsize=(12, 15), gridspec_kw = {'wspace':0.1, 'hspace':0.35}, 
                                                                                sharex=True, sharey=True)

        if st.session_state['varlen'] > 1:
            axes = st.session_state['axes'].flatten()
            st.session_state['axes'] = axes
        else:
            axes = {}
            axes[0] = st.session_state['axes']
            st.session_state['axes'] = axes
        
        st.session_state['menuoutput'] = menuoutput
        st.session_state['domain'] = domain
        st.session_state['variables'] = variables
        st.session_state['figout'] = st.empty()
        st.session_state['maskover'] = maskover
        st.session_state['domain'] = domain
        st.session_state['compare'] = compare
        st.session_state['compare2'] = compare2 
        st.session_state['fstexp'] = fstexp

        __plotLonlev__(variables=st.session_state['variables'], level=st.session_state['level'],     
                       menuoutput=st.session_state['menuoutput'], session=st.session_state['session'],
                       cycles=cyclerange)

    else:
        st.empty()

def __plotLonlev__(variables=None, level=None, menuoutput=None, session=None, cycles=None):

    start = str(st.session_state['f'][st.session_state['fstexp']]['PP']['Analysis Increments'].coords['time'].values[cycles[0]])[:16]
    end   = str(st.session_state['f'][st.session_state['fstexp']]['PP']['Analysis Increments'].coords['time'].values[cycles[1]])[:16]

    if 'cross section' in st.session_state['menuoutput']:

        match st.session_state['compare']: 
            case 'Analysis Increments':
                f = st.session_state['f'][st.session_state['fstexp']]['PP']['Analysis Increments']

            case 'Analysis':
                f = st.session_state['f'][st.session_state['fstexp']]['PP'][st.session_state['compare2']]

        if f'{cycles[0]}' != f'{cycles[1]}':
            f = f.isel(time=slice(cycles[0], cycles[1])).mean(dim='time').copy()
        else:
            f = f.isel(time=cycles[0]).copy()

        f = mask(f)
        f = reduce_domain(f)
        f = f.mean(dim=['lat'])

        map = st.sidebar.radio('Profile:',
                                ['Longitudinal', 'Vertical'], index=0)

        match map:

            case "Longitudinal":

                st.session_state['filtering'] = st.sidebar.radio('Fixed range:',
                                                                ['None', 'Between -2.5 and 2.5'], index=0)

                for var in st.session_state['variables']:

                    pos = st.session_state['variables'].index(var)

                    match var:
                        case 'Temperature':
                            v = 'T'
                        case 'Specific Humidity':
                            v = 'Q'
                        case 'Meridional Wind':
                            v = 'VS'
                        case 'Zonal Wind':
                            v = 'US'

                    match st.session_state['filtering']:
                        
                        case 'None':

                            if var == 'Specific Humidity':

                                vmin = f[v].min(skipna=True).values
                                vmax = f[v].max(skipna=True).values
                                lev = 7

                                if (vmin*-1) > vmax:
                                    vmax = vmin*-1
                                else:
                                    vmin = vmax*-1

                                levels = np.linspace(vmin,vmax,lev)
                                # clevels = np.delete(levels, np.where(levels == 0.00001))
                                clevels = levels

                                p = [vmin, -0.00001, 0.00001, vmax]
                                ff = lambda x: np.interp(x, p, [0, 0.5, 0.5, 1])

                                cmap = LinearSegmentedColormap.from_list('map_white', 
                                            list(zip(np.linspace(0,1), plt.cm.seismic_r(ff(np.linspace(min(p), max(p)))))))
                                cmap.set_extremes(under='darkred', over='midnightblue')
                            
                            else:

                                vmin = f[v].min(skipna=True).values
                                vmax = f[v].max(skipna=True).values

                                lev = 21

                                if (vmin*-1) > vmax:
                                    vmax = vmin*-1
                                else:
                                    vmin = vmax*-1

                                levels = np.linspace(vmin,vmax,lev)
                                clevels = np.delete(levels, np.where(levels == 0))
                                if vmax > 0.8 : p = [vmin, -0.25, 0.25, vmax]
                                if vmax <= 0.8: p = [vmin, -0.1, 0.1, vmax]
                                ff = lambda x: np.interp(x, p, [0, 0.5, 0.5, 1])

                                cmap = LinearSegmentedColormap.from_list('map_white', 
                                            list(zip(np.linspace(0,1), plt.cm.seismic_r(ff(np.linspace(min(p), max(p)))))))
                                cmap.set_extremes(under='darkred', over='midnightblue')
                        
                        case _:

                            vmin = -2.5
                            vmax = 2.5
                            lev = 21

                            levels = np.linspace(vmin,vmax,lev)
                            clevels = np.delete(levels, np.where(levels == 0))
                            
                            p = [vmin, -0.25, 0.25, vmax]
                            ff = lambda x: np.interp(x, p, [0, 0.5, 0.5, 1])

                            cmap = LinearSegmentedColormap.from_list('map_white', 
                                        list(zip(np.linspace(0,1), plt.cm.seismic_r(ff(np.linspace(min(p), max(p)))))))
                            cmap.set_extremes(under='darkred', over='midnightblue')

                    st.session_state['figure'][pos] = f[v].plot.contourf(ax=st.session_state['axes'][pos], add_colorbar=True, 
                                                                                    cbar_kwargs={"orientation": "vertical", 
                                                                                    "pad": 0.008, "aspect": 35, "extend": 'both',
                                                                                    'spacing': 'uniform'},
                                                                                    yincrease=False, extend='both',  cmap=cmap, levels=levels) # 

                    st.session_state['figure'][pos] = f[v].plot.contour(ax=st.session_state['axes'][pos], colors='k', linewidths=0.4,
                                                                                    yincrease=False, vmin=vmin, vmax=vmax, levels=clevels)
                                            
                    plt.clabel(st.session_state['figure'][pos], inline=True, fontsize=7)
                    st.session_state['axes'][pos].set(yscale="log")
                    st.session_state['axes'][pos].set_ylabel('')
                    st.session_state['axes'][pos].set_title(f"{var} - {st.session_state['domain']} - {st.session_state['fstexp'].upper()} \n {start} to {end}", fontsize=11)

                print(st.session_state['fstexp'])
                st.session_state['fig'].subplots_adjust(top=0.95)
                st.session_state['figout'].pyplot(st.session_state['fig'])

            case "Vertical":

                fig, axes = plt.subplots(ncols=1, nrows=1, figsize=(11,6), gridspec_kw = {'wspace':0.05, 'hspace':0.13})
                # axes.set_xlim(-1.5,1.5)
                f = f.mean(dim=['lon'])
                figure = plt.plot([0,0],[992.5561,0], linestyle='dashed', color='black', linewidth=0.7)

                if 'Temperature' in st.session_state['variables']:
                    figure = f.T.plot(ax=axes, yincrease=False, y='lev', label='Temperature', color='red', linewidth=0.9)

                if 'Zonal Wind' in st.session_state['variables']:
                    figure = f.US.plot(ax=axes, yincrease=False, y='lev', label='Zonal Wind', color='blue', linewidth=0.9)

                if 'Meridional Wind' in st.session_state['variables']:
                    figure = f.VS.plot(ax=axes, yincrease=False, y='lev', label='Meridional Wind', color='green', linewidth=0.9)

                if 'Specific Humidity' in st.session_state['variables']:
                    figure = f.Q.plot(ax=axes,yincrease=False, y='lev', label='Specific Humidity', color='brown', linewidth=0.9)
                
                fig.subplots_adjust(top=0.95)
                axes.set_ylabel('')
                axes.grid()
                axes.set_title(f"{st.session_state['domain']} - {st.session_state['fstexp'].upper()} \n {start} to {end}", fontsize=11)
                axes.set(yscale="log")
                plt.legend(loc='best')
                st.session_state['figout'].pyplot(fig)
        
    if 'probability density distributions' in st.session_state['menuoutput']:

        st.session_state['over'] = st.sidebar.radio('Frequency:',
            ['Cycles', 'Mean by day', 'Mean by time of day'], index=0)

        match st.session_state['compare']: 
            case 'Analysis Increments':
                f = st.session_state['f'][st.session_state['fstexp']]['PP']['Analysis Increments']

            case 'Analysis':
                f = st.session_state['f'][st.session_state['fstexp']]['PP'][st.session_state['compare2']]

        # if st.session_state['name'][0] == st.session_state['compare']:
        #     f = st.session_state['f'][st.session_state['name'][0]]['PP']['Analysis Increments']
        # else:
        #     f = st.session_state['f'][st.session_state['name'][0]]['PP'][st.session_state['compare']]

        if f'{cycles[0]}' != f'{cycles[1]}':
            f = f.isel(time=slice(cycles[0], cycles[1])).copy()
        else:
            f = f.isel(time=cycles[0]).copy()
            f = f.expand_dims('time')

        f = mask(f)
        f = reduce_domain(f)
        f = f.mean(dim=['lat'])

        match st.session_state['over']:
            case 'Cycles':
                hue_dim = 'time'
            case 'Mean by day':
                f = f.resample(time='1D').mean()
                hue_dim = 'time'
            case 'Mean by time of day':
                f = f.groupby('time.hour').mean()
                hue_dim = 'hour'

        match st.session_state['varlen']:

            case 1: 
                st.session_state['fig'] = plt.figure(figsize=(8, 10))#, layout="constrained"
                st.session_state['fig'].subplots_adjust(bottom=0.05, left=0.025, top = 0.95, right=0.95)
                spec = st.session_state['fig'].add_gridspec(2, 6, wspace=0.75, hspace=0.25)

                st.session_state['axes'][0] = st.session_state['fig'].add_subplot(spec[0, :2]) # , sharex=st.session_state['axes'][1]
                st.session_state['axes'][1] = st.session_state['fig'].add_subplot(spec[0, 2:])
                st.session_state['axes'][2] = st.session_state['fig'].add_subplot(spec[1, :2])
                st.session_state['axes'][3] = st.session_state['fig'].add_subplot(spec[1, 2:])
            
            case _:
                
                st.session_state['axes'] = {}

                st.session_state['fig'] = plt.figure(figsize=(8, 8*st.session_state['varlen']))#, layout="constrained"
                st.session_state['fig'].subplots_adjust(bottom=0.05, left=0.025, top = 0.95, right=0.95)
                gs = st.session_state['fig'].add_gridspec(st.session_state['varlen']*2, 6, wspace=0.75, hspace=0.25)
                j = 0
                n = 0
                for i in np.arange(0,4*st.session_state['varlen'],4):
                    st.session_state['axes'][i] = st.session_state['fig'].add_subplot(gs[j, :2]) # , sharex=st.session_state['axes'][1]
                    st.session_state['axes'][i+1] = st.session_state['fig'].add_subplot(gs[j, 2:])
                    st.session_state['axes'][i+2] = st.session_state['fig'].add_subplot(gs[j+1, :2])
                    st.session_state['axes'][i+3] = st.session_state['fig'].add_subplot(gs[j+1, 2:])
                    j+=2
                    n+=1
        
        st.session_state['filtering'] = st.sidebar.radio('Filtering:',
            ['None', 'Between -0.25 and 0.25'], index=0)

        vmin = -2.5
        vmax = 2.5

        levels = np.linspace(vmin,vmax,21)
        clevels = np.delete(levels, np.where(levels == 0))
        ticks = [2.5, 2, 1.5, 1, 0.5, 0, -0.5, -1, -1.5, -2, -2.5]

        cmap = 'gist_rainbow_r' # brg  dist_rainbow

        for var in st.session_state['variables']:

            pos = st.session_state['variables'].index(var)

            match var:
                case 'Temperature':
                    v = 'T'
                case 'Specific Humidity':
                    v = 'Q'
                case 'Meridional Wind':
                    v = 'VS'
                case 'Zonal Wind':
                    v = 'US'

            if pos == 0: x0=0; x1=1; x2=2; x3=3
            if pos == 1: x0=4; x1=5; x2=6; x3=7 
            if pos == 2: x0=8; x1=9; x2=10; x3=11
            if pos == 3: x0=12; x1=13; x2=14; x3=15

            df = f[v].to_dataframe().reset_index()

            if st.session_state['filtering'] == 'None': 
                lim = 0 
            else: 
                lim = .25

            idex1 = np.where(df[v] > lim)
            df1 = df.loc[idex1]

            idex2 = np.where(df[v] < lim * -1)
            df2 = df.loc[idex2]

            df1 = df1.set_index([hue_dim, 'lon', 'lev'])
            df2 = df2.set_index([hue_dim, 'lon', 'lev'])

            if st.session_state['filtering'] != 'None': 
                df = pd.concat([df1, df2])

            # #  OFICIAL FIGURE 0 SPLIT POSITIVES AND NEGATIVES
            # sns.kdeplot(data=df1, x=v, color='b',
            #             ax=st.session_state['axes'][x0], label=f'{v} > {lim}') # hue='time', multiple='stack', alpha=.9
            # # st.session_state['axes'][x0].set_xlim(vmin,vmax)
            # sns.kdeplot(data=df2, x=v, color='r',
            #             ax=st.session_state['axes'][x0], label=f'{v} < {lim*-1}') # hue='time', multiple='stack', alpha=.9
            # st.session_state['axes'][x0].set_title(f"{st.session_state['name'][0].upper()}", fontsize=13, loc='left', alpha=.8)
            # st.session_state['axes'][x0].grid(True, alpha=.7)
            # st.session_state['axes'][x0].legend(loc='upper left', prop={'size': 8})

            #  OFICIAL FIGURE 0
            sns.kdeplot(data=df, x=v, color='black',
                        ax=st.session_state['axes'][x0], label=f'{v}')
            st.session_state['axes'][x0].set_title(f"{st.session_state['fstexp'].upper()}", fontsize=13, loc='left', alpha=.8)
            st.session_state['axes'][x0].grid(True, alpha=.7)
            st.session_state['axes'][x0].legend(loc='upper left', prop={'size': 8})
            
            #  OFICIAL FIGURE 1
            if pos == 0 :
                sns.scatterplot(data=df, x=v, y='lev', hue=hue_dim, alpha=0.3,
                                ax=st.session_state['axes'][x1],marker='o', size=0.7, 
                                legend='full', edgecolor = 'none', palette=cmap)
                                        
                box = st.session_state['axes'][x1].get_position()
                st.session_state['axes'][x1].set_position([box.x0, box.y0, box.width, box.height])                 
                st.session_state['axes'][x1].legend(loc='center left', bbox_to_anchor=(1, 0), 
                                                    ncol=1, fancybox=True, shadow=False, fontsize=9)
            else:
                sns.scatterplot(data=df, x=v, y='lev', hue=hue_dim, alpha=0.3,
                                ax=st.session_state['axes'][x1],marker='o', size=0.7, 
                                legend=False, edgecolor = 'none', palette=cmap)
                                        
            st.session_state['axes'][x1].set(yscale="log")
            st.session_state['axes'][x1].invert_yaxis()
            st.session_state['axes'][x1].set(ylabel="")
            st.session_state['axes'][x1].set_title(f"{var}", fontsize=15, loc='left', alpha=.8)
            st.session_state['axes'][x1].grid(True, alpha=.7)

            # st.session_state['axes'][x1].set(yincrease=False)
            # st.session_state['axes'][x1].set_ylim(bottom=-2)

            #  OFICIAL FIGURE 2
            sns.ecdfplot(data=df, x=v, ax=st.session_state['axes'][x2],
                            color='k', stat='proportion', 
                            linewidth=1.5, legend=False) # hue="hour", palette='Dark2'alpha=.7,
            # st.session_state['axes'][x2].set_xlim(vmin,vmax)
            st.session_state['axes'][x2].set_ylim(0,1.02)
            st.session_state['axes'][x2].set_title(f"{st.session_state['domain']}", fontsize=15, loc='left', alpha=.8)
            st.session_state['axes'][x2].grid(True, alpha=.7)
            
        
            #  OFICIAL FIGURE 3
            df = df.reset_index().groupby([hue_dim, 'lev']).mean().drop('lon', axis=1)
            sns.scatterplot(data=df, x=v, y='lev', hue=hue_dim, alpha=0.8,
                            ax=st.session_state['axes'][x3], marker='o', size=0.9, 
                            legend=False, edgecolor = 'none', palette=cmap)
            st.session_state['axes'][x3].set(yscale="log")
            st.session_state['axes'][x3].invert_yaxis()
            st.session_state['axes'][x3].set(ylabel="") 
            st.session_state['axes'][x3].set_title(f"{start} to {end}", fontsize=15, loc='left', alpha=.8)
            st.session_state['axes'][x3].grid(True, alpha=.7)

        st.pyplot(st.session_state['fig'])

def __menuMaps__(menuoutput=None, compare=None, maskover=None, domain=None, fstexp=None, compare2=None):
    
    pt = st.sidebar.radio('Plot:', ['Shaded', 'Lineplot'])

    if pt == 'Shaded':

        
        st.slider('Cycle:', min_value=0, key="frame",
                            value=st.session_state['frame2'],
                            max_value=len(st.session_state['f'][fstexp]['default']['Background'].coords['time']),
                            step=1)

    else:
        level = 0.0
        st.slider('Cycle:', min_value=0, key="frame",
                            value=st.session_state['frame2'],
                            max_value=len(st.session_state['f'][fstexp]['default']['Background'].coords['time']),
                            step=1)
    
    
    match st.session_state['product']:
        case 'OUTPUT':

            if pt == 'Shaded':
                variables = st.sidebar.multiselect(
                                        'Select variables:',
                                        ['Temperature', 'Specific Humidity', 'Meridional Wind', 
                                        'Zonal Wind', 'Surface Pressure'], default=['Temperature'])

                level = st.sidebar.selectbox(
                                    "Select Level",
                                    ['992','850','500','250','150','50','3', '1', '0.1', '0.01'], index=0
                                )
                st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
                # level = st.sidebar.selectbox(
                #                 "Select Level",
                #                 ['992','850','500','250','150','50','3', '1', '0.1', '0.01'], index=0
                #             )
                # st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
            else:
                variables = st.sidebar.multiselect(
                        'Select variables:',
                        ['Temperature', 'Specific Humidity', 'Meridional Wind', 
                        'Zonal Wind'], default=['Temperature'])

        case 'MODEL SPACE':
            
            variables = st.sidebar.multiselect(
                                    'Select variables:',
                                    ['Temperature', 'Specific Humidity', 'Meridional Wind', 
                                    'Zonal Wind'], default=['Temperature'])
            if pt == 'Shaded':
                level = st.sidebar.selectbox(
                                "Select Level",
                                ['1000', '975', '950', '925', '900', '875', 
                                '850', '825', '800', '775', '750', '700', 
                                '650', '600', '550', '500', '450', '400', 
                                '350', '300', '250', '225', '200', '175', 
                                '150', '125', '100', '70', '50', '30', 
                                '20', '10', '7', '5', '3', '2', '1'], index=0
                        )
            else:
                level = 0.0
            st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)    
    
    if len(variables) > 0:

    # if (variables != st.session_state['variables']) | (st.session_state['level'] != float(level)) | \
    #    (st.session_state['menuoutput'] != menuoutput) | (st.session_state['session_change'] == True ) | \
    #    (st.session_state['compare'] != compare) | (st.session_state['maskover'] != maskover) | \
    #    (st.session_state['domain'] != domain) | (st.session_state['fstexp'] != fstexp) | \
    #    (st.session_state['compare2'] != compare2) | (st.session_state['plot'] != pt) | (pt == 'Lineplot'):

        st.session_state['session_change'] = False
        st.session_state['varlen'] = len(variables)


        
        st.session_state['fstexp'] = fstexp
        st.session_state['plot'] = pt
        st.session_state['maskover'] = maskover
        st.session_state['domain'] = domain
        st.session_state['menuoutput'] = menuoutput
        st.session_state['compare'] = compare
        st.session_state['level'] = float(level)
        st.session_state['variables'] = variables
        st.session_state['figout'] = st.empty()
        st.session_state['compare2'] = compare2

        __plotMap__(variables=st.session_state['variables'], level=st.session_state['level'],     
                    menuoutput=st.session_state['menuoutput'], session=st.session_state['session'])
  
def __plotMap__(variables=None, level=None, menuoutput=None, session=None):

    # st.info("Page under constrution!", icon="ℹ️")

    match st.session_state['product']:
        case 'OUTPUT':
            f = st.session_state['f'][st.session_state['fstexp']]['default'][st.session_state['menuoutput']]
        case 'MODEL SPACE':
            if st.session_state['compare'] == 'Analysis Increments':
                f = st.session_state['f'][st.session_state['fstexp']]['PP']['Analysis Increments']
            else:
                f = st.session_state['f'][st.session_state['fstexp']]['PP'][st.session_state['compare2']]
        


    match st.session_state['plot']:
    
        case 'Shaded':

            match st.session_state['varlen']:
                case 1: 
                    st.session_state['fig'], st.session_state['axes'] = plt.subplots(ncols=1, nrows=1, figsize=(8,5), gridspec_kw = {'wspace':0.05, 'hspace':0.2}, 
                                                                                    subplot_kw={'projection': ccrs.PlateCarree()})
                case 2: 
                    st.session_state['fig'], st.session_state['axes'] = plt.subplots(ncols=1, nrows=2, figsize=(8, 9), gridspec_kw = {'wspace':0.05, 'hspace':0.0}, 
                                                                                    subplot_kw={'projection': ccrs.PlateCarree()}, sharex=True, sharey=True)
                case 3: 
                    st.session_state['fig'], st.session_state['axes'] = plt.subplots(ncols=2, nrows=2, figsize=(11, 7), gridspec_kw = {'wspace':0.1, 'hspace':0.0}, 
                                                                                    subplot_kw={'projection': ccrs.PlateCarree()}, sharex=True, sharey=True)
                case 4: 
                    st.session_state['fig'], st.session_state['axes'] = plt.subplots(ncols=2, nrows=2, figsize=(11, 7), gridspec_kw = {'wspace':0.1, 'hspace':0.0}, 
                                                                                    subplot_kw={'projection': ccrs.PlateCarree()}, sharex=True, sharey=True)
                case 5: 
                    st.session_state['fig'], st.session_state['axes'] = plt.subplots(ncols=2, nrows=3, figsize=(12, 12), gridspec_kw = {'wspace':0.1, 'hspace':0.0}, 
                                                                                    subplot_kw={'projection': ccrs.PlateCarree()}, sharex=True, sharey=True)

            if st.session_state['varlen'] > 1:
                axes = st.session_state['axes'].flatten()
                st.session_state['axes'] = axes
            else:
                axes = {}
                axes[0] = st.session_state['axes']
                st.session_state['axes'] = axes

            f = mask(f)
            f = reduce_domain(f)

            dt = str(f.coords['time'].values[st.session_state['frame']])[:16]
            rlevel = f.sel(lev=level, method='nearest').lev.values

            # Define colors to plot
            if (st.session_state['menuoutput'] == "Analysis Increments") :
                levels = 17
                tcmap = plt.get_cmap('BrBG').copy()
                tcmap.set_extremes(under='darkred', over='green')

                spcmap = plt.get_cmap('BrBG').copy()
                spcmap.set_extremes(under='darkred', over='green')
                
                wcmap = plt.get_cmap('BrBG').copy()
                wcmap.set_extremes(under='darkred', over='green')
                
                shcmap = plt.get_cmap('BrBG').copy()
                shcmap.set_extremes(under='darkred', over='orange')  
                
                extend = 'both'

            elif ('Inflation' in st.session_state['menuoutput']) | ('Standard' in st.session_state['menuoutput']):

                levels = 21
                tcmap = plt.get_cmap('gist_ncar_r').copy()
                tcmap.set_extremes(under='antiquewhite', over='black')
                spcmap = tcmap
                wcmap = tcmap
                shcmap = tcmap

                extend = 'max'

            elif st.session_state['menuoutput'] == "Difference":

                levels = 17
                tcmap = plt.get_cmap('BrBG').copy()
                tcmap.set_extremes(under='darkred', over='green')

                spcmap = plt.get_cmap('BrBG').copy()
                spcmap.set_extremes(under='darkred', over='green')
                
                wcmap = plt.get_cmap('BrBG').copy()
                wcmap.set_extremes(under='darkred', over='green')
                
                shcmap = plt.get_cmap('BrBG').copy()
                shcmap.set_extremes(under='darkred', over='orange')  
                extend = 'both'

            else:
                        
                levels = 21
                tcmap = plt.get_cmap('gist_ncar_r').copy()
                tcmap.set_extremes(under='antiquewhite', over='black')

                spcmap = plt.get_cmap('Spectral').copy()
                spcmap.set_extremes(under='darkred', over='dimgray')
                
                wcmap = plt.get_cmap('jet_r').copy()
                wcmap.set_extremes(under='darkred', over='darkblue')
                
                shcmap = plt.get_cmap('gist_earth_r').copy()
                shcmap.set_extremes(under='antiquewhite', over='darkred')
                extend = 'both'

            if 'Temperature' in st.session_state['variables']:

                pos = st.session_state['variables'].index('Temperature')
                st.session_state['figure'][pos] = f.sel(time=dt, lev=level, method='nearest').T.plot.pcolormesh(ax=st.session_state['axes'][pos], add_colorbar=True, 
                                                                                                cmap=tcmap, 
                                                                                                levels=levels, cbar_kwargs={"orientation": "horizontal", 
                                                                                                "pad": 0.008, "aspect": 50, "extend": extend,  'spacing': 'uniform'})

                st.session_state['figure'][pos] = f.sel(time=dt, lev=level, method='nearest').T.plot.contour(ax=st.session_state['axes'][pos], 
                                                                                                levels=levels,colors='k', linewidths=0.2)


                # st.session_state['axes'][pos].set_title(f"Time = {dt}, Lev = {rlevel}", fontsize=11)
                st.session_state['axes'][pos].set_title('')
                st.session_state['axes'][pos].add_feature(cfeature.COASTLINE, alpha=0.5) 

            if 'Surface Pressure' in st.session_state['variables']:

                pos = st.session_state['variables'].index('Surface Pressure')                        
                st.session_state['figure'][pos] =  f.sel(time=dt).PS.plot.pcolormesh(ax=st.session_state['axes'][pos], add_colorbar=True, cmap=spcmap, 
                                                                levels=15, cbar_kwargs={"orientation": "horizontal", 
                                                                "pad": 0.008, "aspect": 50, "extend": extend,  'spacing': 'uniform'})
                
                st.session_state['figure'][pos] = f.sel(time=dt, lev=level, method='nearest').PS.plot.contour(ax=st.session_state['axes'][pos], 
                                                                                        levels=levels,colors='k', linewidths=0.2)
                # st.session_state['axes'][pos].set_title(f"Time = {dt}, Surface", fontsize=11)
                st.session_state['axes'][pos].set_title('')
                st.session_state['axes'][pos].add_feature(cfeature.COASTLINE, alpha=0.5) 

            if 'Zonal Wind' in st.session_state['variables']:

                pos = st.session_state['variables'].index('Zonal Wind')
                st.session_state['figure'][pos] =  f.sel(time=dt, lev=level, method='nearest').US.plot.pcolormesh(ax=st.session_state['axes'][pos], add_colorbar=True, cmap=wcmap, 
                                                                                                levels=levels, cbar_kwargs={"orientation": "horizontal", 
                                                                                                "pad": 0.008, "aspect": 50, "extend": extend,  'spacing': 'uniform'})

                st.session_state['figure'][pos] = f.sel(time=dt, lev=level, method='nearest').US.plot.contour(ax=st.session_state['axes'][pos], 
                                                                                                levels=levels,colors='k', linewidths=0.2)

                # st.session_state['axes'][pos].set_title(f"Time = {dt}, Lev = {rlevel}", fontsize=11)
                st.session_state['axes'][pos].set_title('')
                st.session_state['axes'][pos].add_feature(cfeature.COASTLINE, alpha=0.5) 

            if 'Meridional Wind' in st.session_state['variables']:
                
                pos = st.session_state['variables'].index('Meridional Wind')
                st.session_state['figure'][pos] =  f.sel(time=dt, lev=level, method='nearest').VS.plot.pcolormesh(ax=st.session_state['axes'][pos], add_colorbar=True, cmap=wcmap, 
                                                                                                levels=levels, cbar_kwargs={"orientation": "horizontal", 
                                                                                                "pad": 0.008, "aspect": 50, "extend": extend,  'spacing': 'uniform' }) # 'extendfrac' : None
                st.session_state['figure'][pos] = f.sel(time=dt, lev=level, method='nearest').VS.plot.contour(ax=st.session_state['axes'][pos], 
                                                                                                levels=levels,colors='k', linewidths=0.2)

                # st.session_state['axes'][pos].set_title(f"Time = {dt}, Lev = {rlevel}", fontsize=11)
                st.session_state['axes'][pos].set_title('')
                st.session_state['axes'][pos].add_feature(cfeature.COASTLINE, alpha=0.5) 

            if 'Specific Humidity' in st.session_state['variables']:
                
                pos = st.session_state['variables'].index('Specific Humidity')
                st.session_state['figure'][pos] =  f.sel(time=dt, lev=level, method='nearest').Q.plot.pcolormesh(ax=st.session_state['axes'][pos], add_colorbar=True, cmap=shcmap,
                                                                                                                levels=levels, cbar_kwargs={"orientation": "horizontal", "pad": 0.008, "aspect": 50,
                                                                                                                "extend": 'max',  'spacing': 'uniform'})
                st.session_state['figure'][pos] = f.sel(time=dt, lev=level, method='nearest').Q.plot.contour(ax=st.session_state['axes'][pos], 
                                                                                                levels=levels,colors='k', linewidths=0.2)

                # st.session_state['axes'][pos].set_title(f"Time = {dt}, Lev = {rlevel}", fontsize=11)#
                st.session_state['axes'][pos].set_title('')
                st.session_state['axes'][pos].add_feature(cfeature.COASTLINE, alpha=0.5) 
                
            if len(variables) == 3: st.session_state['axes'][3].set_axis_off(); plt.delaxes(st.session_state['axes'][3])
            if len(variables) == 5: st.session_state['axes'][5].set_axis_off(); plt.delaxes(st.session_state['axes'][5])
            plt.suptitle(f"{st.session_state['fstexp'].upper()} - {rlevel} [hPa] \n {dt}")

            st.session_state['fig'].subplots_adjust(top=0.95)
    
        case 'Lineplot':
            
            match st.session_state['varlen']:
                case 1: 
                    st.session_state['fig'], st.session_state['axes'] = plt.subplots(ncols=1, nrows=1, figsize=(5,4), gridspec_kw = {'wspace':0.05, 'hspace':0.2})
                case 2: 
                    st.session_state['fig'], st.session_state['axes'] = plt.subplots(ncols=1, nrows=2, figsize=(7, 8), gridspec_kw = {'wspace':0.05, 'hspace':0.15})#, sharex=True, sharey=True
                case 3: 
                    st.session_state['fig'], st.session_state['axes'] = plt.subplots(ncols=2, nrows=2, figsize=(12, 10), gridspec_kw = {'wspace':0.2, 'hspace':0.15})
                case 4: 
                    st.session_state['fig'], st.session_state['axes'] = plt.subplots(ncols=2, nrows=2, figsize=(12, 10), gridspec_kw = {'wspace':0.2, 'hspace':0.15})
                case 5: 
                    st.session_state['fig'], st.session_state['axes'] = plt.subplots(ncols=2, nrows=3, figsize=(12, 12), gridspec_kw = {'wspace':0.1, 'hspace':0.3})

            if st.session_state['varlen'] > 1:
                axes = st.session_state['axes'].flatten()
                st.session_state['axes'] = axes
            else:
                axes = {}
                axes[0] = st.session_state['axes']
                st.session_state['axes'] = axes

            f = mask(f)
            f = reduce_domain(f, lineplot=True)

            dt = str(f['Global'].coords['time'].values[st.session_state['frame']])[:16]
            print(f['Global']['lev'])

            for var in st.session_state['variables']:

                pos = st.session_state['variables'].index(var)

                match var:
                    case 'Temperature':
                        v = 'T'
                        unit = '[K]'
                    case 'Specific Humidity':
                        v = 'Q'
                        unit = '[kg/kg]'
                    case 'Meridional Wind':
                        v = 'VS'
                        unit = '(staggered) [m/s]'
                    case 'Zonal Wind':
                        v = 'US'
                        unit = '(staggered) [m/s]'
                    case 'Surface Pressure':
                        v = 'PS'
                        unit = '[Pa]'
                                
                for idx in f:

                    match idx:
                        case 'Global':
                            clr = 'black'
                        case 'NH':
                            clr = 'blue'
                        case 'SH':
                            clr = 'green'
                        case 'Tropics': 
                            clr = 'orange'
                        case 'Box':
                            clr = 'red'

                    figure = f[idx].sel(time=dt)[v].plot(ax=st.session_state['axes'][pos], yincrease=False, y='lev', label=idx, color=clr, linewidth=1, add_legend=True)
                    st.session_state['axes'][pos].set(yscale="log")
                    st.session_state['axes'][pos].set_title(f"", fontsize=11)#f"{var} {unit}"
                    st.session_state['axes'][pos].set_ylabel(f"{var} {unit}")
                    st.session_state['axes'][pos].set_xlabel('')
                    st.session_state['axes'][pos].grid()
                    st.session_state['axes'][pos].legend(frameon=False, loc='best')

            if len(variables) == 3: st.session_state['axes'][3].set_axis_off(); plt.delaxes(st.session_state['axes'][3])
            if len(variables) == 5: st.session_state['axes'][5].set_axis_off(); plt.delaxes(st.session_state['axes'][5])
            plt.suptitle(f"{st.session_state['fstexp'].upper()} \n {dt}")

    
    st.session_state['figout'].pyplot(st.session_state['fig'])
           
# def expander(products=None, chart=None):
    
#     msg = st.session_state['expander']['PRODUCTS'][products]

#     with st.sidebar.expander("See explanation"):

#         # Experiment loaded
#         # st.subheader(f"Experiment {st.session_state['name']}")
        
#         # Description
#         st.markdown(f"{msg['description']}")
        
#         # info
#         st.markdown(f"{msg['info']}")
        
#         # Description
#         if '|' in msg[chart]['description']:
#             for m in msg[chart]['description'].split('|'): st.write(m)
#         else:
#             st.write(msg[chart]['description'])

#         # Extras
#         if 'extras' in msg[chart]:

#             for m in msg[chart]['extras'].split('|'): 

#                 if 'description' in st.session_state['expander'][m]:
                
#                     st.write(st.session_state['expander'][m]['description'])

#                 match m:

#                     case 'Domain':
#                         st.write(st.session_state['expander'][m][st.session_state['domain']])

#                     case 'Frequency':
#                         st.write(st.session_state['expander'][m][st.session_state['over']])

#                     case 'Filtering':
#                         st.write(st.session_state['expander'][m][st.session_state['filtering']])

#         # Charts
#         if '|' in msg[chart]['chart']:
#             for m in msg[chart]['chart'].split('|'): st.write(m)
#         else:
#             st.write(msg[chart]['chart'])

if __name__ == '__main__':

    get_experiment()

