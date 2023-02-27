import streamlit as st
import matplotlib.pyplot as plt
import xarray as xr
import cartopy, cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from PIL import Image
import glob
import warnings
import os
from PIL import Image
import numpy as np
import time
import io
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as mcolors
from cycler import cycler
from streamlit.elements.utils import _shown_default_value_warning
_shown_default_value_warning = False
from matplotlib.gridspec import GridSpec
import json
import pandas as pd
import geopandas as gpd
import rioxarray
# "description": 
import init

init.load()

# homepath = '/work/csp/mg20022/charts/CESM-DART'
img = Image.open(f"{st.session_state['homepath']}/.thumb.jpg")
st.set_page_config(page_title='CMCC-DART', page_icon=img, layout="wide") # , layout="wide"
hide_menu = """
        <style>
        footer {visibility: hidden;}
        footer:after{
       [pos].set_array(f.sel(time=dt,lev=level     visibility: visible;
            content: 'Copyright @ 2022: CMCC';
            color: gray;
            position:relative;
        }
        </style>
        """
st.markdown(hide_menu, unsafe_allow_html=True)
  
def get_experiment():


    with st.expander("Register or Remove Experiments"):

        with st.container():

            # colpath = st.columns((1,3))
            name = st.text_input('Experiment Name')
            path = st.text_input('Experiment Path')

            col1, col2 = st.columns(2, gap='small')

            st.session_state['register'] = col1.button('Register')
            st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)

            st.session_state['delete'] = col2.button('Delete')
            st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
    
    if st.session_state['register']:
        if name:
            if os.path.exists(f"{path}/{name}"):

                if os.path.exists(f"{st.session_state['homepath']}/data/experiments/{name}"):
                    os.remove(f"{st.session_state['homepath']}/data/experiments/{name}")

                os.symlink(f'{path}/{name}', f"{st.session_state['homepath']}/data/experiments/{name}")

                # Update menu
                exp1 = glob.glob(f"{st.session_state['homepath']}/data/experiments/*")
                exp2 = glob.glob(f"/work/csp/mg20022/charts/CESM-DART/src/experiments/*")
                st.session_state['experiments'] = { 'exp': [x.split('/')[-1] for x in exp1 + exp2 ],
                                                    'path' : exp1 + exp2}
                st.success("Experiment registered successful!")

            else:
                st.error("Experiment directory not found or forbidden the access!")


    if st.session_state['delete']:
        if name:
            if name in st.session_state['experiments']['exp']:
                st.session_state['experiments']['exp'].remove(name)

            os.remove(f"{st.session_state['homepath']}/data/experiments/{name}")
            st.success("Experiment removed successful!")

            # Update menu
            exp1 = glob.glob(f"{st.session_state['homepath']}/data/experiments/*")
            exp2 = glob.glob(f"/work/csp/mg20022/charts/CESM-DART/src/experiments/*")
            st.session_state['experiments'] = { 'exp': [x.split('/')[-1] for x in exp1 + exp2 ],
                                                'path' : exp1 + exp2}

    with st.container():

        st.title("CMCC-DART")
        st.markdown(
                        """
                        CMCC-DART Diagnostics is a framework built specifically for providing a
                        common tool to analyze and evaluate experiments on data assimilation runned by **DART-CESM**.


                        """
                    )

        # **👈 Select a demo from the sidebar** to see some examples
        # of what Streamlit can do!
        # ### Want to learn more?
        # - Check out [streamlit.io](https://streamlit.io)
        # - Jump into our [documentation](https://docs.streamlit.io)
        # - Ask a question in our [community
        #     forums](https://discuss.streamlit.io)
        # ### See more complex demos
        # - Use a neural net to [analyze the Udacity Self-driving Car Image
        #     Dataset](https://github.com/streamlit/demo-self-driving)
        # - Explore a [New York City rideshare dataset](https://github.com/streamlit/demo-uber-nyc-pickups)

    col1, col2, col3 = st.columns(3)

    choose1 = col1.selectbox('First Experiment:',
                                [''] + st.session_state['experiments']['exp'])

    if choose1 != "": 
        menu2 = st.session_state['experiments']['exp'].copy() + ['ERA5-Reanalysis']; menu2.remove(choose1) # , 'NCEP-Reanalysis'
    else: 
        menu2 = st.session_state['experiments']['exp'].copy() + ['ERA5-Reanalysis'] # , 'NCEP-Reanalysis'

    choose2 = col2.selectbox(
                                'Experiment to compare 1:',
                                [''] + menu2)
    
    if choose2 != "": 
        menu3 = menu2.copy() ; menu3.remove(choose2)
    else:
        menu3 = menu2.copy()
        
    choose3 = col3.selectbox(
                                'Experiment to compare 2:',
                                [''] + menu3)
    st.title('')
#    st.markdown(
#                    """
#
#                    \n
#                    CMCC-DART Diagnostics is a framework built specifically for providing a
#                    common tool to analyze and evaluate experiments on data assimilation runned by **DART-CESM**.


#                    """
#                )

    # col4, col5, _ = st.columns(3)
    # over1 = col4.radio('Select', (1, 2))
    # over2 = col5.radio('Select2', (1, 2))

    # st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)


    load = st.button('Load')

    if load:

        experiments = [choose1, choose2, choose3]
        experiments = [x for x in experiments if x]

        if len(experiments) > 0:

            if 'ERA5-Reanalysis' in experiments:
                experiments.remove('ERA5-Reanalysis')
                experiments.append('ERA5-Reanalysis')

            if st.session_state['name'] != experiments:
                st.session_state['session_change'] = True
                st.session_state['session'] = np.random.rand()
                
            st.session_state['name'] = experiments
            st.session_state['path'] = f"{st.session_state['homepath']}/data/experiments"
            st.session_state['load'] = True
            
            openDataset(session=st.session_state['session'])
            st.success('Experiments loaded, please click on Output Fields to visualize it.')

        else:
            st.warning(f"Select two or more experiments before try load.")
            st.session_state['load'] = False

 
@st.cache(hash_funcs={xr.core.dataset.Dataset: id}, allow_output_mutation=True, ttl=7200)
def openDataset(session):

    start = time.time()

    path = st.session_state['path']

    for name in st.session_state['name']:

        match name:
            # ERA5-Reanalysis
            case 'ERA5-Reanalysis':

                def preprocess_era5(f):
                    f.coords['lon'] = (f.coords['lon'] + 180) % 360 - 180
                    f = f.sortby(f.lon)
                    return f

                newlist = st.session_state['name']
                
                if 'ERA5-Reanalysis' in newlist: newlist.remove('ERA5-Reanalysis')
                # if 'ERA5-Reanalysis' in newlist: newlist.remove('ERA5-Reanalysis')

                for name2 in newlist:

                    mon1 = str(st.session_state['date'][name2][0]).split('-')[1]
                    mon2 = str(st.session_state['date'][name2][1]).split('-')[1]

                    if mon1 == mon2:
                        # files = glob.glob(f"{st.session_state['homepath']}/data/ERA5/2017/regrid/{mon1}/*_regrid.nc", recursive=True)
                        files = glob.glob(f"{st.session_state['homepath']}/data/ERA5/regrid/2017/{mon1}/*_merged.nc", recursive=True)

                    else:
                        files = glob.glob(f"{st.session_state['homepath']}/data/ERA5/regrid/2017/*[{mon1}-{mon2}]/*_merged.nc", recursive=True)

                    print(files, mon1, mon2)
                    ds = xr.open_mfdataset(files, chunks={'lon': -1, 'lat': -1, 'level': -1, 'time': 10}, parallel=False, preprocess=preprocess_era5)
                    ds.coords['lat'] = st.session_state['f'][st.session_state['name'][0]]['PP']['Analysis'].coords['lat']
                    ds = ds.sel(time=slice(st.session_state['date'][name2][0], st.session_state['date'][name2][1]))
                    ds = ds.rename({
                                    'level': 'lev',
                                    't':'T', 
                                    'q':'Q', 
                                    'u':'US', 
                                    'v':'VS', 
                                    'z':'PS'
                                    })
                
                    st.session_state['f'][name2]['PP'][name] = st.session_state['f'][name2]['PP']['Analysis']-ds
                    # st.session_state['f'].update({"Tropics" : fout})
                    # print(st.session_state['f'][st.session_state['name'][0]]['PP']['Analysis'])
                    # print('\n')
                    # print(ds)
                    # print(st.session_state['f'][st.session_state['name'][0]]['PP']['ERA5-Reanalysis'])

            case _:

                def preprocess_cesm(f):

                    # Define datetime
                    date = str(f.date.values[0])
                    hours = str(timedelta(seconds=int(f.datesec.values[0]))).split(':')
                    year, mon, day = int(date[0:4]), int(date[4:6]), int(date[6:])
                    hour, min, sec = int(hours[0]), int(hours[1]), int(hours[2])
                    dt = datetime(year,mon,day,hour,min,sec)
                    
                    # Assign coords
                    f = f.assign_coords({'time2': dt})
                    f = f.expand_dims(['time2'])
                    f = f.drop_vars(['date', 'datesec'])
                    f = f.squeeze(['time'])
                    f = f.rename({'time2':'time'})

                    variables = ['T', 'Q', 'US', 'VS', 'PS']
                    f['NUS'] = f['US'].interp(slat=f.coords['lat'].values, method='linear', kwargs={"fill_value": "extrapolate"}).rename({'slat': 'lat'})
                    f['NVS'] = f['VS'].interp(slon=f.coords['lon'].values, method='linear', kwargs={"fill_value": "extrapolate"}).rename({'slon': 'lon'})

                    f = f.drop_vars(['US', 'VS'])
                    f = f.rename({'NUS':'US', 'NVS':'VS'})
                    f = f[variables]
                    f.coords['lon'] = (f.coords['lon'] + 180) % 360 - 180
                    f = f.sortby(f.lon)

                    return f

                ff = {}

                # case 'Background':
                files0 = glob.glob(f'{path}/{name}/{name}-*/{name}.dart.e.cam_forecast_mean.*.nc', recursive=True)
                if len(files0) == 0: files0 = glob.glob(f'{path}/{name}-*/{name}.dart.e.cam_forecast_mean.*.nc', recursive=True)

                # case 'Analysis':
                files1 = glob.glob(f'{path}/{name}/{name}-*/{name}.dart.i.cam_output_mean.*', recursive=True)
                if len(files1) == 0: files1 = glob.glob(f'{path}/{name}-*/{name}.dart.i.cam_output_mean.*', recursive=True)

                # case 'Background Standard Deviation':
                files2 = glob.glob(f'{path}/{name}/{name}-*/{name}.dart.e.cam_forecast_sd.*', recursive=True)
                if len(files2) == 0: files2 = glob.glob(f'{path}/{name}-*/{name}.dart.e.cam_forecast_sd.*', recursive=True)
                
                # case 'Analysis Standard Deviation':
                files3 = glob.glob(f'{path}/{name}/{name}-*/{name}.dart.i.cam_output_sd.*', recursive=True)
                if len(files3) == 0: files3 = glob.glob(f'{path}/{name}-*/{name}.dart.i.cam_output_sd.*', recursive=True)

                # case 'Analysis Inflation':
                files4 = glob.glob(f'{path}/{name}/{name}-*/{name}.dart.rh.cam_output_priorinf_mean.*', recursive=True)
                if len(files4) == 0: files4 = glob.glob(f'{path}/{name}-*/{name}.dart.rh.cam_output_priorinf_mean.*', recursive=True)

                # case 'Analysis Inflation Standard Deviation':
                files5 = glob.glob(f'{path}/{name}/{name}-*/{name}.dart.rh.cam_output_priorinf_sd.*', recursive=True)
                if len(files5) == 0: files5 = glob.glob(f'{path}/{name}-*/{name}.dart.rh.cam_output_priorinf_sd*', recursive=True)

                # case 'Background Inflation':
                files6 = glob.glob(f'{path}/{name}/{name}-*/{name}.dart.rh.cam_forecast_priorinf_mean.*', recursive=True)
                if len(files6) == 0: files6 = glob.glob(f'{path}/{name}-*/{name}.dart.rh.cam_forecast_priorinf_mean.*', recursive=True)

                # case 'Background Inflation Standard Deviation':
                files7 = glob.glob(f'{path}/{name}/{name}-*/{name}.dart.rh.cam_forecast_priorinf_sd.*', recursive=True)
                if len(files7) == 0: files7 = glob.glob(f'{path}/{name}-*/{name}.dart.rh.cam_forecast_priorinf_sd*', recursive=True)

                idx = ['Background', 'Analysis', 'Background Standard Deviation', 
                       'Analysis Standard Deviation', 'Analysis Inflation',
                       'Analysis Inflation Standard Deviation', 'Background Inflation',
                       'Background Inflation Standard Deviation']

                for id, files in enumerate([files0, files1, files2, files3, files4, files5, files6, files7]):

                    if len(files) > 0:        
                        ff.update({f"{idx[id]}": xr.open_mfdataset(files, preprocess=preprocess_cesm, chunks={'lon': -1, 'lat': -1, 'lev': -1, 'time': 10}, parallel=False)})
                        # ff[idx[id]] = ff[idx[id]].assign_coords(lon=(((ff[idx[id]].lon + 180) % 360) - 180))
                ff.update({"Analysis Increments" : (ff['Analysis']-ff['Background'])})

                nff = {}
                levs = [1,2,3,5,7,10,20,30,50,70,100,125, 150,175,200,
                        225,250,300,350,400,450,500,550,600,650,700,750,
                        775,800,825,850,875,900,925,950,975,1000]
                
                # levs = [np.format_float_scientific(lv) for lv in levs]
                # print(levs)

                for id in idx:
                    nff[id] = ff[id].interp(lev=levs, 
                                            method='linear', kwargs={"fill_value": "extrapolate"})
                
                st.session_state['f'][name] = {}
                nff.update({"Analysis Increments" : (nff['Analysis']-nff['Background'])})
                st.session_state['f'][name]['PP'] = nff.copy()
                
                levs = [992,850,500,250,150,50,3,1,0.1,0.01]
                # levs = [np.format_float_scientific(lv) for lv in levs]
                # print(levs)

                for id in idx:
                    #ff[id] = ff[id].sel(lev=['992','850','500','250','150','50','3', '1', '0.1', '0.01'], method='nearest')    
                    ff[id] = ff[id].interp(lev=levs, method='linear', kwargs={"fill_value": "extrapolate"})                
                
                st.session_state['f'][name]['default'] = ff.copy()

                st.session_state['date'][name] = []
                st.session_state['date'][name] = [st.session_state['f'][name]['default']['Background'].coords['time'].values[0],
                                                  st.session_state['f'][name]['default']['Background'].coords['time'].values[-1]]

    st.session_state['f']['shape'] = gpd.read_file(f"{st.session_state['homepath']}/data/shape/ne_110m_ocean/ne_110m_ocean.shp", crs="epsg:4326")

    print(f"FINISH {time.time()-start}")

def expander(products=None, chart=None):
    
    msg = st.session_state['expander']['PRODUCTS'][products]

    with st.sidebar.expander("See explanation"):

        # Experiment loaded
        # st.subheader(f"Experiment {st.session_state['name']}")
        
        # Description
        st.markdown(f"{msg['description']}")
        
        # info
        st.markdown(f"{msg['info']}")
        
        # Description
        if '|' in msg[chart]['description']:
            for m in msg[chart]['description'].split('|'): st.write(m)
        else:
            st.write(msg[chart]['description'])

        # Extras
        if 'extras' in msg[chart]:

            for m in msg[chart]['extras'].split('|'): 

                if 'description' in st.session_state['expander'][m]:
                
                    st.write(st.session_state['expander'][m]['description'])

                match m:

                    case 'Domain':
                        st.write(st.session_state['expander'][m][st.session_state['domain']])

                    case 'Frequency':
                        st.write(st.session_state['expander'][m][st.session_state['over']])

                    case 'Filtering':
                        st.write(st.session_state['expander'][m][st.session_state['filtering']])

        # Charts
        if '|' in msg[chart]['chart']:
            for m in msg[chart]['chart'].split('|'): st.write(m)
        else:
            st.write(msg[chart]['chart'])

if __name__ == '__main__':

    get_experiment()

