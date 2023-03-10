from PIL import Image
import streamlit as st
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import cartopy, cartopy.crs as ccrs
import cartopy.feature
from cartopy.mpl.patch import geos_to_path
import cartopy.crs as ccrs
from matplotlib.collections import LineCollection
import itertools
import numpy as np
import init
import subprocess, os, glob
import cartopy.feature as cfeature
import time
init.load()

img = Image.open(f"{st.session_state['homepath']}/.thumb.jpg")
img2 = Image.open(f"{st.session_state['homepath']}/.thumb2.jpg")
st.set_page_config(page_title='CMCC-DART', page_icon=img2, layout="wide") # , layout="wide"

hide_menu = """
        <style>
        footer {visibility: hidden;}
        footer:after{
            visibility: visible;
            content: 'Copyright @ 2023: CMCC';
            color: gray;
            position:relative;
        }
        </style>
        """
st.markdown(hide_menu, unsafe_allow_html=True)

test = st.sidebar.selectbox("Select test:", ['test1', 'test2', 'test3'], index=2)

# vardict = { 

#         2 : "Temperature (k)",
#         3 : "Zonal Wind (m/s)",
#         4 : "Meridional Wind (m/s)"


# }
match test:
    case 'test1':
        # varlist = [20, 21, 22]
        refname = st.sidebar.selectbox("Reference:", ['ARC'])
    case 'test2': 
        # varlist = [34, 35]
        refname = st.sidebar.selectbox("Reference:", ['ARC', 'AMV'], index=0)
    case 'test3':
        refname = st.sidebar.selectbox("Reference:", ['ARC', 'AMV', 'SND'], index=2)

# if refname == 'ARC':
#     varlist = [20, 21, 22]
# if refname == 'AMV':
#     varlist = [34, 35]
# if test not in st.session_state:
    # conn = sqlite3.connect("../data/db/ACAR.1.db", uri=True)
    # cursor = conn.cursor()

    # body = pd.read_sql_query("select * from body", conn)
    # hdr = pd.read_sql_query("select * from hdr", conn)
    # body = body.rename(columns={'hdr_id': 'id'})
    # ds = pd.merge(hdr, body, on="id", how='left')
    # st.session_state['d4o'] = ds
    # if os.path.exists("source/tmp/"+str(st.session_state['session'])+"/spreads/"+str(refname)+".db"):
    #     os.remove("source/tmp/"+str(st.session_state['session'])+"/spreads/"+str(refname)+".db")
    # if os.path.exists("source/tmp/"+str(st.session_state['session'])+"/dart/"+str(refname)+".db"):
    #     os.remove("source/tmp/"+str(st.session_state['session'])+"/dart/"+str(refname)+".db")

    # shutil.copy("/work/csp/lg07622/work/d4o/CMCC-DART/d4o/scripts/d4omerge", "source/")
# os.makedirs(f"source/tmp/{str(st.session_state['session'])}/{test}/spreads", exist_ok=True)
# os.makedirs(f"source/tmp/{str(st.session_state['session'])}/{test}/dart", exist_ok=True)
# files = glob.glob("/work/csp/lg07622/spreads/EXP/"+str(test)+"/TC/"+str(refname)+"*.db")
# command1 = "/work/csp/mg20022/charts/CESM-DART/DEV/source/d4omerge source/tmp/"+str(st.session_state['session'])+"/spreads/"+str(refname)+".db "
# command2 = " "
# command2 = command2.join(str(f) for f in files)

# Confirmar com gustavo sobre test1 sem tall

if f"{test+refname}_spreads" not in st.session_state['d40view']:
    if test == 'test1':
        pathdata = f"/work/csp/lg07622/spreads/EXP/{str(test)}_back"
    else:
        pathdata = f"/work/csp/lg07622/spreads/EXP/{str(test)}"

    # if not os.path.exists(f"source/tmp/{str(st.session_state['session'])}/{test}/spreads/{str(refname)}.csv"):
    # SPREADS
    df_temp = []

    for file in glob.glob(f"{pathdata}/TC/{str(refname)}.*.db"):
        conn = sqlite3.connect(file, uri=True)
        start = time.time()
        dtemp = pd.read_sql_query("select *, (select distinct description from toc where kind=body.kind) as description from hdr join body on id=body.hdr_id join ens on id=ens.hdr_id and entryno=body_entryno ", conn)
        print(f"END1 : {time.time()-start}")
        dtemp = dtemp.loc[:,~dtemp.columns.duplicated()]
        dtemp = dtemp[dtemp['dart_qc'] == 0]
        df_temp.append(dtemp)
    
    # print(f'columns: {df_temp[0].columns}, len:{len(df_temp[0])}')
    # print(f'columns: {df_temp[1].columns}, len:{len(df_temp[1])}')
    df = pd.concat(df_temp, ignore_index=True)
    # df.to_csv(f"source/tmp/{str(st.session_state['session'])}/{test}/spreads/{str(refname)}.csv", index=False)
    st.session_state['d40view'].update({f"{test+refname}_spreads" : df.copy()})
    
    # DART
    df_temp = []

    for file in glob.glob(f"{pathdata}/tall/TC/{str(refname)}.*.db"):
        conn = sqlite3.connect(file, uri=True)
        # dtemp = pd.read_sql_query("select *, (select distinct description from toc where kind=body.kind) as description, obsvalue, prior_mean, posterior_mean, qc, dart_qc, member, prior, posterior from hdr join body on id=body.hdr_id join ens on id=ens.hdr_id and entryno=body_entryno ", conn)
        start = time.time()
        dtemp = pd.read_sql_query("select *, (select distinct description from toc where kind=body.kind) as description from hdr join body on id=body.hdr_id join ens on id=ens.hdr_id and entryno=body_entryno ", conn)
        print(f"END1 : {time.time()-start}")
        dtemp = dtemp.loc[:,~dtemp.columns.duplicated()]
        dtemp = dtemp[dtemp['dart_qc'] == 0]
        df_temp.append(dtemp)
    
    # print(f'columns: {df_temp[0].columns}, len:{len(df_temp[0])}')
    # print(f'columns: {df_temp[1].columns}, len:{len(df_temp[1])}')
    df = pd.concat(df_temp, ignore_index=True)
    # df.to_csv(f"source/tmp/{str(st.session_state['session'])}/{test}/spreads/{str(refname)}.csv", index=False)
    st.session_state['d40view'].update({f"{test+refname}_dart" : df.copy()})


        # print(f'columns: {df.columns}, len:{len(df)}')
        # else:
        #     st.session_state['d40view'][f"{test+refname}"] = pd.read_csv(f"source/tmp/{str(st.session_state['session'])}/{test}/spreads/{str(refname)}.csv")
        #     subprocess.call(f"source/d4omerge source/tmp/{str(st.session_state['session'])}/{test}/spreads/{str(refname)}.db /work/csp/lg07622/spreads/EXP/{str(test)}_back/TC/{str(refname)}.*.db", shell=True, executable='/bin/bash')
        # if not os.path.exists(f"source/tmp/{str(st.session_state['session'])}/{test}/dart/{str(refname)}.csv"):
        #     subprocess.call(f"source/d4omerge source/tmp/{str(st.session_state['session'])}/{test}/dart/{str(refname)}.db /work/csp/lg07622/spreads/EXP/{str(test)}_back/tall/TC/{str(refname)}.*.db", shell=True, executable='/bin/bash')
    # else:
        # if not os.path.exists(f"source/tmp/{str(st.session_state['session'])}/{test}/spreads/{str(refname)}.db"):
        #     subprocess.call(f"source/d4omerge source/tmp/{str(st.session_state['session'])}/{test}/spreads/{str(refname)}.db /work/csp/lg07622/spreads/EXP/{str(test)}/TC/{str(refname)}.*.db", shell=True, executable='/bin/bash')
        # if not os.path.exists(f"source/tmp/{str(st.session_state['session'])}/{test}/dart/{str(refname)}.db"):
        #     subprocess.call(f"source/d4omerge source/tmp/{str(st.session_state['session'])}/{test}/dart/{str(refname)}.db /work/csp/lg07622/spreads/EXP/{str(test)}/tall/TC/{str(refname)}.*.db", shell=True, executable='/bin/bash')
# if test+refname != st.session_state['d40view']:
#     match test:
#         case 'test1' | 'test2':
#             conn_spreads = sqlite3.connect(f"source/tmp/{str(st.session_state['session'])}/{test}/spreads/{str(refname)}.db", uri=True)
#             st.session_state['ds_spreads'] = pd.read_sql_query("select hhmmss,yyyymmdd,lon,height,timeslot,deglat,deglon,id, reportype, entryno, kind, (select distinct description from toc where kind=body.kind) as description, obsvalue, prior_mean, posterior_mean, qc, dart_qc, member, prior, posterior from hdr join body on id=body.hdr_id join ens on id=ens.hdr_id and entryno=body_entryno ", conn_spreads)

#             conn_dart = sqlite3.connect(f"source/tmp/{str(st.session_state['session'])}/{test}/dart/{str(refname)}.db", uri=True)
#             st.session_state['ds_dart'] = pd.read_sql_query("select hhmmss,yyyymmdd,lon, height, timeslot, deglat, deglon, id, reportype, entryno, kind, (select distinct description from toc where kind=body.kind) as description, obsvalue, prior_mean, posterior_mean, qc, dart_qc, member, prior, posterior from hdr join body on id=body.hdr_id join ens on id=ens.hdr_id and entryno=body_entryno ", conn_dart)
#         case 'test3':
#             # "select id, reportype, entryno, kind, (select distinct description from toc where kind=body.kind) as description, obsvalue, prior_mean, posterior_mean, qc, dart_qc, member, prior, posterior from hdr join body on id=body.hdr_id join ens on id=ens.hdr_id and entryno=body_entryno "
#             conn_spreads = sqlite3.connect(f"source/tmp/{str(st.session_state['session'])}/{test}/spreads/{str(refname)}.db", uri=True)
#             st.session_state['ds_spreads'] = pd.read_sql_query("select hhmmss,yyyymmdd,lon,height,timeslot,deglat,deglon,id, reportype, entryno, kind, (select distinct description from toc where kind=body.kind) as description, obsvalue, prior_mean, posterior_mean, qc, dart_qc, member, prior, posterior from hdr join body on id=body.hdr_id join ens on id=ens.hdr_id and entryno=body_entryno ", conn_spreads)

#             conn_dart = sqlite3.connect(f"source/tmp/{str(st.session_state['session'])}/{test}/dart/{str(refname)}.db", uri=True)
#             st.session_state['ds_dart'] = pd.read_sql_query("select hhmmss,yyyymmdd,lon, height, timeslot, deglat, deglon, id, reportype, entryno, kind, (select distinct description from toc where kind=body.kind) as description, obsvalue, prior_mean, posterior_mean, qc, dart_qc, member, prior, posterior from hdr join body on id=body.hdr_id join ens on id=ens.hdr_id and entryno=body_entryno ", conn_dart)
    
    
#     st.session_state['d40view'] = test+refname
# X = st.sidebar.slider("X", min_value=-90, max_value=90, step=1)
# Y = st.sidebar.slider("Y", min_value=-90, max_value=90, step=1)

var = st.sidebar.selectbox("Variable:", st.session_state['d40view'][f"{test+refname}_dart"]['description'].unique(), index=0)
data_spreads = st.session_state['d40view'][f"{test+refname}_spreads"][ st.session_state['d40view'][f"{test+refname}_spreads"]['description'] == var ]
# data_spreads = data_spreads[data_spreads['dart_qc'] == 0]

# print(f"DATA SPREADS {data_spreads}")
data_dart    = st.session_state['d40view'][f"{test+refname}_dart"][ st.session_state['d40view'][f"{test+refname}_dart"]['description'] == var ]
# data_dart    = data_dart[data_dart['dart_qc'] == 0]
# print(f"DATA DART {data_dart}")

charts = st.sidebar.radio(
    "Visualization:",
    ('2D', '3D'))
st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)

# Index(['id', 'file_id', 'bufrtype', 'subtype', 'obstype', 'codetype',
#        'reportype', 'group_id', 'statid', 'stalt', 'lat', 'lon', 'deglat',
#        'deglon', 'geoarea', 'timeslot', 'yyyymmdd', 'hhmmss', 'timestamp',
#        'epoch', 'status', 'nbody', 'nsat', 'ngpsro', 'seqno', 'entryno',
#        'hdr_id', 'varno', 'kind', 'ppcode', 'vertco_type', 'which_vert',
#        'levelht', 'press', 'height', 'fg_error_variance', 'obs_error_variance',
#        'fg_error', 'obs_error', 'channel', 'obsvalue', 'prior_mean',
#        'posterior_mean', 'prior_spread', 'posterior_spread', 'qc', 'dart_qc',
#        'body_entryno', 'member', 'istatus_prior', 'prior', 'posterior',
#        'description'

field = st.sidebar.selectbox("Field", ['obsvalue', 'prior_mean',
                                       'posterior_mean', 'dart_qc', 
                                       'prior', 'posterior', 'fg_error_variance', 
                                       'obs_error_variance', 'obs_error','fg_error',
                                       'prior_spread', 
                                       'posterior_spread'], index=0)

min, max = data_dart['hhmmss'].min(), data_dart['hhmmss'].max()
hours = st.sidebar.select_slider("Filter hhmmss:", np.linspace(min, max, max+1, dtype=int), value=[min, max])
data_spreads = data_spreads[(data_spreads['hhmmss'] >= hours[0]) & (data_spreads['hhmmss'] <= hours[1])]
data_dart = data_dart[(data_dart['hhmmss'] >= hours[0]) & (data_dart['hhmmss'] <= hours[1])]

# lev = st.sidebar.select_slider("Filter Levs:", np.linspace(0,14000,201), value=[0, 14000])
# data = data[(data['height'] >= lev[0]) & (data['height'] <= lev[1])]
match refname:
    case 'ARC':
        levopt = st.sidebar.radio(
            "Level type:",
            ('Single Level', 'Range', 'Mean over Range'))
        st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
        
        match levopt:
            case 'Single Level':
                lev = st.sidebar.number_input('Insert a level number (meter) between 0 and 14000', min_value=0, max_value=14000)
                data_spreads = data_spreads[data_spreads['height'] == lev ][['deglat', 'deglon', field, 'yyyymmdd']]
                data_dart    = data_dart[data_dart['height'] == lev ][['deglat', 'deglon', field, 'yyyymmdd']]
            case 'Range':
                lev = st.sidebar.select_slider('Select a range level number (meter)', np.linspace(0, 14000, 14000+1, dtype=int), value=[0, 14000])
                data_spreads = data_spreads[(data_spreads['height'] >= lev[0]) & (data_spreads['height'] <= lev[1])][['deglat', 'deglon', field, 'yyyymmdd']]
                data_dart    = data_dart[(data_dart['height'] >= lev[0]) & (data_dart['height'] <= lev[1])][['deglat', 'deglon', field, 'yyyymmdd']]
                # print(data_spreads.groupby(['hhmmss', 'yyyymmdd', 'lon', 'timeslot', 'deglat', 'deglon',
                #                             'id', 'reportype', 'entryno', 'kind', 'description']).mean().reset_index())
            case 'Mean over Range':
                lev = st.sidebar.select_slider('Select a range level number (meter)', np.linspace(0, 14000, 14000+1, dtype=int), value=[0, 14000])
                data_spreads = data_spreads[(data_spreads['height'] >= lev[0]) & (data_spreads['height'] <= lev[1])][['deglat', 'deglon', field, 'yyyymmdd']]
                data_spreads = data_spreads.groupby(['deglat', 'deglon', 'yyyymmdd']).mean().reset_index()
                data_dart    = data_dart[(data_dart['height'] >= lev[0]) & (data_dart['height'] <= lev[1])][['deglat', 'deglon', field, 'yyyymmdd']]
                data_dart = data_dart.groupby(['deglat', 'deglon', 'yyyymmdd']).mean().reset_index()

    
    # case 'AMV':    
    #         data_spreads = st.session_state['ds_spreads']
    #         data_dart    = st.session_state['ds_dart']
    #         print(data_spreads.head())
    # case 'SDV':
    #         data_spreads = st.session_state['ds_spreads']
    #         data_dart    = st.session_state['ds_dart']
    #         print(data_spreads.head()) 
    #         print(data_spreads.columns)        


    # filtrar qc = 0 e contar numero
    # test3 kind 5 6 
    # histograma de cada tipo em cada timeslot

if (len(data_spreads[field]) > 0 ) and (len(data_dart[field]) > 0 ):
    match charts:
        case '2D':
        
            fig, axes = plt.subplots(2,1, figsize=(10,8), subplot_kw={'projection': ccrs.PlateCarree()})

            figure = axes[0].scatter(data_spreads['deglon'].values, data_spreads['deglat'].values, c=data_spreads[field].values, cmap=plt.cm.jet, s=5)
            axes[0].add_feature(cfeature.COASTLINE, alpha=0.7)
            axes[0].set_extent([-180,180,-90,90])
            axes[0].set_title(f"Spreads - {data_spreads['yyyymmdd'].unique()[0]}")
            # axes[0].annotate("mean="+str(np.around(data_spreads[field].mean(),2))+"\nstd="+str(np.around(data_spreads[field].std(),2)), xy=(-200,-10))
            plt.colorbar(figure, pad=0.009, aspect=30)
            axes[0].text(-187, -40, f"mean:{np.around(data_spreads[field].mean(),2)}, std:{np.around(data_spreads[field].std(),2)}, count={np.around(len(data_spreads[field]),2)}", rotation=90, size=8)

            figure = axes[1].scatter(data_dart['deglon'].values, data_dart['deglat'].values, c=data_dart[field].values, cmap=plt.cm.jet, s=5)
            axes[1].add_feature(cfeature.COASTLINE, alpha=0.7)
            axes[1].set_extent([-180,180,-90,90])
            axes[1].set_title(f"Dart - {data_dart['yyyymmdd'].unique()[0]}")
            plt.colorbar(figure, pad=0.009, aspect=30)
            axes[1].text(-187, -40, f"mean:{np.around(data_dart[field].mean(),2)}, std:{np.around(data_dart[field].std(),2)}, count={np.around(len(data_dart[field]),2)}", rotation=90, size=8)
            
            plt.tight_layout()
            st.pyplot(fig)

        case '3D':
            print('3d')
        

else:
    st.warning("No information available to plot.")














# for vnum in vardict.keys():
#     if var == vardict.get(vnum):
#         break
# data = data[data['varno'] == vnum]

# timeslot = st.sidebar.selectbox("Timeslot:", data['timeslot'].unique(), index=0)
# data = data[data['timeslot'] == timeslot]





# # Rotation View
# Z = st.sidebar.slider("Rotation View", min_value=0, max_value=180, step=1, value=40)

# def d4oviewer():

#     fig = plt.figure(figsize=(10,8))
#     axes = fig.add_subplot(111, projection = '3d')
    
#     target_projection = ccrs.PlateCarree()

#     feature = cartopy.feature.NaturalEarthFeature('physical', 'coastline', '110m')

#     geodetic = ccrs.Geodetic(globe=ccrs.Globe(datum='WGS84'))
#     geoms = feature.geometries()

#     geoms = [target_projection.project_geometry(geom, feature.crs)
#             for geom in geoms]

#     paths = list(itertools.chain.from_iterable(geos_to_path(geom) for geom in geoms))

#     segments = []
#     for path in paths:
#         vertices = [vertex for vertex, _ in path.iter_segments()]
#         vertices = np.asarray(vertices)
#         segments.append(vertices)

#     lc = LineCollection(segments, color='black', alpha=0.7)

#     axes.add_collection3d(lc)

#     figure = axes.scatter(data['lon'].values*1/0.017453, data['deglat'].values, data['height'].values , c=data[field].values, cmap=plt.cm.jet, s=10)

#     axes.set_ylim(-90, 90)
#     axes.set_xlim(-180, 180)

#     # Change view to move the angle
#     axes.view_init(Z)
#     _ = axes.text2D(0.98, 0.05, s="mean="+str(np.around(data[field].mean(),2))+"\nstd="+str(np.around(data[field].std(),2)), transform=axes.transAxes)

#     # axes.view_init(X, Y, Z)

#     plt.draw()
#     plt.colorbar(figure,fraction=0.02, pad=0.05)
#     st.pyplot(fig)



# if __name__ == '__main__':

#     d4oviewer()