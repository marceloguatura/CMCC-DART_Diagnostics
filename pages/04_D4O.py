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
from matplotlib.ticker import NullFormatter  # useful for `logit` scale
import numpy as np
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import shutil, os, subprocess, glob

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


# vardict = { 

#         22 : "Temperature (k)",
#         21 : "Zonal Wind (m/s)",
#         20 : "Meridional Wind (m/s)"

# }

test = st.sidebar.selectbox("Select test:", ['test1', 'test2'], index=0)

if test == 'test1': 
    varlist = [20, 21, 22]
    refname = st.sidebar.selectbox("Reference:", ['ARC'])
if test == 'test2': 
    varlist = [34, 35]
    refname = st.sidebar.selectbox("Reference:", ['ARC', 'AMV'], index=0)

if refname == 'ARC':
    varlist = [20, 21, 22]
if refname == 'AMV':
    varlist = [34, 35]

if os.path.exists("source/tmp/"+str(st.session_state['session'])+"/spreads/"+str(refname)+".db"):
  os.remove("source/tmp/"+str(st.session_state['session'])+"/spreads/"+str(refname)+".db")
if os.path.exists("source/tmp/"+str(st.session_state['session'])+"/dart/"+str(refname)+".db"):
  os.remove("source/tmp/"+str(st.session_state['session'])+"/dart/"+str(refname)+".db")

# shutil.copy("/work/csp/lg07622/work/d4o/CMCC-DART/d4o/scripts/d4omerge", "source/")
os.makedirs("source/tmp/"+str(st.session_state['session'])+"/spreads", exist_ok=True)
os.makedirs("source/tmp/"+str(st.session_state['session'])+"/dart", exist_ok=True)
files = glob.glob("/work/csp/lg07622/spreads/EXP/"+str(test)+"/TC/"+str(refname)+"*.db")
command1 = "/work/csp/mg20022/charts/CESM-DART/DEV/source/d4omerge source/tmp/"+str(st.session_state['session'])+"/spreads/"+str(refname)+".db "
command2 = " "
command2 = command2.join(str(f) for f in files)

# Confirmar com gustavo sobre test1 sem tall
if test == 'test1':
    subprocess.call(f"source/d4omerge source/tmp/{str(st.session_state['session'])}/spreads/{str(refname)}.db /work/csp/lg07622/spreads/EXP/{str(test)}/TC/{str(refname)}.*.db", shell=True, executable='/bin/bash')
    subprocess.call(f"source/d4omerge source/tmp/{str(st.session_state['session'])}/dart/{str(refname)}.db /work/csp/lg07622/spreads/EXP/{str(test)}_back/tall/TC/{str(refname)}.*.db", shell=True, executable='/bin/bash')
else:
    subprocess.call(f"source/d4omerge source/tmp/{str(st.session_state['session'])}/spreads/{str(refname)}.db /work/csp/lg07622/spreads/EXP/{str(test)}/TC/{str(refname)}.*.db", shell=True, executable='/bin/bash')
    subprocess.call(f"source/d4omerge source/tmp/{str(st.session_state['session'])}/dart/{str(refname)}.db /work/csp/lg07622/spreads/EXP/{str(test)}/tall/TC/{str(refname)}.*.db", shell=True, executable='/bin/bash')
    
arr = {}
for var in varlist:
    
    arr[var] = {"obsvalues1":"", "obsvalues2":"",
                "means1":"", "means2":"",
                "valid":"", "rmse":"",
                "rmse1":"","rmse2":"",
                "corr_coef":"", "tit":""
                }
    
    conn1 = sqlite3.connect(f"source/tmp/{str(st.session_state['session'])}/spreads/{str(refname)}.db")
    c1 = conn1.cursor()
    c1.execute("select id, reportype, entryno, kind, "
            "(select distinct description from toc where kind=body.kind) as description, "
            "obsvalue, prior_mean, posterior_mean, qc, dart_qc, member, prior, posterior "
            "from hdr join body on id=body.hdr_id join ens on id=ens.hdr_id and entryno=body_entryno "
            + "where kind in ("+str(var)+")")
    # fetch all data from second database
    data1 = c1.fetchall()
    # print(f"TAMANHO LEN 1 {var}: ",len(data1))

    # connect to second database and execute query
    conn2 = sqlite3.connect(f"source/tmp/{str(st.session_state['session'])}/dart/{str(refname)}.db")
    c2 = conn2.cursor()
    c2.execute("select id, reportype, entryno, kind, "
           "(select distinct description from toc where kind=body.kind) as description, "
           "obsvalue, prior_mean, posterior_mean, qc, dart_qc, member, prior, posterior "
           "from hdr join body on id=body.hdr_id join ens on id=ens.hdr_id and entryno=body_entryno "
            + "where kind in ("+str(var)+")")

    # fetch all data from second database
    data2 = c2.fetchall()
    # print(f"TAMANHO LEN 2 {var} : ",len(data1))

    obsvalues1 = [row[5] for row in data1]
    obsvalues2 = [row[5] for row in data2]
    means1 = [row[6] for row in data1]
    means2 = [row[6] for row in data2]
    arr[var]["means1"] = np.array(means1, dtype=np.float64)
    arr[var]["means2"] = np.array(means2, dtype=np.float64)
    arr[var]["obsvalues1"] = np.array(obsvalues1, dtype=np.float64)
    arr[var]["obsvalues2"] = np.array(obsvalues2, dtype=np.float64)

    valid = ~np.isnan(arr[var]["means1"]) & ~np.isnan(arr[var]["means2"])

    # calculate RMSE and correlation coefficient
    arr[var]["rmse"] = mean_squared_error(arr[var]["means1"][valid], arr[var]["means2"][valid], squared=False)
    arr[var]["rmse1"] = mean_squared_error(arr[var]["means1"][valid], arr[var]["obsvalues1"][valid], squared=False)
    arr[var]["rmse2"] = mean_squared_error(arr[var]["means2"][valid], arr[var]["obsvalues2"][valid], squared=False)
    arr[var]["corr_coef"] = pearsonr(arr[var]["means1"][valid], arr[var]["means2"][valid])[0]

    arr[var]["tit"] = [row[4] for row in data2]


set1 = {20, 21, 34, 35}
set2 = {22}
set3 = {11, 12, 13, 14, 15}

fig, axes= plt.subplots(ncols=1, nrows=len(varlist), figsize=(12, 12), gridspec_kw = {'wspace':0.1, 'hspace':1}, 
                                                                           sharex=False, sharey=False)

for n,var in enumerate(varlist):
    # check if the number falls within any of the specific sets of numbers
    if var in set1:
        m1 = -20
        m2 = 100
        m3 = m2-m1+1

    elif var in set2:
        m1 = 200
        m2 = 350
        m3 = m2-m1+1

    # plot a scatter plot of the obsvalues and means
    axes[n].scatter(arr[var]["obsvalues1"], arr[var]["obsvalues2"], s=10)
    axes[n].set_xlabel('Prior Mean SPREADS')
    axes[n].set_ylabel('Prior Mean DART')
    axes[n].plot(np.linspace(m1, m2, m3, endpoint=True), np.linspace(m1, m2, m3, endpoint=True))
    axes[n].set_title(f'{arr[var]["tit"][0]}\nRMSD: {arr[var]["rmse"]:.5f}, Corr Coeff: {arr[var]["corr_coef"]:.5f}\nRMSE SPRD: {arr[var]["rmse1"]:.5f}, RMSE DART: {arr[var]["rmse2"]:.5f}')

st.pyplot(fig)


# date = st.sidebar.selectbox("Select date:", st.session_state['d4o']['yyyymmdd'].unique())
# data = st.session_state['d4o'][st.session_state['d4o']['yyyymmdd'] == date]

# var = st.sidebar.selectbox("Variable:", [vardict[v] for v in data['varno'].unique()] , index=0)
# for vnum in vardict.keys():
#     if var == vardict.get(vnum):
#         break
# data = data[data['varno'] == vnum]

# timeslot = st.sidebar.selectbox("Timeslot:", data['timeslot'].unique(), index=0)
# data = data[data['timeslot'] == timeslot]

# field = st.sidebar.selectbox("Field", ['fg_error_variance', 'obsvalue',
#                     'prior_mean', 'posterior_mean', 'prior_spread', 'posterior_spread',
#                     'qc', 'dart_qc'], index=1)

# min, max = data['hhmmss'].min(), data['hhmmss'].max()
# hours = st.sidebar.select_slider("Filter hhmmss:", np.linspace(min, max, max+1, dtype=int), value=[min, max])
# data = data[(data['hhmmss'] >= hours[0]) & (data['hhmmss'] <= hours[1])]
# # lev = st.sidebar.select_slider("Filter Levs:", np.sort(st.session_state['d4o']['height'].unique()), value=[st.session_state['d4o']['height'].min(), st.session_state['d4o']['height'].max()])
# lev = st.sidebar.select_slider("Filter Levs:", np.linspace(0,14000,201), value=[0, 14000])
# data = data[(data['height'] >= lev[0]) & (data['height'] <= lev[1])]

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