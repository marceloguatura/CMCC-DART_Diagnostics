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

test = st.sidebar.selectbox("Select test:", ['test1', 'test2'], index=0)

vardict = { 

        2 : "Temperature (k)",
        3 : "Zonal Wind (m/s)",
        4 : "Meridional Wind (m/s)"


}

if 'd4o' not in st.session_state:
    conn = sqlite3.connect("../data/db/ACAR.1.db", uri=True)
    cursor = conn.cursor()

    body = pd.read_sql_query("select * from body", conn)
    hdr = pd.read_sql_query("select * from hdr", conn)
    body = body.rename(columns={'hdr_id': 'id'})

    ds = pd.merge(hdr, body, on="id", how='left')

    # data = ds[ds['varno'] == 2 ]
    # data = data[data['timeslot'] == 1 ]
    st.session_state['d4o'] = ds

# X = st.sidebar.slider("X", min_value=-90, max_value=90, step=1)
# Y = st.sidebar.slider("Y", min_value=-90, max_value=90, step=1)

# Index(['id', 'file_id', 'bufrtype', 'subtype', 'obstype', 'codetype',
#        'reportype', 'group_id', 'statid', 'stalt', 'lat', 'lon', 'deglat',
#        'deglon', 'geoarea', 'timeslot', 'yyyymmdd', 'hhmmss', 'timestamp',
#        'epoch', 'status_x', 'nbody', 'nsat', 'ngpsro', 'seqno', 'entryno',
#        'varno', 'kind', 'ppcode', 'vertco_type', 'which_vert', 'levelht',
#        'press', 'height', 'fg_error_variance', 'obs_error_variance',
#        'fg_error', 'obs_error', 'channel', 'status_y', 'obsvalue',
#        'prior_mean', 'posterior_mean', 'prior_spread', 'posterior_spread',
#        'qc', 'dart_qc'],
#       dtype='object')

date = st.sidebar.selectbox("Select date:", st.session_state['d4o']['yyyymmdd'].unique())
data = st.session_state['d4o'][st.session_state['d4o']['yyyymmdd'] == date]

var = st.sidebar.selectbox("Variable:", [vardict[v] for v in data['varno'].unique()] , index=0)
for vnum in vardict.keys():
    if var == vardict.get(vnum):
        break
data = data[data['varno'] == vnum]

timeslot = st.sidebar.selectbox("Timeslot:", data['timeslot'].unique(), index=0)
data = data[data['timeslot'] == timeslot]

field = st.sidebar.selectbox("Field", ['fg_error_variance', 'obsvalue',
                    'prior_mean', 'posterior_mean', 'prior_spread', 'posterior_spread',
                    'qc', 'dart_qc'], index=1)

min, max = data['hhmmss'].min(), data['hhmmss'].max()
hours = st.sidebar.select_slider("Filter hhmmss:", np.linspace(min, max, max+1, dtype=int), value=[min, max])
data = data[(data['hhmmss'] >= hours[0]) & (data['hhmmss'] <= hours[1])]
# lev = st.sidebar.select_slider("Filter Levs:", np.sort(st.session_state['d4o']['height'].unique()), value=[st.session_state['d4o']['height'].min(), st.session_state['d4o']['height'].max()])
lev = st.sidebar.select_slider("Filter Levs:", np.linspace(0,14000,201), value=[0, 14000])
data = data[(data['height'] >= lev[0]) & (data['height'] <= lev[1])]

# Rotation View
Z = st.sidebar.slider("Rotation View", min_value=0, max_value=180, step=1, value=40)

def d4oviewer():

    fig = plt.figure(figsize=(10,8))
    axes = fig.add_subplot(111, projection = '3d')
    
    target_projection = ccrs.PlateCarree()

    feature = cartopy.feature.NaturalEarthFeature('physical', 'coastline', '110m')

    geodetic = ccrs.Geodetic(globe=ccrs.Globe(datum='WGS84'))
    geoms = feature.geometries()

    geoms = [target_projection.project_geometry(geom, feature.crs)
            for geom in geoms]

    paths = list(itertools.chain.from_iterable(geos_to_path(geom) for geom in geoms))

    segments = []
    for path in paths:
        vertices = [vertex for vertex, _ in path.iter_segments()]
        vertices = np.asarray(vertices)
        segments.append(vertices)

    lc = LineCollection(segments, color='black', alpha=0.7)

    axes.add_collection3d(lc)

    figure = axes.scatter(data['lon'].values*1/0.017453, data['deglat'].values, data['height'].values , c=data[field].values, cmap=plt.cm.jet, s=10)

    axes.set_ylim(-90, 90)
    axes.set_xlim(-180, 180)

    # Change view to move the angle
    axes.view_init(Z)
    _ = axes.text2D(0.98, 0.05, s="mean="+str(np.around(data[field].mean(),2))+"\nstd="+str(np.around(data[field].std(),2)), transform=axes.transAxes)

    # axes.view_init(X, Y, Z)

    plt.draw()
    plt.colorbar(figure,fraction=0.02, pad=0.05)
    st.pyplot(fig)



if __name__ == '__main__':

    d4oviewer()