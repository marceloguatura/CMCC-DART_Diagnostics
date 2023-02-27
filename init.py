import streamlit as st
import glob
import numpy as np

def load():
    homepath = '/work/csp/mg20022/charts/CESM-DART'

    if 'load' not in st.session_state: st.session_state['load'] = False
    if 'name' not in st.session_state: st.session_state['name'] = ''
    if 'path' not in st.session_state: st.session_state['path'] = ''
    if 'homepath' not in st.session_state: st.session_state['homepath'] = homepath
    if 'session' not in st.session_state: st.session_state['session'] = np.random.rand()
    if 'varlen' not in st.session_state: st.session_state['varlen'] = 0
    if 'f' not in st.session_state: st.session_state['f'] = {}
    if 'fig' not in st.session_state: st.session_state['fig'] = None
    if 'axes' not in st.session_state: st.session_state['axes'] = []
    if 'level' not in st.session_state: st.session_state['level'] = -0.3
    if 'figure' not in st.session_state: st.session_state['figure'] = {}
    if 'variables' not in st.session_state: st.session_state['variables'] = []
    if 'frame' not in st.session_state: st.session_state['frame'] = 0
    if 'menuoutput' not in st.session_state: st.session_state['menuoutput'] = None
    if 'figout' not in st.session_state: st.session_state['figout'] = None
    if 'frame2' not in st.session_state: st.session_state['frame2'] = 0
    if 'frames' not in st.session_state: st.session_state['frames'] = 0
    if 'register' not in st.session_state: st.session_state['register'] = False
    if 'session_change' not in st.session_state: st.session_state['session_change'] = True
    if 'domain' not in st.session_state: st.session_state['domain'] = None
    if 'expander' not in st.session_state: st.session_state['expander'] = None
    if 'over' not in st.session_state: st.session_state['over'] = None
    if 'loadpath' not in st.session_state: st.session_state['loadpath'] = False
    if 'filtering' not in st.session_state: st.session_state['filtering'] = None
    if 'delete' not in st.session_state: st.session_state['delete'] = False
    if 'date' not in st.session_state: st.session_state['date'] = {}
    if 'compare' not in st.session_state: st.session_state['compare'] = None
    if 'compare2' not in st.session_state: st.session_state['compare2'] = None
    if 'product' not in st.session_state: st.session_state['product'] = None
    if 'maskover' not in st.session_state: st.session_state['maskover'] = None
    if 'fstexp' not in st.session_state: st.session_state['fstexp'] = ''
    if 'lon' not in st.session_state: st.session_state['lon'] = np.arange(-180, 180, 0.25)
    if 'lat' not in st.session_state: st.session_state['lat'] = np.arange(-90, 91, 1)
    if 'plot' not in st.session_state: st.session_state['plot'] = None

    if 'experiments' not in st.session_state: 

        exp1 = glob.glob(f"{st.session_state['homepath']}/data/experiments/*")
        exp2 = glob.glob(f"/work/csp/mg20022/charts/CESM-DART/src/experiments/*")
        st.session_state['experiments'] = { 'exp': [x.split('/')[-1] for x in exp1 + exp2 ],
                                            'path' : exp1 + exp2}