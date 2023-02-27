import streamlit as st
from PIL import Image
import init
import glob
import numpy as np

init.load()

# homepath = '/work/csp/mg20022/charts/CESM-DART'
img = Image.open(f"{st.session_state['homepath']}/.thumb.jpg")

st.set_page_config(page_title='CMCC-DART', page_icon=img) # , layout="wide"

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

if 'load' not in st.session_state: st.session_state['load'] = False
if 'name' not in st.session_state: st.session_state['name'] = ''
if 'path' not in st.session_state: st.session_state['path'] = ''
if 'cesmsession' not in st.session_state: st.session_state['cesmsession'] = np.random.rand()

if 'exp' not in st.session_state: 
    st.session_state['exp'] = glob.glob(f"{st.session_state['homepath']}/data/experiments/*")
    st.session_state['exp'] =  ['']+[x.split('/')[-1] for x in st.session_state['exp'] ]

def main():

    st.write("""
    # CMCC-DART 
    ### Diagnostics ###
    #""")


    col = st.empty()
    col.image(img, use_column_width=True)

    # st.sidebar.success("Load Experiment to start.")
    # st.session_state['load'] = False

    # st.title("Stre")
    # st.subheader('sub header')

    # st.markdown(
    # """
    # Streamlit is an open-source app framework built specifically for
    # Machine Learning and Data Science projects.
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
    # """
    # )

if __name__ == '__main__':
    main()