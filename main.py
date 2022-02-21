import streamlit as st
from datetime import datetime
from ibm import show_data

st.sidebar.title("Stocks App")

submit = st.sidebar.button('Show Result')

if submit:
    st.success('Executed Successfully')
    st.balloons()
    st.title('Result check for 21 days lockdown')
    show_data()
