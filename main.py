import streamlit as st
from datetime import datetime
from ibm import show_data

st.title('Project on "Predict how well can 21 days lockdown perform in containing spread of COVID19 Virus."')

submit = st.button('Show Result')

if submit:
    st.success('Executed Successfully')
    st.balloons()
    st.title('Result check for 21 days lockdown')
    show_data()
