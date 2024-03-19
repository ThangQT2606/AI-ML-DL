import pandas as pd
import streamlit as st
from loan_data import build_model_loan
from banking import build_model_banking
from Carbon import build_model_carbon
from stock import build_model_stock
uploaded_file = st.file_uploader("Choose a file")
st.write(uploaded_file)
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    if uploaded_file.name == 'loan_data.csv':
        build_model_loan()
    elif uploaded_file.name == 'banking.csv':
        build_model_banking()
    elif uploaded_file.name == 'Carbon Emission.csv':
        build_model_carbon()
    elif uploaded_file.name == 'Dữ liệu Lịch sử VNM 2013_2023.csv':
        build_model_stock()