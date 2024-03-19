import pandas as pd
import numpy as np 
import random as rd
import matplotlib.pyplot as plt 
import streamlit as st
import altair as alt
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from category_encoders import OneHotEncoder, BaseNEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

df = pd.read_csv("Carbon Emission.csv")
def plot_regression_results(x_test, y_test, y_pred, score_test, mse, mae, model1):
    current_date = datetime.now().strftime("%d/%m/%Y")
    col1, col2, col3 = st.columns([6, 2, 6])
    with col2:
        st.write(f"<span style='color:green'>{current_date}</span>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([3, 7, 2.5])
    with col2:
        st.title("Data Visualization")
    st.balloons()
    st.text_area('Description')
    st.write(df)
    col1, col2, col3 = st.columns([4, 3.5, 4])
    with col2:
        st.text(f"MSE:{mse}")
    col1, col2, col3 = st.columns([4, 3.5, 4])
    with col2:
        st.text(f"MAE:{mae}")
    col1, col2, col3 = st.columns([4, 3.5, 4])
    with col2:
        st.text(f"Score:{score_test}")
    number_test = rd.randint(0, x_test.shape[0])
    x_new = x_test.iloc[number_test]
    y_new = y_test.iloc[number_test]
    x_new = np.expand_dims(x_new, axis=0)
    y_predict = model1.predict(x_new)

    col1, col2, col3 = st.columns([4, 3.5, 4])
    with col2:
        st.write(f"<span style='color:red'>{'Predict Values: ', str(round(y_predict[0]))}</span>", unsafe_allow_html=True)
        st.write(f"<span style='color:red'>{'Real Values: ', str(y_new)}</span>", unsafe_allow_html=True)
    result_df = pd.DataFrame({'y_test': y_test, 'y_pred': y_pred})

    # Vẽ biểu đồ scatter plot
    scatter_plot = alt.Chart(result_df).mark_circle().encode(
        x='y_test',
        y='y_pred'
    )

    # Tính toán giá trị tối đa và tối thiểu của y_test và y_pred để vẽ đường thẳng
    min_value = result_df['y_test'].min()
    max_value = result_df['y_test'].max()

    # Tạo DataFrame cho đường thẳng
    line_df = pd.DataFrame({
        'x': [min_value, max_value],
        'y': [min_value, max_value]
    })

    # Vẽ đường thẳng màu đỏ
    line_plot = alt.Chart(line_df).mark_line(color='red').encode(
        x='x',
        y='y'
    )

    # Kết hợp cả scatter plot và đường thẳng vào một biểu đồ
    chart = (scatter_plot + line_plot).properties(width=600, height=400)

    # Hiển thị biểu đồ
    st.altair_chart(chart, use_container_width=True)
    # st.pyplot(fig, ax, bbox_inches='tight')

def build_model_carbon():
    df = pd.read_csv("Carbon Emission.csv")
    df['Vehicle Type'].fillna(df['Vehicle Type'].mode()[0], inplace = True)
    object_columns = df.select_dtypes("object").columns.to_list()
    q25, q75 = np.quantile(df["Vehicle Monthly Distance Km"], 0.25), np.quantile(df["Vehicle Monthly Distance Km"], 0.75)
    iqr = q75 - q25
    lower, upper = q25 - 0.1*iqr, q75 + 0.1*iqr
    df_iqr = df[(df["Vehicle Monthly Distance Km"] < upper) & (df["Vehicle Monthly Distance Km"] > lower)]
    q25, q75 = np.quantile(df_iqr["CarbonEmission"], 0.25), np.quantile(df_iqr["CarbonEmission"], 0.75)
    iqr = q75 - q25
    lower, upper = q25 - 1.5*iqr, q75 + 1.5*iqr
    df_iqr = df_iqr[(df["CarbonEmission"] < upper) & (df_iqr["CarbonEmission"] > lower)]
    object_columns.remove("Recycling")
    object_columns.remove("Cooking_With")
    encode = OneHotEncoder(cols= object_columns, return_df= True)
    encode.fit(df_iqr)
    df = encode.transform(df_iqr)
    encode1 = BaseNEncoder(cols= ['Recycling', 'Cooking_With'], return_df= True, base=16)
    encode1.fit(df)
    df = encode1.transform(df)
    x = df.drop("CarbonEmission", axis=1)
    y = df["CarbonEmission"]
    smote = RandomOverSampler(random_state=42)
    x, y = smote.fit_resample(x, y)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, shuffle=True, stratify=y)
    model1 = RandomForestRegressor(n_estimators=50, random_state=42)
    model1.fit(x_train, y_train)
    score_test = model1.score(x_test, y_test)
    y_pred = model1.predict(x_test)
    # Tính toán MSE
    mse = mean_squared_error(y_test, y_pred)
    # Tính toán MAE
    mae = mean_absolute_error(y_test, y_pred)
    # Tính toán R-squared score
    r_squared = r2_score(y_test, y_pred)
    plot_regression_results(x_test, y_test, y_pred, score_test, mse, mae, model1)

if __name__ == '__main__':
    build_model_carbon()
