import pandas as pd
import numpy as np 
import random as rd
import matplotlib.pyplot as plt 
import streamlit as st
import altair as alt
from keras import Sequential
from keras.layers import Dense
from datetime import datetime
from matplotlib.dates import YearLocator, DateFormatter, MonthLocator
from sklearn.preprocessing import LabelEncoder
from keras.losses import mean_absolute_error, mean_squared_error
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from category_encoders import OneHotEncoder, BaseNEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def build_model_stock():
    current_date = datetime.now().strftime("%d/%m/%Y")
    col1, col2, col3 = st.columns([6.2, 2, 6])
    with col2:
        st.write(f"<span style='color:green'>{current_date}</span>", unsafe_allow_html=True)
    _, col2, _ = st.columns([6, 3, 6])
    with col2:
        st.title('Stocks')
    df = pd.read_csv('Dữ liệu Lịch sử VNM 2013_2023.csv')
    _, col2, _ = st.columns([1.5, 10, 1.5])
    with col2:
        st.write(df)
    df.drop(columns=['KL', '% Thay đổi'], axis=1, inplace=True)
    df["Ngày"] = pd.to_datetime(df.Ngày,format="%d/%m/%Y")
    df = df.sort_values(by='Ngày')
    # Chuyển đổi định dạng các cột giá thành số thực
    df['Đóng cửa'] = df['Đóng cửa'].str.replace(',', '').astype(float)
    df['Mở cửa'] = df['Mở cửa'].str.replace(',', '').astype(float)
    df['Cao nhất'] = df['Cao nhất'].str.replace(',', '').astype(float)
    df['Thấp nhất'] = df['Thấp nhất'].str.replace(',', '').astype(float)
    df1 = pd.DataFrame(df,columns=['Ngày','Đóng cửa'])
    df1.index = df1.Ngày
    df1.drop('Ngày',axis=1,inplace=True)
    data = df1.values 
    train_data = data[:2000]
    test_data = data[2000:]
    sc = MinMaxScaler(feature_range=(0,1))
    sc_train = sc.fit_transform(data)
    x_train,y_train=[],[]
    for i in range(50,len(train_data)):
        x_train.append(sc_train[i-50:i,0]) #lấy 50 giá đóng cửa liên tục
        y_train.append(sc_train[i,0]) #lấy ra giá đóng cửa ngày hôm sau
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    test = df1[len(train_data)-50:].values
    test = test.reshape(-1,1)
    sc_test = sc.transform(test)

    x_test = []
    for i in range(50,test.shape[0]):
        x_test.append(sc_test[i-50:i,0])
    x_test = np.array(x_test)
    y_test = test_data
    model = Sequential()
    model.add(Dense(512, input_dim=x_train.shape[1], activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(1))
    model.compile(loss = 'mse', optimizer='adam')
    model.fit(x_train, y_train, epochs=100, batch_size=256)
    y_pred = model.predict(x_test)
    y_predict = sc.inverse_transform(y_pred)
    y_train = np.reshape(y_train,(y_train.shape[0],1))
    y_train = sc.inverse_transform(y_train)
    y_train_predict = model.predict(x_train) #dự đoán giá đóng cửa trên tập đã train
    y_train_predict = sc.inverse_transform(y_train_predict) #giá dự đoán
    #lập biểu đồ so sánh
    train_data1 = df1[50:2000]
    test_data1 = df1[2000:]
    
    fig, ax = plt.subplots()
    # plt.figure(figsize=(20,10))
    plt.plot(df1,label='Giá thực tế',color='red') #đường giá thực
    train_data1['Dự đoán'] = y_train_predict #thêm dữ liệu
    plt.plot(train_data1['Dự đoán'],label='Giá dự đoán train',color='green') #đường giá dự báo train
    test_data1['Dự đoán'] = y_predict #thêm dữ liệu
    plt.plot(test_data1['Dự đoán'],label='Giá dự đoán test',color='blue') #đường giá dự báo test
    plt.title('So sánh giá dự báo và giá thực tế') #đặt tên biểu đồ
    plt.xlabel('Thời gian') #đặt tên hàm x
    plt.ylabel('Giá đóng cửa (VNĐ)') #đặt tên hàm y
    plt.legend() #chú thích
    st.balloons()
    st.pyplot(fig, bbox_inches='tight')

if __name__ == '__main__':
    build_model_stock()