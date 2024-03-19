import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import streamlit as st
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

df = pd.read_csv("banking.csv")
def plot_classification_results(y_test, y_pred, _score, y):
    current_date = datetime.now().strftime("%d/%m/%Y")
    col1, col2, col3 = st.columns([6, 2, 6])
    with col2:
        st.write(f"<span style='color:green'>{current_date}</span>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([3, 7, 2.5])

    with col2:
        st.title("Data Visualization")
    
    colors = ['pink', 'white','red', 'purple', 'orange', 'brown', 'yellow', 'blue', 'green']
    cmap = plt.cm.colors.ListedColormap(colors)
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    img = ax.imshow(cm, interpolation='nearest', cmap="rocket")
    fig.colorbar(img, ax=ax)
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    ax.set_xticks([0, 1, 2])
    ax.set_yticks([0, 1, 2])
    ax.set_xticklabels(['Standard', 'Poor', 'Good'])
    ax.set_yticklabels(['Standard', 'Poor', 'Good'])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), horizontalalignment='center', color='black' if cm[i, j] > cm.max() / 2 else 'white')
    st.balloons()
    st.text_area('Description')
    st.write(df)
    col1, col2, col3 = st.columns([6.5, 3, 6])
    with col2:
        st.write(f"<span style='color:red'>{'Confusion Matrix'}</span>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([4, 3.5, 4])
    with col2:
        st.text(f"Score:{_score}")
    st.pyplot(fig, ax, bbox_inches='tight')

def build_model_banking():
    object_columns = df.select_dtypes("object").columns.to_list()
    number_columns = df.drop(object_columns, axis=1).columns.to_list()
    process_noise = ["Amount_invested_monthly", "Num_Credit_Inquiries", "Delay_from_due_date", "Monthly_Inhand_Salary", "Outstanding_Debt",
                    "Monthly_Balance", "Annual_Income", "Changed_Credit_Limit", "Credit_Utilization_Ratio", "Total_EMI_per_month"]
    df_iqr = df.copy()
    for _encode in process_noise:
        q25, q75 = np.quantile(df_iqr[_encode], 0.25), np.quantile(df_iqr[_encode], 0.75)
        iqr = q75 - q25
        lower, upper = q25 - 1.25*iqr, q75 + 1.25*iqr
        df_iqr = df_iqr[(df_iqr[_encode] < upper) & (df_iqr[_encode] > lower)]

    encode = LabelEncoder()
    df_iqr["Credit_Label"] = encode.fit_transform(df_iqr["Credit_Score"])
    Credit_Labels = encode.classes_
    df_iqr.drop("Credit_Score", axis=1, inplace=True)
    x = df_iqr.drop("Credit_Label", axis=1)
    y = df_iqr["Credit_Label"]
    object_columns = x.select_dtypes("object").columns.to_list()
    number_columns = x.drop(object_columns, axis=1).columns.to_list()
    encoder = LabelEncoder()
    for i in object_columns:
        x[i] = encoder.fit_transform(x[i])
    smote = RandomOverSampler(random_state=42)
    x, y = smote.fit_resample(x, y)
    sc = MinMaxScaler(feature_range=(-1,1))
    sc.fit(x)
    x = sc.transform(x)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, random_state= 42, shuffle= True, stratify= y)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    model = RandomForestClassifier(n_estimators=100, random_state= 42, class_weight='balanced')
    model.fit(x_train, y_train)
    _score = model.score(x_test, y_test)
    y_pred = model.predict(x_test)
    plot_classification_results(y_test, y_pred, _score, y)

if __name__ == '__main__':
    build_model_banking()