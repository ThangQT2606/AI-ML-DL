import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN

image_path = "QTM.jpg"
def plot_classification_results(y_test, y_pred, _score):
    current_date = datetime.now().strftime("%d/%m/%Y")
    col1, col2, col3 = st.columns([6, 2, 6])
    with col2:
        st.write(f"<span style='color:green'>{current_date}</span>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([2.5, 5, 2.5])

    with col2:
        st.title("Confusion Matrix")
        
    col1, col2, col3 = st.columns([4, 3.5, 4])
    with col2:
        st.text(f"Score:{_score}")
    
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.xticks([0, 1], ['Negative', 'Positive'])
    plt.yticks([0, 1], ['Negative', 'Positive'])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), horizontalalignment='center', color='white' if cm[i, j] > cm.max() / 2 else 'black')
    st.pyplot(fig, bbox_inches='tight')


def build_model():
    df = pd.read_csv("loan_data.csv")
    df_iqr = df.copy()
    noise = ['int.rate','installment', 'log.annual.inc', 'fico', 'days.with.cr.line', 'revol.bal', 'inq.last.6mths']
    index = [1.43, 1.48, 1.455, 1.545, 1.32, 0.826, 2]
    pos = 0
    for _index in noise:
        q25, q75 = np.quantile(df_iqr[_index], 0.25), np.quantile(df_iqr[_index], 0.75)
        iqr = q75 - q25
        lower, upper = q25 - index[pos]*iqr, q75 + index[pos]*iqr
        df_iqr = df_iqr[(df_iqr[_index] < upper) & (df_iqr[_index] > lower)]
        pos += 1
    cat_feats = ['purpose'] 
    df_iqr = pd.get_dummies(df_iqr,columns=cat_feats,drop_first=True)
    x = df_iqr.drop('not.fully.paid', axis=1)
    y = df_iqr['not.fully.paid']
    smote1 = RandomOverSampler(random_state=1)
    x1, y1 = smote1.fit_resample(x, y)
    sc = MinMaxScaler(feature_range=(-1, 1))
    sc.fit(x1)
    x1 = sc.transform(x1)
    x_train, x_test, y_train, y_test = train_test_split(x1, y1, test_size=0.2, random_state=42, shuffle=True, stratify=y1)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    model = RandomForestClassifier(n_estimators=50, random_state=42, class_weight='balanced', min_samples_split=2)
    model.fit(x_train, y_train)
    _score = model.score(x_test, y_test)
    y_pred = model.predict(x_test)
    plot_classification_results(y_test, y_pred, _score)


if __name__ == '__main__':
    build_model()
    