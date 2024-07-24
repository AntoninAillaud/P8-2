import streamlit as st
import pandas as pd
import json
import requests
import dill as pickle
import shap
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import plotly.graph_objects as go
import seaborn as sns

st.set_page_config(
    page_title="Prediction banking scoring",
    page_icon="✅",
    layout="wide",
)

@st.cache_data
def startup():
    #Récupération du modèle et du seuil
    best_model = pickle.load(open('BEST_MODEL.sav', 'rb'))
    f = open("seuil.txt","r")
    seuil = float(f.read())
    f.close()
    
    #Récupération des données
    df_id = pd.read_csv('test_id.csv')
    df = df_id.drop(['SK_ID_CURR'], axis=1)
    
    explainer = shap.Explainer(best_model)
    shap_values = explainer(df)

    proba_all = best_model.predict_proba(df)[:,1]
    df_id['Score'] = proba_all

    return best_model, seuil, df_id, df, shap_values

best_model, seuil, df_id, df, shap_values = startup()

seuil_inv = abs(seuil - 1)

st.set_option('deprecation.showPyplotGlobalUse', False)

st.title("Client banking credit score - Dashboard")
st.write("")

id = 0
selected_client = 0
st.sidebar.title('Client selection')
selected_client = st.sidebar.selectbox('Identifiant client :', df_id['SK_ID_CURR'], help = "Client ID selection among the ones in the dataset")
id = df_id.set_index('SK_ID_CURR').index.get_loc(selected_client)

st.sidebar.title('Graphs')
univariate_options = [col for col in df.columns]
bivariate_options = [col for col in df.columns]
univariate_feature = st.sidebar.selectbox('Univariate variable :', univariate_options, help = "Univariate variable selection among the ones used as features")

bivariate_feature1 = st.sidebar.selectbox('Bivariate variable 1 :', bivariate_options, help = "First bivariate variable selection among the ones used as features")
bivariate_feature2 = st.sidebar.selectbox('Bivariate variable 2 :', bivariate_options, help = "Second bivariate variable selection among the ones used as features")

inputs = {"idx": id}


#url = "http://127.0.0.1:8000/predict"
#request = requests.post(url, data = json.dumps(inputs))
#req = request.json()
#proba_api = req[0]['proba']
#proba_api_inv = abs(proba_api - 1)
#rep_api = req[0]['rep']

@st.cache_data()
def predict(id):
    proba_api = df_id.iloc[[id]]['Score'].values[0]
    #proba_api = best_model.predict_proba(df.values[id].reshape(1,-1))[:,1][0]
    rep_api = "Rejetée" if proba_api > seuil else 'Acceptée'
    return proba_api, rep_api

proba_api, rep_api = predict(id)
proba_api_inv = abs(proba_api - 1)

st.write("General information about the selected client :")
dd = df_id.iloc[[id]][['SK_ID_CURR','CODE_GENDER','DAYS_BIRTH','CNT_CHILDREN','FLAG_OWN_CAR']]
dd.columns = ['Client ID', 'Gender', 'Age', 'Number of Children', 'Have a car']
dd['Gender'] = dd['Gender'].replace({0: 'Female', 1: 'Male'})
dd['Have a car'] = dd['Have a car'].replace({0: 'No', 1: 'Yes'})
dd['Age'] = round(dd['Age'].abs()/365, 0)
st.write(dd)
st.write("")
st.header("Predicting from the client data :")
st.write("")

if(rep_api == "Rejetée"):
    s  = "<span style=\"font-size: 70px;\">:red[Rejected]</span>"
    s1 = "<span style=\"font-size: 35px;\">:red[" + str(round(proba_api_inv,3)) + "  <  " + str(round(seuil_inv,3)) + "]</span>" 
    s2 = "<span style=\"font-size: 35px;\">Score:red[  <  ] Threshold</span>" 
else:
    s  = "<span style=\"font-size: 70px;\">:green[Accepted]</span>"
    s1 = "<span style=\"font-size: 35px;\">:green[" + str(round(proba_api_inv,3)) + "  >  " + str(round(seuil_inv,3)) + "]</span>" 
    s2 = "<span style=\"font-size: 35px;\">Score:green[  >  ] Threshold</span>" 

st.markdown(s2, unsafe_allow_html=True)
st.markdown(s1, unsafe_allow_html=True)
st.markdown(s, unsafe_allow_html=True)
st.write("")

@st.cache_data(hash_funcs={matplotlib.figure.Figure: lambda _: None})
def displayGauge(proba_api_inv):
    return go.Figure(go.Indicator(
    mode = "number+gauge+delta",
    gauge = {'shape': "bullet", 'axis': {'range': [None, 1]},
             'steps' : [
                 {'range': [0, seuil_inv], 'color': "lightcoral"},
                 {'range': [seuil_inv, 1], 'color': "lightgreen"}],
             'threshold' : {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': seuil_inv}},
    delta = {'reference': seuil_inv},
    value = proba_api_inv,
    domain = {'x': [0.1, 1], 'y': [0.2, 0.9]},
    title = {'text': "Scoring"}))

st.write(displayGauge(proba_api_inv))

col1, col2 = st.columns(2)

@st.cache_data(hash_funcs={matplotlib.figure.Figure: lambda _: None})
def displayShapGlobal():
    fig = plt.figure()  
    shap.plots.bar(shap_values,max_display=20)
    return fig

@st.cache_data(hash_funcs={matplotlib.figure.Figure: lambda _: None})
def displayShapLocal(id):
    fig = plt.figure()  
    shap.plots.bar(shap_values[id],max_display=20)
    return fig

with col1:
    st.header("Global Feature Importance")
    fig1 = displayShapGlobal()
    st.pyplot(fig1)

with col2:
    s = "Local Feature Importance for ID " + str(selected_client)
    st.header(s)
    fig2 = displayShapLocal(id)
    st.pyplot(fig2)

col3, col4 = st.columns(2)

@st.cache_data(hash_funcs={matplotlib.figure.Figure: lambda _: None})
def displayUnivariate(univariate_feature, id):
    plt.figure()
    plt.hist(df_id[univariate_feature], color='skyblue', label='Population')
    plt.xlabel(univariate_feature)
    plt.axvline(df_id.iloc[[id]][univariate_feature].values[0], color='salmon', linestyle='--', label='Selected Client')
    plt.legend()
    return plt.gcf()

@st.cache_data(hash_funcs={matplotlib.figure.Figure: lambda _: None})
def displayBivariate(bivariate_feature1, bivariate_feature2, id):
    plt.figure()
    sns.scatterplot(data=df_id, x=bivariate_feature1, y=bivariate_feature2, c=abs(df_id['Score'] - 1), 
                    cmap='viridis', alpha=0.5, label='Population')
    sns.scatterplot(data=df_id.iloc[[id]], x=bivariate_feature1, y=bivariate_feature2, 
                    color='salmon', marker='o', s=100, label='Selected Client')
    plt.xlabel(bivariate_feature1)
    plt.ylabel(bivariate_feature2)
    plt.legend()
    return plt.gcf()

with col3:
    s = "Univariate Analysis of " + str(univariate_feature)
    st.header(s)
    fig3 = displayUnivariate(univariate_feature, id)
    st.pyplot(fig3)

with col4:
    s = "Bivariate Analysis for " + str(bivariate_feature1) + " and " + str(bivariate_feature2)
    st.header(s)
    fig4 = displayBivariate(bivariate_feature1, bivariate_feature2, id)
    st.pyplot(fig4)
