# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
import streamlit as st
from sklearn.ensemble import RandomForestClassifier

import warnings
warnings.filterwarnings("ignore")
# -

model=pickle.load(open('random1.pkl','rb'))

model

# +
st.title("Bankruptcy Prevention")
st.sidebar.header("User Input")

def user_input_features():
  Industrial_risk=st.sidebar.slider("Industrial Risk",0.0,0.5,1.0)
  Management_risk=st.sidebar.slider("Management Risk",0.0,0.5,1.0)
  Financial_flexibility=st.sidebar.slider("Financial Flexibility",0.0,0.5,1.0)
  Credibility=st.sidebar.slider("Credibility",0.0,0.5,1.0)
  Competitiveness=st.sidebar.slider("Competitiveness",0.0,0.5,1.0)
  Operating_risk=st.sidebar.slider("Operating Risk",0.0,0.5,1.0)
  data={'industrial_risk':Industrial_risk, # column names changed to match the training data
        'management_risk':Management_risk,
        'financial_flexibility':Financial_flexibility,
        'credibility':Credibility,
        'competitiveness':Competitiveness,
        'operating_risk':Operating_risk}
  features=pd.DataFrame(data,index=[0])
  return features


# +
df = user_input_features()
st.subheader("User Input Parameters")
st.write(df)

bank = pd.read_excel("bankruptcy_prevention.xlsx")

bank['class']=bank['class'].replace({'bankruptcy':1,'non-bankruptcy':0})
y=bank['class']
x=bank.drop('class',axis=1)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=52)
rf=RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(x_train,y_train)


pred = rf.predict(df)

prediction_proba=rf.predict_proba(df)

st.subheader("Prediction")
st.write('bankruptcy' if prediction_proba[0][1] > 0.5 else 'not bankrupt')

st.subheader("Prediction Probability")
st.write(prediction_proba)

# -


