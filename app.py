# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Configuração da página
st.set_page_config(page_title="Previsão de Vendas", layout="centered")

st.title("📊 Previsão de Vendas de Produtos")
st.write("Este sistema utiliza Machine Learning para prever a demanda de produtos com base no histórico de vendas.")

# Carregar os dados
@st.cache_data
def carregar_dados():
    df = pd.read_csv("dados/train.csv")
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    return df

df = carregar_dados()

# Treinar o modelo
@st.cache_resource
def treinar_modelo(df):
    X = df[['store', 'item', 'month', 'day']]
    y = df['sales']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    modelo = RandomForestRegressor(n_estimators=100, random_state=42)
    modelo.fit(X_train, y_train)
    return modelo

modelo = treinar_modelo(df)

# Interface de entrada
st.subheader("📥 Insira os dados para prever as vendas:")

store = st.number_input("Loja (1 a 10)", min_value=1, max_value=10, value=1)
item = st.number_input("Produto (1 a 50)", min_value=1, max_value=50, value=1)
month = st.number_input("Mês (1 a 12)", min_value=1, max_value=12, value=1)
day = st.number_input("Dia do Mês (1 a 31)", min_value=1, max_value=31, value=1)

# Previsão
if st.button("Prever Vendas"):
    entrada = pd.DataFrame([[store, item, month, day]], columns=['store', 'item', 'month', 'day'])
    pred = modelo.predict(entrada)
    st.success(f"🔮 Previsão de vendas: {int(pred[0])} unidades")
