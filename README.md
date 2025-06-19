# 📊 Previsão de Vendas com Streamlit

Este projeto é uma aplicação web que prevê a demanda de vendas de produtos com base em dados históricos, utilizando Random Forest em Python.

## Como executar

1. Clone ou baixe o projeto.
2. Instale as dependências:
   ```
   pip install -r requirements.txt
   ```

3. Baixe o dataset de vendas do Kaggle:
   [Retail Demand Forecasting](https://www.kaggle.com/c/demand-forecasting-kernels-only/data)
   e coloque o arquivo `train.csv` dentro da pasta `dados`.

4. Rode a aplicação:
   ```
   streamlit run app.py
   ```

## Estrutura

```
projeto_previsao_vendas/
├── app.py
├── dados/
│   └── train.csv (você precisa adicionar)
├── requirements.txt
└── README.md
```
