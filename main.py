import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

# Função para converter colunas categóricas em numéricas
def preprocess_data(df):
    # Converter colunas categóricas usando one-hot encoding
    df = pd.get_dummies(df, drop_first=True)
    return df

def read_all_csv(csv_files):
    dataframes = []
    for file in csv_files:
        try:
            # Tentar ler com codificação padrão UTF-8
            df = pd.read_csv(file)
        except UnicodeDecodeError:
            try:
                # Se falhar, tentar com ISO-8859-1
                df = pd.read_csv(file, encoding='ISO-8859-1')
            except Exception as e:
                print(f"Erro ao ler o arquivo {file}: {e}")
                continue
        dataframes.append(df)
    return dataframes

csv_file = "Loan-data-sem-primeira-linha-change-dependent-variable.csv"

dataframes = read_all_csv([csv_file])

target="loan_status"
for i, df in enumerate(dataframes):
    df = df.dropna()
    print(f"numero de linhas sem dados faltantes : {df.shape[0]}")
    X = df.drop(target, axis=1)
    y = df[target]

    # Pré-processar os dados
    X = preprocess_data(X)

    # Dividir os dados em conjuntos de treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Criar e treinar o modelo
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    # Prever no conjunto de teste
    y_pred = clf.predict(X_test)
    # Mostrar relatório de classificação
    print(f"\nRelatório de classificação para o arquivo {csv_file}:")
    print(classification_report(y_test, y_pred))
