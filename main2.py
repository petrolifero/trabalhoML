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

# 1. Ler todos os arquivos CSV da pasta atual
csv_files = [file for file in os.listdir('.') if file.endswith('.csv')]

# 2. Armazenar os dados de cada CSV em variáveis separadas (usaremos uma lista)
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

# 3. Verificar quais arquivos têm dados faltantes e quantos valores estão faltando em cada arquivo
for i, df in enumerate(dataframes):
    missing_data = df.isnull().sum().sum()
    if missing_data > 0:
        print(f"Arquivo {csv_files[i]} tem {missing_data} valores faltantes.")

# 4. Usar um algoritmo básico de árvore de decisão para prever a variável `loan_status`
for i, df in enumerate(dataframes):
    # Identificar o nome correto da coluna 'loan_status'
    if 'Y' in df.columns:
        target = 'Y'
    elif 'Loan_Status' in df.columns:
        target = 'Loan_Status'
    elif 'loan_status' in df.columns:
        target = 'loan_status'
    else:
        print(f"\nO arquivo {csv_files[i]} não contém a coluna 'loan_status'.")
        continue
    
    df = df.dropna()  # Remover linhas com dados faltantes
    if df.shape[0] > 0:  # Certificar que há dados suficientes
        X = df.drop(target, axis=1)
        y = df[target]

        # Pré-processar os dados
        X = preprocess_data(X)

        # Dividir os dados em conjuntos de treino e teste
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Verificar se as colunas são numéricas
        if X_train.select_dtypes(include=[np.number]).shape[1] == X_train.shape[1]:
            # Criar e treinar o modelo
            clf = DecisionTreeClassifier()
            clf.fit(X_train, y_train)

            # Prever no conjunto de teste
            y_pred = clf.predict(X_test)

            # Mostrar relatório de classificação
            print(f"\nRelatório de classificação para o arquivo {csv_files[i]}:")
            print(classification_report(y_test, y_pred))
        else:
            print(f"\nO arquivo {csv_files[i]} contém colunas não numéricas. Por favor, converta-as para numéricas antes de usar o modelo de árvore de decisão.")
    else:
        print(f"\nO arquivo {csv_files[i]} não possui dados suficientes após a remoção de valores faltantes.")
