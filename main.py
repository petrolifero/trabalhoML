import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score



def explain_decision_tree(clf):
    n_nodes = clf.tree_.node_count
    children_left = clf.tree_.children_left
    children_right = clf.tree_.children_right
    feature = clf.tree_.feature
    threshold = clf.tree_.threshold
    values = clf.tree_.value
    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, 0)]  # start with the root node id (0) and its depth (0)
    while len(stack) > 0:
        # `pop` ensures each node is only visited once
        node_id, depth = stack.pop()
        node_depth[node_id] = depth
        
        # If the left and right child of a node is not the same we have a split
        # node
        is_split_node = children_left[node_id] != children_right[node_id]
        # If a split node, append left and right children and depth to `stack`
        # so we can loop through them
        if is_split_node:
            stack.append((children_left[node_id], depth + 1))
            stack.append((children_right[node_id], depth + 1))
        else:
            is_leaves[node_id] = True
            
            print(
                "The binary tree structure has {n} nodes and has "
                "the following tree structure:\n".format(n=n_nodes)
            )
            for i in range(n_nodes):
                if is_leaves[i]:
                    print(
                        "{space}node={node} is a leaf node with value={value}.".format(
                            space=node_depth[i] * "\t", node=i, value=values[i]
                        )
                    )
                else:
                    print(
                        "{space}node={node} is a split node with value={value}: "
                        "go to node {left} if X[:, {feature}] <= {threshold} "
                        "else to node {right}.".format(
                            space=node_depth[i] * "\t",
                            node=i,
                            left=children_left[i],
                            feature=feature[i],
                            threshold=threshold[i],
                            right=children_right[i],
                            value=values[i],
                        )
        )

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

    # Dividir os dados em conjuntos de treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Criar e treinar o modelo
    clf = DecisionTreeClassifier(max_depth=10)
    clf2 = DecisionTreeClassifier(max_depth=100)
    clf.fit(X_train, y_train)
    clf2.fit(X_train, y_train)
    # Prever no conjunto de teste
    y_pred = clf.predict(X_test)
    y_pred2 = clf2.predict(X_test)
    # Mostrar relatório de classificação
    print(f"\nRelatório de classificação para o arquivo {csv_file}:")
    print(classification_report(y_test, y_pred))
    print(classification_report(y_test, y_pred2))
    scores = cross_val_score(clf, X, y, cv=20)
    scores2 = cross_val_score(clf2, X, y, cv=20)
    print(scores.mean(),scores.std())
    print(scores2.mean(),scores2.std())