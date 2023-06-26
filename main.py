# Aluno: Pedro Wilson Rodrigues de Lima.
# Nº de Matrícula:2020267.

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score

# Aqui carrega os dados
data = pd.read_csv('dados_pacientes.csv')  

data['enjoo'] = data['enjoo'].map({'S': 1, 'N': 0})
data['dor_no_corpo'] = data['dor_no_corpo'].map({'S': 1, 'N': 0})
data['manchas_na_pele'] = data['manchas_na_pele'].map({'S': 1, 'N': 0})

# Aqui divide os dados em features (X) e classes (y)
X = data[['idade', 'sexo', 'temperatura_corporal', 'enjoo', 'dor_no_corpo', 'manchas_na_pele']]
y = data['dengue']

X = pd.get_dummies(X, drop_first=True)

# Dividindo os dados em conjunto de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)

print('Acurácia:', accuracy)
print('Recall:', recall)
print('Precisão:', precision)

# Aqui é a utilizando de um modelo para prever a probabilidade de um paciente ter dengue
idade = 25
sexo = 'M'
temperatura_corporal = 38.5
enjoo = 1
dor_no_corpo = 1
manchas_na_pele = 0

input_data = pd.DataFrame([[idade, temperatura_corporal, enjoo, dor_no_corpo, manchas_na_pele]],
                          columns=['idade', 'temperatura_corporal', 'enjoo', 'dor_no_corpo', 'manchas_na_pele'])

input_data = pd.get_dummies(input_data, drop_first=True)

# Fazendo a previsão
probabilidade_dengue = model.predict_proba(input_data)[:, 1]

print('Probabilidade de ter dengue:', probabilidade_dengue)
