import os
import pandas as pd 

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from joblib import dump, load

from src.service.balance_df import generate_balaced_dataset

# validamos que haya un archivo de salida balanceado

def train_model():
    global df_balanced
    balanced_file = os.listdir('src/dataset/balanced/')
    # si no existe, lo generamos
    if len(balanced_file) == 0:
        print('No existe un archivo balanceado, generando...')
        df_balanced = generate_balaced_dataset()
    else:
        print('Existe un archivo balanceado, leyendo...')
        df_balanced = pd.read_csv('src/dataset/balanced/balanced_out.csv')

    # generamos el vectorizador
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df_balanced['Text'])
    y = df_balanced['Positive']

    # dividimos la data en entrenamiento y testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)

    # empleamos un modelo de regresión logística para el entrenamiento
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # evaluamos el modelo 
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print("Model Accuracy: ", accuracy)

    # guardamos el modelo y el vectorizador
    dump(model, 'src/models/model.joblib')
    dump(vectorizer, 'src/models/vectorizer.joblib')

    print('Modelo y vectorizador guardados en src/models/')

def use_model():
    
    print('Cargando modelo y vectorizador...')
    model = load('src/models/model.joblib')
    vectorizer = load('src/models/vectorizer.joblib')
    print('Modelo y vectorizador cargados')
    return model, vectorizer
    
    

