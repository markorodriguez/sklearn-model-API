from src.service.train_model import use_model, train_model

def predict_value(params):
    new_data = [params]
    model, vectorizer = use_model()
    X_new = vectorizer.transform(new_data)

    # simulamos una prediccion
    y_pred = model.predict(X_new)

    # imprimimos nuestros "labels"
    if y_pred == 1:
        return "Mensaje positivo"
    else:
        return "Mensaje negativo"


