from src.service.train_model import use_model, train_model
from datetime import datetime

def predict_value(params):
    response = []
    model, vectorizer = use_model()

    for param in params:
        
        new_data = [param["textMessage"]]
        # print(new_data)
        X_new = vectorizer.transform(new_data)
        y_pred = model.predict(X_new)
        if y_pred == 1:
            param["qualification"] = "Mensaje positivo"
        else:
            param["qualification"] = "Mensaje negativo"

        param["model"] = "SKlearn"
        # print(param['qualification'])
        response.append(param)

    # print(len(response))
    return response
