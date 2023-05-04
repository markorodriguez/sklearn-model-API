from flask import Flask, request

from src.service.predict_value import predict_value

app = Flask(__name__)

# Main route

@app.route('/')
def main():
    return 'Main route'

@app.post('/api/trained-model')
def trained_model():
    body = request.get_json()
    prediction = predict_value(body['text'])
    json_response = {
        'prediction': prediction
    }

    return json_response

