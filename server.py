from flask import Flask, render_template, request, jsonify
# import tensorflow as tf
# import keras
from keras.models import load_model
import joblib
import numpy as np

app = Flask(__name__)
# Load the pre-trained model
model_naive_bayes = joblib.load("nb_classifier.pkl")
model_cnn = load_model("./cnn_model")
model_cnn_bert = load_model("./cnn_bert_model")
label_encoder = joblib.load('./label_encoder.joblib')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        title = request.form['title']
        abstract = request.form['abstract']
        model = request.form['model']
        title_abstract = title + ' ' + abstract
        journal_ref_naive_bayes = ''
        if model == 'naive_bayes':
            journal_ref_naive_bayes=predict_naive_bayes(title_abstract)
            return render_template('index.html', naive_bayes_response=journal_ref_naive_bayes)
        elif model == 'cnn':
            response_cnn = model_cnn.predict([title_abstract])
            y_pred_classes = response_cnn.argmax(axis=1)
            # get the class name of the predicted class
            journal_ref_naive_bayes = label_encoder.inverse_transform([y_pred_classes])[0]
            return render_template('index.html', cnn_response=journal_ref_naive_bayes)
        else:
            return render_template('index.html', naive_bayes_response=journal_ref_naive_bayes)


        

        return render_template('index.html', naive_bayes_response=journal_ref_naive_bayes)

    return render_template('index.html', naive_bayes_response = '')



def predict_naive_bayes(title_abstract: str):
    
    predictions = model_naive_bayes.predict([title_abstract])
    # create a response payload with the predicted answer
    return predictions[0]



print("Predicted class:", model_naive_bayes.predict(["ola"])[0])



if __name__ == '__main__':
    app.run()
