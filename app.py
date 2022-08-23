from flask import Flask, render_template, url_for, request
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
# import h5py
# h5py.run_tests()
from keras.preprocessing.text import Tokenizer
import tensorflow as tf

app = Flask(__name__)
model = tf.keras.models.load_model('bi_LSTMmodel.h5')

tokenizer = Tokenizer()

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [x for x in request.form.values()]
    filterdataframe = pd.read_csv('FilteredDataframe.csv')
    tokenizer.fit_on_texts(list(filterdataframe['FinalProcessedData']))
    final_test = tokenizer.texts_to_sequences(features)
    padded_texts = pad_sequences(final_test, maxlen=343, value=0.0)
    prediction = model.predict(padded_texts)
    labels = ['GRP_0', 'GRP_12', 'GRP_24', 'GRP_8', 'GRP_9']
    return render_template('result.html', prediction = 'Issue belongs to {}'.format(labels[np.argmax(prediction)]))



if __name__ == '__main__':
    app.run(debug=True)
