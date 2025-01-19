from flask import Flask, request, jsonify
from nltk import PorterStemmer
from nltk.corpus import stopwords
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
from joblib import load
import re
import string

model_saves_directory = "model-saves"
cnn_text_classification_model = load_model(os.path.join(model_saves_directory, 'cnn_text_classification_model.h5'))
cnn_mental_health_types_classification_model = load(os.path.join(model_saves_directory, 'non_cnn_mental_health_types_classification.joblib'))

tfidf = load(os.path.join(model_saves_directory, 'tfidf_vectorizer.joblib'))
max_sequence_length = 25000
tokenizer = Tokenizer()

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [stemmer.stem(word) for word in tokens]
    text = ' '.join(tokens)
    return text

app = Flask(__name__)

@app.route('/classify', methods=['POST'])
def classify_text():
    data = request.get_json()
    input_text = data.get('text', '')

    if input_text == '':
        return jsonify({'error': 'No text provided'}), 400

    preprocessed_text = preprocess_text(input_text)

    input_sequence = tokenizer.texts_to_sequences([preprocessed_text])
    input_sequence_padded = pad_sequences(input_sequence, maxlen=max_sequence_length)

    mental_health_pred = cnn_text_classification_model.predict(input_sequence_padded)

    print(mental_health_pred)

    if mental_health_pred[0][0] > 0.5:
        return jsonify({'message': 'The text is not related to mental health issues'}), 200

    input_text_tfidf = tfidf.transform([preprocessed_text])
    
    issue_category_pred = cnn_mental_health_types_classification_model.predict(input_text_tfidf.toarray())

    return jsonify({
        'message': 'The text is about mental health issues',
        'category': issue_category_pred[0]
    }), 200

if __name__ == '__main__':
    app.run(debug=True)