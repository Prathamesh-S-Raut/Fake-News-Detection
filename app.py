from flask import Flask, render_template, request 
import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification
import re
from nltk.stem import PorterStemmer


app = Flask(__name__)

# Load the trained BERT model
model_path = 'content/bertv3_model'
model = TFBertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Preprocessing function
def preprocess_text(text):
    stemmer = PorterStemmer()
    text = re.sub('[^a-zA-Z]', ' ', text)  
    text = text.lower()
    text = ' '.join([stemmer.stem(word) for word in text.split()])
    return text

# Home route
@app.route('/')
def index():
    return render_template('index.html')

# About route
@app.route('/about', methods=['GET', 'POST'])
def about():
    return render_template('about.html')

# Prediction route
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('prediction.html', prediction_text='')  # Show empty form initially
    
    if request.method == 'POST':
        input_text = request.form['news']
        
        # Preprocess input text
        preprocessed_text = preprocess_text(input_text)

        # Tokenize and prepare inputs for BERT
        inputs = tokenizer(preprocessed_text, truncation=True, padding='max_length', max_length=42, return_tensors='tf')

        # Make prediction
        predictions = model(inputs)
        logits = predictions[0][0]
        probabilities = tf.nn.softmax(logits).numpy()
        predicted_label = tf.argmax(probabilities).numpy()

        
        predicted_label = 1 - predicted_label
        
        if predicted_label == 0:
            prediction_text = f"\nFake: {probabilities[0]*100:.2f}% | Real: {probabilities[1]*100:.2f}%"
        else:
            prediction_text = f"\nFake: {probabilities[0]*100:.2f}% | Real: {probabilities[1]*100:.2f}%"

        return render_template('prediction.html', prediction_text=prediction_text)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
