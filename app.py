from flask import Flask, render_template, request
import re
import pickle
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

# Load the trained model
with open('logistic_regression_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Load the TfidfVectorizer
with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

ps = PorterStemmer()

# Function for text preprocessing
def preprocess_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    text = text.split()
    text = [ps.stem(word) for word in text if word not in set(stopwords.words('english'))]
    text = ' '.join(text)
    return text

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        input_text = request.form['text']
        processed_text = preprocess_text(input_text)
        input_vector = vectorizer.transform([processed_text])
        prediction = model.predict(input_vector)

        if prediction[0] == 0:
            result = 'Real News'
        else:
            result = 'Fake News'

        return render_template('index.html', prediction=result, input_text=input_text)

if __name__ == '__main__':
    app.run(debug=True)
