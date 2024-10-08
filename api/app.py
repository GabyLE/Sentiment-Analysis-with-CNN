# Importar las bibliotecas necesarias
from flask import Flask, request, jsonify, render_template
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import nltk
from bs4 import BeautifulSoup

# Descargar los recursos necesarios de NLTK
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Inicializar la aplicación Flask
app = Flask(__name__)

# Definir variables necesarias
MAX_SEQUENCE_LENGTH = 100
lemmatizer = WordNetLemmatizer()
palabras_vacias = set(stopwords.words('english'))

# Función para limpiar etiquetas HTML
def limpiar_html(texto):
    return BeautifulSoup(texto, "html.parser").get_text()

# Función de preprocesamiento del texto
def preprocesar_texto_lemmatization(texto):
    if isinstance(texto, str):
        texto = limpiar_html(texto)
        texto = texto.lower()
        texto = re.sub(r'\W+', ' ', texto)
        tokens = [palabra for palabra in texto.split() if palabra not in palabras_vacias]
        tokens = [lemmatizer.lemmatize(palabra) for palabra in tokens]
        return ' '.join(tokens)
    else:
        return ''

# Cargar el modelo previamente entrenado y el tokenizador
model = load_model('sentiment_cnn_model.keras')

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Cargar el LabelEncoder para las etiquetas
le = LabelEncoder()
le.classes_ = np.array(['negative', 'positive'])

# Ruta para la página principal
@app.route('/')
def home():
    return render_template('index.html')

# Ruta para realizar predicción de sentimiento
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        review = request.form['review']  # Obtener la reseña del formulario
        preprocessed_review = preprocesar_texto_lemmatization(review)
        sequence = tokenizer.texts_to_sequences([preprocessed_review])
        padded_sequence = pad_sequences(sequence, maxlen=MAX_SEQUENCE_LENGTH)

        # Hacer predicción con el modelo cargado
        prediction = model.predict(padded_sequence)
        predicted_label = np.argmax(prediction, axis=1)[0]
        sentiment = le.inverse_transform([predicted_label])[0]
        
        return jsonify({'sentiment': sentiment})

# Iniciar la aplicación Flask
if __name__ == '__main__':
    app.run(debug=True)