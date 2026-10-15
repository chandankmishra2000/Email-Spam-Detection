import string
import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from flask import Flask, render_template, request, jsonify

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

# Initialize Flask app
app = Flask(__name__)

# Ensure NLTK stopwords are downloaded
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    """Replicates the exact preprocessing steps from your Jupyter notebook."""
    if not isinstance(text, str):
        return ""
    
    # 1. Lowercase
    text = text.lower()
    
    # 2. Remove punctuation
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    
    # 3. Remove 'subject ' prefix if present
    text = text.replace('subject ', '')
    
    # 4. Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # 5. Remove URLs and links
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # 6. Remove emojis and special symbols
    text = re.sub(r'[^\w\s]', '', text)
    
    # 7. Remove stopwords
    words = [word for word in text.split() if word not in stop_words]
    
    return ' '.join(words)

print("Loading dataset and training the SVM model...")
try:
    # Load dataset
    df = pd.read_csv('emails.csv')
    
    # Drop missing values if any
    df = df.dropna(subset=['text', 'spam'])
    
    # Apply preprocessing pipeline to training data text
    df['processed_text'] = df['text'].apply(preprocess_text)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df['processed_text'],
        df['spam'],
        test_size=0.20,
        random_state=42
    )
    
    # Vectorize using TF-IDF (as done in your notebook)
    tfidf = TfidfVectorizer()
    X_train_tfidf = tfidf.fit_transform(X_train)
    
    # Train the best model (SVM / LinearSVC)
    svm_model = LinearSVC()
    svm_model.fit(X_train_tfidf, y_train)
    
    print("Model trained successfully! Ready to accept requests.")

except Exception as e:
    print(f"Error during model initialization: {e}")
    df = None
    tfidf = None
    svm_model = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        raw_email = data.get('email_text', '')
        
        if not raw_email.strip():
            return jsonify({'error': 'Please provide valid email text.'}), 400
            
        if svm_model is None or tfidf is None:
            return jsonify({'error': 'Model is not initialized properly. Check server logs.'}), 500

        # Preprocess input email text
        processed = preprocess_text(raw_email)
        
        # Transform using fitted TF-IDF vectorizer
        vectorized_input = tfidf.transform([processed])
        
        # Predict using SVM model
        prediction = svm_model.predict(vectorized_input)[0]
        
        # Format result
        result_label = "Spam" if prediction == 1 else "Not Spam (Safe)"
        confidence_status = "High Confidence (SVM Model)"

        return jsonify({
            'prediction': result_label,
            'is_spam': int(prediction),
            'processed_text': processed,
            'details': confidence_status
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)