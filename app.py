from flask import Flask, render_template, request, jsonify
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json

app = Flask(__name__)

# Load embeddings and knowledge base
embeddings = np.load('embeddings.npy')
with open('documents.json', 'r') as f:
    knowledge_base = json.load(f)

# Initialize the sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

def get_most_relevant_doc(query):
    # Encode the user's query
    query_embedding = model.encode([query])

    # Compute cosine similarities between the query and document embeddings
    similarities = cosine_similarity(query_embedding, embeddings)[0]

    # Find the index of the most similar document
    best_match_idx = np.argmax(similarities)
    return knowledge_base[best_match_idx]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    user_input = request.form['user_input']
    relevant_doc = get_most_relevant_doc(user_input)
    response = relevant_doc['content']
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)
