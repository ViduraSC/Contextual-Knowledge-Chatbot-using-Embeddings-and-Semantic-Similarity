import json
from sentence_transformers import SentenceTransformer
import numpy as np

# Load the knowledge base
with open('knowledge_base.json', 'r') as f:
    knowledge_base = json.load(f)

# Initialize the sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Preprocess function to clean and normalize text
def preprocess_text(text):
    # You can add more preprocessing steps as needed
    return text.strip()

# Create embeddings for each document
embeddings = []
for doc in knowledge_base:
    # Preprocess content before encoding
    content = preprocess_text(doc['content'])
    embedding = model.encode(content)
    
    # Add the embedding to the list
    embeddings.append({
        'embedding': embedding.tolist(),  # Convert to list for JSON serialization
        'title': doc['title'],
        'content': content,
        # Include metadata if needed, e.g., synonyms or categories
        'synonyms': doc.get('synonyms', []),  # If you add synonyms in your knowledge base
        'categories': doc.get('categories', [])  # If you add categories in your knowledge base
    })

# Save the embeddings and knowledge base
np.save('embeddings.npy', np.array([e['embedding'] for e in embeddings]))  # Save only embeddings
with open('documents.json', 'w') as f:
    json.dump(embeddings, f)  # Save full entries with embeddings

print("Embeddings created and saved!")
